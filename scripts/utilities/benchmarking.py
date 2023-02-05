import pytorch_lightning as pl
import timm
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from lightly.utils import knn_predict
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score


# modified from https://github.com/lightly-ai/lightly/blob/master/lightly/utils/benchmarking.py
# source is https://arxiv.org/abs/1805.01978
class KNNBenchmarkModule(pl.LightningModule):
    """A PyTorch Lightning Module for automated kNN callback with support for torchmetrics.

    Modified from https://github.com/lightly-ai/lightly/blob/master/lightly/utils/benchmarking.py

    At the end of every training epoch we create a feature bank by feeding the
    `dataloader_kNN` passed to the module through the backbone.
    At every validation step we predict features on the validation data.
    After all predictions on validation data (validation_epoch_end) we evaluate
    the predictions on a kNN classifier on the validation data using the
    feature_bank features from the train data.

    We can access the highest test accuracy during a kNN prediction
    using the `max_accuracy` attribute.

    Attributes:
        backbone:
            The backbone model used for kNN validation. Make sure that you set the
            backbone when inheriting from `BenchmarkModule`.
        max_accuracy:
            Floating point number between 0.0 and 1.0 representing the maximum
            test accuracy the benchmarked model has achieved.
        dataloader_kNN:
            Dataloader to be used after each training epoch to create feature bank.
        num_classes:
            Number of classes. E.g. for cifar10 we have 10 classes. (default: 10)
        knn_k:
            Number of nearest neighbors for kNN
        knn_t:
            Temperature parameter for kNN
    """

    def __init__(
        self,
        dataloader_kNN: DataLoader,
        num_classes: int,
        knn_k: int = 27,  # TODO: find a good default value, 200 is too high for class imbalance
        knn_t: float = 0.1,
    ):
        super().__init__()
        self.backbone = nn.Module()
        self.max_accuracy = 0.0
        self.max_f1 = 0.0
        self.dataloader_kNN = dataloader_kNN
        self.num_classes = num_classes
        self.knn_k = knn_k
        self.knn_t = knn_t

        # Initialize metrics for validation; imbalanced classes, so use macro average
        self.val_accuracy = MulticlassAccuracy(num_classes=num_classes, average="macro")
        self.val_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")

        # create dummy param to keep track of the device the model is using
        self.dummy_param = nn.Parameter(torch.empty(0))

    def training_epoch_end(self, outputs):
        # update feature bank at the end of each training epoch
        self.backbone.eval()
        self.feature_bank = []
        self.targets_bank = []
        with torch.no_grad():
            for data in self.dataloader_kNN:
                img, target, _ = data
                img = img.to(self.dummy_param.device)
                target = target.to(self.dummy_param.device)
                feature = self.backbone(img).squeeze()
                feature = F.normalize(feature, dim=1)
                self.feature_bank.append(feature)
                self.targets_bank.append(target)
        self.feature_bank = torch.cat(self.feature_bank, dim=0).t().contiguous()
        self.targets_bank = torch.cat(self.targets_bank, dim=0).t().contiguous()
        self.backbone.train()

    def validation_step(self, batch, batch_idx):
        # we can only do kNN predictions once we have a feature bank
        if hasattr(self, "feature_bank") and hasattr(self, "targets_bank"):
            images, targets, _ = batch
            feature = self.backbone(images).squeeze()
            feature = F.normalize(feature, dim=1)
            pred_labels = knn_predict(
                feature,
                self.feature_bank,
                self.targets_bank,
                self.num_classes,
                self.knn_k,
                self.knn_t,
            )
            return (pred_labels[:, 0], targets)

    def validation_epoch_end(self, outputs):
        # Compute classification metrics once we full feature bank
        if outputs:
            # concatenate all predictions and targets
            all_preds = torch.cat([x[0] for x in outputs], dim=0)
            all_targets = torch.cat([x[1] for x in outputs], dim=0)

            # update metrics
            self.val_accuracy(all_preds, all_targets)
            self.val_f1(all_preds, all_targets)

            # update maxima
            if self.val_accuracy.compute().item() > self.max_accuracy:
                self.max_accuracy = self.val_accuracy.compute().item()
            if self.val_f1.compute().item() > self.max_f1:
                self.max_f1 = self.val_f1.compute().item()

            # log metrics
            self.log("knn_accuracy", self.val_accuracy, on_epoch=True, prog_bar=True)
            self.log("knn_f1", self.val_f1, on_epoch=True, prog_bar=True)

    def predict_step(self, batch, batch_idx):
        # Recommended usage: preds = trainer.predict(model, dataloader)
        # preds = torch.cat(preds, dim=0)
        images, _, _ = batch
        return self.backbone(images)
