import lightly
import pytorch_lightning as pl
import timm
import torch
from lightly.loss import NegativeCosineSimilarity
from lightly.models.modules import SimSiamPredictionHead, SimSiamProjectionHead
from lightly.utils.debug import std_of_l2_normalized
from torch import nn


class FastSiam(pl.LightningModule):
    """Model class for FastSiam.

    Parameters
    ----------
    backbone : _type_
        Backbone encoder for self-supervised learning. Typically any model
        from timm or torchvision without the last classification head.
    feat_dim : int
        Output dimension of the backbone encoder. For resnet18 this is 512.
    rep_dim : int, optional
        Output dimension of SimSiam projection and prediction heads, by default 1024.
        The original paper uses 2048, but we use 1024 here for lower complexity.
    log_std : bool, optional
        Whether to log the standard deviation of the l2-normalized features, by default False.
    """

    def __init__(
        self, backbone, feat_dim: int, rep_dim: int = 1024, log_std: bool = True
    ):
        super().__init__()
        self.backbone = backbone
        # Original paper uses dimension d=2048. We use 1024 here for lower complexity.
        self.projection_head = SimSiamProjectionHead(feat_dim, rep_dim, rep_dim)
        # prediction MLP’s hidden layer dimension is always 1/4 of the output dimension
        self.prediction_head = SimSiamPredictionHead(rep_dim, rep_dim // 4, rep_dim)
        self.criterion = NegativeCosineSimilarity()
        self.log_std = log_std

    def forward(self, x):
        f = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(f)
        p = self.prediction_head(z)
        z = z.detach()
        return z, p

    def training_step(self, batch, batch_idx):
        # Unpack augmented views
        views, _, _ = batch
        x1, x2, x3, x4 = views

        # Pass each view through projector to get z, and predictor to get p
        z1, p1 = self.forward(x1)
        z2, p2 = self.forward(x2)
        z3, p3 = self.forward(x3)
        z4, p4 = self.forward(x4)

        # Use mean of the last N - 1 projected views
        mean = (z2 + z3 + z4) / 3

        # Compute loss using prediction of 1st view, mean of remaining projected views
        loss = self.criterion(p1, mean)

        # Keep a log of the loss
        self.log("loss", loss)
        # Monitor the STD of L2-normalized representation to check if it collapses (bad)
        if self.log_std:
            self.log("z1 std", std_of_l2_normalized(z1))
            self.log("z2 std", std_of_l2_normalized(z2))
            self.log("z3 std", std_of_l2_normalized(z3))
            self.log("z4 std", std_of_l2_normalized(z4))

            self.log("mean std", std_of_l2_normalized(mean))

            self.log("p1 std", std_of_l2_normalized(p1))
            self.log("p2 std", std_of_l2_normalized(p2))
            self.log("p3 std", std_of_l2_normalized(p3))
            self.log("p4 std", std_of_l2_normalized(p4))
        return loss

    def configure_optimizers(self):
        # FastSiam authors use lr=0.125 (?!), SimSiam would use 0.00625 here. 0.06 is a happy medium :)
        optim = torch.optim.SGD(self.parameters(), lr=0.06)
        return optim


def fastsiam_resnet18():
    backbone = timm.create_model("resnet18", num_classes=0)
    model = FastSiam(backbone, 512)
    return model


def fastsiam_resnet18_lightly():
    # Lightly uses a different ResNet implementation; need to add a global average pooling layer
    resnet = lightly.models.ResNetGenerator("resnet-18")
    backbone = nn.Sequential(*list(resnet.children())[:-1], nn.AdaptiveAvgPool2d(1))
    model = FastSiam(backbone, 512)
    return model


def fastsiam_convnextv2_nano():
    backbone = timm.create_model("convnextv2_nano", num_classes=0)
    model = FastSiam(backbone, 640)
    return model
