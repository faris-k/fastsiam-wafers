# fastsiam-wafers
Self-Supervised Representation Learning of Wafer Maps with FastSiam

This repository includes an unofficial PyTorch Lightning implementation of ["FastSiam: Resource-Efficient Self-supervised Learning on a Single GPU"](https://link.springer.com/chapter/10.1007/978-3-031-16788-1_4). See [`fastsiam.py`](scripts/models/fastsiam.py) for our implementation, in which we simply modify [lightly AI's](https://github.com/lightly-ai/lightly) implementation of SimSiam. We wanted to make our implementation more or less plug-and-play. The only other thing you'd need is a custom collate function to extract 4 augmented views per image instead of 2, which is pretty simple. Below you'll find a full example.

For this project, we applied FastSiam to the [WM-811K semiconductor wafer map dataset](http://mirlab.org/dataset/public/) (or rather, a [subset](https://www.kaggle.com/datasets/mohammedfariskhan/wm811k-clean-subset) of it). We also benchmarked our implementation of FastSiam on the [Imagenette benchmark used by lightly](https://docs.lightly.ai/self-supervised-learning/getting_started/benchmarks.html#imagenette) (see [`benchmarking.py`](scripts/benchmarking.py) in `scripts/`).


## Example Usage of FastSiam

```python
import lightly
import pytorch_lightning as pl
import timm
import torch
import torchvision
from lightly.data import LightlyDataset
from lightly.data.collate import MultiViewCollateFunction, SimCLRCollateFunction
from lightly.loss import NegativeCosineSimilarity
from lightly.models.modules import SimSiamPredictionHead, SimSiamProjectionHead
from torch import nn
from torch.utils.data import DataLoader


class FastSiam(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # Compatible with any backbone encoder, just remove the classification head
        self.backbone = timm.create_model("resnet18", num_classes=0)
        feat_dim = timm.create_model("resnet18").get_classifier().in_features
        self.projection_head = SimSiamProjectionHead(feat_dim, 1024, 1024)
        self.prediction_head = SimSiamPredictionHead(1024, 256, 1024)
        self.criterion = NegativeCosineSimilarity()

    def forward(self, x):
        f = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(f)
        p = self.prediction_head(z)
        z = z.detach()
        return z, p

    def training_step(self, batch, batch_idx):
        # Unpack N augmented views
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
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.parameters(), lr=0.06)
        return optim


model = FastSiam()

cifar10 = torchvision.datasets.CIFAR10("datasets/cifar10", download=True)
dataset = LightlyDataset.from_torch_dataset(cifar10)

# or create a dataset from a folder containing images or videos:
# dataset = LightlyDataset("path/to/folder")

simclr_collate_fn = SimCLRCollateFunction(input_size=32)
base_transforms = simclr_collate_fn.transform
# or any of your own transforms, like this:
# base_transforms = T.Compose([
#     # your transforms here
# ])

fastsiam_collate_fn = MultiViewCollateFunction([base_transforms] * 4)

dataloader = DataLoader(
    dataset,
    batch_size=32,
    collate_fn=fastsiam_collate_fn
)

accelerator = "gpu" if torch.cuda.is_available() else "cpu"

trainer = pl.Trainer(max_epochs=10, accelerator=accelerator, devices=-1)
trainer.fit(model=model, train_dataloaders=dataloader)
```

## BibTeX
FastSiam:
```bibtex
@inproceedings{pototzky2022fastsiam,
    title={FastSiam: Resource-Efficient Self-supervised Learning on a Single GPU},
    author={Pototzky, Daniel and Sultan, Azhar and Schmidt-Thieme, Lars},
    booktitle={DAGM German Conference on Pattern Recognition},
    pages={53--67},
    year={2022},
    organization={Springer}
}
```

WM-811K Dataset:
```bibtex
@article{wu2014wafer,
    title={Wafer map failure pattern recognition and similarity ranking for large-scale data sets},
    author={Wu, Ming-Ju and Jang, Jyh-Shing R and Chen, Jui-Long},
    journal={IEEE Transactions on Semiconductor Manufacturing},
    volume={28},
    number={1},
    pages={1--12},
    year={2014},
    publisher={IEEE}
}
```


Lightly: 
```bibtex
@article{susmelj2020lightly,
    title={Lightly},
    author={Igor Susmelj and Matthias Heller and Philipp Wirth and Jeremy Prescott and Malte Ebner et al.},
    journal={GitHub. Note: https://github.com/lightly-ai/lightly},
    year={2020}
}
```


