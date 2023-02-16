# Scripts
- `benchmarking.py`: Imagenette benchmark we ran for FastSiam. Mostly copied from lightly's benchmark [here](https://github.com/lightly-ai/lightly/blob/442e54bc1af40cc904c41a8856f36882c7f9701b/docs/source/getting_started/benchmarks/imagenette_benchmark.py).
- `wm811k.ipynb`: Analyzing, cleaning, and splitting the WM-811K dataset, plus some prototyping of collate functions that make sense for semiconductor wafer maps.
- `fastsiam_wafers.ipynb`: Training FastSiam on our full training data in a self-supervised fashion.
- `fully_supervised.ipynb`: Training a modified ResNet-18 end-to-end in a fully supervised fashion on different splits of our training data, as a baseline for comparison.
- `linear_probe.ipynb`: Analyzing frozen features from a FastSiam pretrained in a self-supervised fashion with UMAP embedding visualizations. We also fit a shallow MLP on these frozen features using different amounts of labeled data to see how we compare against a fully supervised ResNet-18 baseline.

## KNN Benchmarking results
Run on a single GTX 1080 Ti GPU. All models used a ResNet-18 backbone unless specified otherwise. See [`wafer_benchmarks.py`](wafer_benchmarks.py) for full implementation details, and [`utilities/data.py`](utilities/data.py) for the collate functions used for each model (the collate functions determine the augmentation pipelines for self-supervised learning).

| Model           | Batch Size | Epochs | #param. | KNN Test Accuracy | KNN Test F1 |    Time    | Peak GPU Usage |
|-----------------|:----------:|:------:|:-------:|:-----------------:|:-----------:|:----------:|:--------------:|
| SupervisedR18   |     32     |   200  |  11.2M  |       0.751       |    0.738    |  266.1 Min |    0.9 GByte   |
| BarlowTwins     |     32     |   200  |  20.6M  |       0.584       |    0.611    |  554.8 Min |    1.8 GByte   |
| BYOL            |     32     |   200  |  16.4M  |       0.611       |    0.636    |  446.4 Min |    1.8 GByte   |
| DCLW            |     32     |   200  |  11.5M  |       0.637       |    0.637    |  374.5 Min |    1.6 GByte   |
| Moco            |     32     |   200  |  12.5M  |       0.604       |    0.614    |  513.1 Min |    1.8 GByte   |
| SimCLR          |     32     |   200  |  11.5M  |       0.628       |    0.635    |  392.0 Min |    1.6 GByte   |
| SimSiam         |     32     |   200  |  22.7M  |       0.461       |    0.472    |  374.0 Min |    1.7 GByte   |
| FastSiam        |     32     |   200  |  22.7M  |       0.467       |    0.455    |  744.4 Min |    3.0 GByte   |
| FastSiam(sym)   |     32     |   200  |  22.7M  |       0.514       |    0.528    |  785.1 Min |    3.0 GByte   |
| SwaV            |     32     |   200  |  12.6M  |       0.597       |    0.619    | 1092.6 Min |    2.7 GByte   |
| VICReg*         |     32     |   200  |  20.6M  |       0.590       |    0.608    |  258.0 Min*|    1.7 GByte   |
| DINO            |     32     |   200  |  17.5M  |       0.531       |    0.557    | 1041.0 Min |    2.8 GByte   |
| DINO (ViT-S/16) |     32     |   200  |  27.7M  |       0.562       |    0.566    | 1919.2 Min |    7.6 GByte   |
| MSN (ViT-S/16)  |     32     |   200  |  27.8M  |       0.609       |    0.621    | 1519.7 Min |    6.4 GByte   |
| PMSN* (ViT-S/16)|     32     |   200  |  27.8M  |       0.622       |    0.646    |  795.4 Min*|    6.4 GByte   |
| MAE (ViT-B/32)  |     32     |   200  |  93.4M  |       0.669       |    0.697    |  423.4 Min |    1.9 GByte   |

*VICReg was run on a 3080 Ti GPU, hence the much shorter training time. It trains about as fast as SimSiam, so it would probably take around 380 minutes to train on a 1080 Ti. PMSN was also run on a 3080 Ti, and it should take the same time to train as MSN.