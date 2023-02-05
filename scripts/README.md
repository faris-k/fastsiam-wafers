# Scripts
- `benchmarking.py`: Imagenette benchmark we ran for FastSiam. Mostly copied from lightly's benchmark [here](https://github.com/lightly-ai/lightly/blob/442e54bc1af40cc904c41a8856f36882c7f9701b/docs/source/getting_started/benchmarks/imagenette_benchmark.py).
- `wm811k.ipynb`: Analyzing, cleaning, and splitting the WM-811K dataset, plus some prototyping of collate functions that make sense for semiconductor wafer maps.
- `fastsiam_wafers.ipynb`: Training FastSiam on our full training data in a self-supervised fashion.
- `fully_supervised.ipynb`: Training a modified ResNet-18 end-to-end in a fully supervised fashion on different splits of our training data, as a baseline for comparison.
- `linear_probe.ipynb`: Analyzing frozen features from a FastSiam pretrained in a self-supervised fashion with UMAP embedding visualizations. We also fit a shallow MLP on these frozen features using different amounts of labeled data to see how we compare against a fully supervised ResNet-18 baseline.

## KNN Benchmarking results
Run on a single GTX 1080 Ti GPU. All models used a ResNet-18 backbone unless specified otherwise. See [`wafer_benchmarks.py`](wafer_benchmarks.py) for full implementation details, and [`utilities/data.py`](utilities/data.py) for the collate functions used for each model (the collate functions determine the augmentation pipelines for self-supervised learning).

| Model           | Batch Size | Epochs | KNN Test Accuracy | KNN Test F1 |       Time | Peak GPU Usage |
|-----------------|-----------:|-------:|------------------:|------------:|-----------:|---------------:|
| BarlowTwins     |         32 |    200 |             0.461 |       0.417 |  547.4 Min |      1.9 GByte |
| BYOL            |         32 |    200 |             0.474 |       0.478 |  442.5 Min |      1.8 GByte |
| DCLW            |         32 |    200 |             0.523 |       0.529 |  372.2 Min |      1.6 GByte |
| MoCo            |         32 |    200 |             0.502 |       0.500 |  438.0 Min |      1.8 GByte |
| SimCLR          |         32 |    200 |             0.537 |       0.561 |  365.0 Min |      1.6 GByte |
| SimSiam         |         32 |    200 |             0.380 |       0.392 |  373.9 Min |      1.7 GByte |
| FastSiam        |         32 |    200 |             0.384 |       0.357 |  467.8 Min |      3.0 GByte |
| SwaV            |         32 |    200 |             0.519 |       0.525 | 1092.5 Min |      2.7 GByte |
| DINO            |         32 |    200 |             0.358 |       0.337 | 1030.0 Min |      2.8 GByte |
| DINO (ViT-S/16) |         32 |    200 |             0.329 |       0.300 | 1911.3 Min |      7.6 GByte |
| MSN (ViT-S/16)  |         32 |    200 |             0.484 |       0.483 | 1512.7 Min |      6.4 GByte |
| MAE (ViT-B/32)  |         32 |    200 |             0.506 |       0.514 |  412.3 Min |      1.9 GByte |