# Tutorials

Step-by-step lessons to learn RadiObject from scratch.

Start here if you're new to RadiObject. Each tutorial builds on the previous one.

## Prerequisites

Before starting the tutorials, download the test datasets:

```bash
# Download BraTS data for tutorials 1-4
python scripts/download_dataset.py msd-brain-tumour

# Download full MSD Lung for ML tutorials 5-7
python scripts/download_dataset.py msd-lung
```

See [Download Datasets](../how-to/datasets.md) for details and alternative download methods.

## Learning Path

| # | Tutorial | What You'll Learn |
|---|----------|-------------------|
| 1 | [Ingest NIfTI Data](../notebooks/00_ingest_brats.ipynb) | Create your first RadiObject |
| 2 | [RadiObject API](../notebooks/01_radi_object.ipynb) | Core data access patterns |
| 3 | [VolumeCollection](../notebooks/02_volume_collection.ipynb) | Group volumes by modality |
| 4 | [Volume Operations](../notebooks/03_volume.ipynb) | Slicing and export |
| 5 | [Storage Config](../notebooks/04_storage_configuration.ipynb) | Tiling and compression |
| 6 | [Multi-Collection](../notebooks/05_ingest_msd.ipynb) | Complex datasets |
| 7 | [ML Training](../notebooks/06_ml_training.ipynb) | MONAI/TorchIO integration |
