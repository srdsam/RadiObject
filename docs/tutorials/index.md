# Tutorials

Step-by-step lessons to learn RadiObject from scratch. Each tutorial builds on the previous one. Unfamiliar with radiology or TileDB terms? See the [Lexicon](../reference/lexicon.md).

## Prerequisites

Download the test datasets before starting:

```bash
# BraTS data for tutorials 1-5
python scripts/download_dataset.py msd-brain-tumour

# Full MSD Lung for ML tutorial 7
python scripts/download_dataset.py msd-lung
```

See [Datasets](../reference/datasets.md) for details and alternative download methods.

## Learning Path

| # | Tutorial | What You Learn |
|---|----------|----------------|
| 1 | [Ingest NIfTI Data](../notebooks/00_ingest_brats.ipynb) | Create your first RadiObject from NIfTI files |
| 2 | [RadiObject API](../notebooks/01_radi_object.ipynb) | Core data access patterns: indexing, filtering, metadata |
| 3 | [VolumeCollection](../notebooks/02_volume_collection.ipynb) | Group volumes by modality, cross-collection operations |
| 4 | [Volume Operations](../notebooks/03_volume.ipynb) | Slicing, partial reads, statistics, NIfTI export |
| 5 | [Storage Config](../notebooks/04_configuration.ipynb) | Tiling strategies, compression, performance tuning |
| 6 | [Multi-Collection Ingestion](../notebooks/05_ingest_msd.ipynb) | Complex datasets with multiple modalities |
| 7 | [ML Training](../notebooks/06_ml_training.ipynb) | MONAI and TorchIO DataLoader integration |
