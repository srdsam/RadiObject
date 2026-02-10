# Tutorials

Step-by-step lessons to learn RadiObject from scratch. Each tutorial builds on the previous one. Unfamiliar with radiology or TileDB terms? See the [Lexicon](../reference/lexicon.md).

## Prerequisites

Download the test datasets before starting:

```bash
# BraTS data for tutorials 1-3
python scripts/download_dataset.py msd-brain-tumour

# Full MSD Lung for tutorials 4-5
python scripts/download_dataset.py msd-lung
```

See [Datasets](../reference/datasets.md) for details and alternative download methods.

## Learning Path

| # | Tutorial | What You Learn |
|---|----------|----------------|
| 1 | [Ingest NIfTI Data](../notebooks/00_ingest_brats.ipynb) | Create your first RadiObject from NIfTI files |
| 2 | [Explore Data](../notebooks/01_explore_data.ipynb) | RadiObject, collections, volumes: indexing, filtering, partial reads |
| 3 | [Configuration](../notebooks/02_configuration.ipynb) | Write settings, read tuning, S3 config |
| 4 | [Multi-Collection Ingestion](../notebooks/03_ingest_msd.ipynb) | Complex datasets with multiple modalities |
| 5 | [ML Training](../notebooks/04_ml_training.ipynb) | MONAI segmentation training with DataLoader integration |
