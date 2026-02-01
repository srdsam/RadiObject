# RadiObject Tutorials

Interactive notebooks demonstrating RadiObject's API for medical imaging data.

## Quick Start

```bash
# 1. Install with tutorial dependencies
pip install radiobject[tutorials]
# or: uv sync --extra tutorials

# 2. Download sample data (~1.5GB)
python scripts/download_tutorial_data.py

# 3. Run notebooks
cd notebooks
jupyter notebook
```

## Notebooks

| Notebook | Description |
|----------|-------------|
| **00_ingest_brats.ipynb** | Ingest NIfTI files into TileDB (auto-grouping by modality) |
| **01_radi_object.ipynb** | Core RadiObject API |
| **02_volume_collection.ipynb** | Working with VolumeCollections |
| **03_volume.ipynb** | Volume operations and slicing |
| **04_storage_configuration.ipynb** | Tile orientation, compression |
| **05_ingest_msd.ipynb** | Multi-collection ingestion with `images` dict API |
| **06_ml_training.ipynb** | MONAI/TorchIO integration |

## Data Requirements

**Notebooks 00-04**: BraTS sample data (downloaded via script)

**Notebooks 05-06**: MSD Lung data (requires separate download or S3 access)

## Using S3 Instead

Edit `config.py` to use S3 URIs:

```python
# config.py
BRATS_URI = "s3://your-bucket/brats-tutorial"
MSD_LUNG_URI = "s3://your-bucket/msd-lung"
S3_REGION = "us-east-2"
```

See [S3 Setup Guide](../docs/S3_SETUP.md) for AWS configuration.
