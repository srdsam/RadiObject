# RadiObject Tutorials

Interactive notebooks demonstrating RadiObject's API for medical imaging data.

## Quick Start

```bash
# 1. Install with tutorial dependencies
pip install radiobject[tutorials]
# or: uv sync --extra tutorials

# 2. Download sample data (~1.5GB)
python scripts/download_dataset.py msd-brain-tumour

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

## Using S3

Configure S3 via environment variables:

```bash
export RADIOBJECT_S3_BUCKET=your-bucket
export RADIOBJECT_S3_REGION=us-east-2
```

Or use helper functions:

```python
from radiobject.data import get_brats_uri

BRATS_URI = get_brats_uri()  # Returns S3 URI if credentials available
```

See [S3 Setup Guide](https://srdsam.github.io/RadiObject/S3_SETUP/) for AWS configuration.
