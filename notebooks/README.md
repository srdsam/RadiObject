# RadiObject Tutorials

Interactive notebooks demonstrating RadiObject's API for medical imaging data.

## Quick Start

```bash
# 1. Install with tutorial dependencies
pip install radiobject[tutorials]
# or: uv sync --extra tutorials

# 2. Download sample data
pip install radiobject[download]
python scripts/download_dataset.py msd-brain-tumour   # Notebooks 00-04 (~1.5 GB)
python scripts/download_dataset.py msd-lung            # Notebooks 05-06 (~8.5 GB)

# 3. Run notebooks
cd notebooks
jupyter notebook
```

## Notebooks

| Notebook | Description | Data Required |
|----------|-------------|---------------|
| **00_ingest_brats.ipynb** | Ingest NIfTI files into TileDB | BraTS (msd-brain-tumour) |
| **01_radi_object.ipynb** | Core RadiObject API | Run 00 first |
| **02_volume_collection.ipynb** | Working with VolumeCollections | Run 00 first |
| **03_volume.ipynb** | Volume operations and slicing | Run 00 first |
| **04_configuration.ipynb** | Tile orientation, compression | None (uses synthetic data) |
| **05_ingest_msd.ipynb** | Multi-collection ingestion with `images` dict API | MSD Lung (msd-lung) |
| **06_ml_training.ipynb** | MONAI/TorchIO integration | Run 05 first |

## Storage: S3 vs Local

Each notebook has a URI variable at the top that controls where data is stored:

```python
# Default: S3 (requires AWS credentials)
BRATS_URI = "s3://souzy-scratch/radiobject/brats-tutorial"
# For local storage, comment out the line above and uncomment:
# BRATS_URI = "./data/brats_radiobject"
```

**To run locally:** Comment out the S3 URI and uncomment the local path. Everything else works the same — S3 and local paths are interchangeable.

**To run with S3:** Use the default S3 URI. Ensure AWS credentials are configured (see [Cloud Setup](https://srdsam.github.io/RadiObject/how-to/cloud-setup/)).
