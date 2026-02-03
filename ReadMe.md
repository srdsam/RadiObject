# RadiObject

[![PyPI version](https://badge.fury.io/py/radiobject.svg)](https://pypi.org/project/radiobject/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://srdsam.github.io/RadiObject/)

**What?** A TileDB-backed data structure for radiology data at scale.

**Why?** NIfTI/DICOM must be read from local disk and don't support partial reads.
TileDB enables cloud-native storage (S3), efficient partial reads, and
hierarchical organization of multi-volume datasets.

*[Thoughts](https://souzy.up.railway.app/thoughts/radiology-object)*

## Installation

```bash
pip install radiobject
```

## Quick Start

```python
from radiobject import RadiObject, configure, WriteConfig, ReadConfig, TileConfig

# Configure how volumes are written (important for performance!)
configure(write=WriteConfig(
    tile=TileConfig(orientation=SliceOrientation.AXIAL), # Can be ISOTROPIC for 3D patch extraction...
    compression=CompressionConfig(algorithm=Compressor.ZSTD, level=3),
))

# Create from NIfTI files using images dict (recommended)
radi = RadiObject.from_niftis(
    uri="./my-dataset",
    images={
        "CT": "./imagesTr/*.nii.gz",      # Glob pattern
        "seg": "./labelsTr",               # Directory path
    },
    validate_alignment=True,               # Ensure matching subjects across collections
    obs_meta=metadata_df,                  # Optional subject-level metadata
)

# Access data (pandas-like)
vol = radi.CT.iloc[0]            # First CT volume
data = vol[100:200, :, :]        # Partial read (only loads needed tiles)

# Filtering (returns views)
subset = radi.filter("age > 40")       # Query expression
subset = radi.head(10)                 # First 10 subjects
subset.materialize("./subset")         # Write to storage
```

Works with local paths or S3 URIs (`s3://bucket/dataset`).

## How It Works

NIfTI requires decompressing entire volumes; TileDB reads only the tiles needed.
This enables **200-660x faster** partial reads. [See benchmarks →](docs/reference/benchmarks.md)

## Sample Data

Download sample datasets for tutorials and testing:

```bash
# Install download dependencies
pip install radiobject[download]

# Download BraTS brain tumor data (for tutorials 00-04)
python scripts/download_dataset.py msd-brain-tumour

# List all available datasets
python scripts/download_dataset.py --list
```

## Documentation

- **[Tutorials](notebooks/README.md)** - Interactive notebooks
- **[Ingest Data](docs/how-to/ingest-data.md)** - NIfTI/DICOM ingestion
- **[Query & Filter](docs/how-to/query-filter-data.md)** - Data access patterns
- **[ML Integration](docs/how-to/ml-training.md)** - MONAI/TorchIO setup
- **[Architecture](docs/explanation/architecture.md)** - Design decisions
- **[Benchmarks](docs/reference/benchmarks.md)** - Performance analysis
- **[Datasets](docs/how-to/datasets.md)** - Available datasets and download instructions

## License

MIT
