# RadiObject

[![PyPI version](https://badge.fury.io/py/radiobject.svg)](https://pypi.org/project/radiobject/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://srdsam.github.io/RadiObject/)

**What?** A TileDB-backed data structure for radiology data at scale.

**Why?** NIfTI/DICOM must be read from local disk and don't support partial reads.
TileDB enables cloud-native storage (S3), efficient partial reads, and
hierarchical organization of large datasets.


## First Principles

- Contextualised data: *Data is always read/written alongside annotations and context, aligned on shared and labelled indexes. Minimize manual joins.*
- Interoperability: *Software should complement the ecosystem of tooling, not compete.*
- Independence: *Each component can exist independently from it's parent (e.g. a `VolumeCollection` can exist without a `RadiObject`)*

*See full [thoughts here](https://souzy.up.railway.app/thoughts/radiology-object).*


## Installation

```bash
pip install radiobject
```

## Quick Start

```python
import numpy as np
from radiobject import (
    RadiObject, VolumeCollection, Volume,
    configure, WriteConfig, ReadConfig, TileConfig,
    SliceOrientation, CompressionConfig, Compressor,
)

# Configure how volumes are written (important for performance!)
configure(write=WriteConfig(
    tile=TileConfig(orientation=SliceOrientation.AXIAL),
    compression=CompressionConfig(algorithm=Compressor.ZSTD, level=3),
))

# Create RadiObject (read NIfTI/DICOM; write TileDB)
radi = RadiObject.from_niftis(
    uri="./my-dataset",
    images={
        "CT": "./imagesTr/*.nii.gz",  # Glob pattern
        "seg": "./labelsTr",          # Directory path
    },
    validate_alignment=True,          # Ensure matching subjects across collections
    obs_meta=metadata_df,             # Optional subject-level metadata (must include obs_id and subject_obs_id)
)

# Access data (pandas-like)
vol: Volume = radi.CT.iloc[0]          # First CT volume
data: np.ndarray = vol[100:200, :, :]  # Or vol.axial(155) works for partial read

# Transform data (polars-like)
ct_resampled: VolumeCollection = radi.CT.map(resample).write(name="CT_resampled")
radi.add_collection(name="CT_resampled", vc=ct_resampled)

# Filter data (returns views)
subset: RadiObject = radi.filter("age > 40")   # Query expression
subset = radi.head(10)                         # First 10 subjects
subset.write("./subset_with_resampled_CT")
```

Works with local paths or S3 URIs (`s3://bucket/dataset`).

## How It Works

NIfTI requires decompressing entire volumes; TileDB reads only the tiles needed.
This enables **200-660x faster** partial reads. [See benchmarks →](docs/reference/benchmarks.md)

![Benchmark overview](benchmarks/results/figures/benchmark_hero.png)

*N.B. Missing comparison with [Zarr](https://github.com/zarr-developers/zarr-python)* 

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

| | |
|---|---|
| **[Tutorials](docs/tutorials/index.md)** | Interactive notebooks: ingestion, querying, volumes, ML |
| **[How-to Guides](docs/how-to/index.md)** | Task-oriented recipes for ingestion, access, ML, cloud |
| **[Reference](docs/reference/index.md)** | API docs, configuration, benchmarks, lexicon |
| **[Explanation](docs/explanation/architecture.md)** | Architecture, performance analysis |

## Development

### Setup

```bash
git clone https://github.com/srdsam/RadiObject.git
cd RadiObject
uv sync --all-extras
```

### Sample Data

Download datasets for tutorials and tests:

```bash
pip install radiobject[download]

# BraTS brain tumour (tutorials 00-04, ~1.5 GB)
python scripts/download_dataset.py msd-brain-tumour

# MSD Lung tumour (tutorials 05-06, ~8.5 GB)
python scripts/download_dataset.py msd-lung

# All test datasets
python scripts/download_dataset.py --all-tests
```

### Tests

```bash
uv run pytest test/ --ignore=test/ml -v
```

S3 integration tests are automatically skipped without AWS credentials. To run the full suite including S3 tests:

```bash
eval $(aws configure export-credentials --profile xxx-s3 --format env)
uv run pytest test/ --ignore=test/ml -v
```

### Notebooks (Local Storage)

Notebooks default to S3 URIs. To run locally, change the URI variable at the top of each notebook from the S3 path to a local path:

```python
# Comment out the S3 URI:
# BRATS_URI = "s3://souzy-scratch/radiobject/brats-tutorial"
# Uncomment the local path:
BRATS_URI = "./data/brats_radiobject"
```

Then run `00_ingest_brats.ipynb` to ingest data, followed by notebooks 01-04. For MSD Lung (notebooks 05-06), change `MSD_LUNG_URI` similarly.

## License

MIT
