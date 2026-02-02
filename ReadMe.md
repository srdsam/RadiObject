# RadiObject

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
from radiobject import RadiObject

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
This enables **200-660x faster** partial reads. [See benchmarks →](docs/BENCHMARKS.md)

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
- **[Data Access](docs/DATA_ACCESS.md)** - Ingestion, queries, filtering
- **[ML Integration](docs/ML_INTEGRATION.md)** - MONAI/TorchIO setup
- **[Design](docs/DESIGN.md)** - Architecture decisions
- **[Benchmarks](docs/BENCHMARKS.md)** - Performance analysis
- **[Datasets](docs/DATASETS.md)** - Available datasets and download instructions

## License

MIT
