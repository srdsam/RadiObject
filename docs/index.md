# RadiObject

TileDB-backed radiology data management for ML pipelines.

## Quick Start

```python
from radiobject import RadiObject

# Ingest NIfTI files
radi = RadiObject.from_niftis(
    uri="./dataset",
    images={"CT": "./imagesTr", "seg": "./labelsTr"},
)

# Access volumes
vol = radi.CT.iloc[0]
slice_data = vol.axial(z=50)  # Partial read
```

See [README](https://github.com/srdsam/RadiObject) for installation.

## Documentation

- [Ingest Data](how-to/ingest-data.md) - Ingest NIfTI and DICOM files
- [Query & Filter](how-to/query-filter-data.md) - Access and filter stored data
- [Architecture](explanation/architecture.md) - Design and data model
- [ML Integration](how-to/ml-training.md) - MONAI/TorchIO integration
- [Benchmarks](reference/benchmarks.md) - Performance comparisons
