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

See [README](https://github.com/samueldsouza/radiobject) for installation.

## Documentation

- [Data Access](DATA_ACCESS.md) - Ingestion, querying, and slicing
- [Design](DESIGN.md) - Architecture and design decisions
- [ML Integration](ML_INTEGRATION.md) - MONAI/TorchIO integration
- [Benchmarks](BENCHMARKS.md) - Performance comparisons
