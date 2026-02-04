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

## Find What You Need

### Ingest

- [Ingest Data](how-to/ingest-data.md) - NIfTI/DICOM ingestion, managing existing data
- [Streaming Writes](how-to/streaming-writes.md) - Memory-efficient large dataset writes
- [S3 Setup](how-to/s3-setup.md) - Cloud storage configuration

### Explore

- [Indexing & Filtering](how-to/query-filter-data.md) - `iloc`, `loc`, `filter()`, `describe()`
- [Working with Metadata](how-to/working-with-metadata.md) - Subject and volume metadata
- [Volume Operations](how-to/volume-operations.md) - Partial reads, statistics, NIfTI export

### Transform

- [Lazy Pipelines](how-to/lazy-queries.md) - ETL with `map()`, `iter_volumes()`, `materialize()`
- [Configuration](reference/configuration.md) - Tiling, compression, concurrency settings

### Train

- [ML Integration](how-to/ml-training.md) - MONAI and TorchIO DataLoader factories
- [Tuning Concurrency](how-to/tuning-concurrency.md) - Worker and thread configuration
- [Benchmarks](reference/benchmarks.md) - Performance comparisons vs MONAI/TorchIO

### Understand

- [Architecture](explanation/architecture.md) - TileDB data model and design rationale
- [Tutorials](tutorials/index.md) - Step-by-step notebooks (start here if learning)

## Documentation

- [How-to Guides](how-to/index.md) - Task-oriented guides
- [Tutorials](tutorials/index.md) - Step-by-step learning notebooks
- [Reference](reference/index.md) - Configuration, API, benchmarks, lexicon
- [Explanation](explanation/index.md) - Architecture, threading, performance analysis
