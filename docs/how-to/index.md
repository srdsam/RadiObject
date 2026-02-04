# How-to Guides

Task-oriented guides organized by workflow stage. Follow the journey from ingestion through training, or jump to the guide you need.

## Ingest → Explore → Transform → Train

### 1. Get Data In

| I want to... | Guide |
|--------------|-------|
| Ingest NIfTI/DICOM files or manage existing data | [Ingest Data](ingest-data.md) |
| Write large datasets incrementally | [Streaming Writes](streaming-writes.md) / [Append Data](append-data.md) |

**Next:** Once data is ingested, explore it with [Indexing & Filtering](query-filter-data.md).

### 2. Explore & Access

| I want to... | Guide |
|--------------|-------|
| Filter subjects, index by position or ID | [Indexing & Filtering](query-filter-data.md) |
| Browse subject/volume metadata | [Working with Metadata](working-with-metadata.md) |
| Read slices, get statistics, or export to NIfTI | [Volume Operations](volume-operations.md) |

**Next:** For ETL or ML data prep, continue to [Lazy Pipelines](lazy-queries.md).

### 3. Transform & Train

| I want to... | Guide |
|--------------|-------|
| Build ETL pipelines with transforms or streaming | [Lazy Pipelines](lazy-queries.md) |
| Train with MONAI or TorchIO | [ML Integration](ml-training.md) |
| Tune DataLoader and thread settings | [Tuning Concurrency](tuning-concurrency.md) |

### 4. Infrastructure

[S3 Setup](s3-setup.md) | [Profiling](profiling.md) | [Troubleshooting](troubleshooting.md) | [Datasets](datasets.md) | [Contributing](contributing.md)
