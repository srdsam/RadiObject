# Ingest Data

RadiObject ingests NIfTI and DICOM files into TileDB arrays for efficient storage and retrieval.

## NIfTI Ingestion

The `images` dict API is the recommended way to create a RadiObject from NIfTI files:

```python
from radiobject import RadiObject

# Ingest multiple collections in a single call
radi = RadiObject.from_niftis(
    uri="./my-dataset",
    images={
        "CT": "./imagesTr/*.nii.gz",      # Glob pattern
        "seg": "./labelsTr",               # Directory path
    },
    validate_alignment=True,               # Ensure matching subject IDs
    obs_meta=metadata_df,                  # Optional subject metadata
    progress=True,
)
```

**Source formats** (values in `images` dict):

| Format | Example | Description |
|--------|---------|-------------|
| Glob pattern | `"./data/*.nii.gz"` | Match files by pattern |
| Directory | `"./data/imagesTr"` | Discover all NIfTIs in directory |
| Pre-resolved list | `[(path, subject_id), ...]` | Explicit mapping |

**Options:**

| Parameter | Description |
|-----------|-------------|
| `validate_alignment` | Verify all collections have matching subject IDs |
| `obs_meta` | DataFrame with subject-level metadata (must have `obs_subject_id` column) |
| `reorient` | Reorient volumes to canonical orientation during ingestion |
| `progress` | Show progress bar |

## Legacy APIs

These APIs are still supported for backwards compatibility:

```python
# Directory-based (single collection)
radi = RadiObject.from_niftis(uri, image_dir="./data", collection_name="CT")

# Tuple list with auto-grouping by modality
radi = RadiObject.from_niftis(uri, niftis=[(path, subject_id), ...])
```

## DICOM Ingestion

For DICOM data, use `from_dicoms` which automatically extracts metadata and groups by modality:

```python
radi = RadiObject.from_dicoms(
    uri="./dicom-dataset",
    dicom_dir="./dicom_source",
    progress=True,
)
```

## Writing to S3

Ingest directly to S3 by using an S3 URI:

```python
RadiObject.from_niftis(
    "s3://your-bucket/new-dataset",
    images={"CT": "./local/images"},
    obs_meta=df,
)
```

See [S3 Setup](s3-setup.md) for AWS credential configuration.

## Large Dataset Ingestion

For datasets too large to fit in memory, use streaming writes:

- [Streaming Writes](streaming-writes.md) - Incremental writes with `StreamingWriter`
- [Append Data](append-data.md) - Add subjects to existing RadiObjects

## Related Documentation

- [Query & Filter](query-filter-data.md) - Access ingested data
- [Configuration](../reference/configuration.md) - Tile orientation, compression settings
