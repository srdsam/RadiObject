# Ingest Data

RadiObject ingests NIfTI and DICOM files into TileDB arrays for efficient storage and retrieval. For terminology (NIfTI, DICOM, voxel spacing), see the [Lexicon](../reference/lexicon.md).

## Choosing an Ingestion Method

| Method | Best For | Memory Requirement |
|--------|----------|-------------------|
| `RadiObject.from_niftis()` | Small-medium datasets (<1000 subjects) | Fits in memory |
| `StreamingWriter` | Large single-collection datasets | Constant memory |
| `RadiObjectWriter` | Complex multi-collection builds | Controlled batches |

**Decision guide:**

1. **Can all data fit in memory?** → Use `RadiObject.from_niftis()`
2. **Single collection, too large for memory?** → Use `StreamingWriter`
3. **Multiple collections with custom logic?** → Use `RadiObjectWriter`

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

## Handling Orientation

Medical images use coordinate systems (RAS, LPS, etc.) to map voxels to physical space. RadiObject can reorient volumes during ingestion.

### Automatic Reorientation

```python
from radiobject import configure, WriteConfig, OrientationConfig

configure(write=WriteConfig(
    orientation=OrientationConfig(
        canonical_target="RAS",
        reorient_on_load=True
    )
))

# All subsequent ingestions will reorient to RAS
radi = RadiObject.from_niftis(uri, images={"CT": "./data"})
```

### When to Reorient

| Scenario | Recommendation |
|----------|----------------|
| Multi-site studies with inconsistent scanner orientations | **Do reorient** |
| Preserving original acquisition geometry matters | **Don't reorient** |
| ML training (standardized input expected) | **Do reorient** |
| Clinical review (radiologist expects native orientation) | **Don't reorient** |

See [Lexicon: Coordinate Systems](../reference/lexicon.md#coordinate-systems-orientation) for terminology.

## Large Dataset Ingestion

For datasets too large to fit in memory, use streaming writes:

- [Streaming Writes](streaming-writes.md) - Incremental writes with `StreamingWriter`
- [Append Data](append-data.md) - Add subjects to existing RadiObjects

## Related Documentation

- [Query & Filter](query-filter-data.md) - Access ingested data
- [Configuration](../reference/configuration.md) - Tile orientation, compression settings
