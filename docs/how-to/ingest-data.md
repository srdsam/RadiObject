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

## DICOM Ingestion

For DICOM data, use `from_dicoms` which automatically extracts metadata and groups by modality:

```python
radi = RadiObject.from_dicoms(
    uri="./dicom-dataset",
    dicom_dirs=[
        ("./dicom/sub01/CT_HEAD", "sub-01"),
        ("./dicom/sub02/CT_HEAD", "sub-02"),
    ],
    progress=True,
)
```

Each tuple maps a DICOM directory to a subject ID. Collections are auto-grouped by modality and dimensions.

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

## 4D / Functional Data (fMRI, DTI)

RadiObject natively supports 4D NIfTI volumes (3 spatial + 1 temporal/channel dimension).
No special configuration is needed—4D files are ingested the same way as 3D:

```python
radi = RadiObject.from_niftis(
    uri="./fmri-study",
    images={"bold": "./func/*_bold.nii.gz"},
)

vol = radi.bold.iloc[0]
print(vol.shape)   # (64, 64, 32, 200)  — 200 timepoints
print(vol.ndim)    # 4
```

**How dimensions work with 4D data:**

- **Collection shape** (`radi.bold.shape`) reports spatial dimensions only: `(64, 64, 32)`
- **Volume shape** (`vol.shape`) reports the full shape including time: `(64, 64, 32, 200)`
- **Metadata** (`obs.dimensions`) captures the full 4D shape as a string
- **Shape validation** compares spatial dims only—volumes with different timepoint counts
  but the same spatial grid pass `validate_dimensions=True`

## Managing Existing Data

### Check If Data Exists

Before ingesting, verify whether a TileDB object already exists at the target URI:

```python
from radiobject import uri_exists

if uri_exists("s3://bucket/my-dataset"):
    print("Dataset already exists")
```

### Delete and Reingest

Remove an existing TileDB object to replace it:

```python
from radiobject import delete_tiledb_uri

delete_tiledb_uri("s3://bucket/my-dataset")
radi = RadiObject.from_niftis("s3://bucket/my-dataset", images={"CT": "./data"})
```

### Create from Existing VolumeCollections

Assemble a RadiObject from pre-existing VolumeCollections — useful after materializing transformed collections:

```python
from radiobject import RadiObject

# Collections already at expected sub-paths (no copy)
ct = radi.CT.lazy().map(transform).materialize(uri=f"{URI}/collections/CT")
seg = radi.seg.lazy().map(transform).materialize(uri=f"{URI}/collections/seg")

new_radi = RadiObject.from_collections(
    uri=URI,
    collections={"CT": ct, "seg": seg},
)

# Collections from elsewhere (will be copied to target)
new_radi = RadiObject.from_collections(
    uri="./new-dataset",
    collections={"T1w": existing_collection},
    obs_meta=metadata_df,  # Optional; derived from collections if omitted
)
```

For full API details, see [RadiObject API](../api/radi_object.md).

## Large Dataset Ingestion

For datasets too large to fit in memory, use streaming writes:

- [Streaming Writes](streaming-writes.md) - Incremental writes with `StreamingWriter`
- [Append Data](append-data.md) - Add subjects to existing RadiObjects

## Next Step

**Data ingested?** Explore it with [Indexing & Filtering](query-filter-data.md) — filter subjects, inspect collections, and access individual volumes.

## Related Documentation

- [Configuration](../reference/configuration.md) - Tile orientation, compression settings
