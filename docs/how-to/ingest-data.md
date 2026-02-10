# Ingest Data

RadiObject ingests NIfTI and DICOM files into TileDB arrays for efficient storage and retrieval. For terminology, see the [Lexicon](../reference/lexicon.md).

## Choosing an Ingestion Method

| Method | Best For | Memory |
|--------|----------|--------|
| `RadiObject.from_images()` | Small-medium datasets (<1000 subjects) | Fits in memory |
| `VolumeCollectionWriter` | Large single-collection datasets | Constant |
| `RadiObjectWriter` | Complex multi-collection builds | Controlled batches |

**Decision guide:**

1. **Can all data fit in memory?** Use `RadiObject.from_images()`
2. **Single collection, too large for memory?** Use `VolumeCollectionWriter`
3. **Multiple collections with custom logic?** Use `RadiObjectWriter`

## Creating a RadiObject

`from_images()` auto-detects NIfTI and DICOM sources per collection:

```python
from radiobject import RadiObject

radi = RadiObject.from_images(
    uri="./my-dataset",
    images={
        "CT": "./imagesTr/*.nii.gz",      # Glob pattern (NIfTI)
        "seg": "./labelsTr",               # Directory path (NIfTI)
    },
    validate_alignment=True,               # Ensure matching subject IDs
    obs_meta=metadata_df,                  # Optional subject metadata
    progress=True,
)
```

**Source formats** (values in `images` dict):

| Format | Auto-detected as | Example |
|--------|-----------------|---------|
| Glob pattern | NIfTI | `"./data/*.nii.gz"` |
| Directory (contains `.nii`) | NIfTI | `"./data/imagesTr"` |
| Directory (contains DICOM subdirs) | DICOM | `"./data/dicom_series"` |
| Pre-resolved list | Inspects first path | `[(path, subject_id), ...]` |

**Options:**

| Parameter | Description |
|-----------|-------------|
| `validate_alignment` | Verify all collections have matching subject IDs |
| `obs_meta` | DataFrame with subject-level metadata (must have `obs_subject_id` column) |
| `reorient` | Reorient volumes to canonical orientation during ingestion |
| `format_hint` | Dict mapping collection names to `"nifti"` or `"dicom"` for ambiguous sources |
| `progress` | Show progress bar |

### DICOM Sources

DICOM directories work through the same `images` dict. Each immediate subdirectory is treated as one DICOM series:

```python
radi = RadiObject.from_images(
    uri="./dicom-dataset",
    images={
        "CT_head": [
            ("./dicom/sub01/CT_HEAD", "sub-01"),
            ("./dicom/sub02/CT_HEAD", "sub-02"),
        ],
    },
    progress=True,
)
```

For ambiguous directories, use `format_hint`:

```python
radi = RadiObject.from_images(
    uri="./study",
    images={"CT": "/ambiguous/directory/"},
    format_hint={"CT": "dicom"},
)
```

## Streaming Writes

For datasets too large to fit in memory, use streaming writers to build RadiObjects incrementally.

### VolumeCollectionWriter

Incremental writes to a single VolumeCollection:

```python
from radiobject.writers import VolumeCollectionWriter

with VolumeCollectionWriter("./large-collection", name="CT") as writer:
    for path, subject_id in discover_niftis("./source"):
        data = load_nifti(path)
        writer.write_volume(
            data=data,
            obs_id=f"{subject_id}_CT",
            obs_subject_id=subject_id,
            voxel_spacing=(1.0, 1.0, 1.0),
        )

collection = VolumeCollection("./large-collection")
```

`write_volume()` accepts `(X, Y, Z)` or `(X, Y, Z, T)` numpy arrays. Additional keyword arguments become volume-level obs attributes. `write_batch()` accepts a list of `(data, obs_id, obs_subject_id, attrs_dict)` tuples.

### RadiObjectWriter

Builds complete RadiObjects with multiple collections:

```python
from radiobject.writers import RadiObjectWriter
import pandas as pd

with RadiObjectWriter("./large-dataset") as writer:
    writer.write_obs_meta(pd.DataFrame({
        "obs_subject_id": ["sub-01", "sub-02"],
        "age": [45, 52],
        "diagnosis": ["tumor", "healthy"],
    }))

    with writer.add_collection("T1w") as t1_writer:
        for subject_id, data in load_t1w_data():
            t1_writer.write_volume(data, obs_id=f"{subject_id}_T1w", obs_subject_id=subject_id)

    with writer.add_collection("FLAIR") as flair_writer:
        for subject_id, data in load_flair_data():
            flair_writer.write_volume(data, obs_id=f"{subject_id}_FLAIR", obs_subject_id=subject_id)

radi = RadiObject("./large-dataset")
```

Streaming writers work with S3 URIs. For S3 write performance, configure parallel uploads:

```python
from radiobject import configure, S3Config
configure(s3=S3Config(max_parallel_ops=16, multipart_part_size_mb=100))
```

## Appending Data

Add new subjects to existing RadiObjects without rebuilding:

```python
radi = RadiObject("./my-dataset")

new_obs_meta = pd.DataFrame({
    "obs_subject_id": ["sub-100", "sub-101"],
    "age": [38, 45],
    "diagnosis": ["healthy", "tumor"],
})

radi.append(
    images={
        "T1w": [
            ("sub-100_T1w.nii.gz", "sub-100"),
            ("sub-101_T1w.nii.gz", "sub-101"),
        ],
    },
    obs_meta=new_obs_meta,
    progress=True,
)
```

Appends are atomic: both `obs_meta` and volumes are written together. Cached properties are automatically invalidated.

Append directly to a VolumeCollection:

```python
radi.T1w.append(niftis=[("sub-102_T1w.nii.gz", "sub-102")], progress=True)
```

**Constraints:**

1. Append adds to existing collections only. To add new collections, use `add_collection()`.
2. Appended `obs_subject_id` values must not already exist in the RadiObject.
3. New subjects require corresponding metadata in the `obs_meta` DataFrame.

## Handling Orientation

Medical images use coordinate systems (RAS, LPS) to map voxels to physical space. RadiObject can reorient volumes during ingestion.

```python
from radiobject import configure, WriteConfig, OrientationConfig

configure(write=WriteConfig(
    orientation=OrientationConfig(canonical_target="RAS", reorient_on_load=True)
))

radi = RadiObject.from_images(uri, images={"CT": "./data"})
```

| Scenario | Recommendation |
|----------|----------------|
| Multi-site studies with inconsistent orientations | Reorient |
| Preserving original acquisition geometry | Don't reorient |
| ML training (standardized input expected) | Reorient |
| Clinical review (native orientation expected) | Don't reorient |

See [Lexicon: Coordinate Systems](../reference/lexicon.md#coordinate-systems-and-orientation) for terminology.

## 4D Data (fMRI, DTI)

RadiObject natively supports 4D NIfTI volumes. No special configuration needed:

```python
radi = RadiObject.from_images(
    uri="./fmri-study",
    images={"bold": "./func/*_bold.nii.gz"},
)

vol = radi.bold.iloc[0]
print(vol.shape)   # (64, 64, 32, 200) — 200 timepoints
```

**How dimensions work:**

- **Collection shape** reports spatial dimensions only: `(64, 64, 32)`
- **Volume shape** reports the full shape including time: `(64, 64, 32, 200)`
- **Shape validation** compares spatial dims only — volumes with different timepoint counts pass `validate_dimensions=True`

## Managing Existing Data

### Check If Data Exists

```python
from radiobject import uri_exists

if uri_exists("s3://bucket/my-dataset"):
    print("Dataset already exists")
```

### Delete and Reingest

```python
from radiobject import delete_tiledb_uri

delete_tiledb_uri("s3://bucket/my-dataset")
radi = RadiObject.from_images("s3://bucket/my-dataset", images={"CT": "./data"})
```

### Create from Existing VolumeCollections

Assemble a RadiObject from pre-existing VolumeCollections:

```python
new_radi = RadiObject.from_collections(
    uri="./new-dataset",
    collections={"T1w": existing_collection},
    obs_meta=metadata_df,
)
```

For full API details, see [RadiObject API](../api/radi_object.md).
