# Append Data

Add new subjects and volumes to existing RadiObjects without rebuilding from scratch.

## RadiObject Append

The `append()` method atomically adds new subjects from NIfTI files:

```python
from radiobject import RadiObject
import pandas as pd

radi = RadiObject("./my-dataset")

# Prepare metadata for new subjects
new_obs_meta = pd.DataFrame({
    "obs_subject_id": ["sub-100", "sub-101"],
    "age": [38, 45],
    "diagnosis": ["healthy", "tumor"],
})

# Append NIfTI files — tuples of (path, obs_subject_id)
radi.append(
    niftis=[
        ("sub-100_T1w.nii.gz", "sub-100"),
        ("sub-101_T1w.nii.gz", "sub-101"),
    ],
    obs_meta=new_obs_meta,
    progress=True,
)
```

### Atomic Writes

Appends are atomic: both `obs_meta` and volumes are written together to maintain referential integrity.

### Cache Invalidation

After an append, cached properties (`_index`, `_metadata`, `collection_names`) are automatically invalidated:

```python
print(len(radi))  # Before: 50 subjects

radi.append(niftis=new_niftis, obs_meta=new_obs_meta)

print(len(radi))  # After: 52 subjects (cache refreshed)
```

## VolumeCollection Append

Append volumes directly to a VolumeCollection:

```python
collection = radi.T1w

# Append new NIfTI files
collection.append(
    niftis=[
        ("sub-102_T1w.nii.gz", "sub-102"),
        ("sub-103_T1w.nii.gz", "sub-103"),
    ],
    progress=True,
)
```

## Append to S3

Appending works with S3-backed RadiObjects:

```python
radi = RadiObject("s3://bucket/my-dataset")
radi.append(niftis=new_niftis, obs_meta=new_obs_meta)
```

For best S3 performance, configure parallel operations:

```python
from radiobject import configure, S3Config

configure(s3=S3Config(max_parallel_ops=16))
```

## Append vs Streaming Writes

| Scenario | Use |
|----------|-----|
| Adding a few subjects to existing data | `append()` |
| Building new RadiObject from large source | [Streaming Writes](streaming-writes.md) |
| Initial bulk ingestion | `from_niftis()` or `RadiObjectWriter` |

## Constraints

1. **Existing collections only**: Append adds to existing collections. To add new collections, use `add_collection()`.

2. **Subject ID uniqueness**: Appended `obs_subject_id` values must not already exist in the RadiObject.

3. **obs_meta required**: New subjects require corresponding metadata in the `obs_meta` DataFrame.

## Related Documentation

- [Streaming Writes](streaming-writes.md) - Build large RadiObjects incrementally
- [Ingest Data](ingest-data.md) - Initial data ingestion
- [Indexing & Filtering](query-filter-data.md) - Access appended data
