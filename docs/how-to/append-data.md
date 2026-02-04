# Append Data

Add new subjects and volumes to existing RadiObjects without rebuilding from scratch.

## Basic Append

The `append()` method atomically adds new subjects:

```python
from radiobject import RadiObject
import pandas as pd

# Open existing RadiObject
radi = RadiObject("./my-dataset")

# Prepare new subject metadata
new_obs_meta = pd.DataFrame({
    "obs_subject_id": ["sub-100", "sub-101"],
    "age": [38, 45],
    "diagnosis": ["healthy", "tumor"],
})

# Prepare new volumes (dict mapping collection names to volume data)
new_volumes = {
    "T1w": {
        "sub-100": load_volume("sub-100_T1w.nii.gz"),
        "sub-101": load_volume("sub-101_T1w.nii.gz"),
    },
    "FLAIR": {
        "sub-100": load_volume("sub-100_FLAIR.nii.gz"),
        "sub-101": load_volume("sub-101_FLAIR.nii.gz"),
    },
}

# Atomic append
radi.append(obs_meta=new_obs_meta, volumes=new_volumes)
```

## Atomic Writes

Appends are atomic: both `obs_meta` and volumes are written together to maintain referential integrity. If either write fails, the operation is rolled back.

## Cache Invalidation

After an append, cached properties are automatically invalidated:

```python
print(len(radi))  # Before: 50 subjects

radi.append(obs_meta=new_obs_meta, volumes=new_volumes)

print(len(radi))  # After: 52 subjects (cache refreshed)
```

Invalidated caches include:

- `_index` - Subject index mapping
- `_metadata` - Group metadata
- `collection_names` - Available collections

## VolumeCollection Append

Append volumes directly to a VolumeCollection:

```python
collection = radi.T1w

# Append single volume
collection.append(
    data=volume_array,
    obs_id="sub-102_T1w",
    obs_subject_id="sub-102",
)

# Append multiple volumes
collection.append_batch([
    {"data": arr1, "obs_id": "sub-103_T1w", "obs_subject_id": "sub-103"},
    {"data": arr2, "obs_id": "sub-104_T1w", "obs_subject_id": "sub-104"},
])
```

## Append to S3

Appending works with S3-backed RadiObjects:

```python
radi = RadiObject("s3://bucket/my-dataset")
radi.append(obs_meta=new_obs_meta, volumes=new_volumes)
```

For best S3 performance, configure parallel operations:

```python
from radiobject import configure
from radiobject.ctx import S3Config

configure(s3=S3Config(max_parallel_ops=16))
```

## Append vs Streaming Writes

| Scenario | Use |
|----------|-----|
| Adding a few subjects to existing data | `append()` |
| Building new RadiObject from large source | [Streaming Writes](streaming-writes.md) |
| Incremental pipeline output | `append()` in a loop |
| Initial bulk ingestion | `from_niftis()` or `RadiObjectWriter` |

## Constraints

1. **Existing collections only**: Append adds to existing collections. To add new collections, use `add_collection()` instead.

2. **Subject ID uniqueness**: Appended `obs_subject_id` values must not already exist in the RadiObject.

3. **Volume ID uniqueness**: Appended `obs_id` values must be unique across all collections.

## Related Documentation

- [Streaming Writes](streaming-writes.md) - Build large RadiObjects incrementally
- [Ingest Data](ingest-data.md) - Initial data ingestion
- [Query & Filter](query-filter-data.md) - Access appended data
