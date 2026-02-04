# Streaming Writes

For datasets too large to fit in memory, use streaming writers to build RadiObjects incrementally.

## StreamingWriter for VolumeCollections

`StreamingWriter` enables incremental writes to a single VolumeCollection:

```python
from radiobject import VolumeCollection
from radiobject.streaming import StreamingWriter

# Create collection incrementally
with StreamingWriter("./large-collection", name="CT") as writer:
    for path, subject_id in discover_niftis("./source"):
        data = load_nifti(path)
        writer.write_volume(
            data=data,
            obs_id=f"{subject_id}_CT",
            obs_subject_id=subject_id,
            voxel_spacing=(1.0, 1.0, 1.0),
        )

# Access the created collection
collection = VolumeCollection("./large-collection")
```

### write_volume()

Write a single volume:

```python
writer.write_volume(
    data=volume_array,         # np.ndarray (X, Y, Z) or (X, Y, Z, T)
    obs_id="unique_volume_id", # Unique within RadiObject
    obs_subject_id="subject",  # Links to obs_meta
    voxel_spacing=(1.0, 1.0, 1.0),  # Optional metadata
    series_type="T1w",         # Optional metadata
)
```

### write_batch()

Write multiple volumes at once for better performance:

```python
volumes = [
    (arr1, "vol1", "sub1", {"voxel_spacing": (1.0, 1.0, 1.0)}),
    (arr2, "vol2", "sub2", {"voxel_spacing": (1.0, 1.0, 1.0)}),
]
writer.write_batch(volumes)
```

Each tuple is `(data, obs_id, obs_subject_id, attrs_dict)`.

## RadiObjectWriter for Full RadiObjects

`RadiObjectWriter` builds complete RadiObjects with multiple collections:

```python
from radiobject.streaming import RadiObjectWriter
import pandas as pd

# Build RadiObject with multiple collections
with RadiObjectWriter("./large-dataset") as writer:
    # Write subject metadata
    obs_meta = pd.DataFrame({
        "obs_subject_id": ["sub-01", "sub-02"],
        "age": [45, 52],
        "diagnosis": ["tumor", "healthy"],
    })
    writer.write_obs_meta(obs_meta)

    # Add T1w collection
    with writer.add_collection("T1w") as t1_writer:
        for subject_id, data in load_t1w_data():
            t1_writer.write_volume(data, obs_id=f"{subject_id}_T1w", obs_subject_id=subject_id)

    # Add FLAIR collection
    with writer.add_collection("FLAIR") as flair_writer:
        for subject_id, data in load_flair_data():
            flair_writer.write_volume(data, obs_id=f"{subject_id}_FLAIR", obs_subject_id=subject_id)

# Finalize and access
radi = RadiObject("./large-dataset")
```

## Writing to S3

Streaming writers work with S3 URIs:

```python
with RadiObjectWriter("s3://bucket/large-dataset") as writer:
    writer.write_obs_meta(metadata_df)
    with writer.add_collection("CT") as ct_writer:
        for subject_id, data in stream_source_data():
            ct_writer.write_volume(data, obs_id=f"{subject_id}_CT", obs_subject_id=subject_id)
```

For S3 performance, configure parallel uploads:

```python
from radiobject import configure, S3Config

configure(s3=S3Config(
    max_parallel_ops=16,
    multipart_part_size_mb=100,
))
```

## Memory Management

Streaming writers process one volume at a time, keeping memory usage constant:

```python
# Memory-efficient processing of 10,000 volumes
with StreamingWriter(uri, name="CT") as writer:
    for i, (path, subject_id) in enumerate(nifti_paths):
        data = nibabel.load(path).get_fdata()  # Load one volume
        writer.write_volume(data, obs_id=f"{subject_id}_CT", obs_subject_id=subject_id)
        # Memory released after write completes

        if i % 100 == 0:
            print(f"Processed {i} volumes")
```

## Use Cases

| Scenario | Approach |
|----------|----------|
| Large dataset ingestion | `RadiObjectWriter` with `add_collection()` |
| Single collection from stream | `StreamingWriter` |
| Processing pipeline output | `StreamingWriter` with transforms |
| Incremental updates | [Append Data](append-data.md) |

## Related Documentation

- [VolumeCollection API](../api/volume_collection.md) - Full `StreamingWriter` reference
- [Append Data](append-data.md) - Add subjects to existing RadiObjects
- [Ingest Data](ingest-data.md) - Standard ingestion APIs
- [Tuning Concurrency](tuning-concurrency.md) - S3 write optimization
