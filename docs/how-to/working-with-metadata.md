# Working with Metadata

How to browse and query subject-level metadata (`obs_meta`) and volume-level metadata (`obs`).

## Metadata Overview

RadiObject stores two levels of metadata:

| Level | Attribute | Description |
|-------|-----------|-------------|
| Subject | `radi.obs_meta` | One row per subject (demographics, diagnosis, split) |
| Volume | `collection.obs` | One row per volume (voxel_spacing, dimensions, series_type) |

## Viewing Subject Metadata

Subject metadata is stored in the `obs_meta` TileDB DataFrame:

```python
radi = RadiObject("s3://bucket/study")

# Read all subject metadata as pandas DataFrame
obs = radi.obs_meta.read()
print(obs.head())

# Available columns depend on your data
# Common: obs_subject_id, age, diagnosis, split, site
```

### Filtering with QueryConditions

Use TileDB query conditions to filter subjects efficiently:

```python
# Filter during read (more efficient than pandas filtering)
obs = radi.obs_meta.read(query_condition="age > 40 and diagnosis == 'tumor'")

# Or use RadiObject filter (returns view)
subset = radi.filter("split == 'train'")
```

## Viewing Volume Metadata

Each VolumeCollection has an `obs` attribute with volume-level metadata:

```python
collection = radi.T1w

# Read volume metadata
vol_obs = collection.obs.read()
print(vol_obs.head())

# Common columns: obs_id, obs_subject_id, voxel_spacing, shape, orientation
```

### Per-Volume Attributes

Individual volumes expose metadata as properties:

```python
vol = radi.T1w.iloc[0]

# Access volume properties
print(vol.shape)          # (240, 240, 155)
print(vol.voxel_spacing)  # (1.0, 1.0, 1.0)
print(vol.obs_id)         # "BraTS001_T1w"
```

## Joining Metadata

Correlate subject and volume metadata using `obs_subject_id`:

```python
# Get subject metadata
subject_obs = radi.obs_meta.read()

# Get volume metadata for T1w collection
vol_obs = radi.T1w.obs.read()

# Join on subject ID
merged = vol_obs.merge(
    subject_obs,
    on="obs_subject_id",
    how="left"
)

# Now you have volume info + subject demographics
print(merged[["obs_id", "voxel_spacing", "age", "diagnosis"]])
```

## Common Patterns

### List All Subject IDs

```python
subject_ids = radi.obs_subject_ids  # Cached property, O(1)
```

### List All Collection Names

```python
collections = radi.collection_names  # e.g., ["T1w", "FLAIR", "T1gd", "T2w"]
```

### Count Subjects by Category

```python
obs = radi.obs_meta.read()
print(obs["diagnosis"].value_counts())
```

### Find Subjects with Missing Modalities

```python
# Check which subjects have all expected modalities
for sid in radi.obs_subject_ids:
    subject = radi.loc[sid]
    collections = subject.collection_names
    if "FLAIR" not in collections:
        print(f"{sid} missing FLAIR")
```

## Related Documentation

- [Query & Filter](query-filter-data.md) - Filter subjects using metadata
- [Ingest Data](ingest-data.md) - How metadata is populated during ingestion
- [Dataframe API](../api/dataframe.md) - TileDB DataFrame reference
