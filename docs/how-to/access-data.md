# Access Data

Indexing, filtering, metadata inspection, volume reads, and cross-collection alignment. For the full API, see [RadiObject API](../api/radi_object.md) and [VolumeCollection API](../api/volume_collection.md).

## Explore

Start by inspecting what's in a RadiObject:

```python
radi = RadiObject("s3://bucket/study")

print(radi.describe())          # Summary: subjects, collections, shapes, labels
radi.collection_names            # ("T1w", "FLAIR", "T1gd", "T2w")
len(radi)                        # 484
```

Read subject metadata:

```python
obs = radi.obs_meta.read()
print(obs.head())
# Common columns: obs_subject_id, age, diagnosis, split
```

Read volume-level metadata:

```python
vol_obs = radi.T1w.obs.read()
print(vol_obs.head())
# Common columns: obs_id, obs_subject_id, voxel_spacing, dimensions
```

## Index

Direct indexing returns lightweight views for immediate access:

```python
subset = radi.iloc[0:10]              # First 10 subjects
subset = radi.loc["sub-01"]           # Single subject by ID
subset = radi.head(5)                 # Quick preview
subset = radi.tail(5)                 # Last 5 subjects
subset = radi.sample(10, seed=42)     # Random sample
```

`sel()` selects by `obs_subject_id` without requiring knowledge of the obs_id naming convention:

```python
subject = radi.sel(subject="BraTS001")
multi = radi.sel(subject=["BraTS001", "BraTS002"])

# On VolumeCollection — returns Volume (single match) or view (multiple)
vol = radi.T1w.sel(subject="BraTS001")
```

VolumeCollections support the same patterns:

```python
vol = radi.T1w.iloc[0]                # By position
vol = radi.T1w.loc["BraTS001_T1w"]    # By volume ID
```

## Filter

`filter()` accepts [TileDB QueryCondition](https://docs.tiledb.com/main/how-to/arrays/read/query-condition) expressions on `obs_meta` columns:

```python
subset = radi.filter("age > 40")
subset = radi.filter("age > 40 and diagnosis == 'tumor'")
subset = radi.filter("split == 'train'").select_collections(["T1w", "FLAIR"])
```

Filter during metadata reads (more efficient than pandas filtering for large datasets):

```python
obs = radi.obs_meta.read(value_filter="age > 40 and diagnosis == 'tumor'")
```

## Metadata

RadiObject stores two levels of metadata:

| Level | Attribute | Description |
|-------|-----------|-------------|
| Subject | `radi.obs_meta` | One row per subject (demographics, diagnosis, split) |
| Volume | `collection.obs` | One row per volume (voxel_spacing, dimensions, series_type) |

### Joining Subject and Volume Metadata

```python
subject_obs = radi.obs_meta.read()
vol_obs = radi.T1w.obs.read()

merged = vol_obs.merge(subject_obs, on="obs_subject_id", how="left")
print(merged[["obs_id", "voxel_spacing", "age", "diagnosis"]])
```

### Editing Metadata

```python
import numpy as np

# Add a column
radi.obs_meta.add_column("reviewed", np.bool_, fill=False)

# Update values
annotations = pd.DataFrame({
    "obs_subject_id": ["subj_1"],
    "obs_id": ["vol_1"],
    "reviewed": [True],
})
radi.obs_meta.update(annotations)

# Remove a column
radi.obs_meta.drop_column("obsolete_field")

# Delete rows
radi.obs_meta.delete("split == 'excluded'")
```

### Common Patterns

```python
subject_ids = radi.obs_subject_ids       # Cached, O(1)
collections = radi.collection_names      # e.g., ("T1w", "FLAIR")

obs = radi.obs_meta.read()
print(obs["diagnosis"].value_counts())   # Count subjects by category
```

## Volumes

### Partial Reads

RadiObject stores volumes as tiled arrays, enabling reads of individual slices or sub-regions without loading the full volume. This is 200-600x faster than NIfTI for single slices (see [Benchmarks](../reference/benchmarks.md)).

```python
vol = radi.CT.iloc[0]

# Named slice methods
axial = vol.axial(z=77)          # (240, 240)
sagittal = vol.sagittal(x=120)   # (240, 155)
coronal = vol.coronal(y=120)     # (240, 155)

# Arbitrary 3D patch
patch = vol.slice(
    x=slice(50, 114), y=slice(50, 114), z=slice(30, 94)
)  # (64, 64, 64)

# NumPy-style indexing
patch = vol[50:114, 50:114, 30:94]

# For 4D volumes, pass temporal index
axial = vol.axial(z=77, t=0)
```

### Full Volume Read

```python
data = vol.to_numpy()  # np.ndarray, shape matches vol.shape
```

### Volume Properties

```python
vol.shape              # (240, 240, 155)
vol.ndim               # 3 (or 4 for temporal data)
vol.dtype              # float32
vol.obs_id             # "BraTS001_CT"
vol.tile_orientation   # SliceOrientation.AXIAL
vol.orientation_info   # OrientationInfo(axcodes=('R', 'A', 'S'), ...)
```

### Statistics and Histogram

```python
stats = vol.get_statistics()
# {'mean': 142.3, 'std': 89.1, 'min': 0.0, 'max': 1024.0, 'median': 118.0}

stats = vol.get_statistics(percentiles=[5, 25, 75, 95])

counts, bin_edges = vol.compute_histogram(bins=256)
counts, bin_edges = vol.compute_histogram(bins=100, value_range=(0.0, 500.0))
```

### NIfTI Export

```python
vol.to_nifti("./output/scan.nii.gz")
vol.to_nifti("./output/scan.nii", compression=False)
```

Export preserves the original affine matrix, sform/qform codes, and scaling parameters.

## Cross-Collection Alignment

Use the `subjects` property and Index set operations to validate and compare collections:

```python
radi.T1w.subjects.is_aligned(radi.seg.subjects)  # True

common = radi.T1w.subjects & radi.seg.subjects    # Intersection
only_t1w = radi.T1w.subjects - radi.seg.subjects  # Difference

for subject_id, vols in radi.T1w.groupby_subject():
    for vol in vols:
        print(f"{subject_id}: mean={vol.to_numpy().mean():.1f}")
```

## Write Views to Storage

Views can be exported to new storage:

```python
subset = radi.filter("split == 'train'")
subset.materialize("s3://bucket/subset")

radi_subset = RadiObject("s3://bucket/subset")
```

For transform pipelines (eager and lazy), see [Pipelines](pipelines.md).
