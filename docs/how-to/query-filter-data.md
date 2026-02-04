# Indexing & Filtering

RadiObject provides pandas-like indexing and filtering for accessing stored data. For the full API, see [RadiObject API](../api/radi_object.md) and [Query API](../api/query.md).

## Exploring Your Data

Start by inspecting what's in a RadiObject before filtering:

```python
radi = RadiObject("s3://bucket/study")

# Summary: subjects, collections, shapes, label distributions
print(radi.describe())

# List available collections
radi.collection_names  # ["T1w", "FLAIR", "T1gd", "T2w"]

# Count subjects
len(radi)  # 484
```

## Choosing Your Approach

| I want to... | Use | Guide |
|--------------|-----|-------|
| Explore data interactively | Direct filtering (`iloc`, `loc`, `filter()`) | This page |
| Apply transforms (normalize, resample) | Lazy queries with `map()` | [Lazy Pipelines](lazy-queries.md) |
| Stream large datasets without loading all | `iter_volumes()` / `iter_batches()` | [Lazy Pipelines: Materialization](lazy-queries.md#materialization-methods) |
| Write filtered results to new storage | `materialize()` | [Lazy Pipelines](lazy-queries.md#materialize) |

> **Rule of thumb**: Use direct filtering for exploration and debugging. Use lazy queries when you need transforms or memory-controlled streaming.

## Direct Indexing

Direct indexing with `iloc`, `loc`, `head()`, `tail()`, `sample()`, and `filter()` returns lightweight views for immediate access:

```python
radi = RadiObject("s3://bucket/study")

# Filtering returns views (RadiObject with is_view=True)
subset = radi.iloc[0:10]              # First 10 subjects
subset = radi.loc["sub-01"]           # Single subject by ID
subset = radi.head(5)                 # Quick preview
subset = radi.filter("age > 40")      # Query expression
subset = radi.select_collections(["T1w", "FLAIR"])

# Views are immediate - no deferred execution
subset.is_view  # True
len(subset)     # Works immediately
```

## Accessing Volumes

Views provide immediate access to collections and volumes. For partial reads, statistics, and export, see [Volume Operations](volume-operations.md).

```python
# Access collections within view
vol = subset.T1w.iloc[0]        # Get first T1w volume
data = vol.to_numpy()           # Load into memory

# Partial reads for slices
axial_slice = vol.axial(z=77)   # Single 2D slice
patch = vol.slice(x=slice(50, 114), y=slice(50, 114), z=slice(30, 94))  # 64³ ROI
```

## Filter Expressions

The `filter()` method accepts [TileDB QueryCondition](https://docs.tiledb.com/main/how-to/arrays/read/query-condition) expressions on `obs_meta` columns:

```python
# Single condition
subset = radi.filter("age > 40")

# Compound conditions
subset = radi.filter("age > 40 and diagnosis == 'tumor'")

# Combined with collection selection
subset = radi.filter("split == 'train'").select_collections(["T1w", "FLAIR"])
```

## Materializing Views

Views can be exported to new storage with `materialize()`. For streaming materialization and transform pipelines, see [Lazy Pipelines: Materialization](lazy-queries.md#materialization-methods).

```python
# Copy filtered data to new location
subset.materialize("s3://bucket/subset")

# Access the new RadiObject
subset_radi = RadiObject("s3://bucket/subset")
```

## When to Use Direct Filtering

Views provide immediate access to data and feel like working with pandas DataFrames. Use this mode for:

- Interactive exploration
- Debugging
- Analysis notebooks
- Quick data inspection

For ETL pipelines with transforms, use [Lazy Pipelines](lazy-queries.md) instead.

## Subject-Based Selection with `sel()`

`sel()` selects by `obs_subject_id` without requiring knowledge of the `obs_id` naming convention:

```python
# On RadiObject — returns a view
subject_view = radi.sel(subject="BraTS001")
multi_view = radi.sel(subject=["BraTS001", "BraTS002"])

# On VolumeCollection — returns Volume (single match) or view (multiple)
vol = radi.T1w.sel(subject="BraTS001")       # Volume
data = vol.to_numpy()

# Cross-modal access without string construction
for mod in ["FLAIR", "T1w", "T1gd", "T2w"]:
    vol = radi.collection(mod).sel(subject="BraTS001")
    print(vol.to_numpy().mean())
```

## Cross-Collection Alignment

Use the `subjects` property and Index set operations to validate and compare collections:

```python
# Check that two modalities have matching subjects
radi.T1w.subjects.is_aligned(radi.seg.subjects)  # True

# Find subjects present in both
common = radi.T1w.subjects & radi.seg.subjects

# Per-subject iteration
for subject_id, vols in radi.T1w.groupby_subject():
    for vol in vols:
        print(f"{subject_id}: mean={vol.to_numpy().mean():.1f}")
```

## VolumeCollection Indexing

VolumeCollections support the same indexing patterns:

```python
collection = radi.T1w

# By position
vol = collection.iloc[0]

# By volume ID
vol = collection.loc["BraTS001_T1w"]

# Iteration
for vol in collection:
    process(vol.to_numpy())
```

## Next Step

**Found the data you need?** Read slices and compute statistics with [Volume Operations](volume-operations.md), or build transform pipelines with [Lazy Pipelines](lazy-queries.md).

## Related Documentation

- [Working with Metadata](working-with-metadata.md) - Subject and volume metadata
- [Ingest Data](ingest-data.md) - Create RadiObjects from NIfTI/DICOM
- [Configuration](../reference/configuration.md) - Read performance settings
