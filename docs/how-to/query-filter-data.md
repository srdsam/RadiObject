# Query & Filter Data

RadiObject provides pandas-like indexing and filtering for accessing stored data.

## Choosing Your Approach

| I want to... | Use | Guide |
|--------------|-----|-------|
| Explore data interactively | Direct filtering (`iloc`, `loc`, `filter()`) | This page |
| Apply transforms (normalize, resample) | Lazy queries with `map()` | [Lazy Queries](lazy-queries.md) |
| Stream large datasets without loading all | `iter_volumes()` / `iter_batches()` | [Lazy Queries - Materialization](lazy-queries.md#materialization-methods) |
| Write filtered results to new storage | `materialize()` | [Lazy Queries](lazy-queries.md#materialize) |

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

Views provide immediate access to collections and volumes:

```python
# Access collections within view
vol = subset.T1w.iloc[0]        # Get first T1w volume
data = vol.to_numpy()           # Load into memory

# Partial reads for slices
axial_slice = vol.axial(77)     # Single 2D slice
patch = vol.slice(x=slice(50, 114), y=slice(50, 114), z=slice(30, 94))  # 64³ ROI
```

## Filter Expressions

The `filter()` method accepts TileDB QueryCondition expressions on `obs_meta` columns:

```python
# Single condition
subset = radi.filter("age > 40")

# Compound conditions
subset = radi.filter("age > 40 and diagnosis == 'tumor'")

# Combined with collection selection
subset = radi.filter("split == 'train'").select_collections(["T1w", "FLAIR"])
```

## Materializing Views

Export a view to new storage:

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

For ETL pipelines with transforms, use [Lazy Queries](lazy-queries.md) instead.

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

## Related Documentation

- [Ingest Data](ingest-data.md) - Create RadiObjects from NIfTI/DICOM
- [Lazy Queries](lazy-queries.md) - ETL pipelines with transforms
- [Configuration](../reference/configuration.md) - Read performance settings
