# Lazy Pipelines

The `lazy()` method returns a query builder for ETL pipelines with transforms. Unlike direct filtering, lazy queries accumulate operations without touching data until explicit materialization. For the full API, see [Query API](../api/query.md).

## Basic Usage

```python
radi = RadiObject("s3://bucket/study")

# Build a lazy query
result = (
    radi.lazy()
    .filter("age > 40 and diagnosis == 'tumor'")
    .select_collections(["T1w", "FLAIR"])
    .sample(100, seed=42)
    .map(normalize_intensity)  # Transform during materialization
    .materialize("s3://bucket/subset", streaming=True)
)
```

## Query vs Direct Filtering

| Feature | Direct Filter | Lazy Query |
|---------|---------------|------------|
| Returns | View (`RadiObject` with `is_view=True`) | `Query` builder |
| Execution | Immediate | Deferred |
| Transforms | Not supported | `map()` support |
| Memory | Loads data on access | Streaming capable |
| Use case | Exploration, debugging | ETL pipelines, ML data prep |

## Filter Chaining

Chain multiple filters that combine with AND logic:

```python
query = (
    radi.lazy()
    .filter("age > 40")
    .filter("diagnosis == 'tumor'")
    .filter("split == 'train'")
    .select_collections(["T1w"])
)
```

## Apply Transforms with map()

The `map()` method applies a transform function during materialization:

```python
def normalize_intensity(volume: np.ndarray) -> np.ndarray:
    """Normalize to zero mean, unit variance."""
    return (volume - volume.mean()) / volume.std()

def resample_to_1mm(volume: np.ndarray) -> np.ndarray:
    """Resample to 1mm isotropic."""
    # Implementation using scipy.ndimage
    ...

# Transforms compose - applied in order
query = (
    radi.lazy()
    .filter("split == 'train'")
    .map(normalize_intensity)
    .map(resample_to_1mm)
    .materialize("s3://bucket/processed")
)
```

## Materialization Methods

### materialize()

Write query results to new storage:

```python
# Write to new RadiObject
query.materialize("s3://bucket/subset")

# Enable streaming for large datasets
query.materialize("s3://bucket/subset", streaming=True)
```

### iter_volumes()

Stream volumes one at a time:

```python
for vol in query.iter_volumes():
    data = vol.to_numpy()
    process(data)
```

### iter_batches()

Stream batches for ML training:

```python
for batch in radi.lazy().filter("split == 'train'").iter_batches(batch_size=32):
    train_step(batch.volumes["T1w"], batch.volumes["FLAIR"])
```

The `VolumeBatch` dataclass contains:

- `volumes`: Dict mapping collection names to stacked numpy arrays `(N, X, Y, Z)`
- `subject_ids`: Tuple of subject IDs in the batch
- `obs_ids`: Dict mapping collection names to tuples of volume IDs

### count()

Count results without loading volume data:

```python
result = query.count()
print(f"Subjects: {result.n_subjects}")
print(f"T1w volumes: {result.n_volumes['T1w']}")
```

## VolumeCollection Queries

VolumeCollections also support lazy queries via `CollectionQuery`:

```python
collection = radi.T1w

# Filter on volume-level obs attributes
query = collection.lazy().filter("voxel_spacing == '(1.0, 1.0, 1.0)'")

# Iterate results
for vol in query.iter_volumes():
    process(vol)

# Stack all matching volumes as (N, X, Y, Z) array
data = collection.lazy().to_numpy_stack()
```

## When to Use Lazy Mode

| Use Case | Mode | Why |
|----------|------|-----|
| Quick inspection | Direct | Immediate feedback |
| Jupyter notebooks | Direct | Interactive feel |
| Subsetting for export | Direct + `materialize()` | Explicit write |
| Apply transforms | **Lazy** | Transform during materialization |
| ML training data | **Lazy** | Batched iteration, memory control |
| ETL pipelines | **Lazy** | Streaming, composable |

For most use cases, direct filtering is sufficient. Use `lazy()` only when you need:

- Transforms via `map()`
- Memory-controlled streaming via `iter_batches()`
- Complex pipelines with deferred execution

## Next Step

**Ready for ML training?** Set up MONAI or TorchIO DataLoaders with [ML Integration](ml-training.md).

## Related Documentation

- [Indexing & Filtering](query-filter-data.md) - Direct filtering for exploration
- [Volume Operations](volume-operations.md) - Working with individual volumes
- [Streaming Writes](streaming-writes.md) - Large dataset writes
