# Lazy Pipelines

The `lazy()` method returns a query builder for ETL pipelines with transforms. Unlike direct filtering, lazy queries accumulate operations without touching data until explicit materialization. For the full API, see [Query API](../api/query.md).

## When to Use Lazy

| Use Case | Mode | Why |
|----------|------|-----|
| Quick inspection, Jupyter | Direct (`iloc`, `filter`) | Immediate feedback |
| Apply transforms (normalize, resample) | **Lazy** | `map()` during materialization |
| Stream large datasets | **Lazy** | Memory-controlled iteration |
| ETL pipelines | **Lazy** | Composable, deferred execution |

Use direct filtering for exploration. Use `lazy()` when you need transforms, streaming, or deferred execution.

## Building Queries

```python
radi = RadiObject("s3://bucket/study")

result = (
    radi.lazy()
    .filter("age > 40 and diagnosis == 'tumor'")
    .select_collections(["T1w", "FLAIR"])
    .sample(100, seed=42)
    .map(normalize_intensity)
    .materialize("s3://bucket/subset", streaming=True)
)
```

Chain multiple filters (combined with AND logic):

```python
query = (
    radi.lazy()
    .filter("age > 40")
    .filter("diagnosis == 'tumor'")
    .filter("split == 'train'")
    .select_collections(["T1w"])
)
```

Subject-level and collection-level filtering:

```python
query = radi.lazy().filter_subjects(["sub-01", "sub-02"])
query = radi.lazy().filter_collection("T1w", "voxel_spacing == '(1.0, 1.0, 1.0)'")
```

## Transforms

`map()` applies a transform function during materialization:

```python
def normalize_intensity(volume: np.ndarray) -> np.ndarray:
    return (volume - volume.mean()) / volume.std()

def resample_to_1mm(volume: np.ndarray) -> np.ndarray:
    ...  # scipy.ndimage implementation

# Transforms compose — applied in order
query = (
    radi.lazy()
    .filter("split == 'train'")
    .map(normalize_intensity)
    .map(resample_to_1mm)
    .materialize("s3://bucket/processed")
)
```

## Streaming

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

`VolumeBatch` contains:

- `volumes`: Dict mapping collection names to stacked numpy arrays `(N, X, Y, Z)`
- `subject_ids`: Tuple of subject IDs in the batch
- `obs_ids`: Dict mapping collection names to tuples of volume IDs

### count()

Count results without loading volume data:

```python
result = query.count()
print(f"Subjects: {result.n_subjects}, T1w: {result.n_volumes['T1w']}")
```

## Materialization

Write query results to new storage:

```python
query.materialize("s3://bucket/subset")
query.materialize("s3://bucket/subset", streaming=True)  # For large datasets
```

## VolumeCollection Queries

VolumeCollections also support lazy queries via `CollectionQuery`:

```python
query = radi.T1w.lazy().filter("voxel_spacing == '(1.0, 1.0, 1.0)'")

for vol in query.iter_volumes():
    process(vol)

# Stack all matching volumes as (N, X, Y, Z) array
data = radi.T1w.lazy().to_numpy_stack()
```
