# Data Access Patterns

RadiObject provides patterns for data ingestion, filtering, and access, each designed for different workflows.

## Ingestion

The `images` dict API is the recommended way to create a RadiObject from NIfTI files:

```python
from radiobject import RadiObject

# Ingest multiple collections in a single call
radi = RadiObject.from_niftis(
    uri="./my-dataset",
    images={
        "CT": "./imagesTr/*.nii.gz",      # Glob pattern
        "seg": "./labelsTr",               # Directory path
    },
    validate_alignment=True,               # Ensure matching subject IDs
    obs_meta=metadata_df,                  # Optional subject metadata
    progress=True,
)
```

**Source formats** (values in `images` dict):

| Format | Example | Description |
|--------|---------|-------------|
| Glob pattern | `"./data/*.nii.gz"` | Match files by pattern |
| Directory | `"./data/imagesTr"` | Discover all NIfTIs in directory |
| Pre-resolved list | `[(path, subject_id), ...]` | Explicit mapping |

**Options:**

| Parameter | Description |
|-----------|-------------|
| `validate_alignment` | Verify all collections have matching subject IDs |
| `obs_meta` | DataFrame with subject-level metadata (must have `obs_subject_id` column) |
| `reorient` | Reorient volumes to canonical orientation during ingestion |
| `progress` | Show progress bar |

**Legacy APIs** (still supported):

```python
# Directory-based (single collection)
radi = RadiObject.from_niftis(uri, image_dir="./data", collection_name="CT")

# Tuple list with auto-grouping by modality
radi = RadiObject.from_niftis(uri, niftis=[(path, subject_id), ...])
```

## Filtering (Returns Views)

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

# Access collections within view
vol = subset.T1w.iloc[0]        # Get first T1w volume
data = vol.to_numpy()           # Load into memory

# Materialize view to new storage
subset.materialize("s3://bucket/subset")
```

Views provide immediate access to data and feel like working with pandas DataFrames. Use this mode for interactive exploration, debugging, and analysis notebooks.

## Lazy Mode (Transform Pipelines)

The `lazy()` method returns a lazy filter builder for ETL pipelines with transforms:

```python
# Lazy filtering with transforms
result = (
    radi.lazy()
    .filter("age > 40 and diagnosis == 'tumor'")
    .select_collections(["T1w", "FLAIR"])
    .sample(100, seed=42)
    .map(normalize_intensity)  # Transform during materialization
    .materialize("s3://bucket/subset", streaming=True)
)

# Streaming iteration for ML training
for batch in radi.lazy().filter("split == 'train'").iter_batches(batch_size=32):
    train_step(batch.volumes["T1w"], batch.volumes["FLAIR"])
```

`Query` accumulates filters without touching data. Explicit materialization methods (`materialize()`, `iter_volumes()`, `count()`) trigger execution. Use lazy mode when you need to apply transforms via `map()`.

## When to Use Each

| Use Case | Mode | Method | Why |
|----------|------|--------|-----|
| Quick inspection | Filter | `head()`, `iloc[]` | Immediate feedback |
| Jupyter notebooks | Filter | `filter()`, `sample()` | Interactive feel |
| Subsetting for export | Filter + Materialize | `.materialize(uri)` | Explicit write |
| Apply transforms | Lazy | `.lazy().map().materialize()` | Transform during materialization |
| ML training data | Lazy | `.lazy().iter_batches()` | Batched iteration, memory control |

For most use cases, direct filtering is sufficient. Use `lazy()` only when you need to apply transforms via `map()`.

## Context Configuration

RadiObject uses a global configuration pattern to manage TileDB settings. Configuration is organized into **write-time** settings (immutable after array creation) and **read-time** settings (affect all reads):

```python
from radiobject import ctx, configure, WriteConfig, ReadConfig
from radiobject.ctx import TileConfig, SliceOrientation, CompressionConfig

# Use defaults
array = tiledb.open(uri, ctx=ctx())

# Configure write-time settings (affect new arrays only)
configure(write=WriteConfig(
    tile=TileConfig(orientation=SliceOrientation.ISOTROPIC),
    compression=CompressionConfig(level=5)
))

# Configure read-time settings (affect all reads)
configure(read=ReadConfig(memory_budget_mb=2048, concurrency=8))

# Legacy flat API still supported (maps to write.*)
configure(tile=TileConfig(orientation=SliceOrientation.AXIAL))
```

## Default Settings

| Setting | Path | Default | Description |
|---------|------|---------|-------------|
| Tile orientation | `write.tile.orientation` | `AXIAL` | X-Y slices optimized for neuroimaging |
| Compression | `write.compression` | `ZSTD` level 3 | Balanced speed/ratio |
| Canonical orientation | `write.orientation.canonical_target` | `RAS` | Target coordinate system |
| Reorient on load | `write.orientation.reorient_on_load` | `False` | Preserves original orientation |
| Memory budget | `read.memory_budget_mb` | 1024 MB | TileDB operation limit |
| I/O concurrency | `read.concurrency` | 4 threads | Parallel read/write |

Tile size is auto-computed from array shape based on orientation. For example, `AXIAL` uses full X-Y slices with Z=1, `ISOTROPIC` uses 64³ chunks (in all directions).
