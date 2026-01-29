# RadiObject

**What?** A TileDB-backed data structure to represent radiology data.

**Why?** The two standard formats for radiology data (NIfTI and DICOM) are challenging to use for analysis at scale. They must be read from local disk and don't fully support partial reads. 

**Why TileDB?** TileDB supports cloud-native object stores, partial reads, and a hierarchical format (for organising multiple volumes into an atlas). *The limitation being, transform/filter operations are less smooth UX given that the in-memory TileDB data is tied directly to the on-disk TileDB data.*

## Design

Inspired by the [SOMA specification](https://github.com/single-cell-data/SOMA/blob/main/abstract_specification.md), RadiObject is a hierarchical composition of entities aligned on shared indexes.

```
RadiObject
├── ObsMeta                           # Observational metadata per patient
│   └── (indexed per patient)         # Patient height, weight....
│
├── VolumeCollection                  # Collection of volumes (e.g., all T1w scans)
│   │                                 # Enforces consistent X/Y/Z dimensions
│   │
│   ├── Obs                           # Observational metadata per volume/scan
│   │   └── (indexed per volume)      # Acquisition params, labels...
│   │
│   └── Volume[]                      # Individual 3D volumes
│       ├── Volume[0]
│       │   └── Tile[]                # Segmented chunks - handled by TileDB
│       │       ├── Tile[0,0,0]
│       │       ├── Tile[0,0,1]
│       │       └── ...
│       ├── Volume[1]
│       │   └── Tile[]
│       └── ...
```

| Component | Description |
|-----------|-------------|
| **Tile** | Segmented chunk of a volume for partial reads |
| **Volume** | A single 3D acquisition (maps to DICOM Series) |
| **VolumeCollection** | Collection of volumes of a given type (e.g., all T1w) with consistent X/Y/Z dimensions |
| **Obs** | Observational metadata aligned on the index of each volume |
| **ObsMeta** | Observational metadata aligned on the index of each subject |

| Index | Description |
|-----------|-------------|
| **obs_id** | Index on each individual 'series' - present in `Obs` |
| **obs_subject_id** | Index on each 'subject' - present in `Obs` and `ObsMeta` |

## Organisation

Radiology datasets follow the DICOM hierarchy: patients undergo studies containing multiple series (acquisitions), each composed of instances (slices/frames). Volume dimensionality varies by acquisition type—structural scans are 3D while functional and diffusion data are 4D.

Critically, **dimensions are irregular across a dataset**:
- Different scanners produce different matrix sizes
- Protocols vary by site and evolve over time  
- Preprocessing (resampling, registration) changes dimensions

This irregularity makes batch analysis and ML challenging, as most frameworks expect uniform tensor shapes.

`VolumeCollection` addresses this by grouping volumes with consistent X/Y/Z dimensions, while `RadiObject` organizes heterogeneous collections (e.g., T1w at 1mm³, fMRI at 3mm³) under a unified structure.

### Composition

The TileDB entities are a public property of each given entity. This allows direct access to the TileDB object for power users, while persisting a simple API surface. The individual entites are **stateless** - meaning file handles are not cached in memory (to prevent file handle exhaustion).

### Context

RadiObject uses a global configuration pattern to manage TileDB settings. The context is lazily built from configuration and automatically invalidated when settings change. It is important to optimize the context for a given usecase.

```python
from radiobject import ctx, configure
from radiobject.ctx import TileConfig, SliceOrientation

# Use defaults
array = tiledb.open(uri, ctx=ctx())

# Customize settings
configure(tile=TileConfig(orientation=SliceOrientation.ISOTROPIC))
```

**Defaults:**

| Setting | Default | Description |
|---------|---------|-------------|
| Tile orientation | `AXIAL` | X-Y slices optimized for neuroimaging |
| Compression | `ZSTD` level 3 | Balanced speed/ratio |
| Memory budget | 1024 MB | TileDB operation limit |
| I/O concurrency | 4 threads | Parallel read/write |
| Canonical orientation | `RAS` | Target coordinate system |
| Reorient on load | `False` | Preserves original orientation |

Tile size is auto-computed from array shape based on orientation. For example, `AXIAL` uses full X-Y slices with Z=1, `ISOTROPIC` uses 64³ chunks (in all directions).

## Data Access

RadiObject provides two complementary patterns for filtering and accessing data, each designed for different workflows.

### Exploration Mode (Pandas-like)

Direct indexing with `iloc`, `loc`, `head()`, `tail()`, and `sample()` returns lightweight views for interactive exploration:

```python
radi = RadiObject("s3://bucket/study")

# Direct indexing - immediate results
subset = radi.iloc[0:10]        # First 10 subjects
subset = radi.loc["sub-01"]     # Single subject by ID
subset = radi.head(5)           # Quick preview

# Access collections within view
vol = subset.T1w.iloc[0]        # Get first T1w volume
data = vol.to_numpy()           # Load into memory
```

Views (`RadiObjectView`) provide immediate access to data and feel like working with pandas DataFrames. Use this mode when interactively exploring data, debugging, or building analysis notebooks.

### Pipeline Mode (Lazy Query Builder)

The `query()` method returns a lazy filter builder for ETL pipelines and ML training:

```python
# Lazy filtering - no data loaded until materialized
result = (
    radi.query()
    .filter("age > 40 and diagnosis == 'tumor'")
    .select_collections(["T1w", "FLAIR"])
    .sample(100, seed=42)
    .to_radi_object("s3://bucket/subset", streaming=True)
)

# Streaming iteration for ML training
for batch in radi.query().filter("split == 'train'").iter_batches(batch_size=32):
    train_step(batch.volumes["T1w"], batch.volumes["FLAIR"])
```

Queries (`Query`) accumulate filters without touching data. Explicit materialization methods (`to_radi_object()`, `iter_volumes()`, `count()`) trigger execution. Use this mode for reproducible pipelines, memory-efficient exports, and ML data loading.

### When to Use Each

| Use Case | Mode | Why |
|----------|------|-----|
| Debugging / quick inspection | Exploration | Immediate feedback |
| Jupyter notebooks | Exploration | Interactive feel |
| ETL pipelines | Pipeline | Explicit execution, streaming |
| ML training data | Pipeline | Batched iteration, memory control |
| Subsetting for export | Pipeline | Memory-efficient streaming writes |

Both modes can be combined: start with exploration to understand your data, then use `view.to_query()` to transition to pipeline mode for production.

## Validation

RadiObject enforces data integrity through two validation strategies:

### Eager Validation (Automatic)

Index uniqueness is validated automatically when building indexes. If duplicate `obs_id` or `obs_subject_id` values are detected, a `ValueError` is raised immediately.

```python
# This will raise ValueError if duplicates exist
radi = RadiObject(uri)  # Index built on first access
```

### Lazy Validation (On-Demand)

Expensive consistency checks are available via the `validate()` method. Call this after data migrations, before long computations, or when debugging data issues.

```python
# Validate a VolumeCollection
collection = radi.T1w
collection.validate()  # Checks referential integrity, position coherence, metadata counts

# Validate entire RadiObject (cascades to all collections)
radi.validate()
```

**Checks performed by `validate()`:**

| Check | Level | Description |
|-------|-------|-------------|
| Referential integrity | VolumeCollection | All volumes have corresponding obs rows and vice versa |
| Position coherence | VolumeCollection | Volume position matches obs row order |
| Metadata counts | Both | Stored counts match actual data |
| Subject count | RadiObject | `subject_count` matches `obs_meta` row count |
| Collection count | RadiObject | `n_collections` matches actual collection count |

## DevEx

Package management is done with `uv`. Reference the [pyproject](pyproject.toml) to see available package and dependency groups.
