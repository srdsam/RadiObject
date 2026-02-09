# Architecture

Inspired by the [SOMA specification](https://github.com/single-cell-data/SOMA/blob/main/abstract_specification.md), RadiObject is a hierarchical composition of entities aligned on shared indexes. The hierarchy maps to [TileDB](https://docs.tiledb.com/main/) primitives (Groups and Arrays).

## TileDB Structure

```
RadiObject (TileDB Group)
├── metadata: { subject_count, n_collections }
│
├── obs_meta (Sparse Array) ────────────────────────────────────────────────┐
│   │                                                                        │
│   │  DIMENSIONS (Indexes):                                                 │
│   │    dim[0]: obs_subject_id  (ascii)  <- Primary subject identifier      │
│   │                                                                        │
│   │  ATTRIBUTES (Data):                                                    │
│   │    obs_ids            (U2048)  <- JSON list of volume obs_ids (system) │
│   │    User-defined columns (age, sex, diagnosis, labels, etc.)            │
│   └────────────────────────────────────────────────────────────────────────┘
│
└── collections (TileDB Group)
    │
    ├── T1w (VolumeCollection - TileDB Group)
    │   ├── metadata: { n_volumes, name, [x_dim, y_dim, z_dim]? }
    │   │              └── shape fields only present if collection is uniform
    │   │
    │   ├── obs (Sparse Array) ─────────────────────────────────────────────┐
    │   │   │                                                                │
    │   │   │  DIMENSIONS (Indexes):                                         │
    │   │   │    dim[0]: obs_subject_id  (ascii)  <- FK to RadiObject.obs_meta
    │   │   │    dim[1]: obs_id          (ascii)  <- Unique volume ID        │
    │   │   │                                                                │
    │   │   │  ATTRIBUTES (Data):                                            │
    │   │   │    series_type, voxel_spacing, dimensions, axcodes, ...         │
    │   │   └────────────────────────────────────────────────────────────────┘
    │   │
    │   └── volumes (TileDB Group)
    │       ├── 0 (Volume - Dense Array) ───────────────────────────────────┐
    │       │   │  DIMENSIONS: x, y, z [, t]                                │
    │       │   │  ATTRIBUTES: voxels (float32/int16)                       │
    │       │   │  METADATA:   obs_id, slice_orientation, orientation info   │
    │       │   └────────────────────────────────────────────────────────────┘
    │       ├── 1 (Volume) ...  <- may have different shape than Volume[0]
    │       └── N (Volume) ...
    │
    ├── FLAIR (VolumeCollection) ...
    └── seg (VolumeCollection) ...
```

## Key Relationships

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           INDEX RELATIONSHIPS                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  RadiObject.obs_meta                     VolumeCollection.obs                │
│  ┌────────────────────┐                  ┌─────────────────────┐             │
│  │ obs_subject_id (D) │<────────────────>│ obs_subject_id (D)  │  FK         │
│  │ obs_ids        (A) │                  │ obs_id          (D) │             │
│  │ age            (A) │                  │ series_type     (A) │             │
│  │ sex            (A) │                  │ voxel_spacing   (A) │             │
│  │ tumor_grade    (A) │                  │ dimensions      (A) │             │
│  └────────────────────┘                  └─────────────────────┘             │
│         │                                          │                         │
│         │ 1:N relationship                         │ 1:1 relationship        │
│         │ (one subject, many volumes)              │                         │
│         v                                          v                         │
│  ┌──────────────────────────────────────────────────────────────┐           │
│  │                        Volume (Dense Array)                   │           │
│  │  DIMENSIONS: x, y, z [, t]                                   │           │
│  │  ATTRIBUTES: voxels                                          │           │
│  │  METADATA:   obs_id (links to obs dataframe)                 │           │
│  └──────────────────────────────────────────────────────────────┘           │
│                                                                              │
│  (D) = Dimension    (A) = Attribute                                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Component Summary

| Component | TileDB Type | Dimensions (Indexes) | Attributes (Data) |
|-----------|-------------|----------------------|-------------------|
| **RadiObject** | Group | — | metadata: subject_count, n_collections |
| **obs_meta** | Sparse Array | `obs_subject_id` | obs_ids (system), user-defined (age, labels, etc.) |
| **VolumeCollection** | Group | — | metadata: n_volumes, name, [shape]? |
| **obs** | Sparse Array | `obs_subject_id`, `obs_id` | series_type, voxel_spacing, dimensions, etc. |
| **Volume** | Dense Array | `x`, `y`, `z` [, `t`] | `voxels` (intensity values) |

## Index Design

`Index` is an immutable, named dataclass providing bidirectional mapping between string IDs and integer positions:

- **RadiObject.index**: `Index(name="obs_subject_id")` — subject-level ordering
- **VolumeCollection.index**: `Index(name="obs_id")` — volume-level ordering
- **VolumeCollection.subjects**: `Index(name="obs_subject_id")` — deduplicated subject IDs

Index supports set algebra (`&`, `|`, `-`, `^`) with order preservation from the left operand, positional selection (`take`, `mask`), alignment checking (`is_aligned`), and subset/superset comparison (`<=`, `>=`):

```python
radi.T1w.subjects.is_aligned(radi.seg.subjects)  # True
common = radi.T1w.subjects & radi.seg.subjects
train.index | val.index  # all subjects
train.index & val.index  # empty = no overlap
```

The standalone `align(*indexes)` function computes the intersection of multiple indexes, preserving order from the first.

**Shapes:**

- **Uniform collections**: `x_dim, y_dim, z_dim` stored in group metadata; `is_uniform=True`
- **Heterogeneous collections**: No shape in group metadata; each volume's shape stored in `obs.dimensions`
- **4D volumes**: Temporal dimension (`t`) is per-volume; not tracked at collection level

## Organisation

Radiology dimensions are irregular across datasets (different scanners, protocols, preprocessing). `VolumeCollection` groups volumes with consistent spatial (X/Y/Z) dimensions — 4D volumes with different time dimensions but the same spatial grid share a collection. `RadiObject` organizes heterogeneous collections (e.g., T1w at 1mm^3, fMRI at 3mm^3) under a unified structure.

## Composition

The TileDB entities are a public property of each entity. This allows direct access to the TileDB object for power users, while presenting a simple API surface. Individual entities are **stateless** — file handles are not cached in memory (preventing file handle exhaustion).

## Anatomical Orientation

Medical images encode spatial orientation via an **affine matrix** mapping voxel indices to physical (world) coordinates. RadiObject preserves this information and optionally standardizes orientation during ingestion.

Orientation is described by three-letter codes (RAS, LPS, LAS) indicating which anatomical direction each axis points. See [Lexicon: Coordinate Systems](../reference/lexicon.md#coordinate-systems-and-orientation) for terminology.

**Tile orientation vs anatomical orientation** — distinct concepts:

- **Anatomical orientation** (`orientation_info`): Physical coordinate system (RAS/LPS)
- **Tile orientation** (`tile_orientation`): Storage chunking strategy for I/O performance

Choose tile orientation based on access patterns (see [Benchmarks](../reference/benchmarks.md)), not anatomical convention.

For reorientation configuration, see [Ingest Data: Handling Orientation](../how-to/ingest-data.md#handling-orientation). For tile options, see [Configuration: TileConfig](../reference/configuration.md#tileconfig).

## Concurrency Model

RadiObject operates across **four concurrency layers**:

```
┌─────────────────────────────────────────────────────────────────┐
│ Layer 4: PyTorch DataLoader Workers (PROCESSES via fork)       │
│          num_workers=4, persistent_workers=True                 │
├─────────────────────────────────────────────────────────────────┤
│ Layer 3: Python ThreadPoolExecutor (THREADS)                   │
│          max_workers from ReadConfig (default: 4)              │
├─────────────────────────────────────────────────────────────────┤
│ Layer 2: TileDB Internal Threads                               │
│          sm.compute_concurrency_level, sm.io_concurrency_level │
├─────────────────────────────────────────────────────────────────┤
│ Layer 1: S3/VFS Level                                          │
│          vfs.s3.max_parallel_ops (default: 8)                  │
└─────────────────────────────────────────────────────────────────┘
```

### Global State Management

`get_tiledb_ctx()` lazily initializes a global TileDB context from `configure()` settings. All data objects accept an optional `ctx` parameter — if `None`, they fall back to the global context.

### Context Injection

```python
class Volume:
    def __init__(self, uri: str, ctx: tiledb.Ctx | None = None):
        self._ctx = ctx  # None = use global

    def _effective_ctx(self) -> tiledb.Ctx:
        return self._ctx if self._ctx else get_tiledb_ctx()
```

### Threads vs Processes

From [TileDB Wiki](https://github.com/TileDB-Inc/TileDB/wiki/Threading-Model): libtiledb is thread-safe, and sharing one `Ctx` across a thread pool is optimal because schema and fragment metadata is cached per-Ctx.

RadiObject provides two semantically distinct context functions:

```python
def ctx_for_threads(ctx=None) -> tiledb.Ctx:
    """Return context for thread pool workers. Shares caching."""
    return ctx if ctx else get_tiledb_ctx()

def ctx_for_process(base_ctx=None) -> tiledb.Ctx:
    """Create new context for forked process. Isolated memory."""
    if base_ctx is not None:
        return tiledb.Ctx(base_ctx.config())
    return get_radiobject_config().to_tiledb_ctx()
```

| Scenario | Function | Behavior |
|----------|----------|----------|
| `ThreadPoolExecutor` | `ctx_for_threads()` | Returns same context (shared caching) |
| `multiprocessing.Pool` | `ctx_for_process()` | Creates new isolated context |
| DataLoader (`num_workers>0`) | `ctx_for_process()` | Creates new isolated context |

For practical tuning recipes, see [ML Training: Performance Tuning](../how-to/ml-training.md#performance-tuning).
