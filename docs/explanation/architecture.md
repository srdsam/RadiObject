# Architecture

Inspired by the [SOMA specification](https://github.com/single-cell-data/SOMA/blob/main/abstract_specification.md), RadiObject is a hierarchical composition of entities aligned on shared indexes. The hierarchy maps to [TileDB](https://docs.tiledb.com/main/) primitives (Groups and Arrays).

## Organisation

```
RadiObject (TileDB Group)
│
├── obs_meta (Sparse Array)
│   dim: obs_subject_id
│   attrs: obs_ids (system), age, sex, diagnosis, ...
│
└── collections/
    ├── T1w (VolumeCollection Group)
    │   ├── obs (Sparse Array)
    │   │   dims: obs_subject_id (FK), obs_id (unique)
    │   │   attrs: series_type, voxel_spacing, dimensions, ...
    │   └── volumes/
    │       ├── 0 (Dense Array: x, y, z [, t] → voxels)
    │       ├── 1 ...
    │       └── N ...
    ├── FLAIR (VolumeCollection) ...
    └── seg (VolumeCollection) ...
```

**Relationships**: obs_meta → VolumeCollection.obs is 1:N via `obs_subject_id` (one subject, many volumes). Each obs row maps 1:1 to a Volume via `obs_id`.

## Mapping to Radiology Standards

RadiObject's data model maps directly to established radiology data standards:

| RadiObject | DICOM | BIDS |
|---|---|---|
| `obs_subject_id` | PatientID | `sub-XX` |
| `VolumeCollection` | Series Description / Modality | Suffix (T1w, FLAIR, seg) |
| `obs_id` | SeriesInstanceUID (unique) | Full filename stem |
| `obs_meta` | Patient-level demographics | `participants.tsv` |
| `obs` | Series-level metadata | Sidecar JSON |

**obs_id uniqueness**: Like DICOM's SeriesInstanceUID, `obs_id` is globally unique across the entire RadiObject — not just within a single collection. The formula is `{obs_subject_id}_{collection_name}` (e.g., `sub-01_T1w`, `sub-01_seg`). This enables unambiguous single-key lookup across all collections while `obs_subject_id` handles the subject-level grouping (analogous to PatientID linking multiple series).

**VolumeCollections as layers**: Each collection represents a distinct imaging "layer" for the same set of subjects — analogous to how a DICOM study contains multiple series (CT, segmentation, MR) for one patient, or how BIDS organizes different suffixes (T1w, FLAIR, bold) under the same subject.

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

## Shapes

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

## Consistency Model

RadiObject writes span multiple TileDB groups and arrays (volumes, obs DataFrame, group metadata). TileDB guarantees **per-array atomicity** but does **not** provide cross-array transactions. This has practical implications:

| Operation | Consistency | Notes |
|-----------|-------------|-------|
| `Volume.from_numpy()` | Atomic | Single dense array write |
| `VolumeCollectionWriter` | Best-effort | Volumes written individually; obs flushed on `__exit__`. A crash mid-write leaves partial volumes without obs rows. |
| `RadiObject.append()` | Best-effort | Modifies collections, obs_meta, and group metadata in sequence. Not transactional across these steps. |
| `RadiObject.from_images()` | Best-effort | Creates collections sequentially; group metadata updated last. |

**Implications for concurrent access:**

- **Single-writer**: RadiObject is designed for single-writer, multiple-reader workflows. Concurrent writers to the same URI are not supported and may corrupt group metadata.
- **Crash recovery**: Use `validate()` after unexpected interruptions to detect inconsistencies (orphan volumes, missing obs rows, metadata count mismatches).
- **Idempotent re-creation**: If a write fails, delete the URI and re-create rather than attempting partial recovery.

**Why not transactional?** TileDB Groups are metadata containers, not transactional databases. Cross-group atomicity would require a write-ahead log or two-phase commit, adding complexity disproportionate to the current use case (batch ETL, not OLTP).

## Cache

RadiObject delegates data caching to TileDB Embedded. There is no application-level data cache — Python `@cached_property` is used only for immutable schema and metadata objects.

### TileDB's Caching Layers

TileDB Embedded maintains two in-memory caches, both scoped to a [`tiledb.Ctx`](https://docs.tiledb.com/main/how-to/configuration):

| Layer | Config Parameter | Default | Scope | Purpose |
|-------|-----------------|---------|-------|---------|
| **Tile cache** | [`sm.tile_cache_size`](https://docs.tiledb.com/main/how-to/configuration#storage-manager) | 512 MB (RadiObject) | Per-context, across queries | LRU cache of uncompressed tiles — avoids re-reading and re-decompressing on repeated access |
| **VFS read-ahead** | [`vfs.read_ahead_cache_size`](https://docs.tiledb.com/main/how-to/configuration#virtual-filesystem) | 10 MB | Per-context | Cloud-only (S3/GCS/Azure). Speculatively fetches extra bytes on small reads to reduce round-trips |

### Tile Cache vs Memory Budget

These are distinct mechanisms:

- **`sm.tile_cache_size`** (default 512 MB): LRU cache that persists uncompressed tiles across queries within the same context. Repeated reads of the same tile region hit this cache instead of disk/S3.
- **`sm.memory_budget`** (default 1 GB): Per-query throttle that caps how much data TileDB will fetch in a single read. Prevents OOM on large subarray reads. Does not cache anything for reuse.

### Context Sharing and Cache Lifetime

The tile cache lives inside `tiledb.Ctx`. This is why context sharing matters:

- **Threads** (`ctx_for_threads`): Workers share one context, so they share one tile cache. A tile read by thread A is available to thread B.
- **Processes** (`ctx_for_process`): Each forked process gets an isolated context with its own empty cache. No cross-process cache sharing.

See [Concurrency Model](#concurrency-model) for context injection details.
