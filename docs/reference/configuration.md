# Configuration Reference

## Quick Links by Task

| I want to configure... | See |
|------------------------|-----|
| Tile chunking (AXIAL vs ISOTROPIC) | [TileConfig](#tileconfig) |
| Memory and thread concurrency | [ReadConfig](#readconfig) |
| S3 cloud storage | [S3Config](#s3config) |
| Anatomical orientation | [OrientationConfig](#orientationconfig) |
| Compression algorithm | [CompressionConfig](#compressionconfig) |

For practical tuning recipes, see [Tuning Concurrency](../how-to/tuning-concurrency.md).

---

RadiObject uses a global configuration pattern to manage TileDB settings. Configuration is organized into **write-time** settings (immutable after array creation) and **read-time** settings (affect all reads).

## Configuration Classes

### RadiObjectConfig

Top-level configuration model with nested settings:

```
RadiObjectConfig
├── write: WriteConfig
│   ├── tile: TileConfig (orientation, extents)
│   ├── compression: CompressionConfig (algorithm, level)
│   └── orientation: OrientationConfig (canonical_target, reorient_on_load)
├── read: ReadConfig
│   ├── memory_budget_mb (default: 1024)
│   ├── concurrency (default: 4) → TileDB thread pools
│   └── max_workers (default: 4) → Python ThreadPoolExecutor
└── s3: S3Config
    ├── region, endpoint
    ├── max_parallel_ops (default: 8)
    └── multipart_part_size_mb (default: 50)
```

**Location:** `src/radiobject/ctx.py`

### WriteConfig

Settings applied when creating new TileDB arrays. These are immutable after array creation.

| Setting | Type | Description |
|---------|------|-------------|
| `tile` | `TileConfig` | Tile chunking configuration |
| `compression` | `CompressionConfig` | Compression settings |
| `orientation` | `OrientationConfig` | Anatomical orientation handling |

### ReadConfig

Settings for reading TileDB arrays. Affects all reads.

| Setting | Default | Description |
|---------|---------|-------------|
| `memory_budget_mb` | 1024 | TileDB operation memory limit |
| `concurrency` | 4 | TileDB thread pool size (`sm.compute/io_concurrency_level`) |
| `max_workers` | 4 | Python `ThreadPoolExecutor` workers |

### S3Config

Cloud storage settings for S3 backends.

| Setting | Default | Description |
|---------|---------|-------------|
| `region` | `us-east-2` | AWS region |
| `endpoint` | `None` | Custom endpoint URL (for MinIO, etc.) |
| `max_parallel_ops` | 8 | Max concurrent S3 operations (`vfs.s3.max_parallel_ops`) |
| `multipart_part_size_mb` | 50 | Multipart upload chunk size |

### TileConfig

Controls how data is chunked on disk.

| Setting | Default | Description |
|---------|---------|-------------|
| `orientation` | `SliceOrientation.AXIAL` | Tiling strategy |

**SliceOrientation options:**

| Value | Tile Extents | Best For |
|-------|--------------|----------|
| `AXIAL` | `[X, Y, 1]` | 2D slice viewers, axial slice extraction |
| `SAGITTAL` | `[1, Y, Z]` | Sagittal slice viewers |
| `CORONAL` | `[X, 1, Z]` | Coronal slice viewers |
| `ISOTROPIC` | `[64, 64, 64]` | 3D patches, ML training |

### CompressionConfig

| Setting | Default | Description |
|---------|---------|-------------|
| `algorithm` | `ZSTD` | Compression algorithm |
| `level` | 3 | Compression level (1-9) |

### OrientationConfig

Controls anatomical orientation handling during ingestion.

| Setting | Default | Description |
|---------|---------|-------------|
| `canonical_target` | `"RAS"` | Target coordinate system |
| `reorient_on_load` | `False` | Reorient volumes during ingestion |

## configure() API

Use `configure()` to update global settings:

```python
from radiobject import ctx, configure
from radiobject.ctx import WriteConfig, ReadConfig, TileConfig, SliceOrientation, CompressionConfig

# Use defaults
array = tiledb.open(uri, ctx=ctx())

# Configure write-time settings (affect new arrays only)
configure(write=WriteConfig(
    tile=TileConfig(orientation=SliceOrientation.ISOTROPIC),
    compression=CompressionConfig(level=5)
))

# Configure read-time settings (affect all reads)
configure(read=ReadConfig(memory_budget_mb=2048, concurrency=8))
```

## ctx() Function

The global `ctx()` function returns a TileDB context built from `configure()` settings:

```python
from radiobject import ctx

# Get configured TileDB context
tdb_ctx = ctx()

# Use in Volume/VolumeCollection operations
vol = Volume(uri, ctx=tdb_ctx)
```

See [Threading Model](../explanation/threading-model.md) for how contexts are managed across threads and processes.

## Default Settings Summary

| Setting | Path | Default | Description |
|---------|------|---------|-------------|
| Tile orientation | `write.tile.orientation` | `AXIAL` | X-Y slices optimized for neuroimaging |
| Compression | `write.compression` | `ZSTD` level 3 | Balanced speed/ratio |
| Canonical orientation | `write.orientation.canonical_target` | `RAS` | Target coordinate system |
| Reorient on load | `write.orientation.reorient_on_load` | `False` | Preserves original orientation |
| Memory budget | `read.memory_budget_mb` | 1024 MB | TileDB operation limit |
| I/O concurrency | `read.concurrency` | 4 threads | Parallel read/write |
| Max workers | `read.max_workers` | 4 | Python thread pool |
| S3 parallel ops | `s3.max_parallel_ops` | 8 | Concurrent S3 operations |

Tile size is auto-computed from array shape based on orientation. For example, `AXIAL` uses full X-Y slices with Z=1, `ISOTROPIC` uses 64³ chunks.

## Related Documentation

- [Tuning Concurrency](../how-to/tuning-concurrency.md) - Practical tuning recipes
- [Threading Model](../explanation/threading-model.md) - Context management architecture
- [Performance Analysis](../explanation/performance-analysis.md) - Benchmark data
