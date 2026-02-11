# Configuration Reference

## Quick Links by Task

| I want to configure... | See |
|------------------------|-----|
| Tile chunking (AXIAL vs ISOTROPIC) | [TileConfig](#tileconfig) |
| Memory and thread concurrency | [ReadConfig](#readconfig) |
| S3 cloud storage | [S3Config](#s3config) |
| Anatomical orientation | [OrientationConfig](#orientationconfig) |
| Compression algorithm | [CompressionConfig](#compressionconfig) |

For practical tuning recipes, see [ML Training: Performance Tuning](../how-to/ml-training.md#performance-tuning).

---

RadiObject uses a global configuration pattern to manage [TileDB](https://docs.tiledb.com/main/how-to/configuration) settings. Configuration is organized into **write-time** settings (immutable after array creation) and **read-time** settings (affect all reads).

## Configuration Tree

```
RadiObjectConfig
├── write: WriteConfig
│   ├── tile: TileConfig (orientation, extents)
│   ├── compression: CompressionConfig (algorithm, level)
│   └── orientation: OrientationConfig (canonical_target, reorient_on_load)
├── read: ReadConfig
│   ├── memory_budget_mb (default: 1024)
│   ├── tile_cache_size_mb (default: 512) → TileDB LRU tile cache
│   ├── concurrency (default: 4) → TileDB thread pools
│   └── max_workers (default: 4) → Python ThreadPoolExecutor
└── s3: S3Config
    ├── region, endpoint
    ├── max_parallel_ops (default: 8)
    └── multipart_part_size_mb (default: 50)
```

## WriteConfig

Settings applied when creating new TileDB arrays. Immutable after array creation.

| Setting | Type | Description |
|---------|------|-------------|
| `tile` | `TileConfig` | Tile chunking configuration |
| `compression` | `CompressionConfig` | Compression settings |
| `orientation` | `OrientationConfig` | Anatomical orientation handling |

## ReadConfig

Settings for reading TileDB arrays. Affects all reads.

| Setting | Default | Description |
|---------|---------|-------------|
| `memory_budget_mb` | 1024 | Per-query memory limit (`sm.memory_budget`) |
| `tile_cache_size_mb` | 512 | In-memory LRU tile cache (`sm.tile_cache_size`) |
| `concurrency` | 4 | TileDB thread pool size (`sm.compute/io_concurrency_level`) |
| `max_workers` | 4 | Python `ThreadPoolExecutor` workers |

## S3Config

Cloud storage settings. For credential setup, see [Cloud Setup](../how-to/cloud-setup.md).

| Setting | Default | Description |
|---------|---------|-------------|
| `region` | `us-east-1` | AWS region |
| `endpoint` | `None` | Custom endpoint URL (for MinIO, etc.) |
| `max_parallel_ops` | 8 | Max concurrent S3 operations (`vfs.s3.max_parallel_ops`) |
| `multipart_part_size_mb` | 50 | Multipart upload chunk size |

## TileConfig

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

For 4D volumes, tile extents automatically include a temporal extent of `1` (e.g., `[X, Y, 1, 1]` for AXIAL).

## CompressionConfig

| Setting | Default | Description |
|---------|---------|-------------|
| `algorithm` | `ZSTD` | Compression algorithm |
| `level` | 3 | Compression level (1-9) |

## OrientationConfig

Controls anatomical orientation handling during ingestion.

| Setting | Default | Description |
|---------|---------|-------------|
| `canonical_target` | `"RAS"` | Target coordinate system |
| `reorient_on_load` | `False` | Reorient volumes during ingestion |
| `auto_detect` | `True` | Automatically detect orientation from file headers |
| `store_original_affine` | `True` | Store original affine in metadata when reorienting |

## configure() API

```python
from radiobject import configure, WriteConfig, ReadConfig, TileConfig, SliceOrientation, CompressionConfig

# Write-time settings (affect new arrays only)
configure(write=WriteConfig(
    tile=TileConfig(orientation=SliceOrientation.ISOTROPIC),
    compression=CompressionConfig(level=5)
))

# Read-time settings (affect all reads)
configure(read=ReadConfig(memory_budget_mb=2048, concurrency=8))
```

## TileDB Context Management

`get_tiledb_ctx()` returns a TileDB context built from `configure()` settings:

```python
from radiobject import get_tiledb_ctx

ctx = get_tiledb_ctx()
vol = Volume(uri, ctx=ctx)
```

TileDB contexts are thread-safe. Sharing a context across threads enables metadata caching. For forked processes (DataLoader workers), use `ctx_for_process()` to create isolated contexts. See [Architecture: Concurrency Model](../explanation/architecture.md#concurrency-model) for details.

## Defaults Summary

| Setting | Path | Default |
|---------|------|---------|
| Tile orientation | `write.tile.orientation` | `AXIAL` |
| Compression | `write.compression` | `ZSTD` level 3 |
| Canonical orientation | `write.orientation.canonical_target` | `RAS` |
| Reorient on load | `write.orientation.reorient_on_load` | `False` |
| Memory budget | `read.memory_budget_mb` | 1024 MB |
| Tile cache | `read.tile_cache_size_mb` | 512 MB |
| I/O concurrency | `read.concurrency` | 4 threads |
| Max workers | `read.max_workers` | 4 |
| S3 parallel ops | `s3.max_parallel_ops` | 8 |

## Recipes

### Radiology Viewer (2D Slice Access)

```python
configure(write=WriteConfig(
    tile=TileConfig(orientation=SliceOrientation.AXIAL),
    compression=CompressionConfig(algorithm=Compressor.LZ4, level=1),
))
```

### ML Training (3D Patch Extraction)

```python
configure(write=WriteConfig(
    tile=TileConfig(orientation=SliceOrientation.ISOTROPIC),
    compression=CompressionConfig(algorithm=Compressor.ZSTD, level=3),
))
```

### S3 Cloud Storage

```python
configure(
    read=ReadConfig(max_workers=8, concurrency=2),
    s3=S3Config(region="us-east-1", max_parallel_ops=32),
)
```
