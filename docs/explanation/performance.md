# Performance Analysis

For measured data, see [Benchmarks](../reference/benchmarks.md). This page explains *why* operations have their observed characteristics and provides scaling guidance.

---

## Why Full Volume is Slower Than Raw NIfTI

RadiObject full volume load (705ms axial, 99ms isotropic) is slower than nibabel (33ms uncompressed) because TileDB adds tile assembly and metadata overhead. This is the trade-off for enabling random access.

```
NIfTI:      [full blob] → decompress all → extract slice
RadiObject: [tile]      → read 1 tile    → done
```

- Full-volume sequential processing has similar performance across formats
- RadiObject's advantage is **partial reads** (200-560x faster for slices/patches)
- Use RadiObject for I/O, then apply MONAI/TorchIO transforms to loaded data

---

## Why Axial Tiling Gives 200-600x Speedup for Slices

With axial tiling (`tile_extent=[X, Y, 1]`), each axial slice is one contiguous tile (~230 KB). NIfTI requires decompressing the entire 35.6 MB volume.

```
NIfTI:      Load 35,600 KB → decompress → extract 230 KB
RadiObject: Load 230 KB tile directly
Ratio:      ~155x theoretical, ~310x observed vs MONAI (includes decompression savings)
```

---

## Why Isotropic is Best for 3D Patches

Isotropic tiling (`tile_extent=[64, 64, 64]`) matches typical patch sizes. A 64^3 patch spans 1-8 isotropic tiles vs 64 axial tiles.

| operation   | axial_tiles_read | isotropic_tiles_read |
|-------------|------------------|----------------------|
| axial_slice | 1                | ~16                  |
| 64^3_patch  | 64               | 1-8                  |
| full_volume | 155              | ~60                  |

This is a tile-count analysis — see [Benchmarks: Tiling Strategy Impact](../reference/benchmarks.md#tiling-strategy-impact) for measured numbers.

---

## Tiling Strategy Guide

![Tiling heatmap](../assets/benchmarks/tiling_heatmap.png)

| use_case              | tiling        | reason                    |
|-----------------------|---------------|---------------------------|
| 2D slice viewer       | **Axial**     | Single tile per slice (3.4ms) |
| 3D patch training     | **Isotropic** | Patches match tile size (1.5ms) |
| Full volume analysis  | **Isotropic** | Fewer total tiles (99ms) |
| Mixed workload        | **Isotropic** | Best general-purpose      |

---

## Why S3 is ~14x Slower for Full Volumes

Each tile request incurs ~50-150ms S3 round-trip latency. For 155 axial tiles, this compounds to ~8.3s.

**Mitigation:** Use patch-based reads (~203-238ms per patch from S3) or parallel workers to amortize latency.

---

## Why Multi-Worker DataLoaders Slow Down Small Datasets

For small datasets, `num_workers=0` beats multi-worker because IPC serialization (~100-200ms per 35MB tensor) and process spawn overhead (~1-2s) dominate.

**Recommendation:** `num_workers=0` for <100 volumes, `num_workers=4-8` for >1000 volumes.

See [Benchmarks: Multi-Worker Scaling](../reference/benchmarks.md#multi-worker-scaling) for measured numbers.

---

## Scaling Guidance

### What Scales Well

1. **Metadata operations: O(1)** — `iloc[i]`, `loc["id"]`, `obs_subject_ids` are constant-time regardless of dataset size.

2. **Selective access: O(accessed data)** — Loading 1 volume from 10,000 subjects costs the same as from 3 subjects.

3. **Parallel writes** — Throughput scales linearly (~140 MB/s per worker on local SSD). See [Benchmarks: Parallel Write Scaling](../reference/benchmarks.md#parallel-write-scaling).

### Recommendations by Scale

| scale            | storage          | approach                                      |
|------------------|------------------|-----------------------------------------------|
| <1,000 subjects  | Local SSD        | Direct access, in-memory processing           |
| 1,000-10,000     | S3 + local cache | Metadata-first filtering, parallel workers    |
| >10,000          | S3               | Streaming, patch-based training               |

---

## TileDB vs Zarr: Chunked Format Comparison

Both TileDB and Zarr v3 support chunk-based partial reads with ZSTD compression. The benchmarks use identical settings (ZSTD clevel=3, same chunk shapes) to isolate format-level differences.

**Key architectural differences:**

- **Storage layer:** TileDB uses its own VFS (Virtual File System) layer with built-in S3 support and tile caching. Zarr uses fsspec/s3fs for remote storage.
- **Metadata model:** TileDB uses a fragment-based metadata model optimized for concurrent writes. Zarr v3 stores chunks as files within a directory hierarchy, with array metadata in `zarr.json`.
- **Caching:** TileDB maintains a built-in LRU tile cache (`sm.tile_cache_size`) within its context, amortizing repeated reads — 85-95% hit rates for sequential access. Zarr v3 has **no built-in chunk cache**: local reads benefit from OS page cache, but S3 reads re-fetch every chunk on every access. For training loops with overlapping patches or repeated epoch access, TileDB's cache gives a structural advantage.
- **Partial reads:** Both formats read only the chunks/tiles intersecting a query region. TileDB's VFS can issue parallel range requests; Zarr issues one request per chunk via fsspec.
- **Raw throughput:** For single isolated reads (no caching benefit), Zarr is slightly faster due to lower per-chunk overhead — e.g. 1.4ms vs 3.4ms for axial slice, 138 vs 128 samples/sec in DataLoader benchmarks.

See [Benchmarks: Framework Speedups](../reference/benchmarks.md#framework-speedups) and [Cache Hit Rates](../reference/benchmarks.md#cache-hit-rates-tiledb) for measured numbers.

---

### Key Pattern: Filter via Metadata First

```python
# Select cohort without volume reads
obs = radi.obs_meta.read()
tumor_ids = obs[obs["diagnosis"] == "tumor"]["obs_subject_id"].tolist()

# Only read the volumes you need
for sid in tumor_ids[:100]:
    data = radi.loc[sid].T1w.iloc[0].to_numpy()
```
