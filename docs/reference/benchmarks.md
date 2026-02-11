# Benchmarks

> RadiObject enables **200-560x faster** partial reads and native S3 access.

![Benchmark overview](../assets/benchmarks/benchmark_hero.png)

**Configuration:** batch_size=4, patch_size=64^3, n_runs=10, 20 MSD subjects, local SSD + S3 (us-east-2)

---

## Full Volume Load

![Full volume load times](../assets/benchmarks/full_volume_load.png)

| framework   | scenario | tiling    | time_ms | cpu_pct | heap_mb |
|-------------|----------|-----------|---------|---------|---------|
| RadiObject  | local    | isotropic | 99      | 62      | 304     |
| nibabel     | local    | -         | 33      | 31      | 608     |
| numpy       | local    | -         | 42      | 38      | 608     |
| nibabel     | local    | gzip      | 418     | 24      | 912     |
| RadiObject  | local    | axial     | 705     | 40      | 304     |
| TorchIO     | local    | -         | 716     | 38      | 304     |
| MONAI       | local    | -         | 1028    | 26      | 608     |
| RadiObject  | s3       | axial     | 26256   | 16      | 304     |
| zarr        | local    | axial     | 49      | 51      | 146     |
| zarr        | local    | isotropic | 22      | 59      | 177     |
| zarr        | s3       | axial     | 1351    | 34      | 142     |

RadiObject isotropic (99ms) is competitive with raw nibabel (33ms) while enabling random access. Zarr full volume loads are fast (22-49ms local) due to lightweight metadata, but lack RadiObject's integrated metadata and caching layer.

See [Performance: Why Full Volume is Slower](../explanation/performance.md#why-full-volume-is-slower-than-raw-nifti) for interpretation.

---

## 2D Slice Extraction

![Slice extraction times](../assets/benchmarks/slice_extraction.png)

| method              | scenario | tiling    | time_ms   |
|---------------------|----------|-----------|-----------|
| RadiObject          | local    | axial     | **3.4**   |
| zarr                | local    | axial     | 1.4       |
| RadiObject          | local    | isotropic | 24        |
| RadiObject          | s3       | axial     | 203       |
| zarr                | s3       | axial     | 93        |
| MONAI               | local    | -         | 1115      |
| TorchIO             | local    | -         | 731       |

Both chunked formats achieve sub-5ms local slice extraction. Zarr's lower overhead gives a slight edge for individual slices, while RadiObject's caching benefits repeated access.

See [Performance: Why Axial Tiling Gives 200-600x Speedup](../explanation/performance.md#why-axial-tiling-gives-200-600x-speedup-for-slices) for interpretation.

---

## 3D ROI Extraction

![ROI extraction times](../assets/benchmarks/roi_extraction.png)

| method              | scenario | tiling    | time_ms   |
|---------------------|----------|-----------|-----------|
| RadiObject          | local    | isotropic | **2.0**   |
| zarr                | local    | isotropic | 2.9       |
| RadiObject          | local    | axial     | 20        |
| RadiObject          | s3       | isotropic | 238       |
| zarr                | s3       | isotropic | 63        |
| MONAI               | local    | -         | 1106      |
| TorchIO             | local    | -         | 739       |

**559x faster** than MONAI, **374x faster** than TorchIO for isotropic 64^3 ROIs. RadiObject and Zarr are comparable for local 3D partial reads.

See [Performance: Why Isotropic is Best for 3D Patches](../explanation/performance.md#why-isotropic-is-best-for-3d-patches) for interpretation.

---

## Framework Speedups

![Speedup vs MONAI](../assets/benchmarks/speedup_vs_monai.png)

![Speedup vs TorchIO](../assets/benchmarks/speedup_vs_torchio.png)

![Speedup vs Zarr](../assets/benchmarks/speedup_vs_zarr.png)

| operation | monai_ms | torchio_ms | zarr_ms | radiobject_ms | vs_monai | vs_torchio | vs_zarr |
|-----------|----------|------------|---------|---------------|----------|------------|---------|
| slice_2d  | 1115     | 731        | 1.4     | 3.4           | 325x     | 213x       | 0.4x    |
| roi_3d    | 1106     | 739        | 2.9     | 2.0           | 559x     | 374x       | 1.5x    |

MONAI and TorchIO must load the full volume for any access pattern. RadiObject and Zarr both read only the chunks/tiles needed. RadiObject adds integrated metadata, caching, and S3 VFS — Zarr is a raw array format.

---

## S3 vs Local

| framework   | operation   | scenario | tiling    | time_ms   |
|-------------|-------------|----------|-----------|-----------|
| RadiObject  | full_volume | local    | axial     | 705       |
| RadiObject  | full_volume | s3       | axial     | 26256     |
| RadiObject  | slice_2d    | local    | axial     | 3.4       |
| RadiObject  | slice_2d    | s3       | axial     | 203       |
| RadiObject  | roi_3d      | local    | isotropic | 2.0       |
| RadiObject  | roi_3d      | s3       | isotropic | 238       |
| zarr        | full_volume | local    | axial     | 49        |
| zarr        | full_volume | s3       | axial     | 1351      |
| zarr        | slice_2d    | local    | axial     | 1.4       |
| zarr        | slice_2d    | s3       | axial     | 93        |
| zarr        | roi_3d      | local    | isotropic | 2.9       |
| zarr        | roi_3d      | s3       | isotropic | 63        |

Partial reads on S3 (203-238ms) are **110-130x faster** than full volume S3 reads (26256ms). Zarr S3 reads via fsspec are faster for individual operations but lack TileDB's VFS-level parallelism for batch access.

See [Performance: Why S3 is ~14x Slower](../explanation/performance.md#why-s3-is-14x-slower-for-full-volumes) for interpretation.

---

## Format and Storage Overhead

![Disk space comparison](../assets/benchmarks/disk_space_comparison.png)

| format          | size_gb | compression | partial_read |
|-----------------|---------|-------------|--------------|
| NIfTI (.nii.gz) | 2.1     | 2.84x       | No           |
| NIfTI (.nii)    | 6.5     | 0.91x       | No           |
| NumPy (.npy)    | 13.0    | 0.46x       | No           |
| TileDB (axial)  | 6.1     | 0.98x       | Yes          |
| TileDB (iso)    | 5.6     | 1.06x       | Yes          |
| Zarr (axial)    | 0.2     | 35.8x       | Yes          |
| Zarr (iso)      | 0.2     | 35.9x       | Yes          |

TileDB uses ~3x more space than gzipped NIfTI, but enables partial reads that are 200-560x faster. Zarr achieves high compression (36x) on sparse medical imaging data (MSD brain tumour) due to low per-chunk overhead with ZSTD.

---

## Tiling Strategy Impact

![Tiling heatmap](../assets/benchmarks/tiling_heatmap.png)

| access_pattern | axial_ms | isotropic_ms | best_choice |
|----------------|----------|--------------|-------------|
| axial_slice    | **3.8**  | 24           | Axial       |
| coronal_slice  | 85       | **15**       | Isotropic   |
| sagittal_slice | 83       | **12**       | Isotropic   |
| roi_32         | 10       | **1.6**      | Isotropic   |
| roi_64         | 20       | **1.5**      | Isotropic   |
| roi_128        | 51       | **4.4**      | Isotropic   |

**Zarr Chunking:**

| access_pattern | zarr_axial_ms | zarr_isotropic_ms |
|----------------|---------------|-------------------|
| axial_slice    | 1.3           | 7.4               |
| coronal_slice  | 29            | 6.2               |
| sagittal_slice | 28            | 6.4               |
| roi_32         | 6.0           | 2.8               |
| roi_64         | 12            | 2.9               |
| roi_128        | 31            | 6.2               |

See [Performance: Tiling Strategy Guide](../explanation/performance.md#tiling-strategy-guide).

---

## Memory Efficiency

| operation        | radiobject_mb | nifti_load_mb |
|------------------|---------------|---------------|
| slice_extraction | **1**         | 300-900       |
| full_volume      | 304           | 300-600       |

Partial reads use minimal memory because only the requested tiles are loaded.

---

## ML Training Throughput

![Dataloader throughput](../assets/benchmarks/dataloader_throughput.png)

| framework   | ms_per_batch | samples_per_sec | notes                   |
|-------------|--------------|-----------------|-------------------------|
| RadiObject  | 31           | 128             | isotropic, local        |
| zarr        | 29           | 138             | isotropic, local        |
| TorchIO     | 3306         | 1.2             | local, full volume load |
| zarr        | 4423         | 0.9             | isotropic, S3           |
| RadiObject  | 31434        | 0.1             | isotropic, S3           |

Zarr and RadiObject achieve comparable local throughput (~128-138 samples/sec). Both are **100x faster** than TorchIO for patch-based training (batch_size=4, patch_size=64^3). On S3, Zarr's fsspec issues async chunk requests more efficiently (0.9 vs 0.1 samples/sec).

### Patch-Based I/O Reduction

| loading_mode | data_per_sample | 10k_subject_epoch_io |
|--------------|-----------------|----------------------|
| FULL_VOLUME  | 35.6 MB         | 356 GB               |
| PATCH (64^3) | 262 KB          | 2.6 GB               |
| PATCH (128^3)| 2.1 MB          | 21 GB                |

---

## Multi-Worker Scaling

| num_workers | time_3_volumes | time_per_volume |
|-------------|----------------|-----------------|
| 0           | 0.16s          | 0.05s           |
| 1           | 6.06s          | 2.02s           |
| 2           | 11.14s         | 3.71s           |

`num_workers=0` for <100 volumes, `num_workers=4-8` for >1000 volumes.

See [Performance: Why Multi-Worker DataLoaders Slow Down](../explanation/performance.md#why-multi-worker-dataloaders-slow-down-small-datasets) for interpretation.

---

## Cache Hit Rates (TileDB)

TileDB maintains a built-in LRU tile cache within its context object. Zarr v3 has **no built-in chunk cache** — it relies on OS page cache for local reads and has no caching for S3. This is a key TileDB advantage for workloads with repeated or overlapping access patterns.

| access_pattern                | tiledb_hit_rate | zarr  |
|-------------------------------|-----------------|-------|
| Sequential (shared context)   | 85-95%          | OS page cache only |
| Random (shared context)       | 60-75%          | OS page cache only |
| Repeated slices (same volume) | 90-99%          | OS page cache only |
| S3 repeated access            | 85-95%          | No cache (re-fetches) |
| Isolated contexts             | 0%              | 0%    |

---

## Parallel Write Scaling

| workers | local_ssd_mb_s | s3_mb_s |
|---------|----------------|---------|
| 1       | ~140           | ~50     |
| 4       | ~410           | ~150    |
| 8       | ~650           | ~250    |
| 16      | ~800           | ~300    |

Throughput scales linearly on local SSD (~140 MB/s per worker).

---

## GIL Interaction

| operation            | gil_released | parallel_speedup          |
|----------------------|--------------|---------------------------|
| TileDB I/O           | Yes          | ~3-6x with 4 workers     |
| TileDB decompression | Yes          | ~3-4x with 4 workers     |
| NumPy array ops      | Partially    | ~1.5-2x                  |
| Pure Python          | No           | ~1x                       |

---

## RadiObject + MONAI/TorchIO

RadiObject is a **storage layer** that complements MONAI and TorchIO transforms:

```python
from radiobject.ml.compat import as_torchio_subject

subject = as_torchio_subject(radi.T1w.iloc[0], radi.seg.iloc[0])
augmented = my_torchio_transform(subject)
```

The benchmarks compare **I/O performance only**. Use RadiObject for data loading, then apply MONAI/TorchIO transforms.

---

## Running Benchmarks

```bash
# Export AWS credentials for S3 benchmarks (use your own profile)
eval $(aws configure export-credentials --profile <your-profile> --format env)
python benchmarks/run_experiments.py --all
```

See `benchmarks/README.md` for details.
