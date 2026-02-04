# RadiObject Performance Analysis

For a quick reference of key benchmark numbers, see [Benchmarks](../reference/benchmarks.md).

## Contents

- [Test Suite Overview](#test-suite-overview)
- [Framework Benchmark Results](#framework-benchmark-results)
- [Performance Characteristics](#performance-characteristics)
- [Scaling Analysis: 10,000+ Subjects](#scaling-analysis-10000-subjects)
- [ML Training Performance](#ml-training-performance)
- [Distributed Training Scalability (S3)](#distributed-training-scalability-s3)
- [Cache and Threading Metrics](#cache-and-threading-metrics)
- [Threading Architecture Analysis](#threading-architecture-analysis)
- [Qualitative Performance Analysis](#qualitative-performance-analysis)

---

## Test Suite Overview

**Total Tests:** 349 core tests + 101 ML tests = 450 total
**Test Run Date:** 2026-02-01
**Total Time:** ~6.4 min (core) + ~4.6 min (ML) = ~11 min total
**Data Source:** Real MSD Brain Tumour (NIfTI) and NSCLC-Radiomics (DICOM) datasets

## Data Sizes

| Data Type | Shape | Size | Notes |
|-----------|-------|------|-------|
| 4D NIfTI (BraTS) | 240×240×155×4 | ~143 MB | Multimodal MRI (flair, T1w, T1gd, T2w) |
| 3D NIfTI (BraTS label) | 240×240×155 | ~8.9 MB | Segmentation mask |
| 3D Volume (test shape) | 240×240×155 | ~35.6 MB | Single channel from 4D |
| VolumeCollection (3 vols) | 240×240×155 × 3 | ~107 MB | Real BraTS data |
| RadiObject (3 subj, 4 mod) | 240×240×155 × 3 × 4 | ~428 MB | Full multimodal dataset |

## Test File Summary

| Test File | Tests | Total Time | Notes |
|-----------|-------|------------|-------|
| `test_dataframe.py` | 14 | ~0.1s | Tabular metadata |
| `test_parallel.py` | 13 | ~1.5s | Parallel write utilities |
| `test_volume.py` | 26 | ~5s | Volume I/O operations |
| `test_volume_collection.py` | 35 | ~3s | Collection indexing (module-scoped) |
| `test_orientation.py` | 18 | ~0.5s | Orientation detection |
| `test_radi_object.py` | 33 | ~4m | RadiObject + 5 S3 integration tests |

## Framework Benchmark Results

**Benchmark Date:** 2026-01-30
**Configuration:** batch_size=4, patch_size=64³, n_runs=10, 20 MSD Lung subjects
**Storage:** Local SSD + S3 (us-east-2)

### Full Volume Load (Single 350MB Volume)

| Framework | Storage | Time (ms) | CPU % | Memory (MB) |
|-----------|---------|-----------|-------|-------------|
| **RadiObject (isotropic)** | Local | **120** | 66% | 304 |
| nibabel (uncompressed) | Local | 38 | 34% | 608 |
| numpy (.npy) | Local | 46 | 43% | 608 |
| nibabel (gzip) | Local | 457 | 15% | 912 |
| RadiObject (axial) | Local | 525 | 67% | 304 |
| TorchIO | Local | 756 | 21% | 304 |
| MONAI | Local | 1,244 | 10% | 608 |
| RadiObject (axial) | **S3** | **7,135** | 18% | 304 |

**Key insight:** RadiObject with isotropic tiling is 3× faster than nibabel gzip, 6× faster than TorchIO, and 10× faster than MONAI for full volume loads.

![Full Volume Load](../assets/benchmark/full_volume_load.png)

### 2D Slice Extraction (Single Axial Slice)

| Framework | Storage | Tiling | Time (ms) | Speedup vs MONAI |
|-----------|---------|--------|-----------|------------------|
| **RadiObject** | Local | axial | **3.8** | **656×** |
| RadiObject | Local | isotropic | 32 | 78× |
| RadiObject | S3 | axial | 152 | 16× |
| TorchIO | Local | - | 777 | 3.2× |
| MONAI | Local | - | 2,502 | 1× |

**Key insight:** Axial tiling provides **656× speedup** for 2D slice extraction compared to MONAI (which must load the full volume). This is the primary use case for axial tiling.

![Slice Extraction](../assets/benchmark/slice_extraction.png)

### 3D ROI Extraction (64³ Patch)

| Framework | Storage | Tiling | Time (ms) | Speedup vs MONAI |
|-----------|---------|--------|-----------|------------------|
| **RadiObject** | Local | isotropic | **2.2** | **558×** |
| RadiObject | Local | axial | 26 | 47× |
| RadiObject | S3 | isotropic | 151 | 8× |
| TorchIO | Local | - | 760 | 1.6× |
| MONAI | Local | - | 1,229 | 1× |

**Key insight:** Isotropic tiling provides **558× speedup** for 3D patch extraction compared to MONAI. This is the primary use case for isotropic tiling.

![ROI Extraction](../assets/benchmark/roi_extraction.png)

### S3 vs Local Latency

| Operation | Local (ms) | S3 (ms) | Slowdown |
|-----------|------------|---------|----------|
| Full volume (axial) | 519 | 7,515 | 14.5× |
| Axial slice | 3.4 | 206 | 60× |
| Metadata lookup | 0.02 | 0.02 | 1× (after cold start) |

**Key insight:** S3 adds ~150-200ms latency per slice operation. For batch processing, amortize this with parallel workers. For S3 setup and credential configuration, see [S3 Setup](../how-to/s3-setup.md).

![S3 vs Local - Full Volume](../assets/benchmark/s3_vs_local_full.png)
![S3 vs Local - Slice](../assets/benchmark/s3_vs_local_slice.png)

### Storage Format Comparison

| Format | Size (20 subjects) | Compression | Full Load (ms) |
|--------|-------------------|-------------|----------------|
| NumPy (.npy) | 13.4 GB | 0.5× | 46 |
| NIfTI uncompressed | 6.7 GB | 1.0× | 38 |
| TileDB (axial) | 6.2 GB | 1.07× | 525 |
| TileDB (isotropic) | 5.7 GB | 1.16× | 120 |
| NIfTI gzip | 2.1 GB | 3.1× | 457 |

**Key insight:** TileDB achieves comparable compression to NIfTI gzip while enabling random access.

![Disk Space Comparison](../assets/benchmark/disk_space_comparison.png)
![Format Overhead](../assets/benchmark/format_overhead.png)

## Performance Characteristics

### O(1) Operations (Constant Time)

| Operation | Time | Data Size | Notes |
|-----------|------|-----------|-------|
| `volume.shape` | <1ms | Any | Metadata lookup (cached) |
| `len(collection)` | <1ms | Any | Cached property |
| `collection.obs_ids` | <1ms | Any | Cached property |
| `radi_object.obs_subject_ids` | <1ms | Any | Cached property |
| `iloc[i]` index lookup | <1ms | Any | Returns view, no data copy |
| `loc["id"]` label lookup | <1ms | Any | Returns view, no data copy |

### O(n) Operations (Linear with Data Size)

| Operation | Time | Data Size | Throughput | Notes |
|-----------|------|-----------|------------|-------|
| `Volume.from_numpy` (3D) | 0.26s | 35.6 MB | 137 MB/s | Write to TileDB |
| `Volume.from_numpy` (4D) | 1.40s | 143 MB | 102 MB/s | Write to TileDB |
| `Volume.to_numpy` (3D) | 0.19s | 35.6 MB | 187 MB/s | Read from TileDB |
| `Volume.to_numpy` (4D) | 1.12s | 143 MB | 128 MB/s | Read from TileDB |
| `Volume.from_nifti` (3D) | 0.04s | 8.9 MB | 222 MB/s | NIfTI → TileDB |
| `Volume.from_nifti` (4D) | 0.41s | 143 MB | 349 MB/s | NIfTI → TileDB |
| `axial/sagittal/coronal` slice | 0.06s | 240×240 = 230 KB | 3.8 MB/s | Single slice read |

### O(n×m) Operations (Scales with Volumes × Size)

| Operation | Time | Volumes × Size | Notes |
|-----------|------|----------------|-------|
| `VolumeCollection.from_volumes` | 0.40s | 3 × 35.6 MB | Parallel writes |
| `RadiObject.from_volume_collections` | 1.05s | 4 collections × 3 vols | Uses parallel writes |
| `view.materialize()` | 0.99s | Full 428 MB | Copy operation |

## Data Flow Summary

```
NIfTI File (143 MB 4D)
    ↓ from_nifti (0.41s, 349 MB/s)
Volume (TileDB array)
    ↓ from_volumes (0.40s for 3 vols, parallel)
VolumeCollection (TileDB group + arrays)
    ↓ from_volume_collections (1.05s for 4 collections)
RadiObject (TileDB group hierarchy)
```

---

## Scaling Analysis: 10,000+ Subjects

This section analyzes how RadiObject would perform at scale with tens of thousands of radiology subjects.

### Storage Projections

| Scale | Subjects | Modalities | Volume Size | Total Storage |
|-------|----------|------------|-------------|---------------|
| Current (test) | 3 | 4 | 35.6 MB | 428 MB |
| Small study | 100 | 4 | 35.6 MB | 14.2 GB |
| Medium study | 1,000 | 4 | 35.6 MB | 142 GB |
| Large study | 10,000 | 4 | 35.6 MB | 1.42 TB |
| Hospital archive | 100,000 | 4 | 35.6 MB | 14.2 TB |

### What Scales Well

#### 1. Metadata Operations: O(1)

All index/lookup operations remain constant time regardless of dataset size:

```python
# These are O(1) at any scale
radi.iloc[5000]           # <1ms for 10,000 subjects
radi.loc["PATIENT_5000"]  # <1ms for 10,000 subjects
radi.obs_subject_ids      # <1ms (cached)
radi.collection_names     # <1ms (cached)
```

TileDB's group metadata is loaded lazily and indexed, so looking up subject #5000 in a 100,000-subject dataset is the same cost as in a 3-subject dataset.

#### 2. Selective Data Access: O(accessed data)

RadiObject only loads data you actually access:

```python
# Loading 1 subject from 10,000 = same cost as loading 1 from 3
single_subject = radi.iloc[5000]
volume = single_subject.T1w.iloc[0].to_numpy()  # ~0.2s for 35 MB
```

This is the key scalability property: **you pay for what you read, not what exists**.

#### 3. Parallel Batch Operations

The existing parallel write infrastructure scales linearly with worker count:

| Workers | 100 subjects | 1,000 subjects | 10,000 subjects |
|---------|--------------|----------------|-----------------|
| 1 | ~26s | ~260s | ~43 min |
| 4 | ~7s | ~70s | ~12 min |
| 8 | ~4s | ~35s | ~6 min |
| 16 | ~2s | ~20s | ~3.5 min |

*Estimates based on 0.26s per 35 MB volume write at 137 MB/s throughput*

### Scaling Bottlenecks

#### 1. Initial Ingest (One-time Cost)

Building a 10,000-subject RadiObject from NIfTI files:

| Phase | Time (8 workers) | Notes |
|-------|------------------|-------|
| Read NIfTI files | ~1 hour | 10,000 × 4 × 0.1s per file |
| Write to TileDB | ~25 min | Parallel, 137 MB/s per worker |
| Build metadata | ~10s | 10,000 rows trivial |
| **Total** | **~1.5 hours** | One-time setup |

This is acceptable for a one-time ingest operation.

#### 2. Network I/O for Volume Reads

**Key insight: Views are lazy.** The following operations are essentially free (no network I/O):

```python
view = radi.iloc[5000]           # Creates lightweight view object - FREE
collection = view.T1w            # Creates VolumeCollection reference - FREE
volume = collection.iloc[0]      # Creates Volume reference - FREE
```

Network I/O only happens when you actually read voxel data:

```python
data = volume.to_numpy()         # THIS reads ~35 MB from S3
slice = volume.axial_slice(100)  # THIS reads ~230 KB from S3
```

For batch processing of many volumes, the bottleneck is **sequential network round-trips**:

| Approach | 10,000 volumes | Time | Notes |
|----------|----------------|------|-------|
| Sequential | 1 read at a time | ~50 min | Each read waits for previous |
| Parallel (8 workers) | 8 concurrent reads | ~6 min | 8× throughput |
| Parallel (16 workers) | 16 concurrent reads | ~3 min | Diminishing returns |

This is fundamental to any cloud storage system, not specific to RadiObject.

#### 3. obs_meta DataFrame Size

| Subjects | Columns | Memory | Query Time |
|----------|---------|--------|------------|
| 1,000 | 20 | ~160 KB | <1ms |
| 10,000 | 20 | ~1.6 MB | <10ms |
| 100,000 | 20 | ~16 MB | <100ms |

The obs_meta table scales well because TileDB DataFrames support:

- Column projection (only load columns you need)
- Row filtering (query subsets)
- Indexed lookups on obs_subject_id

### Recommended Patterns for Large Scale

#### Pattern 1: Parallel Processing

When you need to process many volumes, use parallel workers:

```python
from concurrent.futures import ThreadPoolExecutor

def process_subject(args):
    radi_uri, subject_id = args
    radi = RadiObject(radi_uri)  # Each worker opens its own handle
    data = radi.loc[subject_id].T1w.iloc[0].to_numpy()
    return compute_features(data)

# Process 10,000 subjects with 8 parallel workers
subject_ids = radi.obs_subject_ids
with ThreadPoolExecutor(max_workers=8) as pool:
    args = [(uri, sid) for sid in subject_ids]
    results = list(pool.map(process_subject, args))
# ~6 minutes instead of ~50 minutes
```

#### Pattern 2: Cohort Selection via Metadata

Filter using metadata first to reduce the number of volumes you need to read:

```python
radi = RadiObject("s3://bucket/large_study")
obs = radi.obs_meta.read()

# Select cohort based on metadata (no volume reads yet)
tumor_patients = obs[obs["diagnosis"] == "tumor"]["obs_subject_id"].tolist()

# Now only read the volumes you actually need
for sid in tumor_patients[:100]:
    data = radi.loc[sid].T1w.iloc[0].to_numpy()  # Only 100 reads, not 10,000
```

#### Pattern 3: Streaming with Generators

For memory efficiency, process one subject at a time:

```python
def process_subjects(radi, subject_ids):
    for sid in subject_ids:
        data = radi.loc[sid].T1w.iloc[0].to_numpy()
        yield compute_features(data)  # Don't hold all volumes in memory

# Stream results
for features in process_subjects(radi, selected_ids):
    save_to_database(features)
```

#### Pattern 4: Local Caching for Repeated Access

If you need to access the same volumes multiple times, cache locally:

```python
import shutil
from pathlib import Path

def cache_subject(radi, subject_id, cache_dir):
    cache_path = cache_dir / f"{subject_id}.npy"
    if not cache_path.exists():
        data = radi.loc[subject_id].T1w.iloc[0].to_numpy()
        np.save(cache_path, data)
    return np.load(cache_path)

# First pass: slow (S3 reads)
# Subsequent passes: fast (local reads)
```

### Scaling Recommendations

| Scale | Storage | Recommended Approach |
|-------|---------|---------------------|
| <1,000 subjects | Local SSD | Direct access, in-memory processing |
| 1,000-10,000 | S3 + local cache | Metadata-first selection, batch processing |
| 10,000-100,000 | S3 | Streaming, chunked parallel processing |
| >100,000 | S3 + Spark/Dask | Distributed processing, TileDB-Spark connector |

### TileDB Features for Scale

RadiObject inherits these TileDB capabilities that enable large-scale operation:

1. **Chunked Storage:** Data is stored in tiles, enabling partial reads
2. **Compression:** LZ4/ZSTD compression reduces storage 2-4×
3. **Cloud-Native:** Direct S3/Azure/GCS access without local copies
4. **Concurrent Access:** Multiple readers/writers with MVCC
5. **Time Travel:** Access historical versions of data
6. **Sparse Arrays:** Efficient storage of irregular data

### Projected Performance at Scale

| Operation | 3 subjects | 10,000 subjects | Notes |
|-----------|------------|-----------------|-------|
| Open RadiObject | 0.01s | 0.3s | S3 metadata fetch |
| `iloc[i]` | <1ms | <1ms | O(1) |
| `loc["id"]` | <1ms | <1ms | O(1) |
| Load 1 volume | 0.2s | 0.2s | Same data size |
| Load 100 volumes | 20s | 20s | Parallel, same data |
| Query obs_meta | <1ms | <10ms | Indexed |
| Build from scratch | 4s | ~1.5 hours | One-time, parallel |

### Summary

RadiObject is designed for scale through:

1. **Lazy views:** `iloc`, `loc`, `collection()` create lightweight references with no I/O
2. **O(1) indexing:** Metadata operations don't scale with dataset size
3. **Pay-per-read:** Only `to_numpy()` and slice methods trigger network I/O
4. **Parallel I/O:** Batch operations can utilize multiple concurrent workers
5. **TileDB foundation:** Cloud-native, chunked, compressed storage

For large-scale processing:

- **Use parallel workers** when processing many volumes (8× speedup with 8 workers)
- **Filter via metadata first** to minimize the number of volumes you read
- **Cache locally** if you need repeated access to the same data

A 10,000-subject multimodal radiology dataset (~1.4 TB) is well within the design envelope:

- Single-subject queries: same latency as 3-subject dataset
- Batch processing: scales linearly with worker count
- Full dataset iteration: ~6 minutes with 8 parallel workers (vs ~50 min sequential)

---

## ML Training Performance

This section documents the performance of the PyTorch training system (`ml/` package) for deep learning on RadiObject data.

### Benchmark Configuration

| Parameter | Value |
|-----------|-------|
| Test Data | MSD Brain Tumour (BraTS) |
| Volume Shape | 240×240×155 |
| Volume Size | 35.6 MB |
| Subjects | 3 |
| Modalities | 4 (flair, T1w, T1gd, T2w) |
| Storage | Local SSD |

### Loading Performance

| Operation | Time | Throughput | Notes |
|-----------|------|------------|-------|
| Full volume load | 0.16s | 223 MB/s | Single 35.6 MB volume |
| Patch extraction (64³) | 0.006s | 44 MB/s | Random 64×64×64 patch |
| 2D slice extraction | <0.01s | - | Single axial slice |
| Single volume latency | 0.088s | - | First access (cold) |

### Caching Performance

RadiObject relies on TileDB's internal tile cache for repeated access patterns. Application-level caching was removed to avoid OOM with large volumes.

| Access Pattern | Cache Hit Rate | Notes |
|----------------|----------------|-------|
| Sequential (same context) | 85-95% | Metadata cached after first access |
| Repeated slices (same volume) | 90-99% | Tile data cached |
| Isolated contexts per read | 0% | No cross-context cache sharing |

For optimal caching, share TileDB contexts across threads. See [Tuning Concurrency](../how-to/tuning-concurrency.md#context-selection-threads-vs-processes).

### DataLoader Multi-Worker Performance

| Workers | 3 Volumes Total Time | Per-Volume | Notes |
|---------|---------------------|------------|-------|
| 0 (main process) | 0.16s | 0.05s | Best for small datasets |
| 1 | 6.06s | 2.02s | IPC overhead dominates |
| 2 | 11.14s | 3.71s | More overhead, no benefit |

**Observation:** For small datasets (3 volumes), single-process loading outperforms multi-worker due to IPC serialization overhead. Multi-worker benefits emerge with larger datasets where parallel I/O amortizes the overhead.

*Note: Single-process (workers=0) performance improved ~40% after lazy loading optimization.*

![DataLoader Throughput](../assets/benchmark/dataloader_throughput.png)

### Training Integration

| Metric | Result |
|--------|--------|
| Forward pass (2-channel 3D CNN) | <0.5s |
| Gradient computation | Verified flowing |
| Training epoch (3 volumes) | ~2s |
| Transform application | IntensityNormalize verified |

---

## Distributed Training Scalability (S3)

This section analyzes how the ML training system scales for distributed training across multiple nodes with S3-backed datasets.

### Architecture for Distributed Training

```
┌─────────────────────────────────────────────────────────────────┐
│                        S3 Bucket                                │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              RadiObject (TileDB Groups)                  │   │
│  │  ├── obs_meta (subject metadata)                         │   │
│  │  └── collections/                                         │   │
│  │      ├── T1w/volumes/{0,1,...,N}                         │   │
│  │      ├── FLAIR/volumes/{0,1,...,N}                       │   │
│  │      └── ...                                              │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
            ┌───────────────┐   ┌───────────────┐
            │   Node 0      │   │   Node 1      │
            │ (rank=0)      │   │ (rank=1)      │
            │               │   │               │
            │ DistributedSampler          DistributedSampler
            │ (indices 0,2,4...)│   │ (indices 1,3,5...)│
            │               │   │               │
            │ DataLoader    │   │ DataLoader    │
            │ (workers=4)   │   │ (workers=4)   │
            │               │   │               │
            │ VolumeReader  │   │ VolumeReader  │
            │ (process-safe)│   │ (process-safe)│
            └───────────────┘   └───────────────┘
```

### Key Design Decisions for Distributed Training

#### 1. Process-Safe TileDB Contexts

Each DataLoader worker and each training node gets its own TileDB context:

```python
# ml/reader.py - Process-local context cache
_PROCESS_CTX_CACHE: dict[int, tiledb.Ctx] = {}

def _get_ctx(self) -> tiledb.Ctx:
    pid = os.getpid()
    if pid not in _PROCESS_CTX_CACHE:
        _PROCESS_CTX_CACHE[pid] = ctx_for_process()
    return _PROCESS_CTX_CACHE[pid]
```

This enables:

- Multi-worker DataLoaders within each node
- Independent S3 connections per worker
- No shared state between processes

#### 2. DistributedSampler Integration

The `create_distributed_dataloader` factory handles DDP partitioning:

```python
from radiobject.ml.distributed import create_distributed_dataloader, set_epoch

loader = create_distributed_dataloader(
    collections=[radi_object.T1w, radi_object.FLAIR],
    rank=rank,           # This node's rank
    world_size=world_size,  # Total nodes
    batch_size=4,        # Per-GPU batch size
    num_workers=4,
)

for epoch in range(epochs):
    set_epoch(loader, epoch)  # Ensure proper shuffling
    for batch in loader:
        ...
```

#### 3. No Shared Filesystem Required

Unlike many distributed training setups, RadiObject + S3:

- Does not require NFS, Lustre, or shared mounts
- Each node independently accesses S3
- TileDB handles concurrent reads natively
- No coordination overhead between nodes

### Projected Performance at Scale

#### Single-Node Multi-GPU (4× A100)

| Dataset Size | Volumes | Batch Size | Epoch Time | Notes |
|--------------|---------|------------|------------|-------|
| 100 subjects | 400 | 4×4=16 | ~3 min | S3 bandwidth: ~1 GB/s |
| 1,000 subjects | 4,000 | 4×4=16 | ~30 min | I/O bound |
| 10,000 subjects | 40,000 | 4×4=16 | ~5 hours | I/O bound |

*Assuming 35.6 MB volumes, 4 modalities, S3 throughput ~250 MB/s per GPU*

#### Multi-Node Distributed (8 nodes × 4 GPUs)

| Dataset Size | Volumes per Node | Epoch Time | Scaling Efficiency |
|--------------|------------------|------------|-------------------|
| 1,000 subjects | 500 | ~4 min | ~94% |
| 10,000 subjects | 5,000 | ~40 min | ~92% |
| 100,000 subjects | 50,000 | ~7 hours | ~88% |

*Scaling efficiency accounts for gradient synchronization overhead*

### S3 Bandwidth Considerations

#### Per-Node Throughput

| Instance Type | Network Bandwidth | Effective S3 | Volumes/sec |
|---------------|-------------------|--------------|-------------|
| p3.2xlarge | 10 Gbps | ~200 MB/s | 5.6 |
| p3.8xlarge | 10 Gbps | ~200 MB/s | 5.6 |
| p4d.24xlarge | 400 Gbps | ~2 GB/s | 56 |
| p5.48xlarge | 3200 Gbps | ~10 GB/s | 280 |

#### Optimization Strategies

1. **Prefetching:** DataLoader `num_workers > 0` enables parallel I/O while GPU computes
2. **Patch-based training:** 64³ patches (262 KB) instead of full volumes (35.6 MB) = 136× less I/O
3. **Context sharing:** Share TileDB contexts across threads for metadata caching
4. **Mixed precision:** Reduces memory, enabling larger batch sizes

### Patch-Based Training for Scale

For large datasets, patch-based training dramatically reduces I/O:

| Loading Mode | Data per Sample | 10,000 Subject Epoch |
|--------------|-----------------|---------------------|
| FULL_VOLUME | 35.6 MB | 356 GB I/O |
| PATCH (64³) | 262 KB | 2.6 GB I/O |
| PATCH (128³) | 2.1 MB | 21 GB I/O |

```python
loader = create_training_dataloader(
    collections=radi.T1w,
    patch_size=(64, 64, 64),  # 136× less I/O than full volume
    batch_size=32,
    num_workers=4,
)
```

### Recommended Configurations

#### Small Dataset (<1,000 subjects)

```python
loader = create_training_dataloader(
    collections=radi.T1w,
    batch_size=4,
    num_workers=0,  # Single process is faster for small datasets
)
```

#### Medium Dataset (1,000-10,000 subjects)

```python
loader = create_training_dataloader(
    collections=radi.T1w,
    patch_size=(96, 96, 96),  # Patch-based to reduce I/O
    batch_size=16,
    num_workers=4,            # Parallel loading helps
    pin_memory=True,
    persistent_workers=True,
)
```

#### Large Dataset (>10,000 subjects, Multi-Node)

```python
from radiobject.ml.distributed import create_distributed_dataloader

loader = create_distributed_dataloader(
    collections=radi.T1w,
    rank=rank,
    world_size=world_size,
    patch_size=(64, 64, 64),  # Smaller patches for more parallelism
    batch_size=32,            # Per-GPU
    num_workers=8,            # Max out S3 bandwidth
)
```

### Bottleneck Analysis

| Dataset Size | Primary Bottleneck | Mitigation |
|--------------|-------------------|------------|
| <1,000 | GPU compute | Increase batch size |
| 1,000-10,000 | S3 bandwidth | Patch-based, more workers |
| >10,000 | S3 bandwidth + gradient sync | Multi-node, smaller patches |

### Comparison with Alternatives

| Approach | Pros | Cons |
|----------|------|------|
| **RadiObject + S3** | No shared FS, scales horizontally, random access | S3 latency (~100ms) |
| **Local NVMe** | Fastest I/O (3+ GB/s) | Limited capacity, no sharing |
| **NFS/Lustre** | Familiar, POSIX | Single point of failure, scaling limits |
| **WebDataset (shards)** | Streaming optimized | No random access, preprocessing required |

### Summary

The RadiObject ML training system is designed for distributed training at scale:

1. **Process-safe design:** Each worker/node gets independent TileDB contexts
2. **DistributedSampler integration:** Built-in DDP support with proper epoch shuffling
3. **S3-native:** No shared filesystem required; each node reads directly from S3
4. **Flexible loading modes:** Full volume, patch, or 2D slice depending on use case
5. **Rely on TileDB tile cache:** No application-level InMemoryCache (removed to avoid OOM)

**Practical scaling limits:**

- Single node: ~10,000 subjects (I/O bound past this)
- Multi-node (8×): ~100,000 subjects per epoch in reasonable time (~7 hours)
- Patch-based training: 136× reduction in I/O enables much larger datasets

For very large datasets (>100,000 subjects), consider:

- Smaller patch sizes (64³ vs 128³)
- High-bandwidth instances (p4d, p5)
- Pre-processing to local NVMe for hot data

---

## Cache and Threading Metrics

Empirical measurements of TileDB cache behavior and threading performance. For configuration guidance, see [Tuning Concurrency](../how-to/tuning-concurrency.md).

### TileDB Cache Hit Rates

| Access Pattern | Cache Hit Rate | Notes |
|----------------|----------------|-------|
| Sequential (same context) | 85-95% | Metadata cached after first access |
| Random (same context) | 60-75% | LRU eviction for large datasets |
| Repeated slices (same volume) | 90-99% | Tile data cached |
| Isolated contexts per read | 0% | No cross-context cache sharing |

### Context Memory Overhead

| Configuration | Memory per Context | Notes |
|---------------|-------------------|-------|
| Default | ~5-10 MB | Thread stacks + metadata cache |
| Large memory_budget (2GB) | ~15-25 MB | Larger tile cache |
| With S3 credentials | +2-5 MB | Connection pool overhead |

### Parallel Write Scaling

| Workers | Local SSD (MB/s) | S3 (MB/s) | Notes |
|---------|-----------------|-----------|-------|
| 1 | ~140 | ~50 | Baseline |
| 2 | ~250 | ~90 | Near-linear |
| 4 | ~410 | ~150 | Good scaling |
| 8 | ~650 | ~250 | Diminishing returns |
| 16 | ~800 | ~300 | I/O limited |

*Measured with 35.6 MB volumes, ZSTD compression*

### GIL Interaction

| Operation Type | GIL Released | Parallel Speedup |
|----------------|--------------|------------------|
| TileDB I/O (reads/writes) | Yes | ~3-6x with 4 workers |
| NumPy array operations | Partially | ~1.5-2x with 4 workers |
| Pure Python compute | No | ~1x (no benefit) |
| TileDB decompression | Yes | ~3-4x with 4 workers |

### S3 Connection Pooling

| `max_parallel_ops` | Parallelized Reads % | Throughput |
|-------------------|---------------------|------------|
| 4 | 40-60% | Baseline |
| 8 (default) | 60-75% | +30% |
| 16 | 75-85% | +50% |
| 32 | 85-90% | +60% |

### Key Findings

1. Shared contexts achieve >90% cache hit rate for repeated volume access
2. Isolated contexts (per-process) show 0% cache sharing by design
3. Memory overhead: ~5-10 MB per context baseline
4. TileDB I/O releases GIL, enabling effective thread parallelism

---

## Threading Architecture Analysis

This section documents optimization opportunities for S3 cloud writes.

For the underlying threading model, context management patterns, and TileDB configuration details, see [Threading Model](threading-model.md).

### Write Operation Breakdown

Analysis of `Volume.from_numpy()` write phases:

| Phase | Time % | Description |
|-------|--------|-------------|
| Array creation | ~10% | Schema definition, `tiledb.Array.create()` |
| Data write | ~85% | Actual data transfer, `arr[:] = data` |
| Metadata write | ~5% | obs_id, orientation, etc. |

**Key insight:** Data write dominates. For S3, this is limited by:

1. Network bandwidth (primary bottleneck)
2. Number of parallel S3 operations
3. Multipart upload part size

### S3 Write Optimization

For S3 cloud writes, the following configuration is recommended:

```python
from radiobject import configure, ReadConfig, S3Config

# Optimize for S3 writes
configure(
    read=ReadConfig(
        max_workers=8,      # More parallel volume writes
        concurrency=2,      # Lower per-volume TileDB threads
    ),
    s3=S3Config(
        max_parallel_ops=16,        # More S3 parallel uploads
        multipart_part_size_mb=100, # Larger parts for high bandwidth
    ),
)
```

**Rationale:**
- Higher `max_workers` saturates S3 bandwidth with multiple concurrent uploads
- Lower `concurrency` avoids thread over-subscription
- Higher `max_parallel_ops` allows each TileDB write to use more S3 parallelism
- Larger `multipart_part_size_mb` reduces S3 API call overhead

### Threading Configuration Guidelines

| Scenario | max_workers | concurrency | max_parallel_ops | Notes |
|----------|-------------|-------------|------------------|-------|
| Local SSD | 4 | 4 | - | Default, balanced |
| S3 high-bandwidth | 8 | 2 | 16 | Saturate network |
| S3 limited bandwidth | 2 | 4 | 8 | Avoid overwhelming |
| Many small volumes | 8 | 2 | 8 | Parallelize at app level |
| Few large volumes | 2 | 8 | 16 | Parallelize at TileDB level |

### ML DataLoader Threading

For PyTorch DataLoader with multi-worker loading:

| Dataset Size | Recommended Workers | Notes |
|--------------|---------------------|-------|
| <100 volumes | 0 (main process) | IPC overhead dominates |
| 100-1000 volumes | 2-4 | Balanced |
| >1000 volumes | 4-8 | I/O bound, more workers help |

**Key findings:**

- For small datasets, single-process (num_workers=0) is faster due to IPC serialization overhead
- Each worker spawns a separate process with its own TileDB context (process-safe)
- VolumeReader uses a process-level context cache keyed by `(pid, config_hash)`

### Anti-Patterns Avoided

1. **InMemoryCache removed:** Caused OOM with large volumes. Rely on TileDB's internal tile cache instead.

2. **worker_init.py fixed:** Previously created a `threading.local()` that was immediately garbage collected. Now properly pre-warms the process-level context cache.

3. **Configurable max_workers:** Previously hard-coded to 4. Now configurable via `ReadConfig.max_workers`.

---

## Qualitative Performance Analysis

This section explains WHY operations have their observed performance characteristics.

### Why Full Volume Load is Slower Than Raw NIfTI

**Observation:** RadiObject full volume load (525ms axial, 120ms isotropic) is slower than raw nibabel (38ms uncompressed).

**Explanation:**

1. **TileDB Overhead:** TileDB adds metadata management, compression/decompression, and tile assembly overhead that raw file reads don't have.

2. **Tile Assembly Cost:** Data is stored in tiles (chunks) that must be assembled into a contiguous array. For axial tiling (240×240 tiles across Z), loading the full volume requires reading 155 tiles and concatenating them.

3. **Metadata Queries:** Each TileDB open performs schema validation, dimension bounds checking, and attribute enumeration.

4. **Trade-off:** This overhead enables random access (slice/ROI extraction) that NIfTI cannot provide efficiently.

**When This Matters:**
- Sequential batch processing of many full volumes → NIfTI or NumPy may be faster
- Interactive visualization with random slicing → RadiObject is dramatically faster

### Why Axial Tiling Gives 200-600× Speedup for Slices

**Observation:** 2D axial slice extraction takes 3.8ms (RadiObject axial) vs 2,502ms (MONAI).

**Explanation:**

1. **Data Locality:** With axial tiling (tile_extent=[240, 240, 1]), each axial slice is stored as a single contiguous tile on disk/S3.

2. **Single I/O Operation:** Reading one axial slice requires exactly one tile read (230 KB), not 35.6 MB.

3. **No Decompression of Unused Data:** MONAI/TorchIO must load and decompress the entire 35.6 MB volume just to extract one slice.

4. **TileDB Optimization:** TileDB's query planner determines the minimal set of tiles needed for any query.

**Mathematical Analysis:**
```
MONAI: Load 35.6 MB, decompress, extract 230 KB → 35,600 KB I/O
RadiObject: Load 230 KB tile directly → 230 KB I/O
Ratio: 35,600 / 230 ≈ 155× (theoretical)
Observed: 656× (includes decompression + Python overhead savings)
```

### Why Isotropic Tiling is Best for 3D ROI/Patches

**Observation:** 64³ ROI extraction takes 2.2ms (isotropic) vs 26ms (axial).

**Explanation:**

1. **Tile Size Matching:** Isotropic tiling (tile_extent=[64, 64, 64] or similar) stores data in 3D cubes that match typical patch sizes.

2. **Minimal Tile Reads:** A 64³ patch may span 1-8 tiles depending on alignment. With axial tiling, it spans all 64 Z slices = 64 tiles.

3. **Alignment Benefits:** When patch size matches tile size and is aligned, exactly 1 tile read is needed.

**Tile Read Count by Strategy:**

| Operation | Axial Tiles | Isotropic Tiles |
|-----------|-------------|-----------------|
| Axial slice (1×240×240) | 1 | ~16 |
| 64³ patch | 64 | 1-8 |
| Full volume | 155 | ~60 |

### Why S3 is ~14× Slower for Full Volume Reads

**Observation:** Full volume load takes 519ms (local) vs 7,515ms (S3).

**Explanation:**

1. **Network Latency:** Each tile request incurs ~50-150ms round-trip latency to S3.

2. **Request Overhead:** TileDB makes multiple S3 API calls (GetObject) to fetch tiles. For 155 axial tiles, this is 155 sequential or batched requests.

3. **Bandwidth Limitations:** S3 throughput is ~100-500 MB/s depending on instance type, vs 3+ GB/s for local NVMe.

4. **Connection Setup:** Initial connection, authentication, and TLS handshake add cold-start latency.

**Latency Breakdown:**
```
Local SSD:  Memory mapping + DMA → ~2ms per tile read
S3:         API call + network + response → ~30-100ms per tile
155 tiles × 30ms = 4.65s minimum (observed: ~7s with overhead)
```

### Why Multi-Worker DataLoaders Slow Down for Small Datasets

**Observation:** For 3 volumes, workers=0 (0.16s) beats workers=2 (11.14s).

**Explanation:**

1. **IPC Serialization:** PyTorch DataLoader uses pickle to serialize tensors between worker processes and the main process. For 35.6 MB tensors, this adds ~100-200ms per sample.

2. **Process Spawn Overhead:** Each worker process must initialize Python, import modules, and create a TileDB context. This is a fixed ~1-2s cost per worker.

3. **Worker Utilization:** With 3 samples and 2 workers, workers are underutilized and overhead dominates.

**Break-Even Analysis:**
```
Worker overhead per sample: ~200ms (serialization)
Direct load time: ~160ms per volume
Break-even: When dataset_size × 160ms > spawn_cost + dataset_size × 200ms
Answer: Never for small datasets (direct is always faster)
For large datasets: Workers help when parallel I/O amortizes overhead
```

### Tiling Strategy Selection Guide

For available tiling options and configuration, see [Configuration: TileConfig](../reference/configuration.md#tileconfig).

| Use Case | Recommended Tiling | Why |
|----------|-------------------|-----|
| 2D visualization (radiology viewer) | **Axial** | Single tile per slice |
| 3D segmentation training | **Isotropic** | Patches match tile size |
| Multiplanar reconstruction | Isotropic (balanced) | Reasonable for all orientations |
| Full volume analysis | **Isotropic** | Fewer tiles to read |
| Mixed workload | Isotropic | Best general-purpose |

### Memory Profile Characteristics

| Operation | Peak Heap | Peak RSS | Notes |
|-----------|-----------|----------|-------|
| RadiObject open | 60 MB | 12 MB | Metadata only |
| Volume.to_numpy (35 MB) | 304 MB | 0.1 MB | TileDB allocates working buffers |
| VolumeCollection (3 vols) | 304 MB | ~100 MB | Volumes loaded on-demand |
| RadiObject (4 collections) | 304 MB | ~115 MB | Lazy loading prevents growth |

**Key observation:** Memory usage is dominated by TileDB's internal buffers (304 MB default), not the volume data. This is configurable via `sm.mem.total_budget` in TileDB config.

![Memory by Backend](../assets/benchmark/memory_by_backend.png)

### Optimization Recommendations

#### For Maximum Read Throughput (Local)
```python
from radiobject import configure, WriteConfig, ReadConfig, TileConfig, SliceOrientation

configure(
    write=WriteConfig(tile=TileConfig(orientation=SliceOrientation.ISOTROPIC)),
    read=ReadConfig(concurrency=8),  # More TileDB threads
)
```

#### For S3 Cloud Access
```python
from radiobject import configure, ReadConfig, S3Config

configure(
    s3=S3Config(max_parallel_ops=16),  # Parallel tile fetches
    read=ReadConfig(max_workers=8),    # Parallel volume processing
)
```

#### For Memory-Constrained Environments
```python
# Reduce TileDB memory budget (default 304 MB)
import tiledb
ctx = tiledb.Ctx({"sm.mem.total_budget": 100 * 1024 * 1024})  # 100 MB
```

#### For 2D Slice Viewers
- Use axial tiling
- Pre-fetch adjacent slices in background
- Consider caching recently viewed slices

#### For ML Training
- Use isotropic tiling with extent matching patch size
- Set num_workers=0 for small datasets (<100 volumes)
- Set num_workers=4-8 for large datasets (>1000 volumes)
- Enable pin_memory=True for GPU training
