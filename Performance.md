# RadiObject Performance Analysis

## Test Suite Overview

**Total Tests:** 203 tests across 6 test files (+ 17 ML tests)
**Test Run Date:** 2026-01-29
**Total Time:** ~2m (core) + ~43s (ML)
**Data Source:** Real MSD Brain Tumour (NIfTI) and NSCLC-Radiomics (DICOM) datasets

### Test Optimization (2026-01-28)

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Total tests | 182 | 139 | -43 (S3 consolidation) |
| Total time | 17m 36s | 4m 27s | **4× faster** |
| S3 tests | 48 | 5 | 90% reduction |

**Key optimizations:**
1. Module-scoped fixtures for read-only tests (eliminates redundant 428MB data creation)
2. Consolidated S3 tests to essential integration checks (logic tested locally)

### API Simplification (2026-01-29)

| Metric | Before | After | Notes |
|--------|--------|-------|-------|
| Total tests | 139 | 203 (+17 ML) | Additional coverage |
| Core test time | ~4m | ~2m | **2× faster** |
| Dead code | ~60 lines | 0 | Removed |
| Code savings | - | ~255 lines | DRY consolidation |

**Key optimizations:**
1. Volume metadata caching (single TileDB open per Volume)
2. Lazy volume loading in VolumeCollection (on-demand instantiation)
3. Efficient obs_meta indexing (only load index column)
4. InMemoryCache now provides >1000× speedup for repeated access

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
| `view.to_radi_object` (materialize) | 0.99s | Full 428 MB | Copy operation |

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

| Strategy | Repeated Access | Speedup | Memory |
|----------|-----------------|---------|--------|
| NoCache | 0.226s | 1× | 0 |
| InMemoryCache | <0.001s | **>1000×** | ~36 MB/volume |
| DiskCache | ~0.02s | ~10× | Disk space |

Cache hit rate for 3 volumes accessed 3× each: **66.7%** (3 misses on first access, 6 hits on repeats)

*Note: InMemoryCache speedup dramatically improved after API simplification (metadata caching, lazy loading).*

### DataLoader Multi-Worker Performance

| Workers | 3 Volumes Total Time | Per-Volume | Notes |
|---------|---------------------|------------|-------|
| 0 (main process) | 0.16s | 0.05s | Best for small datasets |
| 1 | 6.06s | 2.02s | IPC overhead dominates |
| 2 | 11.14s | 3.71s | More overhead, no benefit |

**Observation:** For small datasets (3 volumes), single-process loading outperforms multi-worker due to IPC serialization overhead. Multi-worker benefits emerge with larger datasets where parallel I/O amortizes the overhead.

*Note: Single-process (workers=0) performance improved ~40% after lazy loading optimization.*

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
        _PROCESS_CTX_CACHE[pid] = create_worker_ctx()
    return _PROCESS_CTX_CACHE[pid]
```

This enables:
- Multi-worker DataLoaders within each node
- Independent S3 connections per worker
- No shared state between processes

#### 2. DistributedSampler Integration

The `create_distributed_dataloader` factory handles DDP partitioning:

```python
from ml.distributed import create_distributed_dataloader, set_epoch

loader = create_distributed_dataloader(
    radi_object,
    rank=rank,           # This node's rank
    world_size=world_size,  # Total nodes
    modalities=["T1w", "FLAIR"],
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
3. **Caching:** InMemoryCache provides 12× speedup for repeated access patterns
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
    radi,
    modalities=["T1w"],
    patch_size=(64, 64, 64),  # 136× less I/O than full volume
    batch_size=32,
    num_workers=4,
)
```

### Recommended Configurations

#### Small Dataset (<1,000 subjects)

```python
loader = create_training_dataloader(
    radi,
    batch_size=4,
    num_workers=0,           # Single process is faster
    cache_strategy=CacheStrategy.IN_MEMORY,  # Cache all volumes
)
```

#### Medium Dataset (1,000-10,000 subjects)

```python
loader = create_training_dataloader(
    radi,
    patch_size=(96, 96, 96),  # Patch-based to reduce I/O
    batch_size=16,
    num_workers=4,            # Parallel loading helps
    pin_memory=True,
    persistent_workers=True,
)
```

#### Large Dataset (>10,000 subjects, Multi-Node)

```python
from ml.distributed import create_distributed_dataloader

loader = create_distributed_dataloader(
    radi,
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
5. **Caching layers:** 12× speedup with in-memory caching for repeated access

**Practical scaling limits:**
- Single node: ~10,000 subjects (I/O bound past this)
- Multi-node (8×): ~100,000 subjects per epoch in reasonable time (~7 hours)
- Patch-based training: 136× reduction in I/O enables much larger datasets

For very large datasets (>100,000 subjects), consider:
- Smaller patch sizes (64³ vs 128³)
- More aggressive caching
- High-bandwidth instances (p4d, p5)
- Pre-processing to local NVMe for hot data
