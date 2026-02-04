# Tuning Concurrency Settings

This guide provides practical configuration recipes for RadiObject's concurrency settings.

For the underlying threading architecture and context management, see [Threading Model](../explanation/threading-model.md).

## Quick Configuration Reference

### Local SSD Storage

```python
from radiobject import configure, ReadConfig

# Default - balanced for most local workloads
configure(
    read=ReadConfig(
        max_workers=4,
        concurrency=4,
        memory_budget_mb=1024,
    )
)
```

### S3 Cloud Storage

For AWS credential setup and general S3 guidance, see [S3 Setup](s3-setup.md).

```python
from radiobject import configure, ReadConfig, S3Config

# High-bandwidth instance (p4d, p5)
configure(
    read=ReadConfig(
        max_workers=8,      # More parallel volume operations
        concurrency=2,      # Fewer threads per operation
    ),
    s3=S3Config(
        max_parallel_ops=32,        # Maximize S3 parallelism
        multipart_part_size_mb=100, # Larger parts for bandwidth
    ),
)

# Limited bandwidth or many small files
configure(
    read=ReadConfig(
        max_workers=4,
        concurrency=4,
    ),
    s3=S3Config(
        max_parallel_ops=8,
        multipart_part_size_mb=50,
    ),
)
```

## Context Selection: Threads vs Processes

RadiObject provides `ctx_for_threads()` and `ctx_for_process()` for correct context handling in parallel code. For the underlying threading model and TileDB caching behavior, see [Threading Model](../explanation/threading-model.md#worker-context-functions).

| Scenario | Function | Behavior |
|----------|----------|----------|
| `ThreadPoolExecutor` | `ctx_for_threads(ctx)` | Returns same context (shared caching) |
| `multiprocessing.Pool` | `ctx_for_process()` | Creates isolated context |
| PyTorch DataLoader (`num_workers>0`) | `ctx_for_process()` | Creates isolated context |

**Quick reference:**

```python
# Threads: share context for caching
from radiobject import tdb_ctx
from radiobject.parallel import ctx_for_threads

shared_ctx = tdb_ctx()
def thread_worker(uri):
    vol = Volume(uri, ctx=ctx_for_threads(shared_ctx))
    return vol.to_numpy()

# Processes: isolated contexts (required)
from radiobject.parallel import ctx_for_process

def process_worker(uri):
    vol = Volume(uri, ctx=ctx_for_process())
    return vol.to_numpy()
```

## [PyTorch DataLoader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) Configuration

### Small Datasets (<100 volumes)

```python
from radiobject.ml import create_training_dataloader

loader = create_training_dataloader(
    collections=radi.T1w,
    batch_size=4,
    num_workers=0,  # Main process - avoids IPC overhead
)
```

**Why `num_workers=0`?**
- IPC serialization (~100-200ms per 35MB tensor) dominates for small datasets
- Single process is faster than multi-worker for <100 volumes

### Large Datasets (>1000 volumes)

```python
loader = create_training_dataloader(
    collections=[radi.T1w, radi.FLAIR],  # Multi-modal: stacked as channels
    batch_size=16,
    num_workers=4,       # Parallel loading
    pin_memory=True,     # Faster GPU transfer
    persistent_workers=True,  # Avoid respawn overhead
)
```

### Distributed Training

```python
from radiobject.ml.distributed import create_distributed_dataloader

loader = create_distributed_dataloader(
    collections=radi.T1w,
    rank=rank,
    world_size=world_size,
    batch_size=8,        # Per-GPU batch size
    num_workers=4,
)
```

## Measuring Cache Performance

Use `TileDBStats` to understand your workload's cache behavior. For comprehensive profiling guidance including interpretation and common scenarios, see [Profiling](profiling.md).

```python
from radiobject import TileDBStats

# Measure cache hit rate
with TileDBStats() as stats:
    for uri in volume_uris:
        vol = Volume(uri)
        _ = vol.to_numpy()

cache = stats.cache_stats()
print(f"Cache hits: {cache.cache_hits}")
print(f"Cache misses: {cache.cache_misses}")
print(f"Hit rate: {cache.hit_rate:.1%}")

# For S3 operations
s3 = stats.s3_stats()
print(f"Read ops: {s3.read_ops}")
print(f"Parallelized: {s3.parallelized_reads}")
print(f"Parallelization rate: {s3.parallelization_rate:.1%}")
```

## Common Tuning Scenarios

### Scenario 1: Slow S3 Full Volume Reads

**Symptoms:** Full volume reads from S3 taking 10+ seconds

**Solution:** Increase parallelism at VFS level:

```python
configure(
    s3=S3Config(
        max_parallel_ops=32,  # Up from default 8
    ),
)
```

### Scenario 2: Memory Exhaustion with Many Workers

**Symptoms:** OOM errors when using many DataLoader workers

**Solution:** Reduce workers or memory budget:

```python
configure(
    read=ReadConfig(
        max_workers=2,         # Reduce parallel operations
        memory_budget_mb=512,  # Reduce TileDB buffer
    ),
)
```

### Scenario 3: Poor Cache Hit Rate

**Symptoms:** Cache hit rate below 50% for repeated access

**Solution:** Ensure context sharing for threads:

```python
# Wrong - creates new context each time
def bad_load(uri):
    vol = Volume(uri)  # Uses new global context lookup
    return vol.to_numpy()

# Right - shares context explicitly
shared_ctx = tdb_ctx()
def good_load(uri):
    vol = Volume(uri, ctx=ctx_for_threads(shared_ctx))
    return vol.to_numpy()
```

### Scenario 4: GIL Contention in Compute-Heavy Workloads

**Symptoms:** Parallel processing not faster than serial for NumPy operations

**Explanation:** Python's GIL limits parallelism for CPU-bound operations.

**Solutions:**
1. Use `multiprocessing` instead of threads for CPU-bound work
2. Move compute to native code (NumPy operations release GIL for array operations)
3. Use separate processes for compute-heavy transforms

```python
# For CPU-bound transforms, use process pool
from multiprocessing import Pool

def compute_features(uri):
    ctx = ctx_for_process()  # Isolated context
    vol = Volume(uri, ctx=ctx)
    data = vol.to_numpy()
    # Heavy NumPy operations here
    return np.mean(data), np.std(data)

with Pool(processes=4) as pool:
    results = pool.map(compute_features, volume_uris)
```

## Performance Guidelines Summary

| Workload | `max_workers` | `concurrency` | `num_workers` (DL) |
|----------|---------------|---------------|-------------------|
| Local, small dataset | 4 | 4 | 0 |
| Local, large dataset | 4-8 | 4 | 4 |
| S3, high bandwidth | 8 | 2 | 4-8 |
| S3, limited bandwidth | 2-4 | 4 | 2-4 |
| Memory constrained | 2 | 2 | 2 |

## Related Documentation

- [Configuration](../reference/configuration.md) - All configuration options and defaults
- [Threading Model](../explanation/threading-model.md) - Context management architecture
- [Performance Analysis](../explanation/performance-analysis.md) - Benchmark data and scaling
- [ML Integration](ml-training.md) - DataLoader factory functions
