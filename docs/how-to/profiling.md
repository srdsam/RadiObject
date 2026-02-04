# Profiling & Performance Monitoring

How to measure RadiObject and TileDB performance.

## TileDBStats

Collect read/write statistics using a context manager:

```python
from radiobject import TileDBStats

with TileDBStats() as stats:
    vol = radi.T1w.iloc[0]
    data = vol.to_numpy()

print(stats)  # Shows timing breakdown
```

The stats object provides detailed breakdowns including:

- Total read/write time
- Tile fetch counts
- Decompression time

## CacheStats

Monitor TileDB tile cache effectiveness:

```python
from radiobject import CacheStats

with TileDBStats() as stats:
    # Load same volume multiple times
    for _ in range(5):
        data = vol.to_numpy()

cache = stats.cache_stats()
print(f"Cache hits: {cache.cache_hits}")
print(f"Cache misses: {cache.cache_misses}")
print(f"Hit rate: {cache.hit_rate:.1%}")
```

### Interpreting Cache Hit Rates

| Hit Rate | Interpretation |
|----------|---------------|
| >90% | Excellent - metadata well cached |
| 60-90% | Good - some LRU eviction |
| <60% | Poor - consider sharing contexts |

## S3Stats

Track S3 operations for cloud deployments:

```python
from radiobject import S3Stats

with TileDBStats() as stats:
    vol = RadiObject("s3://bucket/dataset").T1w.iloc[0]
    data = vol.to_numpy()

s3 = stats.s3_stats()
print(f"Read ops: {s3.read_ops}")
print(f"Parallelized: {s3.parallelized_reads}")
print(f"Parallelization rate: {s3.parallelization_rate:.1%}")
```

## Common Profiling Scenarios

### Diagnosing Slow Reads

```python
with TileDBStats() as stats:
    data = vol.to_numpy()

# Check if decompression is the bottleneck
print(f"Decompression time: {stats.decompression_time:.3f}s")
print(f"I/O time: {stats.io_time:.3f}s")
```

### Measuring Cache Hit Rates

```python
from radiobject import ctx

# Share context across operations for caching
shared_ctx = ctx()

with TileDBStats() as stats:
    for uri in volume_uris:
        vol = Volume(uri, ctx=shared_ctx)
        _ = vol.to_numpy()

cache = stats.cache_stats()
print(f"Hit rate: {cache.hit_rate:.1%}")
```

### S3 Latency Analysis

```python
import time

# Measure cold start
start = time.perf_counter()
radi = RadiObject("s3://bucket/dataset")
cold_start = time.perf_counter() - start

# Measure subsequent access
start = time.perf_counter()
_ = radi.T1w.iloc[0].to_numpy()
warm_access = time.perf_counter() - start

print(f"Cold start: {cold_start:.2f}s")
print(f"Warm access: {warm_access:.2f}s")
```

## Diagnosing Performance Issues

Use this decision tree to identify and fix performance problems:

```
Slow reads?
├── Check cache hit rate
│   ├── <60%? → Share TileDB contexts (see Scenario 3 below)
│   └── >60%? → Check I/O vs decompression time
│       ├── I/O dominant? → Increase max_parallel_ops (S3) or check disk
│       └── Decompression dominant? → Lower compression level or use faster codec
│
Slow S3 access?
├── Cold start >5s? → Expected for first connection
├── Warm reads still slow?
│   ├── Check parallelization rate
│   │   ├── <50%? → Increase max_parallel_ops
│   │   └── >50%? → Network bandwidth limited
│   └── Consider patch-based reads instead of full volumes
│
Memory issues (OOM)?
├── Reduce num_workers in DataLoader
├── Reduce memory_budget_mb
└── Use streaming with iter_volumes() instead of loading all
```

## Related Documentation

- [Performance Analysis](../explanation/performance-analysis.md) - Detailed benchmark data
- [Tuning Concurrency](tuning-concurrency.md) - Optimize threading settings
