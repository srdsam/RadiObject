# Troubleshooting

Common issues and solutions when working with RadiObject.

## macOS DataLoader Fork Issue

**Symptom:** Hang or crash when using `num_workers > 0` in PyTorch DataLoader with S3-backed data.

**Cause:** macOS uses `fork()` for multiprocessing by default, which is unsafe with S3 connections.

**Fix:** Use `num_workers=0` (single-process loading):

```python
loader = create_training_dataloader(
    collections=radi.T1w,
    num_workers=0,  # Required on macOS with S3
)
```

Or set the multiprocessing start method to `spawn`:

```python
import torch.multiprocessing as mp
mp.set_start_method("spawn", force=True)
```

## Out of Memory (OOM)

**Symptom:** Process killed or `MemoryError` during large dataset operations.

**Fixes:**

1. **Reduce memory budget:**
   ```python
   from radiobject import configure, ReadConfig
   configure(read=ReadConfig(memory_budget_mb=512))
   ```

2. **Use streaming instead of loading all volumes:**
   ```python
   for vol in radi.lazy().filter("split == 'train'").iter_volumes():
       process(vol.to_numpy())
   ```

3. **Reduce DataLoader workers:**
   ```python
   loader = create_training_dataloader(collections=radi.T1w, num_workers=2)
   ```

4. **Use patch-based training** to avoid loading full volumes:
   ```python
   loader = create_segmentation_dataloader(
       image=radi.CT, mask=radi.seg, patch_size=(96, 96, 96),
   )
   ```

## Slow Reads Diagnostic Tree

```
Slow reads?
├── Check cache hit rate (stats.cache_stats().hit_rate)
│   ├── <60%? → Share TileDB contexts across operations
│   └── >60%? → Check all_counters() for read latency breakdown
│       ├── Network/disk dominant? → Increase max_parallel_ops (S3) or check disk
│       └── Consider lower compression level or faster codec (LZ4)
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

Use `TileDBStats` to measure:

```python
from radiobject import TileDBStats

with TileDBStats() as stats:
    data = vol.to_numpy()

counters = stats.all_counters()
for key, value in sorted(counters.items()):
    if value > 0:
        print(f"{key}: {value}")
```

## S3 Latency

**Cold start (>5s):** Expected for first S3 connection. Subsequent reads are faster.

**Warm reads still slow:**

```python
from radiobject import configure, S3Config
configure(s3=S3Config(max_parallel_ops=16))
```

Prefer partial reads (`vol.axial(z)`, `vol.slice(...)`) over `vol.to_numpy()` when possible.

## Context Errors

**Symptom:** `tiledb.TileDBError: Context already finalized` or similar.

**Cause:** Sharing TileDB contexts across forked processes.

**Fix:** Use `ctx_for_process()` in forked workers, not `ctx_for_threads()`:

```python
from radiobject.parallel import ctx_for_process

def worker_fn():
    ctx = ctx_for_process()
    vol = Volume(uri, ctx=ctx)
```

See [Architecture: Concurrency Model](../explanation/architecture.md#concurrency-model) for details.
