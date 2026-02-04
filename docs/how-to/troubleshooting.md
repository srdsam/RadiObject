# Troubleshooting

Common issues and solutions when working with RadiObject.

## macOS DataLoader Workers with S3

**Symptom:** Hang or crash when using `num_workers > 0` in PyTorch DataLoader with S3-backed data.

**Cause:** macOS uses `fork()` for multiprocessing by default, which is unsafe with S3 connections. The forked child inherits the parent's TileDB context, but S3 connections are not fork-safe.

**Fix:** Use `num_workers=0` (single-process loading):

```python
loader = create_training_dataloader(
    collections=radi.T1w,
    num_workers=0,  # Required on macOS with S3
)
```

Alternatively, set the multiprocessing start method to `spawn`:

```python
import torch.multiprocessing as mp
mp.set_start_method("spawn", force=True)
```

## TileDB Context Errors

**Symptom:** `tiledb.TileDBError: Context already finalized` or similar context-related errors.

**Cause:** Sharing TileDB contexts across forked processes.

**Fix:** Use `ctx_for_process()` in forked workers, not `ctx_for_threads()`:

```python
from radiobject.parallel import ctx_for_process

def worker_fn():
    ctx = ctx_for_process()  # Creates a new context for this process
    vol = Volume(uri, ctx=ctx)
```

See [Threading Model](../explanation/threading-model.md) for details on context isolation.

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

## Slow S3 Performance

**Symptom:** Operations on S3-backed data are slower than expected.

**Checklist:**

1. **Same-region access?** Ensure compute and S3 bucket are in the same AWS region
2. **Parallel ops configured?**
   ```python
   from radiobject import configure, S3Config
   configure(s3=S3Config(max_parallel_ops=16))
   ```
3. **Using partial reads?** Prefer `vol.axial(z)` or `vol.slice(...)` over `vol.to_numpy()` when possible
4. **Check network bandwidth** with `iperf3` or similar tools

See [Tuning Concurrency](tuning-concurrency.md) and [S3 Setup](s3-setup.md) for optimization guidance.

## Related Documentation

- [Configuration](../reference/configuration.md) - All configuration options
- [Profiling](profiling.md) - Measure and diagnose performance
- [Threading Model](../explanation/threading-model.md) - Context isolation details
