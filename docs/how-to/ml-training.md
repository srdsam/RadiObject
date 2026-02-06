# ML Training

RadiObject handles efficient data loading from TileDB/S3 with partial reads. Use [MONAI](https://docs.monai.io/en/stable/transforms.html) or [TorchIO](https://torchio.readthedocs.io/) for transforms and augmentation.

## Installation

```bash
pip install radiobject[monai]     # MONAI only
pip install radiobject[torchio]   # TorchIO only
pip install radiobject[ml]        # Both frameworks
```

## Choosing a DataLoader

| Approach | Best For | Framework |
|----------|----------|-----------|
| `create_training_dataloader()` | Standard training with dict transforms | MONAI |
| `create_segmentation_dataloader()` | Segmentation with foreground sampling | MONAI |
| `VolumeCollectionSubjectsDataset` | Patch-based training with queues | TorchIO |

**Decision guide:**

1. **Using MONAI dict transforms?** Use `create_training_dataloader()`
2. **Segmentation with foreground sampling?** Use `create_segmentation_dataloader()`
3. **Using TorchIO patch-based sampling?** Use `VolumeCollectionSubjectsDataset` with `tio.Queue`
4. **Custom dataset logic?** Subclass `VolumeCollectionDataset`

![Dataloader throughput comparison](../assets/benchmarks/dataloader_throughput.png)

## MONAI Integration

`VolumeCollectionDataset` outputs `{"image": tensor, ...}` — compatible with MONAI dict transforms:

```python
from monai.transforms import Compose, NormalizeIntensityd, RandFlipd
from radiobject.ml import create_training_dataloader

transform = Compose([
    NormalizeIntensityd(keys="image"),
    RandFlipd(keys="image", prob=0.5, spatial_axis=[0, 1, 2]),
])

loader = create_training_dataloader(
    collections=radi.T1w,
    labels="diagnosis",
    transform=transform,
    batch_size=8,
    num_workers=4,
)
```

### Validation and Inference

```python
from radiobject.ml import create_validation_dataloader, create_inference_dataloader

val_loader = create_validation_dataloader(
    collections=radi.T1w, labels="diagnosis", batch_size=8,
)

inf_loader = create_inference_dataloader(
    collections=radi.T1w, batch_size=1,
)
```

## Segmentation

```python
from radiobject.ml import create_segmentation_dataloader

loader = create_segmentation_dataloader(
    image=radi.CT,
    mask=radi.seg,
    batch_size=4,
    patch_size=(96, 96, 96),
    foreground_sampling=True,   # Bias toward non-zero mask regions
)
```

## TorchIO Integration

```python
from radiobject.ml import VolumeCollectionSubjectsDataset
import torchio as tio

dataset = VolumeCollectionSubjectsDataset(collections=radi.T1w)
transform = tio.Compose([tio.ZNormalization(), tio.RandomFlip()])
queue = tio.Queue(dataset, max_length=100, samples_per_volume=10)
```

## DatasetConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `loading_mode` | `FULL_VOLUME` | `FULL_VOLUME`, `PATCH`, or `SLICE_2D` |
| `patch_size` | `None` | Patch dimensions (required for `PATCH` mode) |
| `patches_per_volume` | `1` | Patches extracted per volume per epoch |

For complete API reference, see [ML Module API](../api/ml.md).

## Performance Tuning

### Worker Configuration

`num_workers=0` for <100 volumes (avoids IPC overhead), `num_workers=4-8` for >1000 volumes with `pin_memory=True` and `persistent_workers=True`. For S3, increase `max_parallel_ops`.

```python
# Large dataset example
loader = create_training_dataloader(
    collections=[radi.T1w, radi.FLAIR],
    batch_size=16,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
)
```

### Context Handling: Threads vs Processes

RadiObject provides `ctx_for_threads()` and `ctx_for_process()` for correct context handling in parallel code:

| Scenario | Function | Behavior |
|----------|----------|----------|
| `ThreadPoolExecutor` | `ctx_for_threads(ctx)` | Returns same context (shared caching) |
| `multiprocessing.Pool` | `ctx_for_process()` | Creates isolated context |
| DataLoader (`num_workers>0`) | `ctx_for_process()` | Creates isolated context |

```python
from radiobject import get_tiledb_ctx
from radiobject.parallel import ctx_for_threads, ctx_for_process

# Threads: share context for caching
shared_ctx = get_tiledb_ctx()
def thread_worker(uri):
    vol = Volume(uri, ctx=ctx_for_threads(shared_ctx))
    return vol.to_numpy()

# Processes: isolated contexts (required)
def process_worker(uri):
    vol = Volume(uri, ctx=ctx_for_process())
    return vol.to_numpy()
```

### ReadConfig Tuning

```python
from radiobject import configure, ReadConfig, S3Config

# Local SSD
configure(read=ReadConfig(max_workers=4, concurrency=4, memory_budget_mb=1024))

# S3, high bandwidth
configure(
    read=ReadConfig(max_workers=8, concurrency=2),
    s3=S3Config(max_parallel_ops=32, multipart_part_size_mb=100),
)
```

### Measuring Cache Performance

```python
from radiobject import TileDBStats

with TileDBStats() as stats:
    for uri in volume_uris:
        vol = Volume(uri)
        _ = vol.to_numpy()

cache = stats.cache_stats()
print(f"Hit rate: {cache.hit_rate:.1%}")

s3 = stats.s3_stats()
print(f"Parallelization rate: {s3.parallelization_rate:.1%}")
```

### Common Tuning Scenarios

**Slow S3 full volume reads:** Increase `S3Config(max_parallel_ops=32)`.

**OOM with many workers:** Reduce `ReadConfig(max_workers=2, memory_budget_mb=512)`.

**Poor cache hit rate (<50%):** Ensure threads share context via `ctx_for_threads()` rather than creating new contexts per call.

**GIL contention for CPU-bound transforms:** Use `multiprocessing.Pool` with `ctx_for_process()` instead of threads.

**S3-backed training latency:** Use patch-based training to reduce I/O (64^3 patch = 136x less data). For small datasets (<100 vols), use `num_workers=0` to avoid IPC overhead.
