# ML Integration

RadiObject focuses on efficient data loading from TileDB/S3 with partial reads. Use [MONAI](https://docs.monai.io/en/stable/transforms.html) or [TorchIO](https://torchio.readthedocs.io/) for transforms and augmentation. For preprocessing concepts (normalization, augmentation), see the [Lexicon](../reference/lexicon.md#data-processing-concepts).

## Choosing a DataLoader

| Approach | Best For | Framework |
|----------|----------|-----------|
| `create_training_dataloader()` | Standard training with dict transforms | MONAI |
| `VolumeCollectionSubjectsDataset` | Patch-based training with queues | TorchIO |

**Decision guide:**

1. **Using MONAI dict transforms?** → Use `create_training_dataloader()`
2. **Using TorchIO patch-based sampling?** → Use `VolumeCollectionSubjectsDataset` with `tio.Queue`
3. **Custom dataset logic needed?** → Subclass `VolumeCollectionDataset`

## With MONAI Transforms

`VolumeCollectionDataset` outputs `{"image": tensor, ...}` — compatible with MONAI dict transforms:

```python
from monai.transforms import Compose, NormalizeIntensityd, RandFlipd
from radiobject.ml import create_training_dataloader

transform = Compose([
    NormalizeIntensityd(keys="image"),
    RandFlipd(keys="image", prob=0.5, spatial_axis=[0, 1, 2]),
])

loader = create_training_dataloader(collections=radi.CT, transform=transform)
```

## With TorchIO Transforms

Use `VolumeCollectionSubjectsDataset` for TorchIO's Queue-based training:

```python
from radiobject.ml import VolumeCollectionSubjectsDataset
import torchio as tio

dataset = VolumeCollectionSubjectsDataset(collections=radi.T1w)
transform = tio.Compose([tio.ZNormalization(), tio.RandomFlip()])
queue = tio.Queue(dataset, max_length=100, samples_per_volume=10)
```

## Installation

```bash
# MONAI only
pip install radiobject[monai]

# TorchIO only
pip install radiobject[torchio]

# Both frameworks
pip install radiobject[ml]
```

## DataLoader Factory Functions

RadiObject provides factory functions for common training scenarios:

### Classification / Regression

```python
from radiobject.ml import create_training_dataloader

loader = create_training_dataloader(
    collections=radi.T1w,      # Single or list of VolumeCollections
    labels="diagnosis",         # Column name, DataFrame, dict, or callable
    batch_size=8,
    num_workers=4,
)
```

### Segmentation

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

### Validation

```python
from radiobject.ml import create_validation_dataloader

loader = create_validation_dataloader(
    collections=radi.T1w,
    labels="diagnosis",
    batch_size=8,
    # No shuffle, no drop_last
)
```

### Inference

```python
from radiobject.ml import create_inference_dataloader

loader = create_inference_dataloader(
    collections=radi.T1w,
    batch_size=1,
    # Full volumes, no shuffle
)
```

### DatasetConfig Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `loading_mode` | `FULL_VOLUME` | `FULL_VOLUME`, `PATCH`, or `SLICE_2D` |
| `patch_size` | `None` | Patch dimensions if using `PATCH` mode |
| `patches_per_volume` | `1` | Patches extracted per volume per epoch |

For complete API reference, see [ML Module API](../api/ml.md).

## Best Practices

1. **Use RadiObject for I/O**: Let RadiObject handle data loading from TileDB/S3
2. **Use MONAI/TorchIO for transforms**: Apply augmentation after loading
3. **Partial reads for patches**: RadiObject excels at loading small regions efficiently
4. **Full volumes for heavy augmentation**: Load complete volumes when doing spatial transforms

## Performance Notes

**S3-backed training** adds latency (~100-200ms per volume) compared to local storage. Use patch-based training to reduce I/O (64³ patch = 136x less data than full volume), and for small datasets (<100 volumes), use `num_workers=0` to avoid IPC overhead.

For worker/thread configuration recipes, see [Tuning Concurrency](tuning-concurrency.md). For detailed benchmarks and scaling analysis, see [Performance Analysis: ML Training](../explanation/performance-analysis.md#ml-training-performance).

## Next Step

**Training running but slow?** Tune worker counts and threading with [Tuning Concurrency](tuning-concurrency.md). For diagnosing bottlenecks, see [Profiling](profiling.md).

## Related Documentation

- [Benchmarks](../reference/benchmarks.md) - Performance comparisons
- [Volume Operations](volume-operations.md) - Partial reads for patch-based training
