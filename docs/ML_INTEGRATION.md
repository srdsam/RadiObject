# ML Integration

RadiObject focuses on efficient data loading from TileDB/S3 with partial reads. Use MONAI or TorchIO for transforms and augmentation.

## With MONAI Transforms

RadiObjectDataset outputs `{"image": tensor, ...}` - compatible with MONAI dict transforms:

```python
from monai.transforms import Compose, NormalizeIntensityd, RandFlipd
from radiobject.ml import create_training_dataloader

transform = Compose([
    NormalizeIntensityd(keys="image"),
    RandFlipd(keys="image", prob=0.5, spatial_axis=[0, 1, 2]),
])

loader = create_training_dataloader(radi, modalities=["CT"], transform=transform)
```

## With TorchIO Transforms

Use `RadiObjectSubjectsDataset` for TorchIO's Queue-based training:

```python
from radiobject.ml import RadiObjectSubjectsDataset
import torchio as tio

dataset = RadiObjectSubjectsDataset(radi, modalities=["T1w"])
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

## Best Practices

1. **Use RadiObject for I/O**: Let RadiObject handle data loading from TileDB/S3
2. **Use MONAI/TorchIO for transforms**: Apply augmentation after loading
3. **Partial reads for patches**: RadiObject excels at loading small regions efficiently
4. **Full volumes for heavy augmentation**: Load complete volumes when doing spatial transforms

See [Benchmarks](BENCHMARKS.md) for performance comparisons.
