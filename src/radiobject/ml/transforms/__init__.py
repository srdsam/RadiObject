"""MONAI-backed transforms for medical imaging.

This module provides transforms using MONAI's well-tested implementations.
All transforms operate on dict data with an "image" key containing the tensor.
"""

from typing import Any, Callable, Sequence

from monai.transforms import (
    Compose as _MonaiCompose,
)
from monai.transforms import (
    NormalizeIntensityd,
    RandFlipd,
    RandGaussianNoised,
    RandSpatialCropd,
    Resized,
    ScaleIntensityRanged,
)


class Compose:
    """Compose multiple transforms into a single callable."""

    def __init__(self, transforms: Sequence[Callable[[dict[str, Any]], dict[str, Any]]]):
        self._compose = _MonaiCompose(transforms)

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        return self._compose(data)

    def __repr__(self) -> str:
        return repr(self._compose)


class IntensityNormalize:
    """Normalize intensity to zero mean and unit variance."""

    def __init__(self, channel_wise: bool = True):
        self._transform = NormalizeIntensityd(
            keys=["image"],
            channel_wise=channel_wise,
        )

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        return self._transform(data)


class WindowLevel:
    """Apply window/level (contrast) adjustment for CT data."""

    def __init__(self, window: float, level: float):
        self.window = window
        self.level = level
        min_val = level - window / 2
        max_val = level + window / 2
        self._transform = ScaleIntensityRanged(
            keys=["image"],
            a_min=min_val,
            a_max=max_val,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        )

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        return self._transform(data)


class RandomNoise:
    """Add random Gaussian noise."""

    def __init__(self, std: float = 0.1, prob: float = 0.5):
        self._transform = RandGaussianNoised(
            keys=["image"],
            std=std,
            prob=prob,
        )

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        return self._transform(data)


class RandomFlip3D:
    """Random flip along specified axes."""

    def __init__(self, axes: tuple[int, ...] = (0, 1, 2), prob: float = 0.5):
        self._transforms = [
            RandFlipd(keys=["image"], spatial_axis=axis, prob=prob) for axis in axes
        ]
        self._compose = _MonaiCompose(self._transforms)

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        return self._compose(data)


class RandomCrop3D:
    """Random crop to specified size."""

    def __init__(self, size: tuple[int, int, int]):
        self.size = size
        self._transform = RandSpatialCropd(
            keys=["image"],
            roi_size=size,
            random_size=False,
        )

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        original_shape = data["image"].shape[1:]
        result = self._transform(data)
        # Record crop_start for backwards compatibility
        # When crop equals input size, start is (0, 0, 0)
        if result["image"].shape[1:] == original_shape:
            result["crop_start"] = (0, 0, 0)
        else:
            # MONAI doesn't expose start position, estimate from center crop behavior
            result["crop_start"] = tuple(
                (orig - crop) // 2 for orig, crop in zip(original_shape, self.size)
            )
        return result


class Resample3D:
    """Resample to target shape using MONAI's Resized."""

    def __init__(self, target_shape: tuple[int, int, int], mode: str = "trilinear"):
        self._transform = Resized(
            keys=["image"],
            spatial_size=target_shape,
            mode=mode,
        )

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        return self._transform(data)


__all__ = [
    "Compose",
    "IntensityNormalize",
    "RandomCrop3D",
    "RandomFlip3D",
    "RandomNoise",
    "Resample3D",
    "WindowLevel",
]
