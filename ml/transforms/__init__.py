"""MONAI-compatible transforms for medical imaging."""

from typing import Any, Callable, Sequence

from ml.transforms.intensity import IntensityNormalize, RandomNoise, WindowLevel
from ml.transforms.spatial import RandomCrop3D, RandomFlip3D, Resample3D


class Compose:
    """Compose multiple transforms into a single callable."""

    def __init__(self, transforms: Sequence[Callable[[dict[str, Any]], dict[str, Any]]]):
        self.transforms = list(transforms)

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        for t in self.transforms:
            data = t(data)
        return data

    def __repr__(self) -> str:
        names = [t.__class__.__name__ for t in self.transforms]
        return f"Compose({names})"


__all__ = [
    "Compose",
    "IntensityNormalize",
    "RandomCrop3D",
    "RandomFlip3D",
    "RandomNoise",
    "Resample3D",
    "WindowLevel",
]
