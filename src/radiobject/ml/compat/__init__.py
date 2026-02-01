"""MONAI/TorchIO compatibility module."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from radiobject.ml.compat.torchio import VolumeCollectionSubjectsDataset

try:
    from monai.transforms import Compose
except ImportError:
    try:
        from torchio import Compose
    except ImportError:

        class Compose:
            """Minimal Compose fallback when MONAI/TorchIO unavailable."""

            def __init__(self, transforms: Sequence[Callable[[Any], Any]]):
                self.transforms = list(transforms)

            def __call__(self, data: Any) -> Any:
                for t in self.transforms:
                    data = t(data)
                return data

            def __repr__(self) -> str:
                names = [t.__class__.__name__ for t in self.transforms]
                return f"Compose({names})"


__all__ = ["VolumeCollectionSubjectsDataset", "Compose"]
