"""Spatial transforms for 3D volumes."""

from __future__ import annotations

from typing import Any

import torch


class RandomFlip3D:
    """Random flip along specified axes."""

    def __init__(self, axes: tuple[int, ...] = (0, 1, 2), prob: float = 0.5):
        self.axes = axes
        self.prob = prob

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        image = data["image"]
        for axis in self.axes:
            if torch.rand(1).item() < self.prob:
                spatial_axis = axis + 1
                image = torch.flip(image, dims=[spatial_axis])
        data["image"] = image
        return data


class RandomCrop3D:
    """Random crop to specified size."""

    def __init__(self, size: tuple[int, int, int]):
        self.size = size

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        image = data["image"]
        shape = image.shape[1:]

        max_start = tuple(max(0, shape[i] - self.size[i]) for i in range(3))
        start = tuple(
            int(torch.randint(0, max_start[i] + 1, (1,)).item()) if max_start[i] > 0 else 0
            for i in range(3)
        )

        data["image"] = image[
            :,
            start[0] : start[0] + self.size[0],
            start[1] : start[1] + self.size[1],
            start[2] : start[2] + self.size[2],
        ]
        data["crop_start"] = start
        return data


class Resample3D:
    """Resample to target shape using interpolation."""

    def __init__(self, target_shape: tuple[int, int, int], mode: str = "trilinear"):
        self.target_shape = target_shape
        self.mode = mode

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        image = data["image"]
        image = image.unsqueeze(0).float()
        resampled = torch.nn.functional.interpolate(
            image,
            size=self.target_shape,
            mode=self.mode,
            align_corners=False if self.mode != "nearest" else None,
        )
        data["image"] = resampled.squeeze(0)
        return data
