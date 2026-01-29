"""Intensity transforms for medical imaging."""

from __future__ import annotations

from typing import Any

import torch


class IntensityNormalize:
    """Normalize intensity to zero mean and unit variance."""

    def __init__(self, channel_wise: bool = True):
        self.channel_wise = channel_wise

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        image = data["image"].float()
        if self.channel_wise:
            for c in range(image.shape[0]):
                channel = image[c]
                mean = channel.mean()
                std = channel.std()
                if std > 0:
                    image[c] = (channel - mean) / std
        else:
            mean = image.mean()
            std = image.std()
            if std > 0:
                image = (image - mean) / std
        data["image"] = image
        return data


class WindowLevel:
    """Apply window/level (contrast) adjustment for CT data."""

    def __init__(self, window: float, level: float):
        self.window = window
        self.level = level
        self.min_val = level - window / 2
        self.max_val = level + window / 2

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        image = data["image"].float()
        image = torch.clamp(image, self.min_val, self.max_val)
        image = (image - self.min_val) / (self.max_val - self.min_val)
        data["image"] = image
        return data


class RandomNoise:
    """Add random Gaussian noise."""

    def __init__(self, std: float = 0.1, prob: float = 0.5):
        self.std = std
        self.prob = prob

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        if torch.rand(1).item() < self.prob:
            image = data["image"]
            noise = torch.randn_like(image) * self.std
            data["image"] = image + noise
        return data
