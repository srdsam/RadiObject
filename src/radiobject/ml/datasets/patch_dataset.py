"""Specialized patch extraction dataset."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import torch
from torch.utils.data import Dataset

from radiobject.ml.reader import VolumeReader

if TYPE_CHECKING:
    from radiobject.volume_collection import VolumeCollection


class PatchVolumeDataset(Dataset):
    """Dataset for extracting patches from a single VolumeCollection."""

    def __init__(
        self,
        collection: VolumeCollection,
        patch_size: tuple[int, int, int],
        patches_per_volume: int = 1,
        transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ):
        self._reader = VolumeReader(collection)
        self._patch_size = patch_size
        self._patches_per_volume = patches_per_volume
        self._transform = transform

        self._n_volumes = len(self._reader)
        self._volume_shape = self._reader.shape
        self._length = self._n_volumes * patches_per_volume

        for i, dim in enumerate(patch_size):
            if dim > self._volume_shape[i]:
                raise ValueError(
                    f"Patch dimension {i} ({dim}) exceeds volume dimension ({self._volume_shape[i]})"
                )

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        volume_idx = idx // self._patches_per_volume
        patch_idx = idx % self._patches_per_volume

        rng = np.random.default_rng(seed=idx)

        max_start = tuple(
            max(0, self._volume_shape[i] - self._patch_size[i])
            for i in range(3)
        )
        start = tuple(
            rng.integers(0, max_start[i] + 1) if max_start[i] > 0 else 0
            for i in range(3)
        )

        data = self._reader.read_patch(volume_idx, start, self._patch_size)

        result: dict[str, Any] = {
            "image": torch.from_numpy(data).unsqueeze(0),
            "idx": volume_idx,
            "patch_idx": patch_idx,
            "patch_start": start,
            "obs_id": self._reader.get_obs_id(volume_idx),
        }

        if self._transform is not None:
            result = self._transform(result)

        return result

    @property
    def volume_shape(self) -> tuple[int, int, int]:
        """Shape of each volume."""
        return self._volume_shape

    @property
    def patch_size(self) -> tuple[int, int, int]:
        """Patch dimensions."""
        return self._patch_size


class GridPatchDataset(Dataset):
    """Dataset for extracting patches on a regular grid (for inference)."""

    def __init__(
        self,
        collection: VolumeCollection,
        patch_size: tuple[int, int, int],
        stride: tuple[int, int, int] | None = None,
        transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ):
        self._reader = VolumeReader(collection)
        self._patch_size = patch_size
        self._stride = stride or patch_size
        self._transform = transform

        self._n_volumes = len(self._reader)
        self._volume_shape = self._reader.shape

        self._grid_positions = self._compute_grid_positions()
        self._patches_per_volume = len(self._grid_positions)
        self._length = self._n_volumes * self._patches_per_volume

    def _compute_grid_positions(self) -> list[tuple[int, int, int]]:
        """Compute all valid patch start positions."""
        positions = []
        for x in range(0, self._volume_shape[0] - self._patch_size[0] + 1, self._stride[0]):
            for y in range(0, self._volume_shape[1] - self._patch_size[1] + 1, self._stride[1]):
                for z in range(0, self._volume_shape[2] - self._patch_size[2] + 1, self._stride[2]):
                    positions.append((x, y, z))
        if not positions:
            positions.append((0, 0, 0))
        return positions

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        volume_idx = idx // self._patches_per_volume
        patch_idx = idx % self._patches_per_volume
        start = self._grid_positions[patch_idx]

        data = self._reader.read_patch(volume_idx, start, self._patch_size)

        result: dict[str, Any] = {
            "image": torch.from_numpy(data).unsqueeze(0),
            "idx": volume_idx,
            "patch_idx": patch_idx,
            "patch_start": start,
            "obs_id": self._reader.get_obs_id(volume_idx),
        }

        if self._transform is not None:
            result = self._transform(result)

        return result

    @property
    def grid_positions(self) -> list[tuple[int, int, int]]:
        """All patch start positions in the grid."""
        return self._grid_positions
