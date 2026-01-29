"""Core RadiObjectDataset implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import torch
from torch.utils.data import Dataset

from ml.cache import BaseCache, InMemoryCache, NoOpCache
from ml.config import CacheStrategy, DatasetConfig, LoadingMode
from ml.reader import VolumeReader

if TYPE_CHECKING:
    from src.radi_object import RadiObject


class RadiObjectDataset(Dataset):
    """PyTorch Dataset for loading volumes from RadiObject via TileDB."""

    def __init__(
        self,
        radi_object: RadiObject,
        config: DatasetConfig,
        transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ):
        self._config = config
        self._transform = transform

        # Initialize cache based on strategy
        if config.cache_strategy == CacheStrategy.IN_MEMORY:
            self._cache: BaseCache = InMemoryCache()
        else:
            self._cache = NoOpCache()

        modalities = config.modalities or list(radi_object.collection_names)
        if not modalities:
            raise ValueError("No modalities specified and RadiObject has no collections")

        self._modalities = modalities
        self._readers = {
            mod: VolumeReader(radi_object.collection(mod), ctx=radi_object._ctx)
            for mod in modalities
        }

        first_reader = self._readers[modalities[0]]
        self._n_volumes = len(first_reader)
        self._volume_shape = first_reader.shape

        self._labels: dict[int, Any] | None = None
        if config.label_column:
            self._load_labels(radi_object, config.label_column, config.value_filter)

        if config.loading_mode == LoadingMode.PATCH:
            self._length = self._n_volumes * config.patches_per_volume
        elif config.loading_mode == LoadingMode.SLICE_2D:
            self._length = self._n_volumes * self._volume_shape[2]
        else:
            self._length = self._n_volumes

    def _load_labels(
        self,
        radi_object: RadiObject,
        label_column: str,
        value_filter: str | None,
    ) -> None:
        """Load labels from obs_meta dataframe."""
        obs_meta = radi_object.obs_meta.read(value_filter=value_filter)
        if label_column not in obs_meta.columns:
            raise ValueError(f"Label column '{label_column}' not found in obs_meta")

        first_reader = self._readers[self._modalities[0]]
        self._labels = {}
        for idx in range(self._n_volumes):
            obs_id = first_reader.get_obs_id(idx)
            parts = obs_id.rsplit("_", 1)
            subject_id = parts[0] if len(parts) > 1 else obs_id
            match = obs_meta[obs_meta["obs_subject_id"] == subject_id]
            if len(match) > 0:
                self._labels[idx] = match[label_column].iloc[0]

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # Check cache first
        cached = self._cache.get(idx)
        if cached is not None:
            return cached

        if self._config.loading_mode == LoadingMode.PATCH:
            result = self._get_patch_item(idx)
        elif self._config.loading_mode == LoadingMode.SLICE_2D:
            result = self._get_slice_item(idx)
        else:
            result = self._get_full_volume_item(idx)

        # Store in cache
        self._cache.set(idx, result)
        return result

    def _get_full_volume_item(self, idx: int) -> dict[str, torch.Tensor]:
        """Load full volume for all modalities."""
        volumes = [self._readers[mod].read_full(idx) for mod in self._modalities]

        stacked = np.stack(volumes, axis=0)
        result: dict[str, Any] = {
            "image": torch.from_numpy(stacked),
            "idx": idx,
        }

        if self._labels is not None and idx in self._labels:
            label = self._labels[idx]
            if isinstance(label, (int, float, np.integer, np.floating)):
                result["label"] = torch.tensor(label)
            else:
                result["label"] = label

        if self._transform is not None:
            result = self._transform(result)

        return result

    def _get_patch_item(self, idx: int) -> dict[str, torch.Tensor]:
        """Load a random patch from the volume."""
        volume_idx = idx // self._config.patches_per_volume
        patch_idx = idx % self._config.patches_per_volume

        rng = np.random.default_rng(seed=idx)
        patch_size = self._config.patch_size
        assert patch_size is not None

        max_start = tuple(
            max(0, self._volume_shape[i] - patch_size[i])
            for i in range(3)
        )
        start = tuple(
            rng.integers(0, max_start[i] + 1) if max_start[i] > 0 else 0
            for i in range(3)
        )

        volumes = [
            self._readers[mod].read_patch(volume_idx, start, patch_size)
            for mod in self._modalities
        ]

        stacked = np.stack(volumes, axis=0)
        result: dict[str, Any] = {
            "image": torch.from_numpy(stacked),
            "idx": volume_idx,
            "patch_idx": patch_idx,
            "patch_start": start,
        }

        if self._labels is not None and volume_idx in self._labels:
            label = self._labels[volume_idx]
            if isinstance(label, (int, float, np.integer, np.floating)):
                result["label"] = torch.tensor(label)
            else:
                result["label"] = label

        if self._transform is not None:
            result = self._transform(result)

        return result

    def _get_slice_item(self, idx: int) -> dict[str, torch.Tensor]:
        """Load a 2D slice from the volume."""
        volume_idx = idx // self._volume_shape[2]
        slice_idx = idx % self._volume_shape[2]

        slices = [
            self._readers[mod].read_slice(volume_idx, axis=2, position=slice_idx)
            for mod in self._modalities
        ]

        stacked = np.stack(slices, axis=0)
        result: dict[str, Any] = {
            "image": torch.from_numpy(stacked),
            "idx": volume_idx,
            "slice_idx": slice_idx,
        }

        if self._labels is not None and volume_idx in self._labels:
            label = self._labels[volume_idx]
            if isinstance(label, (int, float, np.integer, np.floating)):
                result["label"] = torch.tensor(label)
            else:
                result["label"] = label

        if self._transform is not None:
            result = self._transform(result)

        return result

    @property
    def modalities(self) -> list[str]:
        """List of modalities being loaded."""
        return self._modalities

    @property
    def volume_shape(self) -> tuple[int, int, int]:
        """Shape of each volume (X, Y, Z)."""
        return self._volume_shape

    @property
    def cache(self) -> BaseCache:
        """Access to the sample cache."""
        return self._cache
