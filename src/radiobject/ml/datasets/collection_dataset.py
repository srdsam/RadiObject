"""VolumeCollectionDataset - primary PyTorch Dataset for VolumeCollection(s)."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from radiobject._types import LabelSource
from radiobject.ml.config import DatasetConfig, LoadingMode
from radiobject.ml.utils.labels import load_labels
from radiobject.ml.utils.validation import validate_collection_alignment, validate_uniform_shapes

if TYPE_CHECKING:
    from radiobject.volume_collection import VolumeCollection


class VolumeCollectionDataset(Dataset):
    """PyTorch Dataset for VolumeCollection(s) - primary ML interface."""

    def __init__(
        self,
        collections: VolumeCollection | Sequence[VolumeCollection],
        config: DatasetConfig | None = None,
        labels: LabelSource = None,
        transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ):
        """Initialize dataset from VolumeCollection(s).

        Args:
            collections: Single VolumeCollection or sequence of collections.
                Multiple collections are stacked along channel dimension.
            config: Dataset configuration (loading mode, patch size, etc.).
                If None, uses full volume mode.
            labels: Label source. Can be:
                - str: Column name in collection's obs DataFrame
                - pd.DataFrame: With obs_id as column/index and label values
                - dict[str, Any]: Mapping from obs_id to label
                - Callable[[str], Any]: Function taking obs_id, returning label
                - None: No labels
            transform: Transform function applied to each sample dict.
                MONAI dict transforms (e.g., RandFlipd) work directly.
        """
        self._config = config or DatasetConfig()
        self._transform = transform

        # Normalize to list
        if not isinstance(collections, Sequence):
            collections = [collections]
        if not collections:
            raise ValueError("At least one collection required")

        self._collections = list(collections)
        self._collection_names = [c.name or f"collection_{i}" for i, c in enumerate(collections)]

        # Build dict for validation
        collections_dict: dict[str, VolumeCollection] = {}
        for name, coll in zip(self._collection_names, self._collections):
            collections_dict[name] = coll

        # Validate alignment if multi-modal
        if len(collections_dict) > 1:
            validate_collection_alignment(collections_dict)

        # Validate uniform shapes (required for full volume and slice modes, not patch mode)
        if self._config.loading_mode == LoadingMode.PATCH:
            self._volume_shape: tuple[int, int, int] | None = None
        else:
            self._volume_shape = validate_uniform_shapes(collections_dict)

        first_coll = self._collections[0]
        self._n_volumes = len(first_coll)

        # Load labels from first collection's obs
        self._labels: dict[int, Any] | None = None
        if labels is not None:
            obs_df = first_coll.obs.read() if isinstance(labels, str) else None
            self._labels = load_labels(first_coll, labels, obs_df)

        # Compute dataset length based on loading mode
        if self._config.loading_mode == LoadingMode.PATCH:
            self._length = self._n_volumes * self._config.patches_per_volume
        elif self._config.loading_mode == LoadingMode.SLICE_2D:
            self._length = self._n_volumes * self._volume_shape[2]
        else:
            self._length = self._n_volumes

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if self._config.loading_mode == LoadingMode.PATCH:
            return self._get_patch_item(idx)
        elif self._config.loading_mode == LoadingMode.SLICE_2D:
            return self._get_slice_item(idx)
        else:
            return self._get_full_volume_item(idx)

    def _get_full_volume_item(self, idx: int) -> dict[str, Any]:
        """Load full volume for all collections."""
        volumes = [coll.iloc[idx].to_numpy() for coll in self._collections]

        stacked = np.stack(volumes, axis=0)
        result: dict[str, Any] = {
            "image": torch.from_numpy(stacked),
            "idx": idx,
        }

        self._add_label(result, idx)

        if self._transform is not None:
            result = self._transform(result)

        return result

    def _get_patch_item(self, idx: int) -> dict[str, Any]:
        """Load a random patch from the volume."""
        volume_idx = idx // self._config.patches_per_volume
        patch_idx = idx % self._config.patches_per_volume

        rng = np.random.default_rng(seed=idx)
        patch_size = self._config.patch_size
        assert patch_size is not None

        # Use per-volume shape (supports heterogeneous collections)
        vol_shape = self._collections[0].iloc[volume_idx].shape

        max_start = tuple(max(0, vol_shape[i] - patch_size[i]) for i in range(3))
        start = tuple(
            rng.integers(0, max_start[i] + 1) if max_start[i] > 0 else 0 for i in range(3)
        )

        volumes = []
        for coll in self._collections:
            vol = coll.iloc[volume_idx]
            patch = vol.slice(
                slice(start[0], start[0] + patch_size[0]),
                slice(start[1], start[1] + patch_size[1]),
                slice(start[2], start[2] + patch_size[2]),
            )
            volumes.append(patch)

        stacked = np.stack(volumes, axis=0)
        result: dict[str, Any] = {
            "image": torch.from_numpy(stacked),
            "idx": volume_idx,
            "patch_idx": patch_idx,
            "patch_start": start,
        }

        self._add_label(result, volume_idx)

        if self._transform is not None:
            result = self._transform(result)

        return result

    def _get_slice_item(self, idx: int) -> dict[str, Any]:
        """Load a 2D slice from the volume."""
        volume_idx = idx // self._volume_shape[2]
        slice_idx = idx % self._volume_shape[2]

        slices = [coll.iloc[volume_idx].axial(slice_idx) for coll in self._collections]

        stacked = np.stack(slices, axis=0)
        result: dict[str, Any] = {
            "image": torch.from_numpy(stacked),
            "idx": volume_idx,
            "slice_idx": slice_idx,
        }

        self._add_label(result, volume_idx)

        if self._transform is not None:
            result = self._transform(result)

        return result

    def _add_label(self, result: dict[str, Any], volume_idx: int) -> None:
        """Add label column to sample dict from label source."""
        if self._labels is not None and volume_idx in self._labels:
            label = self._labels[volume_idx]
            if isinstance(label, (int, float, np.integer, np.floating)):
                result["label"] = torch.tensor(label)
            else:
                result["label"] = label

    @property
    def collection_names(self) -> list[str]:
        """Names of collections being loaded (channel order)."""
        return self._collection_names

    @property
    def volume_shape(self) -> tuple[int, int, int]:
        """Shape of each volume (X, Y, Z)."""
        return self._volume_shape

    @property
    def n_channels(self) -> int:
        """Number of channels (collections) in output tensors."""
        return len(self._collections)
