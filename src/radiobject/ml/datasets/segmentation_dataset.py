"""SegmentationDataset - specialized dataset for image/mask segmentation training."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch.utils.data import Dataset

from radiobject.ml.config import DatasetConfig, LoadingMode
from radiobject.ml.utils.validation import validate_collection_alignment, validate_uniform_shapes

if TYPE_CHECKING:
    from radiobject.volume_collection import VolumeCollection


class SegmentationDataset(Dataset):
    """PyTorch Dataset for segmentation training with explicit image/mask separation."""

    def __init__(
        self,
        image: VolumeCollection,
        mask: VolumeCollection,
        config: DatasetConfig | None = None,
        image_transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        spatial_transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        foreground_sampling: bool = False,
        foreground_threshold: float = 0.01,
        foreground_max_retries: int = 10,
    ):
        """Initialize segmentation dataset.

        Args:
            image: VolumeCollection containing input images (CT, MRI, etc.).
            mask: VolumeCollection containing segmentation masks.
            config: Dataset configuration (loading mode, patch size, etc.).
            image_transform: Transform applied to image only (e.g., normalization).
                Should operate on keys=["image"].
            spatial_transform: Transform applied to both image and mask (e.g., flips).
                Should operate on keys=["image", "mask"].
            foreground_sampling: If True, bias patch sampling toward regions with
                foreground (non-zero mask values).
            foreground_threshold: Minimum fraction of foreground voxels in patch
                when foreground_sampling is enabled.
            foreground_max_retries: Maximum random attempts before accepting any patch.
        """
        self._config = config or DatasetConfig()
        self._image_transform = image_transform
        self._spatial_transform = spatial_transform
        self._foreground_sampling = foreground_sampling
        self._foreground_threshold = foreground_threshold
        self._foreground_max_retries = foreground_max_retries

        # Store collections directly
        self._image = image
        self._mask = mask

        # Cache obs_ids and obs_subject_ids for fast access
        self._obs_ids = image.obs_ids
        self._obs_subject_ids = image.obs_subject_ids

        # Validate alignment between image and mask collections
        collections = {"image": self._image, "mask": self._mask}
        validate_collection_alignment(collections)

        # Validate uniform shapes
        self._volume_shape = validate_uniform_shapes(collections)
        self._n_volumes = len(self._image)

        # Compute dataset length
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
        """Load full volume for image and mask."""
        image_data = self._image.iloc[idx].to_numpy()
        mask_data = self._mask.iloc[idx].to_numpy()

        result: dict[str, Any] = {
            "image": torch.from_numpy(image_data).unsqueeze(0),
            "mask": torch.from_numpy(mask_data).unsqueeze(0),
            "idx": idx,
            "obs_id": self._obs_ids[idx],
            "obs_subject_id": self._obs_subject_ids[idx],
        }

        return self._apply_transforms(result)

    def _get_patch_item(self, idx: int) -> dict[str, Any]:
        """Load a random patch from image and mask."""
        volume_idx = idx // self._config.patches_per_volume
        patch_idx = idx % self._config.patches_per_volume

        patch_size = self._config.patch_size
        assert patch_size is not None

        max_start = tuple(max(0, self._volume_shape[i] - patch_size[i]) for i in range(3))

        if self._foreground_sampling:
            # Try to find a patch with sufficient foreground
            start = self._sample_foreground_patch(volume_idx, max_start, patch_size, idx)
        else:
            rng = np.random.default_rng(seed=idx)
            start = tuple(
                rng.integers(0, max_start[i] + 1) if max_start[i] > 0 else 0 for i in range(3)
            )

        image_vol = self._image.iloc[volume_idx]
        mask_vol = self._mask.iloc[volume_idx]

        image_data = image_vol.slice(
            slice(start[0], start[0] + patch_size[0]),
            slice(start[1], start[1] + patch_size[1]),
            slice(start[2], start[2] + patch_size[2]),
        )
        mask_data = mask_vol.slice(
            slice(start[0], start[0] + patch_size[0]),
            slice(start[1], start[1] + patch_size[1]),
            slice(start[2], start[2] + patch_size[2]),
        )

        result: dict[str, Any] = {
            "image": torch.from_numpy(image_data).unsqueeze(0),
            "mask": torch.from_numpy(mask_data).unsqueeze(0),
            "idx": volume_idx,
            "patch_idx": patch_idx,
            "patch_start": start,
            "obs_id": self._obs_ids[volume_idx],
            "obs_subject_id": self._obs_subject_ids[volume_idx],
        }

        return self._apply_transforms(result)

    def _sample_foreground_patch(
        self,
        volume_idx: int,
        max_start: tuple[int, ...],
        patch_size: tuple[int, int, int],
        seed: int,
    ) -> tuple[int, int, int]:
        """Sample a patch position biased toward foreground regions."""
        rng = np.random.default_rng(seed=seed)
        mask_vol = self._mask.iloc[volume_idx]

        for attempt in range(self._foreground_max_retries):
            start = tuple(
                rng.integers(0, max_start[i] + 1) if max_start[i] > 0 else 0 for i in range(3)
            )

            mask_patch = mask_vol.slice(
                slice(start[0], start[0] + patch_size[0]),
                slice(start[1], start[1] + patch_size[1]),
                slice(start[2], start[2] + patch_size[2]),
            )
            foreground_ratio = np.count_nonzero(mask_patch) / mask_patch.size

            if foreground_ratio >= self._foreground_threshold:
                return start  # type: ignore[return-value]

        # Fallback: return last sampled position
        return start  # type: ignore[return-value]

    def _get_slice_item(self, idx: int) -> dict[str, Any]:
        """Load a 2D slice from image and mask."""
        volume_idx = idx // self._volume_shape[2]
        slice_idx = idx % self._volume_shape[2]

        image_data = self._image.iloc[volume_idx].axial(slice_idx)
        mask_data = self._mask.iloc[volume_idx].axial(slice_idx)

        result: dict[str, Any] = {
            "image": torch.from_numpy(image_data).unsqueeze(0),
            "mask": torch.from_numpy(mask_data).unsqueeze(0),
            "idx": volume_idx,
            "slice_idx": slice_idx,
            "obs_id": self._obs_ids[volume_idx],
            "obs_subject_id": self._obs_subject_ids[volume_idx],
        }

        return self._apply_transforms(result)

    def _apply_transforms(self, result: dict[str, Any]) -> dict[str, Any]:
        """Apply spatial and image transforms to sample dict."""
        # Spatial transform affects both image and mask
        if self._spatial_transform is not None:
            result = self._spatial_transform(result)

        # Image transform affects only image
        if self._image_transform is not None:
            result = self._image_transform(result)

        return result

    @property
    def volume_shape(self) -> tuple[int, int, int]:
        """Shape of each volume (X, Y, Z)."""
        return self._volume_shape

    @property
    def n_volumes(self) -> int:
        """Number of image/mask pairs."""
        return self._n_volumes
