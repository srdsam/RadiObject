"""Factory functions for creating training dataloaders."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Sequence

from torch.utils.data import DataLoader

from radiobject._types import LabelSource
from radiobject.ml.config import DatasetConfig, LoadingMode
from radiobject.ml.datasets.collection_dataset import VolumeCollectionDataset
from radiobject.ml.datasets.segmentation_dataset import SegmentationDataset
from radiobject.ml.utils.worker_init import worker_init_fn

if TYPE_CHECKING:
    from radiobject.volume_collection import VolumeCollection


def create_training_dataloader(
    collections: VolumeCollection | Sequence[VolumeCollection],
    labels: LabelSource = None,
    batch_size: int = 4,
    patch_size: tuple[int, int, int] | None = None,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    patches_per_volume: int = 1,
) -> DataLoader:
    """Create a DataLoader configured for training from VolumeCollection(s).

    Args:
        collections: Single VolumeCollection or list for multi-modal training.
            Multi-modal collections are stacked along channel dimension.
        labels: Label source. Can be:
            - str: Column name in collection's obs DataFrame
            - pd.DataFrame: With obs_id as column/index and label values
            - dict[str, Any]: Mapping from obs_id to label
            - Callable[[str], Any]: Function taking obs_id, returning label
            - None: No labels
        batch_size: Samples per batch.
        patch_size: If provided, extract random patches of this size.
        num_workers: DataLoader worker processes.
        pin_memory: Pin tensors to CUDA memory.
        persistent_workers: Keep workers alive between epochs.
        transform: Transform function applied to each sample.
            MONAI dict transforms (e.g., RandFlipd) work directly.
        patches_per_volume: Number of patches to extract per volume per epoch.

    Returns:
        DataLoader configured for training with shuffle enabled.
    """
    loading_mode = LoadingMode.PATCH if patch_size else LoadingMode.FULL_VOLUME

    config = DatasetConfig(
        loading_mode=loading_mode,
        patch_size=patch_size,
        patches_per_volume=patches_per_volume,
    )

    dataset = VolumeCollectionDataset(
        collections, config=config, labels=labels, transform=transform
    )

    effective_workers = num_workers if num_workers > 0 else 0
    effective_persistent = persistent_workers and effective_workers > 0

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=effective_workers,
        pin_memory=pin_memory and effective_workers > 0,
        persistent_workers=effective_persistent,
        worker_init_fn=worker_init_fn if effective_workers > 0 else None,
        drop_last=True,
    )


def create_validation_dataloader(
    collections: VolumeCollection | Sequence[VolumeCollection],
    labels: LabelSource = None,
    batch_size: int = 4,
    patch_size: tuple[int, int, int] | None = None,
    num_workers: int = 4,
    pin_memory: bool = True,
    transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> DataLoader:
    """Create a DataLoader configured for validation (no shuffle, no drop_last).

    Args:
        collections: Single VolumeCollection or list for multi-modal validation.
        labels: Label source (see create_training_dataloader for options).
        batch_size: Samples per batch.
        patch_size: If provided, extract patches of this size.
        num_workers: DataLoader worker processes.
        pin_memory: Pin tensors to CUDA memory.
        transform: Transform function applied to each sample.
            MONAI dict transforms work directly.

    Returns:
        DataLoader configured for validation.

    Example::

        from monai.transforms import Compose, NormalizeIntensityd

        transform = Compose([NormalizeIntensityd(keys="image")])
        loader = create_validation_dataloader(radi.CT, labels="has_tumor", transform=transform)
    """
    loading_mode = LoadingMode.PATCH if patch_size else LoadingMode.FULL_VOLUME

    config = DatasetConfig(
        loading_mode=loading_mode,
        patch_size=patch_size,
        patches_per_volume=1,
    )

    dataset = VolumeCollectionDataset(
        collections, config=config, labels=labels, transform=transform
    )

    effective_workers = num_workers if num_workers > 0 else 0

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=effective_workers,
        pin_memory=pin_memory and effective_workers > 0,
        worker_init_fn=worker_init_fn if effective_workers > 0 else None,
        drop_last=False,
    )


def create_inference_dataloader(
    collections: VolumeCollection | Sequence[VolumeCollection],
    batch_size: int = 1,
    num_workers: int = 4,
    pin_memory: bool = True,
    transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> DataLoader:
    """Create a DataLoader configured for inference (full volumes, no shuffle).

    Args:
        collections: Single VolumeCollection or list for multi-modal inference.
        batch_size: Samples per batch.
        num_workers: DataLoader worker processes.
        pin_memory: Pin tensors to CUDA memory.
        transform: Transform function applied to each sample.

    Returns:
        DataLoader configured for inference.

    Example::

        from monai.transforms import NormalizeIntensityd

        transform = NormalizeIntensityd(keys="image")
        loader = create_inference_dataloader(radi.CT, transform=transform)
    """
    config = DatasetConfig(loading_mode=LoadingMode.FULL_VOLUME)

    dataset = VolumeCollectionDataset(collections, config=config, transform=transform)

    effective_workers = num_workers if num_workers > 0 else 0

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=effective_workers,
        pin_memory=pin_memory and effective_workers > 0,
        worker_init_fn=worker_init_fn if effective_workers > 0 else None,
    )


def create_segmentation_dataloader(
    image: VolumeCollection,
    mask: VolumeCollection,
    batch_size: int = 4,
    patch_size: tuple[int, int, int] | None = None,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    image_transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    spatial_transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    foreground_sampling: bool = False,
    foreground_threshold: float = 0.01,
    foreground_max_retries: int = 10,
    patches_per_volume: int = 1,
) -> DataLoader:
    """Create a DataLoader for segmentation training with separate image/mask handling.

    Unlike create_training_dataloader which stacks collections as channels, this
    returns separate "image" and "mask" tensors. This is cleaner for segmentation
    workflows where different transforms need to be applied to images vs masks.

    Args:
        image: VolumeCollection containing input images (CT, MRI, etc.).
        mask: VolumeCollection containing segmentation masks.
        batch_size: Samples per batch.
        patch_size: If provided, extract random patches of this size.
        num_workers: DataLoader worker processes.
        pin_memory: Pin tensors to CUDA memory.
        persistent_workers: Keep workers alive between epochs.
        image_transform: Transform applied only to "image" key (e.g., normalization).
        spatial_transform: Transform applied to both "image" and "mask" keys
            (e.g., random flips, rotations).
        foreground_sampling: If True, bias patch sampling toward regions with
            foreground (non-zero mask values).
        foreground_threshold: Minimum fraction of foreground voxels in patch
            when foreground_sampling is enabled.
        foreground_max_retries: Maximum random attempts before accepting any patch.
        patches_per_volume: Number of patches to extract per volume per epoch.

    Returns:
        DataLoader yielding {"image": (B,1,X,Y,Z), "mask": (B,1,X,Y,Z), ...}
    """
    loading_mode = LoadingMode.PATCH if patch_size else LoadingMode.FULL_VOLUME

    config = DatasetConfig(
        loading_mode=loading_mode,
        patch_size=patch_size,
        patches_per_volume=patches_per_volume,
    )

    dataset = SegmentationDataset(
        image=image,
        mask=mask,
        config=config,
        image_transform=image_transform,
        spatial_transform=spatial_transform,
        foreground_sampling=foreground_sampling,
        foreground_threshold=foreground_threshold,
        foreground_max_retries=foreground_max_retries,
    )

    effective_workers = num_workers if num_workers > 0 else 0
    effective_persistent = persistent_workers and effective_workers > 0

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=effective_workers,
        pin_memory=pin_memory and effective_workers > 0,
        persistent_workers=effective_persistent,
        worker_init_fn=worker_init_fn if effective_workers > 0 else None,
        drop_last=True,
    )
