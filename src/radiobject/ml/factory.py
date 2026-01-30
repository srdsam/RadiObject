"""Factory functions for creating training dataloaders."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from torch.utils.data import DataLoader

from radiobject.ml.config import CacheStrategy, DatasetConfig, LoadingMode
from radiobject.ml.datasets.volume_dataset import RadiObjectDataset
from radiobject.ml.utils.worker_init import worker_init_fn

if TYPE_CHECKING:
    from radiobject.radi_object import RadiObject


def create_training_dataloader(
    radi_object: RadiObject,
    modalities: list[str] | None = None,
    label_column: str | None = None,
    batch_size: int = 4,
    patch_size: tuple[int, int, int] | None = None,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    value_filter: str | None = None,
    transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    cache_strategy: CacheStrategy = CacheStrategy.NONE,
    patches_per_volume: int = 1,
) -> DataLoader:
    """Create a DataLoader configured for training.

    Args:
        radi_object: RadiObject to load data from.
        modalities: List of collection names to load. None uses all.
        label_column: Column name in obs_meta for labels.
        batch_size: Samples per batch.
        patch_size: If provided, extract random patches of this size.
        num_workers: DataLoader worker processes.
        pin_memory: Pin tensors to CUDA memory.
        persistent_workers: Keep workers alive between epochs.
        value_filter: TileDB filter for subject selection.
        transform: Transform function applied to each sample.
            MONAI dict transforms (e.g., RandFlipd) work directly.
        cache_strategy: Caching strategy (NONE, IN_MEMORY).
        patches_per_volume: Number of patches to extract per volume per epoch.

    Example with MONAI transforms::

        from monai.transforms import Compose, RandFlipd, NormalizeIntensityd

        transform = Compose([
            NormalizeIntensityd(keys="image"),
            RandFlipd(keys="image", prob=0.5, spatial_axis=[0, 1, 2]),
        ])
        loader = create_training_dataloader(radi, transform=transform)

    Example with TorchIO (use RadiObjectSubjectsDataset instead)::

        from radiobject.ml import RadiObjectSubjectsDataset
        import torchio as tio

        transform = tio.Compose([tio.ZNormalization(), tio.RandomFlip()])
        dataset = RadiObjectSubjectsDataset(radi, modalities=["T1w"], transform=transform)
        loader = DataLoader(dataset, batch_size=4)
    """
    loading_mode = LoadingMode.PATCH if patch_size else LoadingMode.FULL_VOLUME

    config = DatasetConfig(
        loading_mode=loading_mode,
        patch_size=patch_size,
        patches_per_volume=patches_per_volume,
        cache_strategy=cache_strategy,
        modalities=modalities,
        label_column=label_column,
        value_filter=value_filter,
    )

    dataset = RadiObjectDataset(radi_object, config, transform=transform)

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
    radi_object: RadiObject,
    modalities: list[str] | None = None,
    label_column: str | None = None,
    batch_size: int = 4,
    patch_size: tuple[int, int, int] | None = None,
    num_workers: int = 4,
    pin_memory: bool = True,
    value_filter: str | None = None,
    transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> DataLoader:
    """Create a DataLoader configured for validation (no shuffle, no drop_last).

    Args:
        radi_object: RadiObject to load data from.
        modalities: List of collection names to load. None uses all.
        label_column: Column name in obs_meta for labels.
        batch_size: Samples per batch.
        patch_size: If provided, extract patches of this size.
        num_workers: DataLoader worker processes.
        pin_memory: Pin tensors to CUDA memory.
        value_filter: TileDB filter for subject selection.
        transform: Transform function applied to each sample.
            MONAI dict transforms work directly.

    Example::

        from monai.transforms import Compose, NormalizeIntensityd

        transform = Compose([NormalizeIntensityd(keys="image")])
        loader = create_validation_dataloader(radi, transform=transform)
    """
    loading_mode = LoadingMode.PATCH if patch_size else LoadingMode.FULL_VOLUME

    config = DatasetConfig(
        loading_mode=loading_mode,
        patch_size=patch_size,
        patches_per_volume=1,
        modalities=modalities,
        label_column=label_column,
        value_filter=value_filter,
    )

    dataset = RadiObjectDataset(radi_object, config, transform=transform)

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
    radi_object: RadiObject,
    modalities: list[str] | None = None,
    batch_size: int = 1,
    num_workers: int = 4,
    pin_memory: bool = True,
    transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> DataLoader:
    """Create a DataLoader configured for inference (full volumes, no shuffle).

    Args:
        radi_object: RadiObject to load data from.
        modalities: List of collection names to load. None uses all.
        batch_size: Samples per batch.
        num_workers: DataLoader worker processes.
        pin_memory: Pin tensors to CUDA memory.
        transform: Transform function applied to each sample.

    Example::

        from monai.transforms import NormalizeIntensityd

        transform = NormalizeIntensityd(keys="image")
        loader = create_inference_dataloader(radi, transform=transform)
    """
    config = DatasetConfig(
        loading_mode=LoadingMode.FULL_VOLUME,
        modalities=modalities,
    )

    dataset = RadiObjectDataset(radi_object, config, transform=transform)

    effective_workers = num_workers if num_workers > 0 else 0

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=effective_workers,
        pin_memory=pin_memory and effective_workers > 0,
        worker_init_fn=worker_init_fn if effective_workers > 0 else None,
    )
