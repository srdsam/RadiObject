"""Distributed training utilities for DDP."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from radiobject.ml.config import CacheStrategy, DatasetConfig, LoadingMode
from radiobject.ml.datasets.volume_dataset import RadiObjectDataset
from radiobject.ml.utils.worker_init import worker_init_fn

if TYPE_CHECKING:
    from radiobject.radi_object import RadiObject


def create_distributed_dataloader(
    radi_object: RadiObject,
    rank: int,
    world_size: int,
    modalities: list[str] | None = None,
    label_column: str | None = None,
    batch_size: int = 4,
    patch_size: tuple[int, int, int] | None = None,
    num_workers: int = 4,
    pin_memory: bool = True,
    cache_strategy: CacheStrategy = CacheStrategy.NONE,
    value_filter: str | None = None,
    transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> DataLoader:
    """Create a DataLoader for distributed training with DDP.

    Args:
        radi_object: RadiObject to load data from.
        rank: Current process rank.
        world_size: Total number of processes.
        modalities: List of collection names to load.
        label_column: Column name in obs_meta for labels.
        batch_size: Samples per batch per GPU.
        patch_size: If provided, extract random patches.
        num_workers: DataLoader worker processes.
        pin_memory: Pin tensors to CUDA memory.
        cache_strategy: Caching strategy.
        value_filter: TileDB filter for subject selection.
        transform: Transform function.
    """
    loading_mode = LoadingMode.PATCH if patch_size else LoadingMode.FULL_VOLUME

    config = DatasetConfig(
        loading_mode=loading_mode,
        patch_size=patch_size,
        patches_per_volume=1,
        cache_strategy=cache_strategy,
        modalities=modalities,
        label_column=label_column,
        value_filter=value_filter,
    )

    dataset = RadiObjectDataset(radi_object, config, transform=transform)

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )

    effective_workers = num_workers if num_workers > 0 else 0

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=effective_workers,
        pin_memory=pin_memory and effective_workers > 0,
        worker_init_fn=worker_init_fn if effective_workers > 0 else None,
        drop_last=True,
    )


def set_epoch(dataloader: DataLoader, epoch: int) -> None:
    """Set epoch for DistributedSampler to ensure proper shuffling."""
    if hasattr(dataloader, "sampler") and isinstance(dataloader.sampler, DistributedSampler):
        dataloader.sampler.set_epoch(epoch)
