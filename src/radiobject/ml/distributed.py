"""Distributed training utilities for DDP."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Sequence

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from radiobject._types import LabelSource
from radiobject.ml.config import DatasetConfig, LoadingMode
from radiobject.ml.datasets.collection_dataset import VolumeCollectionDataset
from radiobject.ml.utils.worker_init import worker_init_fn

if TYPE_CHECKING:
    from radiobject.volume_collection import VolumeCollection


def create_distributed_dataloader(
    collections: VolumeCollection | Sequence[VolumeCollection],
    rank: int,
    world_size: int,
    labels: LabelSource = None,
    batch_size: int = 4,
    patch_size: tuple[int, int, int] | None = None,
    num_workers: int = 4,
    pin_memory: bool = True,
    transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> DataLoader:
    """Create a DataLoader for distributed training with DDP.

    Args:
        collections: Single VolumeCollection or list for multi-modal training.
        rank: Current process rank.
        world_size: Total number of processes.
        labels: Label source (see create_training_dataloader for options).
        batch_size: Samples per batch per GPU.
        patch_size: If provided, extract random patches.
        num_workers: DataLoader worker processes.
        pin_memory: Pin tensors to CUDA memory.
        transform: Transform function.

    Returns:
        DataLoader with DistributedSampler configured.
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
