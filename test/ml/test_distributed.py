"""Tests for ml/distributed.py - DDP utilities for distributed training."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from radiobject.ml.config import CacheStrategy
from radiobject.ml.distributed import create_distributed_dataloader, set_epoch

if TYPE_CHECKING:
    from radiobject.radi_object import RadiObject


class TestCreateDistributedDataloader:
    """Tests for create_distributed_dataloader function."""

    def test_returns_dataloader(
        self, populated_radi_object_module: "RadiObject"
    ) -> None:
        loader = create_distributed_dataloader(
            populated_radi_object_module,
            rank=0,
            world_size=1,
            modalities=["flair"],
            num_workers=0,
        )

        assert isinstance(loader, DataLoader)

    def test_uses_distributed_sampler(
        self, populated_radi_object_module: "RadiObject"
    ) -> None:
        loader = create_distributed_dataloader(
            populated_radi_object_module,
            rank=0,
            world_size=2,
            modalities=["flair"],
            num_workers=0,
        )

        assert isinstance(loader.sampler, DistributedSampler)

    def test_sampler_configured_correctly(
        self, populated_radi_object_module: "RadiObject"
    ) -> None:
        loader = create_distributed_dataloader(
            populated_radi_object_module,
            rank=1,
            world_size=4,
            modalities=["flair"],
            num_workers=0,
        )

        sampler = loader.sampler
        assert isinstance(sampler, DistributedSampler)
        assert sampler.num_replicas == 4
        assert sampler.rank == 1
        assert sampler.shuffle is True

    def test_batch_size_respected(
        self, populated_radi_object_module: "RadiObject"
    ) -> None:
        loader = create_distributed_dataloader(
            populated_radi_object_module,
            rank=0,
            world_size=1,
            modalities=["flair"],
            batch_size=2,
            num_workers=0,
        )

        assert loader.batch_size == 2

    def test_drop_last_enabled(
        self, populated_radi_object_module: "RadiObject"
    ) -> None:
        loader = create_distributed_dataloader(
            populated_radi_object_module,
            rank=0,
            world_size=1,
            modalities=["flair"],
            num_workers=0,
        )

        assert loader.drop_last is True

    def test_full_volume_mode_default(
        self, populated_radi_object_module: "RadiObject"
    ) -> None:
        loader = create_distributed_dataloader(
            populated_radi_object_module,
            rank=0,
            world_size=1,
            modalities=["flair"],
            batch_size=1,
            num_workers=0,
        )

        batch = next(iter(loader))
        assert batch["image"].shape[-3:] == (240, 240, 155)

    def test_patch_mode_with_patch_size(
        self, populated_radi_object_module: "RadiObject"
    ) -> None:
        loader = create_distributed_dataloader(
            populated_radi_object_module,
            rank=0,
            world_size=1,
            modalities=["flair"],
            patch_size=(64, 64, 64),
            batch_size=1,
            num_workers=0,
        )

        batch = next(iter(loader))
        assert batch["image"].shape[-3:] == (64, 64, 64)

    def test_multiple_modalities(
        self, populated_radi_object_module: "RadiObject"
    ) -> None:
        loader = create_distributed_dataloader(
            populated_radi_object_module,
            rank=0,
            world_size=1,
            modalities=["flair", "T1w"],
            batch_size=1,
            num_workers=0,
        )

        batch = next(iter(loader))
        assert batch["image"].shape[1] == 2

    def test_cache_strategy_passed(
        self, populated_radi_object_module: "RadiObject"
    ) -> None:
        loader = create_distributed_dataloader(
            populated_radi_object_module,
            rank=0,
            world_size=1,
            modalities=["flair"],
            cache_strategy=CacheStrategy.IN_MEMORY,
            num_workers=0,
        )

        assert isinstance(loader, DataLoader)

    def test_pin_memory_disabled_with_no_workers(
        self, populated_radi_object_module: "RadiObject"
    ) -> None:
        loader = create_distributed_dataloader(
            populated_radi_object_module,
            rank=0,
            world_size=1,
            modalities=["flair"],
            num_workers=0,
            pin_memory=True,
        )

        assert loader.pin_memory is False

    def test_transform_applied(
        self, populated_radi_object_module: "RadiObject"
    ) -> None:
        def add_key(data: dict) -> dict:
            data["custom_key"] = True
            return data

        loader = create_distributed_dataloader(
            populated_radi_object_module,
            rank=0,
            world_size=1,
            modalities=["flair"],
            batch_size=1,
            num_workers=0,
            transform=add_key,
        )

        batch = next(iter(loader))
        assert "custom_key" in batch


class TestSetEpoch:
    """Tests for set_epoch function."""

    def test_set_epoch_updates_sampler(
        self, populated_radi_object_module: "RadiObject"
    ) -> None:
        loader = create_distributed_dataloader(
            populated_radi_object_module,
            rank=0,
            world_size=2,
            modalities=["flair"],
            num_workers=0,
        )

        set_epoch(loader, 5)

        sampler = loader.sampler
        assert isinstance(sampler, DistributedSampler)
        assert sampler.epoch == 5

    def test_set_epoch_no_op_for_regular_dataloader(
        self, populated_radi_object_module: "RadiObject"
    ) -> None:
        from radiobject.ml.config import DatasetConfig, LoadingMode
        from radiobject.ml.datasets.volume_dataset import RadiObjectDataset

        config = DatasetConfig(
            loading_mode=LoadingMode.FULL_VOLUME,
            modalities=["flair"],
        )
        dataset = RadiObjectDataset(populated_radi_object_module, config)
        loader = DataLoader(dataset, batch_size=1)

        set_epoch(loader, 5)


class TestDistributedDataloaderIteration:
    """Tests for iterating over distributed dataloaders."""

    def test_can_iterate_single_rank(
        self, populated_radi_object_module: "RadiObject"
    ) -> None:
        loader = create_distributed_dataloader(
            populated_radi_object_module,
            rank=0,
            world_size=1,
            modalities=["flair"],
            batch_size=1,
            num_workers=0,
        )

        batches = list(loader)
        assert len(batches) == 3

    def test_iteration_with_epoch_change(
        self, populated_radi_object_module: "RadiObject"
    ) -> None:
        loader = create_distributed_dataloader(
            populated_radi_object_module,
            rank=0,
            world_size=1,
            modalities=["flair"],
            batch_size=1,
            num_workers=0,
        )

        set_epoch(loader, 0)
        epoch0_batches = [b["idx"].tolist() for b in loader]

        set_epoch(loader, 1)
        epoch1_batches = [b["idx"].tolist() for b in loader]

        assert len(epoch0_batches) == len(epoch1_batches)

    def test_batch_contains_expected_keys(
        self, populated_radi_object_module: "RadiObject"
    ) -> None:
        loader = create_distributed_dataloader(
            populated_radi_object_module,
            rank=0,
            world_size=1,
            modalities=["flair"],
            batch_size=1,
            num_workers=0,
        )

        batch = next(iter(loader))

        assert "image" in batch
        assert "idx" in batch
        assert isinstance(batch["image"], torch.Tensor)
