"""Performance benchmarks for ML data loading."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import pytest

from radiobject.ml.config import DatasetConfig, LoadingMode
from radiobject.ml.datasets import VolumeCollectionDataset

if TYPE_CHECKING:
    from radiobject.radi_object import RadiObject

logger = logging.getLogger(__name__)


class TestLoadingPerformance:
    """Benchmark tests for volume loading."""

    def test_full_volume_load_time(self, ml_dataset: VolumeCollectionDataset) -> None:
        """Benchmark full volume loading."""
        _ = ml_dataset[0]

        start = time.perf_counter()
        for i in range(3):
            _ = ml_dataset[i]
        elapsed = time.perf_counter() - start

        logger.info("3 full volumes loaded in %.2fs (%.2fs per volume)", elapsed, elapsed / 3)
        assert elapsed < 30

    def test_patch_extraction_time(self, ml_dataset_patch: VolumeCollectionDataset) -> None:
        """Benchmark patch extraction."""
        _ = ml_dataset_patch[0]

        n_patches = min(10, len(ml_dataset_patch))
        start = time.perf_counter()
        for i in range(n_patches):
            _ = ml_dataset_patch[i]
        elapsed = time.perf_counter() - start

        logger.info(
            "%d patches extracted in %.2fs (%.3fs per patch)",
            n_patches,
            elapsed,
            elapsed / n_patches,
        )
        assert elapsed < 15


class TestMultiWorkerPerformance:
    """Benchmark tests for multi-worker loading."""

    @pytest.mark.parametrize("num_workers", [0, 1, 2])
    def test_dataloader_scaling(
        self, populated_radi_object_module: "RadiObject", num_workers: int
    ) -> None:
        """Benchmark DataLoader with different worker counts."""
        import torch

        config = DatasetConfig(loading_mode=LoadingMode.FULL_VOLUME)
        collection = populated_radi_object_module.collection("flair")
        dataset = VolumeCollectionDataset(collection, config=config)

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            num_workers=num_workers,
            shuffle=False,
        )

        _ = next(iter(loader))

        start = time.perf_counter()
        for batch in loader:
            _ = batch["image"]
        elapsed = time.perf_counter() - start

        logger.info("num_workers=%d: %.2fs total", num_workers, elapsed)


class TestParameterizedPerformance:
    """Performance tests parameterized by storage backend."""

    def test_backend_latency(self, ml_dataset_param: VolumeCollectionDataset) -> None:
        """Measure loading latency for current backend."""
        _ = ml_dataset_param[0]

        start = time.perf_counter()
        _ = ml_dataset_param[0]
        latency = time.perf_counter() - start

        logger.info("Single volume load latency: %.3fs", latency)
        assert latency < 10
