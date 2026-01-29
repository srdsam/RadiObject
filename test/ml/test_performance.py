"""Performance benchmarks for ML data loading."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import pytest

from ml.config import CacheStrategy, DatasetConfig, LoadingMode
from ml.datasets.volume_dataset import RadiObjectDataset

if TYPE_CHECKING:
    from src.radi_object import RadiObject


class TestLoadingPerformance:
    """Benchmark tests for volume loading."""

    def test_full_volume_load_time(self, ml_dataset: RadiObjectDataset) -> None:
        """Benchmark full volume loading."""
        _ = ml_dataset[0]

        start = time.perf_counter()
        for i in range(3):
            _ = ml_dataset[i]
        elapsed = time.perf_counter() - start

        print(f"\n3 full volumes loaded in {elapsed:.2f}s ({elapsed/3:.2f}s per volume)")
        assert elapsed < 30

    def test_patch_extraction_time(self, ml_dataset_patch: RadiObjectDataset) -> None:
        """Benchmark patch extraction."""
        _ = ml_dataset_patch[0]

        n_patches = min(10, len(ml_dataset_patch))
        start = time.perf_counter()
        for i in range(n_patches):
            _ = ml_dataset_patch[i]
        elapsed = time.perf_counter() - start

        print(f"\n{n_patches} patches extracted in {elapsed:.2f}s ({elapsed/n_patches:.3f}s per patch)")
        assert elapsed < 15


class TestCachePerformance:
    """Benchmark tests for caching strategies."""

    def test_cache_speedup(self, populated_radi_object_module: "RadiObject") -> None:
        """Compare NoCache vs InMemoryCache performance."""
        config_no_cache = DatasetConfig(
            loading_mode=LoadingMode.FULL_VOLUME,
            cache_strategy=CacheStrategy.NONE,
            modalities=["flair"],
        )
        dataset_no_cache = RadiObjectDataset(populated_radi_object_module, config_no_cache)

        config_cached = DatasetConfig(
            loading_mode=LoadingMode.FULL_VOLUME,
            cache_strategy=CacheStrategy.IN_MEMORY,
            modalities=["flair"],
        )
        dataset_cached = RadiObjectDataset(populated_radi_object_module, config_cached)

        _ = dataset_no_cache[0]
        start = time.perf_counter()
        for _ in range(3):
            _ = dataset_no_cache[0]
        uncached_time = time.perf_counter() - start

        _ = dataset_cached[0]
        start = time.perf_counter()
        for _ in range(3):
            _ = dataset_cached[0]
        cached_time = time.perf_counter() - start

        print(f"\nUncached: {uncached_time:.3f}s, Cached: {cached_time:.3f}s")
        print(f"Speedup: {uncached_time / cached_time:.1f}x")

        assert cached_time < uncached_time

    def test_cache_hit_rate(self, ml_dataset_cached: RadiObjectDataset) -> None:
        """Test cache achieves expected hit rate."""
        ml_dataset_cached.cache.clear()

        for _ in range(3):
            for i in range(len(ml_dataset_cached)):
                _ = ml_dataset_cached[i]

        cache = ml_dataset_cached.cache
        total = cache.hits + cache.misses
        hit_rate = cache.hits / total if total > 0 else 0

        print(f"\nCache hits: {cache.hits}, misses: {cache.misses}, hit rate: {hit_rate:.1%}")
        assert hit_rate > 0.5


class TestMultiWorkerPerformance:
    """Benchmark tests for multi-worker loading."""

    @pytest.mark.parametrize("num_workers", [0, 1, 2])
    def test_dataloader_scaling(
        self, populated_radi_object_module: "RadiObject", num_workers: int
    ) -> None:
        """Benchmark DataLoader with different worker counts."""
        import torch

        config = DatasetConfig(
            loading_mode=LoadingMode.FULL_VOLUME,
            modalities=["flair"],
        )
        dataset = RadiObjectDataset(populated_radi_object_module, config)

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

        print(f"\nnum_workers={num_workers}: {elapsed:.2f}s total")


class TestParameterizedPerformance:
    """Performance tests parameterized by storage backend."""

    def test_backend_latency(self, ml_dataset_param: RadiObjectDataset) -> None:
        """Measure loading latency for current backend."""
        _ = ml_dataset_param[0]

        start = time.perf_counter()
        _ = ml_dataset_param[0]
        latency = time.perf_counter() - start

        print(f"\nSingle volume load latency: {latency:.3f}s")
        assert latency < 10
