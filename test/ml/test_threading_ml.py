"""ML-specific threading and multiprocessing tests.

These tests investigate DataLoader worker isolation, memory scaling,
and thread safety of VolumeCollection access.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

from radiobject.ml.config import DatasetConfig, LoadingMode
from radiobject.ml.datasets import VolumeCollectionDataset

if TYPE_CHECKING:
    from radiobject.radi_object import RadiObject

logger = logging.getLogger(__name__)


class TestDataLoaderWorkerContextIsolation:
    """Verify each DataLoader worker has independent TileDB context."""

    def test_worker_process_ids_differ(self, populated_radi_object_module: "RadiObject") -> None:
        """Test that multi-worker DataLoader uses separate processes."""
        config = DatasetConfig(loading_mode=LoadingMode.FULL_VOLUME)
        collection = populated_radi_object_module.collection("flair")
        dataset = VolumeCollectionDataset(collection, config=config)

        # We can't directly test process IDs from here, but we can verify
        # that the worker_init_fn is being called properly by checking
        # that data loads work
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            num_workers=2,
            shuffle=False,
        )

        batches = list(loader)
        assert len(batches) == 3

        for batch in batches:
            assert "image" in batch
            assert batch["image"].shape == (1, 1, 240, 240, 155)

    def test_zero_workers_uses_main_process(
        self, populated_radi_object_module: "RadiObject"
    ) -> None:
        """Test that num_workers=0 uses main process context."""
        config = DatasetConfig(loading_mode=LoadingMode.FULL_VOLUME)
        collection = populated_radi_object_module.collection("flair")
        dataset = VolumeCollectionDataset(collection, config=config)

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            num_workers=0,
            shuffle=False,
        )

        # Should complete without errors
        batches = list(loader)
        assert len(batches) == 3


class TestMultiWorkerMemoryScaling:
    """Measure memory usage with different worker counts."""

    @pytest.mark.parametrize("num_workers", [0, 1, 2])
    def test_memory_with_workers(
        self, populated_radi_object_module: "RadiObject", num_workers: int
    ) -> None:
        """Track memory usage across worker configurations."""
        import psutil

        config = DatasetConfig(loading_mode=LoadingMode.FULL_VOLUME)
        collection = populated_radi_object_module.collection("flair")
        dataset = VolumeCollectionDataset(collection, config=config)

        process = psutil.Process()
        mem_before = process.memory_info().rss / 1e6

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            num_workers=num_workers,
            shuffle=False,
        )

        # Iterate to trigger worker spawning
        for batch in loader:
            _ = batch["image"]

        mem_after = process.memory_info().rss / 1e6
        mem_delta = mem_after - mem_before

        logger.info(
            "num_workers=%d: before=%.1fMB, after=%.1fMB, delta=%.1fMB",
            num_workers,
            mem_before,
            mem_after,
            mem_delta,
        )


class TestIPCSerializationOverhead:
    """Quantify tensor serialization cost in multi-worker DataLoader."""

    def test_ipc_overhead_comparison(self, populated_radi_object_module: "RadiObject") -> None:
        """Compare single-process vs multi-worker loading time."""
        config = DatasetConfig(loading_mode=LoadingMode.FULL_VOLUME)
        collection = populated_radi_object_module.collection("flair")
        dataset = VolumeCollectionDataset(collection, config=config)

        # Single process (no IPC)
        loader_0 = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            num_workers=0,
            shuffle=False,
        )

        # Warm up
        _ = list(loader_0)

        start = time.perf_counter()
        for batch in loader_0:
            _ = batch["image"]
        time_0 = time.perf_counter() - start

        # Multi-worker (with IPC)
        loader_2 = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            num_workers=2,
            shuffle=False,
        )

        # Warm up
        _ = list(loader_2)

        start = time.perf_counter()
        for batch in loader_2:
            _ = batch["image"]
        time_2 = time.perf_counter() - start

        ipc_overhead = time_2 - time_0
        logger.info(
            "0 workers: %.3fs, 2 workers: %.3fs, IPC overhead: %.3fs",
            time_0,
            time_2,
            ipc_overhead,
        )

        # For small datasets, multi-worker is often slower due to IPC
        if time_2 > time_0:
            logger.info("IPC overhead dominates for this dataset size")


class TestConcurrentVolumeReadsThreadSafety:
    """Stress test VolumeCollection thread safety."""

    def test_concurrent_reads_same_volume(self, populated_radi_object_module: "RadiObject") -> None:
        """Test concurrent reads of the same volume."""
        collection = populated_radi_object_module.collection("flair")

        errors: list[Exception] = []
        results: list[np.ndarray] = []
        lock = threading.Lock()

        def read_volume(idx: int) -> None:
            try:
                data = collection.iloc[idx % len(collection)].to_numpy()
                with lock:
                    results.append(data)
            except Exception as e:
                with lock:
                    errors.append(e)

        # Spawn many concurrent reads
        n_concurrent = 20
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(read_volume, i) for i in range(n_concurrent)]
            for f in as_completed(futures):
                pass

        assert len(errors) == 0, f"Errors during concurrent reads: {errors}"
        assert len(results) == n_concurrent

        # All results should have same shape
        shapes = [r.shape for r in results]
        assert all(s == shapes[0] for s in shapes)

    def test_concurrent_reads_different_volumes(
        self, populated_radi_object_module: "RadiObject"
    ) -> None:
        """Test concurrent reads of different volumes."""
        collection = populated_radi_object_module.collection("flair")

        errors: list[Exception] = []
        results: dict[int, np.ndarray] = {}
        lock = threading.Lock()

        def read_volume(idx: int) -> None:
            try:
                data = collection.iloc[idx].to_numpy()
                with lock:
                    results[idx] = data
            except Exception as e:
                with lock:
                    errors.append(e)

        # Read all volumes concurrently
        n_volumes = len(collection)
        with ThreadPoolExecutor(max_workers=n_volumes) as executor:
            futures = [executor.submit(read_volume, i) for i in range(n_volumes)]
            for f in as_completed(futures):
                pass

        assert len(errors) == 0, f"Errors during concurrent reads: {errors}"
        assert len(results) == n_volumes

    def test_concurrent_patch_reads(self, populated_radi_object_module: "RadiObject") -> None:
        """Test concurrent patch extraction."""
        collection = populated_radi_object_module.collection("flair")
        patch_size = (64, 64, 64)

        errors: list[Exception] = []
        results: list[np.ndarray] = []
        lock = threading.Lock()

        def read_patch(idx: int) -> None:
            try:
                vol_idx = idx % len(collection)
                rng = np.random.default_rng(seed=idx)
                shape = collection.shape
                start = tuple(rng.integers(0, shape[i] - patch_size[i] + 1) for i in range(3))
                vol = collection.iloc[vol_idx]
                data = vol.slice(
                    slice(start[0], start[0] + patch_size[0]),
                    slice(start[1], start[1] + patch_size[1]),
                    slice(start[2], start[2] + patch_size[2]),
                )
                with lock:
                    results.append(data)
            except Exception as e:
                with lock:
                    errors.append(e)

        n_patches = 30
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(read_patch, i) for i in range(n_patches)]
            for f in as_completed(futures):
                pass

        assert len(errors) == 0, f"Errors during concurrent patch reads: {errors}"
        assert len(results) == n_patches

        # All patches should have correct shape
        for patch in results:
            assert patch.shape == patch_size
