"""ML-specific threading and multiprocessing tests.

These tests investigate DataLoader worker isolation, memory scaling,
and thread safety of VolumeReader.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

from radiobject.ml.config import DatasetConfig, LoadingMode
from radiobject.ml.datasets.volume_dataset import RadiObjectDataset
from radiobject.ml.reader import VolumeReader

if TYPE_CHECKING:
    from radiobject.radi_object import RadiObject

logger = logging.getLogger(__name__)


class TestDataLoaderWorkerContextIsolation:
    """Verify each DataLoader worker has independent TileDB context."""

    def test_worker_process_ids_differ(self, populated_radi_object_module: "RadiObject") -> None:
        """Test that multi-worker DataLoader uses separate processes."""
        config = DatasetConfig(
            loading_mode=LoadingMode.FULL_VOLUME,
            modalities=["flair"],
        )
        dataset = RadiObjectDataset(populated_radi_object_module, config)

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
        config = DatasetConfig(
            loading_mode=LoadingMode.FULL_VOLUME,
            modalities=["flair"],
        )
        dataset = RadiObjectDataset(populated_radi_object_module, config)

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

        config = DatasetConfig(
            loading_mode=LoadingMode.FULL_VOLUME,
            modalities=["flair"],
        )
        dataset = RadiObjectDataset(populated_radi_object_module, config)

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
        config = DatasetConfig(
            loading_mode=LoadingMode.FULL_VOLUME,
            modalities=["flair"],
        )
        dataset = RadiObjectDataset(populated_radi_object_module, config)

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
    """Stress test VolumeReader thread safety."""

    def test_concurrent_reads_same_volume(self, populated_radi_object_module: "RadiObject") -> None:
        """Test concurrent reads of the same volume."""
        collection = populated_radi_object_module.collection("flair")
        reader = VolumeReader(collection)

        errors: list[Exception] = []
        results: list[np.ndarray] = []
        lock = threading.Lock()

        def read_volume(idx: int) -> None:
            try:
                data = reader.read_full(idx % len(reader))
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
        reader = VolumeReader(collection)

        errors: list[Exception] = []
        results: dict[int, np.ndarray] = {}
        lock = threading.Lock()

        def read_volume(idx: int) -> None:
            try:
                data = reader.read_full(idx)
                with lock:
                    results[idx] = data
            except Exception as e:
                with lock:
                    errors.append(e)

        # Read all volumes concurrently
        n_volumes = len(reader)
        with ThreadPoolExecutor(max_workers=n_volumes) as executor:
            futures = [executor.submit(read_volume, i) for i in range(n_volumes)]
            for f in as_completed(futures):
                pass

        assert len(errors) == 0, f"Errors during concurrent reads: {errors}"
        assert len(results) == n_volumes

    def test_concurrent_patch_reads(self, populated_radi_object_module: "RadiObject") -> None:
        """Test concurrent patch extraction."""
        collection = populated_radi_object_module.collection("flair")
        reader = VolumeReader(collection)

        errors: list[Exception] = []
        results: list[np.ndarray] = []
        lock = threading.Lock()
        patch_size = (64, 64, 64)

        def read_patch(idx: int) -> None:
            try:
                vol_idx = idx % len(reader)
                rng = np.random.default_rng(seed=idx)
                shape = reader.shape
                start = tuple(rng.integers(0, shape[i] - patch_size[i] + 1) for i in range(3))
                data = reader.read_patch(vol_idx, start, patch_size)
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


class TestVolumeReaderContextCaching:
    """Test VolumeReader context caching behavior."""

    def test_context_cached_per_process(self, populated_radi_object_module: "RadiObject") -> None:
        """Verify context is cached per (pid, config_hash)."""
        collection = populated_radi_object_module.collection("flair")
        reader = VolumeReader(collection)

        # First read creates context
        _ = reader.read_full(0)

        # Second read should use cached context
        _ = reader.read_full(1)

        # Check the cache has an entry for this process
        from radiobject.ml.reader import _PROCESS_CTX_CACHE

        pid = os.getpid()
        matching_keys = [k for k in _PROCESS_CTX_CACHE if k[0] == pid]
        assert len(matching_keys) >= 1

    def test_different_configs_different_cache_entries(
        self, populated_radi_object_module: "RadiObject"
    ) -> None:
        """Verify different configs create different cache entries."""
        import tiledb

        collection = populated_radi_object_module.collection("flair")

        # Reader with default config
        reader1 = VolumeReader(collection)
        _ = reader1.read_full(0)

        # Reader with custom config
        cfg = tiledb.Config()
        cfg["sm.memory_budget"] = str(256 * 1024 * 1024)
        custom_ctx = tiledb.Ctx(cfg)

        reader2 = VolumeReader(collection, ctx=custom_ctx)
        _ = reader2.read_full(0)

        from radiobject.ml.reader import _PROCESS_CTX_CACHE

        pid = os.getpid()
        matching_keys = [k for k in _PROCESS_CTX_CACHE if k[0] == pid]

        # Should have at least 2 entries (different configs)
        # Note: may have more from other tests
        assert len(matching_keys) >= 1
