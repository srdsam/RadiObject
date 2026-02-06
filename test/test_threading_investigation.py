"""Threading and parallelization investigation tests.

These tests empirically measure the impact of threading configuration on RadiObject
performance, particularly for S3 cloud write operations.
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING

import numpy as np
import pytest
import tiledb

from radiobject import configure, get_radiobject_config, get_tiledb_ctx, reset_radiobject_config
from radiobject.ctx import RadiObjectConfig, ReadConfig
from radiobject.parallel import ctx_for_threads, map_on_threads
from radiobject.volume import Volume

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


class TestWritePhaseTimingBreakdown:
    """Instrument write path to identify slowest phases."""

    def test_volume_write_phases(self, temp_dir: "Path", array_3d: np.ndarray) -> None:
        """Break down Volume.from_numpy into individual phases."""
        phases = {}

        # Phase 1: Array creation (schema definition)
        uri = str(temp_dir / "timing_test")
        start = time.perf_counter()
        schema = tiledb.ArraySchema(
            domain=tiledb.Domain(
                tiledb.Dim(name="x", domain=(0, array_3d.shape[0] - 1), dtype=np.int32),
                tiledb.Dim(name="y", domain=(0, array_3d.shape[1] - 1), dtype=np.int32),
                tiledb.Dim(name="z", domain=(0, array_3d.shape[2] - 1), dtype=np.int32),
            ),
            attrs=[tiledb.Attr(name="data", dtype=array_3d.dtype)],
        )
        tiledb.Array.create(uri, schema)
        phases["array_create"] = time.perf_counter() - start

        # Phase 2: Data write
        start = time.perf_counter()
        with tiledb.open(uri, "w") as arr:
            arr[:] = {"data": array_3d}
        phases["data_write"] = time.perf_counter() - start

        # Phase 3: Metadata write
        start = time.perf_counter()
        with tiledb.open(uri, "w") as arr:
            arr.meta["obs_id"] = "test_volume"
            arr.meta["orientation_axcodes"] = "RAS"
        phases["metadata_write"] = time.perf_counter() - start

        total = sum(phases.values())
        for phase, duration in phases.items():
            pct = (duration / total) * 100
            logger.info("Phase %-16s: %.3fs (%.1f%%)", phase, duration, pct)

        logger.info("Total: %.3fs", total)

        # Verify data integrity
        vol = Volume(uri)
        assert vol.shape == array_3d.shape

    def test_parallel_vs_serial_volume_writes(self, temp_dir: "Path", array_3d: np.ndarray) -> None:
        """Compare serial vs parallel volume write performance."""
        n_volumes = 4

        # Serial writes
        start = time.perf_counter()
        for i in range(n_volumes):
            uri = str(temp_dir / f"serial_{i}")
            Volume.from_numpy(uri, array_3d)
        serial_time = time.perf_counter() - start

        # Parallel writes using map_on_threads
        def write_volume(idx: int) -> str:
            uri = str(temp_dir / f"parallel_{idx}")
            Volume.from_numpy(uri, array_3d)
            return uri

        start = time.perf_counter()
        _ = map_on_threads(write_volume, range(n_volumes))
        parallel_time = time.perf_counter() - start

        speedup = serial_time / parallel_time if parallel_time > 0 else 1.0
        logger.info(
            "Serial: %.2fs, Parallel: %.2fs, Speedup: %.2fx",
            serial_time,
            parallel_time,
            speedup,
        )

        assert parallel_time < serial_time


class TestIOConcurrencyImpact:
    """Measure impact of TileDB io_concurrency_level setting."""

    @pytest.mark.parametrize("io_concurrency", [1, 2, 4, 8])
    def test_read_concurrency_settings(
        self,
        temp_dir: "Path",
        array_3d: np.ndarray,
        io_concurrency: int,
    ) -> None:
        """Vary TileDB io_concurrency, measure read throughput."""
        uri = str(temp_dir / f"concurrency_test_{io_concurrency}")
        Volume.from_numpy(uri, array_3d)

        cfg = tiledb.Config()
        cfg["sm.io_concurrency_level"] = str(io_concurrency)
        cfg["sm.compute_concurrency_level"] = str(io_concurrency)
        test_ctx = tiledb.Ctx(cfg)

        vol = Volume(uri, ctx=test_ctx)
        _ = vol.to_numpy()

        n_reads = 3
        start = time.perf_counter()
        for _ in range(n_reads):
            _ = vol.to_numpy()
        elapsed = time.perf_counter() - start

        throughput_mb = (array_3d.nbytes * n_reads / 1e6) / elapsed
        logger.info(
            "io_concurrency=%d: %.2fs for %d reads (%.1f MB/s)",
            io_concurrency,
            elapsed,
            n_reads,
            throughput_mb,
        )


class TestMemoryBudgetImpact:
    """Measure impact of sm.memory_budget on write performance."""

    @pytest.mark.parametrize("memory_budget_mb", [256, 512, 1024, 2048])
    def test_memory_budget_write_performance(
        self,
        temp_dir: "Path",
        array_3d: np.ndarray,
        memory_budget_mb: int,
    ) -> None:
        """Vary TileDB memory_budget, measure write throughput."""
        cfg = tiledb.Config()
        cfg["sm.memory_budget"] = str(memory_budget_mb * 1024 * 1024)
        test_ctx = tiledb.Ctx(cfg)

        uri = str(temp_dir / f"memory_test_{memory_budget_mb}")
        start = time.perf_counter()
        Volume.from_numpy(uri, array_3d, ctx=test_ctx)
        elapsed = time.perf_counter() - start

        throughput_mb = (array_3d.nbytes / 1e6) / elapsed
        logger.info(
            "memory_budget=%dMB: %.2fs (%.1f MB/s)",
            memory_budget_mb,
            elapsed,
            throughput_mb,
        )


class TestThreadPoolVsTileDBThreading:
    """Compare application-level vs TileDB internal threading."""

    def test_4x4_vs_1x16_threading(self, temp_dir: "Path", array_3d: np.ndarray) -> None:
        """Compare 4 workers x 4 TileDB threads vs 1 worker x 16 TileDB threads."""
        n_volumes = 4

        # Configuration A: 4 workers x 4 TileDB threads
        cfg_a = RadiObjectConfig(read=ReadConfig(concurrency=4, max_workers=4))
        configure(read=cfg_a.read)

        def write_vol_a(idx: int) -> str:
            uri = str(temp_dir / f"config_a_{idx}")
            Volume.from_numpy(uri, array_3d, ctx=cfg_a.to_tiledb_ctx())
            return uri

        start = time.perf_counter()
        _ = map_on_threads(write_vol_a, range(n_volumes), max_workers=4)
        time_a = time.perf_counter() - start

        # Configuration B: 1 worker x 16 TileDB threads
        cfg_b = RadiObjectConfig(read=ReadConfig(concurrency=16, max_workers=1))

        def write_vol_b(idx: int) -> str:
            uri = str(temp_dir / f"config_b_{idx}")
            Volume.from_numpy(uri, array_3d, ctx=cfg_b.to_tiledb_ctx())
            return uri

        start = time.perf_counter()
        _ = map_on_threads(write_vol_b, range(n_volumes), max_workers=1)
        time_b = time.perf_counter() - start

        reset_radiobject_config()

        logger.info(
            "4x4 config: %.2fs, 1x16 config: %.2fs, ratio: %.2f",
            time_a,
            time_b,
            time_a / time_b if time_b > 0 else float("inf"),
        )


class TestConcurrentTileDBOpensContention:
    """Detect thread contention in concurrent TileDB opens."""

    def test_concurrent_open_contention(self, temp_dir: "Path", array_3d: np.ndarray) -> None:
        """Measure if concurrent tiledb.open() calls cause contention."""
        uris = []
        for i in range(8):
            uri = str(temp_dir / f"contention_test_{i}")
            Volume.from_numpy(uri, array_3d)
            uris.append(uri)

        start = time.perf_counter()
        for uri in uris:
            with tiledb.open(uri) as arr:
                _ = arr.shape
        serial_time = time.perf_counter() - start

        def open_and_read_shape(uri: str) -> tuple:
            with tiledb.open(uri) as arr:
                return arr.shape

        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(open_and_read_shape, uri) for uri in uris]
            for f in as_completed(futures):
                _ = f.result()
        parallel_time = time.perf_counter() - start

        speedup = serial_time / parallel_time if parallel_time > 0 else 1.0
        logger.info(
            "Serial opens: %.3fs, Parallel opens: %.3fs, Speedup: %.2fx",
            serial_time,
            parallel_time,
            speedup,
        )

        if speedup < 0.8:
            logger.warning("Detected potential contention in parallel opens")


class TestMaxWorkersConfiguration:
    """Test that max_workers is properly configurable."""

    def test_max_workers_from_config(self) -> None:
        """Verify max_workers defaults and can be configured."""
        default_config = get_radiobject_config()
        assert default_config.read.max_workers == 4

        configure(read=ReadConfig(max_workers=8))
        new_config = get_radiobject_config()
        assert new_config.read.max_workers == 8

        reset_radiobject_config()

    def test_map_on_threads_respects_config(self, temp_dir: "Path", array_3d: np.ndarray) -> None:
        """Verify map_on_threads uses configured max_workers."""
        configure(read=ReadConfig(max_workers=2))

        worker_pids: list[int] = []

        def record_worker_pid(idx: int) -> int:
            worker_pids.append(os.getpid())
            return idx

        _ = map_on_threads(record_worker_pid, range(10))

        # All should run in same process (threads, not processes)
        assert len(set(worker_pids)) == 1

        reset_radiobject_config()


class TestWorkerContextIsolation:
    """Test that worker contexts are properly isolated."""

    def test_ctx_for_threads_returns_context(self) -> None:
        """Verify ctx_for_threads returns a valid context."""
        ctx1 = ctx_for_threads()
        ctx2 = ctx_for_threads()

        # ctx_for_threads returns shared context (thread-safe)
        assert isinstance(ctx1, tiledb.Ctx)
        assert isinstance(ctx2, tiledb.Ctx)

    def test_worker_ctx_inherits_base_config(self) -> None:
        """Verify ctx_for_threads with base context returns valid context."""
        base_ctx = get_tiledb_ctx()
        worker_ctx = ctx_for_threads(base_ctx)

        assert isinstance(worker_ctx, tiledb.Ctx)
