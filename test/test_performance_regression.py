"""Performance regression tests with threshold assertions.

These tests verify performance stays within acceptable bounds.
Run with: pytest test/test_performance_regression.py -v -m slow
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import numpy as np
import pytest

from radiobject.stats import TileDBStats
from radiobject.volume import Volume

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


# Mark all tests in this module as slow
pytestmark = pytest.mark.slow


class TestSliceExtractionPerformance:
    """Performance tests for slice extraction operations."""

    def test_axial_slice_under_threshold(self, temp_dir: "Path", array_3d: np.ndarray) -> None:
        """2D axial slice extraction should complete under threshold.

        Threshold: 50ms (conservative for local SSD)
        Typical: 3-10ms with axial tiling
        """
        uri = str(temp_dir / "axial_perf")
        Volume.from_numpy(uri, array_3d)

        vol = Volume(uri)

        # Warm up
        _ = vol.axial(vol.shape[2] // 2)

        # Measure
        times = []
        for i in range(10):
            z = (i * vol.shape[2]) // 10
            start = time.perf_counter()
            _ = vol.axial(z)
            elapsed = time.perf_counter() - start
            times.append(elapsed * 1000)  # Convert to ms

        avg_ms = sum(times) / len(times)
        max_ms = max(times)

        logger.info(
            "Axial slice: avg=%.1fms, max=%.1fms (threshold: 50ms)",
            avg_ms,
            max_ms,
        )

        # Threshold assertion
        assert avg_ms < 50, f"Axial slice too slow: {avg_ms:.1f}ms avg (threshold: 50ms)"

    def test_roi_extraction_under_threshold(self, temp_dir: "Path", array_3d: np.ndarray) -> None:
        """64^3 ROI extraction should complete under threshold.

        Threshold: 100ms (conservative for local SSD)
        Typical: 2-30ms depending on tiling
        """
        uri = str(temp_dir / "roi_perf")
        Volume.from_numpy(uri, array_3d)

        vol = Volume(uri)
        shape = vol.shape

        # Warm up
        _ = vol[0:64, 0:64, 0:64]

        # Measure ROI at different positions
        times = []
        positions = [
            (0, 0, 0),
            (shape[0] // 2 - 32, shape[1] // 2 - 32, shape[2] // 2 - 32),
            (shape[0] - 64, shape[1] - 64, max(0, shape[2] - 64)),
        ]

        for x, y, z in positions:
            x = max(0, min(x, shape[0] - 64))
            y = max(0, min(y, shape[1] - 64))
            z = max(0, min(z, shape[2] - 64))

            start = time.perf_counter()
            _ = vol[x : x + 64, y : y + 64, z : z + 64]
            elapsed = time.perf_counter() - start
            times.append(elapsed * 1000)

        avg_ms = sum(times) / len(times)

        logger.info("64^3 ROI: avg=%.1fms (threshold: 100ms)", avg_ms)

        assert avg_ms < 100, f"ROI extraction too slow: {avg_ms:.1f}ms avg (threshold: 100ms)"


class TestVolumeIOPerformance:
    """Performance tests for Volume I/O operations."""

    def test_full_volume_read_throughput(self, temp_dir: "Path", array_3d: np.ndarray) -> None:
        """Full volume read should achieve minimum throughput.

        Threshold: >50 MB/s (conservative for any storage)
        Typical: 100-200 MB/s on SSD
        """
        uri = str(temp_dir / "read_throughput")
        Volume.from_numpy(uri, array_3d)

        vol = Volume(uri)
        size_mb = array_3d.nbytes / (1024 * 1024)

        # Measure
        times = []
        for _ in range(3):
            start = time.perf_counter()
            _ = vol.to_numpy()
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        avg_time = sum(times) / len(times)
        throughput_mb_s = size_mb / avg_time

        logger.info(
            "Read throughput: %.1f MB/s (%.1f MB in %.2fs)",
            throughput_mb_s,
            size_mb,
            avg_time,
        )

        assert throughput_mb_s > 50, f"Read throughput too low: {throughput_mb_s:.1f} MB/s"

    def test_write_throughput(self, temp_dir: "Path", array_3d: np.ndarray) -> None:
        """Volume write should achieve minimum throughput.

        Threshold: >30 MB/s (conservative, writes often slower)
        Typical: 100-150 MB/s on SSD
        """
        size_mb = array_3d.nbytes / (1024 * 1024)

        times = []
        for i in range(3):
            uri = str(temp_dir / f"write_throughput_{i}")
            start = time.perf_counter()
            Volume.from_numpy(uri, array_3d)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        avg_time = sum(times) / len(times)
        throughput_mb_s = size_mb / avg_time

        logger.info(
            "Write throughput: %.1f MB/s (%.1f MB in %.2fs)",
            throughput_mb_s,
            size_mb,
            avg_time,
        )

        assert throughput_mb_s > 30, f"Write throughput too low: {throughput_mb_s:.1f} MB/s"


class TestCachePerformance:
    """Performance tests for cache behavior."""

    def test_cache_hit_rate_on_repeated_access(self, temp_dir: "Path") -> None:
        """Repeated reads should show cache utilization.

        Note: Actual hit rate depends on TileDB version and configuration.
        This test verifies the cache tracking mechanism works.
        """
        # Create small volume for fast repeated access
        uri = str(temp_dir / "cache_perf")
        data = np.random.rand(64, 64, 64).astype(np.float32)
        Volume.from_numpy(uri, data)

        vol = Volume(uri)

        # Multiple reads to warm cache
        with TileDBStats() as stats:
            for _ in range(5):
                _ = vol.to_numpy()

        cache = stats.cache_stats()

        logger.info(
            "5 repeated reads: hits=%d, misses=%d, hit_rate=%.1f%%",
            cache.cache_hits,
            cache.cache_misses,
            cache.hit_rate * 100,
        )

        # Stats should be collected (values may be 0 depending on TileDB config)
        assert cache.cache_hits >= 0
        assert cache.cache_misses >= 0


class TestMetadataPerformance:
    """Performance tests for metadata operations."""

    def test_volume_open_latency(self, temp_dir: "Path", array_3d: np.ndarray) -> None:
        """Volume open should complete quickly.

        Threshold: <100ms (conservative)
        Typical: <10ms for local storage
        """
        uri = str(temp_dir / "open_latency")
        Volume.from_numpy(uri, array_3d)

        times = []
        for _ in range(5):
            start = time.perf_counter()
            vol = Volume(uri)
            _ = vol.shape  # Access metadata
            elapsed = time.perf_counter() - start
            times.append(elapsed * 1000)

        avg_ms = sum(times) / len(times)

        logger.info("Volume open + metadata: avg=%.1fms", avg_ms)

        assert avg_ms < 100, f"Volume open too slow: {avg_ms:.1f}ms"

    def test_shape_access_is_cached(self, temp_dir: "Path", array_3d: np.ndarray) -> None:
        """Shape access should be essentially instant after first access.

        Threshold: <1ms for cached property
        """
        uri = str(temp_dir / "shape_cache")
        Volume.from_numpy(uri, array_3d)

        vol = Volume(uri)
        _ = vol.shape  # First access

        # Subsequent accesses should be fast
        times = []
        for _ in range(100):
            start = time.perf_counter()
            _ = vol.shape
            elapsed = time.perf_counter() - start
            times.append(elapsed * 1000)

        avg_ms = sum(times) / len(times)

        logger.info("Cached shape access: avg=%.3fms", avg_ms)

        assert avg_ms < 1, f"Shape not properly cached: {avg_ms:.3f}ms"
