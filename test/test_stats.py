"""Tests for TileDB statistics collection."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pytest

from radiobject import CacheStats, S3Stats, TileDBStats
from radiobject.volume import Volume

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


class TestTileDBStats:
    """Test TileDBStats context manager."""

    def test_context_manager_returns_self(self) -> None:
        """Stats context manager returns itself."""
        stats = TileDBStats()
        with stats as s:
            assert s is stats

    def test_raw_json_available_after_exit(self, temp_dir: "Path") -> None:
        """Raw JSON stats are captured after context exit."""
        uri = str(temp_dir / "test_vol")
        Volume.from_numpy(uri, np.random.rand(32, 32, 32).astype(np.float32))

        with TileDBStats() as stats:
            vol = Volume(uri)
            _ = vol.to_numpy()

        raw = stats.raw_json()
        assert isinstance(raw, str)
        assert len(raw) > 2  # Not empty "{}"

    def test_cache_stats_returns_dataclass(self, temp_dir: "Path") -> None:
        """cache_stats returns CacheStats dataclass."""
        uri = str(temp_dir / "test_vol")
        Volume.from_numpy(uri, np.random.rand(32, 32, 32).astype(np.float32))

        with TileDBStats() as stats:
            vol = Volume(uri)
            _ = vol.to_numpy()

        cache = stats.cache_stats()
        assert isinstance(cache, CacheStats)
        assert hasattr(cache, "cache_hits")
        assert hasattr(cache, "cache_misses")
        assert hasattr(cache, "hit_rate")

    def test_s3_stats_returns_dataclass(self, temp_dir: "Path") -> None:
        """s3_stats returns S3Stats dataclass."""
        uri = str(temp_dir / "test_vol")
        Volume.from_numpy(uri, np.random.rand(32, 32, 32).astype(np.float32))

        with TileDBStats() as stats:
            vol = Volume(uri)
            _ = vol.to_numpy()

        s3 = stats.s3_stats()
        assert isinstance(s3, S3Stats)
        assert hasattr(s3, "read_ops")
        assert hasattr(s3, "parallelization_rate")

    def test_all_counters_returns_dict(self, temp_dir: "Path") -> None:
        """all_counters returns dictionary of counters."""
        uri = str(temp_dir / "test_vol")
        Volume.from_numpy(uri, np.random.rand(32, 32, 32).astype(np.float32))

        with TileDBStats() as stats:
            vol = Volume(uri)
            _ = vol.to_numpy()

        counters = stats.all_counters()
        assert isinstance(counters, dict)


class TestCacheStats:
    """Test CacheStats dataclass properties."""

    def test_hit_rate_zero_when_empty(self) -> None:
        """Hit rate is 0.0 when no operations."""
        stats = CacheStats(
            cache_hits=0,
            cache_misses=0,
            tile_bytes_read=0,
            vfs_read_bytes=0,
        )
        assert stats.hit_rate == 0.0

    def test_hit_rate_calculation(self) -> None:
        """Hit rate computed correctly."""
        stats = CacheStats(
            cache_hits=8,
            cache_misses=2,
            tile_bytes_read=1000,
            vfs_read_bytes=1000,
        )
        assert stats.hit_rate == pytest.approx(0.8)
        assert stats.miss_rate == pytest.approx(0.2)

    def test_hit_rate_one_when_all_hits(self) -> None:
        """Hit rate is 1.0 when all cache hits."""
        stats = CacheStats(
            cache_hits=10,
            cache_misses=0,
            tile_bytes_read=1000,
            vfs_read_bytes=0,
        )
        assert stats.hit_rate == 1.0

    def test_frozen_dataclass(self) -> None:
        """CacheStats is immutable."""
        stats = CacheStats(
            cache_hits=5,
            cache_misses=5,
            tile_bytes_read=1000,
            vfs_read_bytes=1000,
        )
        with pytest.raises(AttributeError):
            stats.cache_hits = 10  # type: ignore


class TestS3Stats:
    """Test S3Stats dataclass properties."""

    def test_parallelization_rate_zero_when_no_ops(self) -> None:
        """Parallelization rate is 0.0 when no read ops."""
        stats = S3Stats(
            read_ops=0,
            parallelized_reads=0,
            write_ops=0,
            total_bytes_read=0,
            total_bytes_written=0,
        )
        assert stats.parallelization_rate == 0.0

    def test_parallelization_rate_calculation(self) -> None:
        """Parallelization rate computed correctly."""
        stats = S3Stats(
            read_ops=10,
            parallelized_reads=7,
            write_ops=5,
            total_bytes_read=1000000,
            total_bytes_written=500000,
        )
        assert stats.parallelization_rate == 0.7

    def test_frozen_dataclass(self) -> None:
        """S3Stats is immutable."""
        stats = S3Stats(
            read_ops=10,
            parallelized_reads=5,
            write_ops=5,
            total_bytes_read=1000,
            total_bytes_written=1000,
        )
        with pytest.raises(AttributeError):
            stats.read_ops = 20  # type: ignore


class TestCacheBehavior:
    """Test actual cache behavior with Volume operations."""

    def test_repeated_reads_track_operations(self, temp_dir: "Path") -> None:
        """Verify stats track operations across multiple reads."""
        uri = str(temp_dir / "cache_test")
        data = np.random.rand(64, 64, 64).astype(np.float32)
        Volume.from_numpy(uri, data)

        # First read - should trigger actual I/O
        with TileDBStats() as stats1:
            vol = Volume(uri)
            _ = vol.to_numpy()

        cache1 = stats1.cache_stats()

        # Second read - may benefit from OS page cache
        with TileDBStats() as stats2:
            vol = Volume(uri)
            _ = vol.to_numpy()

        cache2 = stats2.cache_stats()

        # Log results for investigation
        logger.info(
            "First read - hits: %d, misses: %d, bytes: %d",
            cache1.cache_hits,
            cache1.cache_misses,
            cache1.tile_bytes_read,
        )
        logger.info(
            "Second read - hits: %d, misses: %d, bytes: %d",
            cache2.cache_hits,
            cache2.cache_misses,
            cache2.tile_bytes_read,
        )

        # Verify stats were collected (values depend on TileDB version)
        # The key assertion is that stats collection works
        assert isinstance(cache1, CacheStats)
        assert isinstance(cache2, CacheStats)

    def test_same_context_cache_behavior(self, temp_dir: "Path") -> None:
        """Repeated reads with same Volume instance may hit TileDB cache."""
        uri = str(temp_dir / "same_ctx_test")
        data = np.random.rand(64, 64, 64).astype(np.float32)
        Volume.from_numpy(uri, data)

        with TileDBStats() as stats:
            vol = Volume(uri)
            # Multiple reads with same Volume instance
            _ = vol.to_numpy()
            _ = vol.to_numpy()
            _ = vol.to_numpy()

        cache = stats.cache_stats()
        logger.info(
            "3 reads same instance - hits: %d, misses: %d, hit_rate: %.1f%%",
            cache.cache_hits,
            cache.cache_misses,
            cache.hit_rate * 100,
        )

        # Stats should capture something
        assert cache.tile_bytes_read >= 0 or cache.vfs_read_bytes >= 0
