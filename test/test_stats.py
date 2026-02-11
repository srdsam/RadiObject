"""Tests for TileDB statistics collection."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from radiobject import CacheStats, S3Stats, TileDBStats
from radiobject.ctx import RadiObjectConfig
from radiobject.volume import Volume

if TYPE_CHECKING:
    from pathlib import Path


class TestTileDBStats:
    """Test TileDBStats context manager."""

    def test_context_manager_returns_self(self) -> None:
        stats = TileDBStats()
        with stats as s:
            assert s is stats

    def test_raw_json_available_after_exit(self, small_volume: Volume) -> None:
        with TileDBStats() as stats:
            vol = Volume(small_volume.uri)
            _ = vol.to_numpy()

        raw = stats.raw_json()
        assert isinstance(raw, str)
        assert len(raw) > 2

    def test_cache_stats_returns_dataclass(self, small_volume: Volume) -> None:
        with TileDBStats() as stats:
            vol = Volume(small_volume.uri)
            _ = vol.to_numpy()

        cache = stats.cache_stats()
        assert isinstance(cache, CacheStats)
        assert hasattr(cache, "cache_hits")
        assert hasattr(cache, "cache_misses")
        assert hasattr(cache, "hit_rate")

    def test_s3_stats_returns_dataclass(self, small_volume: Volume) -> None:
        with TileDBStats() as stats:
            vol = Volume(small_volume.uri)
            _ = vol.to_numpy()

        s3 = stats.s3_stats()
        assert isinstance(s3, S3Stats)
        assert hasattr(s3, "read_ops")
        assert hasattr(s3, "parallelization_rate")

    def test_all_counters_returns_dict(self, small_volume: Volume) -> None:
        with TileDBStats() as stats:
            vol = Volume(small_volume.uri)
            _ = vol.to_numpy()

        counters = stats.all_counters()
        assert isinstance(counters, dict)


class TestCacheStats:
    """Test CacheStats dataclass properties."""

    def test_hit_rate_zero_when_empty(self) -> None:
        stats = CacheStats(
            cache_hits=0,
            cache_misses=0,
            tile_bytes_read=0,
            vfs_read_bytes=0,
        )
        assert stats.hit_rate == 0.0

    def test_hit_rate_calculation(self) -> None:
        stats = CacheStats(
            cache_hits=8,
            cache_misses=2,
            tile_bytes_read=1000,
            vfs_read_bytes=1000,
        )
        assert stats.hit_rate == pytest.approx(0.8)
        assert stats.miss_rate == pytest.approx(0.2)

    def test_hit_rate_one_when_all_hits(self) -> None:
        stats = CacheStats(
            cache_hits=10,
            cache_misses=0,
            tile_bytes_read=1000,
            vfs_read_bytes=0,
        )
        assert stats.hit_rate == 1.0

    def test_frozen_dataclass(self) -> None:
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
        stats = S3Stats(
            read_ops=0,
            parallelized_reads=0,
            write_ops=0,
            total_bytes_read=0,
            total_bytes_written=0,
        )
        assert stats.parallelization_rate == 0.0

    def test_parallelization_rate_calculation(self) -> None:
        stats = S3Stats(
            read_ops=10,
            parallelized_reads=7,
            write_ops=5,
            total_bytes_read=1000000,
            total_bytes_written=500000,
        )
        assert stats.parallelization_rate == 0.7

    def test_frozen_dataclass(self) -> None:
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
        uri = str(temp_dir / "cache_test")
        data = np.random.rand(64, 64, 64).astype(np.float32)
        Volume.from_numpy(uri, data)

        with TileDBStats() as stats1:
            vol = Volume(uri)
            _ = vol.to_numpy()

        cache1 = stats1.cache_stats()

        with TileDBStats() as stats2:
            vol = Volume(uri)
            _ = vol.to_numpy()

        cache2 = stats2.cache_stats()

        assert isinstance(cache1, CacheStats)
        assert isinstance(cache2, CacheStats)

    def test_same_context_cache_behavior(self, temp_dir: "Path") -> None:
        uri = str(temp_dir / "same_ctx_test")
        data = np.random.rand(64, 64, 64).astype(np.float32)
        Volume.from_numpy(uri, data)

        with TileDBStats() as stats:
            vol = Volume(uri)
            _ = vol.to_numpy()
            _ = vol.to_numpy()
            _ = vol.to_numpy()

        cache = stats.cache_stats()
        assert cache.cache_hits > 0


class TestTileCacheConfig:
    """Test tile_cache_size_mb wiring to TileDB config."""

    def test_tile_cache_size_in_config(self) -> None:
        cfg = RadiObjectConfig().to_tiledb_config()
        assert cfg["sm.tile_cache_size"] == str(512 * 1024 * 1024)

    def test_custom_tile_cache_size_in_config(self) -> None:
        from radiobject.ctx import ReadConfig

        config = RadiObjectConfig(read=ReadConfig(tile_cache_size_mb=256))
        cfg = config.to_tiledb_config()
        assert cfg["sm.tile_cache_size"] == str(256 * 1024 * 1024)
