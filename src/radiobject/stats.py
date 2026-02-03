"""TileDB statistics collection for performance analysis."""

from __future__ import annotations

import json
from dataclasses import dataclass
from types import TracebackType

import tiledb


@dataclass(frozen=True)
class CacheStats:
    """TileDB cache statistics."""

    cache_hits: int
    cache_misses: int
    tile_bytes_read: int
    vfs_read_bytes: int

    @property
    def hit_rate(self) -> float:
        """Compute cache hit rate (0.0 to 1.0)."""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0

    @property
    def miss_rate(self) -> float:
        """Compute cache miss rate (0.0 to 1.0)."""
        return 1.0 - self.hit_rate


@dataclass(frozen=True)
class S3Stats:
    """S3/VFS statistics."""

    read_ops: int
    parallelized_reads: int
    write_ops: int
    total_bytes_read: int
    total_bytes_written: int

    @property
    def parallelization_rate(self) -> float:
        """Fraction of reads that were parallelized (0.0 to 1.0)."""
        if self.read_ops == 0:
            return 0.0
        return self.parallelized_reads / self.read_ops


class TileDBStats:
    """Context manager for collecting TileDB statistics.

    Wraps TileDB's internal stats collection to provide structured
    access to cache and VFS metrics.

    Example:
        with TileDBStats() as stats:
            vol = Volume(uri)
            data = vol.to_numpy()

        cache = stats.cache_stats()
        print(f"Cache hit rate: {cache.hit_rate:.1%}")
    """

    def __init__(self) -> None:
        """Initialize stats collector."""
        self._raw_stats: str = "{}"

    def __enter__(self) -> TileDBStats:
        """Enable TileDB statistics collection."""
        tiledb.stats_enable()
        tiledb.stats_reset()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Capture stats and disable collection."""
        try:
            self._raw_stats = tiledb.stats_dump(json=True)
        except (IndexError, Exception):
            # TileDB stats_dump can fail if no operations were performed
            self._raw_stats = "{}"
        finally:
            tiledb.stats_disable()

    def raw_json(self) -> str:
        """Return raw TileDB stats JSON string."""
        return self._raw_stats

    def _get_counters(self) -> dict:
        """Parse counters from raw stats."""
        try:
            stats = json.loads(self._raw_stats)
            return stats.get("counters", {})
        except json.JSONDecodeError:
            return {}

    def cache_stats(self) -> CacheStats:
        """Parse cache statistics from TileDB stats dump.

        Note:
            Counter names may vary between TileDB versions.
            Falls back to 0 if counter not found.
        """
        counters = self._get_counters()

        # TileDB 2.x counter names (may vary by version)
        # Try multiple possible counter names for compatibility
        cache_hits = (
            counters.get("Context.StorageManager.Query.Reader.cache_lru_read_hits", 0)
            or counters.get("cache_lru_read_hits", 0)
            or counters.get("tile_cache_hits", 0)
        )
        cache_misses = (
            counters.get("Context.StorageManager.Query.Reader.cache_lru_read_misses", 0)
            or counters.get("cache_lru_read_misses", 0)
            or counters.get("tile_cache_misses", 0)
        )
        tile_bytes = counters.get(
            "Context.StorageManager.Query.Reader.num_tile_bytes_read", 0
        ) or counters.get("num_tile_bytes_read", 0)
        vfs_bytes = counters.get("Context.StorageManager.VFS.read_total_bytes", 0) or counters.get(
            "vfs_read_total_bytes", 0
        )

        return CacheStats(
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            tile_bytes_read=tile_bytes,
            vfs_read_bytes=vfs_bytes,
        )

    def s3_stats(self) -> S3Stats:
        """Parse S3/VFS statistics.

        Note:
            These counters are only populated for S3 operations.
            Local filesystem operations will show zeros.
        """
        counters = self._get_counters()

        read_ops = counters.get("Context.StorageManager.VFS.read_ops", 0) or counters.get(
            "vfs_read_ops", 0
        )
        parallel_reads = counters.get(
            "Context.StorageManager.VFS.read_ops_parallelized", 0
        ) or counters.get("vfs_read_ops_parallelized", 0)
        write_ops = counters.get("Context.StorageManager.VFS.write_ops", 0) or counters.get(
            "vfs_write_ops", 0
        )
        bytes_read = counters.get("Context.StorageManager.VFS.read_total_bytes", 0) or counters.get(
            "vfs_read_total_bytes", 0
        )
        bytes_written = counters.get(
            "Context.StorageManager.VFS.write_total_bytes", 0
        ) or counters.get("vfs_write_total_bytes", 0)

        return S3Stats(
            read_ops=read_ops,
            parallelized_reads=parallel_reads,
            write_ops=write_ops,
            total_bytes_read=bytes_read,
            total_bytes_written=bytes_written,
        )

    def all_counters(self) -> dict:
        """Return all TileDB counters as a dictionary.

        Useful for debugging and discovering available counter names.
        """
        return self._get_counters()
