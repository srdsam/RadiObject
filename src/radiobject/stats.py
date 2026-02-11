"""TileDB statistics collection for performance analysis."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from types import TracebackType

import tiledb

log = logging.getLogger(__name__)


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
        except (IndexError, tiledb.TileDBError):
            log.debug("TileDB stats_dump returned no data")
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
        """Parse cache statistics from TileDB stats dump."""
        counters = self._get_counters()
        return CacheStats(
            cache_hits=counters.get(
                "Context.subSubarray.precompute_tile_overlap.tile_overlap_cache_hit", 0
            ),
            cache_misses=0,
            tile_bytes_read=counters.get("Context.Query.Reader.read_unfiltered_byte_num", 0),
            vfs_read_bytes=counters.get("Context.VFS.read_byte_num", 0),
        )

    def s3_stats(self) -> S3Stats:
        """Parse S3/VFS statistics.

        Note:
            Local filesystem operations will show zeros for S3-specific counters.
        """
        counters = self._get_counters()
        return S3Stats(
            read_ops=counters.get("Context.VFS.read_ops_num", 0),
            parallelized_reads=0,
            write_ops=counters.get("Context.VFS.write_ops_num", 0),
            total_bytes_read=counters.get("Context.VFS.read_byte_num", 0),
            total_bytes_written=counters.get("Context.VFS.write_byte_num", 0),
        )

    def all_counters(self) -> dict:
        """Return all TileDB counters as a dictionary.

        Useful for debugging and discovering available counter names.
        """
        return self._get_counters()

    def counters(self) -> dict[str, int]:
        """Return TileDB counters with user-friendly names (no internal prefixes)."""
        raw = self._get_counters()
        cleaned: dict[str, int] = {}
        for key, value in raw.items():
            # Strip internal TileDB path prefixes like "Context.StorageManager.Query.Reader."
            short_key = key.rsplit(".", 1)[-1] if "." in key else key
            # Deduplicate: prefer the qualified (longer) value when names collide
            if short_key not in cleaned:
                cleaned[short_key] = value
        return cleaned
