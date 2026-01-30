"""Caching utilities for ML datasets."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseCache(ABC):
    """Abstract base class for sample caching."""

    def __init__(self) -> None:
        self._hits = 0
        self._misses = 0

    @property
    def hits(self) -> int:
        """Number of cache hits."""
        return self._hits

    @property
    def misses(self) -> int:
        """Number of cache misses."""
        return self._misses

    @abstractmethod
    def get(self, key: int) -> Any | None:
        """Get cached sample by key, or None if not cached."""
        ...

    @abstractmethod
    def set(self, key: int, value: Any) -> None:
        """Cache a sample."""
        ...

    @abstractmethod
    def clear(self) -> None:
        """Clear all cached samples."""
        ...


class NoOpCache(BaseCache):
    """Cache that doesn't cache (passthrough)."""

    def get(self, key: int) -> Any | None:
        self._misses += 1
        return None

    def set(self, key: int, value: Any) -> None:
        pass

    def clear(self) -> None:
        pass
