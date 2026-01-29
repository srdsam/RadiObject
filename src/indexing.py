"""Shared indexing utilities for RadiObject entities."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class Index:
    """Bidirectional index mapping string keys to integer positions."""

    keys: tuple[str, ...]
    key_to_idx: dict[str, int]

    @classmethod
    def build(cls, keys: list[str]) -> Index:
        """Build an index from a list of string keys."""
        if len(keys) != len(set(keys)):
            duplicates = [k for k, count in Counter(keys).items() if count > 1]
            raise ValueError(f"Duplicate keys detected: {duplicates[:5]}")
        return cls(
            keys=tuple(keys),
            key_to_idx={key: idx for idx, key in enumerate(keys)},
        )

    def __len__(self) -> int:
        return len(self.keys)

    def __contains__(self, key: str) -> bool:
        return key in self.key_to_idx

    def get_index(self, key: str) -> int:
        """Get integer index for a key. Raises KeyError if not found."""
        idx = self.key_to_idx.get(key)
        if idx is None:
            raise KeyError(f"Key '{key}' not found in index")
        return idx

    def get_key(self, idx: int) -> str:
        """Get key at integer index. Raises IndexError if out of bounds."""
        n = len(self.keys)
        if idx < 0 or idx >= n:
            raise IndexError(f"Index {idx} out of range [0, {n})")
        return self.keys[idx]
