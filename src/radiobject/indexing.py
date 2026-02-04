"""Shared indexing utilities for RadiObject entities."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class Index:
    """Bidirectional index mapping string keys to integer positions.

    Supports set algebra, positional selection, and alignment checks.
    """

    keys: tuple[str, ...]
    key_to_idx: dict[str, int]
    name: str = ""

    @classmethod
    def build(cls, keys: list[str], name: str = "") -> Index:
        """Build an index from a list of string keys."""
        if len(keys) != len(set(keys)):
            duplicates = [k for k, count in Counter(keys).items() if count > 1]
            raise ValueError(f"Duplicate keys detected: {duplicates[:5]}")
        return cls(
            keys=tuple(keys),
            key_to_idx={key: idx for idx, key in enumerate(keys)},
            name=name,
        )

    # -- Lookups ---------------------------------------------------------------

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

    # -- Set operations --------------------------------------------------------

    @staticmethod
    def _as_set(other: Index | frozenset[str]) -> frozenset[str] | dict[str, int]:
        """Normalize Index | frozenset to a set-like object supporting `in`."""
        return other.key_to_idx if isinstance(other, Index) else other

    def intersection(self, other: Index | frozenset[str]) -> Index:
        """Keys present in both, preserving order from self."""
        lookup = self._as_set(other)
        return Index.build([k for k in self.keys if k in lookup], name=self.name)

    def union(self, other: Index) -> Index:
        """Keys from self then any new keys from other."""
        seen = set(self.keys)
        extra = [k for k in other.keys if k not in seen]
        return Index.build(list(self.keys) + extra, name=self.name)

    def difference(self, other: Index | frozenset[str]) -> Index:
        """Keys in self but not in other, preserving order."""
        lookup = self._as_set(other)
        return Index.build([k for k in self.keys if k not in lookup], name=self.name)

    def symmetric_difference(self, other: Index) -> Index:
        """Keys in exactly one of self or other (self-order first, then other-order)."""
        self_set = set(self.keys)
        other_set = set(other.keys)
        result = [k for k in self.keys if k not in other_set]
        result += [k for k in other.keys if k not in self_set]
        return Index.build(result, name=self.name)

    def __and__(self, other: Index | frozenset[str]) -> Index:
        return self.intersection(other)

    def __or__(self, other: Index) -> Index:
        return self.union(other)

    def __sub__(self, other: Index | frozenset[str]) -> Index:
        return self.difference(other)

    def __xor__(self, other: Index) -> Index:
        return self.symmetric_difference(other)

    # -- Positional selection --------------------------------------------------

    def take(self, indices: Sequence[int]) -> Index:
        """Select keys by integer positions."""
        n = len(self.keys)
        for i in indices:
            if i < 0 or i >= n:
                raise IndexError(f"Index {i} out of range [0, {n})")
        return Index.build([self.keys[i] for i in indices], name=self.name)

    def mask(self, bool_array: npt.NDArray[np.bool_]) -> Index:
        """Select keys where bool_array is True."""
        if len(bool_array) != len(self.keys):
            raise ValueError(
                f"Boolean array length {len(bool_array)} != index length {len(self.keys)}"
            )
        return Index.build(
            [k for k, m in zip(self.keys, bool_array) if m],
            name=self.name,
        )

    # -- Alignment / comparison ------------------------------------------------

    def is_aligned(self, other: Index) -> bool:
        """True if same keys in the same order."""
        return self.keys == other.keys

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Index):
            return NotImplemented
        return self.keys == other.keys and self.name == other.name

    def __hash__(self) -> int:
        return hash((self.keys, self.name))

    def __le__(self, other: Index) -> bool:
        """True if self is a subset of other."""
        return set(self.keys) <= set(other.keys)

    def __ge__(self, other: Index) -> bool:
        """True if self is a superset of other."""
        return set(self.keys) >= set(other.keys)

    # -- Conversions / display -------------------------------------------------

    def to_set(self) -> frozenset[str]:
        return frozenset(self.keys)

    def to_list(self) -> list[str]:
        return list(self.keys)

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys)

    def __repr__(self) -> str:
        if self.name:
            return f"Index('{self.name}', {len(self.keys)} keys)"
        return f"Index({len(self.keys)} keys)"


def align(*indexes: Index) -> Index:
    """Intersection of multiple indexes (first-index order)."""
    if not indexes:
        return Index.build([])
    result = indexes[0]
    for idx in indexes[1:]:
        result = result.intersection(idx)
    return result
