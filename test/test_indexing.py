"""Tests for radiobject/indexing.py - Index class for bidirectional key/position mapping."""

from __future__ import annotations

import pytest

from radiobject.indexing import Index


class TestIndexBuild:
    """Tests for Index.build factory method."""

    def test_build_from_keys(self) -> None:
        keys = ["a", "b", "c"]
        index = Index.build(keys)

        assert len(index) == 3
        assert index.keys == ("a", "b", "c")

    def test_build_empty_index(self) -> None:
        index = Index.build([])

        assert len(index) == 0
        assert index.keys == ()

    def test_build_single_key(self) -> None:
        index = Index.build(["only"])

        assert len(index) == 1
        assert index.keys == ("only",)

    def test_build_rejects_duplicates(self) -> None:
        with pytest.raises(ValueError, match="Duplicate keys detected"):
            Index.build(["a", "b", "a"])

    def test_build_rejects_multiple_duplicates(self) -> None:
        with pytest.raises(ValueError, match="Duplicate keys detected"):
            Index.build(["x", "x", "y", "y", "z"])

    def test_build_preserves_order(self) -> None:
        keys = ["c", "a", "b"]
        index = Index.build(keys)

        assert index.keys == ("c", "a", "b")


class TestIndexGetIndex:
    """Tests for Index.get_index (key -> position)."""

    def test_get_index_first(self) -> None:
        index = Index.build(["a", "b", "c"])
        assert index.get_index("a") == 0

    def test_get_index_middle(self) -> None:
        index = Index.build(["a", "b", "c"])
        assert index.get_index("b") == 1

    def test_get_index_last(self) -> None:
        index = Index.build(["a", "b", "c"])
        assert index.get_index("c") == 2

    def test_get_index_not_found(self) -> None:
        index = Index.build(["a", "b", "c"])
        with pytest.raises(KeyError, match="'d' not found in index"):
            index.get_index("d")

    def test_get_index_empty_index(self) -> None:
        index = Index.build([])
        with pytest.raises(KeyError, match="'a' not found in index"):
            index.get_index("a")


class TestIndexGetKey:
    """Tests for Index.get_key (position -> key)."""

    def test_get_key_first(self) -> None:
        index = Index.build(["a", "b", "c"])
        assert index.get_key(0) == "a"

    def test_get_key_last(self) -> None:
        index = Index.build(["a", "b", "c"])
        assert index.get_key(2) == "c"

    def test_get_key_negative_index_raises(self) -> None:
        index = Index.build(["a", "b", "c"])
        with pytest.raises(IndexError, match="Index -1 out of range"):
            index.get_key(-1)

    def test_get_key_out_of_bounds(self) -> None:
        index = Index.build(["a", "b", "c"])
        with pytest.raises(IndexError, match="Index 3 out of range"):
            index.get_key(3)

    def test_get_key_empty_index(self) -> None:
        index = Index.build([])
        with pytest.raises(IndexError, match="Index 0 out of range"):
            index.get_key(0)


class TestIndexContains:
    """Tests for Index.__contains__ (in operator)."""

    def test_contains_existing_key(self) -> None:
        index = Index.build(["a", "b", "c"])
        assert "a" in index
        assert "b" in index
        assert "c" in index

    def test_contains_missing_key(self) -> None:
        index = Index.build(["a", "b", "c"])
        assert "d" not in index
        assert "" not in index

    def test_contains_empty_index(self) -> None:
        index = Index.build([])
        assert "a" not in index


class TestIndexLen:
    """Tests for Index.__len__."""

    def test_len_empty(self) -> None:
        assert len(Index.build([])) == 0

    def test_len_single(self) -> None:
        assert len(Index.build(["a"])) == 1

    def test_len_multiple(self) -> None:
        assert len(Index.build(["a", "b", "c", "d", "e"])) == 5


class TestIndexImmutability:
    """Tests for Index frozen dataclass behavior."""

    def test_index_is_frozen(self) -> None:
        index = Index.build(["a", "b", "c"])
        with pytest.raises(AttributeError):
            index.keys = ("x", "y", "z")  # type: ignore[misc]

    def test_keys_tuple_immutable(self) -> None:
        keys = ["a", "b", "c"]
        index = Index.build(keys)
        keys.append("d")

        assert len(index) == 3
        assert "d" not in index


class TestIndexRoundtrip:
    """Tests for bidirectional key/position lookups."""

    def test_roundtrip_key_to_idx_to_key(self) -> None:
        index = Index.build(["alpha", "beta", "gamma"])

        for key in ["alpha", "beta", "gamma"]:
            idx = index.get_index(key)
            assert index.get_key(idx) == key

    def test_roundtrip_idx_to_key_to_idx(self) -> None:
        index = Index.build(["alpha", "beta", "gamma"])

        for idx in range(3):
            key = index.get_key(idx)
            assert index.get_index(key) == idx
