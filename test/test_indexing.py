"""Tests for radiobject/indexing.py - Index class for bidirectional key/position mapping."""

from __future__ import annotations

import numpy as np
import pytest

from radiobject.indexing import Index, align


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

    def test_build_with_name(self) -> None:
        index = Index.build(["a", "b"], name="obs_id")
        assert index.name == "obs_id"

    def test_build_default_name_empty(self) -> None:
        index = Index.build(["a"])
        assert index.name == ""


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


# =============================================================================
# Set operations
# =============================================================================


class TestIndexSetOperations:
    """Tests for intersection, union, difference, symmetric_difference."""

    def test_intersection_with_index(self) -> None:
        a = Index.build(["a", "b", "c", "d"], name="test")
        b = Index.build(["c", "a", "e"])
        result = a & b
        assert result.keys == ("a", "c")
        assert result.name == "test"

    def test_intersection_with_frozenset(self) -> None:
        a = Index.build(["a", "b", "c", "d"])
        result = a & frozenset({"b", "d", "f"})
        assert result.keys == ("b", "d")

    def test_intersection_empty_result(self) -> None:
        a = Index.build(["a", "b"])
        b = Index.build(["c", "d"])
        assert len(a & b) == 0

    def test_intersection_identical(self) -> None:
        a = Index.build(["a", "b", "c"])
        b = Index.build(["a", "b", "c"])
        assert (a & b).keys == ("a", "b", "c")

    def test_union(self) -> None:
        a = Index.build(["a", "b", "c"], name="test")
        b = Index.build(["c", "d", "e"])
        result = a | b
        assert result.keys == ("a", "b", "c", "d", "e")
        assert result.name == "test"

    def test_union_no_overlap(self) -> None:
        a = Index.build(["a", "b"])
        b = Index.build(["c", "d"])
        assert (a | b).keys == ("a", "b", "c", "d")

    def test_union_identical(self) -> None:
        a = Index.build(["a", "b"])
        b = Index.build(["a", "b"])
        assert (a | b).keys == ("a", "b")

    def test_difference_with_index(self) -> None:
        a = Index.build(["a", "b", "c", "d"], name="test")
        b = Index.build(["b", "d"])
        result = a - b
        assert result.keys == ("a", "c")
        assert result.name == "test"

    def test_difference_with_frozenset(self) -> None:
        a = Index.build(["a", "b", "c"])
        result = a - frozenset({"b"})
        assert result.keys == ("a", "c")

    def test_difference_no_overlap(self) -> None:
        a = Index.build(["a", "b"])
        b = Index.build(["c", "d"])
        assert (a - b).keys == ("a", "b")

    def test_symmetric_difference(self) -> None:
        a = Index.build(["a", "b", "c"], name="test")
        b = Index.build(["b", "c", "d"])
        result = a ^ b
        assert result.keys == ("a", "d")
        assert result.name == "test"

    def test_symmetric_difference_no_overlap(self) -> None:
        a = Index.build(["a", "b"])
        b = Index.build(["c", "d"])
        assert (a ^ b).keys == ("a", "b", "c", "d")

    def test_symmetric_difference_identical(self) -> None:
        a = Index.build(["a", "b"])
        b = Index.build(["a", "b"])
        assert len(a ^ b) == 0

    def test_set_ops_preserve_self_order(self) -> None:
        a = Index.build(["z", "y", "x", "w"])
        b = Index.build(["x", "z"])
        assert (a & b).keys == ("z", "x")
        assert (a - b).keys == ("y", "w")


# =============================================================================
# Positional selection
# =============================================================================


class TestIndexTake:
    """Tests for Index.take()."""

    def test_take_basic(self) -> None:
        index = Index.build(["a", "b", "c", "d"], name="test")
        result = index.take([0, 2])
        assert result.keys == ("a", "c")
        assert result.name == "test"

    def test_take_single(self) -> None:
        index = Index.build(["a", "b", "c"])
        result = index.take([1])
        assert result.keys == ("b",)

    def test_take_empty(self) -> None:
        index = Index.build(["a", "b"])
        result = index.take([])
        assert len(result) == 0

    def test_take_all(self) -> None:
        index = Index.build(["a", "b", "c"])
        result = index.take([0, 1, 2])
        assert result.keys == ("a", "b", "c")

    def test_take_out_of_bounds_raises(self) -> None:
        index = Index.build(["a", "b"])
        with pytest.raises(IndexError, match="Index 5 out of range"):
            index.take([0, 5])

    def test_take_negative_raises(self) -> None:
        index = Index.build(["a", "b"])
        with pytest.raises(IndexError, match="Index -1 out of range"):
            index.take([-1])


class TestIndexMask:
    """Tests for Index.mask()."""

    def test_mask_basic(self) -> None:
        index = Index.build(["a", "b", "c", "d"], name="test")
        mask = np.array([True, False, True, False])
        result = index.mask(mask)
        assert result.keys == ("a", "c")
        assert result.name == "test"

    def test_mask_all_true(self) -> None:
        index = Index.build(["a", "b"])
        result = index.mask(np.array([True, True]))
        assert result.keys == ("a", "b")

    def test_mask_all_false(self) -> None:
        index = Index.build(["a", "b"])
        result = index.mask(np.array([False, False]))
        assert len(result) == 0

    def test_mask_length_mismatch_raises(self) -> None:
        index = Index.build(["a", "b", "c"])
        with pytest.raises(ValueError, match="Boolean array length 2 != index length 3"):
            index.mask(np.array([True, False]))


# =============================================================================
# Alignment / comparison
# =============================================================================


class TestIndexAlignment:
    """Tests for is_aligned, __eq__, __hash__, __le__, __ge__."""

    def test_is_aligned_same_order(self) -> None:
        a = Index.build(["a", "b", "c"])
        b = Index.build(["a", "b", "c"])
        assert a.is_aligned(b)

    def test_is_aligned_different_order(self) -> None:
        a = Index.build(["a", "b", "c"])
        b = Index.build(["c", "b", "a"])
        assert not a.is_aligned(b)

    def test_is_aligned_different_keys(self) -> None:
        a = Index.build(["a", "b"])
        b = Index.build(["a", "c"])
        assert not a.is_aligned(b)

    def test_eq_same(self) -> None:
        a = Index.build(["a", "b"], name="x")
        b = Index.build(["a", "b"], name="x")
        assert a == b

    def test_eq_different_name(self) -> None:
        a = Index.build(["a", "b"], name="x")
        b = Index.build(["a", "b"], name="y")
        assert a != b

    def test_eq_different_keys(self) -> None:
        a = Index.build(["a", "b"])
        b = Index.build(["a", "c"])
        assert a != b

    def test_eq_not_index(self) -> None:
        a = Index.build(["a"])
        assert a != "not an index"

    def test_hash_consistency(self) -> None:
        a = Index.build(["a", "b"], name="x")
        b = Index.build(["a", "b"], name="x")
        assert hash(a) == hash(b)

    def test_subset(self) -> None:
        a = Index.build(["a", "b"])
        b = Index.build(["a", "b", "c"])
        assert a <= b
        assert not b <= a

    def test_superset(self) -> None:
        a = Index.build(["a", "b", "c"])
        b = Index.build(["a", "b"])
        assert a >= b
        assert not b >= a

    def test_subset_equal(self) -> None:
        a = Index.build(["a", "b"])
        b = Index.build(["a", "b"])
        assert a <= b
        assert a >= b


# =============================================================================
# Conversions / display
# =============================================================================


class TestIndexConversions:
    """Tests for to_set, to_list, __iter__, __repr__."""

    def test_to_set(self) -> None:
        index = Index.build(["a", "b", "c"])
        assert index.to_set() == frozenset({"a", "b", "c"})

    def test_to_list(self) -> None:
        index = Index.build(["c", "a", "b"])
        assert index.to_list() == ["c", "a", "b"]

    def test_iter(self) -> None:
        index = Index.build(["a", "b", "c"])
        assert list(index) == ["a", "b", "c"]

    def test_repr_with_name(self) -> None:
        index = Index.build(["a", "b", "c"], name="obs_id")
        assert repr(index) == "Index('obs_id', 3 keys)"

    def test_repr_without_name(self) -> None:
        index = Index.build(["a", "b"])
        assert repr(index) == "Index(2 keys)"


# =============================================================================
# align() standalone function
# =============================================================================


class TestAlign:
    """Tests for the align() standalone function."""

    def test_align_two_indexes(self) -> None:
        a = Index.build(["a", "b", "c", "d"])
        b = Index.build(["b", "c", "e"])
        result = align(a, b)
        assert result.keys == ("b", "c")

    def test_align_three_indexes(self) -> None:
        a = Index.build(["a", "b", "c", "d"])
        b = Index.build(["b", "c", "d", "e"])
        c = Index.build(["c", "d", "f"])
        result = align(a, b, c)
        assert result.keys == ("c", "d")

    def test_align_single_index(self) -> None:
        a = Index.build(["a", "b"])
        result = align(a)
        assert result.keys == ("a", "b")

    def test_align_no_indexes(self) -> None:
        result = align()
        assert len(result) == 0

    def test_align_disjoint(self) -> None:
        a = Index.build(["a", "b"])
        b = Index.build(["c", "d"])
        assert len(align(a, b)) == 0

    def test_align_preserves_first_order(self) -> None:
        a = Index.build(["z", "y", "x"])
        b = Index.build(["x", "z"])
        result = align(a, b)
        assert result.keys == ("z", "x")
