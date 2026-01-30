"""Tests for ml/reader.py - VolumeReader thread-safe volume reading."""

from __future__ import annotations

import concurrent.futures
from typing import TYPE_CHECKING

import numpy as np
import pytest

from radiobject.ml.reader import VolumeReader

if TYPE_CHECKING:
    from radiobject.volume_collection import VolumeCollection


class TestVolumeReaderBasic:
    """Tests for VolumeReader basic operations."""

    def test_reader_init(self, populated_collection_module: "VolumeCollection") -> None:
        reader = VolumeReader(populated_collection_module)

        assert len(reader) == 3
        assert reader.shape == (240, 240, 155)

    def test_reader_obs_ids(self, populated_collection_module: "VolumeCollection") -> None:
        reader = VolumeReader(populated_collection_module)

        obs_id = reader.get_obs_id(0)
        assert isinstance(obs_id, str)
        assert len(obs_id) > 0


class TestVolumeReaderReadFull:
    """Tests for VolumeReader.read_full."""

    def test_read_full_returns_array(self, populated_collection_module: "VolumeCollection") -> None:
        reader = VolumeReader(populated_collection_module)
        data = reader.read_full(0)

        assert isinstance(data, np.ndarray)
        assert data.shape == (240, 240, 155)

    def test_read_full_all_volumes(self, populated_collection_module: "VolumeCollection") -> None:
        reader = VolumeReader(populated_collection_module)

        for i in range(len(reader)):
            data = reader.read_full(i)
            assert data.shape == reader.shape


class TestVolumeReaderReadPatch:
    """Tests for VolumeReader.read_patch."""

    def test_read_patch_basic(self, populated_collection_module: "VolumeCollection") -> None:
        reader = VolumeReader(populated_collection_module)
        patch = reader.read_patch(0, start=(0, 0, 0), size=(64, 64, 64))

        assert patch.shape == (64, 64, 64)

    def test_read_patch_offset(self, populated_collection_module: "VolumeCollection") -> None:
        reader = VolumeReader(populated_collection_module)
        patch = reader.read_patch(0, start=(10, 20, 30), size=(32, 32, 32))

        assert patch.shape == (32, 32, 32)

    def test_read_patch_matches_full(self, populated_collection_module: "VolumeCollection") -> None:
        reader = VolumeReader(populated_collection_module)

        full = reader.read_full(0)
        patch = reader.read_patch(0, start=(10, 20, 30), size=(32, 32, 32))

        expected = full[10:42, 20:52, 30:62]
        np.testing.assert_array_equal(patch, expected)


class TestVolumeReaderReadSlice:
    """Tests for VolumeReader.read_slice."""

    def test_read_axial_slice(self, populated_collection_module: "VolumeCollection") -> None:
        reader = VolumeReader(populated_collection_module)
        slice_2d = reader.read_slice(0, axis=2, position=77)

        assert slice_2d.shape == (240, 240)

    def test_read_sagittal_slice(self, populated_collection_module: "VolumeCollection") -> None:
        reader = VolumeReader(populated_collection_module)
        slice_2d = reader.read_slice(0, axis=0, position=120)

        assert slice_2d.shape == (240, 155)

    def test_read_coronal_slice(self, populated_collection_module: "VolumeCollection") -> None:
        reader = VolumeReader(populated_collection_module)
        slice_2d = reader.read_slice(0, axis=1, position=120)

        assert slice_2d.shape == (240, 155)

    def test_read_slice_matches_full(self, populated_collection_module: "VolumeCollection") -> None:
        reader = VolumeReader(populated_collection_module)

        full = reader.read_full(0)
        axial = reader.read_slice(0, axis=2, position=50)

        np.testing.assert_array_equal(axial, full[:, :, 50])

    def test_read_slice_invalid_axis(self, populated_collection_module: "VolumeCollection") -> None:
        reader = VolumeReader(populated_collection_module)

        with pytest.raises(ValueError, match="axis must be 0, 1, or 2"):
            reader.read_slice(0, axis=3, position=0)


class TestVolumeReaderThreadSafety:
    """Tests for VolumeReader thread safety."""

    def test_concurrent_reads_same_volume(
        self, populated_collection_module: "VolumeCollection"
    ) -> None:
        reader = VolumeReader(populated_collection_module)
        results: list[np.ndarray] = []

        def read_volume() -> np.ndarray:
            return reader.read_full(0)

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(read_volume) for _ in range(8)]
            results = [f.result() for f in futures]

        assert len(results) == 8
        for arr in results[1:]:
            np.testing.assert_array_equal(arr, results[0])

    def test_concurrent_reads_different_volumes(
        self, populated_collection_module: "VolumeCollection"
    ) -> None:
        reader = VolumeReader(populated_collection_module)

        def read_volume(idx: int) -> tuple[int, np.ndarray]:
            return idx, reader.read_full(idx)

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(read_volume, i % 3) for i in range(9)]
            results = [f.result() for f in futures]

        assert len(results) == 9
        idx_to_arrays: dict[int, list[np.ndarray]] = {}
        for idx, arr in results:
            idx_to_arrays.setdefault(idx, []).append(arr)

        for arrays in idx_to_arrays.values():
            for arr in arrays[1:]:
                np.testing.assert_array_equal(arr, arrays[0])

    def test_concurrent_patch_reads(self, populated_collection_module: "VolumeCollection") -> None:
        reader = VolumeReader(populated_collection_module)

        def read_patch(idx: int) -> np.ndarray:
            start = (idx * 10, idx * 10, idx * 10)
            return reader.read_patch(0, start=start, size=(32, 32, 32))

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(read_patch, i % 5) for i in range(10)]
            results = [f.result() for f in futures]

        assert len(results) == 10
        for arr in results:
            assert arr.shape == (32, 32, 32)


class TestVolumeReaderContextCaching:
    """Tests for VolumeReader context caching behavior."""

    def test_context_reused_same_reader(
        self, populated_collection_module: "VolumeCollection"
    ) -> None:
        reader = VolumeReader(populated_collection_module)

        _ = reader.read_full(0)
        _ = reader.read_full(1)
        _ = reader.read_full(2)

    def test_multiple_readers_same_collection(
        self, populated_collection_module: "VolumeCollection"
    ) -> None:
        reader1 = VolumeReader(populated_collection_module)
        reader2 = VolumeReader(populated_collection_module)

        data1 = reader1.read_full(0)
        data2 = reader2.read_full(0)

        np.testing.assert_array_equal(data1, data2)
