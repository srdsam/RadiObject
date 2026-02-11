"""Tests for src/Volume.py - TileDB-backed radiology volumes."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import tiledb

from radiobject import (
    SliceOrientation,
    TileConfig,
    Volume,
    WriteConfig,
    configure,
    reset_radiobject_config,
)


class TestVolumeConstruction:
    """Tests for Volume.create and basic properties."""

    def test_create_empty_3d_volume(self, volume_uri: str) -> None:
        shape = (64, 64, 32)
        vol = Volume.create(volume_uri, shape=shape, dtype=np.float32)

        assert vol.shape == shape
        assert vol.ndim == 3
        assert vol.dtype == np.float32

    def test_create_empty_4d_volume(self, temp_dir: Path) -> None:
        uri = str(temp_dir / "vol_4d")
        shape = (64, 64, 32, 8)
        vol = Volume.create(uri, shape=shape, dtype=np.float32)

        assert vol.shape == shape
        assert vol.ndim == 4

    def test_create_with_int16_dtype(self, temp_dir: Path) -> None:
        uri = str(temp_dir / "vol_int16")
        shape = (32, 32, 16)
        vol = Volume.create(uri, shape=shape, dtype=np.int16)

        assert vol.dtype == np.int16

    def test_create_rejects_invalid_dimensions(self, temp_dir: Path) -> None:
        uri_2d = str(temp_dir / "vol_2d")
        uri_5d = str(temp_dir / "vol_5d")

        with pytest.raises(ValueError, match="3D or 4D"):
            Volume.create(uri_2d, shape=(64, 64))

        with pytest.raises(ValueError, match="3D or 4D"):
            Volume.create(uri_5d, shape=(64, 64, 32, 8, 2))


class TestVolumeTileOrientation:
    """Tests for tile orientation metadata storage and retrieval."""

    def test_tile_orientation_stored_on_create(self, volume_uri: str) -> None:
        vol = Volume.create(volume_uri, shape=(32, 32, 16))
        assert vol.tile_orientation == SliceOrientation.AXIAL

    def test_tile_orientation_from_config(self, temp_dir: Path) -> None:
        configure(write=WriteConfig(tile=TileConfig(orientation=SliceOrientation.SAGITTAL)))
        try:
            uri = str(temp_dir / "vol_sagittal")
            vol = Volume.create(uri, shape=(32, 32, 16))
            assert vol.tile_orientation == SliceOrientation.SAGITTAL
        finally:
            reset_radiobject_config()


class TestVolumeFromNumpy:
    """Tests for Volume.from_numpy factory method."""

    def test_from_numpy_3d(self, volume_uri: str, array_3d: np.ndarray) -> None:
        vol = Volume.from_numpy(volume_uri, array_3d)

        assert vol.shape == array_3d.shape
        np.testing.assert_array_almost_equal(vol.to_numpy(), array_3d)

    def test_from_numpy_4d(self, temp_dir: Path, array_4d: np.ndarray) -> None:
        uri = str(temp_dir / "vol_4d")
        vol = Volume.from_numpy(uri, array_4d)

        assert vol.shape == array_4d.shape
        np.testing.assert_array_almost_equal(vol.to_numpy(), array_4d)

    def test_from_numpy_preserves_dtype(self, temp_dir: Path) -> None:
        uri = str(temp_dir / "vol_int16")
        data = np.random.randint(0, 1000, size=(32, 32, 16), dtype=np.int16)
        vol = Volume.from_numpy(uri, data)

        assert vol.dtype == np.int16
        np.testing.assert_array_equal(vol.to_numpy(), data)


class TestVolumeFromNifti:
    """Tests for Volume.from_nifti factory method."""

    def test_from_nifti_3d(self, temp_dir: Path, nifti_3d_path: Path) -> None:
        uri = str(temp_dir / "vol_from_nifti_3d")
        vol = Volume.from_nifti(uri, nifti_3d_path)

        assert vol.ndim == 3
        img = nib.load(nifti_3d_path)
        assert vol.shape == img.shape

    def test_from_nifti_4d(self, temp_dir: Path, nifti_4d_path: Path) -> None:
        uri = str(temp_dir / "vol_from_nifti_4d")
        vol = Volume.from_nifti(uri, nifti_4d_path)

        assert vol.ndim == 4
        img = nib.load(nifti_4d_path)
        assert vol.shape == img.shape

    def test_from_nifti_accepts_path_object(self, temp_dir: Path, nifti_3d_path: Path) -> None:
        uri = str(temp_dir / "vol_from_path")
        vol = Volume.from_nifti(uri, nifti_3d_path)

        img = nib.load(nifti_3d_path)
        assert vol.shape == img.shape


class TestVolumeSlicing:
    """Tests for axial, sagittal, coronal, and generic slice methods."""

    def test_axial_slice_3d(self, temp_dir: Path, array_3d: np.ndarray) -> None:
        uri = str(temp_dir / "slice_vol")
        vol = Volume.from_numpy(uri, array_3d)

        z_idx = array_3d.shape[2] // 2
        axial_slice = vol.axial(z_idx)

        expected = array_3d[:, :, z_idx]
        assert axial_slice.shape == (array_3d.shape[0], array_3d.shape[1])
        np.testing.assert_array_almost_equal(axial_slice, expected)

    def test_sagittal_slice_3d(self, temp_dir: Path, array_3d: np.ndarray) -> None:
        uri = str(temp_dir / "sagittal_vol")
        vol = Volume.from_numpy(uri, array_3d)

        x_idx = array_3d.shape[0] // 2
        sagittal_slice = vol.sagittal(x_idx)

        expected = array_3d[x_idx, :, :]
        assert sagittal_slice.shape == (array_3d.shape[1], array_3d.shape[2])
        np.testing.assert_array_almost_equal(sagittal_slice, expected)

    def test_coronal_slice_3d(self, temp_dir: Path, array_3d: np.ndarray) -> None:
        uri = str(temp_dir / "coronal_vol")
        vol = Volume.from_numpy(uri, array_3d)

        y_idx = array_3d.shape[1] // 2
        coronal_slice = vol.coronal(y_idx)

        expected = array_3d[:, y_idx, :]
        assert coronal_slice.shape == (array_3d.shape[0], array_3d.shape[2])
        np.testing.assert_array_almost_equal(coronal_slice, expected)

    def test_axial_slice_4d_with_t(self, temp_dir: Path, array_4d: np.ndarray) -> None:
        uri = str(temp_dir / "slice_vol_4d")
        vol = Volume.from_numpy(uri, array_4d)

        z_idx = array_4d.shape[2] // 2
        t_idx = 2
        axial_slice = vol.axial(z_idx, t=t_idx)

        expected = array_4d[:, :, z_idx, t_idx]
        assert axial_slice.shape == (array_4d.shape[0], array_4d.shape[1])
        np.testing.assert_array_almost_equal(axial_slice, expected)

    def test_partial_slice(self, temp_dir: Path, array_3d: np.ndarray) -> None:
        uri = str(temp_dir / "partial_vol")
        vol = Volume.from_numpy(uri, array_3d)

        x_end = min(16, array_3d.shape[0])
        y_start, y_end = 8, min(24, array_3d.shape[1])
        z_start, z_end = 4, min(12, array_3d.shape[2])

        result = vol.slice(slice(0, x_end), slice(y_start, y_end), slice(z_start, z_end))
        expected = array_3d[0:x_end, y_start:y_end, z_start:z_end]
        assert result.shape == expected.shape
        np.testing.assert_array_almost_equal(result, expected)

    def test_partial_slice_4d_with_t(self, temp_dir: Path, array_4d: np.ndarray) -> None:
        uri = str(temp_dir / "partial_vol_4d")
        vol = Volume.from_numpy(uri, array_4d)

        x_end = min(16, array_4d.shape[0])
        y_start, y_end = 8, min(24, array_4d.shape[1])
        z_start, z_end = 4, min(12, array_4d.shape[2])

        result = vol.slice(
            slice(0, x_end), slice(y_start, y_end), slice(z_start, z_end), slice(1, 3)
        )
        expected = array_4d[0:x_end, y_start:y_end, z_start:z_end, 1:3]
        assert result.shape == expected.shape
        np.testing.assert_array_almost_equal(result, expected)


class TestVolumeToNumpy:
    """Tests for to_numpy roundtrip."""

    def test_to_numpy_3d_roundtrip(self, temp_dir: Path, array_3d: np.ndarray) -> None:
        uri = str(temp_dir / "roundtrip_3d")
        vol = Volume.from_numpy(uri, array_3d)
        result = vol.to_numpy()

        assert result.shape == array_3d.shape
        np.testing.assert_array_almost_equal(result, array_3d)

    def test_to_numpy_4d_roundtrip(self, temp_dir: Path, array_4d: np.ndarray) -> None:
        uri = str(temp_dir / "roundtrip_4d")
        vol = Volume.from_numpy(uri, array_4d)
        result = vol.to_numpy()

        assert result.shape == array_4d.shape
        np.testing.assert_array_almost_equal(result, array_4d)


class TestVolumeCustomContext:
    """Tests for custom tiledb.Ctx parameter."""

    def test_create_with_custom_ctx(self, temp_dir: Path, custom_tiledb_ctx: tiledb.Ctx) -> None:
        uri = str(temp_dir / "vol_custom_ctx")
        vol = Volume.create(uri, shape=(32, 32, 16), ctx=custom_tiledb_ctx)

        assert vol._ctx is custom_tiledb_ctx
        assert vol.shape == (32, 32, 16)

    def test_from_numpy_with_custom_ctx(
        self, temp_dir: Path, custom_tiledb_ctx: tiledb.Ctx, array_3d: np.ndarray
    ) -> None:
        uri = str(temp_dir / "vol_custom_ctx_numpy")
        vol = Volume.from_numpy(uri, array_3d, ctx=custom_tiledb_ctx)

        assert vol._ctx is custom_tiledb_ctx
        np.testing.assert_array_almost_equal(vol.to_numpy(), array_3d)
