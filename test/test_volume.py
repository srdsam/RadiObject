"""Tests for src/Volume.py - TileDB-backed radiology volumes."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import tiledb

from radiobject import SliceOrientation, TileConfig, WriteConfig, configure, radi_reset
from radiobject.volume import Volume

logger = logging.getLogger(__name__)


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
            radi_reset()


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


class TestVolumePerformance:
    """Performance tests with timing output. Run with --log-cli-level=INFO to see throughput."""

    def test_write_throughput_3d(self, temp_dir: Path) -> None:
        shape = (128, 128, 64)
        data = np.random.rand(*shape).astype(np.float32)
        size_mb = data.nbytes / (1024 * 1024)
        uri = str(temp_dir / "perf_write_3d")

        start = time.perf_counter()
        Volume.from_numpy(uri, data)
        duration = time.perf_counter() - start

        throughput = size_mb / duration
        logger.info("Write 3D: %.2f MB in %.3fs (%.2f MB/s)", size_mb, duration, throughput)
        assert duration < 30

    def test_read_throughput_3d(self, temp_dir: Path) -> None:
        shape = (128, 128, 64)
        data = np.random.rand(*shape).astype(np.float32)
        size_mb = data.nbytes / (1024 * 1024)
        uri = str(temp_dir / "perf_read_3d")
        Volume.from_numpy(uri, data)
        vol = Volume(uri)

        start = time.perf_counter()
        _ = vol.to_numpy()
        duration = time.perf_counter() - start

        throughput = size_mb / duration
        logger.info("Read 3D: %.2f MB in %.3fs (%.2f MB/s)", size_mb, duration, throughput)
        assert duration < 30

    def test_axial_slice_throughput(self, temp_dir: Path) -> None:
        shape = (128, 128, 64)
        data = np.random.rand(*shape).astype(np.float32)
        uri = str(temp_dir / "perf_slice")
        Volume.from_numpy(uri, data)
        vol = Volume(uri)
        slice_count = 64
        slice_size_mb = (128 * 128 * 4) / (1024 * 1024)

        start = time.perf_counter()
        for z in range(slice_count):
            _ = vol.axial(z)
        duration = time.perf_counter() - start

        total_mb = slice_size_mb * slice_count
        throughput = total_mb / duration
        logger.info(
            "Axial slices (%dx): %.2f MB in %.3fs (%.2f MB/s)",
            slice_count,
            total_mb,
            duration,
            throughput,
        )
        assert duration < 60

    def test_write_throughput_4d(self, temp_dir: Path) -> None:
        shape = (128, 128, 64, 4)
        data = np.random.rand(*shape).astype(np.float32)
        size_mb = data.nbytes / (1024 * 1024)
        uri = str(temp_dir / "perf_write_4d")

        start = time.perf_counter()
        Volume.from_numpy(uri, data)
        duration = time.perf_counter() - start

        throughput = size_mb / duration
        logger.info("Write 4D: %.2f MB in %.3fs (%.2f MB/s)", size_mb, duration, throughput)
        assert duration < 60


class TestDtypePreservation:
    """Tests for comprehensive dtype preservation across all numpy types."""

    @pytest.mark.parametrize(
        "dtype",
        [
            np.uint8,
            np.int8,
            np.uint16,
            np.int16,
            np.int32,
            np.int64,
            np.float32,
            np.float64,
        ],
    )
    def test_dtype_roundtrip(self, temp_dir: Path, dtype: type) -> None:
        """Test that all numpy dtypes survive roundtrip through TileDB."""
        uri = str(temp_dir / f"vol_{dtype.__name__}")
        shape = (16, 16, 8)

        # Create test data with values appropriate for dtype
        if np.issubdtype(dtype, np.integer):
            info = np.iinfo(dtype)
            # Use range that fits in dtype and has diverse values
            data = np.random.randint(
                max(info.min, -1000),
                min(info.max, 1000),
                size=shape,
                dtype=dtype,
            )
        else:
            data = np.random.rand(*shape).astype(dtype)

        vol = Volume.from_numpy(uri, data)

        assert vol.dtype == dtype, f"Expected dtype {dtype}, got {vol.dtype}"

        recovered = vol.to_numpy()
        if np.issubdtype(dtype, np.integer):
            np.testing.assert_array_equal(recovered, data)
        else:
            np.testing.assert_array_almost_equal(recovered, data, decimal=4)

    def test_uint16_max_value_preserved(self, temp_dir: Path) -> None:
        """Test uint16 max value (65535) is preserved exactly."""
        uri = str(temp_dir / "vol_uint16_max")
        data = np.array([[[0, 65535], [32768, 1]]], dtype=np.uint16)

        vol = Volume.from_numpy(uri, data)
        recovered = vol.to_numpy()

        np.testing.assert_array_equal(recovered, data)
        assert recovered.max() == 65535

    def test_int16_negative_preserved(self, temp_dir: Path) -> None:
        """Test int16 negative values are preserved exactly."""
        uri = str(temp_dir / "vol_int16_neg")
        data = np.array([[[-32768, 32767], [-1, 0]]], dtype=np.int16)

        vol = Volume.from_numpy(uri, data)
        recovered = vol.to_numpy()

        np.testing.assert_array_equal(recovered, data)
        assert recovered.min() == -32768
        assert recovered.max() == 32767


class TestFloatingPointEdgeCases:
    """Tests for floating-point special values preservation."""

    def test_nan_preserved(self, temp_dir: Path) -> None:
        """Test NaN values are preserved through roundtrip."""
        uri = str(temp_dir / "vol_nan")
        data = np.array([[[1.0, np.nan], [np.nan, 2.0]]], dtype=np.float32)

        vol = Volume.from_numpy(uri, data)
        recovered = vol.to_numpy()

        # Check finite values match
        finite_mask = np.isfinite(data)
        np.testing.assert_array_equal(recovered[finite_mask], data[finite_mask])

        # Check NaN positions match
        np.testing.assert_array_equal(np.isnan(recovered), np.isnan(data))

    def test_inf_preserved(self, temp_dir: Path) -> None:
        """Test Inf values are preserved through roundtrip."""
        uri = str(temp_dir / "vol_inf")
        data = np.array([[[np.inf, -np.inf], [1.0, -np.inf]]], dtype=np.float32)

        vol = Volume.from_numpy(uri, data)
        recovered = vol.to_numpy()

        np.testing.assert_array_equal(np.isinf(recovered), np.isinf(data))
        np.testing.assert_array_equal(
            np.sign(recovered[np.isinf(recovered)]), np.sign(data[np.isinf(data)])
        )

    def test_denormalized_float_preserved(self, temp_dir: Path) -> None:
        """Test denormalized (subnormal) floats are preserved."""
        uri = str(temp_dir / "vol_denorm")
        # Smallest positive denormalized float32
        denorm = np.finfo(np.float32).tiny / 2
        data = np.array([[[denorm, 1.0], [0.0, denorm]]], dtype=np.float32)

        vol = Volume.from_numpy(uri, data)
        recovered = vol.to_numpy()

        np.testing.assert_array_almost_equal(recovered, data, decimal=45)

    def test_mixed_special_values(self, temp_dir: Path) -> None:
        """Test volume with mixed NaN, Inf, and regular values."""
        uri = str(temp_dir / "vol_mixed")
        data = np.array(
            [
                [[np.nan, np.inf, -np.inf], [0.0, 1.0, -1.0]],
                [[1e-38, 1e38, np.nan], [np.inf, -np.inf, 0.0]],
            ],
            dtype=np.float32,
        )

        vol = Volume.from_numpy(uri, data)
        recovered = vol.to_numpy()

        # Check NaN positions
        np.testing.assert_array_equal(np.isnan(recovered), np.isnan(data))
        # Check Inf positions and signs
        np.testing.assert_array_equal(np.isinf(recovered), np.isinf(data))
        # Check finite values
        finite_mask = np.isfinite(data)
        np.testing.assert_array_almost_equal(recovered[finite_mask], data[finite_mask])


class TestNiftiExportRoundtrip:
    """Tests for NIfTI export fidelity."""

    def test_nifti_affine_roundtrip(self, temp_dir: Path, nifti_3d_path: Path) -> None:
        """Test affine matrix is preserved through TileDB roundtrip."""
        # Load original NIfTI
        original_img = nib.load(nifti_3d_path)
        original_affine = original_img.affine.copy()

        # Store in TileDB
        uri = str(temp_dir / "nifti_roundtrip")
        vol = Volume.from_nifti(uri, nifti_3d_path, reorient=False)

        # Export back to NIfTI
        export_path = temp_dir / "exported.nii.gz"
        vol.to_nifti(export_path)

        # Load exported and compare affine
        exported_img = nib.load(export_path)
        exported_affine = exported_img.affine

        np.testing.assert_array_almost_equal(
            exported_affine,
            original_affine,
            decimal=6,
            err_msg="Affine matrix precision loss exceeds 1e-6",
        )

    def test_nifti_header_fields_roundtrip(self, temp_dir: Path, nifti_3d_path: Path) -> None:
        """Test NIfTI header fields are preserved through roundtrip."""
        original_img = nib.load(nifti_3d_path)
        original_header = original_img.header

        uri = str(temp_dir / "nifti_header_roundtrip")
        vol = Volume.from_nifti(uri, nifti_3d_path, reorient=False)

        export_path = temp_dir / "exported_header.nii.gz"
        vol.to_nifti(export_path)

        exported_img = nib.load(export_path)
        exported_header = exported_img.header

        # Check sform/qform codes
        assert int(exported_header.get("sform_code", 0)) == int(
            original_header.get("sform_code", 0)
        )
        assert int(exported_header.get("qform_code", 0)) == int(
            original_header.get("qform_code", 0)
        )

        # Check scaling factors with tolerance
        orig_slope = float(original_header.get("scl_slope", 1.0))
        exp_slope = float(exported_header.get("scl_slope", 1.0))
        assert abs(orig_slope - exp_slope) < 1e-6 or (np.isnan(orig_slope) and np.isnan(exp_slope))

        orig_inter = float(original_header.get("scl_inter", 0.0))
        exp_inter = float(exported_header.get("scl_inter", 0.0))
        assert abs(orig_inter - exp_inter) < 1e-6 or (np.isnan(orig_inter) and np.isnan(exp_inter))

    def test_nifti_data_roundtrip(self, temp_dir: Path, nifti_3d_path: Path) -> None:
        """Test voxel data is preserved through NIfTI roundtrip."""
        original_img = nib.load(nifti_3d_path)
        original_data = np.asarray(original_img.dataobj)

        uri = str(temp_dir / "nifti_data_roundtrip")
        vol = Volume.from_nifti(uri, nifti_3d_path, reorient=False)

        export_path = temp_dir / "exported_data.nii.gz"
        vol.to_nifti(export_path)

        exported_img = nib.load(export_path)
        exported_data = np.asarray(exported_img.dataobj)

        np.testing.assert_array_almost_equal(
            exported_data,
            original_data,
            decimal=5,
            err_msg="Voxel data changed through NIfTI roundtrip",
        )


class TestDicomDtypeHandling:
    """Tests for DICOM dtype handling."""

    def test_from_dicom_preserves_original_dtype(
        self, temp_dir: Path, sample_dicom_series: Path
    ) -> None:
        """Test from_dicom with dtype=None preserves original DICOM dtype."""
        uri = str(temp_dir / "dicom_orig_dtype")
        vol = Volume.from_dicom(uri, sample_dicom_series, dtype=None)

        # DICOM is typically uint16 or int16
        assert vol.dtype in (np.uint16, np.int16), f"Unexpected DICOM dtype: {vol.dtype}"

    def test_from_dicom_explicit_float32(self, temp_dir: Path, sample_dicom_series: Path) -> None:
        """Test from_dicom with explicit float32 conversion."""
        uri = str(temp_dir / "dicom_float32")
        vol = Volume.from_dicom(uri, sample_dicom_series, dtype=np.float32)

        assert vol.dtype == np.float32

    def test_from_dicom_explicit_float64(self, temp_dir: Path, sample_dicom_series: Path) -> None:
        """Test from_dicom with explicit float64 conversion."""
        uri = str(temp_dir / "dicom_float64")
        vol = Volume.from_dicom(uri, sample_dicom_series, dtype=np.float64)

        assert vol.dtype == np.float64


class TestLargeVolumeIntegrity:
    """Tests for large volume data integrity."""

    def test_large_volume_integrity(self, temp_dir: Path) -> None:
        """Test 256x256x256 volume integrity across tile boundaries."""
        uri = str(temp_dir / "large_vol")
        shape = (256, 256, 256)
        rng = np.random.default_rng(42)
        data = rng.random(shape, dtype=np.float32)

        vol = Volume.from_numpy(uri, data)
        recovered = vol.to_numpy()

        # Verify exact shape
        assert recovered.shape == shape

        # Verify data integrity
        np.testing.assert_array_almost_equal(recovered, data)

        # Verify corners (most likely to have tile boundary issues)
        assert recovered[0, 0, 0] == data[0, 0, 0]
        assert recovered[-1, -1, -1] == data[-1, -1, -1]
        assert recovered[0, 0, -1] == data[0, 0, -1]
        assert recovered[-1, 0, 0] == data[-1, 0, 0]

    def test_large_volume_partial_read(self, temp_dir: Path) -> None:
        """Test partial reads from large volume span tile boundaries correctly."""
        uri = str(temp_dir / "large_vol_partial")
        shape = (256, 256, 256)
        rng = np.random.default_rng(42)
        data = rng.random(shape, dtype=np.float32)

        vol = Volume.from_numpy(uri, data)

        # Read a region that spans multiple tiles
        partial = vol[100:200, 100:200, 100:200]

        np.testing.assert_array_almost_equal(partial, data[100:200, 100:200, 100:200])
