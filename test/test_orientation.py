"""Tests for orientation detection and reorientation functionality."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import tiledb
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from radiobject.orientation import (
    OrientationInfo,
    detect_dicom_orientation,
    detect_nifti_orientation,
    is_orientation_valid,
    metadata_to_orientation_info,
    orientation_info_to_metadata,
    reorient_to_canonical,
)
from radiobject.volume import Volume

# ----- Test Helpers for Synthetic Orientation Data -----


def create_synthetic_nifti_ras(temp_dir: Path) -> Path:
    """Create a synthetic NIfTI file in RAS (canonical) orientation."""
    rng = np.random.default_rng(42)
    data = rng.random((32, 32, 16), dtype=np.float32)

    affine = np.array(
        [
            [1.0, 0.0, 0.0, -16.0],
            [0.0, 1.0, 0.0, -16.0],
            [0.0, 0.0, 1.0, -8.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    img = nib.Nifti1Image(data, affine)
    img.header.set_sform(affine, code=1)

    nifti_path = temp_dir / "synthetic_ras.nii.gz"
    nib.save(img, nifti_path)
    return nifti_path


def create_synthetic_nifti_lps(temp_dir: Path) -> Path:
    """Create a synthetic NIfTI file in LPS orientation."""
    rng = np.random.default_rng(42)
    data = rng.random((32, 32, 16), dtype=np.float32)

    affine = np.array(
        [
            [-1.0, 0.0, 0.0, 16.0],
            [0.0, -1.0, 0.0, 16.0],
            [0.0, 0.0, 1.0, -8.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    img = nib.Nifti1Image(data, affine)
    img.header.set_sform(affine, code=1)

    nifti_path = temp_dir / "synthetic_lps.nii.gz"
    nib.save(img, nifti_path)
    return nifti_path


def create_synthetic_dicom_series(temp_dir: Path) -> Path:
    """Create a synthetic DICOM series directory."""
    series_dir = temp_dir / "dicom_series"
    series_dir.mkdir()

    rng = np.random.default_rng(42)
    series_uid = generate_uid()
    study_uid = generate_uid()

    for z in range(16):
        file_path = str(series_dir / f"slice_{z:04d}.dcm")
        sop_instance_uid = generate_uid()

        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
        file_meta.MediaStorageSOPInstanceUID = sop_instance_uid
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

        ds = FileDataset(file_path, {}, file_meta=file_meta, preamble=b"\x00" * 128)

        ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
        ds.SOPInstanceUID = sop_instance_uid
        ds.SeriesInstanceUID = series_uid
        ds.StudyInstanceUID = study_uid
        ds.Modality = "CT"
        ds.PatientName = "Test^Patient"
        ds.PatientID = "TEST001"
        ds.InstanceNumber = z + 1

        ds.Rows = 32
        ds.Columns = 32
        ds.PixelSpacing = [1.0, 1.0]
        ds.SliceThickness = 1.0
        ds.ImagePositionPatient = [0.0, 0.0, float(z)]
        ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]

        ds.BitsAllocated = 16
        ds.BitsStored = 12
        ds.HighBit = 11
        ds.PixelRepresentation = 0
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"

        pixel_data = (rng.random((32, 32)) * 4095).astype(np.uint16)
        ds.PixelData = pixel_data.tobytes()

        ds.save_as(file_path)

    return series_dir


class TestOrientationDetection:
    """Tests for orientation detection from image headers."""

    def test_detect_nifti_ras_orientation(self, temp_dir: Path):
        """RAS NIfTI should be detected as canonical."""
        nifti_path = create_synthetic_nifti_ras(temp_dir)
        img = nib.load(nifti_path)
        info = detect_nifti_orientation(img)

        assert info.axcodes == ("R", "A", "S")
        assert info.is_canonical is True
        assert info.confidence == "header"
        assert info.source == "nifti_sform"

    def test_detect_nifti_lps_orientation(self, temp_dir: Path):
        """LPS NIfTI should be detected as non-canonical."""
        nifti_path = create_synthetic_nifti_lps(temp_dir)
        img = nib.load(nifti_path)
        info = detect_nifti_orientation(img)

        assert info.axcodes == ("L", "P", "S")
        assert info.is_canonical is False
        assert info.confidence == "header"
        assert info.source == "nifti_sform"

    def test_detect_nifti_identity_affine(self, temp_dir: Path):
        """Identity affine should be flagged as low confidence."""
        data = np.zeros((10, 10, 10), dtype=np.float32)
        img = nib.Nifti1Image(data, np.eye(4))
        img.header.set_sform(None, code=0)
        img.header.set_qform(None, code=0)

        nifti_path = temp_dir / "identity.nii.gz"
        nib.save(img, nifti_path)

        loaded = nib.load(nifti_path)
        info = detect_nifti_orientation(loaded)

        assert info.confidence == "unknown"
        assert info.source == "identity"

    def test_detect_dicom_orientation(self, temp_dir: Path):
        """DICOM series orientation should be detected from IOP tag."""
        dicom_dir = create_synthetic_dicom_series(temp_dir)
        info = detect_dicom_orientation(dicom_dir)

        assert info.confidence == "header"
        assert info.source == "dicom_iop"
        assert len(info.axcodes) == 3

    def test_detect_dicom_empty_dir(self, temp_dir: Path):
        """Empty directory should return identity orientation."""
        empty_dir = temp_dir / "empty_dicom"
        empty_dir.mkdir()

        info = detect_dicom_orientation(empty_dir)

        assert info.source == "identity"
        assert info.confidence == "unknown"


class TestOrientationValidation:
    """Tests for orientation validation."""

    def test_valid_orientation(self, temp_dir: Path):
        """Valid RAS orientation should pass validation."""
        nifti_path = create_synthetic_nifti_ras(temp_dir)
        img = nib.load(nifti_path)
        info = detect_nifti_orientation(img)

        assert is_orientation_valid(info) is True

    def test_identity_affine_invalid(self):
        """Identity affine should fail validation."""
        info = OrientationInfo(
            axcodes=("R", "A", "S"),
            affine=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            is_canonical=True,
            confidence="unknown",
            source="identity",
        )

        assert is_orientation_valid(info) is False

    def test_degenerate_affine_invalid(self):
        """Degenerate (zero determinant) affine should fail validation."""
        info = OrientationInfo(
            axcodes=("R", "A", "S"),
            affine=[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]],
            is_canonical=True,
            confidence="unknown",
            source="identity",
        )

        assert is_orientation_valid(info) is False


def voxel_to_world(affine: np.ndarray, voxel: tuple[int, int, int]) -> np.ndarray:
    """Convert voxel coordinates to world coordinates using affine."""
    voxel_homogeneous = np.array([*voxel, 1.0])
    return (affine @ voxel_homogeneous)[:3]


class TestReorientation:
    """Tests for reorientation to canonical orientation."""

    def test_reorient_lps_to_ras(self, temp_dir: Path):
        """LPS volume should be correctly reoriented to RAS."""
        nifti_path = create_synthetic_nifti_lps(temp_dir)
        img = nib.load(nifti_path)
        data = np.asarray(img.dataobj)
        affine = img.affine

        reoriented_data, reoriented_affine = reorient_to_canonical(data, affine, target="RAS")

        reoriented_img = nib.Nifti1Image(reoriented_data, reoriented_affine)
        info = detect_nifti_orientation(reoriented_img)

        assert info.axcodes == ("R", "A", "S")
        assert info.is_canonical is True

    def test_reorient_preserves_data(self, temp_dir: Path):
        """Reorientation should preserve total data content (sum/mean)."""
        nifti_path = create_synthetic_nifti_lps(temp_dir)
        img = nib.load(nifti_path)
        original_data = np.asarray(img.dataobj)
        original_sum = np.sum(original_data)

        reoriented_data, _ = reorient_to_canonical(original_data, img.affine, target="RAS")

        assert np.isclose(np.sum(reoriented_data), original_sum, rtol=1e-5)

    def test_reorient_ras_noop(self, temp_dir: Path):
        """Reorienting RAS to RAS should produce same data."""
        nifti_path = create_synthetic_nifti_ras(temp_dir)
        img = nib.load(nifti_path)
        data = np.asarray(img.dataobj)

        reoriented_data, _ = reorient_to_canonical(data, img.affine, target="RAS")

        assert np.allclose(reoriented_data, data)

    def test_reorient_to_las_world_coordinates(self, temp_dir: Path):
        """Reorienting RAS to LAS should preserve world coordinate mapping."""
        nifti_path = create_synthetic_nifti_ras(temp_dir)
        img = nib.load(nifti_path)
        data = np.asarray(img.dataobj)
        affine = img.affine

        reoriented_data, reoriented_affine = reorient_to_canonical(data, affine, target="LAS")

        # Check orientation is LAS
        reoriented_img = nib.Nifti1Image(reoriented_data, reoriented_affine)
        info = detect_nifti_orientation(reoriented_img)
        assert info.axcodes == ("L", "A", "S")

        # Verify world coordinates: corner voxels should map to same world location
        # Original corner (0, 0, 0) maps to some world coordinate
        # After flip along X, this becomes (nx-1, 0, 0) in reoriented space
        nx = data.shape[0]
        original_corner = (0, 0, 0)
        reoriented_corner = (nx - 1, 0, 0)

        world_original = voxel_to_world(affine, original_corner)
        world_reoriented = voxel_to_world(reoriented_affine, reoriented_corner)

        assert np.allclose(world_original, world_reoriented, atol=1e-5)

        # Also check opposite corner
        original_far = (nx - 1, 0, 0)
        reoriented_far = (0, 0, 0)

        world_original_far = voxel_to_world(affine, original_far)
        world_reoriented_far = voxel_to_world(reoriented_affine, reoriented_far)

        assert np.allclose(world_original_far, world_reoriented_far, atol=1e-5)

    def test_reorient_to_lps_world_coordinates(self, temp_dir: Path):
        """Reorienting RAS to LPS should preserve world coordinate mapping."""
        nifti_path = create_synthetic_nifti_ras(temp_dir)
        img = nib.load(nifti_path)
        data = np.asarray(img.dataobj)
        affine = img.affine

        reoriented_data, reoriented_affine = reorient_to_canonical(data, affine, target="LPS")

        # Check orientation is LPS
        reoriented_img = nib.Nifti1Image(reoriented_data, reoriented_affine)
        info = detect_nifti_orientation(reoriented_img)
        assert info.axcodes == ("L", "P", "S")

        # Verify world coordinates for corner mapping
        # After flip along X and Y: (0, 0, 0) -> (nx-1, ny-1, 0)
        nx, ny = data.shape[0], data.shape[1]
        original_corner = (0, 0, 0)
        reoriented_corner = (nx - 1, ny - 1, 0)

        world_original = voxel_to_world(affine, original_corner)
        world_reoriented = voxel_to_world(reoriented_affine, reoriented_corner)

        assert np.allclose(world_original, world_reoriented, atol=1e-5)

        # Check center voxel mapping
        cx, cy, cz = nx // 2, ny // 2, data.shape[2] // 2
        original_center = (cx, cy, cz)
        reoriented_center = (nx - 1 - cx, ny - 1 - cy, cz)

        world_original_center = voxel_to_world(affine, original_center)
        world_reoriented_center = voxel_to_world(reoriented_affine, reoriented_center)

        assert np.allclose(world_original_center, world_reoriented_center, atol=1e-5)


class TestMetadataRoundtrip:
    """Tests for orientation metadata storage and retrieval."""

    def test_metadata_roundtrip(self, temp_dir: Path):
        """OrientationInfo should survive serialization/deserialization."""
        nifti_path = create_synthetic_nifti_ras(temp_dir)
        img = nib.load(nifti_path)
        original_info = detect_nifti_orientation(img)

        metadata = orientation_info_to_metadata(original_info)
        recovered_info = metadata_to_orientation_info(metadata)

        assert recovered_info is not None
        assert recovered_info.axcodes == original_info.axcodes
        assert recovered_info.source == original_info.source
        assert recovered_info.confidence == original_info.confidence

    def test_metadata_with_original_affine(self, temp_dir: Path):
        """Original affine should be preserved in metadata when reorienting."""
        nifti_path = create_synthetic_nifti_lps(temp_dir)
        img = nib.load(nifti_path)
        info = detect_nifti_orientation(img)
        original_affine = img.affine

        metadata = orientation_info_to_metadata(info, original_affine=original_affine)

        assert "original_affine" in metadata


class TestVolumeIntegration:
    """Tests for Volume class orientation integration."""

    def test_from_nifti_stores_orientation(self, volume_uri: str, nifti_4d_path: Path):
        """Volume.from_nifti should store orientation metadata."""
        vol = Volume.from_nifti(volume_uri, nifti_4d_path)

        info = vol.orientation_info
        assert info is not None

    def test_from_nifti_with_reorient(self, temp_dir: Path):
        """from_nifti with reorient=True should convert LPS to RAS."""
        nifti_path = create_synthetic_nifti_lps(temp_dir)
        uri = str(temp_dir / "reoriented_volume")
        vol = Volume.from_nifti(uri, nifti_path, reorient=True)

        info = vol.orientation_info
        assert info is not None
        assert info.axcodes == ("R", "A", "S")

    def test_from_nifti_without_reorient(self, temp_dir: Path):
        """from_nifti with reorient=False should preserve LPS orientation."""
        nifti_path = create_synthetic_nifti_lps(temp_dir)
        uri = str(temp_dir / "preserved_volume")
        vol = Volume.from_nifti(uri, nifti_path, reorient=False)

        info = vol.orientation_info
        assert info is not None
        assert info.axcodes == ("L", "P", "S")

    def test_from_dicom_stores_orientation(self, temp_dir: Path):
        """Volume.from_dicom should store orientation metadata."""
        dicom_dir = create_synthetic_dicom_series(temp_dir)
        uri = str(temp_dir / "dicom_volume")
        vol = Volume.from_dicom(uri, dicom_dir)

        info = vol.orientation_info
        assert info is not None
        assert info.source == "dicom_iop"

    def test_original_affine_stored_on_reorient(self, temp_dir: Path):
        """Reorienting should store original affine for provenance."""
        nifti_path = create_synthetic_nifti_lps(temp_dir)
        uri = str(temp_dir / "reoriented_with_provenance")
        Volume.from_nifti(uri, nifti_path, reorient=True)

        with tiledb.open(uri, "r") as arr:
            assert "original_affine" in arr.meta


class TestMetadataFloatPrecision:
    """Tests for metadata float precision through string serialization."""

    def test_affine_precision_roundtrip(self, temp_dir: Path):
        """Test affine matrix values survive string serialization with acceptable precision."""
        nifti_path = create_synthetic_nifti_ras(temp_dir)
        img = nib.load(nifti_path)
        original_info = detect_nifti_orientation(img)
        original_affine = np.array(original_info.affine)

        # Serialize and deserialize
        metadata = orientation_info_to_metadata(original_info)
        recovered_info = metadata_to_orientation_info(metadata)
        recovered_affine = np.array(recovered_info.affine)

        # Verify precision within 1e-6 (acceptable for medical imaging)
        np.testing.assert_array_almost_equal(
            recovered_affine,
            original_affine,
            decimal=6,
            err_msg="Affine precision loss exceeds acceptable threshold (1e-6)",
        )

    def test_scaling_factors_metadata_storage(self, temp_dir: Path):
        """Test scl_slope and scl_inter values stored in TileDB metadata correctly.

        Note: nibabel does not persist scl_slope/scl_inter for float data that doesn't
        require scaling. This test verifies the RadiObject metadata storage path works
        correctly for the values that ARE set in the original file.
        """
        nifti_path = create_synthetic_nifti_ras(temp_dir)

        # Load the original to see what values nibabel actually provides
        original_img = nib.load(nifti_path)
        original_slope = original_img.header.get("scl_slope", 1.0)
        original_inter = original_img.header.get("scl_inter", 0.0)

        # Handle nibabel's NaN for unset values
        if np.isnan(original_slope):
            original_slope = 1.0
        if np.isnan(original_inter):
            original_inter = 0.0

        # Store in TileDB
        uri = str(temp_dir / "scaled_vol")
        vol = Volume.from_nifti(uri, nifti_path, reorient=False)

        # Verify metadata is stored correctly
        meta = vol._metadata
        stored_slope = float(meta.get("nifti_scl_slope", 1.0))
        stored_inter = float(meta.get("nifti_scl_inter", 0.0))

        # The stored values should match what was read from the file
        assert (
            abs(stored_slope - float(original_slope)) < 1e-6
        ), f"Slope storage error: {original_slope} -> {stored_slope}"
        assert (
            abs(stored_inter - float(original_inter)) < 1e-6
        ), f"Inter storage error: {original_inter} -> {stored_inter}"

    def test_extreme_affine_values_precision(self, temp_dir: Path):
        """Test precision with extreme but valid affine values."""
        rng = np.random.default_rng(42)
        data = rng.random((10, 10, 10), dtype=np.float32)

        # Affine with large translation and small voxel size (realistic MRI scenario)
        affine = np.array(
            [
                [0.001, 0.0, 0.0, -150000.0],  # Very small voxel, large offset
                [0.0, 0.001, 0.0, -150000.0],
                [0.0, 0.0, 0.001, -150000.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        img = nib.Nifti1Image(data, affine)
        img.header.set_sform(affine, code=1)

        nifti_path = temp_dir / "extreme_affine.nii.gz"
        nib.save(img, nifti_path)

        # Roundtrip through TileDB
        uri = str(temp_dir / "extreme_vol")
        vol = Volume.from_nifti(uri, nifti_path, reorient=False)

        export_path = temp_dir / "extreme_export.nii.gz"
        vol.to_nifti(export_path)

        exported_img = nib.load(export_path)
        exported_affine = exported_img.affine

        # Check relative precision for non-zero elements
        for i in range(4):
            for j in range(4):
                orig = affine[i, j]
                exp = exported_affine[i, j]
                if abs(orig) > 1e-10:
                    rel_error = abs(orig - exp) / abs(orig)
                    assert rel_error < 1e-5, (
                        f"Relative error at [{i},{j}]: {rel_error:.2e} "
                        f"(original={orig}, exported={exp})"
                    )
                else:
                    assert abs(orig - exp) < 1e-10

    def test_voxel_spacing_precision(self, temp_dir: Path):
        """Test voxel spacing (diagonal of affine) precision."""
        rng = np.random.default_rng(42)
        data = rng.random((10, 10, 10), dtype=np.float32)

        # Typical high-resolution MRI voxel sizes
        voxel_sizes = [0.5, 0.5, 1.2]  # mm
        affine = np.diag([*voxel_sizes, 1.0])

        img = nib.Nifti1Image(data, affine)
        img.header.set_sform(affine, code=1)

        nifti_path = temp_dir / "voxel_spacing.nii.gz"
        nib.save(img, nifti_path)

        uri = str(temp_dir / "voxel_spacing_vol")
        vol = Volume.from_nifti(uri, nifti_path, reorient=False)

        export_path = temp_dir / "voxel_spacing_export.nii.gz"
        vol.to_nifti(export_path)

        exported_img = nib.load(export_path)
        exported_zooms = exported_img.header.get_zooms()[:3]

        for i, (orig, exp) in enumerate(zip(voxel_sizes, exported_zooms)):
            assert (
                abs(orig - exp) < 1e-6
            ), f"Voxel spacing axis {i}: {orig} -> {exp} (diff={abs(orig - exp):.2e})"
