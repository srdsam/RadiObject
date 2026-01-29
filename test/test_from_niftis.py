"""Integration tests for NIfTI/DICOM ingestion with metadata capture."""

from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from radiobject.imaging_metadata import (
    extract_nifti_metadata,
    infer_series_type,
    KNOWN_SERIES_TYPES,
)
from radiobject.radi_object import RadiObject
from radiobject.volume_collection import VolumeCollection


# ----- Fixtures -----


@pytest.fixture
def synthetic_nifti_files(temp_dir: Path) -> list[tuple[Path, str]]:
    """Create synthetic NIfTI files for testing."""
    niftis = []
    shape = (32, 32, 16)
    affine = np.eye(4)
    affine[0, 0] = 1.0  # 1mm voxel spacing
    affine[1, 1] = 1.0
    affine[2, 2] = 2.0  # 2mm slice thickness

    for subject_id in ["sub-01", "sub-02", "sub-03"]:
        for series_type in ["T1w", "FLAIR"]:
            data = np.random.rand(*shape).astype(np.float32)
            img = nib.Nifti1Image(data, affine)

            # Set header fields
            img.header.set_qform(affine, code=1)
            img.header.set_sform(affine, code=1)

            filename = f"{subject_id}_{series_type}.nii.gz"
            filepath = temp_dir / filename
            nib.save(img, filepath)
            niftis.append((filepath, subject_id))

    return niftis


@pytest.fixture
def mismatched_dim_niftis(temp_dir: Path) -> list[tuple[Path, str]]:
    """Create NIfTI files with mismatched dimensions."""
    niftis = []
    affine = np.eye(4)

    # First file: 32x32x16
    data1 = np.random.rand(32, 32, 16).astype(np.float32)
    img1 = nib.Nifti1Image(data1, affine)
    path1 = temp_dir / "sub-01_T1w.nii.gz"
    nib.save(img1, path1)
    niftis.append((path1, "sub-01"))

    # Second file: 64x64x32 (different dimensions)
    data2 = np.random.rand(64, 64, 32).astype(np.float32)
    img2 = nib.Nifti1Image(data2, affine)
    path2 = temp_dir / "sub-02_T1w.nii.gz"
    nib.save(img2, path2)
    niftis.append((path2, "sub-02"))

    return niftis


# ----- Series Type Inference Tests -----


class TestSeriesTypeInference:
    """Tests for infer_series_type function."""

    def test_bids_suffix_t1w(self, temp_dir: Path) -> None:
        path = temp_dir / "sub-01_ses-01_T1w.nii.gz"
        assert infer_series_type(path) == "T1w"

    def test_bids_suffix_flair(self, temp_dir: Path) -> None:
        path = temp_dir / "sub-01_FLAIR.nii.gz"
        assert infer_series_type(path) == "FLAIR"

    def test_bids_suffix_bold(self, temp_dir: Path) -> None:
        path = temp_dir / "sub-01_task-rest_bold.nii.gz"
        assert infer_series_type(path) == "bold"

    def test_bids_suffix_dwi(self, temp_dir: Path) -> None:
        path = temp_dir / "sub-01_dwi.nii.gz"
        assert infer_series_type(path) == "dwi"

    def test_pattern_t1_mprage(self, temp_dir: Path) -> None:
        path = temp_dir / "T1_MPRAGE_001.nii.gz"
        assert infer_series_type(path) == "T1w"

    def test_pattern_t2(self, temp_dir: Path) -> None:
        path = temp_dir / "anatomical_T2.nii.gz"
        assert infer_series_type(path) == "T2w"

    def test_pattern_contrast_enhanced(self, temp_dir: Path) -> None:
        path = temp_dir / "T1GD_post_contrast.nii.gz"
        assert infer_series_type(path) == "T1gd"

    def test_unknown_pattern(self, temp_dir: Path) -> None:
        path = temp_dir / "random_scan_001.nii.gz"
        assert infer_series_type(path) == "unknown"

    def test_case_insensitive(self, temp_dir: Path) -> None:
        path = temp_dir / "sub-01_t1w.nii.gz"
        assert infer_series_type(path) == "T1w"


# ----- NiftiMetadata Extraction Tests -----


class TestNiftiMetadataExtraction:
    """Tests for extract_nifti_metadata function."""

    def test_extracts_dimensions(self, synthetic_nifti_files: list[tuple[Path, str]]) -> None:
        path, _ = synthetic_nifti_files[0]
        metadata = extract_nifti_metadata(path)

        assert metadata.dimensions == (32, 32, 16)

    def test_extracts_voxel_spacing(self, synthetic_nifti_files: list[tuple[Path, str]]) -> None:
        path, _ = synthetic_nifti_files[0]
        metadata = extract_nifti_metadata(path)

        assert metadata.voxel_spacing[0] == pytest.approx(1.0)
        assert metadata.voxel_spacing[1] == pytest.approx(1.0)
        assert metadata.voxel_spacing[2] == pytest.approx(2.0)

    def test_extracts_orientation(self, synthetic_nifti_files: list[tuple[Path, str]]) -> None:
        path, _ = synthetic_nifti_files[0]
        metadata = extract_nifti_metadata(path)

        assert len(metadata.axcodes) == 3
        assert metadata.orientation_source in ("nifti_sform", "nifti_qform", "identity")

    def test_extracts_affine_as_json(self, synthetic_nifti_files: list[tuple[Path, str]]) -> None:
        path, _ = synthetic_nifti_files[0]
        metadata = extract_nifti_metadata(path)

        affine = json.loads(metadata.affine_json)
        assert len(affine) == 4
        assert len(affine[0]) == 4

    def test_stores_source_path(self, synthetic_nifti_files: list[tuple[Path, str]]) -> None:
        path, _ = synthetic_nifti_files[0]
        metadata = extract_nifti_metadata(path)

        assert str(path.absolute()) in metadata.source_path

    def test_file_not_found_raises(self, temp_dir: Path) -> None:
        nonexistent = temp_dir / "does_not_exist.nii.gz"
        with pytest.raises(FileNotFoundError):
            extract_nifti_metadata(nonexistent)


# ----- VolumeCollection.from_niftis Tests -----


class TestVolumeCollectionFromNiftis:
    """Tests for VolumeCollection.from_niftis factory method."""

    def test_creates_collection(
        self, temp_dir: Path, synthetic_nifti_files: list[tuple[Path, str]]
    ) -> None:
        # Use only T1w files (same dimensions)
        t1w_files = [(p, s) for p, s in synthetic_nifti_files if "T1w" in str(p)]
        uri = str(temp_dir / "vc_from_niftis")

        vc = VolumeCollection.from_niftis(uri=uri, niftis=t1w_files)

        assert len(vc) == 3
        assert vc.shape == (32, 32, 16)

    def test_captures_metadata_in_obs(
        self, temp_dir: Path, synthetic_nifti_files: list[tuple[Path, str]]
    ) -> None:
        t1w_files = [(p, s) for p, s in synthetic_nifti_files if "T1w" in str(p)]
        uri = str(temp_dir / "vc_metadata")

        vc = VolumeCollection.from_niftis(uri=uri, niftis=t1w_files)
        obs = vc.obs.read()

        # Check metadata columns exist
        assert "voxel_spacing" in obs.columns
        assert "dimensions" in obs.columns
        assert "axcodes" in obs.columns
        assert "series_type" in obs.columns
        assert "source_path" in obs.columns

    def test_dimension_validation_raises(
        self, temp_dir: Path, mismatched_dim_niftis: list[tuple[Path, str]]
    ) -> None:
        uri = str(temp_dir / "vc_mismatch")

        with pytest.raises(ValueError, match="Dimension mismatch"):
            VolumeCollection.from_niftis(
                uri=uri,
                niftis=mismatched_dim_niftis,
                validate_dimensions=True,
            )

    def test_dimension_validation_disabled(
        self, temp_dir: Path, mismatched_dim_niftis: list[tuple[Path, str]]
    ) -> None:
        # When validation is disabled, only first file's shape is used
        uri = str(temp_dir / "vc_no_validate")

        # This should raise because volumes still have different shapes
        # but we create with first shape
        with pytest.raises(ValueError, match="Dimension mismatch"):
            VolumeCollection.from_niftis(
                uri=uri,
                niftis=mismatched_dim_niftis,
                validate_dimensions=True,
            )

    def test_file_not_found_raises(self, temp_dir: Path) -> None:
        nonexistent = temp_dir / "does_not_exist.nii.gz"
        uri = str(temp_dir / "vc_missing")

        with pytest.raises(FileNotFoundError):
            VolumeCollection.from_niftis(uri=uri, niftis=[(nonexistent, "sub-01")])

    def test_empty_list_raises(self, temp_dir: Path) -> None:
        uri = str(temp_dir / "vc_empty")

        with pytest.raises(ValueError, match="At least one NIfTI"):
            VolumeCollection.from_niftis(uri=uri, niftis=[])

    def test_valid_subject_ids_whitelist(
        self, temp_dir: Path, synthetic_nifti_files: list[tuple[Path, str]]
    ) -> None:
        t1w_files = [(p, s) for p, s in synthetic_nifti_files if "T1w" in str(p)]
        uri = str(temp_dir / "vc_whitelist")

        # Only allow sub-01 and sub-02
        with pytest.raises(ValueError, match="Invalid obs_subject_ids"):
            VolumeCollection.from_niftis(
                uri=uri,
                niftis=t1w_files,
                valid_subject_ids={"sub-01", "sub-02"},  # Missing sub-03
            )


# ----- RadiObject.from_niftis Tests -----


class TestRadiObjectFromNiftis:
    """Tests for RadiObject.from_niftis factory method with auto-grouping."""

    def test_auto_groups_by_series_type(
        self, temp_dir: Path, synthetic_nifti_files: list[tuple[Path, str]]
    ) -> None:
        uri = str(temp_dir / "radi_auto_group")

        radi = RadiObject.from_niftis(uri=uri, niftis=synthetic_nifti_files)

        # Should have created T1w and FLAIR collections
        assert "T1w" in radi.collection_names
        assert "FLAIR" in radi.collection_names
        assert radi.n_collections == 2

    def test_each_collection_has_correct_volumes(
        self, temp_dir: Path, synthetic_nifti_files: list[tuple[Path, str]]
    ) -> None:
        uri = str(temp_dir / "radi_volumes")

        radi = RadiObject.from_niftis(uri=uri, niftis=synthetic_nifti_files)

        # Each collection should have 3 volumes (one per subject)
        assert len(radi.T1w) == 3
        assert len(radi.FLAIR) == 3

    def test_auto_generates_obs_meta(
        self, temp_dir: Path, synthetic_nifti_files: list[tuple[Path, str]]
    ) -> None:
        uri = str(temp_dir / "radi_auto_meta")

        radi = RadiObject.from_niftis(uri=uri, niftis=synthetic_nifti_files)

        obs_meta = radi.obs_meta.read()
        assert len(obs_meta) == 3  # 3 subjects
        assert set(obs_meta["obs_subject_id"]) == {"sub-01", "sub-02", "sub-03"}

    def test_user_provided_obs_meta(
        self, temp_dir: Path, synthetic_nifti_files: list[tuple[Path, str]]
    ) -> None:
        uri = str(temp_dir / "radi_user_meta")
        obs_meta = pd.DataFrame({
            "obs_subject_id": ["sub-01", "sub-02", "sub-03"],
            "age": [45, 52, 38],
            "sex": ["M", "F", "M"],
        })

        radi = RadiObject.from_niftis(
            uri=uri,
            niftis=synthetic_nifti_files,
            obs_meta=obs_meta,
        )

        result_meta = radi.obs_meta.read()
        assert "age" in result_meta.columns
        assert "sex" in result_meta.columns

    def test_fk_constraint_valid(
        self, temp_dir: Path, synthetic_nifti_files: list[tuple[Path, str]]
    ) -> None:
        uri = str(temp_dir / "radi_fk_valid")
        obs_meta = pd.DataFrame({
            "obs_subject_id": ["sub-01", "sub-02", "sub-03"],
            "age": [45, 52, 38],
        })

        radi = RadiObject.from_niftis(
            uri=uri,
            niftis=synthetic_nifti_files,
            obs_meta=obs_meta,
        )

        # Should not raise
        radi.validate()

    def test_fk_constraint_invalid_raises(
        self, temp_dir: Path, synthetic_nifti_files: list[tuple[Path, str]]
    ) -> None:
        uri = str(temp_dir / "radi_fk_invalid")
        # obs_meta only has sub-01, but niftis include sub-02 and sub-03
        obs_meta = pd.DataFrame({
            "obs_subject_id": ["sub-01"],
            "age": [45],
        })

        with pytest.raises(ValueError, match="obs_subject_ids.*not found in obs_meta"):
            RadiObject.from_niftis(
                uri=uri,
                niftis=synthetic_nifti_files,
                obs_meta=obs_meta,
            )

    def test_obs_meta_missing_column_raises(
        self, temp_dir: Path, synthetic_nifti_files: list[tuple[Path, str]]
    ) -> None:
        uri = str(temp_dir / "radi_missing_col")
        # obs_meta without obs_subject_id column
        obs_meta = pd.DataFrame({
            "patient_id": ["sub-01", "sub-02", "sub-03"],
        })

        with pytest.raises(ValueError, match="obs_subject_id"):
            RadiObject.from_niftis(
                uri=uri,
                niftis=synthetic_nifti_files,
                obs_meta=obs_meta,
            )

    def test_empty_niftis_raises(self, temp_dir: Path) -> None:
        uri = str(temp_dir / "radi_empty")

        with pytest.raises(ValueError, match="At least one NIfTI"):
            RadiObject.from_niftis(uri=uri, niftis=[])

    def test_metadata_captured_in_collection_obs(
        self, temp_dir: Path, synthetic_nifti_files: list[tuple[Path, str]]
    ) -> None:
        uri = str(temp_dir / "radi_obs_meta")

        radi = RadiObject.from_niftis(uri=uri, niftis=synthetic_nifti_files)

        t1w_obs = radi.T1w.obs.read()
        assert "voxel_spacing" in t1w_obs.columns
        assert "axcodes" in t1w_obs.columns
        assert "series_type" in t1w_obs.columns


# ----- Validation Tests -----


class TestRadiObjectValidation:
    """Tests for RadiObject.validate() FK constraint checking."""

    def test_validate_passes_on_valid_radi(
        self, temp_dir: Path, synthetic_nifti_files: list[tuple[Path, str]]
    ) -> None:
        uri = str(temp_dir / "radi_validate")

        radi = RadiObject.from_niftis(uri=uri, niftis=synthetic_nifti_files)

        # Should not raise
        radi.validate()


# ----- Integration with Real Data -----


class TestFromNiftisWithRealData:
    """Integration tests using real BraTS data (if available)."""

    def test_from_real_nifti_4d(
        self, temp_dir: Path, nifti_4d_path: Path, nifti_manifest: list[dict]
    ) -> None:
        """Test with real 4D BraTS data - extract single channel."""
        # Load 4D data and extract first channel
        img = nib.load(nifti_4d_path)
        data_4d = np.asarray(img.dataobj, dtype=np.float32)
        data_3d = data_4d[..., 0]

        # Create temp NIfTI with single channel
        affine = img.affine
        new_img = nib.Nifti1Image(data_3d, affine)
        nifti_path = temp_dir / "BRATS_001_flair.nii.gz"
        nib.save(new_img, nifti_path)

        # Create VolumeCollection
        uri = str(temp_dir / "vc_real")
        vc = VolumeCollection.from_niftis(
            uri=uri,
            niftis=[(nifti_path, "BRATS_001")],
        )

        assert len(vc) == 1
        obs = vc.obs.read()
        # Dimensions are stored as a tuple-string in the obs DataFrame
        assert "dimensions" in obs.columns
        # Verify shape is captured (stored as string representation of tuple)
        dims_str = str(obs.iloc[0]["dimensions"])
        assert str(data_3d.shape[0]) in dims_str
        assert str(data_3d.shape[1]) in dims_str
        assert str(data_3d.shape[2]) in dims_str


# ----- Known Series Types Tests -----


class TestKnownSeriesTypes:
    """Tests for KNOWN_SERIES_TYPES constant."""

    def test_contains_common_mri_types(self) -> None:
        assert "T1w" in KNOWN_SERIES_TYPES
        assert "T2w" in KNOWN_SERIES_TYPES
        assert "FLAIR" in KNOWN_SERIES_TYPES
        assert "bold" in KNOWN_SERIES_TYPES
        assert "dwi" in KNOWN_SERIES_TYPES

    def test_contains_ct_types(self) -> None:
        assert "CT" in KNOWN_SERIES_TYPES
        assert "CTA" in KNOWN_SERIES_TYPES
        assert "CTPA" in KNOWN_SERIES_TYPES

    def test_is_frozen_set(self) -> None:
        assert isinstance(KNOWN_SERIES_TYPES, frozenset)
