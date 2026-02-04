"""Integration tests for NIfTI/DICOM ingestion with metadata capture."""

from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from radiobject.imaging_metadata import (
    KNOWN_SERIES_TYPES,
    extract_nifti_metadata,
    infer_series_type,
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
def synthetic_nifti_images(
    synthetic_nifti_files: list[tuple[Path, str]],
) -> dict[str, list[tuple[Path, str]]]:
    """Convert synthetic_nifti_files to images dict format grouped by series type."""
    images: dict[str, list[tuple[Path, str]]] = {"T1w": [], "FLAIR": []}
    for path, subject_id in synthetic_nifti_files:
        if "T1w" in path.name:
            images["T1w"].append((path, subject_id))
        elif "FLAIR" in path.name:
            images["FLAIR"].append((path, subject_id))
    return images


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

    @pytest.mark.parametrize(
        ("filename", "expected"),
        [
            ("sub-01_ses-01_T1w.nii.gz", "T1w"),
            ("sub-01_FLAIR.nii.gz", "FLAIR"),
            ("sub-01_task-rest_bold.nii.gz", "bold"),
            ("sub-01_dwi.nii.gz", "dwi"),
        ],
    )
    def test_bids_suffix_inference(self, temp_dir: Path, filename: str, expected: str) -> None:
        path = temp_dir / filename
        assert infer_series_type(path) == expected

    @pytest.mark.parametrize(
        ("filename", "expected"),
        [
            ("T1_MPRAGE_001.nii.gz", "T1w"),
            ("anatomical_T2.nii.gz", "T2w"),
            ("T1GD_post_contrast.nii.gz", "T1gd"),
        ],
    )
    def test_pattern_inference(self, temp_dir: Path, filename: str, expected: str) -> None:
        path = temp_dir / filename
        assert infer_series_type(path) == expected

    def test_unknown_pattern_returns_unknown(self, temp_dir: Path) -> None:
        path = temp_dir / "random_scan_001.nii.gz"
        assert infer_series_type(path) == "unknown"

    def test_case_insensitive_matching(self, temp_dir: Path) -> None:
        path = temp_dir / "sub-01_t1w.nii.gz"
        assert infer_series_type(path) == "T1w"


# ----- NiftiMetadata Extraction Tests -----


class TestNiftiMetadataExtraction:
    """Tests for extract_nifti_metadata function."""

    def test_extracts_spatial_metadata(self, synthetic_nifti_files: list[tuple[Path, str]]) -> None:
        path, _ = synthetic_nifti_files[0]
        metadata = extract_nifti_metadata(path)

        assert metadata.dimensions == (32, 32, 16)
        assert metadata.voxel_spacing == pytest.approx((1.0, 1.0, 2.0))
        assert len(metadata.axcodes) == 3
        assert metadata.orientation_source in ("nifti_sform", "nifti_qform", "identity")

    def test_extracts_affine_and_source(
        self, synthetic_nifti_files: list[tuple[Path, str]]
    ) -> None:
        path, _ = synthetic_nifti_files[0]
        metadata = extract_nifti_metadata(path)

        affine = json.loads(metadata.affine_json)
        assert len(affine) == 4 and len(affine[0]) == 4
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
    """Tests for RadiObject.from_niftis factory method with images dict."""

    def test_creates_collections_from_images_dict(
        self, temp_dir: Path, synthetic_nifti_images: dict[str, list[tuple[Path, str]]]
    ) -> None:
        uri = str(temp_dir / "radi_images_dict")

        radi = RadiObject.from_niftis(uri=uri, images=synthetic_nifti_images)

        # Should have created T1w and FLAIR collections
        assert "T1w" in radi.collection_names
        assert "FLAIR" in radi.collection_names
        assert radi.n_collections == 2

    def test_each_collection_has_correct_volumes(
        self, temp_dir: Path, synthetic_nifti_images: dict[str, list[tuple[Path, str]]]
    ) -> None:
        uri = str(temp_dir / "radi_volumes")

        radi = RadiObject.from_niftis(uri=uri, images=synthetic_nifti_images)

        # Each collection should have 3 volumes (one per subject)
        assert len(radi.T1w) == 3
        assert len(radi.FLAIR) == 3

    def test_auto_generates_obs_meta(
        self, temp_dir: Path, synthetic_nifti_images: dict[str, list[tuple[Path, str]]]
    ) -> None:
        uri = str(temp_dir / "radi_auto_meta")

        radi = RadiObject.from_niftis(uri=uri, images=synthetic_nifti_images)

        obs_meta = radi.obs_meta.read()
        assert len(obs_meta) == 3  # 3 subjects
        assert set(obs_meta["obs_subject_id"]) == {"sub-01", "sub-02", "sub-03"}

    def test_user_provided_obs_meta(
        self, temp_dir: Path, synthetic_nifti_images: dict[str, list[tuple[Path, str]]]
    ) -> None:
        uri = str(temp_dir / "radi_user_meta")
        obs_meta = pd.DataFrame(
            {
                "obs_subject_id": ["sub-01", "sub-02", "sub-03"],
                "age": [45, 52, 38],
                "sex": ["M", "F", "M"],
            }
        )

        radi = RadiObject.from_niftis(
            uri=uri,
            images=synthetic_nifti_images,
            obs_meta=obs_meta,
        )

        result_meta = radi.obs_meta.read()
        assert "age" in result_meta.columns
        assert "sex" in result_meta.columns

    def test_fk_constraint_valid(
        self, temp_dir: Path, synthetic_nifti_images: dict[str, list[tuple[Path, str]]]
    ) -> None:
        uri = str(temp_dir / "radi_fk_valid")
        obs_meta = pd.DataFrame(
            {
                "obs_subject_id": ["sub-01", "sub-02", "sub-03"],
                "age": [45, 52, 38],
            }
        )

        radi = RadiObject.from_niftis(
            uri=uri,
            images=synthetic_nifti_images,
            obs_meta=obs_meta,
        )

        # Should not raise
        radi.validate()

    def test_fk_constraint_invalid_raises(
        self, temp_dir: Path, synthetic_nifti_images: dict[str, list[tuple[Path, str]]]
    ) -> None:
        uri = str(temp_dir / "radi_fk_invalid")
        # obs_meta only has sub-01, but images include sub-02 and sub-03
        obs_meta = pd.DataFrame(
            {
                "obs_subject_id": ["sub-01"],
                "age": [45],
            }
        )

        with pytest.raises(ValueError, match="obs_subject_ids.*not found in obs_meta"):
            RadiObject.from_niftis(
                uri=uri,
                images=synthetic_nifti_images,
                obs_meta=obs_meta,
            )

    def test_obs_meta_missing_column_raises(
        self, temp_dir: Path, synthetic_nifti_images: dict[str, list[tuple[Path, str]]]
    ) -> None:
        uri = str(temp_dir / "radi_missing_col")
        # obs_meta without obs_subject_id column
        obs_meta = pd.DataFrame(
            {
                "patient_id": ["sub-01", "sub-02", "sub-03"],
            }
        )

        with pytest.raises(ValueError, match="obs_subject_id"):
            RadiObject.from_niftis(
                uri=uri,
                images=synthetic_nifti_images,
                obs_meta=obs_meta,
            )

    def test_empty_images_raises(self, temp_dir: Path) -> None:
        uri = str(temp_dir / "radi_empty")

        with pytest.raises(ValueError, match="images dict cannot be empty"):
            RadiObject.from_niftis(uri=uri, images={})

    def test_metadata_captured_in_collection_obs(
        self, temp_dir: Path, synthetic_nifti_images: dict[str, list[tuple[Path, str]]]
    ) -> None:
        uri = str(temp_dir / "radi_obs_meta")

        radi = RadiObject.from_niftis(uri=uri, images=synthetic_nifti_images)

        t1w_obs = radi.T1w.obs.read()
        assert "voxel_spacing" in t1w_obs.columns
        assert "axcodes" in t1w_obs.columns
        assert "series_type" in t1w_obs.columns


# ----- Validation Tests -----


class TestRadiObjectValidation:
    """Tests for RadiObject.validate() FK constraint checking."""

    def test_validate_passes_on_valid_radi(
        self, temp_dir: Path, synthetic_nifti_images: dict[str, list[tuple[Path, str]]]
    ) -> None:
        uri = str(temp_dir / "radi_validate")

        radi = RadiObject.from_niftis(uri=uri, images=synthetic_nifti_images)

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

    def test_contains_expected_modality_types(self) -> None:
        mri_types = {"T1w", "T2w", "FLAIR", "bold", "dwi"}
        ct_types = {"CT", "CTA", "CTPA"}

        assert mri_types.issubset(KNOWN_SERIES_TYPES)
        assert ct_types.issubset(KNOWN_SERIES_TYPES)
        assert isinstance(KNOWN_SERIES_TYPES, frozenset)


# ----- Images Dict API Tests -----


@pytest.fixture
def multi_modality_dirs(temp_dir: Path) -> dict[str, Path]:
    """Create separate directories for CT and segmentation NIfTIs."""
    images_dir = temp_dir / "imagesTr"
    labels_dir = temp_dir / "labelsTr"
    images_dir.mkdir()
    labels_dir.mkdir()

    shape = (32, 32, 16)
    affine = np.eye(4)

    for subject_id in ["sub-01", "sub-02", "sub-03"]:
        # CT images
        data = np.random.rand(*shape).astype(np.float32)
        img = nib.Nifti1Image(data, affine)
        nib.save(img, images_dir / f"{subject_id}.nii.gz")

        # Segmentation labels
        seg = np.random.randint(0, 4, shape, dtype=np.int16)
        seg_img = nib.Nifti1Image(seg, affine)
        nib.save(seg_img, labels_dir / f"{subject_id}.nii.gz")

    return {"images": images_dir, "labels": labels_dir}


class TestFromNiftis4D:
    """Tests for 4D NIfTI volume ingestion."""

    def test_ingest_4d_nifti_preserves_shape(self, temp_dir: Path):
        """4D NIfTI ingested via from_niftis() preserves full shape."""
        shape_4d = (64, 64, 32, 200)
        affine = np.eye(4)
        data = np.random.rand(*shape_4d).astype(np.float32)
        img = nib.Nifti1Image(data, affine)
        img.header.set_sform(affine, code=1)
        path = temp_dir / "sub-01_bold.nii.gz"
        nib.save(img, path)

        uri = str(temp_dir / "vc_4d")
        vc = VolumeCollection.from_niftis(uri=uri, niftis=[(path, "sub-01")])

        assert len(vc) == 1
        assert vc.shape == (64, 64, 32)  # Collection shape is spatial only
        vol = vc.iloc[0]
        assert vol.shape == (64, 64, 32, 200)  # Volume shape is full 4D

        obs = vc.obs.read()
        assert "(64, 64, 32, 200)" in str(obs.iloc[0]["dimensions"])

    def test_mixed_3d_4d_same_spatial_shape(self, temp_dir: Path):
        """3D and 4D volumes with same spatial grid coexist in one collection."""
        affine = np.eye(4)

        # 3D volume
        data_3d = np.random.rand(64, 64, 32).astype(np.float32)
        img_3d = nib.Nifti1Image(data_3d, affine)
        img_3d.header.set_sform(affine, code=1)
        path_3d = temp_dir / "sub-01_bold.nii.gz"
        nib.save(img_3d, path_3d)

        # 4D volume with same spatial grid
        data_4d = np.random.rand(64, 64, 32, 100).astype(np.float32)
        img_4d = nib.Nifti1Image(data_4d, affine)
        img_4d.header.set_sform(affine, code=1)
        path_4d = temp_dir / "sub-02_bold.nii.gz"
        nib.save(img_4d, path_4d)

        uri = str(temp_dir / "vc_mixed_3d4d")
        vc = VolumeCollection.from_niftis(
            uri=uri,
            niftis=[(path_3d, "sub-01"), (path_4d, "sub-02")],
            validate_dimensions=True,
        )

        assert len(vc) == 2
        assert vc.shape == (64, 64, 32)

    def test_4d_dimension_validation_uses_spatial(self, temp_dir: Path):
        """Two 4D volumes with same spatial dims but different time dims pass validation."""
        affine = np.eye(4)

        data_a = np.random.rand(64, 64, 32, 100).astype(np.float32)
        img_a = nib.Nifti1Image(data_a, affine)
        img_a.header.set_sform(affine, code=1)
        path_a = temp_dir / "sub-01_bold.nii.gz"
        nib.save(img_a, path_a)

        data_b = np.random.rand(64, 64, 32, 200).astype(np.float32)
        img_b = nib.Nifti1Image(data_b, affine)
        img_b.header.set_sform(affine, code=1)
        path_b = temp_dir / "sub-02_bold.nii.gz"
        nib.save(img_b, path_b)

        uri = str(temp_dir / "vc_4d_diff_time")
        vc = VolumeCollection.from_niftis(
            uri=uri,
            niftis=[(path_a, "sub-01"), (path_b, "sub-02")],
            validate_dimensions=True,
        )

        assert len(vc) == 2
        assert vc.shape == (64, 64, 32)


class TestImagesDictAPI:
    """Tests for the new images dict parameter in RadiObject.from_niftis()."""

    def test_images_with_directories(
        self, temp_dir: Path, multi_modality_dirs: dict[str, Path]
    ) -> None:
        """Test images dict with directory paths."""
        uri = str(temp_dir / "radi_images_dirs")

        radi = RadiObject.from_niftis(
            uri=uri,
            images={
                "CT": multi_modality_dirs["images"],
                "seg": multi_modality_dirs["labels"],
            },
        )

        assert "CT" in radi.collection_names
        assert "seg" in radi.collection_names
        assert len(radi.CT) == 3
        assert len(radi.seg) == 3

    def test_images_with_glob_patterns(
        self, temp_dir: Path, multi_modality_dirs: dict[str, Path]
    ) -> None:
        """Test images dict with glob patterns."""
        uri = str(temp_dir / "radi_images_globs")

        radi = RadiObject.from_niftis(
            uri=uri,
            images={
                "CT": str(multi_modality_dirs["images"] / "*.nii.gz"),
                "seg": str(multi_modality_dirs["labels"] / "*.nii.gz"),
            },
        )

        assert "CT" in radi.collection_names
        assert "seg" in radi.collection_names
        assert len(radi.CT) == 3
        assert len(radi.seg) == 3

    def test_images_with_pre_resolved_list(
        self, temp_dir: Path, multi_modality_dirs: dict[str, Path]
    ) -> None:
        """Test images dict with pre-resolved (path, subject_id) tuples."""
        uri = str(temp_dir / "radi_images_list")

        images_dir = multi_modality_dirs["images"]
        labels_dir = multi_modality_dirs["labels"]

        radi = RadiObject.from_niftis(
            uri=uri,
            images={
                "CT": [
                    (images_dir / "sub-01.nii.gz", "patient-001"),
                    (images_dir / "sub-02.nii.gz", "patient-002"),
                ],
                "seg": [
                    (labels_dir / "sub-01.nii.gz", "patient-001"),
                    (labels_dir / "sub-02.nii.gz", "patient-002"),
                ],
            },
        )

        assert len(radi.CT) == 2
        assert len(radi.seg) == 2
        assert "patient-001" in radi.obs_subject_ids
        assert "patient-002" in radi.obs_subject_ids

    def test_images_empty_dict_raises(self, temp_dir: Path) -> None:
        """Error when images dict is empty."""
        uri = str(temp_dir / "radi_empty_images")

        with pytest.raises(ValueError, match="images dict cannot be empty"):
            RadiObject.from_niftis(uri=uri, images={})

    def test_validate_alignment_passes(
        self, temp_dir: Path, multi_modality_dirs: dict[str, Path]
    ) -> None:
        """Alignment validation passes when subjects match."""
        uri = str(temp_dir / "radi_align_pass")

        # Should not raise - subjects are aligned
        radi = RadiObject.from_niftis(
            uri=uri,
            images={
                "CT": multi_modality_dirs["images"],
                "seg": multi_modality_dirs["labels"],
            },
            validate_alignment=True,
        )

        assert len(radi) == 3

    def test_validate_alignment_fails_mismatch(
        self, temp_dir: Path, multi_modality_dirs: dict[str, Path]
    ) -> None:
        """Alignment validation fails when subjects don't match."""
        # Add extra file to labels only
        labels_dir = multi_modality_dirs["labels"]
        extra_seg = np.random.randint(0, 4, (32, 32, 16), dtype=np.int16)
        extra_img = nib.Nifti1Image(extra_seg, np.eye(4))
        nib.save(extra_img, labels_dir / "sub-99.nii.gz")

        uri = str(temp_dir / "radi_align_fail")

        with pytest.raises(ValueError, match="Subject ID mismatch"):
            RadiObject.from_niftis(
                uri=uri,
                images={
                    "CT": multi_modality_dirs["images"],
                    "seg": multi_modality_dirs["labels"],
                },
                validate_alignment=True,
            )
