"""Tests for NIfTI discovery and raw ingestion."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from radiobject.ingest import NiftiSource, discover_nifti_pairs
from radiobject.radi_object import RadiObject

# ===== Fixtures =====


@pytest.fixture
def nifti_dir(temp_dir: Path) -> Path:
    """Create directory with NIfTI files."""
    images_dir = temp_dir / "imagesTr"
    images_dir.mkdir()

    # Create 3 volumes with 2mm spacing
    affine_2mm = np.diag([2.0, 2.0, 2.0, 1.0])
    shape_2mm = (32, 32, 16)

    for i in range(3):
        data = np.random.rand(*shape_2mm).astype(np.float32) * 100
        img = nib.Nifti1Image(data, affine_2mm)
        img.header.set_qform(affine_2mm, code=1)
        img.header.set_sform(affine_2mm, code=1)
        nib.save(img, images_dir / f"lung_{i:03d}.nii.gz")

    return images_dir


@pytest.fixture
def label_dir(temp_dir: Path, nifti_dir: Path) -> Path:
    """Create directory with label NIfTIs matching images."""
    labels_dir = temp_dir / "labelsTr"
    labels_dir.mkdir()

    affine_2mm = np.diag([2.0, 2.0, 2.0, 1.0])
    shape_2mm = (32, 32, 16)

    for i in range(3):
        label = np.zeros(shape_2mm, dtype=np.int16)
        if i < 2:  # First 2 have tumors
            label[10:20, 10:20, 5:10] = 1
        img = nib.Nifti1Image(label, affine_2mm)
        img.header.set_qform(affine_2mm, code=1)
        nib.save(img, labels_dir / f"lung_{i:03d}.nii.gz")

    return labels_dir


@pytest.fixture
def mixed_shape_niftis(temp_dir: Path) -> Path:
    """Create NIfTIs with different shapes (heterogeneous collection)."""
    images_dir = temp_dir / "mixed_shapes"
    images_dir.mkdir()

    shapes = [(32, 32, 16), (48, 48, 24), (64, 64, 32)]
    affine = np.eye(4)

    for i, shape in enumerate(shapes):
        data = np.random.rand(*shape).astype(np.float32)
        img = nib.Nifti1Image(data, affine)
        img.header.set_qform(affine, code=1)
        nib.save(img, images_dir / f"vol_{i:03d}.nii.gz")

    return images_dir


# ===== NiftiSource Tests =====


class TestNiftiSource:
    """Tests for NiftiSource dataclass."""

    def test_has_label_true(self, temp_dir: Path) -> None:
        source = NiftiSource(
            image_path=temp_dir / "image.nii.gz",
            subject_id="sub-01",
            label_path=temp_dir / "label.nii.gz",
        )
        assert source.has_label is True

    def test_has_label_false(self, temp_dir: Path) -> None:
        source = NiftiSource(
            image_path=temp_dir / "image.nii.gz",
            subject_id="sub-01",
        )
        assert source.has_label is False


# ===== discover_nifti_pairs Tests =====


class TestDiscoverNiftiPairs:
    """Tests for discover_nifti_pairs."""

    def test_discovers_images(self, nifti_dir: Path) -> None:
        sources = discover_nifti_pairs(nifti_dir)
        assert len(sources) == 3

    def test_pairs_with_labels(self, nifti_dir: Path, label_dir: Path) -> None:
        sources = discover_nifti_pairs(nifti_dir, label_dir)
        assert len(sources) == 3
        assert all(s.has_label for s in sources)

    def test_extracts_subject_id(self, nifti_dir: Path) -> None:
        sources = discover_nifti_pairs(nifti_dir)
        subject_ids = {s.subject_id for s in sources}
        assert subject_ids == {"lung_000", "lung_001", "lung_002"}

    def test_raises_on_missing_dir(self, temp_dir: Path) -> None:
        with pytest.raises(FileNotFoundError):
            discover_nifti_pairs(temp_dir / "nonexistent")

    def test_custom_subject_id_fn(self, nifti_dir: Path) -> None:
        def custom_fn(p: Path) -> str:
            name = p.stem
            if name.endswith(".nii"):
                name = name[:-4]
            return f"custom_{name.split('_')[1]}"

        sources = discover_nifti_pairs(nifti_dir, subject_id_fn=custom_fn)
        subject_ids = {s.subject_id for s in sources}
        assert "custom_000" in subject_ids


# ===== Raw Ingestion Tests =====


class TestRawIngestion:
    """Tests for RadiObject.from_niftis with raw data storage."""

    def test_from_image_dir(self, temp_dir: Path, nifti_dir: Path) -> None:
        uri = str(temp_dir / "radi_from_dir")

        radi = RadiObject.from_niftis(
            uri=uri,
            image_dir=nifti_dir,
        )

        assert len(radi) == 3
        # lung_xxx files infer to CT modality
        assert "CT" in radi.collection_names

    def test_explicit_collection_name(self, temp_dir: Path, nifti_dir: Path) -> None:
        uri = str(temp_dir / "radi_named")

        radi = RadiObject.from_niftis(
            uri=uri,
            image_dir=nifti_dir,
            collection_name="lung_ct",
        )

        assert "lung_ct" in radi.collection_names
        assert len(radi.collection_names) == 1

    def test_heterogeneous_shapes_allowed(self, temp_dir: Path, mixed_shape_niftis: Path) -> None:
        uri = str(temp_dir / "radi_mixed")

        radi = RadiObject.from_niftis(
            uri=uri,
            image_dir=mixed_shape_niftis,
            collection_name="mixed",
        )

        vc = radi.collection("mixed")
        # Shape is None for heterogeneous collections
        assert vc.shape is None
        assert vc.is_uniform is False
        assert len(vc) == 3

    def test_uniform_collection_has_shape(self, temp_dir: Path, nifti_dir: Path) -> None:
        uri = str(temp_dir / "radi_uniform")

        radi = RadiObject.from_niftis(
            uri=uri,
            image_dir=nifti_dir,
            collection_name="ct",
        )

        vc = radi.collection("ct")
        assert vc.shape is not None
        assert vc.is_uniform is True
        assert vc.shape == (32, 32, 16)

    def test_mutually_exclusive_inputs(self, temp_dir: Path, nifti_dir: Path) -> None:
        uri = str(temp_dir / "radi_error")
        nifti_path = list(nifti_dir.glob("*.nii.gz"))[0]

        with pytest.raises(ValueError, match="Cannot specify both"):
            RadiObject.from_niftis(
                uri=uri,
                niftis=[(nifti_path, "sub-01")],
                image_dir=nifti_dir,
            )

    def test_requires_some_input(self, temp_dir: Path) -> None:
        uri = str(temp_dir / "radi_no_input")

        with pytest.raises(ValueError, match="Must specify either"):
            RadiObject.from_niftis(uri=uri)

    def test_original_niftis_api(
        self, temp_dir: Path, synthetic_nifti_files: list[tuple[Path, str]]
    ) -> None:
        """Verify tuple-based API works."""
        uri = str(temp_dir / "radi_original")

        radi = RadiObject.from_niftis(
            uri=uri,
            niftis=synthetic_nifti_files,
        )

        assert len(radi) == 3
        assert "T1w" in radi.collection_names


# ===== obs_id Uniqueness Tests =====


class TestObsIdUniqueness:
    """Tests for obs_id uniqueness across RadiObject."""

    def test_all_obs_ids_property(
        self, temp_dir: Path, synthetic_nifti_files: list[tuple[Path, str]]
    ) -> None:
        uri = str(temp_dir / "radi_obs_ids")

        radi = RadiObject.from_niftis(
            uri=uri,
            niftis=synthetic_nifti_files,
        )

        # Should have 6 volumes (3 subjects × 2 modalities)
        all_ids = radi.all_obs_ids
        assert len(all_ids) == 6
        # All should be unique
        assert len(set(all_ids)) == 6

    def test_get_volume_by_obs_id(
        self, temp_dir: Path, synthetic_nifti_files: list[tuple[Path, str]]
    ) -> None:
        uri = str(temp_dir / "radi_get_vol")

        radi = RadiObject.from_niftis(
            uri=uri,
            niftis=synthetic_nifti_files,
        )

        # Get first obs_id from T1w collection
        t1w_coll = radi.collection("T1w")
        first_obs_id = t1w_coll.obs_ids[0]

        vol = radi.get_volume(first_obs_id)
        assert vol is not None
        assert vol.to_numpy().shape == (32, 32, 16)

    def test_get_volume_not_found(
        self, temp_dir: Path, synthetic_nifti_files: list[tuple[Path, str]]
    ) -> None:
        uri = str(temp_dir / "radi_not_found")

        radi = RadiObject.from_niftis(
            uri=uri,
            niftis=synthetic_nifti_files,
        )

        with pytest.raises(KeyError, match="not found"):
            radi.get_volume("nonexistent_id")


# ===== rename_collection Tests =====


class TestRenameCollection:
    """Tests for RadiObject.rename_collection()."""

    def test_rename_collection(
        self, temp_dir: Path, synthetic_nifti_files: list[tuple[Path, str]]
    ) -> None:
        uri = str(temp_dir / "radi_rename")

        radi = RadiObject.from_niftis(
            uri=uri,
            niftis=synthetic_nifti_files,
        )

        assert "T1w" in radi.collection_names
        radi.rename_collection("T1w", "anatomical")

        assert "anatomical" in radi.collection_names
        assert "T1w" not in radi.collection_names

    def test_rename_nonexistent_raises(
        self, temp_dir: Path, synthetic_nifti_files: list[tuple[Path, str]]
    ) -> None:
        uri = str(temp_dir / "radi_rename_err")

        radi = RadiObject.from_niftis(
            uri=uri,
            niftis=synthetic_nifti_files,
        )

        with pytest.raises(KeyError, match="not found"):
            radi.rename_collection("nonexistent", "new_name")

    def test_rename_to_existing_raises(
        self, temp_dir: Path, synthetic_nifti_files: list[tuple[Path, str]]
    ) -> None:
        uri = str(temp_dir / "radi_rename_dup")

        radi = RadiObject.from_niftis(
            uri=uri,
            niftis=synthetic_nifti_files,
        )

        with pytest.raises(ValueError, match="already exists"):
            radi.rename_collection("T1w", "FLAIR")


# ===== Fixtures =====


@pytest.fixture
def synthetic_nifti_files(temp_dir: Path) -> list[tuple[Path, str]]:
    """Create synthetic NIfTI files for testing."""
    niftis = []
    shape = (32, 32, 16)
    affine = np.eye(4)

    for subject_id in ["sub-01", "sub-02", "sub-03"]:
        for series_type in ["T1w", "FLAIR"]:
            data = np.random.rand(*shape).astype(np.float32)
            img = nib.Nifti1Image(data, affine)
            img.header.set_qform(affine, code=1)
            img.header.set_sform(affine, code=1)

            filename = f"{subject_id}_{series_type}.nii.gz"
            filepath = temp_dir / filename
            nib.save(img, filepath)
            niftis.append((filepath, subject_id))

    return niftis
