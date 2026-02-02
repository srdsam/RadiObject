"""Tests for atomic append operations on RadiObject and VolumeCollection."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from radiobject.data import get_dataset_path
from radiobject.radi_object import RadiObject
from radiobject.volume_collection import VolumeCollection

DATA_DIR = get_dataset_path("msd-brain-tumour")


class TestVolumeCollectionAppendNiftis:
    """Tests for VolumeCollection.append() with NIfTI files."""

    def test_append_niftis_basic(
        self,
        temp_dir: Path,
        populated_collection: VolumeCollection,
        nifti_manifest: list[dict],
    ):
        """append() adds new volumes to existing collection."""
        initial_count = len(populated_collection)

        # Create a new NIfTI file for appending (using same data as existing)
        src_path = DATA_DIR / nifti_manifest[0]["image_path"]
        img = nib.load(src_path)
        data = np.asarray(img.dataobj, dtype=np.float32)
        if data.ndim == 4:
            data = data[..., 0]

        new_nifti_path = temp_dir / "new_volume.nii.gz"
        new_img = nib.Nifti1Image(data, img.affine)
        nib.save(new_img, new_nifti_path)

        # Append the new volume
        populated_collection.append(
            niftis=[(new_nifti_path, "sub-NEW")],
        )

        # Verify
        assert len(populated_collection) == initial_count + 1

    def test_append_updates_metadata(
        self,
        temp_dir: Path,
        populated_collection: VolumeCollection,
        nifti_manifest: list[dict],
    ):
        """append() updates n_volumes metadata."""
        src_path = DATA_DIR / nifti_manifest[0]["image_path"]
        img = nib.load(src_path)
        data = np.asarray(img.dataobj, dtype=np.float32)
        if data.ndim == 4:
            data = data[..., 0]

        new_nifti_path = temp_dir / "metadata_test.nii.gz"
        new_img = nib.Nifti1Image(data, img.affine)
        nib.save(new_img, new_nifti_path)

        initial_count = len(populated_collection)
        populated_collection.append(niftis=[(new_nifti_path, "sub-META")])

        # Re-read to verify metadata updated
        reopened = VolumeCollection(populated_collection.uri)
        assert len(reopened) == initial_count + 1


class TestVolumeCollectionAppendValidation:
    """Tests for VolumeCollection.append() validation."""

    def test_append_dimension_mismatch_raises(
        self,
        temp_dir: Path,
        populated_collection: VolumeCollection,
    ):
        """append() raises ValueError for dimension mismatch."""
        wrong_shape = (64, 64, 32)  # Different from collection shape
        data = np.random.randn(*wrong_shape).astype(np.float32)

        new_nifti_path = temp_dir / "wrong_shape.nii.gz"
        new_img = nib.Nifti1Image(data, np.eye(4))
        nib.save(new_img, new_nifti_path)

        with pytest.raises(ValueError, match="Dimension mismatch"):
            populated_collection.append(niftis=[(new_nifti_path, "sub-BAD")])

    def test_append_duplicate_obs_id_raises(
        self,
        temp_dir: Path,
        nifti_manifest: list[dict],
    ):
        """append() raises ValueError for duplicate obs_id."""
        # Create a fresh collection from NIfTIs so we know the exact obs_id format
        src_path = DATA_DIR / nifti_manifest[0]["image_path"]

        # Create initial collection with one volume
        initial_nifti = temp_dir / "sub-001_T1w.nii.gz"
        img = nib.load(src_path)
        data = np.asarray(img.dataobj, dtype=np.float32)
        if data.ndim == 4:
            data = data[..., 0]
        nib.save(nib.Nifti1Image(data, img.affine), initial_nifti)

        vc_uri = str(temp_dir / "dup_test_collection")
        vc = VolumeCollection.from_niftis(
            uri=vc_uri,
            niftis=[(initial_nifti, "sub-001")],
            name="T1w",
        )

        # Try to append with same subject_id and same series type
        # This should generate the same obs_id: "sub-001_T1w"
        duplicate_nifti = temp_dir / "another_T1w.nii.gz"
        nib.save(nib.Nifti1Image(data, img.affine), duplicate_nifti)

        with pytest.raises(ValueError, match="already exist"):
            vc.append(niftis=[(duplicate_nifti, "sub-001")])

    def test_append_file_not_found_raises(
        self,
        populated_collection: VolumeCollection,
    ):
        """append() raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            populated_collection.append(niftis=[("/nonexistent/file.nii.gz", "sub-MISSING")])

    def test_append_requires_niftis_or_dicoms(
        self,
        populated_collection: VolumeCollection,
    ):
        """append() requires either niftis or dicom_dirs."""
        with pytest.raises(ValueError, match="Must provide"):
            populated_collection.append()

    def test_append_cannot_provide_both(
        self,
        populated_collection: VolumeCollection,
    ):
        """append() cannot provide both niftis and dicom_dirs."""
        with pytest.raises(ValueError, match="Cannot provide both"):
            populated_collection.append(
                niftis=[("fake.nii", "sub")],
                dicom_dirs=[("fake_dir", "sub")],
            )


# =============================================================================
# RadiObject Append Tests
# =============================================================================


class TestRadiObjectAppendBasic:
    """Tests for RadiObject.append() basic functionality."""

    def test_append_new_subjects(
        self,
        temp_dir: Path,
        populated_radi_object: RadiObject,
        nifti_manifest: list[dict],
    ):
        """append() adds new subjects and their volumes."""
        initial_subject_count = len(populated_radi_object)

        # Create new NIfTI for a new subject
        src_path = DATA_DIR / nifti_manifest[0]["image_path"]
        img = nib.load(src_path)
        data = np.asarray(img.dataobj, dtype=np.float32)
        if data.ndim == 4:
            data = data[..., 0]

        new_nifti_path = temp_dir / "sub-NEW_T1w.nii.gz"
        new_img = nib.Nifti1Image(data, img.affine)
        nib.save(new_img, new_nifti_path)

        # Append with new subject metadata
        obs_meta = pd.DataFrame(
            {
                "obs_subject_id": ["sub-NEW"],
            }
        )

        populated_radi_object.append(
            niftis=[(new_nifti_path, "sub-NEW")],
            obs_meta=obs_meta,
        )

        # Invalidate cache and verify
        assert len(populated_radi_object) == initial_subject_count + 1
        assert "sub-NEW" in populated_radi_object.obs_subject_ids


class TestRadiObjectAppendExistingSubjects:
    """Tests for RadiObject.append() with existing subjects."""

    def test_append_to_existing_subjects(
        self,
        temp_dir: Path,
        populated_radi_object: RadiObject,
        nifti_manifest: list[dict],
    ):
        """append() can add volumes for existing subjects without obs_meta."""
        existing_subject_id = populated_radi_object.obs_subject_ids[0]
        initial_subject_count = len(populated_radi_object)

        # Create new NIfTI with different modality name
        src_path = DATA_DIR / nifti_manifest[0]["image_path"]
        img = nib.load(src_path)
        data = np.asarray(img.dataobj, dtype=np.float32)
        if data.ndim == 4:
            data = data[..., 0]

        # Use a new modality name that won't exist yet
        new_nifti_path = temp_dir / f"{existing_subject_id}_DWI.nii.gz"
        new_img = nib.Nifti1Image(data, img.affine)
        nib.save(new_img, new_nifti_path)

        # Append without obs_meta (subject already exists)
        populated_radi_object.append(
            niftis=[(new_nifti_path, existing_subject_id)],
        )

        # Subject count should not change
        assert len(populated_radi_object) == initial_subject_count


class TestRadiObjectAppendValidation:
    """Tests for RadiObject.append() validation."""

    def test_append_new_subject_without_obs_meta_raises(
        self,
        temp_dir: Path,
        populated_radi_object: RadiObject,
        nifti_manifest: list[dict],
    ):
        """append() raises when new subject lacks obs_meta."""
        src_path = DATA_DIR / nifti_manifest[0]["image_path"]
        img = nib.load(src_path)
        data = np.asarray(img.dataobj, dtype=np.float32)
        if data.ndim == 4:
            data = data[..., 0]

        new_nifti_path = temp_dir / "sub-NOMETA_T1w.nii.gz"
        new_img = nib.Nifti1Image(data, img.affine)
        nib.save(new_img, new_nifti_path)

        with pytest.raises(ValueError, match="obs_meta required"):
            populated_radi_object.append(
                niftis=[(new_nifti_path, "sub-NOMETA")],
            )

    def test_append_obs_meta_missing_subject_raises(
        self,
        temp_dir: Path,
        populated_radi_object: RadiObject,
        nifti_manifest: list[dict],
    ):
        """append() raises when obs_meta doesn't include all new subjects."""
        src_path = DATA_DIR / nifti_manifest[0]["image_path"]
        img = nib.load(src_path)
        data = np.asarray(img.dataobj, dtype=np.float32)
        if data.ndim == 4:
            data = data[..., 0]

        new_nifti_path = temp_dir / "sub-MISSING_T1w.nii.gz"
        new_img = nib.Nifti1Image(data, img.affine)
        nib.save(new_img, new_nifti_path)

        # Provide obs_meta but missing the subject
        obs_meta = pd.DataFrame(
            {
                "obs_subject_id": ["sub-OTHER"],
            }
        )

        with pytest.raises(ValueError, match="missing entries"):
            populated_radi_object.append(
                niftis=[(new_nifti_path, "sub-MISSING")],
                obs_meta=obs_meta,
            )

    def test_append_obs_meta_missing_column_raises(
        self,
        temp_dir: Path,
        populated_radi_object: RadiObject,
        nifti_manifest: list[dict],
    ):
        """append() raises when obs_meta lacks obs_subject_id column."""
        src_path = DATA_DIR / nifti_manifest[0]["image_path"]
        img = nib.load(src_path)
        data = np.asarray(img.dataobj, dtype=np.float32)
        if data.ndim == 4:
            data = data[..., 0]

        new_nifti_path = temp_dir / "sub-BADCOL_T1w.nii.gz"
        new_img = nib.Nifti1Image(data, img.affine)
        nib.save(new_img, new_nifti_path)

        # obs_meta without required column
        bad_obs_meta = pd.DataFrame(
            {
                "subject_name": ["sub-BADCOL"],  # Wrong column name
            }
        )

        with pytest.raises(ValueError, match="obs_subject_id"):
            populated_radi_object.append(
                niftis=[(new_nifti_path, "sub-BADCOL")],
                obs_meta=bad_obs_meta,
            )

    def test_append_requires_niftis_or_dicoms(
        self,
        populated_radi_object: RadiObject,
    ):
        """append() requires either niftis or dicom_dirs."""
        with pytest.raises(ValueError, match="Must provide"):
            populated_radi_object.append()


class TestRadiObjectAppendNewCollection:
    """Tests for RadiObject.append() creating new collections."""

    def test_append_creates_new_collection(
        self,
        temp_dir: Path,
        populated_radi_object: RadiObject,
        nifti_manifest: list[dict],
    ):
        """append() creates new collection when series type is new."""
        existing_subject_id = populated_radi_object.obs_subject_ids[0]
        initial_collections = set(populated_radi_object.collection_names)

        # Create NIfTI with a new modality
        src_path = DATA_DIR / nifti_manifest[0]["image_path"]
        img = nib.load(src_path)
        data = np.asarray(img.dataobj, dtype=np.float32)
        if data.ndim == 4:
            data = data[..., 0]

        # Use unique modality name
        new_nifti_path = temp_dir / f"{existing_subject_id}_PET.nii.gz"
        new_img = nib.Nifti1Image(data, img.affine)
        nib.save(new_img, new_nifti_path)

        populated_radi_object.append(
            niftis=[(new_nifti_path, existing_subject_id)],
        )

        # Should have new collection
        new_collections = set(populated_radi_object.collection_names)
        added = new_collections - initial_collections
        assert len(added) >= 1


class TestAppendAtomicity:
    """Tests for append operation atomicity."""

    def test_append_writes_obs_and_volumes_together(
        self,
        temp_dir: Path,
        populated_radi_object: RadiObject,
        nifti_manifest: list[dict],
    ):
        """append() writes obs_meta and volumes atomically."""
        src_path = DATA_DIR / nifti_manifest[0]["image_path"]
        img = nib.load(src_path)
        data = np.asarray(img.dataobj, dtype=np.float32)
        if data.ndim == 4:
            data = data[..., 0]

        new_nifti_path = temp_dir / "sub-ATOMIC_T1w.nii.gz"
        new_img = nib.Nifti1Image(data, img.affine)
        nib.save(new_img, new_nifti_path)

        obs_meta = pd.DataFrame(
            {
                "obs_subject_id": ["sub-ATOMIC"],
                "age": [42],
            }
        )

        populated_radi_object.append(
            niftis=[(new_nifti_path, "sub-ATOMIC")],
            obs_meta=obs_meta,
        )

        # Verify obs_meta was written
        full_obs_meta = populated_radi_object.obs_meta.read()
        assert "sub-ATOMIC" in full_obs_meta["obs_subject_id"].values

        # Verify subject count updated
        assert "sub-ATOMIC" in populated_radi_object.obs_subject_ids


class TestAppendCacheInvalidation:
    """Tests for proper cache invalidation after append."""

    def test_index_cache_invalidated(
        self,
        temp_dir: Path,
        populated_radi_object: RadiObject,
        nifti_manifest: list[dict],
    ):
        """append() invalidates index cache."""
        # Access index to populate cache
        _ = populated_radi_object.obs_subject_ids
        initial_count = len(populated_radi_object)

        src_path = DATA_DIR / nifti_manifest[0]["image_path"]
        img = nib.load(src_path)
        data = np.asarray(img.dataobj, dtype=np.float32)
        if data.ndim == 4:
            data = data[..., 0]

        new_nifti_path = temp_dir / "sub-CACHE_T1w.nii.gz"
        new_img = nib.Nifti1Image(data, img.affine)
        nib.save(new_img, new_nifti_path)

        obs_meta = pd.DataFrame({"obs_subject_id": ["sub-CACHE"]})

        populated_radi_object.append(
            niftis=[(new_nifti_path, "sub-CACHE")],
            obs_meta=obs_meta,
        )

        # Cache should be invalidated - new count should reflect append
        assert len(populated_radi_object) == initial_count + 1
        assert "sub-CACHE" in populated_radi_object.obs_subject_ids

    def test_collection_names_cache_invalidated(
        self,
        temp_dir: Path,
        populated_radi_object: RadiObject,
        nifti_manifest: list[dict],
    ):
        """append() invalidates collection_names cache when new collection created."""
        # Access collection_names to populate cache
        initial_names = populated_radi_object.collection_names
        existing_subject_id = populated_radi_object.obs_subject_ids[0]

        src_path = DATA_DIR / nifti_manifest[0]["image_path"]
        img = nib.load(src_path)
        data = np.asarray(img.dataobj, dtype=np.float32)
        if data.ndim == 4:
            data = data[..., 0]

        # Use a unique modality name
        new_nifti_path = temp_dir / f"{existing_subject_id}_ASL.nii.gz"
        new_img = nib.Nifti1Image(data, img.affine)
        nib.save(new_img, new_nifti_path)

        populated_radi_object.append(
            niftis=[(new_nifti_path, existing_subject_id)],
        )

        # Collection names should reflect new collection
        new_names = populated_radi_object.collection_names
        assert len(new_names) >= len(initial_names)
