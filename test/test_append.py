"""Tests for atomic append operations on RadiObject and VolumeCollection."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from radiobject.radi_object import RadiObject
from radiobject.volume_collection import VolumeCollection


class TestVolumeCollectionAppendNiftis:
    """Tests for VolumeCollection.append() with NIfTI files."""

    def test_append_niftis_basic(
        self,
        populated_collection: VolumeCollection,
        create_test_nifti,
    ):
        """append() adds new volumes to existing collection."""
        initial_count = len(populated_collection)

        new_nifti_path, _ = create_test_nifti("new_volume.nii.gz")

        populated_collection.append(
            niftis=[(new_nifti_path, "sub-NEW")],
        )

        assert len(populated_collection) == initial_count + 1

    def test_append_updates_metadata(
        self,
        populated_collection: VolumeCollection,
        create_test_nifti,
    ):
        """append() updates n_volumes metadata."""
        new_nifti_path, _ = create_test_nifti("metadata_test.nii.gz")

        initial_count = len(populated_collection)
        populated_collection.append(niftis=[(new_nifti_path, "sub-META")])

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
        wrong_shape = (64, 64, 32)
        data = np.random.randn(*wrong_shape).astype(np.float32)

        new_nifti_path = temp_dir / "wrong_shape.nii.gz"
        new_img = nib.Nifti1Image(data, np.eye(4))
        nib.save(new_img, new_nifti_path)

        with pytest.raises(ValueError, match="Dimension mismatch"):
            populated_collection.append(niftis=[(new_nifti_path, "sub-BAD")])

    def test_append_duplicate_subject_auto_indexes_acquisition(
        self,
        temp_dir: Path,
        create_test_nifti,
    ):
        """append() auto-indexes acquisitions when subject repeats in same collection."""
        initial_nifti, _ = create_test_nifti("sub-001_T1w.nii.gz")

        vc_uri = str(temp_dir / "dup_test_collection")
        vc = VolumeCollection.from_niftis(
            uri=vc_uri,
            niftis=[(initial_nifti, "sub-001")],
            name="T1w",
        )

        duplicate_nifti, data = create_test_nifti("another_T1w.nii.gz")

        vc.append(niftis=[(duplicate_nifti, "sub-001")])
        assert len(vc) == 2
        obs_ids = vc.obs_ids
        assert "sub-001_T1w" in obs_ids
        assert "sub-001_T1w_acq-1" in obs_ids

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
        populated_radi_object: RadiObject,
        create_test_nifti,
    ):
        """append() adds new subjects and their volumes."""
        initial_subject_count = len(populated_radi_object)

        new_nifti_path, _ = create_test_nifti("sub-NEW_T1w.nii.gz")

        obs_meta = pd.DataFrame({"obs_subject_id": ["sub-NEW"]})

        populated_radi_object.append(
            images={"T1w": [(new_nifti_path, "sub-NEW")]},
            obs_meta=obs_meta,
        )

        assert len(populated_radi_object) == initial_subject_count + 1
        assert "sub-NEW" in populated_radi_object.obs_subject_ids


class TestRadiObjectAppendExistingSubjects:
    """Tests for RadiObject.append() with existing subjects."""

    def test_append_to_existing_subjects(
        self,
        populated_radi_object: RadiObject,
        create_test_nifti,
    ):
        """append() can add volumes for existing subjects without obs_meta."""
        existing_subject_id = populated_radi_object.obs_subject_ids[0]
        initial_subject_count = len(populated_radi_object)

        new_nifti_path, _ = create_test_nifti(f"{existing_subject_id}_DWI.nii.gz")

        populated_radi_object.append(
            images={"DWI": [(new_nifti_path, existing_subject_id)]},
        )

        assert len(populated_radi_object) == initial_subject_count


class TestRadiObjectAppendValidation:
    """Tests for RadiObject.append() validation."""

    def test_append_new_subject_without_obs_meta_raises(
        self,
        populated_radi_object: RadiObject,
        create_test_nifti,
    ):
        """append() raises when new subject lacks obs_meta."""
        new_nifti_path, _ = create_test_nifti("sub-NOMETA_T1w.nii.gz")

        with pytest.raises(ValueError, match="obs_meta required"):
            populated_radi_object.append(
                images={"T1w": [(new_nifti_path, "sub-NOMETA")]},
            )

    def test_append_obs_meta_missing_subject_raises(
        self,
        populated_radi_object: RadiObject,
        create_test_nifti,
    ):
        """append() raises when obs_meta doesn't include all new subjects."""
        new_nifti_path, _ = create_test_nifti("sub-MISSING_T1w.nii.gz")

        obs_meta = pd.DataFrame({"obs_subject_id": ["sub-OTHER"]})

        with pytest.raises(ValueError, match="missing entries"):
            populated_radi_object.append(
                images={"T1w": [(new_nifti_path, "sub-MISSING")]},
                obs_meta=obs_meta,
            )

    def test_append_obs_meta_missing_column_raises(
        self,
        populated_radi_object: RadiObject,
        create_test_nifti,
    ):
        """append() raises when obs_meta lacks obs_subject_id column."""
        new_nifti_path, _ = create_test_nifti("sub-BADCOL_T1w.nii.gz")

        bad_obs_meta = pd.DataFrame({"subject_name": ["sub-BADCOL"]})

        with pytest.raises(ValueError, match="obs_subject_id"):
            populated_radi_object.append(
                images={"T1w": [(new_nifti_path, "sub-BADCOL")]},
                obs_meta=bad_obs_meta,
            )

    def test_append_empty_images_raises(
        self,
        populated_radi_object: RadiObject,
    ):
        """append() requires non-empty images dict."""
        with pytest.raises(ValueError, match="images dict cannot be empty"):
            populated_radi_object.append(images={})


class TestRadiObjectAppendNewCollection:
    """Tests for RadiObject.append() creating new collections."""

    def test_append_creates_new_collection(
        self,
        populated_radi_object: RadiObject,
        create_test_nifti,
    ):
        """append() creates new collection when collection name is new."""
        existing_subject_id = populated_radi_object.obs_subject_ids[0]
        initial_collections = set(populated_radi_object.collection_names)

        new_nifti_path, _ = create_test_nifti(f"{existing_subject_id}_PET.nii.gz")

        populated_radi_object.append(
            images={"PET": [(new_nifti_path, existing_subject_id)]},
        )

        new_collections = set(populated_radi_object.collection_names)
        assert "PET" in new_collections - initial_collections


class TestAppendAtomicity:
    """Tests for append operation atomicity."""

    def test_append_writes_obs_and_volumes_together(
        self,
        populated_radi_object: RadiObject,
        create_test_nifti,
    ):
        """append() writes obs_meta and volumes atomically."""
        new_nifti_path, _ = create_test_nifti("sub-ATOMIC_T1w.nii.gz")

        obs_meta = pd.DataFrame({"obs_subject_id": ["sub-ATOMIC"], "age": [42]})

        populated_radi_object.append(
            images={"T1w": [(new_nifti_path, "sub-ATOMIC")]},
            obs_meta=obs_meta,
        )

        full_obs_meta = populated_radi_object.obs_meta.read()
        assert "sub-ATOMIC" in full_obs_meta["obs_subject_id"].values
        assert "sub-ATOMIC" in populated_radi_object.obs_subject_ids


class TestAppendCacheInvalidation:
    """Tests for proper cache invalidation after append."""

    def test_index_cache_invalidated(
        self,
        populated_radi_object: RadiObject,
        create_test_nifti,
    ):
        """append() invalidates index cache."""
        _ = populated_radi_object.obs_subject_ids
        initial_count = len(populated_radi_object)

        new_nifti_path, _ = create_test_nifti("sub-CACHE_T1w.nii.gz")

        obs_meta = pd.DataFrame({"obs_subject_id": ["sub-CACHE"]})

        populated_radi_object.append(
            images={"T1w": [(new_nifti_path, "sub-CACHE")]},
            obs_meta=obs_meta,
        )

        assert len(populated_radi_object) == initial_count + 1
        assert "sub-CACHE" in populated_radi_object.obs_subject_ids

    def test_collection_names_cache_invalidated(
        self,
        populated_radi_object: RadiObject,
        create_test_nifti,
    ):
        """append() invalidates collection_names cache when new collection created."""
        initial_names = populated_radi_object.collection_names
        existing_subject_id = populated_radi_object.obs_subject_ids[0]

        new_nifti_path, _ = create_test_nifti(f"{existing_subject_id}_ASL.nii.gz")

        populated_radi_object.append(
            images={"ASL": [(new_nifti_path, existing_subject_id)]},
        )

        new_names = populated_radi_object.collection_names
        assert len(new_names) >= len(initial_names)
