"""Tests for obs validation, obs_meta 1-dim PK, user-provided obs, and obs_ids auto-population."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from radiobject.dataframe import INDEX_COLUMNS, Dataframe
from radiobject.utils import ensure_obs_columns, validate_no_column_collisions

# --- Shared fixtures ---


@pytest.fixture
def synthetic_niftis(tmp_path):
    """Create 2 minimal synthetic NIfTI files as (path, subject_id) list."""
    import nibabel as nib

    paths = []
    for sid in ["sub-01", "sub-02"]:
        data = np.random.rand(4, 4, 4).astype(np.float32)
        img = nib.Nifti1Image(data, np.eye(4))
        path = tmp_path / f"{sid}_T1w.nii.gz"
        nib.save(img, path)
        paths.append((str(path), sid))
    return paths


@pytest.fixture
def synthetic_nifti_images(tmp_path):
    """Create synthetic NIfTI images as RadiObject images dict."""
    import nibabel as nib

    paths: dict[str, list[tuple[str, str]]] = {"T1w": []}
    for sid in ["sub-01", "sub-02"]:
        data = np.random.rand(4, 4, 4).astype(np.float32)
        img = nib.Nifti1Image(data, np.eye(4))
        path = tmp_path / f"{sid}_T1w.nii.gz"
        nib.save(img, path)
        paths["T1w"].append((str(path), sid))
    return paths


# --- Utility tests ---


class TestEnsureObsColumns:
    def test_valid_with_obs_subject_id(self):
        df = pd.DataFrame({"obs_subject_id": ["s1", "s2"]})
        result = ensure_obs_columns(df)
        assert len(result) == 2

    def test_raises_missing_obs_subject_id(self):
        df = pd.DataFrame({"other": ["s1"]})
        with pytest.raises(ValueError, match="obs_subject_id"):
            ensure_obs_columns(df)

    def test_raises_missing_obs_id_when_required(self):
        df = pd.DataFrame({"obs_subject_id": ["s1"]})
        with pytest.raises(ValueError, match="obs_id"):
            ensure_obs_columns(df, require_obs_id=True)

    def test_passes_when_obs_id_not_required(self):
        df = pd.DataFrame({"obs_subject_id": ["s1"]})
        result = ensure_obs_columns(df, require_obs_id=False)
        assert len(result) == 1

    def test_context_in_error_message(self):
        df = pd.DataFrame({"other": ["s1"]})
        with pytest.raises(ValueError, match="my_context"):
            ensure_obs_columns(df, context="my_context")


class TestValidateNoColumnCollisions:
    def test_no_collision(self):
        validate_no_column_collisions({"age", "sex"}, {"series_type", "voxel_spacing"})

    def test_collision_raises(self):
        with pytest.raises(ValueError, match="collide"):
            validate_no_column_collisions({"series_type", "age"}, {"series_type", "voxel_spacing"})

    def test_index_columns_excluded_from_collision(self):
        validate_no_column_collisions(
            {"obs_id", "obs_subject_id", "age"},
            {"obs_id", "obs_subject_id", "series_type"},
        )

    def test_context_in_error_message(self):
        with pytest.raises(ValueError, match="test_ctx"):
            validate_no_column_collisions({"series_type"}, {"series_type"}, context="test_ctx")


# --- Dataframe schema tests ---


class TestDataframeIndexColumns:
    def test_create_single_dim(self, tmp_path):
        uri = str(tmp_path / "single_dim")
        schema = {"age": np.dtype("int32")}
        df = Dataframe.create(uri, schema=schema, index_columns=("obs_subject_id",))
        assert df.index_columns == ("obs_subject_id",)

    def test_create_default_two_dims(self, tmp_path):
        uri = str(tmp_path / "two_dim")
        schema = {"age": np.dtype("int32")}
        df = Dataframe.create(uri, schema=schema)
        assert df.index_columns == INDEX_COLUMNS

    def test_read_single_dim(self, tmp_path):
        uri = str(tmp_path / "single_read")
        data = pd.DataFrame({"obs_subject_id": ["s1", "s2"], "age": [30, 40]})
        df = Dataframe.from_pandas(uri, data, index_columns=("obs_subject_id",))
        result = df.read()
        assert "obs_subject_id" in result.columns
        assert "obs_id" not in result.columns
        assert len(result) == 2
        assert set(result["obs_subject_id"]) == {"s1", "s2"}

    def test_index_columns_property_reflects_schema(self, tmp_path):
        uri = str(tmp_path / "reflect")
        schema = {"val": np.dtype("float64")}
        _ = Dataframe.create(uri, schema=schema, index_columns=("obs_subject_id",))
        # Re-open from URI to verify schema round-trip
        df2 = Dataframe(uri)
        assert df2.index_columns == ("obs_subject_id",)

    def test_from_pandas_single_dim(self, tmp_path):
        uri = str(tmp_path / "from_pandas_single")
        data = pd.DataFrame(
            {"obs_subject_id": ["s1", "s2"], "age": np.array([30, 40], dtype=np.int32)}
        )
        df = Dataframe.from_pandas(uri, data, index_columns=("obs_subject_id",))
        assert len(df) == 2
        result = df.read()
        assert list(result.columns) == ["obs_subject_id", "age"]


# --- VolumeCollection user-provided obs tests ---


class TestVolumeCollectionCustomObs:
    """Tests requiring real NIfTI data — skipped if dataset not available."""

    def test_from_niftis_custom_obs_ids(self, tmp_path, synthetic_niftis):
        from radiobject.volume_collection import VolumeCollection

        obs = pd.DataFrame(
            {
                "obs_id": ["custom_01", "custom_02"],
                "obs_subject_id": ["sub-01", "sub-02"],
            }
        )
        uri = str(tmp_path / "custom_vc")
        vc = VolumeCollection.from_niftis(
            uri=uri,
            niftis=synthetic_niftis,
            obs=obs,
            validate_dimensions=False,
        )
        assert set(vc.obs_ids) == {"custom_01", "custom_02"}

    def test_from_niftis_extra_columns_stored(self, tmp_path, synthetic_niftis):
        from radiobject.volume_collection import VolumeCollection

        obs = pd.DataFrame(
            {
                "obs_id": ["custom_01", "custom_02"],
                "obs_subject_id": ["sub-01", "sub-02"],
                "scan_date": ["2024-01-01", "2024-01-02"],
            }
        )
        uri = str(tmp_path / "extra_cols_vc")
        vc = VolumeCollection.from_niftis(
            uri=uri,
            niftis=synthetic_niftis,
            obs=obs,
            validate_dimensions=False,
        )
        obs_df = vc.obs.read()
        assert "scan_date" in obs_df.columns
        assert set(obs_df["scan_date"]) == {"2024-01-01", "2024-01-02"}

    def test_from_niftis_imaging_metadata_always_extracted(self, tmp_path, synthetic_niftis):
        from radiobject.volume_collection import VolumeCollection

        obs = pd.DataFrame(
            {
                "obs_id": ["c01", "c02"],
                "obs_subject_id": ["sub-01", "sub-02"],
            }
        )
        uri = str(tmp_path / "imaging_vc")
        vc = VolumeCollection.from_niftis(
            uri=uri,
            niftis=synthetic_niftis,
            obs=obs,
            validate_dimensions=False,
        )
        obs_df = vc.obs.read()
        assert "series_type" in obs_df.columns
        assert "voxel_spacing" in obs_df.columns

    def test_from_niftis_column_collision_raises(self, tmp_path, synthetic_niftis):
        from radiobject.volume_collection import VolumeCollection

        obs = pd.DataFrame(
            {
                "obs_id": ["c01", "c02"],
                "obs_subject_id": ["sub-01", "sub-02"],
                "series_type": ["T1w", "T1w"],  # collides with auto-generated
            }
        )
        uri = str(tmp_path / "collision_vc")
        with pytest.raises(ValueError, match="collide"):
            VolumeCollection.from_niftis(
                uri=uri,
                niftis=synthetic_niftis,
                obs=obs,
                validate_dimensions=False,
            )

    def test_from_niftis_length_mismatch_raises(self, tmp_path, synthetic_niftis):
        from radiobject.volume_collection import VolumeCollection

        obs = pd.DataFrame(
            {
                "obs_id": ["c01"],
                "obs_subject_id": ["sub-01"],
            }
        )
        uri = str(tmp_path / "mismatch_vc")
        with pytest.raises(ValueError, match="obs length"):
            VolumeCollection.from_niftis(
                uri=uri,
                niftis=synthetic_niftis,
                obs=obs,
                validate_dimensions=False,
            )

    def test_from_niftis_subject_id_mismatch_raises(self, tmp_path, synthetic_niftis):
        from radiobject.volume_collection import VolumeCollection

        obs = pd.DataFrame(
            {
                "obs_id": ["c01", "c02"],
                "obs_subject_id": ["sub-02", "sub-01"],  # swapped
            }
        )
        uri = str(tmp_path / "sid_mismatch_vc")
        with pytest.raises(ValueError, match="obs_subject_id mismatch"):
            VolumeCollection.from_niftis(
                uri=uri,
                niftis=synthetic_niftis,
                obs=obs,
                validate_dimensions=False,
            )

    def test_from_niftis_missing_obs_id_raises(self, tmp_path, synthetic_niftis):
        from radiobject.volume_collection import VolumeCollection

        obs = pd.DataFrame({"obs_subject_id": ["sub-01", "sub-02"]})
        uri = str(tmp_path / "no_obs_id_vc")
        with pytest.raises(ValueError, match="obs_id"):
            VolumeCollection.from_niftis(
                uri=uri,
                niftis=synthetic_niftis,
                obs=obs,
                validate_dimensions=False,
            )

    def test_from_niftis_missing_obs_subject_id_raises(self, tmp_path, synthetic_niftis):
        from radiobject.volume_collection import VolumeCollection

        obs = pd.DataFrame({"obs_id": ["c01", "c02"]})
        uri = str(tmp_path / "no_sid_vc")
        with pytest.raises(ValueError, match="obs_subject_id"):
            VolumeCollection.from_niftis(
                uri=uri,
                niftis=synthetic_niftis,
                obs=obs,
                validate_dimensions=False,
            )


# --- RadiObject validation tests ---


class TestRadiObjectObsMetaValidation:
    """Tests that obs_meta uses single-dimension (obs_subject_id only)."""

    def test_from_niftis_obs_meta_single_dim(self, tmp_path, synthetic_nifti_images):
        from radiobject.radi_object import RadiObject

        uri = str(tmp_path / "radi_single_dim")
        radi = RadiObject.from_images(uri=uri, images=synthetic_nifti_images)
        assert radi.obs_meta.index_columns == ("obs_subject_id",)

    def test_from_niftis_missing_subject_id_raises(self, tmp_path, synthetic_nifti_images):
        from radiobject.radi_object import RadiObject

        uri = str(tmp_path / "radi_no_sid")
        obs_meta = pd.DataFrame({"bad_column": ["sub-01", "sub-02"]})
        with pytest.raises(ValueError, match="obs_subject_id"):
            RadiObject.from_images(uri=uri, images=synthetic_nifti_images, obs_meta=obs_meta)

    def test_from_collections_validates_obs_meta(self, tmp_path, synthetic_nifti_images):
        from radiobject.radi_object import RadiObject
        from radiobject.volume_collection import VolumeCollection

        # First create a VC
        vc_uri = str(tmp_path / "vc_for_fc")
        vc = VolumeCollection.from_niftis(
            uri=vc_uri,
            niftis=synthetic_nifti_images["T1w"],
            validate_dimensions=False,
        )

        uri = str(tmp_path / "radi_fc_validate")
        obs_meta = pd.DataFrame({"bad_column": ["sub-01"]})
        with pytest.raises(ValueError, match="obs_subject_id"):
            RadiObject.from_collections(
                uri=uri,
                collections={"T1w": vc},
                obs_meta=obs_meta,
            )

    def test_obs_meta_no_obs_id_column(self, tmp_path, synthetic_nifti_images):
        from radiobject.radi_object import RadiObject

        uri = str(tmp_path / "radi_no_obs_id")
        radi = RadiObject.from_images(uri=uri, images=synthetic_nifti_images)
        obs_meta_df = radi.obs_meta.read()
        # obs_id should NOT be a column or dimension in obs_meta
        # (obs_ids — plural — is the system-managed attribute)
        assert "obs_id" not in obs_meta_df.columns


# --- obs_ids auto-population tests ---


class TestObsIdsAutoPopulation:
    def test_from_niftis_obs_ids_populated(self, tmp_path, synthetic_nifti_images):
        from radiobject.radi_object import RadiObject

        uri = str(tmp_path / "radi_obs_ids")
        radi = RadiObject.from_images(uri=uri, images=synthetic_nifti_images)
        obs_meta_df = radi.obs_meta.read()
        assert "obs_ids" in obs_meta_df.columns
        for _, row in obs_meta_df.iterrows():
            obs_ids = json.loads(row["obs_ids"])
            assert isinstance(obs_ids, list)
            assert len(obs_ids) > 0

    def test_obs_ids_sorted_deterministically(self, tmp_path):
        """obs_ids list is sorted for reproducibility."""
        import nibabel as nib

        from radiobject.radi_object import RadiObject

        # Create two collections for same subject
        images = {"T1w": [], "FLAIR": []}
        for mod in ["T1w", "FLAIR"]:
            for sid in ["sub-01"]:
                data = np.random.rand(4, 4, 4).astype(np.float32)
                img = nib.Nifti1Image(data, np.eye(4))
                path = tmp_path / f"{sid}_{mod}.nii.gz"
                nib.save(img, path)
                images[mod].append((str(path), sid))

        uri = str(tmp_path / "radi_sorted_ids")
        radi = RadiObject.from_images(uri=uri, images=images)
        obs_meta_df = radi.obs_meta.read()
        row = obs_meta_df[obs_meta_df["obs_subject_id"] == "sub-01"].iloc[0]
        obs_ids = json.loads(row["obs_ids"])
        assert obs_ids == sorted(obs_ids)
