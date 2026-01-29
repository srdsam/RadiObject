"""Tests for RadiObject and RadiObjectView."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tiledb

from radiobject.radi_object import RadiObject, RadiObjectView
from radiobject.volume_collection import VolumeCollection


class TestRadiObjectCreate:
    """Tests for RadiObject._create()."""

    def test_create_empty_radi_object(self, radi_object_uri: str):
        """Create an empty RadiObject with default settings."""
        radi = RadiObject._create(radi_object_uri)

        assert len(radi) == 0
        assert radi.n_collections == 0
        assert radi.obs_subject_ids == []

    def test_create_with_obs_meta_schema(self, radi_object_uri: str):
        """Create RadiObject with additional obs_meta schema columns."""
        obs_meta_schema = {"age": np.dtype("int32"), "diagnosis": np.dtype("U64")}
        radi = RadiObject._create(radi_object_uri, obs_meta_schema=obs_meta_schema)

        assert "obs_subject_id" in radi.obs_meta.index_columns
        assert "age" in radi.obs_meta.columns
        assert "diagnosis" in radi.obs_meta.columns


class TestRadiObjectFromVolumeCollections:
    """Tests for RadiObject._from_volume_collections()."""

    def test_from_volume_collections_basic(
        self,
        temp_dir: Path,
        volume_collections: dict[str, VolumeCollection],
        nifti_manifest: list[dict],
    ):
        """Create RadiObject from existing VolumeCollections."""
        uri = str(temp_dir / "from_vc_test")
        subject_ids = [entry["sample_id"] for entry in nifti_manifest[:3]]
        obs_meta_df = pd.DataFrame({
            "obs_subject_id": subject_ids,
        })

        radi = RadiObject._from_volume_collections(
            uri,
            collections=volume_collections,
            obs_meta=obs_meta_df,
        )

        assert len(radi) == 3
        assert radi.n_collections == 4
        assert "T1w" in radi.collection_names
        assert "flair" in radi.collection_names

    def test_from_volume_collections_with_obs_meta(
        self,
        temp_dir: Path,
        volume_collections: dict[str, VolumeCollection],
        nifti_manifest: list[dict],
    ):
        """Create RadiObject with additional obs_meta metadata."""
        uri = str(temp_dir / "from_vc_obs_test")
        subject_ids = [entry["sample_id"] for entry in nifti_manifest[:3]]
        obs_meta_df = pd.DataFrame({
            "obs_subject_id": subject_ids,
            "age": [45, 52, 38],
            "diagnosis": ["healthy", "tumor", "healthy"],
        })

        radi = RadiObject._from_volume_collections(
            uri,
            collections=volume_collections,
            obs_meta=obs_meta_df,
        )

        assert "age" in radi.obs_meta.columns
        assert "diagnosis" in radi.obs_meta.columns

    def test_from_volume_collections_empty_raises(self, temp_dir: Path):
        """Empty collections dict raises ValueError."""
        uri = str(temp_dir / "empty_vc")
        with pytest.raises(ValueError, match="At least one VolumeCollection"):
            RadiObject._from_volume_collections(uri, collections={})


class TestRadiObjectILoc:
    """Tests for iloc (integer-location) indexer."""

    def test_iloc_single_int(self, populated_radi_object_module: RadiObject):
        """iloc[int] returns RadiObjectView with single subject."""
        view = populated_radi_object_module.iloc[0]

        assert isinstance(view, RadiObjectView)
        assert len(view) == 1
        assert view.obs_subject_ids == [populated_radi_object_module.obs_subject_ids[0]]

    def test_iloc_negative_index(self, populated_radi_object_module: RadiObject):
        """iloc[-1] returns view with last subject."""
        view = populated_radi_object_module.iloc[-1]

        assert len(view) == 1
        assert view.obs_subject_ids == [populated_radi_object_module.obs_subject_ids[-1]]

    def test_iloc_slice(self, populated_radi_object_module: RadiObject):
        """iloc[start:stop] returns view with sliced subjects."""
        view = populated_radi_object_module.iloc[0:2]

        assert len(view) == 2
        assert view.obs_subject_ids == populated_radi_object_module.obs_subject_ids[0:2]

    def test_iloc_list(self, populated_radi_object_module: RadiObject):
        """iloc[[0, 2]] returns view with specific subjects."""
        view = populated_radi_object_module.iloc[[0, 2]]

        assert len(view) == 2
        expected = [populated_radi_object_module.obs_subject_ids[i] for i in [0, 2]]
        assert view.obs_subject_ids == expected

    def test_iloc_out_of_range_raises(self, populated_radi_object_module: RadiObject):
        """iloc[99] raises IndexError."""
        with pytest.raises(IndexError):
            populated_radi_object_module.iloc[99]


class TestRadiObjectLoc:
    """Tests for loc (label-based) indexer."""

    def test_loc_single_str(self, populated_radi_object_module: RadiObject):
        """loc[str] returns RadiObjectView with single subject."""
        subject_id = populated_radi_object_module.obs_subject_ids[1]
        view = populated_radi_object_module.loc[subject_id]

        assert isinstance(view, RadiObjectView)
        assert len(view) == 1
        assert view.obs_subject_ids == [subject_id]

    def test_loc_list(self, populated_radi_object_module: RadiObject):
        """loc[[str, str]] returns view with multiple subjects."""
        subject_ids = [
            populated_radi_object_module.obs_subject_ids[0],
            populated_radi_object_module.obs_subject_ids[2],
        ]
        view = populated_radi_object_module.loc[subject_ids]

        assert len(view) == 2
        assert view.obs_subject_ids == subject_ids

    def test_loc_not_found_raises(self, populated_radi_object_module: RadiObject):
        """loc["NONEXISTENT"] raises KeyError."""
        with pytest.raises(KeyError):
            populated_radi_object_module.loc["NONEXISTENT"]


class TestRadiObjectCollectionAccess:
    """Tests for collection access methods."""

    def test_collection_method(self, populated_radi_object_module: RadiObject):
        """collection(name) returns VolumeCollection."""
        vc = populated_radi_object_module.collection("T1w")

        assert isinstance(vc, VolumeCollection)
        assert len(vc) == 3

    def test_collection_attribute_access(self, populated_radi_object_module: RadiObject):
        """radi.T1w returns VolumeCollection via __getattr__."""
        vc = populated_radi_object_module.T1w

        assert isinstance(vc, VolumeCollection)
        assert len(vc) == 3

    def test_collection_not_found_raises(self, populated_radi_object_module: RadiObject):
        """collection("NONEXISTENT") raises KeyError."""
        with pytest.raises(KeyError):
            populated_radi_object_module.collection("NONEXISTENT")


class TestRadiObjectProperties:
    """Tests for RadiObject properties."""

    def test_obs_subject_ids(self, populated_radi_object_module: RadiObject):
        """obs_subject_ids returns list of all subject IDs."""
        ids = populated_radi_object_module.obs_subject_ids

        assert len(ids) == 3

    def test_collection_names(self, populated_radi_object_module: RadiObject):
        """collection_names returns tuple of collection names."""
        names = populated_radi_object_module.collection_names

        assert len(names) == 4
        assert "T1w" in names
        assert "flair" in names

    def test_n_collections(self, populated_radi_object_module: RadiObject):
        """n_collections returns number of collections."""
        assert populated_radi_object_module.n_collections == 4

    def test_len(self, populated_radi_object_module: RadiObject):
        """len(radi) returns number of subjects."""
        assert len(populated_radi_object_module) == 3


class TestRadiObjectIndex:
    """Tests for RadiObject.index property."""

    def test_index_get_index(self, populated_radi_object_module: RadiObject):
        """index.get_index() maps obs_subject_id to integer index."""
        for i, subject_id in enumerate(populated_radi_object_module.obs_subject_ids):
            assert populated_radi_object_module.index.get_index(subject_id) == i

    def test_index_get_key(self, populated_radi_object_module: RadiObject):
        """index.get_key() maps integer index to obs_subject_id."""
        for i, expected_id in enumerate(populated_radi_object_module.obs_subject_ids):
            assert populated_radi_object_module.index.get_key(i) == expected_id

    def test_index_keys(self, populated_radi_object_module: RadiObject):
        """index.keys returns tuple of all obs_subject_ids."""
        assert populated_radi_object_module.index.keys == tuple(
            populated_radi_object_module.obs_subject_ids
        )

    def test_index_len(self, populated_radi_object_module: RadiObject):
        """len(index) returns number of subjects."""
        assert len(populated_radi_object_module.index) == 3

    def test_index_contains(self, populated_radi_object_module: RadiObject):
        """'obs_subject_id' in index works correctly."""
        first_id = populated_radi_object_module.obs_subject_ids[0]
        assert first_id in populated_radi_object_module.index
        assert "NONEXISTENT" not in populated_radi_object_module.index

    def test_index_roundtrip(self, populated_radi_object_module: RadiObject):
        """Roundtrip: index.get_key(index.get_index(id)) == id."""
        first_id = populated_radi_object_module.index.keys[0]
        idx = populated_radi_object_module.index.get_index(first_id)
        assert populated_radi_object_module.index.get_key(idx) == first_id


class TestRadiObjectObsRowRetrieval:
    """Tests for get_obs_row_by_obs_subject_id method."""

    def test_get_obs_row_by_obs_subject_id(self, populated_radi_object_module: RadiObject):
        """Get obs_meta row by obs_subject_id string."""
        subject_id = populated_radi_object_module.obs_subject_ids[1]
        row = populated_radi_object_module.get_obs_row_by_obs_subject_id(subject_id)
        assert row["obs_subject_id"].iloc[0] == subject_id

    def test_get_obs_row_by_index_pattern(self, populated_radi_object_module: RadiObject):
        """Get obs_meta row by integer index using index.get_key() pattern."""
        subject_id = populated_radi_object_module.index.get_key(0)
        row = populated_radi_object_module.get_obs_row_by_obs_subject_id(subject_id)
        assert "obs_subject_id" in row.columns
        assert row["obs_subject_id"].iloc[0] == populated_radi_object_module.obs_subject_ids[0]


class TestRadiObjectViewFiltering:
    """Tests for RadiObjectView filtering methods."""

    def test_select_subjects(self, populated_radi_object_module: RadiObject):
        """select_subjects filters to subset of subjects."""
        view = populated_radi_object_module.iloc[:]
        subject_ids = [
            populated_radi_object_module.obs_subject_ids[0],
            populated_radi_object_module.obs_subject_ids[2],
        ]
        filtered = view.select_subjects(subject_ids)

        assert len(filtered) == 2
        assert filtered.obs_subject_ids == subject_ids

    def test_select_collections(self, populated_radi_object_module: RadiObject):
        """select_collections filters to subset of collections."""
        view = populated_radi_object_module.iloc[:]
        filtered = view.select_collections(["T1w"])

        assert filtered.n_collections == 1
        assert filtered.collection_names == ("T1w",)

    def test_chained_filters(self, populated_radi_object_module: RadiObject):
        """Chained filtering narrows both subjects and collections."""
        subject_ids = [
            populated_radi_object_module.obs_subject_ids[0],
            populated_radi_object_module.obs_subject_ids[1],
        ]
        view = (
            populated_radi_object_module.iloc[:]
            .select_subjects(subject_ids)
            .select_collections(["flair"])
        )

        assert len(view) == 2
        assert view.obs_subject_ids == subject_ids
        assert view.collection_names == ("flair",)


class TestRadiObjectViewMaterialization:
    """Tests for RadiObjectView.to_radi_object()."""

    def test_to_radi_object_basic(
        self,
        temp_dir: Path,
        populated_radi_object: RadiObject,
    ):
        """Materialize full view as new RadiObject."""
        view = populated_radi_object.iloc[:]
        new_uri = str(temp_dir / "materialized_radi")
        new_radi = view.to_radi_object(new_uri)

        assert len(new_radi) == 3
        assert new_radi.n_collections == 4

    def test_to_radi_object_filtered_subjects(
        self,
        temp_dir: Path,
        populated_radi_object: RadiObject,
    ):
        """Materialize view with filtered subjects."""
        view = populated_radi_object.iloc[[0, 2]]
        new_uri = str(temp_dir / "materialized_filtered")
        new_radi = view.to_radi_object(new_uri)

        assert len(new_radi) == 2
        expected_ids = [populated_radi_object.obs_subject_ids[i] for i in [0, 2]]
        assert new_radi.obs_subject_ids == expected_ids

    def test_to_radi_object_filtered_collections(
        self,
        temp_dir: Path,
        populated_radi_object: RadiObject,
    ):
        """Materialize view with filtered collections."""
        view = populated_radi_object.select_collections(["T1w"])
        new_uri = str(temp_dir / "materialized_one_collection")
        new_radi = view.to_radi_object(new_uri)

        assert new_radi.n_collections == 1
        assert "T1w" in new_radi.collection_names


class TestRadiObjectRoundtrip:
    """Integration tests for complete workflows."""

    def test_create_reopen_roundtrip(
        self,
        temp_dir: Path,
        volume_collections: dict[str, VolumeCollection],
        nifti_manifest: list[dict],
    ):
        """Create RadiObject and verify data persists after reopening."""
        uri = str(temp_dir / "roundtrip_test")
        subject_ids = [entry["sample_id"] for entry in nifti_manifest[:3]]
        obs_meta_df = pd.DataFrame({
            "obs_subject_id": subject_ids,
            "age": [45, 52, 38],
        })

        RadiObject._from_volume_collections(
            uri,
            collections=volume_collections,
            obs_meta=obs_meta_df,
        )

        reopened = RadiObject(uri)

        assert len(reopened) == 3
        assert reopened.n_collections == 4
        assert "T1w" in reopened.collection_names

    def test_obs_subject_id_consistency(self, populated_radi_object_module: RadiObject):
        """Verify obs_subject_id consistency across obs_meta and collections."""
        subject_ids = populated_radi_object_module.obs_subject_ids

        for name in populated_radi_object_module.collection_names:
            vc = populated_radi_object_module.collection(name)
            obs_df = vc.obs.read()
            vc_subject_ids = list(obs_df["obs_subject_id"])
            assert set(vc_subject_ids) == set(subject_ids)


# ============================================================================
# S3 Integration Tests (Essential subset - logic tested locally)
# ============================================================================


class TestS3Integration:
    """Essential S3 integration tests. Business logic tested locally; these verify S3 I/O.

    These tests use S3-specific fixtures that skip at fixture level if S3 is
    unavailable, preventing fixture setup pollution when running the full test suite.
    """

    def test_create_empty(
        self,
        s3_radi_object_uri: str,
        s3_tiledb_ctx: tiledb.Ctx,
    ):
        """Verify empty RadiObject can be created on S3."""
        radi = RadiObject._create(s3_radi_object_uri, ctx=s3_tiledb_ctx)

        assert len(radi) == 0
        assert radi.n_collections == 0

    def test_from_volume_collections(
        self,
        s3_test_base_uri: str,
        s3_volume_collections: dict[str, VolumeCollection],
        nifti_manifest: list[dict],
        s3_tiledb_ctx: tiledb.Ctx,
    ):
        """Verify RadiObject can be built from VolumeCollections on S3."""
        subject_ids = [entry["sample_id"] for entry in nifti_manifest[:3]]
        obs_meta_df = pd.DataFrame({"obs_subject_id": subject_ids})

        uri = f"{s3_test_base_uri}/radi_object"
        radi = RadiObject._from_volume_collections(
            uri, collections=s3_volume_collections, obs_meta=obs_meta_df, ctx=s3_tiledb_ctx
        )

        assert len(radi) == 3
        assert radi.n_collections == 2

    def test_roundtrip(
        self,
        s3_test_base_uri: str,
        s3_volume_collections: dict[str, VolumeCollection],
        nifti_manifest: list[dict],
        s3_tiledb_ctx: tiledb.Ctx,
    ):
        """Verify RadiObject persists and can be reopened from S3."""
        subject_ids = [entry["sample_id"] for entry in nifti_manifest[:3]]
        obs_meta_df = pd.DataFrame({"obs_subject_id": subject_ids, "age": [45, 52, 38]})

        uri = f"{s3_test_base_uri}/roundtrip_radi"
        RadiObject._from_volume_collections(
            uri, collections=s3_volume_collections, obs_meta=obs_meta_df, ctx=s3_tiledb_ctx
        )

        reopened = RadiObject(uri, ctx=s3_tiledb_ctx)

        assert len(reopened) == 3
        assert "flair" in reopened.collection_names

    def test_iloc_and_loc(
        self,
        s3_populated_radi_object: RadiObject,
    ):
        """Verify iloc/loc indexing works on S3."""
        view = s3_populated_radi_object.iloc[0]
        assert len(view) == 1

        subject_ids = s3_populated_radi_object.obs_subject_ids
        view = s3_populated_radi_object.loc[subject_ids[1]]
        assert view.obs_subject_ids == [subject_ids[1]]

    def test_view_materialization(
        self,
        s3_test_base_uri: str,
        s3_populated_radi_object: RadiObject,
        s3_tiledb_ctx: tiledb.Ctx,
    ):
        """Verify view materialization works on S3."""
        subject_ids = s3_populated_radi_object.obs_subject_ids
        view = s3_populated_radi_object.iloc[[0, 2]]
        new_uri = f"{s3_test_base_uri}/materialized_radi"
        new_radi = view.to_radi_object(new_uri, ctx=s3_tiledb_ctx)

        assert len(new_radi) == 2
        assert new_radi.obs_subject_ids == [subject_ids[0], subject_ids[2]]
