"""Tests for RadiObject (attached and views)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tiledb

from radiobject.radi_object import RadiObject
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
        obs_meta_df = pd.DataFrame(
            {
                "obs_subject_id": subject_ids,
            }
        )

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
        obs_meta_df = pd.DataFrame(
            {
                "obs_subject_id": subject_ids,
                "age": [45, 52, 38],
                "diagnosis": ["healthy", "tumor", "healthy"],
            }
        )

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


class TestRadiObjectFromCollections:
    """Tests for RadiObject.from_collections() public factory method."""

    def test_from_collections_in_place(
        self,
        temp_dir: Path,
        volume_collections: dict[str, VolumeCollection],
        nifti_manifest: list[dict],
    ):
        """Collections already at expected URIs are linked, not copied."""
        uri = str(temp_dir / "from_collections_inplace")
        collections_uri = f"{uri}/collections"

        # First, create collections at expected locations
        import tiledb

        from radiobject.radi_object import _copy_volume_collection

        tiledb.Group.create(collections_uri)
        in_place_collections = {}
        for name, vc in volume_collections.items():
            expected_uri = f"{collections_uri}/{name}"
            _copy_volume_collection(vc, expected_uri, name=name)
            in_place_collections[name] = VolumeCollection(expected_uri)

        # Use from_collections - should link without copying
        radi = RadiObject.from_collections(
            uri=uri,
            collections=in_place_collections,
        )

        assert len(radi) == 3
        assert radi.n_collections == len(volume_collections)
        assert set(radi.collection_names) == set(volume_collections.keys())

    def test_from_collections_with_copy(
        self,
        temp_dir: Path,
        volume_collections: dict[str, VolumeCollection],
    ):
        """Collections from external URIs are copied to expected locations."""
        uri = str(temp_dir / "from_collections_copy")

        # Use from_collections with collections at different URIs - should copy
        radi = RadiObject.from_collections(
            uri=uri,
            collections=volume_collections,
        )

        assert len(radi) == 3
        assert radi.n_collections == len(volume_collections)

    def test_from_collections_with_obs_meta(
        self,
        temp_dir: Path,
        volume_collections: dict[str, VolumeCollection],
        nifti_manifest: list[dict],
    ):
        """Custom obs_meta is used when provided."""
        uri = str(temp_dir / "from_collections_obs_meta")
        subject_ids = [entry["sample_id"] for entry in nifti_manifest[:3]]
        obs_meta_df = pd.DataFrame(
            {
                "obs_subject_id": subject_ids,
                "age": [45, 52, 38],
                "diagnosis": ["healthy", "tumor", "healthy"],
            }
        )

        radi = RadiObject.from_collections(
            uri=uri,
            collections=volume_collections,
            obs_meta=obs_meta_df,
        )

        assert "age" in radi.obs_meta.columns
        assert "diagnosis" in radi.obs_meta.columns

    def test_from_collections_derives_obs_meta(
        self,
        temp_dir: Path,
        volume_collections: dict[str, VolumeCollection],
    ):
        """obs_meta is derived from collections when not provided."""
        uri = str(temp_dir / "from_collections_derived")

        radi = RadiObject.from_collections(
            uri=uri,
            collections=volume_collections,
        )

        # obs_meta should be auto-derived from collections
        assert len(radi) == 3
        assert len(radi.obs_subject_ids) == 3

    def test_from_collections_string_uri(
        self,
        temp_dir: Path,
        volume_collections: dict[str, VolumeCollection],
    ):
        """String URIs are resolved to VolumeCollection objects."""
        uri = str(temp_dir / "from_collections_string_uri")

        # Pass URI strings instead of VolumeCollection objects
        uri_collections = {name: vc.uri for name, vc in volume_collections.items()}

        radi = RadiObject.from_collections(
            uri=uri,
            collections=uri_collections,
        )

        assert radi.n_collections == len(volume_collections)

    def test_from_collections_empty_raises(self, temp_dir: Path):
        """Empty collections dict raises ValueError."""
        uri = str(temp_dir / "from_collections_empty")
        with pytest.raises(ValueError, match="At least one collection"):
            RadiObject.from_collections(uri=uri, collections={})


class TestRadiObjectILoc:
    """Tests for iloc (integer-location) indexer."""

    @pytest.mark.parametrize(
        ("indexer", "expected_len"),
        [
            (0, 1),  # iloc[int] - single subject
            (-1, 1),  # iloc[-1] - last subject
            (slice(0, 2), 2),  # iloc[start:stop] - range
            ([0, 2], 2),  # iloc[list] - specific indices
        ],
        ids=["single", "negative", "slice", "list"],
    )
    def test_iloc_indexing(self, populated_radi_object_module: RadiObject, indexer, expected_len):
        """iloc returns RadiObject view with expected subjects."""
        view = populated_radi_object_module.iloc[indexer]

        assert isinstance(view, RadiObject)
        assert view.is_view
        assert len(view) == expected_len

    def test_iloc_out_of_range_raises(self, populated_radi_object_module: RadiObject):
        """iloc[99] raises IndexError."""
        with pytest.raises(IndexError):
            populated_radi_object_module.iloc[99]


class TestRadiObjectLoc:
    """Tests for loc (label-based) indexer."""

    def test_loc_single_and_list(self, populated_radi_object_module: RadiObject):
        """loc returns RadiObject view for single ID or list of IDs."""
        subject_ids = populated_radi_object_module.obs_subject_ids

        single_view = populated_radi_object_module.loc[subject_ids[1]]
        assert isinstance(single_view, RadiObject)
        assert single_view.is_view
        assert len(single_view) == 1

        multi_view = populated_radi_object_module.loc[[subject_ids[0], subject_ids[2]]]
        assert len(multi_view) == 2
        assert multi_view.obs_subject_ids == [subject_ids[0], subject_ids[2]]

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

    def test_basic_properties(self, populated_radi_object_module: RadiObject):
        """Verify core properties: length, collections, subject IDs."""
        radi = populated_radi_object_module

        assert len(radi) == 3
        assert len(radi.obs_subject_ids) == 3
        assert radi.n_collections == 4
        assert {"T1w", "flair"} <= set(radi.collection_names)


class TestRadiObjectIndex:
    """Tests for RadiObject.index property."""

    def test_index_bidirectional_mapping(self, populated_radi_object_module: RadiObject):
        """index maps between obs_subject_id and integer index bidirectionally."""
        index = populated_radi_object_module.index
        subject_ids = populated_radi_object_module.obs_subject_ids

        assert len(index) == 3
        assert index.keys == tuple(subject_ids)

        for i, subject_id in enumerate(subject_ids):
            assert index.get_index(subject_id) == i
            assert index.get_key(i) == subject_id

    def test_index_contains(self, populated_radi_object_module: RadiObject):
        """'obs_subject_id' in index works correctly."""
        first_id = populated_radi_object_module.obs_subject_ids[0]

        assert first_id in populated_radi_object_module.index
        assert "NONEXISTENT" not in populated_radi_object_module.index


class TestRadiObjectObsRowRetrieval:
    """Tests for get_obs_row_by_obs_subject_id method."""

    def test_get_obs_row_by_obs_subject_id(self, populated_radi_object_module: RadiObject):
        """Get obs_meta row by obs_subject_id string or via index.get_key() pattern."""
        radi = populated_radi_object_module
        subject_id = radi.obs_subject_ids[1]

        row = radi.get_obs_row_by_obs_subject_id(subject_id)
        assert "obs_subject_id" in row.columns
        assert row["obs_subject_id"].iloc[0] == subject_id

        first_id = radi.index.get_key(0)
        row_via_index = radi.get_obs_row_by_obs_subject_id(first_id)
        assert row_via_index["obs_subject_id"].iloc[0] == radi.obs_subject_ids[0]


class TestRadiObjectFiltering:
    """Tests for RadiObject view filtering methods."""

    def test_select_collections(self, populated_radi_object_module: RadiObject):
        """select_collections filters to subset of collections."""
        view = populated_radi_object_module.iloc[:]
        filtered = view.select_collections(["T1w"])

        assert filtered.n_collections == 1
        assert filtered.collection_names == ("T1w",)


class TestRadiObjectMaterialization:
    """Tests for RadiObject view materialize()."""

    def test_materialize_basic(
        self,
        temp_dir: Path,
        populated_radi_object: RadiObject,
    ):
        """Materialize full view as new RadiObject."""
        view = populated_radi_object.iloc[:]
        new_uri = str(temp_dir / "materialized_radi")
        new_radi = view.materialize(new_uri)

        assert len(new_radi) == 3
        assert new_radi.n_collections == 4

    def test_materialize_filtered_subjects(
        self,
        temp_dir: Path,
        populated_radi_object: RadiObject,
    ):
        """Materialize view with filtered subjects."""
        view = populated_radi_object.iloc[[0, 2]]
        new_uri = str(temp_dir / "materialized_filtered")
        new_radi = view.materialize(new_uri)

        assert len(new_radi) == 2
        expected_ids = [populated_radi_object.obs_subject_ids[i] for i in [0, 2]]
        assert new_radi.obs_subject_ids == expected_ids

    def test_materialize_filtered_collections(
        self,
        temp_dir: Path,
        populated_radi_object: RadiObject,
    ):
        """Materialize view with filtered collections."""
        view = populated_radi_object.select_collections(["T1w"])
        new_uri = str(temp_dir / "materialized_one_collection")
        new_radi = view.materialize(new_uri)

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
        obs_meta_df = pd.DataFrame(
            {
                "obs_subject_id": subject_ids,
                "age": [45, 52, 38],
            }
        )

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


class TestRadiObjectIndexName:
    """Tests for Index name propagation."""

    def test_index_has_obs_subject_id_name(self, populated_radi_object_module: RadiObject):
        """RadiObject index has name='obs_subject_id'."""
        assert populated_radi_object_module.index.name == "obs_subject_id"

    def test_index_repr(self, populated_radi_object_module: RadiObject):
        """Index repr shows name and count."""
        assert "obs_subject_id" in repr(populated_radi_object_module.index)
        assert "3 keys" in repr(populated_radi_object_module.index)


class TestRadiObjectSel:
    """Tests for .sel(subject=...) method."""

    def test_sel_single_subject(self, populated_radi_object_module: RadiObject):
        """sel(subject=str) returns single-subject RadiObject view."""
        subject_id = populated_radi_object_module.obs_subject_ids[1]
        view = populated_radi_object_module.sel(subject=subject_id)
        assert isinstance(view, RadiObject)
        assert view.is_view
        assert len(view) == 1
        assert view.obs_subject_ids == [subject_id]

    def test_sel_list_subjects(self, populated_radi_object_module: RadiObject):
        """sel(subject=list) returns multi-subject RadiObject view."""
        subject_ids = populated_radi_object_module.obs_subject_ids
        view = populated_radi_object_module.sel(subject=[subject_ids[0], subject_ids[2]])
        assert len(view) == 2
        assert view.obs_subject_ids == [subject_ids[0], subject_ids[2]]

    def test_sel_nonexistent_raises(self, populated_radi_object_module: RadiObject):
        """sel(subject='NONEXISTENT') raises KeyError."""
        with pytest.raises(KeyError):
            populated_radi_object_module.sel(subject="NONEXISTENT")


class TestRadiObjectCrossCollectionAlignment:
    """Tests for cross-collection Index set operations."""

    def test_collection_subjects_aligned(self, populated_radi_object_module: RadiObject):
        """All collections' subjects should be aligned."""
        names = populated_radi_object_module.collection_names
        first = populated_radi_object_module.collection(names[0]).subjects
        for name in names[1:]:
            other = populated_radi_object_module.collection(name).subjects
            assert first.is_aligned(other)

    def test_cross_collection_intersection(self, populated_radi_object_module: RadiObject):
        """Cross-collection subject intersection works."""
        flair_subjects = populated_radi_object_module.collection("flair").subjects
        t1w_subjects = populated_radi_object_module.collection("T1w").subjects
        common = flair_subjects & t1w_subjects
        assert len(common) == 3


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
        new_radi = view.materialize(new_uri, ctx=s3_tiledb_ctx)

        assert len(new_radi) == 2
        assert new_radi.obs_subject_ids == [subject_ids[0], subject_ids[2]]
