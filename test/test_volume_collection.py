"""Tests for VolumeCollection."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from radiobject.volume import Volume
from radiobject.volume_collection import VolumeCollection


class TestVolumeCollectionCreate:
    """Tests for VolumeCollection._create()."""

    def test_create_empty_collection(
        self, collection_uri: str, collection_shape: tuple[int, int, int]
    ):
        """Create an empty collection with specified dimensions."""
        collection = VolumeCollection._create(collection_uri, shape=collection_shape)

        assert collection.shape == collection_shape
        assert len(collection) == 0
        assert collection.obs_ids == []

    def test_create_with_obs_schema(
        self, collection_uri: str, collection_shape: tuple[int, int, int]
    ):
        """Create collection with additional obs schema columns."""
        obs_schema = {"age": np.dtype("int32"), "diagnosis": np.dtype("U64")}
        collection = VolumeCollection._create(
            collection_uri, shape=collection_shape, obs_schema=obs_schema
        )

        assert collection.obs.index_columns == ("obs_subject_id", "obs_id")
        assert "age" in collection.obs.columns
        assert "diagnosis" in collection.obs.columns


class TestVolumeCollectionFromVolumes:
    """Tests for VolumeCollection._from_volumes()."""

    def test_from_volumes_basic(
        self,
        temp_dir: Path,
        volumes: list[tuple[str, Volume]],
    ):
        """Create collection from existing volumes."""
        uri = str(temp_dir / "from_volumes_test")
        collection = VolumeCollection._from_volumes(uri, volumes)

        assert len(collection) == 3
        assert len(collection.obs_ids) == 3

    def test_from_volumes_with_obs_data(
        self,
        temp_dir: Path,
        volumes: list[tuple[str, Volume]],
    ):
        """Create collection with additional obs metadata."""
        uri = str(temp_dir / "from_volumes_obs_test")
        obs_ids = [v[0] for v in volumes]
        subject_ids = [obs_id.rsplit("_", 1)[0] for obs_id in obs_ids]

        obs_df = pd.DataFrame({
            "obs_subject_id": subject_ids,
            "obs_id": obs_ids,
            "age": [45, 52, 38],
            "diagnosis": ["healthy", "tumor", "healthy"],
        })
        collection = VolumeCollection._from_volumes(uri, volumes, obs_data=obs_df)

        assert "age" in collection.obs.columns
        obs_id = collection.index.get_key(0)
        row = collection.get_obs_row_by_obs_id(obs_id)
        assert row["age"].iloc[0] == 45

    def test_from_volumes_empty_raises(self, temp_dir: Path):
        """Empty volumes list raises ValueError."""
        uri = str(temp_dir / "empty_volumes")
        with pytest.raises(ValueError, match="At least one volume"):
            VolumeCollection._from_volumes(uri, [])

    def test_from_volumes_shape_mismatch_raises(
        self,
        temp_dir: Path,
        volumes: list[tuple[str, Volume]],
    ):
        """Mismatched volume shapes raise ValueError."""
        rng = np.random.default_rng(42)
        bad_uri = str(temp_dir / "bad_vol")
        bad_data = rng.random((64, 64, 32), dtype=np.float32)
        bad_vol = Volume.from_numpy(bad_uri, bad_data)
        bad_vol.set_obs_id("BAD001")

        volumes_with_mismatch = volumes + [("BAD001", bad_vol)]
        uri = str(temp_dir / "mismatch_test")

        with pytest.raises(ValueError, match="shape"):
            VolumeCollection._from_volumes(uri, volumes_with_mismatch)


class TestVolumeCollectionILoc:
    """Tests for iloc (integer-location) indexer."""

    def test_iloc_single_int(self, populated_collection_module: VolumeCollection):
        """iloc[int] returns single Volume."""
        vol = populated_collection_module.iloc[0]
        assert vol.obs_id == populated_collection_module.obs_ids[0]

    def test_iloc_negative_index(self, populated_collection_module: VolumeCollection):
        """iloc[-1] returns last Volume."""
        vol = populated_collection_module.iloc[-1]
        assert vol.obs_id == populated_collection_module.obs_ids[-1]

    def test_iloc_slice(self, populated_collection_module: VolumeCollection):
        """iloc[start:stop] returns list of Volumes."""
        vols = populated_collection_module.iloc[0:2]
        assert len(vols) == 2
        assert vols[0].obs_id == populated_collection_module.obs_ids[0]
        assert vols[1].obs_id == populated_collection_module.obs_ids[1]

    def test_iloc_slice_with_step(self, populated_collection_module: VolumeCollection):
        """iloc[::2] returns every other Volume."""
        vols = populated_collection_module.iloc[::2]
        assert len(vols) == 2
        assert vols[0].obs_id == populated_collection_module.obs_ids[0]
        assert vols[1].obs_id == populated_collection_module.obs_ids[2]

    def test_iloc_list(self, populated_collection_module: VolumeCollection):
        """iloc[[0, 2]] returns specific Volumes."""
        vols = populated_collection_module.iloc[[0, 2]]
        assert len(vols) == 2
        assert vols[0].obs_id == populated_collection_module.obs_ids[0]
        assert vols[1].obs_id == populated_collection_module.obs_ids[2]

    def test_iloc_out_of_range_raises(self, populated_collection_module: VolumeCollection):
        """iloc[99] raises IndexError."""
        with pytest.raises(IndexError):
            populated_collection_module.iloc[99]

    def test_iloc_invalid_type_raises(self, populated_collection_module: VolumeCollection):
        """iloc[1.5] raises TypeError."""
        with pytest.raises(TypeError):
            populated_collection_module.iloc[1.5]


class TestVolumeCollectionLoc:
    """Tests for loc (label-based) indexer."""

    def test_loc_single_str(self, populated_collection_module: VolumeCollection):
        """loc[str] returns single Volume."""
        obs_id = populated_collection_module.obs_ids[1]
        vol = populated_collection_module.loc[obs_id]
        assert vol.obs_id == obs_id

    def test_loc_list(self, populated_collection_module: VolumeCollection):
        """loc[[str, str]] returns list of Volumes."""
        obs_ids = [populated_collection_module.obs_ids[0], populated_collection_module.obs_ids[2]]
        vols = populated_collection_module.loc[obs_ids]
        assert len(vols) == 2
        assert vols[0].obs_id == obs_ids[0]
        assert vols[1].obs_id == obs_ids[1]

    def test_loc_not_found_raises(self, populated_collection_module: VolumeCollection):
        """loc["NONEXISTENT"] raises KeyError."""
        with pytest.raises(KeyError):
            populated_collection_module.loc["NONEXISTENT"]

    def test_loc_invalid_type_raises(self, populated_collection_module: VolumeCollection):
        """loc[123] raises TypeError."""
        with pytest.raises(TypeError):
            populated_collection_module.loc[123]


class TestVolumeCollectionGetitem:
    """Tests for __getitem__ convenience access."""

    def test_getitem_int(self, populated_collection_module: VolumeCollection):
        """collection[int] delegates to iloc."""
        vol = populated_collection_module[0]
        assert vol.obs_id == populated_collection_module.obs_ids[0]

    def test_getitem_str(self, populated_collection_module: VolumeCollection):
        """collection[str] delegates to loc."""
        obs_id = populated_collection_module.obs_ids[2]
        vol = populated_collection_module[obs_id]
        assert vol.obs_id == obs_id

    def test_getitem_slice(self, populated_collection_module: VolumeCollection):
        """collection[start:stop] delegates to iloc."""
        vols = populated_collection_module[1:3]
        assert len(vols) == 2
        assert vols[0].obs_id == populated_collection_module.obs_ids[1]

    def test_getitem_list_int(self, populated_collection_module: VolumeCollection):
        """collection[[int, int]] delegates to iloc."""
        vols = populated_collection_module[[0, 2]]
        assert len(vols) == 2

    def test_getitem_list_str(self, populated_collection_module: VolumeCollection):
        """collection[[str, str]] delegates to loc."""
        obs_ids = [populated_collection_module.obs_ids[0], populated_collection_module.obs_ids[1]]
        vols = populated_collection_module[obs_ids]
        assert len(vols) == 2

    def test_getitem_empty_list(self, populated_collection_module: VolumeCollection):
        """collection[[]] returns empty list."""
        vols = populated_collection_module[[]]
        assert vols == []

    def test_getitem_invalid_type_raises(self, populated_collection_module: VolumeCollection):
        """collection[1.5] raises TypeError."""
        with pytest.raises(TypeError):
            populated_collection_module[1.5]


class TestVolumeCollectionIndex:
    """Tests for index property."""

    def test_index_get_index(self, populated_collection_module: VolumeCollection):
        """index.get_index() maps obs_id to integer index."""
        for i, obs_id in enumerate(populated_collection_module.obs_ids):
            assert populated_collection_module.index.get_index(obs_id) == i

    def test_index_get_key(self, populated_collection_module: VolumeCollection):
        """index.get_key() maps integer index to obs_id."""
        for i, expected_obs_id in enumerate(populated_collection_module.obs_ids):
            assert populated_collection_module.index.get_key(i) == expected_obs_id

    def test_index_keys(self, populated_collection_module: VolumeCollection):
        """index.keys returns tuple of all obs_ids."""
        assert populated_collection_module.index.keys == tuple(
            populated_collection_module.obs_ids
        )

    def test_index_len(self, populated_collection_module: VolumeCollection):
        """len(index) returns number of volumes."""
        assert len(populated_collection_module.index) == 3

    def test_index_contains(self, populated_collection_module: VolumeCollection):
        """'obs_id' in index works correctly."""
        first_obs_id = populated_collection_module.obs_ids[0]
        assert first_obs_id in populated_collection_module.index
        assert "NONEXISTENT" not in populated_collection_module.index


class TestVolumeCollectionObsRows:
    """Tests for obs row retrieval."""

    def test_get_obs_row_by_obs_id(self, populated_collection_module: VolumeCollection):
        """Get obs row by obs_id string."""
        obs_id = populated_collection_module.obs_ids[1]
        row = populated_collection_module.get_obs_row_by_obs_id(obs_id)
        assert row["obs_id"].iloc[0] == obs_id

    def test_get_obs_row_by_index_pattern(self, populated_collection_module: VolumeCollection):
        """Get obs row by integer index using index.get_key() pattern."""
        obs_id = populated_collection_module.index.get_key(0)
        row = populated_collection_module.get_obs_row_by_obs_id(obs_id)
        assert "obs_id" in row.columns
        assert row["obs_id"].iloc[0] == populated_collection_module.obs_ids[0]


class TestVolumeCollectionProperties:
    """Tests for VolumeCollection properties."""

    def test_shape_property(self, populated_collection_module: VolumeCollection):
        """Shape property returns (X, Y, Z)."""
        assert len(populated_collection_module.shape) == 3

    def test_n_volumes_property(self, populated_collection_module: VolumeCollection):
        """n_volumes returns volume count."""
        assert len(populated_collection_module) == 3

    def test_len(self, populated_collection_module: VolumeCollection):
        """len(collection) returns volume count."""
        assert len(populated_collection_module) == 3

    def test_obs_ids_property(self, populated_collection_module: VolumeCollection):
        """obs_ids returns list of all obs_ids."""
        obs_ids = populated_collection_module.obs_ids
        assert len(obs_ids) == 3

class TestVolumeCollectionRoundtrip:
    """Integration tests for complete workflows."""

    def test_from_volumes_roundtrip(
        self, temp_dir: Path, volumes: list[tuple[str, Volume]]
    ):
        """Create from volumes and verify data roundtrip."""
        uri = str(temp_dir / "roundtrip_test")
        VolumeCollection._from_volumes(uri, volumes)

        reopened = VolumeCollection(uri)
        assert len(reopened) == 3
        first_obs_id = volumes[0][0]
        assert reopened[first_obs_id].obs_id == first_obs_id

    def test_obs_id_consistency(self, populated_collection_module: VolumeCollection):
        """Verify obs_id consistency between metadata and obs dataframe."""
        for i, obs_id in enumerate(populated_collection_module.obs_ids):
            vol = populated_collection_module.iloc[i]
            assert vol.obs_id == obs_id
            row = populated_collection_module.get_obs_row_by_obs_id(obs_id)
            assert row["obs_id"].iloc[0] == obs_id
