"""Tests for LazyQuery and EagerQuery query builders."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from radiobject.query import EagerQuery, LazyQuery
from radiobject.radi_object import RadiObject
from radiobject.volume import Volume
from radiobject.volume_collection import VolumeCollection


class TestLazyQueryCreation:
    """Tests for LazyQuery instantiation."""

    def test_query_from_volume_collection(self, populated_collection_module: VolumeCollection):
        """vc.lazy() returns a LazyQuery instance."""
        query = populated_collection_module.lazy()

        assert isinstance(query, LazyQuery)


class TestLazyQueryFilters:
    """Tests for LazyQuery filtering methods."""

    @pytest.mark.parametrize(
        ("indexer", "expected_count"),
        [
            (0, 1),  # iloc(int)
            (slice(0, 2), 2),  # iloc(slice)
        ],
        ids=["single", "slice"],
    )
    def test_iloc(self, populated_collection_module: VolumeCollection, indexer, expected_count):
        """iloc filters volumes by integer position."""
        query = populated_collection_module.lazy().iloc(indexer)
        assert query.count() == expected_count

    def test_loc(self, populated_collection_module: VolumeCollection):
        """loc(str) filters to single volume by obs_id."""
        obs_id = populated_collection_module.obs_ids[0]
        query = populated_collection_module.lazy().loc(obs_id)
        assert query.count() == 1

    @pytest.mark.parametrize(
        ("method", "n", "expected_count"),
        [
            ("head", 2, 2),
            ("tail", 1, 1),
        ],
    )
    def test_head_tail(
        self, populated_collection_module: VolumeCollection, method, n, expected_count
    ):
        """head/tail filter to first/last n volumes."""
        query = getattr(populated_collection_module.lazy(), method)(n)
        assert query.count() == expected_count

    def test_sample_with_seed(self, populated_collection_module: VolumeCollection):
        """sample(n, seed) returns reproducible random sample."""
        q1 = populated_collection_module.lazy().sample(2, seed=42)
        q2 = populated_collection_module.lazy().sample(2, seed=42)
        assert q1.to_obs()["obs_id"].tolist() == q2.to_obs()["obs_id"].tolist()

    def test_filter_subjects(self, populated_radi_object_module: RadiObject):
        """filter_subjects() narrows to volumes belonging to specific subjects."""
        vc = populated_radi_object_module.T1w
        subject_id = populated_radi_object_module.obs_subject_ids[0]
        query = vc.lazy().filter_subjects([subject_id])
        assert query.count() == 1


class TestLazyQueryMaterialization:
    """Tests for LazyQuery materialization methods."""

    def test_count(self, populated_collection_module: VolumeCollection):
        """count() returns integer volume count."""
        assert populated_collection_module.lazy().count() == 3

    def test_to_obs(self, populated_collection_module: VolumeCollection):
        """to_obs() returns filtered obs DataFrame."""
        df = populated_collection_module.lazy().head(2).to_obs()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    def test_iter_volumes(self, populated_collection_module: VolumeCollection):
        """iter_volumes() yields Volume instances."""
        volumes = list(populated_collection_module.lazy().head(2).iter_volumes())
        assert len(volumes) == 2
        assert all(isinstance(v, Volume) for v in volumes)

    def test_to_numpy_stack(self, populated_collection_module: VolumeCollection):
        """to_numpy_stack() returns stacked 4D array."""
        stack = populated_collection_module.lazy().head(2).to_numpy_stack()
        assert stack.ndim == 4
        assert stack.shape[0] == 2

    def test_write(self, temp_dir: Path, populated_collection: VolumeCollection):
        """write() creates a new VolumeCollection."""
        new_vc = populated_collection.lazy().head(2).write(str(temp_dir / "coll_query_write"))
        assert isinstance(new_vc, VolumeCollection)
        assert len(new_vc) == 2


class TestLazyQueryMap:
    """Tests for LazyQuery.map() transform method."""

    def test_map_applies_transform(
        self,
        temp_dir: Path,
        populated_collection: VolumeCollection,
    ):
        """map() applies transform function during write."""
        query = populated_collection.lazy().head(1)
        original_vol = next(query.iter_volumes())
        original_data = original_vol.to_numpy()

        new_uri = str(temp_dir / "coll_map_transform")
        new_vc = query.map(lambda v, obs: v * 3).write(new_uri)

        new_vol = next(iter(new_vc))
        new_data = new_vol.to_numpy()

        np.testing.assert_array_almost_equal(new_data, original_data * 3)

    def test_map_chained_composes(
        self,
        temp_dir: Path,
        populated_collection: VolumeCollection,
    ):
        """Chained map() calls compose transforms in order."""
        query = populated_collection.lazy().head(1)
        original_vol = next(query.iter_volumes())
        original_data = original_vol.to_numpy()

        new_uri = str(temp_dir / "coll_map_chained")
        new_vc = query.map(lambda v, obs: v * 3).map(lambda v, obs: v - 5).write(new_uri)

        new_vol = next(iter(new_vc))
        new_data = new_vol.to_numpy()

        np.testing.assert_array_almost_equal(new_data, original_data * 3 - 5)

    def test_map_with_obs_updates(
        self,
        temp_dir: Path,
        populated_collection: VolumeCollection,
    ):
        """map() can return (volume, obs_updates_dict) to annotate obs metadata."""
        query = populated_collection.lazy().head(1)

        def annotating_transform(volume, obs):
            return volume, {"processed": True, "vol_mean": float(volume.mean())}

        new_uri = str(temp_dir / "coll_map_obs_updates")
        new_vc = query.map(annotating_transform).write(new_uri)

        obs_df = new_vc.obs.read()
        assert "processed" in obs_df.columns
        assert "vol_mean" in obs_df.columns


# =============================================================================
# EagerQuery Tests
# =============================================================================


class TestEagerQuery:
    """Tests for VolumeCollection.map() returning EagerQuery."""

    def test_map_returns_eager_query(self, populated_collection: VolumeCollection):
        """vc.map(fn) returns an EagerQuery."""
        result = populated_collection.map(lambda v, obs: v * 2)
        assert isinstance(result, EagerQuery)
        assert len(result) == len(populated_collection)

    def test_eager_to_list(self, populated_collection: VolumeCollection):
        """EagerQuery.to_list() returns list of (result, obs_row) tuples."""
        pairs = populated_collection.map(lambda v, obs: v.mean()).to_list()
        assert len(pairs) == len(populated_collection)
        results = [r for r, _ in pairs]
        assert all(isinstance(r, (float, np.floating)) for r in results)

    def test_eager_chained_map(self, populated_collection: VolumeCollection):
        """EagerQuery.map() chains transforms."""
        original_data = populated_collection.iloc[0].to_numpy()
        pairs = populated_collection.map(lambda v, obs: v * 2).map(lambda v, obs: v + 1).to_list()
        result, _ = pairs[0]
        np.testing.assert_array_almost_equal(result, original_data * 2 + 1)

    def test_eager_write(self, temp_dir: Path, populated_collection: VolumeCollection):
        """EagerQuery.write() persists results to a new VolumeCollection."""
        new_uri = str(temp_dir / "eager_write")
        new_vc = populated_collection.map(lambda v, obs: v * 2).write(new_uri)

        assert isinstance(new_vc, VolumeCollection)
        assert len(new_vc) == len(populated_collection)

        original_data = populated_collection.iloc[0].to_numpy()
        new_data = new_vc.iloc[0].to_numpy()
        np.testing.assert_array_almost_equal(new_data, original_data * 2)

    def test_eager_map_with_obs_updates(
        self, temp_dir: Path, populated_collection: VolumeCollection
    ):
        """EagerQuery.map() accumulates obs_updates and writes them."""

        def transform(volume, obs):
            return volume * 2, {"scaled": True}

        new_uri = str(temp_dir / "eager_obs_updates")
        new_vc = populated_collection.map(transform).write(new_uri)

        obs_df = new_vc.obs.read()
        assert "scaled" in obs_df.columns

    def test_eager_chained_obs_updates(
        self, temp_dir: Path, populated_collection: VolumeCollection
    ):
        """Chained EagerQuery.map() calls accumulate obs_updates from all transforms."""

        def step1(volume, obs):
            return volume * 2, {"step1": True}

        def step2(volume, obs):
            return volume + 1, {"step2": True}

        new_uri = str(temp_dir / "eager_chained_updates")
        new_vc = populated_collection.map(step1).map(step2).write(new_uri)

        obs_df = new_vc.obs.read()
        assert "step1" in obs_df.columns
        assert "step2" in obs_df.columns

    def test_eager_iter(self, populated_collection: VolumeCollection):
        """EagerQuery supports iteration yielding (result, obs_row) tuples."""
        eq = populated_collection.map(lambda v, obs: v.shape)
        shapes = [r for r, _ in eq]
        assert len(shapes) == len(populated_collection)

    def test_eager_getitem(self, populated_collection: VolumeCollection):
        """EagerQuery supports indexing returning (result, obs_row)."""
        eq = populated_collection.map(lambda v, obs: v.mean())
        val, obs = eq[0]
        assert isinstance(val, (float, np.floating))
        assert isinstance(obs, pd.Series)


class TestEagerMapBatches:
    """Tests for VolumeCollection.map_batches()."""

    def test_map_batches_returns_eager_query(self, populated_collection: VolumeCollection):
        """map_batches(fn) returns an EagerQuery."""

        def batch_fn(batch):
            return [v * 2 for v, obs in batch]

        result = populated_collection.map_batches(batch_fn, batch_size=2)
        assert isinstance(result, EagerQuery)
        assert len(result) == len(populated_collection)
