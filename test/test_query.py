"""Tests for Query and CollectionQuery lazy filter builders."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from radiobject.query import CollectionQuery, Query, QueryCount, VolumeBatch
from radiobject.radi_object import RadiObject
from radiobject.volume import Volume
from radiobject.volume_collection import VolumeCollection


class TestQueryCreation:
    """Tests for Query instantiation."""

    def test_query_from_radi_object(self, populated_radi_object_module: RadiObject):
        """radi.lazy() returns a Query instance."""
        query = populated_radi_object_module.lazy()

        assert isinstance(query, Query)

    def test_query_from_view(self, populated_radi_object_module: RadiObject):
        """RadiObject view lazy() returns a Query instance."""
        view = populated_radi_object_module.iloc[0:2]
        query = view.lazy()

        assert isinstance(query, Query)
        assert len(query) == 2


class TestQuerySubjectFilters:
    """Tests for subject-level filtering methods."""

    def test_filter_subjects_by_ids(self, populated_radi_object_module: RadiObject):
        """filter_subjects() narrows to specific subject IDs."""
        subject_ids = populated_radi_object_module.obs_subject_ids[:2]
        query = populated_radi_object_module.lazy().filter_subjects(subject_ids)

        assert len(query) == 2

    def test_filter_subjects_chained(self, populated_radi_object_module: RadiObject):
        """Chained filter_subjects() intersects the ID sets."""
        all_ids = populated_radi_object_module.obs_subject_ids
        query = (
            populated_radi_object_module.lazy()
            .filter_subjects(all_ids[:2])
            .filter_subjects([all_ids[1]])
        )

        assert len(query) == 1

    @pytest.mark.parametrize(
        ("indexer", "expected_len"),
        [
            (0, 1),  # iloc(int) - single subject
            (slice(0, 2), 2),  # iloc(slice) - range
            ([0, 2], 2),  # iloc(list) - specific indices
            (-1, 1),  # iloc(-1) - last subject
        ],
        ids=["single", "slice", "list", "negative"],
    )
    def test_iloc(self, populated_radi_object_module: RadiObject, indexer, expected_len):
        """iloc filters subjects by integer position."""
        query = populated_radi_object_module.lazy().iloc(indexer)
        assert len(query) == expected_len

    def test_loc_single(self, populated_radi_object_module: RadiObject):
        """loc(str) filters to single subject by ID."""
        subject_id = populated_radi_object_module.obs_subject_ids[1]
        query = populated_radi_object_module.lazy().loc(subject_id)
        assert len(query) == 1

    def test_loc_list(self, populated_radi_object_module: RadiObject):
        """loc(list[str]) filters to multiple subjects by ID."""
        subject_ids = [
            populated_radi_object_module.obs_subject_ids[0],
            populated_radi_object_module.obs_subject_ids[2],
        ]
        query = populated_radi_object_module.lazy().loc(subject_ids)
        assert len(query) == 2

    @pytest.mark.parametrize(
        ("method", "n", "expected_len"),
        [
            ("head", 2, 2),
            ("tail", 1, 1),
        ],
    )
    def test_head_tail(self, populated_radi_object_module: RadiObject, method, n, expected_len):
        """head/tail filter to first/last n subjects."""
        query = getattr(populated_radi_object_module.lazy(), method)(n)
        assert len(query) == expected_len

    def test_sample_with_seed(self, populated_radi_object_module: RadiObject):
        """sample(n, seed) returns reproducible random sample."""
        q1 = populated_radi_object_module.lazy().sample(2, seed=42)
        q2 = populated_radi_object_module.lazy().sample(2, seed=42)

        assert len(q1) == 2
        assert (
            q1.to_obs_meta()["obs_subject_id"].tolist()
            == q2.to_obs_meta()["obs_subject_id"].tolist()
        )


class TestQueryCollectionFilters:
    """Tests for collection-level filtering methods."""

    def test_select_collections(self, populated_radi_object_module: RadiObject):
        """select_collections() limits output collections."""
        query = populated_radi_object_module.lazy().select_collections(["T1w"])
        count = query.count()

        assert list(count.n_volumes.keys()) == ["T1w"]

    def test_select_collections_chained(self, populated_radi_object_module: RadiObject):
        """Chained select_collections() intersects collections."""
        query = (
            populated_radi_object_module.lazy()
            .select_collections(["T1w", "flair"])
            .select_collections(["T1w"])
        )
        count = query.count()

        assert list(count.n_volumes.keys()) == ["T1w"]


class TestQueryMaskResolution:
    """Tests for internal mask resolution logic."""

    def test_resolve_preserves_all_subjects(self, populated_radi_object_module: RadiObject):
        """Unfiltered query resolves to all subjects."""
        query = populated_radi_object_module.lazy()

        assert len(query) == len(populated_radi_object_module)

    def test_resolve_with_subject_filter(self, populated_radi_object_module: RadiObject):
        """Subject filter correctly narrows resolved mask."""
        subject_ids = populated_radi_object_module.obs_subject_ids[:1]
        query = populated_radi_object_module.lazy().filter_subjects(subject_ids)

        assert len(query) == 1


class TestQueryCount:
    """Tests for count() materialization."""

    def test_count(self, populated_radi_object_module: RadiObject):
        """count() returns QueryCount with subject and volume counts."""
        count = populated_radi_object_module.lazy().count()

        assert isinstance(count, QueryCount)
        assert count.n_subjects == 3
        assert len(count.n_volumes) == 4
        assert all(n == 3 for n in count.n_volumes.values())


class TestQueryToObsMeta:
    """Tests for to_obs_meta() materialization."""

    def test_to_obs_meta_returns_dataframe(self, populated_radi_object_module: RadiObject):
        """to_obs_meta() returns filtered DataFrame."""
        query = populated_radi_object_module.lazy().head(2)
        df = query.to_obs_meta()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "obs_subject_id" in df.columns


class TestQueryIterVolumes:
    """Tests for iter_volumes() materialization."""

    def test_iter_volumes_yields_volumes(self, populated_radi_object_module: RadiObject):
        """iter_volumes() yields Volume instances."""
        query = populated_radi_object_module.lazy().head(1).select_collections(["T1w"])
        volumes = list(query.iter_volumes())

        assert len(volumes) == 1
        assert all(isinstance(v, Volume) for v in volumes)

    def test_iter_volumes_specific_collection(self, populated_radi_object_module: RadiObject):
        """iter_volumes(collection_name) yields from specific collection."""
        query = populated_radi_object_module.lazy().head(1)
        volumes = list(query.iter_volumes(collection_name="T1w"))

        assert len(volumes) == 1


class TestQueryIterBatches:
    """Tests for iter_batches() materialization."""

    def test_iter_batches(self, populated_radi_object_module: RadiObject):
        """iter_batches() yields VolumeBatch with stacked arrays and subject IDs."""
        query = populated_radi_object_module.lazy().head(2).select_collections(["T1w"])
        batch = next(query.iter_batches(batch_size=2))

        assert isinstance(batch, VolumeBatch)
        assert "T1w" in batch.volumes
        assert batch.volumes["T1w"].shape == (2, *batch.volumes["T1w"].shape[1:])
        assert len(batch.subject_ids) == 2


class TestQueryMap:
    """Tests for map() transform method."""

    def test_map_applies_transform(
        self,
        temp_dir: Path,
        populated_radi_object: RadiObject,
    ):
        """map() applies transform function during materialization."""
        import numpy as np

        query = populated_radi_object.lazy().head(1).select_collections(["T1w"])
        original_vol = next(query.iter_volumes())
        original_data = original_vol.to_numpy()

        # Double the values
        new_uri = str(temp_dir / "query_map_transform")
        new_radi = query.map(lambda v: v * 2).materialize(new_uri)

        new_vol = next(iter(new_radi.T1w))
        new_data = new_vol.to_numpy()

        np.testing.assert_array_almost_equal(new_data, original_data * 2)

    def test_map_chained_composes(
        self,
        temp_dir: Path,
        populated_radi_object: RadiObject,
    ):
        """Chained map() calls compose transforms in order."""
        import numpy as np

        query = populated_radi_object.lazy().head(1).select_collections(["T1w"])
        original_vol = next(query.iter_volumes())
        original_data = original_vol.to_numpy()

        # Chain: double, then add 10
        new_uri = str(temp_dir / "query_map_chained")
        new_radi = query.map(lambda v: v * 2).map(lambda v: v + 10).materialize(new_uri)

        new_vol = next(iter(new_radi.T1w))
        new_data = new_vol.to_numpy()

        np.testing.assert_array_almost_equal(new_data, original_data * 2 + 10)

    def test_map_without_transform_unchanged(
        self,
        temp_dir: Path,
        populated_radi_object: RadiObject,
    ):
        """Query without map() preserves original data."""
        import numpy as np

        query = populated_radi_object.lazy().head(1).select_collections(["T1w"])
        original_vol = next(query.iter_volumes())
        original_data = original_vol.to_numpy()

        new_uri = str(temp_dir / "query_no_transform")
        new_radi = query.materialize(new_uri)

        new_vol = next(iter(new_radi.T1w))
        new_data = new_vol.to_numpy()

        np.testing.assert_array_equal(new_data, original_data)


class TestQueryMaterialize:
    """Tests for materialize() (formerly to_radi_object())."""

    def test_materialize_creates_new_radi(
        self,
        temp_dir: Path,
        populated_radi_object: RadiObject,
    ):
        """materialize() creates a new RadiObject."""
        query = populated_radi_object.lazy().head(2).select_collections(["T1w"])
        new_uri = str(temp_dir / "query_materialized")
        new_radi = query.materialize(new_uri)

        assert isinstance(new_radi, RadiObject)
        assert len(new_radi) == 2
        assert new_radi.n_collections == 1

    def test_materialize_streaming(
        self,
        temp_dir: Path,
        populated_radi_object: RadiObject,
    ):
        """materialize(streaming=True) uses memory-efficient writes."""
        query = populated_radi_object.lazy().head(2)
        new_uri = str(temp_dir / "query_streaming")
        new_radi = query.materialize(new_uri, streaming=True)

        assert len(new_radi) == 2


class TestQueryRepr:
    """Tests for Query string representation."""

    def test_repr_shows_counts(self, populated_radi_object_module: RadiObject):
        """Query repr shows subject and volume counts."""
        query = populated_radi_object_module.lazy()
        repr_str = repr(query)

        assert "3 subjects" in repr_str
        assert "volumes" in repr_str


# =============================================================================
# CollectionQuery Tests
# =============================================================================


class TestCollectionQueryCreation:
    """Tests for CollectionQuery instantiation."""

    def test_query_from_volume_collection(self, populated_collection_module: VolumeCollection):
        """vc.lazy() returns a CollectionQuery instance."""
        query = populated_collection_module.lazy()

        assert isinstance(query, CollectionQuery)


class TestCollectionQueryFilters:
    """Tests for CollectionQuery filtering methods."""

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


class TestCollectionQueryMaterialization:
    """Tests for CollectionQuery materialization methods."""

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

    def test_materialize(self, temp_dir: Path, populated_collection: VolumeCollection):
        """materialize() creates a new VolumeCollection."""
        new_vc = (
            populated_collection.lazy()
            .head(2)
            .materialize(str(temp_dir / "coll_query_materialized"))
        )
        assert isinstance(new_vc, VolumeCollection)
        assert len(new_vc) == 2


class TestCollectionQueryMap:
    """Tests for CollectionQuery.map() transform method."""

    def test_map_applies_transform(
        self,
        temp_dir: Path,
        populated_collection: VolumeCollection,
    ):
        """map() applies transform function during materialization."""
        import numpy as np

        query = populated_collection.lazy().head(1)
        original_vol = next(query.iter_volumes())
        original_data = original_vol.to_numpy()

        new_uri = str(temp_dir / "coll_map_transform")
        new_vc = query.map(lambda v: v * 3).materialize(new_uri)

        new_vol = next(iter(new_vc))
        new_data = new_vol.to_numpy()

        np.testing.assert_array_almost_equal(new_data, original_data * 3)

    def test_map_chained_composes(
        self,
        temp_dir: Path,
        populated_collection: VolumeCollection,
    ):
        """Chained map() calls compose transforms in order."""
        import numpy as np

        query = populated_collection.lazy().head(1)
        original_vol = next(query.iter_volumes())
        original_data = original_vol.to_numpy()

        # Chain: triple, then subtract 5
        new_uri = str(temp_dir / "coll_map_chained")
        new_vc = query.map(lambda v: v * 3).map(lambda v: v - 5).materialize(new_uri)

        new_vol = next(iter(new_vc))
        new_data = new_vol.to_numpy()

        np.testing.assert_array_almost_equal(new_data, original_data * 3 - 5)


class TestVolumeCollectionViewMethods:
    """Tests for VolumeCollection convenience filter methods that return views."""

    def test_head_returns_view(self, populated_collection_module: VolumeCollection):
        """head() returns a VolumeCollection view, not CollectionQuery."""
        result = populated_collection_module.head(2)
        assert isinstance(result, VolumeCollection)
        assert result.is_view
        assert len(result) == 2

    def test_tail_returns_view(self, populated_collection_module: VolumeCollection):
        """tail() returns a VolumeCollection view."""
        result = populated_collection_module.tail(1)
        assert isinstance(result, VolumeCollection)
        assert result.is_view
        assert len(result) == 1

    def test_sample_returns_view(self, populated_collection_module: VolumeCollection):
        """sample() returns a VolumeCollection view."""
        result = populated_collection_module.sample(2, seed=42)
        assert isinstance(result, VolumeCollection)
        assert result.is_view
        assert len(result) == 2

    def test_filter_returns_view(self, populated_radi_object_module: RadiObject):
        """filter() returns a VolumeCollection view."""
        vc = populated_radi_object_module.T1w
        # Filter by obs_subject_id (which is in the obs schema)
        result = vc.head(2)  # Use head as filter() needs a valid TileDB query
        assert isinstance(result, VolumeCollection)
        assert result.is_view
