"""Tests for Query and CollectionQuery lazy filter builders."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from radiobject.query import Query, CollectionQuery, QueryCount, VolumeBatch
from radiobject.radi_object import RadiObject, RadiObjectView
from radiobject.volume_collection import VolumeCollection
from radiobject.volume import Volume


class TestQueryCreation:
    """Tests for Query instantiation."""

    def test_query_from_radi_object(self, populated_radi_object_module: RadiObject):
        """radi.query() returns a Query instance."""
        query = populated_radi_object_module.query()

        assert isinstance(query, Query)

    def test_query_from_view(self, populated_radi_object_module: RadiObject):
        """RadiObjectView.to_query() returns a Query instance."""
        view = populated_radi_object_module.iloc[0:2]
        query = view.to_query()

        assert isinstance(query, Query)
        assert len(query) == 2


class TestQuerySubjectFilters:
    """Tests for subject-level filtering methods."""

    def test_filter_subjects_by_ids(self, populated_radi_object_module: RadiObject):
        """filter_subjects() narrows to specific subject IDs."""
        subject_ids = populated_radi_object_module.obs_subject_ids[:2]
        query = populated_radi_object_module.query().filter_subjects(subject_ids)

        assert len(query) == 2

    def test_filter_subjects_chained(self, populated_radi_object_module: RadiObject):
        """Chained filter_subjects() intersects the ID sets."""
        all_ids = populated_radi_object_module.obs_subject_ids
        query = (
            populated_radi_object_module.query()
            .filter_subjects(all_ids[:2])
            .filter_subjects([all_ids[1]])
        )

        assert len(query) == 1

    def test_iloc_single(self, populated_radi_object_module: RadiObject):
        """iloc(int) filters to single subject."""
        query = populated_radi_object_module.query().iloc(0)

        assert len(query) == 1

    def test_iloc_slice(self, populated_radi_object_module: RadiObject):
        """iloc(slice) filters to range of subjects."""
        query = populated_radi_object_module.query().iloc(slice(0, 2))

        assert len(query) == 2

    def test_iloc_list(self, populated_radi_object_module: RadiObject):
        """iloc(list[int]) filters to specific subjects."""
        query = populated_radi_object_module.query().iloc([0, 2])

        assert len(query) == 2

    def test_iloc_negative(self, populated_radi_object_module: RadiObject):
        """iloc(-1) filters to last subject."""
        query = populated_radi_object_module.query().iloc(-1)

        assert len(query) == 1

    def test_loc_single(self, populated_radi_object_module: RadiObject):
        """loc(str) filters to single subject by ID."""
        subject_id = populated_radi_object_module.obs_subject_ids[1]
        query = populated_radi_object_module.query().loc(subject_id)

        assert len(query) == 1

    def test_loc_list(self, populated_radi_object_module: RadiObject):
        """loc(list[str]) filters to multiple subjects by ID."""
        subject_ids = [
            populated_radi_object_module.obs_subject_ids[0],
            populated_radi_object_module.obs_subject_ids[2],
        ]
        query = populated_radi_object_module.query().loc(subject_ids)

        assert len(query) == 2

    def test_head(self, populated_radi_object_module: RadiObject):
        """head(n) filters to first n subjects."""
        query = populated_radi_object_module.query().head(2)

        assert len(query) == 2

    def test_tail(self, populated_radi_object_module: RadiObject):
        """tail(n) filters to last n subjects."""
        query = populated_radi_object_module.query().tail(1)

        assert len(query) == 1

    def test_sample_with_seed(self, populated_radi_object_module: RadiObject):
        """sample(n, seed) returns reproducible random sample."""
        q1 = populated_radi_object_module.query().sample(2, seed=42)
        q2 = populated_radi_object_module.query().sample(2, seed=42)

        assert len(q1) == 2
        assert q1.to_obs_meta()["obs_subject_id"].tolist() == q2.to_obs_meta()["obs_subject_id"].tolist()


class TestQueryCollectionFilters:
    """Tests for collection-level filtering methods."""

    def test_select_collections(self, populated_radi_object_module: RadiObject):
        """select_collections() limits output collections."""
        query = populated_radi_object_module.query().select_collections(["T1w"])
        count = query.count()

        assert list(count.n_volumes.keys()) == ["T1w"]

    def test_select_collections_chained(self, populated_radi_object_module: RadiObject):
        """Chained select_collections() intersects collections."""
        query = (
            populated_radi_object_module.query()
            .select_collections(["T1w", "flair"])
            .select_collections(["T1w"])
        )
        count = query.count()

        assert list(count.n_volumes.keys()) == ["T1w"]


class TestQueryMaskResolution:
    """Tests for internal mask resolution logic."""

    def test_resolve_preserves_all_subjects(self, populated_radi_object_module: RadiObject):
        """Unfiltered query resolves to all subjects."""
        query = populated_radi_object_module.query()

        assert len(query) == len(populated_radi_object_module)

    def test_resolve_with_subject_filter(self, populated_radi_object_module: RadiObject):
        """Subject filter correctly narrows resolved mask."""
        subject_ids = populated_radi_object_module.obs_subject_ids[:1]
        query = populated_radi_object_module.query().filter_subjects(subject_ids)

        assert len(query) == 1


class TestQueryCount:
    """Tests for count() materialization."""

    def test_count_returns_query_count(self, populated_radi_object_module: RadiObject):
        """count() returns QueryCount dataclass."""
        count = populated_radi_object_module.query().count()

        assert isinstance(count, QueryCount)
        assert count.n_subjects == 3

    def test_count_volume_counts(self, populated_radi_object_module: RadiObject):
        """count() includes volume counts per collection."""
        count = populated_radi_object_module.query().count()

        assert len(count.n_volumes) == 4
        for name, n in count.n_volumes.items():
            assert n == 3


class TestQueryToObsMeta:
    """Tests for to_obs_meta() materialization."""

    def test_to_obs_meta_returns_dataframe(self, populated_radi_object_module: RadiObject):
        """to_obs_meta() returns filtered DataFrame."""
        query = populated_radi_object_module.query().head(2)
        df = query.to_obs_meta()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "obs_subject_id" in df.columns


class TestQueryIterVolumes:
    """Tests for iter_volumes() materialization."""

    def test_iter_volumes_yields_volumes(self, populated_radi_object_module: RadiObject):
        """iter_volumes() yields Volume instances."""
        query = populated_radi_object_module.query().head(1).select_collections(["T1w"])
        volumes = list(query.iter_volumes())

        assert len(volumes) == 1
        assert all(isinstance(v, Volume) for v in volumes)

    def test_iter_volumes_specific_collection(self, populated_radi_object_module: RadiObject):
        """iter_volumes(collection_name) yields from specific collection."""
        query = populated_radi_object_module.query().head(1)
        volumes = list(query.iter_volumes(collection_name="T1w"))

        assert len(volumes) == 1


class TestQueryIterBatches:
    """Tests for iter_batches() materialization."""

    def test_iter_batches_yields_volume_batch(self, populated_radi_object_module: RadiObject):
        """iter_batches() yields VolumeBatch instances."""
        query = populated_radi_object_module.query().head(2).select_collections(["T1w"])
        batches = list(query.iter_batches(batch_size=2))

        assert len(batches) == 1
        assert isinstance(batches[0], VolumeBatch)

    def test_batch_contains_stacked_arrays(self, populated_radi_object_module: RadiObject):
        """VolumeBatch.volumes contains stacked numpy arrays."""
        query = populated_radi_object_module.query().head(2).select_collections(["T1w"])
        batch = next(query.iter_batches(batch_size=2))

        assert "T1w" in batch.volumes
        assert batch.volumes["T1w"].ndim == 4
        assert batch.volumes["T1w"].shape[0] == 2

    def test_batch_subject_ids(self, populated_radi_object_module: RadiObject):
        """VolumeBatch.subject_ids contains subject identifiers."""
        query = populated_radi_object_module.query().head(2).select_collections(["T1w"])
        batch = next(query.iter_batches(batch_size=2))

        assert len(batch.subject_ids) == 2


class TestQueryToRadiObject:
    """Tests for to_radi_object() materialization."""

    def test_to_radi_object_creates_new_radi(
        self,
        temp_dir: Path,
        populated_radi_object: RadiObject,
    ):
        """to_radi_object() creates a new RadiObject."""
        query = populated_radi_object.query().head(2).select_collections(["T1w"])
        new_uri = str(temp_dir / "query_materialized")
        new_radi = query.to_radi_object(new_uri)

        assert isinstance(new_radi, RadiObject)
        assert len(new_radi) == 2
        assert new_radi.n_collections == 1

    def test_to_radi_object_streaming(
        self,
        temp_dir: Path,
        populated_radi_object: RadiObject,
    ):
        """to_radi_object(streaming=True) uses memory-efficient writes."""
        query = populated_radi_object.query().head(2)
        new_uri = str(temp_dir / "query_streaming")
        new_radi = query.to_radi_object(new_uri, streaming=True)

        assert len(new_radi) == 2


class TestQueryRepr:
    """Tests for Query string representation."""

    def test_repr_shows_counts(self, populated_radi_object_module: RadiObject):
        """Query repr shows subject and volume counts."""
        query = populated_radi_object_module.query()
        repr_str = repr(query)

        assert "3 subjects" in repr_str
        assert "volumes" in repr_str


# =============================================================================
# CollectionQuery Tests
# =============================================================================


class TestCollectionQueryCreation:
    """Tests for CollectionQuery instantiation."""

    def test_query_from_volume_collection(self, populated_collection_module: VolumeCollection):
        """vc.query() returns a CollectionQuery instance."""
        query = populated_collection_module.query()

        assert isinstance(query, CollectionQuery)


class TestCollectionQueryFilters:
    """Tests for CollectionQuery filtering methods."""

    def test_iloc_single(self, populated_collection_module: VolumeCollection):
        """iloc(int) filters to single volume."""
        query = populated_collection_module.query().iloc(0)

        assert query.count() == 1

    def test_iloc_slice(self, populated_collection_module: VolumeCollection):
        """iloc(slice) filters to range of volumes."""
        query = populated_collection_module.query().iloc(slice(0, 2))

        assert query.count() == 2

    def test_loc_single(self, populated_collection_module: VolumeCollection):
        """loc(str) filters to single volume by obs_id."""
        obs_id = populated_collection_module.obs_ids[0]
        query = populated_collection_module.query().loc(obs_id)

        assert query.count() == 1

    def test_head(self, populated_collection_module: VolumeCollection):
        """head(n) filters to first n volumes."""
        query = populated_collection_module.query().head(2)

        assert query.count() == 2

    def test_tail(self, populated_collection_module: VolumeCollection):
        """tail(n) filters to last n volumes."""
        query = populated_collection_module.query().tail(1)

        assert query.count() == 1

    def test_sample_with_seed(self, populated_collection_module: VolumeCollection):
        """sample(n, seed) returns reproducible random sample."""
        q1 = populated_collection_module.query().sample(2, seed=42)
        q2 = populated_collection_module.query().sample(2, seed=42)

        df1 = q1.to_obs()
        df2 = q2.to_obs()
        assert df1["obs_id"].tolist() == df2["obs_id"].tolist()

    def test_filter_subjects(self, populated_radi_object_module: RadiObject):
        """filter_subjects() narrows to volumes belonging to specific subjects."""
        vc = populated_radi_object_module.T1w
        subject_id = populated_radi_object_module.obs_subject_ids[0]
        query = vc.query().filter_subjects([subject_id])

        assert query.count() == 1


class TestCollectionQueryMaterialization:
    """Tests for CollectionQuery materialization methods."""

    def test_count_returns_int(self, populated_collection_module: VolumeCollection):
        """count() returns integer volume count."""
        count = populated_collection_module.query().count()

        assert isinstance(count, int)
        assert count == 3

    def test_to_obs_returns_dataframe(self, populated_collection_module: VolumeCollection):
        """to_obs() returns filtered obs DataFrame."""
        query = populated_collection_module.query().head(2)
        df = query.to_obs()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    def test_iter_volumes_yields_volumes(self, populated_collection_module: VolumeCollection):
        """iter_volumes() yields Volume instances."""
        query = populated_collection_module.query().head(2)
        volumes = list(query.iter_volumes())

        assert len(volumes) == 2
        assert all(isinstance(v, Volume) for v in volumes)

    def test_to_numpy_stack(self, populated_collection_module: VolumeCollection):
        """to_numpy_stack() returns stacked array."""
        query = populated_collection_module.query().head(2)
        stack = query.to_numpy_stack()

        assert stack.ndim == 4
        assert stack.shape[0] == 2

    def test_to_volume_collection(
        self,
        temp_dir: Path,
        populated_collection: VolumeCollection,
    ):
        """to_volume_collection() creates a new VolumeCollection."""
        query = populated_collection.query().head(2)
        new_uri = str(temp_dir / "coll_query_materialized")
        new_vc = query.to_volume_collection(new_uri)

        assert isinstance(new_vc, VolumeCollection)
        assert len(new_vc) == 2


class TestCollectionQueryConvenience:
    """Tests for VolumeCollection convenience filter methods."""

    def test_vc_head(self, populated_collection_module: VolumeCollection):
        """VolumeCollection.head() returns CollectionQuery."""
        query = populated_collection_module.head(2)

        assert isinstance(query, CollectionQuery)
        assert query.count() == 2

    def test_vc_tail(self, populated_collection_module: VolumeCollection):
        """VolumeCollection.tail() returns CollectionQuery."""
        query = populated_collection_module.tail(1)

        assert isinstance(query, CollectionQuery)
        assert query.count() == 1

    def test_vc_sample(self, populated_collection_module: VolumeCollection):
        """VolumeCollection.sample() returns CollectionQuery."""
        query = populated_collection_module.sample(2, seed=42)

        assert isinstance(query, CollectionQuery)
        assert query.count() == 2
