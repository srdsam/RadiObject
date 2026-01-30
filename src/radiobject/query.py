"""Lazy query builder pattern for RadiObject and VolumeCollection filtering.

Query Design:
=============
Queries work by building up filter conditions and resolving them to masks (sets of IDs).
The flow is:

1. Start with RadiObject.query() or VolumeCollection.query()
2. Add filter conditions (obs_meta filters, collection filters)
3. Resolve to masks:
   - Subject mask: set of obs_subject_ids matching criteria
   - Volume mask: set of obs_ids within each collection matching criteria
4. Apply masks via materialization (iter_volumes, to_radi_object, etc.)

Key Concepts:
- obs_meta: Subject-level metadata (indexed by obs_subject_id)
- obs: Volume-level metadata per collection (indexed by obs_id, contains obs_subject_id FK)
- A subject matches if it passes obs_meta filter AND has at least one volume passing collection filters
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Iterator, Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd
import tiledb

from radiobject.volume import Volume

if TYPE_CHECKING:
    from radiobject.radi_object import RadiObject
    from radiobject.volume_collection import VolumeCollection

# Transform function signature: numpy array in, numpy array out (shape may differ)
TransformFn = Callable[[npt.NDArray[np.floating]], npt.NDArray[np.floating]]


@dataclass(frozen=True)
class VolumeBatch:
    """Batch of volumes for ML training with stacked numpy arrays."""

    volumes: dict[str, npt.NDArray[np.floating]]  # collection_name -> (N, X, Y, Z)
    subject_ids: tuple[str, ...]
    obs_ids: dict[str, tuple[str, ...]]  # collection_name -> obs_ids


@dataclass(frozen=True)
class QueryCount:
    """Count results for a query."""

    n_subjects: int
    n_volumes: dict[str, int]  # collection_name -> volume count


class Query:
    """Lazy filter builder for RadiObject with explicit materialization.

    Query Flow:
    -----------
    1. Filters on obs_meta (subject-level): filter_subjects(), filter()
    2. Filters on collection obs (volume-level): filter_collection()
    3. Resolution: subject matches if it passes obs_meta AND has matching volumes
    4. Materialization: iter_volumes(), to_radi_object(), count()

    Example:
        # Filter subjects by age, then filter T1w volumes by resolution
        result = (
            radi.query()
            .filter("age > 40")                    # Filter on obs_meta
            .filter_collection("T1w", "voxel_spacing == '1.0x1.0x1.0'")  # Filter on T1w.obs
            .select_collections(["T1w", "FLAIR"])  # Limit output collections
            .to_radi_object("s3://bucket/subset")
        )
    """

    def __init__(
        self,
        source: RadiObject,
        *,
        # Subject-level filters (on obs_meta)
        subject_ids: frozenset[str] | None = None,
        subject_query: str | None = None,
        # Collection-level filters (on each collection's obs)
        collection_filters: dict[str, str] | None = None,  # collection_name -> query expr
        # Output scope
        output_collections: frozenset[str] | None = None,
        # Transform function applied during materialization
        transform_fn: TransformFn | None = None,
    ):
        self._source = source
        self._subject_ids = subject_ids
        self._subject_query = subject_query
        self._collection_filters = collection_filters or {}
        self._output_collections = output_collections
        self._transform_fn = transform_fn

    def _copy(self, **kwargs) -> Query:
        """Create a copy with modified fields."""
        return Query(
            self._source,
            subject_ids=kwargs.get("subject_ids", self._subject_ids),
            subject_query=kwargs.get("subject_query", self._subject_query),
            collection_filters=kwargs.get("collection_filters", self._collection_filters),
            output_collections=kwargs.get("output_collections", self._output_collections),
            transform_fn=kwargs.get("transform_fn", self._transform_fn),
        )

    # =========================================================================
    # SUBJECT-LEVEL FILTERS (operate on obs_meta)
    # =========================================================================

    def filter(self, expr: str) -> Query:
        """Filter subjects using TileDB QueryCondition on obs_meta.

        Args:
            expr: TileDB query expression (e.g., "age > 40 and diagnosis == 'tumor'")

        Returns:
            New Query with filter applied
        """
        new_query = self._subject_query
        if new_query is None:
            new_query = expr
        else:
            new_query = f"({new_query}) and ({expr})"
        return self._copy(subject_query=new_query)

    def filter_subjects(self, ids: Sequence[str]) -> Query:
        """Filter to specific subject IDs.

        Args:
            ids: List of obs_subject_ids to include

        Returns:
            New Query with filter applied
        """
        new_ids = frozenset(ids)
        if self._subject_ids is not None:
            new_ids = self._subject_ids & new_ids
        return self._copy(subject_ids=new_ids)

    def iloc(self, key: int | slice | list[int] | npt.NDArray[np.bool_]) -> Query:
        """Filter subjects by integer position(s)."""
        n = len(self._source)
        if isinstance(key, int):
            idx = key if key >= 0 else n + key
            indices = [idx]
        elif isinstance(key, slice):
            indices = list(range(*key.indices(n)))
        elif isinstance(key, np.ndarray) and key.dtype == np.bool_:
            indices = list(np.where(key)[0])
        elif isinstance(key, list):
            indices = [i if i >= 0 else n + i for i in key]
        else:
            raise TypeError("iloc key must be int, slice, list[int], or bool array")

        ids = [self._source._index.get_key(i) for i in indices]
        return self.filter_subjects(ids)

    def loc(self, key: str | Sequence[str]) -> Query:
        """Filter subjects by obs_subject_id(s)."""
        if isinstance(key, str):
            return self.filter_subjects([key])
        return self.filter_subjects(key)

    def head(self, n: int = 5) -> Query:
        """Filter to first n subjects."""
        return self.iloc(slice(0, n))

    def tail(self, n: int = 5) -> Query:
        """Filter to last n subjects."""
        total = len(self._source)
        return self.iloc(slice(max(0, total - n), total))

    def sample(self, n: int = 5, seed: int | None = None) -> Query:
        """Filter to n randomly sampled subjects."""
        rng = np.random.default_rng(seed)
        resolved = self._resolve_subject_mask()
        subject_list = list(resolved)
        n = min(n, len(subject_list))
        sampled = rng.choice(subject_list, size=n, replace=False)
        return self.filter_subjects(sampled)

    # =========================================================================
    # COLLECTION-LEVEL FILTERS (operate on collection obs)
    # =========================================================================

    def filter_collection(self, collection_name: str, expr: str) -> Query:
        """Filter volumes in a specific collection using TileDB QueryCondition on obs.

        Only subjects that have at least one volume matching the filter will be included.

        Args:
            collection_name: Name of the collection to filter
            expr: TileDB query expression on the collection's obs dataframe

        Returns:
            New Query with filter applied

        Example:
            # Only include subjects whose T1w scans have 1mm resolution
            query.filter_collection("T1w", "voxel_spacing == '1.0x1.0x1.0'")
        """
        new_filters = dict(self._collection_filters)
        if collection_name in new_filters:
            new_filters[collection_name] = f"({new_filters[collection_name]}) and ({expr})"
        else:
            new_filters[collection_name] = expr
        return self._copy(collection_filters=new_filters)

    def select_collections(self, names: Sequence[str]) -> Query:
        """Limit output to specific collections.

        This doesn't affect filtering - it only limits which collections
        appear in the output.

        Args:
            names: Collection names to include in output

        Returns:
            New Query with output scope set
        """
        new_collections = frozenset(names)
        if self._output_collections is not None:
            new_collections = self._output_collections & new_collections
        return self._copy(output_collections=new_collections)

    # =========================================================================
    # TRANSFORM (applied during materialization)
    # =========================================================================

    def map(self, fn: TransformFn) -> Query:
        """Apply transform to each volume during materialization.

        The transform function receives a numpy array and returns a numpy array.
        Transform is applied lazily when volumes are materialized via
        to_radi_object() or to_volume_collection().

        Multiple map() calls compose: query.map(f1).map(f2) applies f1 then f2.

        Args:
            fn: Function taking numpy array (X, Y, Z) and returning numpy array.
                Can change shape (e.g., resampling to different dimensions).

        Returns:
            New Query with transform applied

        Example:
            # Double all voxel values
            query.map(lambda v: v * 2).to_radi_object(uri)

            # Resample to uniform shape with MONAI
            from monai.transforms import Spacing
            spacing = Spacing(pixdim=(2.0, 2.0, 2.0))
            query.map(lambda v: spacing(v[np.newaxis, ...])[0]).to_radi_object(uri)
        """
        if self._transform_fn is not None:
            # Compose transforms: apply previous transform, then new one
            prev_fn = self._transform_fn

            def composed_fn(v: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
                return fn(prev_fn(v))

            return self._copy(transform_fn=composed_fn)
        return self._copy(transform_fn=fn)

    # =========================================================================
    # MASK RESOLUTION
    # =========================================================================

    def _resolve_subject_mask(self) -> frozenset[str]:
        """Resolve subject-level filters to a set of obs_subject_ids.

        This applies:
        1. Explicit subject_ids filter
        2. Query expression on obs_meta
        """
        all_ids = set(self._source.obs_subject_ids)

        # Apply explicit subject_ids filter
        if self._subject_ids is not None:
            all_ids &= self._subject_ids

        # Apply query expression on obs_meta
        if self._subject_query is not None:
            filtered = self._source.obs_meta.read(value_filter=self._subject_query)
            query_ids = set(filtered["obs_subject_id"])
            all_ids &= query_ids

        return frozenset(all_ids)

    def _resolve_volume_mask(
        self, collection_name: str, subject_mask: frozenset[str]
    ) -> frozenset[str]:
        """Resolve volume-level filters to a set of obs_ids for a collection.

        Uses TileDB dimension slicing for efficient subject filtering,
        combined with QueryCondition for attribute filters.
        """
        vc = self._source.collection(collection_name)
        effective_ctx = vc.obs._effective_ctx()

        # Build query with dimension slicing for subject_mask
        with tiledb.open(vc.obs.uri, "r", ctx=effective_ctx) as arr:
            # Use multi-index for efficient dimension-based filtering
            subject_list = list(subject_mask) if subject_mask else None

            if subject_list:
                # Query only rows matching subject_mask using dimension slicing
                query = arr.query(dims=["obs_subject_id", "obs_id"])
                if collection_name in self._collection_filters:
                    query = query.cond(self._collection_filters[collection_name])
                result = query.multi_index[subject_list, :]
            else:
                # No subject filter - apply attribute filter only
                if collection_name in self._collection_filters:
                    result = arr.query(cond=self._collection_filters[collection_name])[:]
                else:
                    result = arr.query(attrs=[])[:]["obs_id"]
                    return frozenset(v.decode() if isinstance(v, bytes) else str(v) for v in result)

            obs_ids = result["obs_id"]
            return frozenset(v.decode() if isinstance(v, bytes) else str(v) for v in obs_ids)

    def _resolve_final_subject_mask(self) -> frozenset[str]:
        """Resolve to final subject mask after applying all filters.

        Uses TileDB dimension queries to efficiently find subjects with matching volumes.
        """
        subject_mask = self._resolve_subject_mask()

        # If there are collection filters, further filter subjects
        # to only those with matching volumes
        if self._collection_filters:
            subjects_with_matching_volumes = set()

            for coll_name in self._collection_filters:
                if coll_name not in self._source.collection_names:
                    continue

                vc = self._source.collection(coll_name)
                effective_ctx = vc.obs._effective_ctx()
                expr = self._collection_filters[coll_name]

                # Query with attribute filter, only request obs_subject_id dimension
                with tiledb.open(vc.obs.uri, "r", ctx=effective_ctx) as arr:
                    result = arr.query(cond=expr, dims=["obs_subject_id"], attrs=[])[:]
                    subject_ids = result["obs_subject_id"]
                    for sid in subject_ids:
                        s = sid.decode() if isinstance(sid, bytes) else str(sid)
                        if s in subject_mask:
                            subjects_with_matching_volumes.add(s)

            subject_mask = subject_mask & frozenset(subjects_with_matching_volumes)

        return subject_mask

    def _resolve_output_collections(self) -> tuple[str, ...]:
        """Resolve which collections to include in output."""
        if self._output_collections is not None:
            return tuple(
                name for name in self._source.collection_names if name in self._output_collections
            )
        return self._source.collection_names

    # =========================================================================
    # MATERIALIZATION (triggers data access)
    # =========================================================================

    def count(self) -> QueryCount:
        """Count subjects and volumes matching the query without loading volume data."""
        subject_mask = self._resolve_final_subject_mask()
        output_collections = self._resolve_output_collections()

        volume_counts: dict[str, int] = {}
        for name in output_collections:
            volume_mask = self._resolve_volume_mask(name, subject_mask)
            volume_counts[name] = len(volume_mask)

        return QueryCount(n_subjects=len(subject_mask), n_volumes=volume_counts)

    def to_obs_meta(self) -> pd.DataFrame:
        """Return filtered obs_meta DataFrame."""
        subject_mask = self._resolve_final_subject_mask()
        obs_meta = self._source.obs_meta.read()
        return obs_meta[obs_meta["obs_subject_id"].isin(subject_mask)].reset_index(drop=True)

    def iter_volumes(self, collection_name: str | None = None) -> Iterator[Volume]:
        """Iterate over volumes matching the query.

        Args:
            collection_name: Specific collection to iterate. If None, iterates
                           over all output collections.
        """
        subject_mask = self._resolve_final_subject_mask()
        output_collections = self._resolve_output_collections()

        if collection_name is not None:
            if collection_name not in output_collections:
                raise ValueError(f"Collection '{collection_name}' not in query scope")
            output_collections = (collection_name,)

        for coll_name in output_collections:
            vc = self._source.collection(coll_name)
            volume_mask = self._resolve_volume_mask(coll_name, subject_mask)

            for obs_id in volume_mask:
                yield vc.loc[obs_id]

    def iter_batches(self, batch_size: int = 32) -> Iterator[VolumeBatch]:
        """Iterate over batched volumes for ML training.

        Yields VolumeBatch with stacked numpy arrays for each collection.
        Batches are grouped by subject.
        """
        subject_mask = self._resolve_final_subject_mask()
        subject_list = sorted(subject_mask)
        output_collections = self._resolve_output_collections()

        for i in range(0, len(subject_list), batch_size):
            batch_subjects = subject_list[i : i + batch_size]
            batch_subject_set = frozenset(batch_subjects)

            volumes: dict[str, npt.NDArray[np.floating]] = {}
            obs_ids: dict[str, tuple[str, ...]] = {}

            for coll_name in output_collections:
                volume_mask = self._resolve_volume_mask(coll_name, batch_subject_set)
                vc = self._source.collection(coll_name)

                batch_arrays = []
                batch_obs_ids = []
                for obs_id in sorted(volume_mask):
                    vol = vc.loc[obs_id]
                    batch_arrays.append(vol.to_numpy())
                    batch_obs_ids.append(obs_id)

                if batch_arrays:
                    volumes[coll_name] = np.stack(batch_arrays, axis=0)
                    obs_ids[coll_name] = tuple(batch_obs_ids)

            yield VolumeBatch(
                volumes=volumes,
                subject_ids=tuple(batch_subjects),
                obs_ids=obs_ids,
            )

    def to_radi_object(
        self,
        uri: str,
        streaming: bool = True,
        ctx: tiledb.Ctx | None = None,
    ) -> RadiObject:
        """Materialize query results as a new RadiObject.

        If a transform was set via map(), it is applied to each volume
        during materialization. If the transform changes volume shapes,
        the output collection becomes heterogeneous.

        Args:
            uri: Target URI for the new RadiObject
            streaming: Use streaming writer for memory efficiency (default: True)
            ctx: TileDB context
        """
        from radiobject.radi_object import RadiObject, RadiObjectView
        from radiobject.streaming import RadiObjectWriter

        subject_mask = self._resolve_final_subject_mask()
        output_collections = self._resolve_output_collections()

        # If no transform, delegate to RadiObjectView for efficiency
        if self._transform_fn is None:
            view = RadiObjectView(
                source=self._source,
                obs_subject_ids=list(subject_mask),
                collection_names=list(output_collections),
            )
            if streaming:
                return view.to_radi_object_streaming(uri, ctx=ctx)
            return view.to_radi_object(uri, ctx=ctx)

        # With transform: use streaming writer and apply transform to each volume
        # Build filtered obs_meta
        obs_meta_df = self._source.obs_meta.read()
        filtered_obs_meta = obs_meta_df[
            obs_meta_df["obs_subject_id"].isin(subject_mask)
        ].reset_index(drop=True)

        obs_meta_schema: dict[str, np.dtype] = {}
        for col in filtered_obs_meta.columns:
            if col in ("obs_subject_id", "obs_id"):
                continue
            dtype = filtered_obs_meta[col].to_numpy().dtype
            if dtype == np.dtype("O"):
                dtype = np.dtype("U64")
            obs_meta_schema[col] = dtype

        with RadiObjectWriter(uri, obs_meta_schema=obs_meta_schema, ctx=ctx) as writer:
            writer.write_obs_meta(filtered_obs_meta)

            for coll_name in output_collections:
                src_collection = self._source.collection(coll_name)
                volume_mask = self._resolve_volume_mask(coll_name, subject_mask)

                if not volume_mask:
                    continue

                # Get obs data for filtered volumes
                obs_df = src_collection.obs.read()
                filtered_obs = obs_df[obs_df["obs_id"].isin(volume_mask)]

                # Extract obs schema
                obs_schema: dict[str, np.dtype] = {}
                for col in src_collection.obs.columns:
                    if col in ("obs_id", "obs_subject_id"):
                        continue
                    obs_schema[col] = src_collection.obs.dtypes[col]

                # Transform may change shape, so output is heterogeneous (shape=None)
                with writer.add_collection(
                    coll_name, shape=None, obs_schema=obs_schema
                ) as coll_writer:
                    for _, row in filtered_obs.iterrows():
                        obs_id = row["obs_id"]
                        vol = src_collection.loc[obs_id]
                        data = vol.to_numpy()

                        # Apply transform
                        data = self._transform_fn(data)

                        attrs = {
                            k: v for k, v in row.items() if k not in ("obs_id", "obs_subject_id")
                        }
                        coll_writer.write_volume(
                            data=data,
                            obs_id=obs_id,
                            obs_subject_id=row["obs_subject_id"],
                            **attrs,
                        )

        return RadiObject(uri, ctx=ctx)

    def __len__(self) -> int:
        """Number of subjects matching the query."""
        return len(self._resolve_final_subject_mask())

    def __repr__(self) -> str:
        """Concise representation of the Query."""
        count = self.count()
        collections = ", ".join(count.n_volumes.keys())
        return (
            f"Query({count.n_subjects} subjects, "
            f"{sum(count.n_volumes.values())} volumes across [{collections}])"
        )


class CollectionQuery:
    """Lazy filter builder for VolumeCollection with explicit materialization.

    CollectionQuery operates on a single VolumeCollection's obs dataframe,
    returning masks that can be used to filter volumes.

    Example:
        high_res = (
            radi.T1w.query()
            .filter("voxel_spacing == '1.0x1.0x1.0'")
            .head(100)
        )
        for vol in high_res.iter_volumes():
            process(vol)
    """

    def __init__(
        self,
        source: VolumeCollection,
        *,
        volume_ids: frozenset[str] | None = None,
        volume_query: str | None = None,
        subject_ids: frozenset[str] | None = None,
        transform_fn: TransformFn | None = None,
    ):
        self._source = source
        self._volume_ids = volume_ids
        self._volume_query = volume_query
        self._subject_ids = subject_ids
        self._transform_fn = transform_fn

    def _copy(self, **kwargs) -> CollectionQuery:
        """Create a copy with modified fields."""
        return CollectionQuery(
            self._source,
            volume_ids=kwargs.get("volume_ids", self._volume_ids),
            volume_query=kwargs.get("volume_query", self._volume_query),
            subject_ids=kwargs.get("subject_ids", self._subject_ids),
            transform_fn=kwargs.get("transform_fn", self._transform_fn),
        )

    # =========================================================================
    # FILTER METHODS
    # =========================================================================

    def filter(self, expr: str) -> CollectionQuery:
        """Filter volumes using TileDB QueryCondition on obs."""
        new_query = self._volume_query
        if new_query is None:
            new_query = expr
        else:
            new_query = f"({new_query}) and ({expr})"
        return self._copy(volume_query=new_query)

    def filter_subjects(self, ids: Sequence[str]) -> CollectionQuery:
        """Filter to volumes belonging to specific subject IDs."""
        new_ids = frozenset(ids)
        if self._subject_ids is not None:
            new_ids = self._subject_ids & new_ids
        return self._copy(subject_ids=new_ids)

    def iloc(self, key: int | slice | list[int] | npt.NDArray[np.bool_]) -> CollectionQuery:
        """Filter volumes by integer position(s)."""
        n = len(self._source)
        if isinstance(key, int):
            idx = key if key >= 0 else n + key
            obs_id = self._source._index.get_key(idx)
            return self._copy(volume_ids=frozenset([obs_id]))
        elif isinstance(key, slice):
            indices = list(range(*key.indices(n)))
        elif isinstance(key, np.ndarray) and key.dtype == np.bool_:
            indices = list(np.where(key)[0])
        elif isinstance(key, list):
            indices = [i if i >= 0 else n + i for i in key]
        else:
            raise TypeError("iloc key must be int, slice, list[int], or bool array")

        obs_ids = frozenset(self._source._index.get_key(i) for i in indices)
        new_ids = obs_ids
        if self._volume_ids is not None:
            new_ids = self._volume_ids & obs_ids
        return self._copy(volume_ids=new_ids)

    def loc(self, key: str | Sequence[str]) -> CollectionQuery:
        """Filter volumes by obs_id(s)."""
        if isinstance(key, str):
            ids = frozenset([key])
        else:
            ids = frozenset(key)
        new_ids = ids
        if self._volume_ids is not None:
            new_ids = self._volume_ids & ids
        return self._copy(volume_ids=new_ids)

    def head(self, n: int = 5) -> CollectionQuery:
        """Filter to first n volumes."""
        return self.iloc(slice(0, n))

    def tail(self, n: int = 5) -> CollectionQuery:
        """Filter to last n volumes."""
        total = len(self._source)
        return self.iloc(slice(max(0, total - n), total))

    def sample(self, n: int = 5, seed: int | None = None) -> CollectionQuery:
        """Filter to n randomly sampled volumes."""
        rng = np.random.default_rng(seed)
        resolved = list(self._resolve_volume_mask())
        n = min(n, len(resolved))
        sampled = rng.choice(resolved, size=n, replace=False)
        return self._copy(volume_ids=frozenset(sampled))

    # =========================================================================
    # TRANSFORM (applied during materialization)
    # =========================================================================

    def map(self, fn: TransformFn) -> CollectionQuery:
        """Apply transform to each volume during materialization.

        The transform function receives a numpy array and returns a numpy array.
        Transform is applied lazily when volumes are materialized via
        to_volume_collection().

        Multiple map() calls compose: query.map(f1).map(f2) applies f1 then f2.

        Args:
            fn: Function taking numpy array (X, Y, Z) and returning numpy array.
                Can change shape (e.g., resampling to different dimensions).

        Returns:
            New CollectionQuery with transform applied

        Example:
            # Double all voxel values
            query.map(lambda v: v * 2).to_volume_collection(uri)

            # Resample to uniform shape
            def resample(v):
                from scipy.ndimage import zoom
                target = (128, 128, 128)
                factors = tuple(t / s for t, s in zip(target, v.shape))
                return zoom(v, factors, order=1)
            query.map(resample).to_volume_collection(uri)
        """
        if self._transform_fn is not None:
            # Compose transforms: apply previous transform, then new one
            prev_fn = self._transform_fn

            def composed_fn(v: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
                return fn(prev_fn(v))

            return self._copy(transform_fn=composed_fn)
        return self._copy(transform_fn=fn)

    # =========================================================================
    # MASK RESOLUTION
    # =========================================================================

    def _resolve_volume_mask(self) -> frozenset[str]:
        """Resolve all filters to a set of obs_ids using TileDB-native queries."""
        effective_ctx = self._source.obs._effective_ctx()

        with tiledb.open(self._source.obs.uri, "r", ctx=effective_ctx) as arr:
            # Build query based on filters
            if self._subject_ids is not None and self._volume_query is not None:
                # Both subject and attribute filters - use dimension slicing + QueryCondition
                query = arr.query(cond=self._volume_query, dims=["obs_id"])
                result = query.multi_index[list(self._subject_ids), :]
            elif self._subject_ids is not None:
                # Subject filter only - use dimension slicing
                result = arr.query(dims=["obs_id"]).multi_index[list(self._subject_ids), :]
            elif self._volume_query is not None:
                # Attribute filter only - use QueryCondition
                result = arr.query(cond=self._volume_query, dims=["obs_id"])[:]
            else:
                # No filters - return all obs_ids
                result = arr.query(dims=["obs_id"], attrs=[])[:]

            obs_ids = result["obs_id"]
            all_ids = frozenset(v.decode() if isinstance(v, bytes) else str(v) for v in obs_ids)

        # Apply explicit volume_ids filter (intersection with pre-specified IDs)
        if self._volume_ids is not None:
            all_ids &= self._volume_ids

        return all_ids

    # =========================================================================
    # MATERIALIZATION
    # =========================================================================

    def count(self) -> int:
        """Count volumes matching the query."""
        return len(self._resolve_volume_mask())

    def to_obs(self) -> pd.DataFrame:
        """Return filtered obs DataFrame."""
        volume_mask = self._resolve_volume_mask()
        obs_df = self._source.obs.read()
        return obs_df[obs_df["obs_id"].isin(volume_mask)].reset_index(drop=True)

    def iter_volumes(self) -> Iterator[Volume]:
        """Iterate over volumes matching the query."""
        volume_mask = self._resolve_volume_mask()
        for obs_id in sorted(volume_mask):
            yield self._source.loc[obs_id]

    def to_numpy_stack(self) -> npt.NDArray[np.floating]:
        """Load all matching volumes as stacked numpy array (N, X, Y, Z)."""
        volume_mask = self._resolve_volume_mask()
        if not volume_mask:
            raise ValueError("No volumes match the query")

        arrays = [self._source.loc[obs_id].to_numpy() for obs_id in sorted(volume_mask)]
        return np.stack(arrays, axis=0)

    def to_volume_collection(
        self,
        uri: str,
        name: str | None = None,
        ctx: tiledb.Ctx | None = None,
    ) -> VolumeCollection:
        """Materialize query results as a new VolumeCollection.

        If a transform was set via map(), it is applied to each volume
        during materialization. If the transform changes volume shapes,
        the output collection becomes heterogeneous (shape=None).
        """
        from radiobject.streaming import StreamingWriter

        volume_mask = self._resolve_volume_mask()
        if not volume_mask:
            raise ValueError("No volumes match the query")

        obs_df = self.to_obs()
        collection_name = name or self._source.name

        # Extract obs schema
        obs_schema: dict[str, np.dtype] = {}
        for col in self._source.obs.columns:
            if col in ("obs_id", "obs_subject_id"):
                continue
            obs_schema[col] = self._source.obs.dtypes[col]

        # If transform is set, output is heterogeneous (shape may change)
        output_shape = None if self._transform_fn is not None else self._source.shape

        with StreamingWriter(
            uri=uri,
            shape=output_shape,
            obs_schema=obs_schema,
            name=collection_name,
            ctx=ctx,
        ) as writer:
            for obs_id in sorted(volume_mask):
                vol = self._source.loc[obs_id]
                data = vol.to_numpy()

                # Apply transform if set
                if self._transform_fn is not None:
                    data = self._transform_fn(data)

                obs_row = obs_df[obs_df["obs_id"] == obs_id].iloc[0]
                attrs = {k: v for k, v in obs_row.items() if k not in ("obs_id", "obs_subject_id")}
                writer.write_volume(
                    data=data,
                    obs_id=obs_id,
                    obs_subject_id=obs_row["obs_subject_id"],
                    **attrs,
                )

        return self._source.__class__(uri, ctx=ctx)

    def __len__(self) -> int:
        """Number of volumes matching the query."""
        return len(self._resolve_volume_mask())

    def __repr__(self) -> str:
        """Concise representation of the CollectionQuery."""
        n = len(self)
        name = self._source.name or "unnamed"
        shape = "x".join(str(d) for d in self._source.shape)
        return f"CollectionQuery('{name}', {n} volumes, shape={shape})"
