"""Eager and lazy query builders for VolumeCollection pipelines.

Classes:
    EagerQuery  — returned by VolumeCollection.map(). Holds computed results
                  with chaining (.map), persistence (.write), and extraction (.to_list).
    LazyQuery   — returned by VolumeCollection.lazy(). Defers filter + transform
                  until .write() for single-pass, memory-efficient ETL.

Transform signature:
    All transforms receive (volume: np.ndarray, obs_row: pd.Series) and return
    either a transformed volume, or (volume, obs_updates_dict) to annotate
    obs metadata during the pipeline.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Generic, Iterator, Sequence, TypeVar

import numpy as np
import numpy.typing as npt
import pandas as pd
import tiledb

from radiobject._types import AttrValue, TransformFn, normalize_transform_result
from radiobject.volume import Volume

if TYPE_CHECKING:
    from radiobject.volume_collection import VolumeCollection


def _compose_transforms(prev_fn: TransformFn | None, new_fn: TransformFn) -> TransformFn:
    """Compose two transform functions, applying prev_fn then new_fn.

    Propagates obs_updates through the chain: prev_fn's updates are merged
    into obs_row before passing to new_fn, and all updates are combined.
    """
    if prev_fn is None:
        return new_fn

    def composed_fn(
        volume: npt.NDArray[np.floating], obs_row: pd.Series
    ) -> npt.NDArray[np.floating] | tuple[npt.NDArray[np.floating], dict]:
        prev_result = prev_fn(volume, obs_row)
        prev_vol, prev_updates = normalize_transform_result(prev_result)
        merged_obs = obs_row.copy()
        for k, v in prev_updates.items():
            merged_obs[k] = v
        next_result = new_fn(prev_vol, merged_obs)
        next_vol, next_updates = normalize_transform_result(next_result)
        combined = {**prev_updates, **next_updates}
        if combined:
            return next_vol, combined
        return next_vol

    return composed_fn


def _infer_attr_dtype(value: AttrValue) -> np.dtype:
    """Infer TileDB-compatible numpy dtype from a Python scalar."""
    if isinstance(value, bool):
        return np.dtype("uint8")
    if isinstance(value, int):
        return np.dtype("int64")
    if isinstance(value, float):
        return np.dtype("float64")
    return np.dtype("U256")


_OBS_ID_COLS = frozenset({"obs_id", "obs_subject_id"})


def _build_obs_schema(obs_accessor: object) -> dict[str, np.dtype]:
    """Build obs_schema from a source's obs columns, excluding identity columns."""
    return {
        col: obs_accessor.dtypes[col] for col in obs_accessor.columns if col not in _OBS_ID_COLS
    }


def _extend_obs_schema(
    schema: dict[str, np.dtype],
    updates_iter: Iterator[dict[str, AttrValue]],
) -> None:
    """Extend obs_schema in-place with new columns discovered from obs_updates."""
    for updates in updates_iter:
        for key, val in updates.items():
            if key not in schema and key not in _OBS_ID_COLS:
                schema[key] = _infer_attr_dtype(val)


def _extract_obs_attrs(obs_row: pd.Series) -> dict[str, AttrValue]:
    """Extract non-identity attributes from an obs row."""
    return {k: v for k, v in obs_row.items() if k not in _OBS_ID_COLS}


def _apply_transform_to_volumes(
    transform_fn: TransformFn,
    collection: VolumeCollection,
    filtered_obs: pd.DataFrame,
) -> list[tuple[str, str, npt.NDArray, dict[str, AttrValue], dict[str, AttrValue]]]:
    """Apply a transform function to each volume and collect results with obs_updates.

    Returns list of (obs_id, subject_id, data, base_attrs, obs_updates) tuples.
    """
    results: list[tuple[str, str, npt.NDArray, dict, dict]] = []
    for _, row in filtered_obs.iterrows():
        obs_id = row["obs_id"]
        data = collection.loc[obs_id].to_numpy()

        raw_result = transform_fn(data, row)
        data, obs_updates = normalize_transform_result(raw_result)

        attrs = _extract_obs_attrs(row)
        results.append((obs_id, row["obs_subject_id"], data, attrs, obs_updates))
    return results


T = TypeVar("T")


class EagerQuery(Generic[T]):
    """Computed query results with pipeline methods.

    Holds materialized results paired with obs metadata and accumulated obs_updates,
    enabling chained transforms via `.map()` and persistence via `.write()`.

    Transforms can return either just a value, or (value, obs_updates_dict) to
    annotate obs metadata. Updates accumulate across chained `.map()` calls and
    are merged into obs rows on `.write()`.
    """

    def __init__(
        self,
        results: list[T],
        obs_df: pd.DataFrame,
        source: VolumeCollection,
        obs_updates: list[dict[str, AttrValue]] | None = None,
    ):
        self._results = results
        self._obs_df = obs_df
        self._source = source
        self._obs_updates = obs_updates or [{} for _ in results]

    def map(self, fn: Callable) -> EagerQuery:
        """Chain another transform, applying fn(result, obs_row) immediately.

        If fn returns (value, obs_updates_dict), the updates are accumulated
        and merged into the obs_row for subsequent transforms and on `.write()`.
        """
        new_results = []
        new_updates = []
        for result, (_, obs_row), prev_updates in zip(
            self._results, self._obs_df.iterrows(), self._obs_updates
        ):
            merged_obs = obs_row.copy()
            for k, v in prev_updates.items():
                merged_obs[k] = v
            raw_result = fn(result, merged_obs)
            value, updates = normalize_transform_result(raw_result)
            new_results.append(value)
            new_updates.append({**prev_updates, **updates})
        return EagerQuery(new_results, self._obs_df, self._source, new_updates)

    def write(
        self,
        uri: str,
        name: str | None = None,
        ctx: tiledb.Ctx | None = None,
    ) -> VolumeCollection:
        """Persist results to a new VolumeCollection (results must be np.ndarray)."""
        from radiobject.volume_collection import VolumeCollection
        from radiobject.writers import VolumeCollectionWriter

        collection_name = name or self._source.name

        obs_schema = _build_obs_schema(self._source.obs)
        _extend_obs_schema(obs_schema, iter(self._obs_updates))

        with VolumeCollectionWriter(
            uri=uri,
            shape=None,
            obs_schema=obs_schema,
            name=collection_name,
            ctx=ctx,
        ) as writer:
            for result, (_, obs_row), updates in zip(
                self._results, self._obs_df.iterrows(), self._obs_updates
            ):
                attrs = _extract_obs_attrs(obs_row)
                attrs.update(updates)
                writer.write_volume(
                    data=result,
                    obs_id=obs_row["obs_id"],
                    obs_subject_id=obs_row["obs_subject_id"],
                    **attrs,
                )

        return VolumeCollection(uri, ctx=ctx)

    def to_list(self) -> list[T]:
        """Extract raw results."""
        return list(self._results)

    def __iter__(self) -> Iterator[T]:
        return iter(self._results)

    def __len__(self) -> int:
        return len(self._results)

    def __getitem__(self, idx: int) -> T:
        return self._results[idx]

    def __repr__(self) -> str:
        return f"EagerQuery({len(self._results)} results)"


class LazyQuery:
    """Lazy filter builder for VolumeCollection with deferred transforms.

    Use `.lazy()` to enter lazy mode, then chain `.filter()`, `.head()`, `.map()`,
    and `.write()` to build and execute a transform pipeline.

    Example:
        Filter and write high-resolution volumes:

            high_res = (
                radi.T1w.lazy()
                .filter("voxel_spacing == '1.0x1.0x1.0'")
                .head(100)
                .map(normalize)
                .write("./output")
            )
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

    def __len__(self) -> int:
        """Number of volumes matching the query."""
        return len(self._resolve_volume_mask())

    def __repr__(self) -> str:
        n = len(self)
        name = self._source.name or "unnamed"
        shape = self._source.shape
        shape_str = "x".join(str(d) for d in shape) if shape else "heterogeneous"
        return f"LazyQuery('{name}', {n} volumes, shape={shape_str})"

    def filter(self, expr: str) -> LazyQuery:
        """Filter volumes using TileDB QueryCondition on obs."""
        new_query = self._volume_query
        if new_query is None:
            new_query = expr
        else:
            new_query = f"({new_query}) and ({expr})"
        return self._copy(volume_query=new_query)

    def filter_subjects(self, ids: Sequence[str]) -> LazyQuery:
        """Filter to volumes belonging to specific subject IDs."""
        new_ids = frozenset(ids)
        if self._subject_ids is not None:
            new_ids = self._subject_ids & new_ids
        return self._copy(subject_ids=new_ids)

    def iloc(self, key: int | slice | list[int] | npt.NDArray[np.bool_]) -> LazyQuery:
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

        obs_ids = self._source._index.take(indices).to_set()
        new_ids = obs_ids
        if self._volume_ids is not None:
            new_ids = self._volume_ids & obs_ids
        return self._copy(volume_ids=new_ids)

    def loc(self, key: str | Sequence[str]) -> LazyQuery:
        """Filter volumes by obs_id(s)."""
        if isinstance(key, str):
            ids = frozenset([key])
        else:
            ids = frozenset(key)
        new_ids = ids
        if self._volume_ids is not None:
            new_ids = self._volume_ids & ids
        return self._copy(volume_ids=new_ids)

    def head(self, n: int = 5) -> LazyQuery:
        """Filter to first n volumes."""
        return self.iloc(slice(0, n))

    def tail(self, n: int = 5) -> LazyQuery:
        """Filter to last n volumes."""
        total = len(self._source)
        return self.iloc(slice(max(0, total - n), total))

    def sample(self, n: int = 5, seed: int | None = None) -> LazyQuery:
        """Filter to n randomly sampled volumes."""
        rng = np.random.default_rng(seed)
        resolved = list(self._resolve_volume_mask())
        n = min(n, len(resolved))
        sampled = rng.choice(resolved, size=n, replace=False)
        return self._copy(volume_ids=frozenset(sampled))

    def map(self, fn: TransformFn) -> LazyQuery:
        """Apply transform to each volume during write.

        Multiple map() calls compose: query.map(f1).map(f2) applies f1 then f2.
        Transform receives (volume, obs_row) and can return volume or (volume, obs_updates).
        """
        return self._copy(transform_fn=_compose_transforms(self._transform_fn, fn))

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

    def write(
        self,
        uri: str | None = None,
        name: str | None = None,
        ctx: tiledb.Ctx | None = None,
    ) -> VolumeCollection:
        """Write query results as a new VolumeCollection.

        Applies queued transforms and persists in a single pass.
        Transforms receive (volume, obs_row) and can return volume or (volume, obs_updates).
        """
        from radiobject.volume_collection import _generate_adjacent_uri
        from radiobject.writers import VolumeCollectionWriter

        if uri is None:
            uri = _generate_adjacent_uri(
                self._source.uri, name=name, transform_fn=self._transform_fn
            )

        volume_mask = self._resolve_volume_mask()
        if not volume_mask:
            raise ValueError("No volumes match the query")

        obs_df = self.to_obs()
        collection_name = name or self._source.name
        obs_schema = _build_obs_schema(self._source.obs)

        if self._transform_fn is None:
            with VolumeCollectionWriter(
                uri=uri,
                shape=self._source.shape,
                obs_schema=obs_schema,
                name=collection_name,
                ctx=ctx,
            ) as writer:
                for obs_id in sorted(volume_mask):
                    data = self._source.loc[obs_id].to_numpy()
                    obs_row = obs_df[obs_df["obs_id"] == obs_id].iloc[0]
                    attrs = _extract_obs_attrs(obs_row)
                    writer.write_volume(
                        data=data,
                        obs_id=obs_id,
                        obs_subject_id=obs_row["obs_subject_id"],
                        **attrs,
                    )
        else:
            filtered_obs = obs_df.set_index("obs_id").loc[sorted(volume_mask)].reset_index()
            transform_results = _apply_transform_to_volumes(
                self._transform_fn, self._source, filtered_obs
            )
            _extend_obs_schema(obs_schema, (updates for *_, updates in transform_results))

            with VolumeCollectionWriter(
                uri=uri,
                shape=None,
                obs_schema=obs_schema,
                name=collection_name,
                ctx=ctx,
            ) as writer:
                for obs_id, subject_id, data, attrs, obs_updates in transform_results:
                    attrs.update(obs_updates)
                    writer.write_volume(
                        data=data,
                        obs_id=obs_id,
                        obs_subject_id=subject_id,
                        **attrs,
                    )

        return self._source.__class__(uri, ctx=ctx)

    def _copy(self, **kwargs) -> LazyQuery:
        """Create a copy with modified fields."""
        return LazyQuery(
            self._source,
            volume_ids=kwargs.get("volume_ids", self._volume_ids),
            volume_query=kwargs.get("volume_query", self._volume_query),
            subject_ids=kwargs.get("subject_ids", self._subject_ids),
            transform_fn=kwargs.get("transform_fn", self._transform_fn),
        )

    def _resolve_volume_mask(self) -> frozenset[str]:
        """Resolve all filters to a set of obs_ids using TileDB-native queries."""
        effective_ctx = self._source.obs._effective_ctx()

        with tiledb.open(self._source.obs.uri, "r", ctx=effective_ctx) as arr:
            if self._subject_ids is not None and self._volume_query is not None:
                query = arr.query(cond=self._volume_query, dims=["obs_id"])
                result = query.multi_index[list(self._subject_ids), :]
            elif self._subject_ids is not None:
                result = arr.query(dims=["obs_id"]).multi_index[list(self._subject_ids), :]
            elif self._volume_query is not None:
                result = arr.query(cond=self._volume_query, dims=["obs_id"])[:]
            else:
                result = arr.query(dims=["obs_id"], attrs=[])[:]

            obs_ids = result["obs_id"]
            all_ids = frozenset(v.decode() if isinstance(v, bytes) else str(v) for v in obs_ids)

        if self._volume_ids is not None:
            all_ids &= self._volume_ids

        return all_ids
