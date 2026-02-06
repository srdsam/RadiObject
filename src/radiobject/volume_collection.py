"""VolumeCollection - organizes volumes with consistent dimensions indexed by obs_id."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterator
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence, overload

import numpy as np
import numpy.typing as npt
import pandas as pd
import tiledb

from radiobject._types import TransformFn
from radiobject.ctx import get_tiledb_ctx
from radiobject.dataframe import Dataframe
from radiobject.imaging_metadata import (
    DicomMetadata,
    NiftiMetadata,
    extract_dicom_metadata,
    extract_nifti_metadata,
    infer_series_type,
)
from radiobject.indexing import Index
from radiobject.parallel import WriteResult, ctx_for_threads, map_on_threads
from radiobject.volume import Volume

if TYPE_CHECKING:
    from radiobject.ml.datasets.collection_dataset import VolumeCollectionDataset
    from radiobject.query import CollectionQuery


def _normalize_index(idx: int, length: int) -> int:
    """Convert negative index to positive and validate bounds."""
    if idx < 0:
        idx = length + idx
    if idx < 0 or idx >= length:
        raise IndexError(f"Index {idx} out of range [0, {length})")
    return idx


def _get_volume_by_obs_id(collection: VolumeCollection, obs_id: str) -> Volume:
    """Construct Volume on-demand by obs_id (shared by iloc and loc indexers)."""
    root = collection._root
    idx = root._index.get_index(obs_id)
    return Volume(f"{root.uri}/volumes/{idx}", ctx=root._ctx)


def generate_obs_id(obs_subject_id: str, series_type: str) -> str:
    """Generate a unique obs_id from subject ID and series type."""
    return f"{obs_subject_id}_{series_type}"


def _write_volumes_parallel(
    write_fn,
    write_args: list,
    progress: bool,
    desc: str,
) -> list[WriteResult]:
    """Common helper for parallel volume writes with error handling."""
    results = map_on_threads(write_fn, write_args, progress=progress, desc=desc)
    failures = [r for r in results if not r.success]
    if failures:
        raise RuntimeError(f"Volume write failed: {failures[0].error}")
    return results


def _generate_adjacent_uri(
    source_uri: str,
    name: str | None = None,
    transform_fn: TransformFn | None = None,
) -> str:
    """Generate a URI adjacent to the source for materialization.

    Naming priority: explicit name > transform __name__ > '_materialized' suffix.
    """
    parent = source_uri.rsplit("/", 1)[0]

    if name:
        return f"{parent}/{name}"

    source_name = source_uri.rsplit("/", 1)[1]

    if transform_fn is not None:
        fn_name = getattr(transform_fn, "__name__", None)
        if fn_name and fn_name != "<lambda>":
            return f"{parent}/{source_name}_{fn_name}"

    return f"{parent}/{source_name}_materialized"


class _ILocIndexer:
    """Integer-location based indexer for VolumeCollection (like pandas .iloc)."""

    def __init__(self, collection: VolumeCollection):
        self._collection = collection

    @overload
    def __getitem__(self, key: int) -> Volume: ...
    @overload
    def __getitem__(self, key: slice) -> VolumeCollection: ...
    @overload
    def __getitem__(self, key: list[int]) -> VolumeCollection: ...
    @overload
    def __getitem__(self, key: npt.NDArray[np.bool_]) -> VolumeCollection: ...

    def __getitem__(
        self, key: int | slice | list[int] | npt.NDArray[np.bool_]
    ) -> Volume | VolumeCollection:
        """Index by int, slice, list of ints, or boolean mask."""
        obs_ids = self._collection._effective_obs_ids
        n = len(obs_ids)
        if isinstance(key, int):
            idx = _normalize_index(key, n)
            return _get_volume_by_obs_id(self._collection, obs_ids[idx])

        elif isinstance(key, slice):
            indices = list(range(*key.indices(n)))
            selected_ids = frozenset(obs_ids[i] for i in indices)
            return self._collection._create_view(volume_ids=selected_ids)

        elif isinstance(key, np.ndarray) and key.dtype == np.bool_:
            if len(key) != n:
                raise ValueError(f"Boolean mask length {len(key)} != volume count {n}")
            indices = np.where(key)[0]
            selected_ids = frozenset(obs_ids[int(i)] for i in indices)
            return self._collection._create_view(volume_ids=selected_ids)

        elif isinstance(key, list):
            selected_ids = frozenset(obs_ids[_normalize_index(i, n)] for i in key)
            return self._collection._create_view(volume_ids=selected_ids)

        raise TypeError(
            f"iloc indices must be int, slice, list[int], or boolean array, got {type(key)}"
        )


class _LocIndexer:
    """Label-based indexer for VolumeCollection (like pandas .loc)."""

    def __init__(self, collection: VolumeCollection):
        self._collection = collection

    @overload
    def __getitem__(self, key: str) -> Volume: ...
    @overload
    def __getitem__(self, key: list[str]) -> VolumeCollection: ...

    def __getitem__(self, key: str | list[str]) -> Volume | VolumeCollection:
        """Index by obs_id string or list of obs_id strings."""
        if isinstance(key, str):
            # Validate the key is in the effective set
            if self._collection.is_view and key not in self._collection._volume_ids:
                raise KeyError(f"obs_id '{key}' not in view")
            return _get_volume_by_obs_id(self._collection, key)

        elif isinstance(key, list):
            selected_ids = frozenset(key)
            # Validate all keys are in the effective set
            if self._collection.is_view:
                invalid = selected_ids - self._collection._volume_ids
                if invalid:
                    raise KeyError(f"obs_ids not in view: {sorted(invalid)[:5]}")
            return self._collection._create_view(volume_ids=selected_ids)

        raise TypeError(f"loc indices must be str or list[str], got {type(key)}")


class VolumeCollection:
    """TileDB-backed volume collection indexed by obs_id. Supports uniform or heterogeneous shapes."""

    def __init__(
        self,
        uri: str | None,
        ctx: tiledb.Ctx | None = None,
        *,
        _source: VolumeCollection | None = None,
        _volume_ids: frozenset[str] | None = None,
    ):
        self._uri: str | None = uri
        self._ctx: tiledb.Ctx | None = ctx
        self._source: VolumeCollection | None = _source
        self._volume_ids: frozenset[str] | None = _volume_ids

    def __len__(self) -> int:
        """Number of volumes in collection (respects view filter)."""
        if self._volume_ids is not None:
            return len(self._volume_ids)
        return int(self._metadata["n_volumes"])

    def __iter__(self):
        """Iterate over volumes in index order (respects view filter)."""
        for obs_id in self._effective_obs_ids:
            yield self.loc[obs_id]

    def __repr__(self) -> str:
        """Concise representation of the VolumeCollection."""
        shape = self.shape
        shape_str = "x".join(str(d) for d in shape) if shape else "heterogeneous"
        name_part = f"'{self.name}', " if self.name else ""
        view_part = ", view" if self.is_view else ""
        return f"VolumeCollection({name_part}{len(self)} volumes, shape={shape_str}{view_part})"

    @overload
    def __getitem__(self, key: int) -> Volume: ...
    @overload
    def __getitem__(self, key: str) -> Volume: ...
    @overload
    def __getitem__(self, key: slice) -> VolumeCollection: ...
    @overload
    def __getitem__(self, key: list[int]) -> VolumeCollection: ...
    @overload
    def __getitem__(self, key: list[str]) -> VolumeCollection: ...

    def __getitem__(
        self, key: int | str | slice | list[int] | list[str]
    ) -> Volume | VolumeCollection:
        """Index by int, str, slice, or list. Slices/lists return views."""
        if isinstance(key, int):
            return self.iloc[key]
        elif isinstance(key, str):
            return self.loc[key]
        elif isinstance(key, slice):
            return self.iloc[key]
        elif isinstance(key, list):
            if len(key) == 0:
                return self._create_view(volume_ids=frozenset())
            if isinstance(key[0], int):
                return self.iloc[key]
            elif isinstance(key[0], str):
                return self.loc[key]
        raise TypeError(f"Key must be int, str, slice, or list, got {type(key)}")

    @property
    def uri(self) -> str:
        """URI of the underlying storage (raises if view without storage)."""
        if self._uri is not None:
            return self._uri
        if self._source is not None:
            return self._source.uri
        raise ValueError("VolumeCollection view has no URI. Call materialize(uri) first.")

    @property
    def is_view(self) -> bool:
        """True if this VolumeCollection is a filtered view of another."""
        return self._source is not None

    @property
    def shape(self) -> tuple[int, int, int] | None:
        """Volume dimensions (X, Y, Z) if uniform, None if heterogeneous."""
        m = self._metadata
        if "x_dim" not in m or "y_dim" not in m or "z_dim" not in m:
            return None
        return (int(m["x_dim"]), int(m["y_dim"]), int(m["z_dim"]))

    @property
    def is_uniform(self) -> bool:
        """Whether all volumes in this collection have the same shape."""
        return self.shape is not None

    @property
    def name(self) -> str | None:
        """Collection name (if set during creation)."""
        return self._metadata.get("name")

    @property
    def obs(self) -> Dataframe:
        """Observational metadata per volume."""
        obs_uri = f"{self._root.uri}/obs"
        return Dataframe(uri=obs_uri, ctx=self._ctx)

    @property
    def obs_ids(self) -> list[str]:
        """All obs_id values in index order (respects view filter)."""
        return self._effective_obs_ids

    @property
    def obs_subject_ids(self) -> list[str]:
        """Get obs_subject_id values for this collection (respects view filter)."""
        obs_df = self.to_obs()
        id_to_subject = dict(zip(obs_df["obs_id"], obs_df["obs_subject_id"]))
        return [id_to_subject[obs_id] for obs_id in self._effective_obs_ids]

    @property
    def index(self) -> Index:
        """Volume index for bidirectional ID/position lookups."""
        return self._index

    @cached_property
    def iloc(self) -> _ILocIndexer:
        """Integer-location based indexing for selecting volumes by position."""
        return _ILocIndexer(self)

    @cached_property
    def loc(self) -> _LocIndexer:
        """Label-based indexing for selecting volumes by obs_id."""
        return _LocIndexer(self)

    @cached_property
    def subjects(self) -> Index:
        """Subject-level index (obs_subject_id) for this collection."""
        unique = list(dict.fromkeys(self.obs_subject_ids))
        return Index.build(unique, name="obs_subject_id")

    def sel(self, *, subject: str | list[str]) -> Volume | VolumeCollection:
        """Select volumes by obs_subject_id.

        Args:
            subject: Single subject ID (returns Volume if exactly one match)
                or list of subject IDs (returns VolumeCollection view).
        """
        obs_df = self.to_obs()
        subjects = [subject] if isinstance(subject, str) else subject
        matching = obs_df[obs_df["obs_subject_id"].isin(subjects)]["obs_id"].tolist()

        if not matching:
            raise KeyError(f"obs_subject_id '{subject}' not found in collection")
        if isinstance(subject, str) and len(matching) == 1:
            return self.loc[matching[0]]
        return self._create_view(volume_ids=frozenset(matching))

    def groupby_subject(self) -> Iterator[tuple[str, VolumeCollection]]:
        """Group volumes by obs_subject_id. Yields (subject_id, view) pairs."""
        obs_df = self.to_obs()
        groups: dict[str, list[str]] = defaultdict(list)
        for _, row in obs_df.iterrows():
            groups[row["obs_subject_id"]].append(row["obs_id"])

        for subject_id in self.subjects:
            if subject_id in groups:
                yield subject_id, self._create_view(volume_ids=frozenset(groups[subject_id]))

    def get_obs_row_by_obs_id(self, obs_id: str) -> pd.DataFrame:
        """Get observation row by obs_id string identifier."""
        df = self.obs.read()
        filtered = df[df["obs_id"] == obs_id].reset_index(drop=True)
        return filtered

    def filter(self, expr: str) -> VolumeCollection:
        """Filter volumes using TileDB QueryCondition on obs. Returns view."""
        matching_ids = self._resolve_filter(expr)
        return self._create_view(volume_ids=matching_ids)

    def head(self, n: int = 5) -> VolumeCollection:
        """Return view of first n volumes."""
        return self.iloc[:n]

    def tail(self, n: int = 5) -> VolumeCollection:
        """Return view of last n volumes."""
        total = len(self)
        return self.iloc[max(0, total - n) :]

    def sample(self, n: int = 5, seed: int | None = None) -> VolumeCollection:
        """Return view of n randomly sampled volumes."""
        rng = np.random.default_rng(seed)
        obs_ids = self._effective_obs_ids
        n = min(n, len(obs_ids))
        sampled = rng.choice(obs_ids, size=n, replace=False)
        return self._create_view(volume_ids=frozenset(sampled))

    def lazy(self) -> CollectionQuery:
        """Enter lazy mode for transform pipelines via map()."""
        from radiobject.query import CollectionQuery

        return CollectionQuery(self._root, volume_ids=self._volume_ids)

    def map(self, fn: TransformFn) -> CollectionQuery:
        """Apply transform to all volumes during materialization. Returns lazy query."""
        return self.lazy().map(fn)

    def materialize(
        self,
        uri: str | None = None,
        name: str | None = None,
        ctx: tiledb.Ctx | None = None,
    ) -> VolumeCollection:
        """Write this collection (or view) to new storage.

        Creates a new VolumeCollection at the target URI containing all volumes
        in this view. For views, only the filtered volumes are written.

        Args:
            uri: Target URI. If None, generates adjacent to source collection.
            name: Collection name. Also used to derive URI when uri is None.
            ctx: TileDB context.
        """
        from radiobject.writers import VolumeCollectionWriter

        if uri is None:
            uri = _generate_adjacent_uri(self._root.uri, name=name)

        obs_ids = self._effective_obs_ids
        if not obs_ids:
            raise ValueError("No volumes to materialize")

        # Get obs DataFrame for this view
        obs_df = self.obs.read()
        if self._volume_ids is not None:
            obs_df = obs_df[obs_df["obs_id"].isin(self._volume_ids)].reset_index(drop=True)

        collection_name = name or self._root.name

        # Build obs schema from source
        obs_schema: dict[str, np.dtype] = {}
        for col in self._root.obs.columns:
            if col in ("obs_id", "obs_subject_id"):
                continue
            obs_schema[col] = self._root.obs.dtypes[col]

        effective_ctx = ctx if ctx else self._effective_ctx()

        with VolumeCollectionWriter(
            uri=uri,
            shape=self._root.shape,
            obs_schema=obs_schema,
            name=collection_name,
            ctx=effective_ctx,
        ) as writer:
            for obs_id in obs_ids:
                vol = self.loc[obs_id]
                data = vol.to_numpy()

                obs_row = obs_df[obs_df["obs_id"] == obs_id].iloc[0]
                attrs = {k: v for k, v in obs_row.items() if k not in ("obs_id", "obs_subject_id")}
                writer.write_volume(
                    data=data,
                    obs_id=obs_id,
                    obs_subject_id=obs_row["obs_subject_id"],
                    **attrs,
                )

        return VolumeCollection(uri, ctx=effective_ctx)

    def to_dataset(
        self,
        patch_size: tuple[int, int, int] | None = None,
        labels: pd.DataFrame | dict | str | None = None,
        transform: Callable[..., Any] | None = None,
    ) -> VolumeCollectionDataset:
        """Create PyTorch Dataset from this collection.

        Convenience method for ML training integration.

        Args:
            patch_size: If provided, extract random patches of this size.
            labels: Label source. Can be:
                - str: Column name in this collection's `obs` DataFrame
                - pd.DataFrame: With `obs_id` as column/index and label values
                - dict[str, Any]: Mapping from `obs_id` to label
                - None: No labels
            transform: Transform function applied to each sample.
                MONAI dict transforms (e.g., RandFlipd) work directly.

        Returns:
            VolumeCollectionDataset ready for use with `DataLoader`.

        Examples:
            Full volumes with labels from obs column:

                dataset = radi.CT.to_dataset(labels="has_tumor")

            Patch extraction:

                dataset = radi.CT.to_dataset(patch_size=(64, 64, 64), labels="grade")

            With MONAI transforms:

                from monai.transforms import NormalizeIntensityd
                dataset = radi.CT.to_dataset(
                    labels="has_tumor",
                    transform=NormalizeIntensityd(keys="image"),
                )
        """
        from radiobject.ml.config import DatasetConfig, LoadingMode
        from radiobject.ml.datasets.collection_dataset import VolumeCollectionDataset

        loading_mode = LoadingMode.PATCH if patch_size else LoadingMode.FULL_VOLUME
        config = DatasetConfig(loading_mode=loading_mode, patch_size=patch_size)

        return VolumeCollectionDataset(self, config=config, labels=labels, transform=transform)

    def to_obs(self) -> pd.DataFrame:
        """Return obs DataFrame (respects view filter)."""
        obs_df = self.obs.read()
        if self._volume_ids is not None:
            obs_df = obs_df[obs_df["obs_id"].isin(self._volume_ids)].reset_index(drop=True)
        return obs_df

    def copy(self) -> VolumeCollection:
        """Create detached copy of this collection (views remain views)."""
        if self.is_view:
            return VolumeCollection(
                uri=None,
                ctx=self._ctx,
                _source=self._root,
                _volume_ids=self._volume_ids,
            )
        return VolumeCollection(self._uri, ctx=self._ctx)

    def append(
        self,
        niftis: Sequence[tuple[str | Path, str]] | None = None,
        dicom_dirs: Sequence[tuple[str | Path, str]] | None = None,
        reorient: bool | None = None,
        progress: bool = False,
    ) -> None:
        """Append new volumes atomically.

        Volume data and obs metadata are written together to maintain consistency.
        Cannot be called on views - use `materialize()` first.

        Args:
            niftis: List of (nifti_path, obs_subject_id) tuples.
            dicom_dirs: List of (dicom_dir, obs_subject_id) tuples.
            reorient: Reorient to canonical orientation (None uses config default).
            progress: Show tqdm progress bar during volume writes.

        Example:
            Append new NIfTI files:

                radi.T1w.append(
                    niftis=[
                        ("sub101_T1w.nii.gz", "sub-101"),
                        ("sub102_T1w.nii.gz", "sub-102"),
                    ],
                )
        """
        self._check_not_view("append")

        if niftis is None and dicom_dirs is None:
            raise ValueError("Must provide either niftis or dicom_dirs")
        if niftis is not None and dicom_dirs is not None:
            raise ValueError("Cannot provide both niftis and dicom_dirs")

        effective_ctx = self._effective_ctx()
        current_count = len(self)

        if niftis is not None:
            self._append_niftis(niftis, reorient, effective_ctx, current_count, progress)
        else:
            self._append_dicoms(dicom_dirs, reorient, effective_ctx, current_count, progress)

        # Invalidate cached properties
        for prop in ("_index", "_metadata"):
            if prop in self.__dict__:
                del self.__dict__[prop]

    def validate(self) -> None:
        """Validate internal consistency of obs vs volume metadata."""
        self._check_not_view("validate")

        obs_data = self.obs.read()
        obs_ids_in_dataframe = set(obs_data["obs_id"])

        # Check each volume's obs_id against obs dataframe
        obs_ids_in_volumes = set()
        for i in range(len(self)):
            vol = self.iloc[i]
            if vol.obs_id is None:
                raise ValueError(f"Volume at index {i} lacks required obs_id metadata")
            obs_ids_in_volumes.add(vol.obs_id)

            expected_obs_id = obs_data.iloc[i]["obs_id"]
            if vol.obs_id != expected_obs_id:
                raise ValueError(
                    f"Position mismatch at index {i}: "
                    f"volume.obs_id={vol.obs_id}, obs.iloc[{i}]={expected_obs_id}"
                )

        missing_in_obs = obs_ids_in_volumes - obs_ids_in_dataframe
        if missing_in_obs:
            raise ValueError(f"Volumes without obs rows: {list(missing_in_obs)[:5]}")

        orphan_obs = obs_ids_in_dataframe - obs_ids_in_volumes
        if orphan_obs:
            raise ValueError(f"Obs rows without volumes: {list(orphan_obs)[:5]}")

        with tiledb.Group(f"{self._root.uri}/volumes", "r", ctx=self._effective_ctx()) as grp:
            actual_count = len(list(grp))
        if actual_count != self._metadata["n_volumes"]:
            raise ValueError(
                f"n_volumes mismatch: metadata={self._metadata['n_volumes']}, actual={actual_count}"
            )

    @property
    def _root(self) -> VolumeCollection:
        """The original attached VolumeCollection (follows source chain)."""
        return self._source._root if self._source else self

    @property
    def _effective_obs_ids(self) -> list[str]:
        """Get the list of obs_ids for this collection (filtered if view)."""
        if self._volume_ids is not None:
            return self._root._index.intersection(self._volume_ids).to_list()
        return self._root._index.to_list()

    @cached_property
    def _metadata(self) -> dict:
        """Cached group metadata."""
        with tiledb.Group(self._root.uri, "r", ctx=self._effective_ctx()) as grp:
            return dict(grp.meta)

    @cached_property
    def _index(self) -> Index:
        """Cached bidirectional index for obs_id lookups."""
        n = self._metadata["n_volumes"]
        if n == 0:
            return Index.build([], name="obs_id")
        obs_data = self.obs.read()
        return Index.build(list(obs_data["obs_id"]), name="obs_id")

    def _effective_ctx(self) -> tiledb.Ctx:
        return self._ctx if self._ctx else get_tiledb_ctx()

    def _check_not_view(self, operation: str) -> None:
        """Raise if this is a view (views are immutable)."""
        if self.is_view:
            raise TypeError(f"Cannot {operation} on a view. Call materialize(uri) first.")

    def _create_view(self, volume_ids: frozenset[str]) -> VolumeCollection:
        """Create a view with the given volume IDs, intersecting with current filter."""
        if self._volume_ids is not None:
            volume_ids = self._volume_ids & volume_ids
        return VolumeCollection(
            uri=None,
            ctx=self._ctx,
            _source=self._root,
            _volume_ids=volume_ids,
        )

    def _resolve_filter(self, expr: str) -> frozenset[str]:
        """Resolve filter expression to set of matching obs_ids."""
        effective_ctx = self._effective_ctx()
        obs_uri = f"{self._root.uri}/obs"

        with tiledb.open(obs_uri, "r", ctx=effective_ctx) as arr:
            result = arr.query(cond=expr, dims=["obs_id"])[:]
            obs_ids = result["obs_id"]
            matching = frozenset(v.decode() if isinstance(v, bytes) else str(v) for v in obs_ids)

        # Intersect with current view filter
        if self._volume_ids is not None:
            matching = matching & self._volume_ids

        return matching

    def _append_niftis(
        self,
        niftis: Sequence[tuple[str | Path, str]],
        reorient: bool | None,
        effective_ctx: tiledb.Ctx,
        start_index: int,
        progress: bool = False,
    ) -> None:
        """Internal: append NIfTI files to this collection."""
        # Extract metadata (no dimension validation for heterogeneous collections)
        metadata_list: list[tuple[Path, str, NiftiMetadata, str]] = []

        for nifti_path, obs_subject_id in niftis:
            path = Path(nifti_path)
            if not path.exists():
                raise FileNotFoundError(f"NIfTI file not found: {path}")

            metadata = extract_nifti_metadata(path)
            series_type = infer_series_type(path)

            # Only validate spatial dimensions if collection has uniform shape requirement
            if self.is_uniform and metadata.spatial_dimensions != self.shape:
                raise ValueError(
                    f"Dimension mismatch: {path.name} has spatial shape {metadata.spatial_dimensions}, "
                    f"expected {self.shape}"
                )

            metadata_list.append((path, obs_subject_id, metadata, series_type))

        # Check for duplicate obs_ids
        existing_obs_ids = set(self.obs_ids)
        new_obs_ids = {generate_obs_id(sid, st) for _, sid, _, st in metadata_list}
        duplicates = existing_obs_ids & new_obs_ids
        if duplicates:
            raise ValueError(f"obs_ids already exist: {sorted(duplicates)[:5]}")

        # Write volumes
        def write_volume(args: tuple[int, Path, str, NiftiMetadata, str]) -> WriteResult:
            idx, path, obs_subject_id, metadata, series_type = args
            worker_ctx = ctx_for_threads(self._ctx)
            volume_uri = f"{self.uri}/volumes/{idx}"
            obs_id = generate_obs_id(obs_subject_id, series_type)
            try:
                vol = Volume.from_nifti(volume_uri, path, ctx=worker_ctx, reorient=reorient)
                vol.set_obs_id(obs_id)
                return WriteResult(idx, volume_uri, obs_id, success=True)
            except Exception as e:
                return WriteResult(idx, volume_uri, obs_id, success=False, error=e)

        write_args = [
            (start_index + i, path, sid, meta, st)
            for i, (path, sid, meta, st) in enumerate(metadata_list)
        ]
        results = _write_volumes_parallel(
            write_volume, write_args, progress, f"Writing {self.name or 'volumes'}"
        )

        # Register volumes with group
        with tiledb.Group(f"{self.uri}/volumes", "w", ctx=effective_ctx) as vol_grp:
            for result in results:
                vol_grp.add(result.uri, name=str(result.index))

        # Build and write obs rows
        obs_rows: list[dict] = []
        for path, obs_subject_id, metadata, series_type in metadata_list:
            obs_id = generate_obs_id(obs_subject_id, series_type)
            obs_rows.append(metadata.to_obs_dict(obs_id, obs_subject_id, series_type))

        obs_df = pd.DataFrame(obs_rows)
        obs_uri = f"{self.uri}/obs"
        obs_subject_ids = obs_df["obs_subject_id"].astype(str).to_numpy()
        obs_ids = obs_df["obs_id"].astype(str).to_numpy()

        # Only write attributes that exist in the target schema
        existing_columns = set(self.obs.columns)
        with tiledb.open(obs_uri, "w", ctx=effective_ctx) as arr:
            attr_data = {
                col: obs_df[col].to_numpy()
                for col in obs_df.columns
                if col not in ("obs_subject_id", "obs_id") and col in existing_columns
            }
            arr[obs_subject_ids, obs_ids] = attr_data

        # Update n_volumes metadata
        new_count = start_index + len(niftis)
        with tiledb.Group(self.uri, "w", ctx=effective_ctx) as grp:
            grp.meta["n_volumes"] = new_count

    def _append_dicoms(
        self,
        dicom_dirs: Sequence[tuple[str | Path, str]],
        reorient: bool | None,
        effective_ctx: tiledb.Ctx,
        start_index: int,
        progress: bool = False,
    ) -> None:
        """Internal: append DICOM series to this collection."""
        metadata_list: list[tuple[Path, str, DicomMetadata]] = []

        for dicom_dir, obs_subject_id in dicom_dirs:
            path = Path(dicom_dir)
            if not path.exists():
                raise FileNotFoundError(f"DICOM directory not found: {path}")

            metadata = extract_dicom_metadata(path)
            dims = metadata.dimensions
            shape = (dims[1], dims[0], dims[2])

            # Only validate dimensions if collection has uniform shape requirement
            if self.is_uniform and shape != self.shape:
                raise ValueError(
                    f"Dimension mismatch: {path.name} has shape {shape}, expected {self.shape}"
                )

            metadata_list.append((path, obs_subject_id, metadata))

        # Check for duplicate obs_ids
        existing_obs_ids = set(self.obs_ids)
        new_obs_ids = {generate_obs_id(sid, meta.modality) for _, sid, meta in metadata_list}
        duplicates = existing_obs_ids & new_obs_ids
        if duplicates:
            raise ValueError(f"obs_ids already exist: {sorted(duplicates)[:5]}")

        # Write volumes
        def write_volume(args: tuple[int, Path, str, DicomMetadata]) -> WriteResult:
            idx, path, obs_subject_id, metadata = args
            worker_ctx = ctx_for_threads(self._ctx)
            volume_uri = f"{self.uri}/volumes/{idx}"
            obs_id = generate_obs_id(obs_subject_id, metadata.modality)
            try:
                vol = Volume.from_dicom(volume_uri, path, ctx=worker_ctx, reorient=reorient)
                vol.set_obs_id(obs_id)
                return WriteResult(idx, volume_uri, obs_id, success=True)
            except Exception as e:
                return WriteResult(idx, volume_uri, obs_id, success=False, error=e)

        write_args = [
            (start_index + i, path, sid, meta) for i, (path, sid, meta) in enumerate(metadata_list)
        ]
        results = _write_volumes_parallel(
            write_volume, write_args, progress, f"Writing {self.name or 'volumes'}"
        )

        # Register volumes with group
        with tiledb.Group(f"{self.uri}/volumes", "w", ctx=effective_ctx) as vol_grp:
            for result in results:
                vol_grp.add(result.uri, name=str(result.index))

        # Build and write obs rows
        obs_rows: list[dict] = []
        for path, obs_subject_id, metadata in metadata_list:
            obs_id = generate_obs_id(obs_subject_id, metadata.modality)
            obs_rows.append(metadata.to_obs_dict(obs_id, obs_subject_id))

        obs_df = pd.DataFrame(obs_rows)
        for col in ["kvp", "exposure", "repetition_time", "echo_time", "magnetic_field_strength"]:
            if col in obs_df.columns:
                obs_df[col] = obs_df[col].fillna(np.nan)

        obs_uri = f"{self.uri}/obs"
        obs_subject_ids = obs_df["obs_subject_id"].astype(str).to_numpy()
        obs_ids = obs_df["obs_id"].astype(str).to_numpy()

        # Only write attributes that exist in the target schema
        existing_columns = set(self.obs.columns)
        with tiledb.open(obs_uri, "w", ctx=effective_ctx) as arr:
            attr_data = {
                col: obs_df[col].to_numpy()
                for col in obs_df.columns
                if col not in ("obs_subject_id", "obs_id") and col in existing_columns
            }
            arr[obs_subject_ids, obs_ids] = attr_data

        # Update n_volumes metadata
        new_count = start_index + len(dicom_dirs)
        with tiledb.Group(self.uri, "w", ctx=effective_ctx) as grp:
            grp.meta["n_volumes"] = new_count

    @classmethod
    def from_niftis(
        cls,
        uri: str,
        niftis: Sequence[tuple[str | Path, str]],
        reorient: bool | None = None,
        validate_dimensions: bool = True,
        valid_subject_ids: set[str] | None = None,
        name: str | None = None,
        ctx: tiledb.Ctx | None = None,
        progress: bool = False,
    ) -> VolumeCollection:
        """Create VolumeCollection from NIfTI files with full metadata capture.

        Args:
            uri: Target URI for the VolumeCollection.
            niftis: List of (nifti_path, obs_subject_id) tuples.
            reorient: Reorient to canonical orientation (None uses config default).
            validate_dimensions: Raise if dimensions are inconsistent.
            valid_subject_ids: Optional whitelist for FK validation.
            name: Collection name (stored in metadata).
            ctx: TileDB context.
            progress: Show tqdm progress bar during volume writes.

        Returns:
            VolumeCollection with obs containing NIfTI metadata.
        """
        if not niftis:
            raise ValueError("At least one NIfTI file is required")

        # Validate subject IDs if whitelist provided
        if valid_subject_ids is not None:
            nifti_subject_ids = {sid for _, sid in niftis}
            invalid = nifti_subject_ids - valid_subject_ids
            if invalid:
                raise ValueError(f"Invalid obs_subject_ids: {sorted(invalid)[:5]}")

        # Extract metadata and validate spatial dimensions
        metadata_list: list[tuple[Path, str, NiftiMetadata, str]] = []
        first_spatial_shape: tuple[int, int, int] | None = None

        for nifti_path, obs_subject_id in niftis:
            path = Path(nifti_path)
            if not path.exists():
                raise FileNotFoundError(f"NIfTI file not found: {path}")

            metadata = extract_nifti_metadata(path)
            series_type = infer_series_type(path)

            spatial_shape = metadata.spatial_dimensions
            if first_spatial_shape is None:
                first_spatial_shape = spatial_shape
                all_same_shape = True
            elif spatial_shape != first_spatial_shape:
                if validate_dimensions:
                    raise ValueError(
                        f"Dimension mismatch: {path.name} has spatial shape {spatial_shape}, "
                        f"expected {first_spatial_shape}"
                    )
                all_same_shape = False

            metadata_list.append((path, obs_subject_id, metadata, series_type))

        effective_ctx = ctx if ctx else get_tiledb_ctx()
        # Only set uniform shape if all volumes have same spatial dimensions
        collection_shape = first_spatial_shape if all_same_shape else None

        # Build obs schema from NiftiMetadata fields (tuples serialized as strings)
        obs_schema: dict[str, np.dtype] = {
            "series_type": np.dtype("U32"),
            "voxel_spacing": np.dtype("U64"),  # Tuple serialized as string
            "dimensions": np.dtype("U64"),  # Tuple serialized as string
            "datatype": np.int32,
            "bitpix": np.int32,
            "scl_slope": np.float64,
            "scl_inter": np.float64,
            "xyzt_units": np.int32,
            "spatial_units": np.dtype("U16"),
            "qform_code": np.int32,
            "sform_code": np.int32,
            "axcodes": np.dtype("U8"),
            "affine_json": np.dtype("U512"),
            "orientation_source": np.dtype("U32"),
            "source_path": np.dtype("U512"),
        }

        # Create collection
        cls._create(
            uri,
            shape=collection_shape,
            obs_schema=obs_schema,
            n_volumes=len(niftis),
            name=name,
            ctx=ctx,
        )

        # Write volumes in parallel
        def write_volume(args: tuple[int, Path, str, NiftiMetadata, str]) -> WriteResult:
            idx, path, obs_subject_id, metadata, series_type = args
            worker_ctx = ctx_for_threads(ctx)
            volume_uri = f"{uri}/volumes/{idx}"
            obs_id = generate_obs_id(obs_subject_id, series_type)
            try:
                vol = Volume.from_nifti(volume_uri, path, ctx=worker_ctx, reorient=reorient)
                vol.set_obs_id(obs_id)
                return WriteResult(idx, volume_uri, obs_id, success=True)
            except Exception as e:
                return WriteResult(idx, volume_uri, obs_id, success=False, error=e)

        write_args = [
            (idx, path, sid, meta, st) for idx, (path, sid, meta, st) in enumerate(metadata_list)
        ]
        results = _write_volumes_parallel(
            write_volume, write_args, progress, f"Writing {name or 'volumes'}"
        )

        # Register volumes with group
        with tiledb.Group(f"{uri}/volumes", "w", ctx=effective_ctx) as vol_grp:
            for result in results:
                vol_grp.add(result.uri, name=str(result.index))

        # Build obs DataFrame rows
        obs_rows: list[dict] = []
        for path, obs_subject_id, metadata, series_type in metadata_list:
            obs_id = generate_obs_id(obs_subject_id, series_type)
            obs_rows.append(metadata.to_obs_dict(obs_id, obs_subject_id, series_type))

        obs_df = pd.DataFrame(obs_rows)

        # Write obs data
        obs_uri = f"{uri}/obs"
        obs_subject_ids = obs_df["obs_subject_id"].astype(str).to_numpy()
        obs_ids = obs_df["obs_id"].astype(str).to_numpy()
        with tiledb.open(obs_uri, "w", ctx=effective_ctx) as arr:
            attr_data = {
                col: obs_df[col].to_numpy()
                for col in obs_df.columns
                if col not in ("obs_subject_id", "obs_id")
            }
            arr[obs_subject_ids, obs_ids] = attr_data

        return cls(uri, ctx=ctx)

    @classmethod
    def from_dicoms(
        cls,
        uri: str,
        dicom_dirs: Sequence[tuple[str | Path, str]],
        reorient: bool | None = None,
        validate_dimensions: bool = True,
        valid_subject_ids: set[str] | None = None,
        name: str | None = None,
        ctx: tiledb.Ctx | None = None,
        progress: bool = False,
    ) -> VolumeCollection:
        """Create VolumeCollection from DICOM series with full metadata capture.

        Args:
            uri: Target URI for the VolumeCollection.
            dicom_dirs: List of (dicom_dir, obs_subject_id) tuples.
            reorient: Reorient to canonical orientation (None uses config default).
            validate_dimensions: Raise if dimensions are inconsistent.
            valid_subject_ids: Optional whitelist for FK validation.
            name: Collection name (stored in metadata).
            ctx: TileDB context.
            progress: Show tqdm progress bar during volume writes.

        Returns:
            VolumeCollection with obs containing DICOM metadata.
        """
        if not dicom_dirs:
            raise ValueError("At least one DICOM directory is required")

        # Validate subject IDs if whitelist provided
        if valid_subject_ids is not None:
            dicom_subject_ids = {sid for _, sid in dicom_dirs}
            invalid = dicom_subject_ids - valid_subject_ids
            if invalid:
                raise ValueError(f"Invalid obs_subject_ids: {sorted(invalid)[:5]}")

        # Extract metadata and validate dimensions
        metadata_list: list[tuple[Path, str, DicomMetadata]] = []
        first_shape: tuple[int, int, int] | None = None

        for dicom_dir, obs_subject_id in dicom_dirs:
            path = Path(dicom_dir)
            if not path.exists():
                raise FileNotFoundError(f"DICOM directory not found: {path}")

            metadata = extract_dicom_metadata(path)

            # DICOM dimensions tuple is (rows, columns, n_slices)
            # Swap to (columns, rows, n_slices) to match X, Y, Z convention
            dims = metadata.dimensions
            shape = (dims[1], dims[0], dims[2])
            if first_shape is None:
                first_shape = shape
                all_same_shape = True
            elif shape != first_shape:
                if validate_dimensions:
                    raise ValueError(
                        f"Dimension mismatch: {path.name} has shape {shape}, expected {first_shape}"
                    )
                all_same_shape = False

            metadata_list.append((path, obs_subject_id, metadata))

        effective_ctx = ctx if ctx else get_tiledb_ctx()
        # Only set uniform shape if all volumes have same dimensions
        collection_shape = first_shape if all_same_shape else None

        # Build obs schema from DicomMetadata fields (tuples serialized as strings)
        obs_schema: dict[str, np.dtype] = {
            "voxel_spacing": np.dtype("U64"),  # Tuple serialized as string
            "dimensions": np.dtype("U64"),  # Tuple serialized as string
            "modality": np.dtype("U16"),
            "series_description": np.dtype("U256"),
            "kvp": np.float64,
            "exposure": np.float64,
            "repetition_time": np.float64,
            "echo_time": np.float64,
            "magnetic_field_strength": np.float64,
            "axcodes": np.dtype("U8"),
            "affine_json": np.dtype("U512"),
            "orientation_source": np.dtype("U32"),
            "source_path": np.dtype("U512"),
        }

        # Create collection
        cls._create(
            uri,
            shape=collection_shape,
            obs_schema=obs_schema,
            n_volumes=len(dicom_dirs),
            name=name,
            ctx=ctx,
        )

        # Write volumes in parallel
        def write_volume(args: tuple[int, Path, str, DicomMetadata]) -> WriteResult:
            idx, path, obs_subject_id, metadata = args
            worker_ctx = ctx_for_threads(ctx)
            volume_uri = f"{uri}/volumes/{idx}"
            obs_id = generate_obs_id(obs_subject_id, metadata.modality)
            try:
                vol = Volume.from_dicom(volume_uri, path, ctx=worker_ctx, reorient=reorient)
                vol.set_obs_id(obs_id)
                return WriteResult(idx, volume_uri, obs_id, success=True)
            except Exception as e:
                return WriteResult(idx, volume_uri, obs_id, success=False, error=e)

        write_args = [(idx, path, sid, meta) for idx, (path, sid, meta) in enumerate(metadata_list)]
        results = _write_volumes_parallel(
            write_volume, write_args, progress, f"Writing {name or 'volumes'}"
        )

        # Register volumes with group
        with tiledb.Group(f"{uri}/volumes", "w", ctx=effective_ctx) as vol_grp:
            for result in results:
                vol_grp.add(result.uri, name=str(result.index))

        # Build obs DataFrame rows
        obs_rows: list[dict] = []
        for path, obs_subject_id, metadata in metadata_list:
            obs_id = generate_obs_id(obs_subject_id, metadata.modality)
            obs_rows.append(metadata.to_obs_dict(obs_id, obs_subject_id))

        obs_df = pd.DataFrame(obs_rows)

        # Handle None values for optional fields
        for col in ["kvp", "exposure", "repetition_time", "echo_time", "magnetic_field_strength"]:
            obs_df[col] = obs_df[col].fillna(np.nan)

        # Write obs data
        obs_uri = f"{uri}/obs"
        obs_subject_ids = obs_df["obs_subject_id"].astype(str).to_numpy()
        obs_ids = obs_df["obs_id"].astype(str).to_numpy()
        with tiledb.open(obs_uri, "w", ctx=effective_ctx) as arr:
            attr_data = {
                col: obs_df[col].to_numpy()
                for col in obs_df.columns
                if col not in ("obs_subject_id", "obs_id")
            }
            arr[obs_subject_ids, obs_ids] = attr_data

        return cls(uri, ctx=ctx)

    @classmethod
    def _create(
        cls,
        uri: str,
        shape: tuple[int, int, int] | None = None,
        obs_schema: dict[str, np.dtype] | None = None,
        n_volumes: int = 0,
        name: str | None = None,
        ctx: tiledb.Ctx | None = None,
    ) -> VolumeCollection:
        """Internal: create empty collection with optional uniform dimensions.

        Args:
            uri: Target URI for the collection
            shape: If provided, enforces uniform dimensions. If None, allows heterogeneous shapes.
            obs_schema: Schema for volume-level obs attributes
            n_volumes: Initial volume count (usually 0)
            name: Collection name
            ctx: TileDB context
        """
        effective_ctx = ctx if ctx else get_tiledb_ctx()

        tiledb.Group.create(uri, ctx=effective_ctx)

        volumes_uri = f"{uri}/volumes"
        tiledb.Group.create(volumes_uri, ctx=effective_ctx)

        obs_uri = f"{uri}/obs"
        Dataframe.create(obs_uri, schema=obs_schema or {}, ctx=ctx)

        with tiledb.Group(uri, "w", ctx=effective_ctx) as grp:
            if shape is not None:
                grp.meta["x_dim"] = shape[0]
                grp.meta["y_dim"] = shape[1]
                grp.meta["z_dim"] = shape[2]
            grp.meta["n_volumes"] = n_volumes
            if name is not None:
                grp.meta["name"] = name
            grp.add(volumes_uri, name="volumes")
            grp.add(obs_uri, name="obs")

        return cls(uri, ctx=ctx)

    @classmethod
    def _from_volumes(
        cls,
        uri: str,
        volumes: Sequence[tuple[str, Volume]],
        obs_data: pd.DataFrame | None = None,
        name: str | None = None,
        ctx: tiledb.Ctx | None = None,
    ) -> VolumeCollection:
        """Internal: create collection from existing volumes (write-once)."""
        if not volumes:
            raise ValueError("At least one volume is required")

        first_shape = volumes[0][1].shape[:3]
        for obs_id, vol in volumes:
            if vol.shape[:3] != first_shape:
                raise ValueError(
                    f"Volume '{obs_id}' has shape {vol.shape[:3]}, expected {first_shape}"
                )

        effective_ctx = ctx if ctx else get_tiledb_ctx()

        obs_schema = None
        if obs_data is not None:
            obs_schema = {}
            for col in obs_data.columns:
                if col in ("obs_id", "obs_subject_id"):
                    continue
                dtype = obs_data[col].to_numpy().dtype
                if dtype == np.dtype("O"):
                    dtype = np.dtype("U64")
                obs_schema[col] = dtype

        cls._create(
            uri,
            shape=first_shape,
            obs_schema=obs_schema,
            n_volumes=len(volumes),
            name=name,
            ctx=ctx,
        )

        with tiledb.Group(uri, "w", ctx=effective_ctx) as grp:
            grp.meta["n_volumes"] = len(volumes)

        def write_volume(args: tuple[int, str, Volume]) -> WriteResult:
            idx, obs_id, vol = args
            worker_ctx = ctx_for_threads(ctx)
            volume_uri = f"{uri}/volumes/{idx}"
            try:
                data = vol.to_numpy()
                new_vol = Volume.from_numpy(volume_uri, data, ctx=worker_ctx)
                new_vol.set_obs_id(obs_id)
                return WriteResult(idx, volume_uri, obs_id, success=True)
            except Exception as e:
                return WriteResult(idx, volume_uri, obs_id, success=False, error=e)

        write_args = [(idx, obs_id, vol) for idx, (obs_id, vol) in enumerate(volumes)]
        results = _write_volumes_parallel(
            write_volume, write_args, progress=False, desc="Writing volumes"
        )

        with tiledb.Group(f"{uri}/volumes", "w", ctx=effective_ctx) as vol_grp:
            for result in results:
                vol_grp.add(result.uri, name=str(result.index))

        obs_ids = np.array([obs_id for obs_id, _ in volumes])
        if obs_data is not None and "obs_subject_id" in obs_data.columns:
            obs_subject_ids = obs_data["obs_subject_id"].astype(str).to_numpy()
        else:
            obs_subject_ids = obs_ids.copy()

        obs_uri = f"{uri}/obs"
        with tiledb.open(obs_uri, "w", ctx=effective_ctx) as arr:
            attr_data = {}
            if obs_data is not None:
                for col in obs_data.columns:
                    if col not in ("obs_id", "obs_subject_id"):
                        attr_data[col] = obs_data[col].to_numpy()
            arr[obs_subject_ids, obs_ids] = attr_data

        return cls(uri, ctx=ctx)
