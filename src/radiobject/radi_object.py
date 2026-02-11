"""RadiObject - top-level container for multi-collection radiology data."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from functools import cached_property
from pathlib import Path
from typing import Sequence, overload

import numpy as np
import numpy.typing as npt
import pandas as pd
import tiledb

from radiobject.ctx import get_tiledb_ctx
from radiobject.dataframe import Dataframe
from radiobject.exceptions import AlignmentError, ViewError
from radiobject.indexing import Index
from radiobject.parallel import WriteResult, ctx_for_threads
from radiobject.utils import (
    _aggregate_obs_ids,
    build_obs_ids_mapping,
    build_obs_meta_schema,
    create_and_write_obs_meta,
    ensure_obs_columns,
    merge_obs_ids,
    write_obs_dataframe,
)
from radiobject.volume import Volume
from radiobject.volume_collection import (
    VolumeCollection,
    _normalize_index,
    _write_volumes_parallel,
)

log = logging.getLogger(__name__)


class _SubjectILocIndexer:
    """Integer-location based indexer for RadiObject subjects."""

    def __init__(self, radi_object: RadiObject):
        self._radi_object = radi_object

    @overload
    def __getitem__(self, key: int) -> RadiObject: ...
    @overload
    def __getitem__(self, key: slice) -> RadiObject: ...
    @overload
    def __getitem__(self, key: list[int]) -> RadiObject: ...
    @overload
    def __getitem__(self, key: npt.NDArray[np.bool_]) -> RadiObject: ...

    def __getitem__(self, key: int | slice | list[int] | npt.NDArray[np.bool_]) -> RadiObject:
        """Returns a RadiObject view filtered to selected subject indices."""
        n = len(self._radi_object)
        if isinstance(key, int):
            idx = _normalize_index(key, n)
            return self._radi_object._filter_by_indices([idx])
        elif isinstance(key, slice):
            indices = list(range(*key.indices(n)))
            return self._radi_object._filter_by_indices(indices)
        elif isinstance(key, np.ndarray) and key.dtype == np.bool_:
            if len(key) != n:
                raise ValueError(f"Boolean mask length {len(key)} != subject count {n}")
            indices = list(np.where(key)[0])
            return self._radi_object._filter_by_indices(indices)
        elif isinstance(key, list):
            indices = [_normalize_index(i, n) for i in key]
            return self._radi_object._filter_by_indices(indices)
        raise TypeError(
            f"iloc indices must be int, slice, list[int], or boolean array, got {type(key)}"
        )


class _SubjectLocIndexer:
    """Label-based indexer for RadiObject subjects."""

    def __init__(self, radi_object: RadiObject):
        self._radi_object = radi_object

    @overload
    def __getitem__(self, key: str) -> RadiObject: ...
    @overload
    def __getitem__(self, key: list[str]) -> RadiObject: ...

    def __getitem__(self, key: str | list[str]) -> RadiObject:
        """Returns a RadiObject view filtered to selected obs_subject_ids."""
        if isinstance(key, str):
            return self._radi_object._filter_by_subject_ids([key])
        elif isinstance(key, list):
            return self._radi_object._filter_by_subject_ids(key)
        raise TypeError(f"loc indices must be str or list[str], got {type(key)}")


class RadiObject:
    """Top-level container for multi-collection radiology data with subject metadata.

    RadiObject can be either "attached" (backed by storage at a URI) or a "view"
    (filtered subset referencing a source RadiObject). Views are created by
    filtering operations and read data from their source with filters applied.

    Views are immutable. To persist a view, use `write(uri)`.

    Examples:
        Attached (has URI):

            radi = RadiObject("s3://bucket/dataset")
            radi.is_view  # False
            radi.uri      # "s3://bucket/dataset"

        View (filtered, no URI):

            subset = radi.filter("age > 40")
            subset.is_view  # True
            subset.uri      # None
    """

    def __init__(
        self,
        uri: str | None,
        ctx: tiledb.Ctx | None = None,
        *,
        # View state (internal use only)
        _source: RadiObject | None = None,
        _subject_ids: frozenset[str] | None = None,
        _collection_names: frozenset[str] | None = None,
    ):
        self._uri: str | None = uri
        self._ctx: tiledb.Ctx | None = ctx
        # View state
        self._source: RadiObject | None = _source
        self._subject_ids: frozenset[str] | None = _subject_ids
        self._collection_names_filter: frozenset[str] | None = _collection_names

    def __len__(self) -> int:
        """Number of subjects."""
        if self.is_view:
            return len(self._index)
        return int(self._metadata.get("subject_count", 0))

    def __iter__(self) -> Iterator[str]:
        """Iterate over collection names."""
        return iter(self.collection_names)

    def __repr__(self) -> str:
        """Concise representation of the RadiObject."""
        collections = ", ".join(self.collection_names) if self.collection_names else "none"
        view_indicator = " (view)" if self.is_view else ""
        return (
            f"RadiObject({len(self)} subjects, {self.n_collections} collections: "
            f"[{collections}]){view_indicator}"
        )

    @overload
    def __getitem__(self, key: str) -> RadiObject: ...
    @overload
    def __getitem__(self, key: list[str]) -> RadiObject: ...

    def __getitem__(self, key: str | list[str]) -> RadiObject:
        """Bracket indexing for subjects by obs_subject_id.

        Alias for .loc[] - allows `radi["BraTS001"]` as shorthand for `radi.loc["BraTS001"]`.
        """
        return self.loc[key]

    def __getattr__(self, name: str) -> VolumeCollection:
        """Attribute access to collections (e.g., radi.T1w)."""
        if name.startswith("_"):
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")
        try:
            return self.collection(name)
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' has no collection '{name}'")

    @property
    def uri(self) -> str | None:
        """URI of this RadiObject, or None if this is a view."""
        return self._uri

    @property
    def is_view(self) -> bool:
        """True if this RadiObject is a filtered view of another."""
        return self._source is not None

    @property
    def n_collections(self) -> int:
        """Number of VolumeCollections."""
        return len(self.collection_names)

    @property
    def obs_meta(self) -> pd.DataFrame | Dataframe:
        """Subject-level observational metadata.

        Returns Dataframe for attached RadiObject, pd.DataFrame for views.
        """
        if self.is_view:
            # Return filtered DataFrame
            full_obs_meta = self._root.obs_meta.read()
            if self._subject_ids is not None:
                return full_obs_meta[
                    full_obs_meta["obs_subject_id"].isin(self._subject_ids)
                ].reset_index(drop=True)
            return full_obs_meta
        obs_meta_uri = f"{self._effective_uri()}/obs_meta"
        return Dataframe(uri=obs_meta_uri, ctx=self._ctx)

    @property
    def obs_subject_ids(self) -> list[str]:
        """All obs_subject_id values in index order."""
        return self._index.to_list()

    @property
    def index(self) -> Index:
        """Subject index for bidirectional ID/position lookups."""
        return self._index

    @cached_property
    def all_obs_ids(self) -> list[str]:
        """All obs_ids across all collections (for uniqueness checks)."""
        obs_ids = []
        for name in self.collection_names:
            obs_ids.extend(self.collection(name).obs_ids)
        return obs_ids

    @cached_property
    def collection_names(self) -> tuple[str, ...]:
        """Names of all VolumeCollections."""
        if self.is_view and self._collection_names_filter is not None:
            # Return filtered collection names (preserving root order)
            root_names = self._root.collection_names
            return tuple(name for name in root_names if name in self._collection_names_filter)
        uri = self._effective_uri()
        collections_uri = f"{uri}/collections"
        with tiledb.Group(collections_uri, "r", ctx=self._effective_ctx()) as grp:
            return tuple(obj.name for obj in grp)

    @cached_property
    def iloc(self) -> _SubjectILocIndexer:
        """Integer-location based indexing for selecting subjects by position."""
        return _SubjectILocIndexer(self)

    @cached_property
    def loc(self) -> _SubjectLocIndexer:
        """Label-based indexing for selecting subjects by obs_subject_id."""
        return _SubjectLocIndexer(self)

    def sel(self, *, subject: str | list[str]) -> RadiObject:
        """Select subjects by obs_subject_id. Named-parameter alias for .loc[]."""
        return self.loc[subject]

    def collection(self, name: str) -> VolumeCollection:
        """Get a VolumeCollection by name."""
        if name not in self.collection_names:
            raise KeyError(f"Collection '{name}' not found. Available: {self.collection_names}")
        uri = self._effective_uri()
        collection_uri = f"{uri}/collections/{name}"
        return VolumeCollection(collection_uri, ctx=self._ctx)

    def get_volume(self, obs_id: str) -> Volume:
        """Get a volume by obs_id from any collection."""
        for name in self.collection_names:
            coll = self.collection(name)
            if obs_id in coll.index:
                return coll.loc[obs_id]
        raise KeyError(f"obs_id '{obs_id}' not found in any collection")

    def get_obs_row_by_obs_subject_id(self, obs_subject_id: str) -> pd.DataFrame:
        """Get obs_meta row by obs_subject_id string identifier."""
        if self.is_view:
            obs_meta_df = self.obs_meta
            return obs_meta_df[obs_meta_df["obs_subject_id"] == obs_subject_id].reset_index(
                drop=True
            )
        df = self.obs_meta.read()
        filtered = df[df["obs_subject_id"] == obs_subject_id].reset_index(drop=True)
        return filtered

    def filter(self, expr: str) -> RadiObject:
        """Filter subjects using a query expression on obs_meta.

        Args:
            expr: TileDB QueryCondition string (e.g., "tumor_grade == 'HGG' and age > 40")

        Returns:
            RadiObject view filtered to matching subjects
        """
        if self.is_view:
            # Filter from the obs_meta DataFrame
            obs_meta_df = self.obs_meta
            # Use pandas query for view filtering
            filtered = obs_meta_df.query(expr)
            subject_ids = frozenset(filtered["obs_subject_id"])
        else:
            # Use TileDB QueryCondition for attached RadiObject
            filtered = self.obs_meta.read(value_filter=expr)
            subject_ids = frozenset(filtered["obs_subject_id"])
        return self._create_view(subject_ids=subject_ids)

    def head(self, n: int = 5) -> RadiObject:
        """Return view of first n subjects."""
        n = min(n, len(self))
        return self._filter_by_indices(list(range(n)))

    def tail(self, n: int = 5) -> RadiObject:
        """Return view of last n subjects."""
        total = len(self)
        n = min(n, total)
        return self._filter_by_indices(list(range(total - n, total)))

    def sample(self, n: int = 5, seed: int | None = None) -> RadiObject:
        """Return view of n randomly sampled subjects."""
        rng = np.random.default_rng(seed)
        total = len(self)
        n = min(n, total)
        indices = list(rng.choice(total, size=n, replace=False))
        return self._filter_by_indices(sorted(indices))

    def select_collections(self, names: list[str]) -> RadiObject:
        """Create a view with only specified collections."""
        current_names = set(self.collection_names)
        for name in names:
            if name not in current_names:
                raise KeyError(f"Collection '{name}' not found")
        return self._create_view(collection_names=frozenset(names))

    def write(
        self,
        uri: str,
        ctx: tiledb.Ctx | None = None,
    ) -> RadiObject:
        """Write this RadiObject (or view) to storage.

        For attached RadiObjects, this copies the entire dataset.
        For views, this writes only the filtered subset.

        Args:
            uri: Target URI for the new RadiObject
            ctx: TileDB context

        Returns:
            New attached RadiObject at the target URI
        """
        # Get filtered obs_meta
        if self.is_view:
            filtered_obs_meta = self.obs_meta  # Already filtered DataFrame
        else:
            filtered_obs_meta = self.obs_meta.read()

        obs_meta_schema = build_obs_meta_schema(filtered_obs_meta)
        return self._write_streaming(uri, filtered_obs_meta, obs_meta_schema, ctx)

    def copy(self) -> RadiObject:
        """Create an independent in-memory copy, detached from the view chain.

        Useful when you want to break the reference to the source RadiObject.
        Note: This does NOT persist data. Call write(uri) to write to storage.
        """
        if not self.is_view:
            # For attached RadiObject, just return self (already independent)
            return self
        # Create a new view with the same filters but mark it as "detached"
        # In practice, since we always point to _root, this is already independent
        return RadiObject(
            uri=None,
            ctx=self._ctx,
            _source=self._root,
            _subject_ids=self._subject_ids,
            _collection_names=self._collection_names_filter,
        )

    def describe(self) -> str:
        """Return a summary: subjects, collections, shapes, and label distributions."""
        lines = [
            "RadiObject Summary",
            "==================",
            f"URI: {self.uri or '(view)'}",
            f"Subjects: {len(self)}",
            f"Collections: {self.n_collections}",
            "",
            "Collections:",
        ]

        for name in self.collection_names:
            coll = self.collection(name)
            shape = coll.shape
            shape_str = "x".join(str(d) for d in shape) if shape else "heterogeneous"
            uniform_str = "" if coll.is_uniform else " (mixed shapes)"
            lines.append(f"  - {name}: {len(coll)} volumes, shape={shape_str}{uniform_str}")

        # Find label columns
        obs_meta_df = self.obs_meta if self.is_view else self.obs_meta.read()
        label_cols = []
        for col in obs_meta_df.columns:
            if col in ("obs_subject_id", "obs_id"):
                continue
            dtype = obs_meta_df[col].dtype
            if dtype in (np.int64, np.int32, np.float64, np.float32, object):
                n_unique = obs_meta_df[col].nunique()
                if n_unique <= 10:
                    label_cols.append(col)

        if label_cols:
            lines.append("")
            lines.append("Label Columns:")
            for col in label_cols:
                value_counts = obs_meta_df[col].value_counts().to_dict()
                lines.append(f"  - {col}: {value_counts}")

        return "\n".join(lines)

    def append(
        self,
        images: dict[str, str | Path | Sequence[tuple[str | Path, str]]],
        obs_meta: pd.DataFrame | None = None,
        reorient: bool | None = None,
        format_hint: dict[str, str] | None = None,
        progress: bool = False,
    ) -> None:
        """Append new subjects and their volumes atomically."""
        from radiobject.ingest import ImageFormat, resolve_image_source

        self._check_not_view("append")

        if not images:
            raise ValueError("images dict cannot be empty")

        # Resolve each collection source
        hint_map = format_hint or {}
        groups: dict[str, tuple[list[tuple[Path, str]], ImageFormat]] = {}
        for coll_name, source in images.items():
            hint = ImageFormat(hint_map[coll_name]) if coll_name in hint_map else None
            items, fmt = resolve_image_source(source, format_hint=hint)
            groups[coll_name] = (items, fmt)

        effective_ctx = self._effective_ctx()
        uri = self._effective_uri()

        # Collect all subject IDs from input
        input_subject_ids: set[str] = set()
        for items, _ in groups.values():
            input_subject_ids.update(sid for _, sid in items)

        existing_subject_ids = set(self.obs_subject_ids)
        new_subject_ids = input_subject_ids - existing_subject_ids

        # Validate obs_meta
        if new_subject_ids:
            if obs_meta is None:
                raise ValueError(
                    f"obs_meta required for new subjects: {sorted(new_subject_ids)[:5]}"
                )
            ensure_obs_columns(obs_meta, context="RadiObject.append")
            obs_meta_ids = set(obs_meta["obs_subject_id"])
            missing = new_subject_ids - obs_meta_ids
            if missing:
                raise ValueError(f"obs_meta missing entries for: {sorted(missing)[:5]}")
            obs_meta = obs_meta[obs_meta["obs_subject_id"].isin(new_subject_ids)].copy()

        # Append obs_meta for new subjects (inject obs_ids placeholder)
        if obs_meta is not None and len(obs_meta) > 0:
            if "obs_ids" not in obs_meta.columns:
                obs_meta["obs_ids"] = "[]"
            obs_meta_uri = f"{uri}/obs_meta"
            existing_columns = set(self._root.obs_meta.columns)
            write_obs_dataframe(obs_meta_uri, obs_meta, ctx=effective_ctx, columns=existing_columns)

            new_count = len(self) + len(obs_meta)
            with tiledb.Group(uri, "w", ctx=effective_ctx) as grp:
                grp.meta["subject_count"] = new_count

        # Process each collection
        collections_uri = f"{uri}/collections"
        existing_collections = set(self.collection_names)

        for coll_name, (items, fmt) in groups.items():
            if coll_name in existing_collections:
                vc = self.collection(coll_name)
                if fmt == ImageFormat.NIFTI:
                    vc.append(niftis=items, reorient=reorient, progress=progress)
                else:
                    vc.append(dicom_dirs=items, reorient=reorient, progress=progress)
            else:
                vc_uri = f"{collections_uri}/{coll_name}"
                if fmt == ImageFormat.NIFTI:
                    VolumeCollection.from_niftis(
                        uri=vc_uri,
                        niftis=list(items),
                        reorient=reorient,
                        validate_dimensions=False,
                        name=coll_name,
                        ctx=self._ctx,
                        progress=progress,
                    )
                else:
                    VolumeCollection.from_dicoms(
                        uri=vc_uri,
                        dicom_dirs=list(items),
                        reorient=reorient,
                        validate_dimensions=True,
                        name=coll_name,
                        ctx=self._ctx,
                        progress=progress,
                    )
                with tiledb.Group(collections_uri, "w", ctx=effective_ctx) as grp:
                    grp.add(vc_uri, name=coll_name)
                with tiledb.Group(uri, "w", ctx=effective_ctx) as grp:
                    grp.meta["n_collections"] = self.n_collections + 1
                existing_collections.add(coll_name)

        # Update obs_ids in obs_meta for affected subjects
        obs_meta_uri = f"{uri}/obs_meta"
        # Invalidate first so collection_names is fresh
        for prop in ("_index", "_metadata", "collection_names"):
            if prop in self.__dict__:
                del self.__dict__[prop]

        all_collections = [self.collection(name) for name in self.collection_names]
        obs_ids_map = build_obs_ids_mapping(all_collections)
        affected_subjects = obs_ids_map[obs_ids_map["obs_subject_id"].isin(input_subject_ids)]
        if len(affected_subjects) > 0:
            write_obs_dataframe(
                obs_meta_uri,
                affected_subjects,
                ctx=effective_ctx,
                columns={"obs_ids"},
            )

    def rename_collection(self, old_name: str, new_name: str) -> None:
        """Rename a collection."""
        self._check_not_view("rename_collection")
        if old_name not in self.collection_names:
            raise KeyError(f"Collection '{old_name}' not found")
        if new_name in self.collection_names:
            raise ValueError(f"Collection '{new_name}' already exists")

        effective_ctx = self._effective_ctx()
        uri = self._effective_uri()
        collections_uri = f"{uri}/collections"
        old_uri = f"{collections_uri}/{old_name}"

        with tiledb.Group(old_uri, "w", ctx=effective_ctx) as grp:
            grp.meta["name"] = new_name

        with tiledb.Group(collections_uri, "w", ctx=effective_ctx) as grp:
            grp.remove(old_name)
            grp.add(old_uri, name=new_name)

        if "collection_names" in self.__dict__:
            del self.__dict__["collection_names"]

    def add_collection(self, name: str, vc: VolumeCollection | str) -> None:
        """Register an existing VolumeCollection into this RadiObject.

        Links the collection if it's already at the expected URI
        ({uri}/collections/{name}), otherwise copies it. Updates obs_meta
        with any new subjects found in the collection.
        """
        self._check_not_view("add_collection")
        if name in self.collection_names:
            raise ValueError(f"Collection '{name}' already exists")

        effective_ctx = self._effective_ctx()
        uri = self._effective_uri()
        collections_uri = f"{uri}/collections"
        expected_uri = f"{collections_uri}/{name}"

        # Resolve string URI to VolumeCollection
        if isinstance(vc, str):
            vc = VolumeCollection(vc, ctx=self._ctx)

        # Link or copy
        if vc.uri == expected_uri:
            with tiledb.Group(collections_uri, "w", ctx=effective_ctx) as grp:
                grp.add(vc.uri, name=name)
        else:
            _copy_volume_collection(vc, expected_uri, name=name, ctx=self._ctx)
            with tiledb.Group(collections_uri, "w", ctx=effective_ctx) as grp:
                grp.add(expected_uri, name=name)

        # Update obs_meta with any new subjects
        vc_obs = vc.obs.read()
        vc_subject_ids = set(vc_obs["obs_subject_id"])
        existing_subject_ids = set(self.obs_subject_ids)
        new_subject_ids = vc_subject_ids - existing_subject_ids

        if new_subject_ids:
            obs_meta_uri = f"{uri}/obs_meta"
            new_ids = sorted(new_subject_ids)
            new_meta = pd.DataFrame({"obs_subject_id": new_ids, "obs_ids": ["[]"] * len(new_ids)})
            write_obs_dataframe(obs_meta_uri, new_meta, ctx=effective_ctx)

        # Update obs_ids for all subjects in the new collection
        obs_meta_uri = f"{uri}/obs_meta"
        # Invalidate cached props first to get fresh collection_names
        for prop in ("_index", "_metadata", "collection_names", "all_obs_ids"):
            if prop in self.__dict__:
                del self.__dict__[prop]

        all_collections = [self.collection(n) for n in self.collection_names]
        obs_ids_map = build_obs_ids_mapping(all_collections)
        affected = obs_ids_map[obs_ids_map["obs_subject_id"].isin(vc_subject_ids)]
        if len(affected) > 0:
            write_obs_dataframe(obs_meta_uri, affected, ctx=effective_ctx, columns={"obs_ids"})

        # Update group metadata
        new_subject_count = len(existing_subject_ids | vc_subject_ids)
        with tiledb.Group(uri, "w", ctx=effective_ctx) as grp:
            grp.meta["n_collections"] = self.n_collections
            grp.meta["subject_count"] = new_subject_count

    def validate(self) -> None:
        """Validate internal consistency of the RadiObject and all collections."""
        self._check_not_view("validate")
        obs_meta_data = self.obs_meta.read()
        actual_subject_count = len(obs_meta_data)
        stored_subject_count = self._metadata.get("subject_count", 0)
        if actual_subject_count != stored_subject_count:
            raise AlignmentError(
                f"subject_count mismatch: metadata={stored_subject_count}, actual={actual_subject_count}"
            )

        actual_n_collections = len(self.collection_names)
        stored_n_collections = self._metadata.get("n_collections", 0)
        if actual_n_collections != stored_n_collections:
            raise AlignmentError(
                f"n_collections mismatch: metadata={stored_n_collections}, actual={actual_n_collections}"
            )

        for name in self.collection_names:
            self.collection(name).validate()

        obs_meta_subject_ids = set(obs_meta_data["obs_subject_id"])
        for name in self.collection_names:
            vc = self.collection(name)
            vc_obs = vc.obs.read()
            vc_subject_ids = set(vc_obs["obs_subject_id"])
            orphan_subjects = vc_subject_ids - obs_meta_subject_ids
            if orphan_subjects:
                raise AlignmentError(
                    f"Collection '{name}' has obs_subject_ids not in obs_meta: "
                    f"{sorted(orphan_subjects)[:5]}"
                )

        seen_obs_ids: dict[str, str] = {}
        for name in self.collection_names:
            vc = self.collection(name)
            for obs_id in vc.obs_ids:
                if obs_id in seen_obs_ids:
                    raise AlignmentError(
                        f"obs_id '{obs_id}' is duplicated across collections: "
                        f"'{seen_obs_ids[obs_id]}' and '{name}'"
                    )
                seen_obs_ids[obs_id] = name

        # Validate obs_ids column matches actual collections
        if "obs_ids" in obs_meta_data.columns:
            expected = build_obs_ids_mapping(
                [self.collection(name) for name in self.collection_names]
            )
            for _, row in expected.iterrows():
                sid = row["obs_subject_id"]
                expected_ids = json.loads(row["obs_ids"])
                stored_row = obs_meta_data[obs_meta_data["obs_subject_id"] == sid]
                if len(stored_row) == 0:
                    continue
                stored_ids = json.loads(stored_row.iloc[0]["obs_ids"])
                if sorted(expected_ids) != sorted(stored_ids):
                    raise AlignmentError(
                        f"obs_ids mismatch for subject '{sid}': "
                        f"stored={stored_ids}, expected={expected_ids}"
                    )

    @property
    def _root(self) -> RadiObject:
        """The original attached RadiObject (follows source chain)."""
        if self._source is None:
            return self
        return self._source._root

    @cached_property
    def _metadata(self) -> dict:
        """Cached group metadata."""
        uri = self._effective_uri()
        with tiledb.Group(uri, "r", ctx=self._effective_ctx()) as grp:
            return dict(grp.meta)

    @cached_property
    def _index(self) -> Index:
        """Cached bidirectional index for obs_subject_id lookups."""
        if self.is_view:
            if self._subject_ids is not None:
                return self._root._index.intersection(self._subject_ids)
            return self._root._index
        n = self._metadata.get("subject_count", 0)
        if n == 0:
            return Index.build([], name="obs_subject_id")
        data = self.obs_meta.read(columns=["obs_subject_id"])
        return Index.build(list(data["obs_subject_id"]), name="obs_subject_id")

    def _effective_ctx(self) -> tiledb.Ctx:
        if self._source is not None:
            return self._source._effective_ctx()
        return self._ctx if self._ctx else get_tiledb_ctx()

    def _effective_uri(self) -> str:
        """Get the storage URI (from root if this is a view)."""
        if self._source is not None:
            return self._source._effective_uri()
        if self._uri is None:
            raise ValueError("RadiObject has no URI")
        return self._uri

    def _create_view(
        self,
        subject_ids: frozenset[str] | None = None,
        collection_names: frozenset[str] | None = None,
    ) -> RadiObject:
        """Create a view with specified filters, intersecting with current filters."""
        # Intersect subject_ids with current filter
        if self._subject_ids is not None and subject_ids is not None:
            subject_ids = self._subject_ids & subject_ids
        elif self._subject_ids is not None:
            subject_ids = self._subject_ids
        # subject_ids stays as passed if self._subject_ids is None

        # Intersect collection_names with current filter
        if self._collection_names_filter is not None and collection_names is not None:
            collection_names = self._collection_names_filter & collection_names
        elif self._collection_names_filter is not None:
            collection_names = self._collection_names_filter
        # collection_names stays as passed if self._collection_names_filter is None

        return RadiObject(
            uri=None,
            ctx=self._ctx,
            _source=self._root,  # Always point to root to avoid deep chains
            _subject_ids=subject_ids,
            _collection_names=collection_names,
        )

    def _check_not_view(self, operation: str) -> None:
        """Raise if attempting to modify a view."""
        if self.is_view:
            raise ViewError(
                f"Cannot {operation} on a view. Call write(uri) first to create "
                "an attached RadiObject."
            )

    def _filter_by_indices(self, indices: list[int]) -> RadiObject:
        """Create a view filtered to specific subject indices."""
        subject_ids = self._index.take(indices).to_set()
        return self._create_view(subject_ids=subject_ids)

    def _filter_by_subject_ids(self, obs_subject_ids: list[str]) -> RadiObject:
        """Create a view filtered to specific obs_subject_ids."""
        for sid in obs_subject_ids:
            if sid not in self._index:
                raise KeyError(f"obs_subject_id '{sid}' not found")
        return self._create_view(subject_ids=frozenset(obs_subject_ids))

    def _write_streaming(
        self,
        uri: str,
        obs_meta_df: pd.DataFrame,
        obs_meta_schema: dict[str, np.dtype],
        ctx: tiledb.Ctx | None,
    ) -> RadiObject:
        """Write view to storage using streaming writer."""
        from radiobject.writers import RadiObjectWriter

        subject_ids = set(obs_meta_df["obs_subject_id"])

        # Drop source obs_ids — will recompute from written collections
        meta_to_write = obs_meta_df.drop(columns=["obs_ids"], errors="ignore")

        # Collect obs_id→subject mappings for obs_ids recomputation
        written_pairs: list[tuple[str, str]] = []

        with RadiObjectWriter(uri, obs_meta_schema=obs_meta_schema, ctx=ctx) as writer:
            # Write collections first to collect obs_id pairs
            for coll_name in self.collection_names:
                src_collection = self.collection(coll_name)
                obs_df = src_collection.obs.read()
                filtered_obs = obs_df[obs_df["obs_subject_id"].isin(subject_ids)]

                if len(filtered_obs) == 0:
                    continue

                with writer.add_collection(
                    coll_name, src_collection.shape, _extract_obs_schema(src_collection.obs)
                ) as coll_writer:
                    for _, row in filtered_obs.iterrows():
                        obs_id = row["obs_id"]
                        vol = src_collection.loc[obs_id]
                        attrs = {
                            k: v for k, v in row.items() if k not in ("obs_id", "obs_subject_id")
                        }
                        coll_writer.write_volume(
                            data=vol.to_numpy(),
                            obs_id=obs_id,
                            obs_subject_id=row["obs_subject_id"],
                            **attrs,
                        )
                        written_pairs.append((row["obs_subject_id"], obs_id))

            # Compute obs_ids from written volumes and merge into obs_meta
            if written_pairs:
                pairs_df = pd.DataFrame(written_pairs, columns=["obs_subject_id", "obs_id"])
                obs_ids_map = _aggregate_obs_ids(pairs_df)
                meta_to_write = meta_to_write.merge(obs_ids_map, on="obs_subject_id", how="left")
                meta_to_write["obs_ids"] = meta_to_write["obs_ids"].fillna("[]")

            writer.write_obs_meta(meta_to_write)

        return RadiObject(uri, ctx=ctx)

    @classmethod
    def from_images(
        cls,
        uri: str,
        images: dict[str, str | Path | Sequence[tuple[str | Path, str]]],
        validate_alignment: bool = False,
        obs_meta: pd.DataFrame | None = None,
        reorient: bool | None = None,
        format_hint: dict[str, str] | None = None,
        ctx: tiledb.Ctx | None = None,
        progress: bool = False,
    ) -> RadiObject:
        """Create RadiObject from NIfTI or DICOM images with auto-format detection.

        Each collection's format is auto-detected (or set via format_hint).

        Args:
            uri: Target URI for RadiObject.
            images: Dict mapping collection names to image sources. Sources can be
                a glob pattern, directory path, or pre-resolved list of (path, subject_id).
            validate_alignment: If True, verify all collections have same subject IDs.
            obs_meta: Subject-level metadata keyed by obs_subject_id (one row per subject).
            reorient: Reorient to canonical orientation (None uses config default).
            format_hint: Dict mapping collection names to format strings ("nifti" or "dicom").
            ctx: TileDB context.
            progress: Show tqdm progress bar.

        Examples:
            NIfTI with glob patterns:

                radi = RadiObject.from_images(
                    uri="./dataset",
                    images={
                        "CT": "./imagesTr/*.nii.gz",
                        "seg": "./labelsTr/*.nii.gz",
                    },
                )

            DICOM with pre-resolved tuples:

                radi = RadiObject.from_images(
                    uri="./ct_study",
                    images={
                        "CT_head": [(Path("/dicom/sub01/head"), "sub-01")],
                    },
                )

            Mixed format with explicit hints:

                radi = RadiObject.from_images(
                    uri="./study",
                    images={"CT": "/dicom_dir/", "seg": "/labels/*.nii.gz"},
                    format_hint={"CT": "dicom"},
                )
        """
        from radiobject.ingest import ImageFormat, resolve_image_source

        if not images:
            raise ValueError("images dict cannot be empty")

        hint_map = format_hint or {}
        groups: dict[str, tuple[list[tuple[Path, str]], ImageFormat]] = {}
        for coll_name, source in images.items():
            hint = ImageFormat(hint_map[coll_name]) if coll_name in hint_map else None
            items, fmt = resolve_image_source(source, format_hint=hint)
            groups[coll_name] = (items, fmt)

        # Optional alignment validation
        if validate_alignment and len(groups) > 1:
            subject_sets = {name: {sid for _, sid in items} for name, (items, _) in groups.items()}
            first_name, first_set = next(iter(subject_sets.items()))
            for name, sid_set in subject_sets.items():
                if sid_set != first_set:
                    missing_in_first = sid_set - first_set
                    missing_in_other = first_set - sid_set
                    raise ValueError(
                        f"Subject ID mismatch between '{first_name}' and '{name}': "
                        f"missing in '{first_name}': {sorted(missing_in_first)[:3]}, "
                        f"missing in '{name}': {sorted(missing_in_other)[:3]}"
                    )

        # Validate files/directories exist
        for coll_name, (item_list, fmt) in groups.items():
            for path, _ in item_list:
                if not path.exists():
                    if fmt == ImageFormat.DICOM:
                        raise FileNotFoundError(f"DICOM directory not found: {path}")
                    else:
                        raise FileNotFoundError(f"NIfTI file not found: {path}")

        # Collect all subject IDs
        all_subject_ids: set[str] = set()
        for item_list, _ in groups.values():
            all_subject_ids.update(sid for _, sid in item_list)

        if obs_meta is not None:
            ensure_obs_columns(obs_meta, context="RadiObject.from_images")
            obs_meta_subject_ids = set(obs_meta["obs_subject_id"])
            missing = all_subject_ids - obs_meta_subject_ids
            if missing:
                raise ValueError(
                    f"obs_subject_ids in images not found in obs_meta: {sorted(missing)[:5]}"
                )
        else:
            sorted_ids = sorted(all_subject_ids)
            obs_meta = pd.DataFrame({"obs_subject_id": sorted_ids})

        if all(len(items) == 0 for items, _ in groups.values()):
            raise ValueError("No image files found")

        effective_ctx = ctx if ctx else get_tiledb_ctx()

        tiledb.Group.create(uri, ctx=effective_ctx)
        collections_uri = f"{uri}/collections"
        tiledb.Group.create(collections_uri, ctx=effective_ctx)

        collections: dict[str, VolumeCollection] = {}

        groups_iter = list(groups.items())
        if progress:
            from tqdm.auto import tqdm

            groups_iter = tqdm(groups_iter, desc="Collections", unit="coll")

        for coll_name, (items, fmt) in groups_iter:
            vc_uri = f"{collections_uri}/{coll_name}"
            if fmt == ImageFormat.NIFTI:
                vc = VolumeCollection.from_niftis(
                    uri=vc_uri,
                    niftis=list(items),
                    reorient=reorient,
                    validate_dimensions=False,
                    name=coll_name,
                    ctx=ctx,
                    progress=progress,
                )
            else:
                vc = VolumeCollection.from_dicoms(
                    uri=vc_uri,
                    dicom_dirs=list(items),
                    reorient=reorient,
                    validate_dimensions=True,
                    name=coll_name,
                    ctx=ctx,
                    progress=progress,
                )
            collections[coll_name] = vc

            with tiledb.Group(collections_uri, "w", ctx=effective_ctx) as grp:
                grp.add(vc_uri, name=coll_name)

        obs_meta = merge_obs_ids(obs_meta, collections)
        create_and_write_obs_meta(uri, obs_meta, ctx=ctx)

        with tiledb.Group(uri, "w", ctx=effective_ctx) as grp:
            grp.meta["n_collections"] = len(collections)
            grp.meta["subject_count"] = len(obs_meta)
            grp.add(f"{uri}/obs_meta", name="obs_meta")
            grp.add(collections_uri, name="collections")

        return cls(uri, ctx=ctx)

    @classmethod
    def from_collections(
        cls,
        uri: str,
        collections: dict[str, VolumeCollection | str],
        obs_meta: pd.DataFrame | None = None,
        ctx: tiledb.Ctx | None = None,
    ) -> RadiObject:
        """Create RadiObject from existing VolumeCollections.

        Links collections without copying when they're already at expected URIs
        ({uri}/collections/{name}). Copies collections that are elsewhere.

        Args:
            uri: Target URI for RadiObject.
            collections: Dict mapping collection names to VolumeCollection objects or URIs.
            obs_meta: Subject-level metadata keyed by obs_subject_id (one row per subject).
                If None, derived from collections.
            ctx: TileDB context.

        Examples:
            Collections already at expected locations (no copy):

                ct_vc = radi.CT.map(transform).write(uri=f"{URI}/collections/CT")
                seg_vc = radi.seg.map(transform).write(uri=f"{URI}/collections/seg")
                radi = RadiObject.from_collections(
                    uri=URI,
                    collections={"CT": ct_vc, "seg": seg_vc},
                )

            Collections from elsewhere (will be copied):

                radi = RadiObject.from_collections(
                    uri="./new_dataset",
                    collections={"T1w": existing_t1w_collection},
                )
        """
        if not collections:
            raise ValueError("At least one collection is required")

        effective_ctx = ctx if ctx else get_tiledb_ctx()
        collections_uri = f"{uri}/collections"

        # Resolve string URIs to VolumeCollection objects
        resolved: dict[str, VolumeCollection] = {}
        for name, vc_or_uri in collections.items():
            if isinstance(vc_or_uri, str):
                resolved[name] = VolumeCollection(vc_or_uri, ctx=ctx)
            else:
                resolved[name] = vc_or_uri

        # Determine which collections need copying vs linking
        in_place: dict[str, VolumeCollection] = {}
        to_copy: dict[str, VolumeCollection] = {}

        for name, vc in resolved.items():
            expected_uri = f"{collections_uri}/{name}"
            if vc.uri == expected_uri:
                in_place[name] = vc
            else:
                to_copy[name] = vc

        # Check if collections group already exists (from write)
        collections_group_exists = tiledb.object_type(collections_uri, ctx=effective_ctx) == "group"

        # Create root group (may already exist as directory from write)
        if tiledb.object_type(uri, ctx=effective_ctx) != "group":
            tiledb.Group.create(uri, ctx=effective_ctx)

        # Create or use existing collections group
        if not collections_group_exists:
            tiledb.Group.create(collections_uri, ctx=effective_ctx)

        with tiledb.Group(uri, "w", ctx=effective_ctx) as grp:
            grp.add(collections_uri, name="collections")

        # Link in-place collections (no copy needed)
        with tiledb.Group(collections_uri, "w", ctx=effective_ctx) as grp:
            for name, vc in in_place.items():
                grp.add(vc.uri, name=name)

        # Copy external collections
        for name, vc in to_copy.items():
            new_uri = f"{collections_uri}/{name}"
            _copy_volume_collection(vc, new_uri, name=name, ctx=ctx)
            with tiledb.Group(collections_uri, "w", ctx=effective_ctx) as grp:
                grp.add(new_uri, name=name)

        # Derive obs_meta if not provided
        if obs_meta is not None:
            ensure_obs_columns(obs_meta, context="RadiObject.from_collections")
        else:
            all_subject_ids: set[str] = set()
            for vc in resolved.values():
                obs_df = vc.obs.read()
                all_subject_ids.update(obs_df["obs_subject_id"].tolist())
            sorted_ids = sorted(all_subject_ids)
            obs_meta = pd.DataFrame({"obs_subject_id": sorted_ids})

        obs_meta = merge_obs_ids(obs_meta, resolved)
        create_and_write_obs_meta(uri, obs_meta, ctx=ctx)

        # Link obs_meta to root and set metadata
        with tiledb.Group(uri, "w", ctx=effective_ctx) as grp:
            grp.add(f"{uri}/obs_meta", name="obs_meta")
            grp.meta["n_collections"] = len(resolved)
            grp.meta["subject_count"] = len(obs_meta)

        return cls(uri, ctx=ctx)

    @classmethod
    def _create(
        cls,
        uri: str,
        obs_meta_schema: dict[str, np.dtype] | None = None,
        n_subjects: int = 0,
        ctx: tiledb.Ctx | None = None,
    ) -> RadiObject:
        """Internal: create an empty RadiObject with optional obs_meta schema."""
        effective_ctx = ctx if ctx else get_tiledb_ctx()

        tiledb.Group.create(uri, ctx=effective_ctx)

        obs_meta_uri = f"{uri}/obs_meta"
        Dataframe.create(
            obs_meta_uri,
            schema=obs_meta_schema or {},
            ctx=ctx,
            index_columns=("obs_subject_id",),
        )

        collections_uri = f"{uri}/collections"
        tiledb.Group.create(collections_uri, ctx=effective_ctx)

        with tiledb.Group(uri, "w", ctx=effective_ctx) as grp:
            grp.meta["subject_count"] = n_subjects
            grp.meta["n_collections"] = 0
            grp.add(obs_meta_uri, name="obs_meta")
            grp.add(collections_uri, name="collections")

        return cls(uri, ctx=ctx)

    @classmethod
    def _from_volume_collections(
        cls,
        uri: str,
        collections: dict[str, VolumeCollection],
        obs_meta: pd.DataFrame | None = None,
        ctx: tiledb.Ctx | None = None,
    ) -> RadiObject:
        """Internal: create RadiObject from existing VolumeCollections."""
        if not collections:
            raise ValueError("At least one VolumeCollection is required")

        effective_ctx = ctx if ctx else get_tiledb_ctx()

        if obs_meta is not None:
            obs_meta = merge_obs_ids(obs_meta, collections)

        n_subjects = len(obs_meta) if obs_meta is not None else 0
        obs_meta_schema = build_obs_meta_schema(obs_meta) if obs_meta is not None else None

        cls._create(uri, obs_meta_schema=obs_meta_schema, n_subjects=n_subjects, ctx=ctx)

        if obs_meta is not None and len(obs_meta) > 0:
            write_obs_dataframe(f"{uri}/obs_meta", obs_meta, ctx=effective_ctx)

        collections_uri = f"{uri}/collections"
        for coll_name, vc in collections.items():
            new_vc_uri = f"{collections_uri}/{coll_name}"
            _copy_volume_collection(vc, new_vc_uri, name=coll_name, ctx=ctx)

            with tiledb.Group(collections_uri, "w", ctx=effective_ctx) as grp:
                grp.add(new_vc_uri, name=coll_name)

        with tiledb.Group(uri, "w", ctx=effective_ctx) as grp:
            grp.meta["n_collections"] = len(collections)
            grp.meta["subject_count"] = n_subjects

        radi_result = cls(uri, ctx=ctx)
        return radi_result


def _extract_obs_schema(obs: Dataframe) -> dict[str, np.dtype]:
    """Extract schema from an obs Dataframe (excluding obs_id and obs_subject_id)."""
    schema = {}
    for col in obs.columns:
        if col in ("obs_id", "obs_subject_id"):
            continue
        schema[col] = obs.dtypes[col]
    return schema


def _copy_volume_collection(
    src: VolumeCollection,
    dst_uri: str,
    name: str | None = None,
    obs_subject_ids: list[str] | None = None,
    ctx: tiledb.Ctx | None = None,
) -> None:
    """Copy a VolumeCollection to a new URI, optionally filtering by subject IDs."""
    effective_ctx = ctx if ctx else get_tiledb_ctx()

    collection_name = name if name is not None else src.name

    obs_df = src.obs.read()
    if obs_subject_ids is not None:
        subject_id_set = set(obs_subject_ids)
        obs_df = obs_df[obs_df["obs_subject_id"].isin(subject_id_set)].reset_index(drop=True)
        if len(obs_df) == 0:
            raise ValueError("No volumes match the specified obs_subject_ids")

    VolumeCollection._create(
        dst_uri,
        shape=src.shape,
        obs_schema=_extract_obs_schema(src.obs),
        n_volumes=len(obs_df),
        name=collection_name,
        ctx=ctx,
    )

    write_obs_dataframe(f"{dst_uri}/obs", obs_df, ctx=effective_ctx)

    if obs_subject_ids is not None:
        selected_obs_ids = set(obs_df["obs_id"])
        selected_indices = [i for i, oid in enumerate(src.obs_ids) if oid in selected_obs_ids]

        def write_volume(args: tuple[int, int, str]) -> WriteResult:
            new_idx, orig_idx, obs_id = args
            worker_ctx = ctx_for_threads(ctx)
            new_vol_uri = f"{dst_uri}/volumes/{new_idx}"
            try:
                vol = src.iloc[orig_idx]
                data = vol.to_numpy()
                new_vol = Volume.from_numpy(new_vol_uri, data, ctx=worker_ctx)
                new_vol.set_obs_id(obs_id)
                return WriteResult(new_idx, new_vol_uri, obs_id, success=True)
            except Exception as e:
                return WriteResult(new_idx, new_vol_uri, obs_id, success=False, error=e)

        write_args = [
            (new_idx, orig_idx, src.obs_ids[orig_idx])
            for new_idx, orig_idx in enumerate(selected_indices)
        ]
    else:

        def write_volume(args: tuple[int, str, Volume]) -> WriteResult:
            idx, obs_id, vol = args
            worker_ctx = ctx_for_threads(ctx)
            new_vol_uri = f"{dst_uri}/volumes/{idx}"
            try:
                data = vol.to_numpy()
                new_vol = Volume.from_numpy(new_vol_uri, data, ctx=worker_ctx)
                new_vol.set_obs_id(obs_id)
                return WriteResult(idx, new_vol_uri, obs_id, success=True)
            except Exception as e:
                return WriteResult(idx, new_vol_uri, obs_id, success=False, error=e)

        write_args = [(idx, obs_id, src.iloc[idx]) for idx, obs_id in enumerate(src.obs_ids)]

    results = _write_volumes_parallel(
        write_volume, write_args, progress=False, desc="Copying volumes"
    )

    with tiledb.Group(f"{dst_uri}/volumes", "w", ctx=effective_ctx) as vol_grp:
        for result in results:
            vol_grp.add(result.uri, name=str(result.index))
