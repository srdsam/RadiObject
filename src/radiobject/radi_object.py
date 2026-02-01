"""RadiObject - top-level container for multi-collection radiology data."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Sequence, overload

import numpy as np
import numpy.typing as npt
import pandas as pd
import tiledb

from radiobject.ctx import ctx as global_ctx
from radiobject.dataframe import Dataframe
from radiobject.imaging_metadata import (
    extract_dicom_metadata,
    extract_nifti_metadata,
    infer_series_type,
)
from radiobject.indexing import Index
from radiobject.parallel import WriteResult, create_worker_ctx
from radiobject.volume import Volume
from radiobject.volume_collection import (
    VolumeCollection,
    _normalize_index,
    _write_volumes_parallel,
)

if TYPE_CHECKING:
    from radiobject.query import Query


class _SubjectILocIndexer:
    """Integer-location based indexer for RadiObject subjects."""

    def __init__(self, radi_object: RadiObject):
        self._radi_object = radi_object

    @overload
    def __getitem__(self, key: int) -> RadiObjectView: ...
    @overload
    def __getitem__(self, key: slice) -> RadiObjectView: ...
    @overload
    def __getitem__(self, key: list[int]) -> RadiObjectView: ...
    @overload
    def __getitem__(self, key: npt.NDArray[np.bool_]) -> RadiObjectView: ...

    def __getitem__(self, key: int | slice | list[int] | npt.NDArray[np.bool_]) -> RadiObjectView:
        """Returns a RadiObjectView filtered to selected subject indices."""
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
    def __getitem__(self, key: str) -> RadiObjectView: ...
    @overload
    def __getitem__(self, key: list[str]) -> RadiObjectView: ...

    def __getitem__(self, key: str | list[str]) -> RadiObjectView:
        """Returns a RadiObjectView filtered to selected obs_subject_ids."""
        if isinstance(key, str):
            return self._radi_object._filter_by_subject_ids([key])
        elif isinstance(key, list):
            return self._radi_object._filter_by_subject_ids(key)
        raise TypeError(f"loc indices must be str or list[str], got {type(key)}")


class RadiObject:
    """Top-level container for multi-collection radiology data with subject metadata."""

    def __init__(self, uri: str, ctx: tiledb.Ctx | None = None):
        self.uri: str = uri
        self._ctx: tiledb.Ctx | None = ctx

    def _effective_ctx(self) -> tiledb.Ctx:
        return self._ctx if self._ctx else global_ctx()

    # ===== Subject Indexing =====

    @cached_property
    def iloc(self) -> _SubjectILocIndexer:
        """Integer-location based indexing for selecting subjects by position."""
        return _SubjectILocIndexer(self)

    @cached_property
    def loc(self) -> _SubjectLocIndexer:
        """Label-based indexing for selecting subjects by obs_subject_id."""
        return _SubjectLocIndexer(self)

    # ===== ObsMeta (Subject Metadata) =====

    @property
    def obs_meta(self) -> Dataframe:
        """Subject-level observational metadata."""
        obs_meta_uri = f"{self.uri}/obs_meta"
        return Dataframe(uri=obs_meta_uri, ctx=self._ctx)

    @cached_property
    def _index(self) -> Index:
        """Cached bidirectional index for obs_subject_id lookups."""
        n = self._metadata.get("subject_count", 0)
        if n == 0:
            return Index.build([])
        # Only load the index column for efficiency
        data = self.obs_meta.read(columns=["obs_subject_id"])
        return Index.build(list(data["obs_subject_id"]))

    @property
    def index(self) -> Index:
        """Subject index for bidirectional ID/position lookups."""
        return self._index

    @property
    def obs_subject_ids(self) -> list[str]:
        """All obs_subject_id values in index order."""
        return list(self._index.keys)

    def get_obs_row_by_obs_subject_id(self, obs_subject_id: str) -> pd.DataFrame:
        """Get obs_meta row by obs_subject_id string identifier."""
        df = self.obs_meta.read()
        filtered = df[df["obs_subject_id"] == obs_subject_id].reset_index(drop=True)
        return filtered

    # ===== Volume Access Across Collections =====

    @cached_property
    def all_obs_ids(self) -> list[str]:
        """All obs_ids across all collections (for uniqueness checks)."""
        obs_ids = []
        for name in self.collection_names:
            obs_ids.extend(self.collection(name).obs_ids)
        return obs_ids

    def get_volume(self, obs_id: str) -> Volume:
        """Get a volume by obs_id from any collection.

        obs_id must be unique across the entire RadiObject.

        Args:
            obs_id: Unique volume identifier

        Returns:
            Volume object

        Raises:
            KeyError: If obs_id not found in any collection
        """
        for name in self.collection_names:
            coll = self.collection(name)
            if obs_id in coll.index:
                return coll.loc[obs_id]
        raise KeyError(f"obs_id '{obs_id}' not found in any collection")

    # ===== VolumeCollections =====

    @cached_property
    def _metadata(self) -> dict:
        """Cached group metadata."""
        with tiledb.Group(self.uri, "r", ctx=self._effective_ctx()) as grp:
            return dict(grp.meta)

    @cached_property
    def collection_names(self) -> tuple[str, ...]:
        """Names of all VolumeCollections."""
        collections_uri = f"{self.uri}/collections"
        with tiledb.Group(collections_uri, "r", ctx=self._effective_ctx()) as grp:
            return tuple(obj.name for obj in grp)

    def collection(self, name: str) -> VolumeCollection:
        """Get a VolumeCollection by name."""
        if name not in self.collection_names:
            raise KeyError(f"Collection '{name}' not found. Available: {self.collection_names}")
        collection_uri = f"{self.uri}/collections/{name}"
        return VolumeCollection(collection_uri, ctx=self._ctx)

    def __getattr__(self, name: str) -> VolumeCollection:
        """Attribute access to collections (e.g., radi.T1w)."""
        if name.startswith("_"):
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")
        try:
            return self.collection(name)
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' has no collection '{name}'")

    def rename_collection(self, old_name: str, new_name: str) -> None:
        """Rename a collection.

        Args:
            old_name: Current collection name
            new_name: New collection name

        Raises:
            KeyError: If old_name doesn't exist
            ValueError: If new_name already exists
        """
        if old_name not in self.collection_names:
            raise KeyError(f"Collection '{old_name}' not found")
        if new_name in self.collection_names:
            raise ValueError(f"Collection '{new_name}' already exists")

        effective_ctx = self._effective_ctx()
        collections_uri = f"{self.uri}/collections"
        old_uri = f"{collections_uri}/{old_name}"

        # Update collection's internal name metadata
        with tiledb.Group(old_uri, "w", ctx=effective_ctx) as grp:
            grp.meta["name"] = new_name

        # Update the group member name
        with tiledb.Group(collections_uri, "w", ctx=effective_ctx) as grp:
            grp.remove(old_name)
            grp.add(old_uri, name=new_name)

        # Invalidate cached property
        if "collection_names" in self.__dict__:
            del self.__dict__["collection_names"]

    # ===== Length / Iteration =====

    def __len__(self) -> int:
        """Number of subjects."""
        return int(self._metadata.get("subject_count", 0))

    @property
    def n_collections(self) -> int:
        """Number of VolumeCollections."""
        return len(self.collection_names)

    def __iter__(self) -> Iterator[str]:
        """Iterate over collection names."""
        return iter(self.collection_names)

    @overload
    def __getitem__(self, key: str) -> RadiObjectView: ...
    @overload
    def __getitem__(self, key: list[str]) -> RadiObjectView: ...

    def __getitem__(self, key: str | list[str]) -> RadiObjectView:
        """Bracket indexing for subjects by obs_subject_id.

        Alias for .loc[] - allows `radi["BraTS001"]` as shorthand for `radi.loc["BraTS001"]`.

        Args:
            key: Single obs_subject_id or list of obs_subject_ids

        Returns:
            RadiObjectView filtered to matching subjects
        """
        return self.loc[key]

    def __repr__(self) -> str:
        """Concise representation of the RadiObject."""
        collections = ", ".join(self.collection_names) if self.collection_names else "none"
        return (
            f"RadiObject({len(self)} subjects, {self.n_collections} collections: [{collections}])"
        )

    def describe(self) -> str:
        """Return a summary: subjects, collections, shapes, and label distributions."""
        lines = [
            "RadiObject Summary",
            "==================",
            f"URI: {self.uri}",
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

        # Find label columns (non-string columns that aren't IDs)
        obs_meta = self.obs_meta.read()
        label_cols = []
        for col in obs_meta.columns:
            if col in ("obs_subject_id", "obs_id"):
                continue
            dtype = obs_meta[col].dtype
            # Check if it looks like a label column (categorical or numeric with few values)
            if dtype in (np.int64, np.int32, np.float64, np.float32, object):
                n_unique = obs_meta[col].nunique()
                if n_unique <= 10:  # Likely a label column
                    label_cols.append(col)

        if label_cols:
            lines.append("")
            lines.append("Label Columns:")
            for col in label_cols:
                value_counts = obs_meta[col].value_counts().to_dict()
                lines.append(f"  - {col}: {value_counts}")

        return "\n".join(lines)

    # ===== Query Builder (Pipeline Mode) =====

    def query(self) -> Query:
        """Create a lazy Query builder for pipeline-style filtering."""
        from radiobject.query import Query

        return Query(self)

    # ===== Filtering (Interactive Mode - returns RadiObjectView) =====

    def _filter_by_indices(self, indices: list[int]) -> RadiObjectView:
        """Create a view filtered to specific subject indices."""
        subject_ids = [self._index.get_key(i) for i in indices]
        return RadiObjectView(
            source=self,
            obs_subject_ids=subject_ids,
            collection_names=list(self.collection_names),
        )

    def _filter_by_subject_ids(self, obs_subject_ids: list[str]) -> RadiObjectView:
        """Create a view filtered to specific obs_subject_ids."""
        for sid in obs_subject_ids:
            if sid not in self._index:
                raise KeyError(f"obs_subject_id '{sid}' not found")
        return RadiObjectView(
            source=self,
            obs_subject_ids=obs_subject_ids,
            collection_names=list(self.collection_names),
        )

    def select_collections(self, names: list[str]) -> RadiObjectView:
        """Create a view with only specified collections."""
        for name in names:
            if name not in self.collection_names:
                raise KeyError(f"Collection '{name}' not found")
        return RadiObjectView(
            source=self,
            obs_subject_ids=self.obs_subject_ids,
            collection_names=names,
        )

    def filter(self, expr: str) -> RadiObjectView:
        """Filter subjects using a query expression on obs_meta.

        Args:
            expr: TileDB QueryCondition string (e.g., "tumor_grade == 'HGG' and age > 40")

        Returns:
            RadiObjectView filtered to matching subjects
        """
        filtered = self.obs_meta.read(value_filter=expr)
        subject_ids = list(filtered["obs_subject_id"])
        return self._filter_by_subject_ids(subject_ids)

    def head(self, n: int = 5) -> RadiObjectView:
        """Return view of first n subjects."""
        n = min(n, len(self))
        return self._filter_by_indices(list(range(n)))

    def tail(self, n: int = 5) -> RadiObjectView:
        """Return view of last n subjects."""
        total = len(self)
        n = min(n, total)
        return self._filter_by_indices(list(range(total - n, total)))

    def sample(self, n: int = 5, seed: int | None = None) -> RadiObjectView:
        """Return view of n randomly sampled subjects.

        Args:
            n: Number of subjects to sample
            seed: Random seed for reproducibility
        """
        rng = np.random.default_rng(seed)
        total = len(self)
        n = min(n, total)
        indices = list(rng.choice(total, size=n, replace=False))
        return self._filter_by_indices(sorted(indices))

    # ===== Append Operations (Mutations) =====

    def append(
        self,
        niftis: Sequence[tuple[str | Path, str]] | None = None,
        dicom_dirs: Sequence[tuple[str | Path, str]] | None = None,
        obs_meta: pd.DataFrame | None = None,
        reorient: bool | None = None,
        progress: bool = False,
    ) -> None:
        """Append new subjects and their volumes atomically.

        All data is written together - obs_meta entries and volumes are added
        in a single operation to maintain consistency.

        Args:
            niftis: List of (nifti_path, obs_subject_id) tuples to append
            dicom_dirs: List of (dicom_dir, obs_subject_id) tuples to append
            obs_meta: Subject-level metadata for NEW subjects. Required if any
                      obs_subject_ids don't already exist. Must contain obs_subject_id column.
            reorient: Reorient to canonical orientation (None uses config default)
            progress: Show tqdm progress bar during volume writes

        Example:
            # Append new subjects with their scans
            radi.append(
                niftis=[
                    ("sub101_T1w.nii.gz", "sub-101"),
                    ("sub101_FLAIR.nii.gz", "sub-101"),
                    ("sub102_T1w.nii.gz", "sub-102"),
                ],
                obs_meta=pd.DataFrame({
                    "obs_subject_id": ["sub-101", "sub-102"],
                    "age": [45, 52],
                }),
            )

            # Append scans for existing subjects (no obs_meta needed)
            radi.append(
                niftis=[("sub001_PET.nii.gz", "sub-001")],  # sub-001 already exists
            )
        """
        if niftis is None and dicom_dirs is None:
            raise ValueError("Must provide either niftis or dicom_dirs")
        if niftis is not None and dicom_dirs is not None:
            raise ValueError("Cannot provide both niftis and dicom_dirs")

        effective_ctx = self._effective_ctx()

        # Collect all subject IDs from input
        if niftis is not None:
            input_subject_ids = {sid for _, sid in niftis}
        else:
            input_subject_ids = {sid for _, sid in dicom_dirs}

        existing_subject_ids = set(self.obs_subject_ids)
        new_subject_ids = input_subject_ids - existing_subject_ids

        # Validate obs_meta
        if new_subject_ids:
            if obs_meta is None:
                raise ValueError(
                    f"obs_meta required for new subjects: {sorted(new_subject_ids)[:5]}"
                )
            if "obs_subject_id" not in obs_meta.columns:
                raise ValueError("obs_meta must contain 'obs_subject_id' column")
            obs_meta_ids = set(obs_meta["obs_subject_id"])
            missing = new_subject_ids - obs_meta_ids
            if missing:
                raise ValueError(f"obs_meta missing entries for: {sorted(missing)[:5]}")
            # Filter obs_meta to only new subjects
            obs_meta = obs_meta[obs_meta["obs_subject_id"].isin(new_subject_ids)]

        # Append obs_meta for new subjects
        if obs_meta is not None and len(obs_meta) > 0:
            obs_meta_uri = f"{self.uri}/obs_meta"
            obs_subject_ids_arr = obs_meta["obs_subject_id"].astype(str).to_numpy()
            obs_ids_arr = (
                obs_meta["obs_id"].astype(str).to_numpy()
                if "obs_id" in obs_meta.columns
                else obs_subject_ids_arr
            )
            # Only write attributes that exist in the target schema
            existing_columns = set(self.obs_meta.columns)
            with tiledb.open(obs_meta_uri, "w", ctx=effective_ctx) as arr:
                attr_data = {
                    col: obs_meta[col].to_numpy()
                    for col in obs_meta.columns
                    if col not in ("obs_subject_id", "obs_id") and col in existing_columns
                }
                arr[obs_subject_ids_arr, obs_ids_arr] = attr_data

            # Update subject_count
            new_count = len(self) + len(obs_meta)
            with tiledb.Group(self.uri, "w", ctx=effective_ctx) as grp:
                grp.meta["subject_count"] = new_count

        # Process and group input files
        if niftis is not None:
            self._append_niftis(niftis, reorient, effective_ctx, progress)
        else:
            self._append_dicoms(dicom_dirs, reorient, effective_ctx, progress)

        # Invalidate cached properties
        for prop in ("_index", "_metadata", "collection_names"):
            if prop in self.__dict__:
                del self.__dict__[prop]

    def _append_niftis(
        self,
        niftis: Sequence[tuple[str | Path, str]],
        reorient: bool | None,
        effective_ctx: tiledb.Ctx,
        progress: bool = False,
    ) -> None:
        """Internal: append NIfTI files to existing collections or create new ones."""
        # Extract metadata and group by (shape, series_type)
        file_info: list[tuple[Path, str, tuple[int, int, int], str]] = []
        for nifti_path, obs_subject_id in niftis:
            path = Path(nifti_path)
            if not path.exists():
                raise FileNotFoundError(f"NIfTI file not found: {path}")
            metadata = extract_nifti_metadata(path)
            series_type = infer_series_type(path)
            file_info.append((path, obs_subject_id, metadata.dimensions, series_type))

        groups: dict[tuple[tuple[int, int, int], str], list[tuple[Path, str]]] = defaultdict(list)
        for path, subject_id, shape, series_type in file_info:
            groups[(shape, series_type)].append((path, subject_id))

        collections_uri = f"{self.uri}/collections"
        existing_collections = set(self.collection_names)

        groups_iter = groups.items()
        if progress:
            from tqdm.auto import tqdm

            groups_iter = tqdm(groups_iter, desc="Collections", unit="coll")

        for (shape, series_type), items in groups_iter:
            # Find or create collection
            collection_name = series_type
            if collection_name in existing_collections:
                # Append to existing collection
                vc = self.collection(collection_name)
                if vc.shape != shape:
                    collection_name = f"{series_type}_{shape[0]}x{shape[1]}x{shape[2]}"

            if collection_name in existing_collections:
                # Append to existing
                vc = self.collection(collection_name)
                vc.append(niftis=items, reorient=reorient, progress=progress)
            else:
                # Create new collection
                vc_uri = f"{collections_uri}/{collection_name}"
                VolumeCollection.from_niftis(
                    uri=vc_uri,
                    niftis=items,
                    reorient=reorient,
                    validate_dimensions=True,
                    name=collection_name,
                    ctx=self._ctx,
                    progress=progress,
                )
                with tiledb.Group(collections_uri, "w", ctx=effective_ctx) as grp:
                    grp.add(vc_uri, name=collection_name)
                with tiledb.Group(self.uri, "w", ctx=effective_ctx) as grp:
                    grp.meta["n_collections"] = self.n_collections + 1
                existing_collections.add(collection_name)

    def _append_dicoms(
        self,
        dicom_dirs: Sequence[tuple[str | Path, str]],
        reorient: bool | None,
        effective_ctx: tiledb.Ctx,
        progress: bool = False,
    ) -> None:
        """Internal: append DICOM series to existing collections or create new ones."""
        file_info: list[tuple[Path, str, tuple[int, int, int], str]] = []
        for dicom_dir, obs_subject_id in dicom_dirs:
            path = Path(dicom_dir)
            if not path.exists():
                raise FileNotFoundError(f"DICOM directory not found: {path}")
            metadata = extract_dicom_metadata(path)
            dims = metadata.dimensions
            shape = (dims[1], dims[0], dims[2])  # Swap to X, Y, Z
            file_info.append((path, obs_subject_id, shape, metadata.modality))

        groups: dict[tuple[tuple[int, int, int], str], list[tuple[Path, str]]] = defaultdict(list)
        for path, subject_id, shape, modality in file_info:
            groups[(shape, modality)].append((path, subject_id))

        collections_uri = f"{self.uri}/collections"
        existing_collections = set(self.collection_names)

        groups_iter = groups.items()
        if progress:
            from tqdm.auto import tqdm

            groups_iter = tqdm(groups_iter, desc="Collections", unit="coll")

        for (shape, modality), items in groups_iter:
            collection_name = modality
            if collection_name in existing_collections:
                vc = self.collection(collection_name)
                if vc.shape != shape:
                    collection_name = f"{modality}_{shape[0]}x{shape[1]}x{shape[2]}"

            if collection_name in existing_collections:
                vc = self.collection(collection_name)
                vc.append(dicom_dirs=items, reorient=reorient, progress=progress)
            else:
                vc_uri = f"{collections_uri}/{collection_name}"
                VolumeCollection.from_dicoms(
                    uri=vc_uri,
                    dicom_dirs=items,
                    reorient=reorient,
                    validate_dimensions=True,
                    name=collection_name,
                    ctx=self._ctx,
                    progress=progress,
                )
                with tiledb.Group(collections_uri, "w", ctx=effective_ctx) as grp:
                    grp.add(vc_uri, name=collection_name)
                with tiledb.Group(self.uri, "w", ctx=effective_ctx) as grp:
                    grp.meta["n_collections"] = self.n_collections + 1
                existing_collections.add(collection_name)

    # ===== Validation =====

    def validate(self) -> None:
        """Validate internal consistency of the RadiObject and all collections.

        Checks:
        1. subject_count metadata matches actual obs_meta rows
        2. n_collections metadata matches actual collection count
        3. Each VolumeCollection passes its own validate()
        4. FK constraint: all obs_subject_ids in each collection's obs exist in obs_meta
        5. obs_id uniqueness across all collections
        """
        obs_meta_data = self.obs_meta.read()
        actual_subject_count = len(obs_meta_data)
        stored_subject_count = self._metadata.get("subject_count", 0)
        if actual_subject_count != stored_subject_count:
            raise ValueError(
                f"subject_count mismatch: metadata={stored_subject_count}, actual={actual_subject_count}"
            )

        actual_n_collections = len(self.collection_names)
        stored_n_collections = self._metadata.get("n_collections", 0)
        if actual_n_collections != stored_n_collections:
            raise ValueError(
                f"n_collections mismatch: metadata={stored_n_collections}, actual={actual_n_collections}"
            )

        # Validate each collection
        for name in self.collection_names:
            self.collection(name).validate()

        # Validate FK constraint: obs_subject_ids in collections must exist in obs_meta
        obs_meta_subject_ids = set(obs_meta_data["obs_subject_id"])
        for name in self.collection_names:
            vc = self.collection(name)
            vc_obs = vc.obs.read()
            vc_subject_ids = set(vc_obs["obs_subject_id"])
            orphan_subjects = vc_subject_ids - obs_meta_subject_ids
            if orphan_subjects:
                raise ValueError(
                    f"Collection '{name}' has obs_subject_ids not in obs_meta: "
                    f"{sorted(orphan_subjects)[:5]}"
                )

        # Validate obs_id uniqueness across all collections
        seen_obs_ids: dict[str, str] = {}  # obs_id -> collection_name
        for name in self.collection_names:
            vc = self.collection(name)
            for obs_id in vc.obs_ids:
                if obs_id in seen_obs_ids:
                    raise ValueError(
                        f"obs_id '{obs_id}' is duplicated across collections: "
                        f"'{seen_obs_ids[obs_id]}' and '{name}'"
                    )
                seen_obs_ids[obs_id] = name

    # ===== Factory Methods =====

    @classmethod
    def _create(
        cls,
        uri: str,
        obs_meta_schema: dict[str, np.dtype] | None = None,
        n_subjects: int = 0,
        ctx: tiledb.Ctx | None = None,
    ) -> RadiObject:
        """Internal: create an empty RadiObject with optional obs_meta schema."""
        effective_ctx = ctx if ctx else global_ctx()

        tiledb.Group.create(uri, ctx=effective_ctx)

        obs_meta_uri = f"{uri}/obs_meta"
        Dataframe.create(obs_meta_uri, schema=obs_meta_schema or {}, ctx=ctx)

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

        effective_ctx = ctx if ctx else global_ctx()

        n_subjects = len(obs_meta) if obs_meta is not None else 0

        obs_meta_schema = None
        if obs_meta is not None:
            obs_meta_schema = {}
            for col in obs_meta.columns:
                if col in ("obs_subject_id", "obs_id"):
                    continue
                dtype = obs_meta[col].to_numpy().dtype
                if dtype == np.dtype("O"):
                    dtype = np.dtype("U64")
                obs_meta_schema[col] = dtype

        cls._create(uri, obs_meta_schema=obs_meta_schema, n_subjects=n_subjects, ctx=ctx)

        if obs_meta is not None and len(obs_meta) > 0:
            obs_meta_uri = f"{uri}/obs_meta"
            obs_subject_ids = obs_meta["obs_subject_id"].astype(str).to_numpy()
            obs_ids = (
                obs_meta["obs_id"].astype(str).to_numpy()
                if "obs_id" in obs_meta.columns
                else obs_subject_ids
            )
            with tiledb.open(obs_meta_uri, "w", ctx=effective_ctx) as arr:
                attr_data = {}
                for col in obs_meta.columns:
                    if col not in ("obs_subject_id", "obs_id"):
                        attr_data[col] = obs_meta[col].to_numpy()
                arr[obs_subject_ids, obs_ids] = attr_data

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

    @classmethod
    def from_niftis(
        cls,
        uri: str,
        niftis: Sequence[tuple[str | Path, str]] | None = None,
        image_dir: str | Path | None = None,
        collection_name: str | None = None,
        images: dict[str, str | Path | Sequence[tuple[str | Path, str]]] | None = None,
        validate_alignment: bool = False,
        obs_meta: pd.DataFrame | None = None,
        reorient: bool | None = None,
        ctx: tiledb.Ctx | None = None,
        progress: bool = False,
    ) -> RadiObject:
        """Create RadiObject from NIfTI files with raw data storage.

        Ingestion stores volumes in their original dimensions without any
        preprocessing. Use `collection.map()` for post-hoc transformations.

        Three input modes:
        1. images: Dict mapping collection names to paths/globs/lists (recommended)
        2. niftis: List of (path, subject_id) tuples (legacy)
        3. image_dir: Directory-based discovery (legacy)

        Collection organization:
        - With images dict: each key becomes a collection
        - With collection_name: all volumes go to that single collection
        - Otherwise: auto-group by inferred modality (T1w, FLAIR, CT, etc.)

        Args:
            uri: Target URI for RadiObject
            images: Dict mapping collection names to NIfTI sources. Sources can be:
                   - Glob pattern: "./imagesTr/*.nii.gz"
                   - Directory path: "./imagesTr"
                   - Pre-resolved list: [(path, subject_id), ...]
            niftis: List of (nifti_path, obs_subject_id) tuples (legacy)
            image_dir: Directory containing image NIfTIs (legacy, mutually exclusive with niftis)
            collection_name: Explicit name for collection (legacy, all volumes go here)
            validate_alignment: If True, verify all collections have same subject IDs
            obs_meta: Subject-level metadata. Must contain obs_subject_id column.
            reorient: Reorient to canonical orientation (None uses config default)
            ctx: TileDB context
            progress: Show tqdm progress bar

        Example (images dict with globs):
            radi = RadiObject.from_niftis(
                uri="./dataset",
                images={
                    "CT": "./imagesTr/*.nii.gz",
                    "seg": "./labelsTr/*.nii.gz",
                },
            )

        Example (images dict with directories):
            radi = RadiObject.from_niftis(
                uri="./dataset",
                images={"CT": "./imagesTr", "seg": "./labelsTr"},
            )

        Example (legacy explicit collection name):
            radi = RadiObject.from_niftis(
                uri="s3://bucket/raw",
                image_dir="./imagesTr",
                collection_name="lung_ct",
            )

        Example (legacy auto-group by modality):
            radi = RadiObject.from_niftis(
                uri="s3://bucket/raw",
                niftis=[
                    ("sub01_T1w.nii.gz", "sub-01"),
                    ("sub01_FLAIR.nii.gz", "sub-01"),
                ],
            )
            # Result: radi.T1w, radi.FLAIR collections
        """
        from radiobject.ingest import resolve_nifti_source

        # --- NORMALIZE ALL INPUTS TO images DICT ---

        if images is not None:
            # New mode: images dict provided directly
            if niftis is not None or image_dir is not None or collection_name is not None:
                raise ValueError("Cannot use 'images' with legacy parameters")
            if not images:
                raise ValueError("images dict cannot be empty")
            # images is ready to use

        elif image_dir is not None:
            # Legacy mode: image_dir → discover files and convert to images dict
            if niftis is not None:
                raise ValueError("Cannot specify both 'niftis' and 'image_dir'")
            from radiobject.ingest import discover_nifti_pairs

            sources = discover_nifti_pairs(image_dir)
            niftis = [(s.image_path, s.subject_id) for s in sources]

            if collection_name:
                # Explicit name - all to one collection
                images = {collection_name: niftis}
            else:
                # Group by inferred modality
                modality_groups: dict[str, list[tuple[str | Path, str]]] = defaultdict(list)
                for path, sid in niftis:
                    series_type = infer_series_type(Path(path))
                    modality_groups[series_type].append((path, sid))
                images = dict(modality_groups)

        elif niftis is not None:
            # Legacy mode: niftis list → convert to images dict
            if collection_name:
                # All to one collection
                images = {collection_name: niftis}
            else:
                # Group by inferred modality
                modality_groups: dict[str, list[tuple[str | Path, str]]] = defaultdict(list)
                for path, sid in niftis:
                    series_type = infer_series_type(Path(path))
                    modality_groups[series_type].append((path, sid))
                images = dict(modality_groups)
        else:
            raise ValueError("Must specify 'images', 'niftis', or 'image_dir'")

        # --- SINGLE CODE PATH: Resolve images dict ---

        groups: dict[str, list[tuple[Path, str]]] = {}
        for coll_name, source in images.items():
            groups[coll_name] = resolve_nifti_source(source)

        # Optional alignment validation
        if validate_alignment and len(groups) > 1:
            subject_sets = {
                name: {sid for _, sid in nifti_list} for name, nifti_list in groups.items()
            }
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

        # Validate all files exist
        for coll_name, nifti_list in groups.items():
            for path, _ in nifti_list:
                if not path.exists():
                    raise FileNotFoundError(f"NIfTI file not found: {path}")

        # Collect all subject IDs
        all_subject_ids: set[str] = set()
        for nifti_list in groups.values():
            all_subject_ids.update(sid for _, sid in nifti_list)

        # Validate FK constraint if obs_meta provided
        if obs_meta is not None:
            if "obs_subject_id" not in obs_meta.columns:
                raise ValueError("obs_meta must contain 'obs_subject_id' column")
            obs_meta_subject_ids = set(obs_meta["obs_subject_id"])
            missing = all_subject_ids - obs_meta_subject_ids
            if missing:
                raise ValueError(
                    f"obs_subject_ids in niftis not found in obs_meta: {sorted(missing)[:5]}"
                )
        else:
            # Auto-generate obs_meta from unique subject IDs
            sorted_ids = sorted(all_subject_ids)
            obs_meta = pd.DataFrame(
                {
                    "obs_subject_id": sorted_ids,
                    "obs_id": sorted_ids,
                }
            )

        # Check for empty groups
        if not groups or all(len(nifti_list) == 0 for nifti_list in groups.values()):
            raise ValueError("No NIfTI files found")

        # Create VolumeCollection for each group
        effective_ctx = ctx if ctx else global_ctx()

        # Ensure parent directories exist
        tiledb.Group.create(uri, ctx=effective_ctx)
        collections_uri = f"{uri}/collections"
        tiledb.Group.create(collections_uri, ctx=effective_ctx)

        collections: dict[str, VolumeCollection] = {}

        groups_iter = list(groups.items())
        if progress:
            from tqdm.auto import tqdm

            groups_iter = tqdm(groups_iter, desc="Collections", unit="coll")

        for coll_name, items in groups_iter:
            vc_uri = f"{collections_uri}/{coll_name}"
            nifti_list = [(path, subject_id) for path, subject_id in items]

            # Create collection without shape constraint (heterogeneous shapes allowed)
            vc = VolumeCollection.from_niftis(
                uri=vc_uri,
                niftis=nifti_list,
                reorient=reorient,
                validate_dimensions=False,  # Allow heterogeneous shapes
                name=coll_name,
                ctx=ctx,
                progress=progress,
            )
            collections[coll_name] = vc

            with tiledb.Group(collections_uri, "w", ctx=effective_ctx) as grp:
                grp.add(vc_uri, name=coll_name)

        # Create obs_meta Dataframe
        n_subjects = len(obs_meta)
        obs_meta_schema: dict[str, np.dtype] = {}
        for col in obs_meta.columns:
            if col in ("obs_subject_id", "obs_id"):
                continue
            dtype = obs_meta[col].to_numpy().dtype
            if dtype == np.dtype("O"):
                dtype = np.dtype("U64")
            obs_meta_schema[col] = dtype

        obs_meta_uri = f"{uri}/obs_meta"
        Dataframe.create(obs_meta_uri, schema=obs_meta_schema, ctx=ctx)

        # Write obs_meta data
        if len(obs_meta) > 0:
            obs_subject_ids = obs_meta["obs_subject_id"].astype(str).to_numpy()
            obs_ids = (
                obs_meta["obs_id"].astype(str).to_numpy()
                if "obs_id" in obs_meta.columns
                else obs_subject_ids
            )
            with tiledb.open(obs_meta_uri, "w", ctx=effective_ctx) as arr:
                attr_data = {}
                for col in obs_meta.columns:
                    if col not in ("obs_subject_id", "obs_id"):
                        attr_data[col] = obs_meta[col].to_numpy()
                arr[obs_subject_ids, obs_ids] = attr_data

        # Update group metadata
        with tiledb.Group(uri, "w", ctx=effective_ctx) as grp:
            grp.meta["n_collections"] = len(collections)
            grp.meta["subject_count"] = n_subjects
            grp.add(obs_meta_uri, name="obs_meta")
            grp.add(collections_uri, name="collections")

        return cls(uri, ctx=ctx)

    @classmethod
    def from_dicoms(
        cls,
        uri: str,
        dicom_dirs: Sequence[tuple[str | Path, str]],
        obs_meta: pd.DataFrame | None = None,
        reorient: bool | None = None,
        ctx: tiledb.Ctx | None = None,
        progress: bool = False,
    ) -> RadiObject:
        """Create RadiObject from DICOM series with automatic grouping.

        Files are automatically grouped into VolumeCollections by:
        1. Dimensions (rows, columns, n_slices)
        2. Modality tag (CT, MR) + SeriesDescription

        Args:
            uri: Target URI for RadiObject
            dicom_dirs: List of (dicom_dir, obs_subject_id) tuples
            obs_meta: Subject-level metadata (user-provided). Must contain obs_subject_id column.
            reorient: Reorient to canonical orientation (None uses config default)
            ctx: TileDB context
            progress: Show tqdm progress bar during volume writes

        Example:
            radi = RadiObject.from_dicoms(
                uri="/storage/ct_study",
                dicom_dirs=[
                    ("/dicom/sub01/CT_HEAD", "sub-01"),
                    ("/dicom/sub01/CT_CHEST", "sub-01"),
                    ("/dicom/sub02/CT_HEAD", "sub-02"),
                ],
                obs_meta=obs_meta_df,
            )
        """
        if not dicom_dirs:
            raise ValueError("At least one DICOM directory is required")

        # Collect all subject IDs
        all_subject_ids = {sid for _, sid in dicom_dirs}

        # Validate FK constraint if obs_meta provided
        if obs_meta is not None:
            if "obs_subject_id" not in obs_meta.columns:
                raise ValueError("obs_meta must contain 'obs_subject_id' column")
            obs_meta_subject_ids = set(obs_meta["obs_subject_id"])
            missing = all_subject_ids - obs_meta_subject_ids
            if missing:
                raise ValueError(
                    f"obs_subject_ids in dicom_dirs not found in obs_meta: {sorted(missing)[:5]}"
                )
        else:
            # Auto-generate obs_meta from unique subject IDs
            sorted_ids = sorted(all_subject_ids)
            obs_meta = pd.DataFrame(
                {
                    "obs_subject_id": sorted_ids,
                    "obs_id": sorted_ids,
                }
            )

        # Extract metadata for each DICOM series
        file_info: list[tuple[Path, str, tuple[int, int, int], str]] = []
        for dicom_dir, obs_subject_id in dicom_dirs:
            path = Path(dicom_dir)
            if not path.exists():
                raise FileNotFoundError(f"DICOM directory not found: {path}")

            metadata = extract_dicom_metadata(path)
            # DICOM dimensions tuple is (rows, columns, n_slices)
            # Swap to (columns, rows, n_slices) to match X, Y, Z convention
            dims = metadata.dimensions
            shape = (dims[1], dims[0], dims[2])
            # Use modality as group key (could also use series_description)
            group_key = metadata.modality
            file_info.append((path, obs_subject_id, shape, group_key))

        # Group by (shape, modality)
        groups: dict[tuple[tuple[int, int, int], str], list[tuple[Path, str]]] = defaultdict(list)
        for path, subject_id, shape, group_key in file_info:
            key = (shape, group_key)
            groups[key].append((path, subject_id))

        # Create VolumeCollection for each group
        effective_ctx = ctx if ctx else global_ctx()

        # Ensure parent directories exist
        tiledb.Group.create(uri, ctx=effective_ctx)
        collections_uri = f"{uri}/collections"
        tiledb.Group.create(collections_uri, ctx=effective_ctx)

        collections: dict[str, VolumeCollection] = {}
        used_names: set[str] = set()

        groups_iter = groups.items()
        if progress:
            from tqdm.auto import tqdm

            groups_iter = tqdm(groups_iter, desc="Collections", unit="coll")

        for (shape, modality), items in groups_iter:
            # Generate unique collection name
            collection_name = modality
            if collection_name in used_names:
                collection_name = f"{modality}_{shape[0]}x{shape[1]}x{shape[2]}"
            used_names.add(collection_name)

            vc_uri = f"{collections_uri}/{collection_name}"
            dicom_list = [(path, subject_id) for path, subject_id in items]

            vc = VolumeCollection.from_dicoms(
                uri=vc_uri,
                dicom_dirs=dicom_list,
                reorient=reorient,
                validate_dimensions=True,
                name=collection_name,
                ctx=ctx,
                progress=progress,
            )
            collections[collection_name] = vc

            # Register with group
            with tiledb.Group(collections_uri, "w", ctx=effective_ctx) as grp:
                grp.add(vc_uri, name=collection_name)

        # Create obs_meta Dataframe
        n_subjects = len(obs_meta)
        obs_meta_schema: dict[str, np.dtype] = {}
        for col in obs_meta.columns:
            if col in ("obs_subject_id", "obs_id"):
                continue
            dtype = obs_meta[col].to_numpy().dtype
            if dtype == np.dtype("O"):
                dtype = np.dtype("U64")
            obs_meta_schema[col] = dtype

        obs_meta_uri = f"{uri}/obs_meta"
        Dataframe.create(obs_meta_uri, schema=obs_meta_schema, ctx=ctx)

        # Write obs_meta data
        if len(obs_meta) > 0:
            obs_subject_ids = obs_meta["obs_subject_id"].astype(str).to_numpy()
            obs_ids = (
                obs_meta["obs_id"].astype(str).to_numpy()
                if "obs_id" in obs_meta.columns
                else obs_subject_ids
            )
            with tiledb.open(obs_meta_uri, "w", ctx=effective_ctx) as arr:
                attr_data = {}
                for col in obs_meta.columns:
                    if col not in ("obs_subject_id", "obs_id"):
                        attr_data[col] = obs_meta[col].to_numpy()
                arr[obs_subject_ids, obs_ids] = attr_data

        # Update group metadata
        with tiledb.Group(uri, "w", ctx=effective_ctx) as grp:
            grp.meta["n_collections"] = len(collections)
            grp.meta["subject_count"] = n_subjects
            grp.add(obs_meta_uri, name="obs_meta")
            grp.add(collections_uri, name="collections")

        return cls(uri, ctx=ctx)


class RadiObjectView:
    """Immutable view into a RadiObject with filtered subjects/collections.

    RadiObjectView is designed for interactive mode - quick inspection and
    pandas-like access patterns. For pipeline/ETL workloads, use `to_query()`
    to get a lazy Query builder.
    """

    def __init__(
        self,
        source: RadiObject,
        obs_subject_ids: list[str],
        collection_names: list[str],
    ):
        self._source = source
        self._obs_subject_ids = tuple(obs_subject_ids)
        self._collection_names = tuple(collection_names)

    @property
    def obs_subject_ids(self) -> list[str]:
        """Selected obs_subject_ids."""
        return list(self._obs_subject_ids)

    @property
    def collection_names(self) -> tuple[str, ...]:
        """Selected collection names."""
        return self._collection_names

    @property
    def obs_meta(self) -> pd.DataFrame:
        """Subject metadata for filtered subjects."""
        full_obs_meta = self._source.obs_meta.read()
        return full_obs_meta[
            full_obs_meta["obs_subject_id"].isin(self._obs_subject_ids)
        ].reset_index(drop=True)

    def __len__(self) -> int:
        """Number of subjects in view."""
        return len(self._obs_subject_ids)

    @property
    def n_collections(self) -> int:
        """Number of collections in view."""
        return len(self._collection_names)

    def __repr__(self) -> str:
        """Concise representation of the RadiObjectView."""
        collections = ", ".join(self._collection_names) if self._collection_names else "none"
        return f"RadiObjectView({len(self)} subjects, {self.n_collections} collections: [{collections}])"

    # ===== Bridge to Query (Pipeline Mode) =====

    def to_query(self) -> Query:
        """Convert this view to a Query for pipeline-style operations.

        Use this when you need to chain additional filters or use streaming
        materialization after initial interactive exploration.

        Example:
            view = radi.iloc[0:100]  # Interactive: quick look at first 100
            query = view.to_query()  # Bridge to pipeline mode
            new_radi = query.filter("age > 40").to_radi_object("...", streaming=True)
        """
        from radiobject.query import Query

        return Query(
            self._source,
            subject_ids=frozenset(self._obs_subject_ids),
            output_collections=frozenset(self._collection_names),
        )

    # ===== Further Filtering (chainable) =====

    def select_subjects(self, obs_subject_ids: list[str]) -> RadiObjectView:
        """Further filter to subset of subjects."""
        current_set = set(self._obs_subject_ids)
        for sid in obs_subject_ids:
            if sid not in current_set:
                raise KeyError(f"obs_subject_id '{sid}' not in view")
        return RadiObjectView(
            source=self._source,
            obs_subject_ids=obs_subject_ids,
            collection_names=list(self._collection_names),
        )

    def select_collections(self, names: list[str]) -> RadiObjectView:
        """Further filter to subset of collections."""
        current_set = set(self._collection_names)
        for name in names:
            if name not in current_set:
                raise KeyError(f"Collection '{name}' not in view")
        return RadiObjectView(
            source=self._source,
            obs_subject_ids=list(self._obs_subject_ids),
            collection_names=names,
        )

    # ===== Write to new RadiObject (materialization) =====

    def to_radi_object(self, uri: str, ctx: tiledb.Ctx | None = None) -> RadiObject:
        """Materialize this view as a new RadiObject."""
        effective_ctx = ctx if ctx else self._source._effective_ctx()

        obs_meta_df = self._source.obs_meta.read()
        filtered_obs_meta = obs_meta_df[
            obs_meta_df["obs_subject_id"].isin(self._obs_subject_ids)
        ].reset_index(drop=True)

        obs_meta_schema = {}
        for col in filtered_obs_meta.columns:
            if col in ("obs_subject_id", "obs_id"):
                continue
            dtype = filtered_obs_meta[col].to_numpy().dtype
            if dtype == np.dtype("O"):
                dtype = np.dtype("U64")
            obs_meta_schema[col] = dtype

        RadiObject._create(
            uri,
            obs_meta_schema=obs_meta_schema,
            n_subjects=len(self._obs_subject_ids),
            ctx=ctx,
        )

        obs_meta_uri = f"{uri}/obs_meta"
        obs_subject_ids = filtered_obs_meta["obs_subject_id"].astype(str).to_numpy()
        obs_ids = (
            filtered_obs_meta["obs_id"].astype(str).to_numpy()
            if "obs_id" in filtered_obs_meta.columns
            else obs_subject_ids
        )
        with tiledb.open(obs_meta_uri, "w", ctx=effective_ctx) as arr:
            attr_data = {}
            for col in filtered_obs_meta.columns:
                if col not in ("obs_subject_id", "obs_id"):
                    attr_data[col] = filtered_obs_meta[col].to_numpy()
            arr[obs_subject_ids, obs_ids] = attr_data

        collections_uri = f"{uri}/collections"
        for coll_name in self._collection_names:
            src_collection = self._source.collection(coll_name)
            new_vc_uri = f"{collections_uri}/{coll_name}"

            _copy_filtered_volume_collection(
                src_collection,
                new_vc_uri,
                obs_subject_ids=list(self._obs_subject_ids),
                name=coll_name,
                ctx=ctx,
            )

            with tiledb.Group(collections_uri, "w", ctx=effective_ctx) as grp:
                grp.add(new_vc_uri, name=coll_name)

        with tiledb.Group(uri, "w", ctx=effective_ctx) as grp:
            grp.meta["n_collections"] = len(self._collection_names)
            grp.meta["subject_count"] = len(self._obs_subject_ids)

        return RadiObject(uri, ctx=ctx)

    def to_radi_object_streaming(self, uri: str, ctx: tiledb.Ctx | None = None) -> RadiObject:
        """Materialize this view as a new RadiObject using streaming writes.

        Memory-efficient alternative to to_radi_object() that writes volumes
        one at a time instead of loading all into memory.
        """
        from radiobject.streaming import RadiObjectWriter

        obs_meta_df = self.obs_meta

        # Build obs_meta schema
        obs_meta_schema: dict[str, np.dtype] = {}
        for col in obs_meta_df.columns:
            if col in ("obs_subject_id", "obs_id"):
                continue
            dtype = obs_meta_df[col].to_numpy().dtype
            if dtype == np.dtype("O"):
                dtype = np.dtype("U64")
            obs_meta_schema[col] = dtype

        with RadiObjectWriter(uri, obs_meta_schema=obs_meta_schema, ctx=ctx) as writer:
            writer.write_obs_meta(obs_meta_df)

            for coll_name in self._collection_names:
                src_collection = self._source.collection(coll_name)
                obs_df = src_collection.obs.read()
                filtered_obs = obs_df[obs_df["obs_subject_id"].isin(self._obs_subject_ids)]

                if len(filtered_obs) == 0:
                    continue

                # Extract obs schema
                obs_schema: dict[str, np.dtype] = {}
                for col in src_collection.obs.columns:
                    if col in ("obs_id", "obs_subject_id"):
                        continue
                    obs_schema[col] = src_collection.obs.dtypes[col]

                with writer.add_collection(
                    coll_name, src_collection.shape, obs_schema
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

        return RadiObject(uri, ctx=ctx)

    # ===== Access collections in view =====

    def collection(self, name: str) -> VolumeCollection:
        """Get a VolumeCollection by name (from source)."""
        if name not in self._collection_names:
            raise KeyError(f"Collection '{name}' not in view")
        return self._source.collection(name)

    def __getattr__(self, name: str) -> VolumeCollection:
        """Attribute access to collections."""
        if name.startswith("_"):
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")
        return self.collection(name)


# ===== Helper Functions =====


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
    ctx: tiledb.Ctx | None = None,
) -> None:
    """Copy a VolumeCollection to a new URI."""
    effective_ctx = ctx if ctx else global_ctx()

    # Preserve source name if not explicitly provided
    collection_name = name if name is not None else src.name

    VolumeCollection._create(
        dst_uri,
        shape=src.shape,
        obs_schema=_extract_obs_schema(src.obs),
        n_volumes=len(src),
        name=collection_name,
        ctx=ctx,
    )

    obs_df = src.obs.read()
    obs_uri = f"{dst_uri}/obs"
    obs_subject_ids = obs_df["obs_subject_id"].astype(str).to_numpy()
    obs_ids = obs_df["obs_id"].astype(str).to_numpy()
    with tiledb.open(obs_uri, "w", ctx=effective_ctx) as arr:
        attr_data = {
            col: obs_df[col].to_numpy()
            for col in obs_df.columns
            if col not in ("obs_subject_id", "obs_id")
        }
        arr[obs_subject_ids, obs_ids] = attr_data

    def write_volume(args: tuple[int, str, Volume]) -> WriteResult:
        idx, obs_id, vol = args
        worker_ctx = create_worker_ctx(ctx)
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


def _copy_filtered_volume_collection(
    src: VolumeCollection,
    dst_uri: str,
    obs_subject_ids: list[str],
    name: str | None = None,
    ctx: tiledb.Ctx | None = None,
) -> None:
    """Copy a VolumeCollection, filtering to volumes matching obs_subject_ids."""
    effective_ctx = ctx if ctx else global_ctx()

    # Preserve source name if not explicitly provided
    collection_name = name if name is not None else src.name

    obs_df = src.obs.read()
    subject_id_set = set(obs_subject_ids)

    filtered_obs = obs_df[obs_df["obs_subject_id"].isin(subject_id_set)].reset_index(drop=True)

    if len(filtered_obs) == 0:
        raise ValueError("No volumes match the specified obs_subject_ids")

    VolumeCollection._create(
        dst_uri,
        shape=src.shape,
        obs_schema=_extract_obs_schema(src.obs),
        n_volumes=len(filtered_obs),
        name=collection_name,
        ctx=ctx,
    )

    obs_uri = f"{dst_uri}/obs"
    obs_subject_ids_arr = filtered_obs["obs_subject_id"].astype(str).to_numpy()
    obs_ids_arr = filtered_obs["obs_id"].astype(str).to_numpy()
    with tiledb.open(obs_uri, "w", ctx=effective_ctx) as arr:
        attr_data = {
            col: filtered_obs[col].to_numpy()
            for col in filtered_obs.columns
            if col not in ("obs_subject_id", "obs_id")
        }
        arr[obs_subject_ids_arr, obs_ids_arr] = attr_data

    selected_obs_ids = set(filtered_obs["obs_id"])
    selected_indices = [i for i, oid in enumerate(src.obs_ids) if oid in selected_obs_ids]

    def write_volume(args: tuple[int, int, str]) -> WriteResult:
        new_idx, orig_idx, obs_id = args
        worker_ctx = create_worker_ctx(ctx)
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
    results = _write_volumes_parallel(
        write_volume, write_args, progress=False, desc="Filtering volumes"
    )

    with tiledb.Group(f"{dst_uri}/volumes", "w", ctx=effective_ctx) as vol_grp:
        for result in results:
            vol_grp.add(result.uri, name=str(result.index))
