"""RadiObject - top-level container for multi-collection radiology data."""

from __future__ import annotations

import warnings
from collections import defaultdict
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Sequence, overload

import numpy as np
import numpy.typing as npt
import pandas as pd
import tiledb

from src.ctx import ctx as global_ctx
from src.dataframe import Dataframe
from src.imaging_metadata import (
    extract_nifti_metadata,
    extract_dicom_metadata,
    infer_series_type,
)
from src.indexing import Index
from src.parallel import WriteResult, create_worker_ctx, map_on_threads
from src.volume import Volume
from src.volume_collection import VolumeCollection, _normalize_index

if TYPE_CHECKING:
    from src.query import Query


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

    def __getitem__(
        self, key: int | slice | list[int] | npt.NDArray[np.bool_]
    ) -> RadiObjectView:
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
    def obs_subject_ids(self) -> list[str]:
        """All obs_subject_id values in index order."""
        return list(self._index.keys)

    def obs_subject_id_to_index(self, obs_subject_id: str) -> int:
        """Map obs_subject_id to integer index."""
        return self._index.get_index(obs_subject_id)

    def index_to_obs_subject_id(self, idx: int) -> str:
        """Map integer index to obs_subject_id."""
        return self._index.get_key(idx)

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

    def __repr__(self) -> str:
        """Concise representation of the RadiObject."""
        collections = ", ".join(self.collection_names) if self.collection_names else "none"
        return (
            f"RadiObject({len(self)} subjects, {self.n_collections} collections: [{collections}])"
        )

    # ===== Query Builder (Pipeline Mode) =====

    def query(self) -> Query:
        """Create a lazy Query builder for pipeline-style filtering.

        Query objects accumulate filters without accessing data. Explicit methods
        like iter_volumes(), to_radi_object(), or count() trigger materialization.

        Example:
            result = (
                radi.query()
                .filter("age > 40 and tumor_grade == 'HGG'")
                .filter_collections(["T1w", "FLAIR"])
                .sample(100, seed=42)
                .to_radi_object("s3://bucket/subset", streaming=True)
            )
        """
        from src.query import Query

        return Query(self)

    # ===== Filtering (EDA Mode - returns RadiObjectView) =====

    def _filter_by_indices(self, indices: list[int]) -> RadiObjectView:
        """Create a view filtered to specific subject indices."""
        subject_ids = [self.index_to_obs_subject_id(i) for i in indices]
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
                raise ValueError(
                    f"obs_meta missing entries for: {sorted(missing)[:5]}"
                )
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
            self._append_niftis(niftis, reorient, effective_ctx)
        else:
            self._append_dicoms(dicom_dirs, reorient, effective_ctx)

        # Invalidate cached properties
        for prop in ("_index", "_metadata", "collection_names"):
            if prop in self.__dict__:
                del self.__dict__[prop]

    def _append_niftis(
        self,
        niftis: Sequence[tuple[str | Path, str]],
        reorient: bool | None,
        effective_ctx: tiledb.Ctx,
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

        for (shape, series_type), items in groups.items():
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
                vc.append(niftis=items, reorient=reorient)
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

        for (shape, modality), items in groups.items():
            collection_name = modality
            if collection_name in existing_collections:
                vc = self.collection(collection_name)
                if vc.shape != shape:
                    collection_name = f"{modality}_{shape[0]}x{shape[1]}x{shape[2]}"

            if collection_name in existing_collections:
                vc = self.collection(collection_name)
                vc.append(dicom_dirs=items, reorient=reorient)
            else:
                vc_uri = f"{collections_uri}/{collection_name}"
                VolumeCollection.from_dicoms(
                    uri=vc_uri,
                    dicom_dirs=items,
                    reorient=reorient,
                    validate_dimensions=True,
                    name=collection_name,
                    ctx=self._ctx,
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
            obs_ids = obs_meta["obs_id"].astype(str).to_numpy() if "obs_id" in obs_meta.columns else obs_subject_ids
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
        niftis: Sequence[tuple[str | Path, str]],
        obs_meta: pd.DataFrame | None = None,
        reorient: bool | None = None,
        ctx: tiledb.Ctx | None = None,
    ) -> RadiObject:
        """Create RadiObject from NIfTI files with automatic grouping.

        Files are automatically grouped into VolumeCollections by:
        1. Dimensions (x, y, z) - files with same shape go together
        2. Series type - inferred from filename/header (T1w, FLAIR, T2w, etc.)

        Args:
            uri: Target URI for RadiObject
            niftis: List of (nifti_path, obs_subject_id) tuples
            obs_meta: Subject-level metadata (user-provided). Must contain obs_subject_id column.
            reorient: Reorient to canonical orientation (None uses config default)
            ctx: TileDB context

        Example:
            radi = RadiObject.from_niftis(
                uri="/storage/study",
                niftis=[
                    ("sub01_T1w.nii.gz", "sub-01"),
                    ("sub01_FLAIR.nii.gz", "sub-01"),
                    ("sub02_T1w.nii.gz", "sub-02"),
                    ("sub02_FLAIR.nii.gz", "sub-02"),
                ],
                obs_meta=pd.DataFrame({
                    "obs_subject_id": ["sub-01", "sub-02"],
                    "age": [45, 52],
                }),
            )
            # Results in: radi.T1w, radi.FLAIR collections
        """
        if not niftis:
            raise ValueError("At least one NIfTI file is required")

        # Collect all subject IDs
        all_subject_ids = {sid for _, sid in niftis}

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
            obs_meta = pd.DataFrame({
                "obs_subject_id": sorted_ids,
                "obs_id": sorted_ids,
            })

        # Extract metadata and infer series type for each file
        file_info: list[tuple[Path, str, tuple[int, int, int], str]] = []
        for nifti_path, obs_subject_id in niftis:
            path = Path(nifti_path)
            if not path.exists():
                raise FileNotFoundError(f"NIfTI file not found: {path}")

            metadata = extract_nifti_metadata(path)
            series_type = infer_series_type(path)
            shape = metadata.dimensions
            file_info.append((path, obs_subject_id, shape, series_type))

        # Group by (shape, series_type)
        groups: dict[tuple[tuple[int, int, int], str], list[tuple[Path, str]]] = defaultdict(list)
        for path, subject_id, shape, series_type in file_info:
            key = (shape, series_type)
            groups[key].append((path, subject_id))

        # Create VolumeCollection for each group
        effective_ctx = ctx if ctx else global_ctx()

        # Ensure parent directories exist
        tiledb.Group.create(uri, ctx=effective_ctx)
        collections_uri = f"{uri}/collections"
        tiledb.Group.create(collections_uri, ctx=effective_ctx)

        collections: dict[str, VolumeCollection] = {}
        used_names: set[str] = set()

        for (shape, series_type), items in groups.items():
            # Generate unique collection name
            collection_name = series_type
            if collection_name in used_names:
                collection_name = f"{series_type}_{shape[0]}x{shape[1]}x{shape[2]}"
            used_names.add(collection_name)

            vc_uri = f"{collections_uri}/{collection_name}"
            nifti_list = [(path, subject_id) for path, subject_id in items]

            vc = VolumeCollection.from_niftis(
                uri=vc_uri,
                niftis=nifti_list,
                reorient=reorient,
                validate_dimensions=True,
                name=collection_name,
                ctx=ctx,
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

    @classmethod
    def from_dicoms(
        cls,
        uri: str,
        dicom_dirs: Sequence[tuple[str | Path, str]],
        obs_meta: pd.DataFrame | None = None,
        reorient: bool | None = None,
        ctx: tiledb.Ctx | None = None,
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
            obs_meta = pd.DataFrame({
                "obs_subject_id": sorted_ids,
                "obs_id": sorted_ids,
            })

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

        for (shape, modality), items in groups.items():
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

    RadiObjectView is designed for EDA (exploratory data analysis) mode - quick
    inspection and pandas-like access patterns. For pipeline/ETL workloads,
    use `to_query()` to get a lazy Query builder.
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
        return (
            f"RadiObjectView({len(self)} subjects, {self.n_collections} collections: [{collections}])"
        )

    # ===== Bridge to Query (Pipeline Mode) =====

    def to_query(self) -> Query:
        """Convert this view to a Query for pipeline-style operations.

        Use this when you need to chain additional filters or use streaming
        materialization after initial EDA exploration.

        Example:
            view = radi.iloc[0:100]  # EDA: quick look at first 100
            query = view.to_query()  # Bridge to pipeline mode
            new_radi = query.filter("age > 40").to_radi_object("...", streaming=True)
        """
        from src.query import Query

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
        filtered_obs_meta = obs_meta_df[obs_meta_df["obs_subject_id"].isin(self._obs_subject_ids)].reset_index(drop=True)

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
        obs_ids = filtered_obs_meta["obs_id"].astype(str).to_numpy() if "obs_id" in filtered_obs_meta.columns else obs_subject_ids
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

    def to_radi_object_streaming(
        self, uri: str, ctx: tiledb.Ctx | None = None
    ) -> RadiObject:
        """Materialize this view as a new RadiObject using streaming writes.

        Memory-efficient alternative to to_radi_object() that writes volumes
        one at a time instead of loading all into memory.
        """
        from src.streaming import RadiObjectWriter

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
                filtered_obs = obs_df[
                    obs_df["obs_subject_id"].isin(self._obs_subject_ids)
                ]

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
                            k: v
                            for k, v in row.items()
                            if k not in ("obs_id", "obs_subject_id")
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
        attr_data = {col: obs_df[col].to_numpy() for col in obs_df.columns if col not in ("obs_subject_id", "obs_id")}
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
    results = map_on_threads(write_volume, write_args)

    failures = [r for r in results if not r.success]
    if failures:
        raise RuntimeError(f"Volume copy failed: {failures[0].error}")

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
        attr_data = {col: filtered_obs[col].to_numpy() for col in filtered_obs.columns if col not in ("obs_subject_id", "obs_id")}
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
    results = map_on_threads(write_volume, write_args)

    failures = [r for r in results if not r.success]
    if failures:
        raise RuntimeError(f"Filtered volume copy failed: {failures[0].error}")

    with tiledb.Group(f"{dst_uri}/volumes", "w", ctx=effective_ctx) as vol_grp:
        for result in results:
            vol_grp.add(result.uri, name=str(result.index))
