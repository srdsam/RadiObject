"""Streaming writers for memory-efficient RadiObject and VolumeCollection creation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import pandas as pd
import tiledb

from radiobject.ctx import ctx as global_ctx
from radiobject.dataframe import Dataframe
from radiobject.volume import Volume

# Scalar value types storable in TileDB obs attributes
AttrValue = int | float | bool | str

if TYPE_CHECKING:
    from radiobject.radi_object import RadiObject


class StreamingWriter:
    """Write volumes incrementally without full memory load.

    Context manager that creates a VolumeCollection and writes volumes
    one at a time, keeping memory usage bounded.

    Example:
        # Uniform shape collection
        with StreamingWriter(uri, shape=(256, 256, 128)) as writer:
            for nifti_path, subject_id in niftis:
                data = load_nifti(nifti_path)
                writer.write_volume(data, obs_id=f"{subject_id}_T1w", obs_subject_id=subject_id)

        # Heterogeneous shape collection (raw ingestion)
        with StreamingWriter(uri) as writer:
            for nifti_path, subject_id in niftis:
                data = load_nifti(nifti_path)
                writer.write_volume(data, obs_id=f"{subject_id}_T1w", obs_subject_id=subject_id)
    """

    def __init__(
        self,
        uri: str,
        shape: tuple[int, int, int] | None = None,
        obs_schema: dict[str, np.dtype] | None = None,
        name: str | None = None,
        ctx: tiledb.Ctx | None = None,
    ):
        self.uri = uri
        self.shape = shape  # None = heterogeneous shapes allowed
        self.obs_schema = obs_schema or {}
        self.name = name
        self._ctx = ctx
        self._volume_count = 0
        self._obs_rows: list[dict[str, AttrValue]] = []
        self._initialized = False
        self._finalized = False

    def _effective_ctx(self) -> tiledb.Ctx:
        return self._ctx if self._ctx else global_ctx()

    def __enter__(self) -> StreamingWriter:
        """Initialize the VolumeCollection structure."""
        if self._initialized:
            raise RuntimeError("StreamingWriter already initialized")

        effective_ctx = self._effective_ctx()

        # Create group structure
        tiledb.Group.create(self.uri, ctx=effective_ctx)

        volumes_uri = f"{self.uri}/volumes"
        tiledb.Group.create(volumes_uri, ctx=effective_ctx)

        obs_uri = f"{self.uri}/obs"
        Dataframe.create(obs_uri, schema=self.obs_schema, ctx=self._ctx)

        # Set initial metadata (n_volumes=0, will update on finalize)
        with tiledb.Group(self.uri, "w", ctx=effective_ctx) as grp:
            if self.shape is not None:
                grp.meta["x_dim"] = self.shape[0]
                grp.meta["y_dim"] = self.shape[1]
                grp.meta["z_dim"] = self.shape[2]
            grp.meta["n_volumes"] = 0
            if self.name is not None:
                grp.meta["name"] = self.name
            grp.add(volumes_uri, name="volumes")
            grp.add(obs_uri, name="obs")

        self._initialized = True
        return self

    def write_volume(
        self,
        data: npt.NDArray[np.floating],
        obs_id: str,
        obs_subject_id: str,
        **attrs: AttrValue,
    ) -> None:
        """Write a single volume to the collection.

        Args:
            data: Volume data array (must match shape if uniform collection)
            obs_id: Unique identifier for this volume
            obs_subject_id: Subject identifier (foreign key)
            **attrs: Additional obs attributes matching obs_schema
        """
        if not self._initialized:
            raise RuntimeError("StreamingWriter not initialized. Use as context manager.")
        if self._finalized:
            raise RuntimeError("StreamingWriter already finalized")

        # Only validate shape if collection requires uniform dimensions
        if self.shape is not None and data.shape[:3] != self.shape:
            raise ValueError(
                f"Volume shape {data.shape[:3]} doesn't match collection shape {self.shape}"
            )

        effective_ctx = self._effective_ctx()
        idx = self._volume_count
        volume_uri = f"{self.uri}/volumes/{idx}"

        # Write volume array
        vol = Volume.from_numpy(volume_uri, data, ctx=self._ctx)
        vol.set_obs_id(obs_id)

        # Register with volumes group
        with tiledb.Group(f"{self.uri}/volumes", "w", ctx=effective_ctx) as vol_grp:
            vol_grp.add(volume_uri, name=str(idx))

        # Collect obs row data
        row = {"obs_id": obs_id, "obs_subject_id": obs_subject_id, **attrs}
        self._obs_rows.append(row)

        self._volume_count += 1

    def write_batch(
        self, volumes: list[tuple[npt.NDArray[np.floating], str, str, dict[str, AttrValue]]]
    ) -> None:
        """Write multiple volumes at once.

        Args:
            volumes: List of (data, obs_id, obs_subject_id, attrs) tuples
        """
        for data, obs_id, obs_subject_id, attrs in volumes:
            self.write_volume(data, obs_id, obs_subject_id, **attrs)

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Finalize the collection by writing obs data and updating metadata."""
        if self._finalized:
            return

        if exc_type is not None:
            # Exception occurred, don't finalize
            return

        effective_ctx = self._effective_ctx()

        # Write obs data
        if self._obs_rows:
            obs_uri = f"{self.uri}/obs"
            obs_df = pd.DataFrame(self._obs_rows)

            obs_subject_ids = obs_df["obs_subject_id"].astype(str).to_numpy()
            obs_ids = obs_df["obs_id"].astype(str).to_numpy()

            with tiledb.open(obs_uri, "w", ctx=effective_ctx) as arr:
                attr_data = {
                    col: obs_df[col].to_numpy()
                    for col in obs_df.columns
                    if col not in ("obs_subject_id", "obs_id")
                }
                arr[obs_subject_ids, obs_ids] = attr_data

        # Update final volume count
        with tiledb.Group(self.uri, "w", ctx=effective_ctx) as grp:
            grp.meta["n_volumes"] = self._volume_count

        self._finalized = True

    @property
    def n_written(self) -> int:
        """Number of volumes written so far."""
        return self._volume_count


class RadiObjectWriter:
    """Build RadiObject incrementally from multiple collections.

    Context manager that creates a RadiObject structure and allows
    adding collections one at a time via StreamingWriter instances.

    Example:
        with RadiObjectWriter(uri) as writer:
            writer.write_obs_meta(obs_meta_df)

            with writer.add_collection("T1w", shape=(256, 256, 128)) as t1w_writer:
                for path, subj_id in t1w_files:
                    t1w_writer.write_volume(load_nifti(path), f"{subj_id}_T1w", subj_id)

            with writer.add_collection("FLAIR", shape=(256, 256, 128)) as flair_writer:
                for path, subj_id in flair_files:
                    flair_writer.write_volume(load_nifti(path), f"{subj_id}_FLAIR", subj_id)
    """

    def __init__(
        self,
        uri: str,
        obs_meta_schema: dict[str, np.dtype] | None = None,
        ctx: tiledb.Ctx | None = None,
    ):
        self.uri = uri
        self.obs_meta_schema = obs_meta_schema or {}
        self._ctx = ctx
        self._collection_names: list[str] = []
        self._subject_count = 0
        self._initialized = False
        self._finalized = False
        self._all_obs_ids: set[str] = set()  # Track obs_ids across all collections

    def _effective_ctx(self) -> tiledb.Ctx:
        return self._ctx if self._ctx else global_ctx()

    def __enter__(self) -> RadiObjectWriter:
        """Initialize the RadiObject structure."""
        if self._initialized:
            raise RuntimeError("RadiObjectWriter already initialized")

        effective_ctx = self._effective_ctx()

        # Create main group
        tiledb.Group.create(self.uri, ctx=effective_ctx)

        # Create obs_meta dataframe
        obs_meta_uri = f"{self.uri}/obs_meta"
        Dataframe.create(obs_meta_uri, schema=self.obs_meta_schema, ctx=self._ctx)

        # Create collections group
        collections_uri = f"{self.uri}/collections"
        tiledb.Group.create(collections_uri, ctx=effective_ctx)

        # Set initial metadata
        with tiledb.Group(self.uri, "w", ctx=effective_ctx) as grp:
            grp.meta["subject_count"] = 0
            grp.meta["n_collections"] = 0
            grp.add(obs_meta_uri, name="obs_meta")
            grp.add(collections_uri, name="collections")

        self._initialized = True
        return self

    def write_obs_meta(self, df: pd.DataFrame) -> None:
        """Write subject-level metadata.

        Args:
            df: DataFrame with obs_subject_id column and optional obs_id column
        """
        if not self._initialized:
            raise RuntimeError("RadiObjectWriter not initialized. Use as context manager.")
        if "obs_subject_id" not in df.columns:
            raise ValueError("DataFrame must contain 'obs_subject_id' column")

        effective_ctx = self._effective_ctx()
        obs_meta_uri = f"{self.uri}/obs_meta"

        obs_subject_ids = df["obs_subject_id"].astype(str).to_numpy()
        obs_ids = df["obs_id"].astype(str).to_numpy() if "obs_id" in df.columns else obs_subject_ids

        with tiledb.open(obs_meta_uri, "w", ctx=effective_ctx) as arr:
            attr_data = {
                col: df[col].to_numpy()
                for col in df.columns
                if col not in ("obs_subject_id", "obs_id")
            }
            arr[obs_subject_ids, obs_ids] = attr_data

        self._subject_count = len(df)

    def add_collection(
        self,
        name: str,
        shape: tuple[int, int, int] | None = None,
        obs_schema: dict[str, np.dtype] | None = None,
    ) -> StreamingWriter:
        """Add a new collection and return a StreamingWriter for it.

        Args:
            name: Collection name (e.g., "T1w", "FLAIR")
            shape: Volume dimensions (X, Y, Z). None for heterogeneous shapes.
            obs_schema: Schema for volume-level obs attributes

        Returns:
            StreamingWriter context manager for writing volumes
        """
        if not self._initialized:
            raise RuntimeError("RadiObjectWriter not initialized. Use as context manager.")
        if self._finalized:
            raise RuntimeError("RadiObjectWriter already finalized")
        if name in self._collection_names:
            raise ValueError(f"Collection '{name}' already added")

        collection_uri = f"{self.uri}/collections/{name}"
        writer = _CollectionStreamingWriter(
            uri=collection_uri,
            shape=shape,
            obs_schema=obs_schema,
            name=name,
            ctx=self._ctx,
            parent=self,
        )
        return writer

    def _register_collection(self, name: str, uri: str) -> None:
        """Internal: register a completed collection."""
        effective_ctx = self._effective_ctx()

        with tiledb.Group(f"{self.uri}/collections", "w", ctx=effective_ctx) as grp:
            grp.add(uri, name=name)

        self._collection_names.append(name)

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Finalize the RadiObject by updating metadata."""
        if self._finalized:
            return

        if exc_type is not None:
            return

        effective_ctx = self._effective_ctx()

        with tiledb.Group(self.uri, "w", ctx=effective_ctx) as grp:
            grp.meta["subject_count"] = self._subject_count
            grp.meta["n_collections"] = len(self._collection_names)

        self._finalized = True

    def finalize(self) -> RadiObject:
        """Finalize and return the created RadiObject."""
        from radiobject.radi_object import RadiObject

        if not self._finalized:
            self.__exit__(None, None, None)

        return RadiObject(self.uri, ctx=self._ctx)

    @property
    def collection_names(self) -> list[str]:
        """Names of collections added so far."""
        return list(self._collection_names)


class _CollectionStreamingWriter(StreamingWriter):
    """StreamingWriter that registers with parent RadiObjectWriter on completion."""

    def __init__(
        self,
        uri: str,
        shape: tuple[int, int, int] | None,
        obs_schema: dict[str, np.dtype] | None,
        name: str,
        ctx: tiledb.Ctx | None,
        parent: RadiObjectWriter,
    ):
        super().__init__(uri, shape, obs_schema, name, ctx)
        self._parent = parent

    def write_volume(
        self,
        data: npt.NDArray[np.floating],
        obs_id: str,
        obs_subject_id: str,
        **attrs: AttrValue,
    ) -> None:
        """Write a volume, checking obs_id uniqueness across all collections."""
        if obs_id in self._parent._all_obs_ids:
            raise ValueError(
                f"obs_id '{obs_id}' already exists in RadiObject. "
                f"obs_id must be unique across all collections."
            )
        self._parent._all_obs_ids.add(obs_id)
        super().write_volume(data, obs_id, obs_subject_id, **attrs)

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Finalize and register with parent."""
        super().__exit__(exc_type, exc_val, exc_tb)

        if exc_type is None and self._finalized:
            self._parent._register_collection(self.name, self.uri)
