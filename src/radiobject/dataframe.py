"""Dataframe - a 2D heterogeneous array backed by TileDB."""

from __future__ import annotations

from functools import cached_property

import numpy as np
import pandas as pd
import tiledb

from radiobject.ctx import get_radiobject_config, get_tiledb_ctx

# Mandatory index columns for all Dataframes
INDEX_COLUMNS = ("obs_subject_id", "obs_id")


class Dataframe:
    """TileDB-backed sparse dataframe for observation metadata.

    Used internally for obs_meta (subject-level) and obs (volume-level) storage.
    Indexed by (obs_subject_id, obs_id) with user-defined attribute columns.

    Example:
        df = dataframe.read(columns=["age"], value_filter="age > 40")
    """

    def __init__(self, uri: str, ctx: tiledb.Ctx | None = None):
        self.uri: str = uri
        self._ctx: tiledb.Ctx | None = ctx

    def _effective_ctx(self) -> tiledb.Ctx:
        return self._ctx if self._ctx else get_tiledb_ctx()

    @cached_property
    def _schema(self) -> tiledb.ArraySchema:
        """Cached schema loaded once from disk."""
        return tiledb.ArraySchema.load(self.uri, ctx=self._effective_ctx())

    @property
    def shape(self) -> tuple[int, int]:
        """(n_rows, n_columns) dimensions."""
        with tiledb.open(self.uri, "r", ctx=self._effective_ctx()) as arr:
            n_rows = arr.nonempty_domain()[0][1] if arr.nonempty_domain()[0] else 0
            if isinstance(n_rows, str):
                n_rows = len(arr.query(attrs=[])[:][INDEX_COLUMNS[0]])
        n_cols = self._schema.nattr
        return (n_rows, n_cols)

    @property
    def index_columns(self) -> tuple[str, str]:
        """Index column names (dimension names)."""
        return INDEX_COLUMNS

    @property
    def columns(self) -> list[str]:
        """Attribute column names (excluding index columns)."""
        return [self._schema.attr(i).name for i in range(self._schema.nattr)]

    @property
    def all_columns(self) -> list[str]:
        """All column names including index columns."""
        return list(INDEX_COLUMNS) + self.columns

    @cached_property
    def dtypes(self) -> dict[str, np.dtype]:
        """Column data types (attributes only)."""
        schema = self._schema
        return {schema.attr(i).name: schema.attr(i).dtype for i in range(schema.nattr)}

    def __len__(self) -> int:
        with tiledb.open(self.uri, "r", ctx=self._effective_ctx()) as arr:
            result = arr.query(attrs=[])[:][INDEX_COLUMNS[0]]
            return len(result)

    def __repr__(self) -> str:
        return f"Dataframe(uri={self.uri!r}, shape={self.shape}, columns={self.columns})"

    def read(
        self,
        columns: list[str] | None = None,
        value_filter: str | None = None,
        include_index: bool = True,
    ) -> pd.DataFrame:
        """Read data with optional column selection and value filtering."""
        # Filter out index columns from requested columns (they're dimensions, not attributes)
        if columns is not None:
            attrs = [c for c in columns if c not in INDEX_COLUMNS]
        else:
            attrs = self.columns
        with tiledb.open(self.uri, "r", ctx=self._effective_ctx()) as arr:
            if value_filter is not None:
                result = arr.query(cond=value_filter, attrs=attrs)[:]
            else:
                result = arr.query(attrs=attrs)[:]
        data = {col: result[col] for col in attrs}
        if include_index:
            for idx_col in INDEX_COLUMNS:
                # Convert bytes to strings for index columns
                raw = result[idx_col]
                data[idx_col] = np.array(
                    [v.decode() if isinstance(v, bytes) else str(v) for v in raw]
                )
        df = pd.DataFrame(data)
        if include_index:
            col_order = list(INDEX_COLUMNS) + attrs
            df = df[col_order]
        return df

    @staticmethod
    def _validate_schema(schema: dict[str, np.dtype]) -> None:
        """Validate column names and types (for non-index attributes)."""
        for name, dtype in schema.items():
            if "\x00" in name:
                raise ValueError(f"Column name contains null byte: {name!r}")
            if name in INDEX_COLUMNS:
                raise ValueError(f"Column name conflicts with index column: {name!r}")
            if not isinstance(dtype, np.dtype):
                try:
                    np.dtype(dtype)
                except TypeError as e:
                    raise TypeError(f"Invalid dtype for column {name!r}: {dtype}") from e

    @staticmethod
    def _compression_filters() -> tiledb.FilterList:
        """Build compression filter list from current config."""
        compression_filter = get_radiobject_config().write.compression.as_filter()
        if compression_filter:
            return tiledb.FilterList([compression_filter])
        return tiledb.FilterList()

    def _invalidate_cache(self) -> None:
        """Clear cached schema and dtypes after schema-mutating operations."""
        for prop in ("_schema", "dtypes"):
            self.__dict__.pop(prop, None)

    def add_column(
        self,
        name: str,
        dtype: np.dtype | type,
        fill: object = None,
    ) -> None:
        """Add a new attribute column to the dataframe.

        Args:
            name: Column name (must not conflict with index columns or existing columns).
            dtype: NumPy dtype for the column.
            fill: If provided, write this value to all existing rows.
        """
        if name in INDEX_COLUMNS:
            raise ValueError(f"Column name conflicts with index column: {name!r}")
        if name in self.columns:
            raise ValueError(f"Column {name!r} already exists")

        ctx = self._effective_ctx()
        se = tiledb.ArraySchemaEvolution(ctx=ctx)
        se.add_attribute(
            tiledb.Attr(name=name, dtype=dtype, filters=self._compression_filters(), ctx=ctx)
        )
        se.array_evolve(self.uri)
        self._invalidate_cache()

        if fill is not None and len(self) > 0:
            with tiledb.open(self.uri, "r", ctx=ctx) as arr:
                all_data = arr[:]
                subject_ids = all_data[INDEX_COLUMNS[0]]
                obs_ids = all_data[INDEX_COLUMNS[1]]
                data = {col: all_data[col] for col in self.columns if col != name}
            data[name] = np.full(len(subject_ids), fill, dtype=np.dtype(dtype))
            with tiledb.open(self.uri, "w", ctx=ctx) as arr:
                arr[subject_ids, obs_ids] = data

    def drop_column(self, name: str) -> None:
        """Remove an attribute column from the dataframe."""
        if name in INDEX_COLUMNS:
            raise ValueError(f"Cannot drop index column: {name!r}")
        if name not in self.columns:
            raise ValueError(f"Column {name!r} does not exist")

        ctx = self._effective_ctx()
        se = tiledb.ArraySchemaEvolution(ctx=ctx)
        se.drop_attribute(name)
        se.array_evolve(self.uri)
        self._invalidate_cache()

    def update(self, df: pd.DataFrame) -> None:
        """Upsert rows from a pandas DataFrame.

        Existing coordinates are overwritten, new coordinates are appended.
        All non-index columns in df must already exist in the schema.
        """
        for idx_col in INDEX_COLUMNS:
            if idx_col not in df.columns:
                raise ValueError(f"DataFrame must contain index column: {idx_col!r}")

        attr_cols = [c for c in df.columns if c not in INDEX_COLUMNS]
        existing = set(self.columns)
        unknown = [c for c in attr_cols if c not in existing]
        if unknown:
            raise ValueError(f"Unknown columns: {unknown}. Use add_column() first.")

        ctx = self._effective_ctx()
        subject_ids = df[INDEX_COLUMNS[0]].astype(str).to_numpy()
        obs_ids = df[INDEX_COLUMNS[1]].astype(str).to_numpy()
        data = {col: df[col].to_numpy() for col in attr_cols}

        # TileDB sparse write requires all attributes — read missing ones for existing rows
        all_attrs = self.columns
        missing_attrs = [c for c in all_attrs if c not in attr_cols]
        if missing_attrs:
            with tiledb.open(self.uri, "r", ctx=ctx) as arr:
                existing = arr.query(attrs=missing_attrs)[subject_ids, obs_ids]
                for col in missing_attrs:
                    raw = existing[col]
                    if len(raw) == len(subject_ids):
                        data[col] = raw
                    else:
                        # New rows — fill with dtype default
                        dt = self.dtypes[col]
                        data[col] = np.zeros(len(subject_ids), dtype=dt)

        with tiledb.open(self.uri, "w", ctx=ctx) as arr:
            arr[subject_ids, obs_ids] = data

    def delete(self, cond: str) -> None:
        """Delete rows matching a TileDB query condition."""
        ctx = self._effective_ctx()
        with tiledb.open(self.uri, "d", ctx=ctx) as arr:
            arr.query(cond=cond).submit()

    @classmethod
    def create(
        cls,
        uri: str,
        schema: dict[str, np.dtype],
        ctx: tiledb.Ctx | None = None,
    ) -> Dataframe:
        """Create an empty sparse Dataframe indexed by obs_subject_id and obs_id."""
        cls._validate_schema(schema)
        effective_ctx = ctx if ctx else get_tiledb_ctx()

        dims = [
            tiledb.Dim(name=INDEX_COLUMNS[0], dtype="ascii", ctx=effective_ctx),
            tiledb.Dim(name=INDEX_COLUMNS[1], dtype="ascii", ctx=effective_ctx),
        ]
        domain = tiledb.Domain(*dims, ctx=effective_ctx)

        compression = cls._compression_filters()
        attrs = [
            tiledb.Attr(name=name, dtype=dtype, filters=compression, ctx=effective_ctx)
            for name, dtype in schema.items()
        ]

        array_schema = tiledb.ArraySchema(
            domain=domain,
            attrs=attrs,
            sparse=True,
            ctx=effective_ctx,
        )
        tiledb.Array.create(uri, array_schema, ctx=effective_ctx)

        return cls(uri, ctx=ctx)

    @classmethod
    def from_pandas(
        cls,
        uri: str,
        df: pd.DataFrame,
        ctx: tiledb.Ctx | None = None,
    ) -> Dataframe:
        """Create a new Dataframe from a pandas DataFrame with obs_subject_id and obs_id columns."""
        for idx_col in INDEX_COLUMNS:
            if idx_col not in df.columns:
                raise ValueError(f"DataFrame must contain index column: {idx_col!r}")

        attr_cols = [col for col in df.columns if col not in INDEX_COLUMNS]
        schema = {col: df[col].to_numpy().dtype for col in attr_cols}
        dataframe = cls.create(uri, schema=schema, ctx=ctx)

        effective_ctx = ctx if ctx else get_tiledb_ctx()
        with tiledb.open(uri, mode="w", ctx=effective_ctx) as arr:
            coords = {
                INDEX_COLUMNS[0]: df[INDEX_COLUMNS[0]].astype(str).to_numpy(),
                INDEX_COLUMNS[1]: df[INDEX_COLUMNS[1]].astype(str).to_numpy(),
            }
            data = {col: df[col].to_numpy() for col in attr_cols}
            arr[coords[INDEX_COLUMNS[0]], coords[INDEX_COLUMNS[1]]] = data

        return dataframe
