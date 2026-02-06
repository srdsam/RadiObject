"""Tests for src/Dataframe.py - TileDB-backed tabular data indexed by obs_subject_id and obs_id."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from radiobject.dataframe import INDEX_COLUMNS, Dataframe


@pytest.fixture
def dataframe_uri(temp_dir: Path) -> str:
    """URI for a test Dataframe."""
    return str(temp_dir / "test_dataframe")


@pytest.fixture
def sample_pandas_df() -> pd.DataFrame:
    """Sample pandas DataFrame with mandatory index columns and heterogeneous attributes."""
    return pd.DataFrame(
        {
            "obs_subject_id": ["subj_1", "subj_2", "subj_3", "subj_4", "subj_5"],
            "obs_id": ["vol_1", "vol_2", "vol_3", "vol_4", "vol_5"],
            "age": np.array([45, 67, 32, 58, 41], dtype=np.int32),
            "tumor_volume": np.array([12.5, 8.3, 15.2, 9.1, 11.8], dtype=np.float64),
            "survival_months": np.array([24.5, 18.2, 36.1, 12.8, 29.3], dtype=np.float64),
        }
    )


class TestDataframeConstruction:
    """Tests for Dataframe.create and basic properties."""

    def test_create_empty_dataframe(self, dataframe_uri: str) -> None:
        schema = {"col_a": np.int32, "col_b": np.float64}
        df = Dataframe.create(dataframe_uri, schema=schema)

        assert df.columns == ["col_a", "col_b"]
        assert df.index_columns == INDEX_COLUMNS

    def test_create_with_multiple_types(self, temp_dir: Path) -> None:
        uri = str(temp_dir / "multi_type_df")
        schema = {
            "int_col": np.int32,
            "float_col": np.float64,
            "uint_col": np.uint16,
        }
        df = Dataframe.create(uri, schema=schema)

        dtypes = df.dtypes
        assert dtypes["int_col"] == np.int32
        assert dtypes["float_col"] == np.float64
        assert dtypes["uint_col"] == np.uint16

    def test_schema_rejects_index_column_names(self, temp_dir: Path) -> None:
        uri = str(temp_dir / "bad_schema_df")
        schema = {"obs_subject_id": np.int32}  # conflicts with index

        with pytest.raises(ValueError, match="conflicts with index column"):
            Dataframe.create(uri, schema=schema)


class TestDataframeFromPandas:
    """Tests for Dataframe.from_pandas factory method."""

    def test_from_pandas_creates_dataframe(
        self, dataframe_uri: str, sample_pandas_df: pd.DataFrame
    ) -> None:
        df = Dataframe.from_pandas(dataframe_uri, sample_pandas_df)

        assert len(df) == 5
        assert set(df.columns) == {"age", "tumor_volume", "survival_months"}

    def test_from_pandas_preserves_dtypes(
        self, dataframe_uri: str, sample_pandas_df: pd.DataFrame
    ) -> None:
        df = Dataframe.from_pandas(dataframe_uri, sample_pandas_df)
        dtypes = df.dtypes

        assert dtypes["age"] == np.int32
        assert dtypes["tumor_volume"] == np.float64

    def test_from_pandas_requires_index_columns(self, dataframe_uri: str) -> None:
        bad_df = pd.DataFrame({"age": [1, 2, 3]})

        with pytest.raises(ValueError, match="must contain index column"):
            Dataframe.from_pandas(dataframe_uri, bad_df)

    def test_from_pandas_roundtrip(
        self, dataframe_uri: str, sample_pandas_df: pd.DataFrame
    ) -> None:
        df = Dataframe.from_pandas(dataframe_uri, sample_pandas_df)
        result = df.read()

        assert len(result) == len(sample_pandas_df)
        assert set(result.columns) == set(sample_pandas_df.columns)
        for col in ["age", "tumor_volume", "survival_months"]:
            np.testing.assert_array_almost_equal(result[col].values, sample_pandas_df[col].values)


class TestDataframeReadToPandas:
    """Tests for read conversion to pandas DataFrame."""

    def test_read_full(self, dataframe_uri: str, sample_pandas_df: pd.DataFrame) -> None:
        df = Dataframe.from_pandas(dataframe_uri, sample_pandas_df)
        result = df.read()

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5

    def test_read_with_column_selection(
        self, dataframe_uri: str, sample_pandas_df: pd.DataFrame
    ) -> None:
        df = Dataframe.from_pandas(dataframe_uri, sample_pandas_df)
        result = df.read(columns=["tumor_volume"], include_index=False)

        assert list(result.columns) == ["tumor_volume"]
        assert "age" not in result.columns

    def test_read_includes_index_by_default(
        self, dataframe_uri: str, sample_pandas_df: pd.DataFrame
    ) -> None:
        df = Dataframe.from_pandas(dataframe_uri, sample_pandas_df)
        result = df.read(columns=["age"])

        assert "obs_subject_id" in result.columns
        assert "obs_id" in result.columns
        assert "age" in result.columns

    def test_read_exclude_index(self, dataframe_uri: str, sample_pandas_df: pd.DataFrame) -> None:
        df = Dataframe.from_pandas(dataframe_uri, sample_pandas_df)
        result = df.read(columns=["age"], include_index=False)

        assert "obs_subject_id" not in result.columns
        assert "obs_id" not in result.columns
        assert list(result.columns) == ["age"]


class TestDataframeRead:
    """Tests for partial read functionality."""

    def test_read_all(self, dataframe_uri: str, sample_pandas_df: pd.DataFrame) -> None:
        df = Dataframe.from_pandas(dataframe_uri, sample_pandas_df)
        data = df.read()

        assert "obs_subject_id" in data.columns
        assert "obs_id" in data.columns
        assert "age" in data.columns

    def test_read_specific_columns(
        self, dataframe_uri: str, sample_pandas_df: pd.DataFrame
    ) -> None:
        df = Dataframe.from_pandas(dataframe_uri, sample_pandas_df)
        data = df.read(columns=["tumor_volume"], include_index=False)

        assert list(data.columns) == ["tumor_volume"]
        np.testing.assert_array_almost_equal(
            data["tumor_volume"].values, sample_pandas_df["tumor_volume"].values
        )

    def test_read_preserves_index_column_order(
        self, dataframe_uri: str, sample_pandas_df: pd.DataFrame
    ) -> None:
        df = Dataframe.from_pandas(dataframe_uri, sample_pandas_df)
        data = df.read(columns=["age"])

        assert list(data.columns)[:2] == ["obs_subject_id", "obs_id"]


class TestDataframeMutation:
    """Tests for add_column, drop_column, update, and delete."""

    def test_add_column(self, dataframe_uri: str, sample_pandas_df: pd.DataFrame) -> None:
        df = Dataframe.from_pandas(dataframe_uri, sample_pandas_df)
        df.add_column("site", np.dtype("U10"))

        assert "site" in df.columns
        assert "site" in df.dtypes

    def test_add_column_with_fill(self, dataframe_uri: str, sample_pandas_df: pd.DataFrame) -> None:
        df = Dataframe.from_pandas(dataframe_uri, sample_pandas_df)
        df.add_column("label", np.int32, fill=0)

        result = df.read(columns=["label"], include_index=False)
        np.testing.assert_array_equal(result["label"].values, np.zeros(5, dtype=np.int32))

    def test_add_column_rejects_duplicate(
        self, dataframe_uri: str, sample_pandas_df: pd.DataFrame
    ) -> None:
        df = Dataframe.from_pandas(dataframe_uri, sample_pandas_df)

        with pytest.raises(ValueError, match="already exists"):
            df.add_column("age", np.int32)

    def test_add_column_rejects_index_name(
        self, dataframe_uri: str, sample_pandas_df: pd.DataFrame
    ) -> None:
        df = Dataframe.from_pandas(dataframe_uri, sample_pandas_df)

        with pytest.raises(ValueError, match="conflicts with index column"):
            df.add_column("obs_subject_id", np.int32)

    def test_drop_column(self, dataframe_uri: str, sample_pandas_df: pd.DataFrame) -> None:
        df = Dataframe.from_pandas(dataframe_uri, sample_pandas_df)
        df.drop_column("survival_months")

        assert "survival_months" not in df.columns

    def test_drop_column_rejects_nonexistent(
        self, dataframe_uri: str, sample_pandas_df: pd.DataFrame
    ) -> None:
        df = Dataframe.from_pandas(dataframe_uri, sample_pandas_df)

        with pytest.raises(ValueError, match="does not exist"):
            df.drop_column("nonexistent_col")

    def test_drop_column_rejects_index(
        self, dataframe_uri: str, sample_pandas_df: pd.DataFrame
    ) -> None:
        df = Dataframe.from_pandas(dataframe_uri, sample_pandas_df)

        with pytest.raises(ValueError, match="Cannot drop index column"):
            df.drop_column("obs_id")

    def test_update_existing_values(
        self, dataframe_uri: str, sample_pandas_df: pd.DataFrame
    ) -> None:
        df = Dataframe.from_pandas(dataframe_uri, sample_pandas_df)
        update_df = pd.DataFrame(
            {
                "obs_subject_id": ["subj_1"],
                "obs_id": ["vol_1"],
                "age": np.array([99], dtype=np.int32),
            }
        )
        df.update(update_df)

        result = df.read(value_filter="obs_subject_id == 'subj_1'")
        assert result["age"].iloc[0] == 99

    def test_update_appends_new_rows(
        self, dataframe_uri: str, sample_pandas_df: pd.DataFrame
    ) -> None:
        df = Dataframe.from_pandas(dataframe_uri, sample_pandas_df)
        new_row = pd.DataFrame(
            {
                "obs_subject_id": ["subj_6"],
                "obs_id": ["vol_6"],
                "age": np.array([55], dtype=np.int32),
                "tumor_volume": np.array([7.0], dtype=np.float64),
                "survival_months": np.array([20.0], dtype=np.float64),
            }
        )
        df.update(new_row)

        assert len(df) == 6

    def test_update_rejects_unknown_columns(
        self, dataframe_uri: str, sample_pandas_df: pd.DataFrame
    ) -> None:
        df = Dataframe.from_pandas(dataframe_uri, sample_pandas_df)
        bad_df = pd.DataFrame(
            {"obs_subject_id": ["subj_1"], "obs_id": ["vol_1"], "nonexistent": [1]}
        )

        with pytest.raises(ValueError, match="Unknown columns.*add_column"):
            df.update(bad_df)

    def test_update_rejects_missing_index(
        self, dataframe_uri: str, sample_pandas_df: pd.DataFrame
    ) -> None:
        df = Dataframe.from_pandas(dataframe_uri, sample_pandas_df)
        bad_df = pd.DataFrame({"age": [99]})

        with pytest.raises(ValueError, match="must contain index column"):
            df.update(bad_df)

    def test_delete(self, dataframe_uri: str, sample_pandas_df: pd.DataFrame) -> None:
        df = Dataframe.from_pandas(dataframe_uri, sample_pandas_df)
        df.delete("age < 40")

        result = df.read()
        assert all(result["age"] >= 40)
        assert len(result) < 5

    def test_add_then_update_new_column(
        self, dataframe_uri: str, sample_pandas_df: pd.DataFrame
    ) -> None:
        df = Dataframe.from_pandas(dataframe_uri, sample_pandas_df)
        df.add_column("grade", np.int32)

        update_df = pd.DataFrame(
            {
                "obs_subject_id": ["subj_1", "subj_2"],
                "obs_id": ["vol_1", "vol_2"],
                "grade": np.array([3, 4], dtype=np.int32),
            }
        )
        df.update(update_df)

        result = df.read(value_filter="obs_subject_id == 'subj_1'")
        assert result["grade"].iloc[0] == 3
