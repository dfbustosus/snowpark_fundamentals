"""Tests for data loading module."""

from __future__ import annotations

import pytest

from snowpark_fundamentals.data.loader import (
    explore_dataframe,
    load_table,
    load_with_sql,
    split_data,
)


class TestLoadTable:
    def test_load_table_calls_session(self, mock_session):
        load_table(mock_session, "MY_DB.MY_SCHEMA.MY_TABLE")
        mock_session.table.assert_called_once_with("MY_DB.MY_SCHEMA.MY_TABLE")

    def test_load_with_sql_calls_session(self, mock_session):
        query = "SELECT * FROM MY_TABLE LIMIT 10"
        load_with_sql(mock_session, query)
        mock_session.sql.assert_called_with(query)


class TestExploreDataframe:
    def test_explore_returns_correct_keys(self, mock_dataframe):
        result = explore_dataframe(mock_dataframe)
        assert "row_count" in result
        assert "column_count" in result
        assert "columns" in result
        assert "dtypes" in result

    def test_explore_row_count(self, mock_dataframe):
        result = explore_dataframe(mock_dataframe)
        assert result["row_count"] == 5000

    def test_explore_column_count(self, mock_dataframe):
        result = explore_dataframe(mock_dataframe)
        assert result["column_count"] == 12


class TestSplitData:
    def test_split_returns_two_dataframes(self, mock_dataframe):
        train, test = split_data(mock_dataframe)
        assert train is not None
        assert test is not None

    def test_split_uses_correct_ratios(self, mock_dataframe):
        split_data(mock_dataframe, train_ratio=0.7)
        args, kwargs = mock_dataframe.random_split.call_args
        ratios = args[0]
        assert ratios[0] == pytest.approx(0.7)
        assert ratios[1] == pytest.approx(0.3)
        assert kwargs["seed"] == 42

    def test_split_custom_seed(self, mock_dataframe):
        split_data(mock_dataframe, seed=123)
        args, kwargs = mock_dataframe.random_split.call_args
        ratios = args[0]
        assert ratios[0] == pytest.approx(0.8)
        assert ratios[1] == pytest.approx(0.2)
        assert kwargs["seed"] == 123
