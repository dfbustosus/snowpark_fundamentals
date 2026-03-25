"""Tests for data loading module."""

from __future__ import annotations

import pytest

from snowpark_fundamentals.data.loader import (
    explore_dataframe,
    load_table,
    load_with_sql,
    sample_stratified,
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


class TestSampleStratified:
    def test_default_frac_applied_to_all_classes(self, mock_dataframe):
        """Verify frac=0.5 creates {0: 0.5, 1: 0.5} for binary labels."""
        from unittest.mock import MagicMock

        row0 = MagicMock()
        row0.__getitem__ = MagicMock(return_value=0)
        row1 = MagicMock()
        row1.__getitem__ = MagicMock(return_value=1)
        mock_dataframe.select.return_value.distinct.return_value.collect.return_value = [
            row0,
            row1,
        ]
        mock_dataframe.sample_by.return_value = mock_dataframe

        sample_stratified(mock_dataframe, "CHURNED")

        call_args = mock_dataframe.sample_by.call_args
        fractions_arg = call_args[0][1]
        assert fractions_arg == {0: 0.5, 1: 0.5}

    def test_custom_fractions_passed_through(self, mock_dataframe):
        """Verify explicit fractions dict is used directly."""
        mock_dataframe.sample_by.return_value = mock_dataframe
        custom_fractions = {0: 0.3, 1: 0.8}

        sample_stratified(mock_dataframe, "CHURNED", fractions=custom_fractions)

        call_args = mock_dataframe.sample_by.call_args
        fractions_arg = call_args[0][1]
        assert fractions_arg == {0: 0.3, 1: 0.8}

    def test_custom_frac_value(self, mock_dataframe):
        """Verify custom frac value is applied to all classes."""
        from unittest.mock import MagicMock

        row0 = MagicMock()
        row0.__getitem__ = MagicMock(return_value=0)
        row1 = MagicMock()
        row1.__getitem__ = MagicMock(return_value=1)
        mock_dataframe.select.return_value.distinct.return_value.collect.return_value = [
            row0,
            row1,
        ]
        mock_dataframe.sample_by.return_value = mock_dataframe

        sample_stratified(mock_dataframe, "CHURNED", frac=0.3)

        call_args = mock_dataframe.sample_by.call_args
        fractions_arg = call_args[0][1]
        assert fractions_arg == {0: 0.3, 1: 0.3}
