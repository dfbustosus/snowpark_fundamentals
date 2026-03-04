"""Tests for training set generation and feature retrieval."""

from __future__ import annotations

from unittest.mock import MagicMock

from snowpark_fundamentals.feature_store.training_sets import (
    create_spine_dataframe,
    generate_training_set,
    retrieve_feature_values,
)


class TestCreateSpineDataframe:
    """Tests for create_spine_dataframe()."""

    def test_spine_without_timestamp(self, mock_session):
        """Should select distinct entity keys when no timestamp given."""
        mock_table = MagicMock()
        mock_session.table.return_value = mock_table
        mock_select = MagicMock()
        mock_table.select.return_value = mock_select

        create_spine_dataframe(mock_session, entity_table="CUSTOMERS")

        mock_table.select.assert_called_once_with("CUSTOMER_ID")
        mock_select.distinct.assert_called_once()

    def test_spine_with_timestamp(self, mock_session):
        """Should include timestamp column for point-in-time joins."""
        mock_table = MagicMock()
        mock_session.table.return_value = mock_table

        create_spine_dataframe(
            mock_session,
            entity_table="CUSTOMERS",
            timestamp_col="EVENT_TIMESTAMP",
        )

        mock_table.select.assert_called_once_with("CUSTOMER_ID", "EVENT_TIMESTAMP")

    def test_custom_entity_key(self, mock_session):
        """Should accept custom entity key column name."""
        mock_table = MagicMock()
        mock_session.table.return_value = mock_table
        mock_select = MagicMock()
        mock_table.select.return_value = mock_select

        create_spine_dataframe(
            mock_session,
            entity_table="ACCOUNTS",
            entity_key="ACCOUNT_ID",
        )

        mock_table.select.assert_called_once_with("ACCOUNT_ID")


class TestGenerateTrainingSet:
    """Tests for generate_training_set()."""

    def test_calls_generate_dataset(self, mock_feature_store):
        """Should call fs.generate_dataset with correct params."""
        mock_spine = MagicMock()
        mock_fv = MagicMock()

        generate_training_set(mock_feature_store, mock_spine, [mock_fv])

        mock_feature_store.generate_dataset.assert_called_once_with(
            name="TRAINING_SET",
            spine_df=mock_spine,
            features=[mock_fv],
            spine_timestamp_col=None,
            spine_label_cols=None,
        )

    def test_with_timestamp_and_labels(self, mock_feature_store):
        """Should pass spine_timestamp_col and spine_label_cols."""
        mock_spine = MagicMock()
        mock_fv = MagicMock()

        generate_training_set(
            mock_feature_store,
            mock_spine,
            [mock_fv],
            spine_timestamp_col="EVENT_TS",
            spine_label_cols=["CHURNED"],
        )

        call_kwargs = mock_feature_store.generate_dataset.call_args[1]
        assert call_kwargs["spine_timestamp_col"] == "EVENT_TS"
        assert call_kwargs["spine_label_cols"] == ["CHURNED"]

    def test_custom_name(self, mock_feature_store):
        """Should accept custom dataset name."""
        generate_training_set(
            mock_feature_store,
            MagicMock(),
            [MagicMock()],
            name="CHURN_DATASET_V2",
        )

        call_kwargs = mock_feature_store.generate_dataset.call_args[1]
        assert call_kwargs["name"] == "CHURN_DATASET_V2"

    def test_returns_dataset(self, mock_feature_store):
        """Should return the generated dataset."""
        expected = MagicMock()
        mock_feature_store.generate_dataset.return_value = expected

        result = generate_training_set(mock_feature_store, MagicMock(), [MagicMock()])
        assert result == expected


class TestRetrieveFeatureValues:
    """Tests for retrieve_feature_values()."""

    def test_calls_retrieve(self, mock_feature_store):
        """Should call fs.retrieve_feature_values."""
        mock_spine = MagicMock()
        mock_fv = MagicMock()

        retrieve_feature_values(mock_feature_store, mock_spine, [mock_fv])

        mock_feature_store.retrieve_feature_values.assert_called_once_with(
            spine_df=mock_spine,
            features=[mock_fv],
            spine_timestamp_col=None,
        )

    def test_with_timestamp(self, mock_feature_store):
        """Should pass spine_timestamp_col."""
        mock_spine = MagicMock()
        mock_fv = MagicMock()

        retrieve_feature_values(
            mock_feature_store,
            mock_spine,
            [mock_fv],
            spine_timestamp_col="EVENT_TS",
        )

        call_kwargs = mock_feature_store.retrieve_feature_values.call_args[1]
        assert call_kwargs["spine_timestamp_col"] == "EVENT_TS"

    def test_returns_dataframe(self, mock_feature_store):
        """Should return the feature values DataFrame."""
        expected = MagicMock()
        mock_feature_store.retrieve_feature_values.return_value = expected

        result = retrieve_feature_values(mock_feature_store, MagicMock(), [MagicMock()])
        assert result == expected
