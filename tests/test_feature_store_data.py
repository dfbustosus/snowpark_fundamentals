"""Tests for synthetic temporal data generation."""

from __future__ import annotations

from unittest.mock import MagicMock

from snowpark_fundamentals.feature_store.feature_data import (
    create_behavioral_features,
    create_customer_interactions_dataset,
    create_customer_transactions_dataset,
    create_derived_features,
    create_rfm_features,
)


class TestCreateCustomerTransactions:
    """Tests for create_customer_transactions_dataset()."""

    def test_creates_table_via_sql(self, mock_session):
        """Should execute CREATE OR REPLACE TABLE SQL."""
        create_customer_transactions_dataset(mock_session)

        mock_session.sql.assert_called_once()
        sql_arg = mock_session.sql.call_args[0][0]
        assert "CREATE OR REPLACE TABLE" in sql_arg
        assert "CUSTOMER_TRANSACTIONS" in sql_arg

    def test_returns_table_reference(self, mock_session):
        """Should return session.table() pointing to the created table."""
        mock_session.table.return_value = MagicMock()

        create_customer_transactions_dataset(mock_session)

        mock_session.table.assert_called_once()
        table_arg = mock_session.table.call_args[0][0]
        assert "CUSTOMER_TRANSACTIONS" in table_arg

    def test_custom_table_name(self, mock_session):
        """Should use custom table name."""
        create_customer_transactions_dataset(mock_session, table_name="MY_TXN")

        sql_arg = mock_session.sql.call_args[0][0]
        assert "MY_TXN" in sql_arg

    def test_sql_contains_temporal_columns(self, mock_session):
        """SQL should include date and temporal columns."""
        create_customer_transactions_dataset(mock_session)

        sql_arg = mock_session.sql.call_args[0][0]
        assert "ORDER_DATE" in sql_arg
        assert "DATEADD" in sql_arg
        assert "CUSTOMER_ID" in sql_arg


class TestCreateCustomerInteractions:
    """Tests for create_customer_interactions_dataset()."""

    def test_creates_table_via_sql(self, mock_session):
        """Should execute CREATE OR REPLACE TABLE SQL."""
        create_customer_interactions_dataset(mock_session)

        sql_arg = mock_session.sql.call_args[0][0]
        assert "CREATE OR REPLACE TABLE" in sql_arg
        assert "CUSTOMER_INTERACTIONS" in sql_arg

    def test_sql_contains_interaction_types(self, mock_session):
        """SQL should include interaction type categories."""
        create_customer_interactions_dataset(mock_session)

        sql_arg = mock_session.sql.call_args[0][0]
        assert "PAGE_VIEW" in sql_arg
        assert "SUPPORT_TICKET" in sql_arg
        assert "EMAIL_CLICK" in sql_arg


class TestCreateRfmFeatures:
    """Tests for create_rfm_features()."""

    def test_creates_rfm_table(self, mock_session):
        """Should create RFM features table from transactions."""
        create_rfm_features(mock_session)

        sql_arg = mock_session.sql.call_args[0][0]
        assert "CREATE OR REPLACE TABLE" in sql_arg
        assert "CUSTOMER_RFM_FEATURES" in sql_arg

    def test_sql_contains_time_windows(self, mock_session):
        """SQL should include 30d, 90d, 365d time windows."""
        create_rfm_features(mock_session)

        sql_arg = mock_session.sql.call_args[0][0]
        assert "ORDERS_30D" in sql_arg
        assert "ORDERS_90D" in sql_arg
        assert "ORDERS_365D" in sql_arg
        assert "SPEND_30D" in sql_arg

    def test_custom_table_names(self, mock_session):
        """Should accept custom source and target table names."""
        create_rfm_features(mock_session, source_table="SRC", target_table="TGT")

        sql_arg = mock_session.sql.call_args[0][0]
        assert "TGT" in sql_arg


class TestCreateBehavioralFeatures:
    """Tests for create_behavioral_features()."""

    def test_creates_behavioral_table(self, mock_session):
        """Should create behavioral features table."""
        create_behavioral_features(mock_session)

        sql_arg = mock_session.sql.call_args[0][0]
        assert "CREATE OR REPLACE TABLE" in sql_arg
        assert "CUSTOMER_BEHAVIORAL_FEATURES" in sql_arg

    def test_sql_contains_engagement_metrics(self, mock_session):
        """SQL should include engagement metrics."""
        create_behavioral_features(mock_session)

        sql_arg = mock_session.sql.call_args[0][0]
        assert "TOTAL_PAGE_VIEWS" in sql_arg
        assert "TOTAL_CLICKS" in sql_arg
        assert "SUPPORT_TICKETS_30D" in sql_arg


class TestCreateDerivedFeatures:
    """Tests for create_derived_features()."""

    def test_creates_derived_table(self, mock_session):
        """Should create derived features table."""
        create_derived_features(mock_session)

        sql_arg = mock_session.sql.call_args[0][0]
        assert "CREATE OR REPLACE TABLE" in sql_arg
        assert "CUSTOMER_DERIVED_FEATURES" in sql_arg

    def test_sql_contains_ratios_and_buckets(self, mock_session):
        """SQL should include derived ratios and bucket logic."""
        create_derived_features(mock_session)

        sql_arg = mock_session.sql.call_args[0][0]
        assert "SPEND_PER_ORDER" in sql_arg
        assert "RECENCY_BUCKET" in sql_arg
        assert "SPEND_BUCKET" in sql_arg
        assert "ENGAGEMENT_SCORE" in sql_arg
