"""Shared test fixtures for Snowpark ML Fundamentals.

Uses mocking to avoid requiring a real Snowflake connection during CI/CD.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_session():
    """Create a mock Snowpark Session."""
    session = MagicMock()
    session.get_current_warehouse.return_value = '"TEST_WH"'
    session.get_current_database.return_value = '"TEST_DB"'
    session.get_current_schema.return_value = '"TEST_SCHEMA"'
    session.get_current_role.return_value = '"TEST_ROLE"'

    version_row = MagicMock()
    version_row.__getitem__ = MagicMock(return_value="8.40.0")
    session.sql.return_value.collect.return_value = [version_row]

    return session


@pytest.fixture
def mock_dataframe():
    """Create a mock Snowpark DataFrame."""
    df = MagicMock()
    df.count.return_value = 5000
    df.columns = [
        "CUSTOMER_ID",
        "AGE",
        "PLAN_TYPE",
        "TENURE_MONTHS",
        "MONTHLY_CHARGES",
        "SUPPORT_TICKETS",
        "USAGE_HOURS_PER_WEEK",
        "CONTRACT_TYPE",
        "PAYMENT_METHOD",
        "NUM_DEPENDENTS",
        "TOTAL_CHARGES",
        "CHURNED",
    ]

    field_mock = MagicMock()
    field_mock.name = "AGE"
    field_mock.datatype = "LongType()"
    df.schema.fields = [field_mock]

    df.filter.return_value = df
    df.select.return_value = df
    df.with_column.return_value = df
    df.group_by.return_value.agg.return_value = df
    df.join.return_value = df
    df.sort.return_value = df
    df.random_split.return_value = (df, df)
    df.describe.return_value = df
    df.limit.return_value = df

    return df


@pytest.fixture
def sample_env_vars():
    """Set up environment variables for testing."""
    env = {
        "SNOWFLAKE_ACCOUNT": "test_account",
        "SNOWFLAKE_USER": "test_user",
        "SNOWFLAKE_PASSWORD": "test_password",
        "SNOWFLAKE_ROLE": "test_role",
        "SNOWFLAKE_WAREHOUSE": "test_wh",
        "SNOWFLAKE_DATABASE": "test_db",
        "SNOWFLAKE_SCHEMA": "test_schema",
    }
    with patch.dict("os.environ", env, clear=False):
        yield env
