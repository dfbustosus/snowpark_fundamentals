"""Tests for configuration module."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from snowpark_fundamentals.config import SnowflakeConfig


class TestSnowflakeConfig:
    def test_from_env_success(self, sample_env_vars):
        config = SnowflakeConfig.from_env()
        assert config.account == "test_account"
        assert config.user == "test_user"
        assert config.password == "test_password"
        assert config.role == "test_role"
        assert config.warehouse == "test_wh"
        assert config.database == "test_db"
        assert config.schema == "test_schema"

    @patch("snowpark_fundamentals.config.load_dotenv")
    def test_from_env_missing_vars(self, mock_dotenv):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="Missing required environment variables"):
                SnowflakeConfig.from_env()

    def test_to_connection_params(self, sample_env_vars):
        config = SnowflakeConfig.from_env()
        params = config.to_connection_params()
        assert isinstance(params, dict)
        assert params["account"] == "test_account"
        assert params["user"] == "test_user"
        assert len(params) == 7

    def test_config_is_immutable(self, sample_env_vars):
        config = SnowflakeConfig.from_env()
        with pytest.raises(AttributeError):
            config.account = "new_account"

    @patch("snowpark_fundamentals.config.load_dotenv")
    def test_partial_env_vars_raises(self, mock_dotenv):
        partial_env = {
            "SNOWFLAKE_ACCOUNT": "test",
            "SNOWFLAKE_USER": "test",
        }
        with patch.dict("os.environ", partial_env, clear=True):
            with pytest.raises(ValueError, match="SNOWFLAKE_PASSWORD"):
                SnowflakeConfig.from_env()
