"""Snowflake connection configuration.

Loads connection parameters from environment variables (.env file)
following the 12-factor app methodology.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class SnowflakeConfig:
    """Immutable Snowflake connection configuration."""

    account: str
    user: str
    password: str
    role: str
    warehouse: str
    database: str
    schema: str

    @classmethod
    def from_env(cls, env_path: str | Path | None = None) -> SnowflakeConfig:
        """Load configuration from environment variables.

        Args:
            env_path: Optional path to .env file. Defaults to project root.

        Returns:
            SnowflakeConfig instance with validated parameters.

        Raises:
            ValueError: If any required parameter is missing.
        """
        if env_path:
            load_dotenv(env_path)
        else:
            load_dotenv()

        required_vars = {
            "account": "SNOWFLAKE_ACCOUNT",
            "user": "SNOWFLAKE_USER",
            "password": "SNOWFLAKE_PASSWORD",
            "role": "SNOWFLAKE_ROLE",
            "warehouse": "SNOWFLAKE_WAREHOUSE",
            "database": "SNOWFLAKE_DATABASE",
            "schema": "SNOWFLAKE_SCHEMA",
        }

        params = {}
        missing = []
        for field_name, env_var in required_vars.items():
            value = os.getenv(env_var)
            if not value:
                missing.append(env_var)
            params[field_name] = value or ""

        if missing:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing)}. "
                "Copy .env.example to .env and fill in your credentials."
            )

        return cls(**params)

    def to_connection_params(self) -> dict[str, str]:
        """Convert to Snowpark Session connection parameters dict."""
        return {
            "account": self.account,
            "user": self.user,
            "password": self.password,
            "role": self.role,
            "warehouse": self.warehouse,
            "database": self.database,
            "schema": self.schema,
        }
