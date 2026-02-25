"""Snowpark Session factory.

Provides a single entry point for creating and managing Snowpark sessions.
Implements the factory pattern to ensure consistent session configuration.
"""

from __future__ import annotations

from pathlib import Path

from snowflake.snowpark import Session

from snowpark_fundamentals.config import SnowflakeConfig


def create_session(env_path: str | Path | None = None) -> Session:
    """Create a Snowpark session from environment configuration.

    This is the primary entry point for connecting to Snowflake
    throughout the tutorial modules.

    Args:
        env_path: Optional path to .env file.

    Returns:
        Configured Snowpark Session ready for use.
    """
    config = SnowflakeConfig.from_env(env_path)
    session: Session = Session.builder.configs(config.to_connection_params()).create()
    return session


def validate_session(session: Session) -> dict[str, str]:
    """Validate that a session is working and return environment info.

    Useful for verifying connectivity at the start of a tutorial session.

    Args:
        session: Active Snowpark session.

    Returns:
        Dict with current warehouse, database, schema, role, and version.
    """
    info = {
        "warehouse": session.get_current_warehouse() or "N/A",
        "database": session.get_current_database() or "N/A",
        "schema": session.get_current_schema() or "N/A",
        "role": session.get_current_role() or "N/A",
        "snowflake_version": session.sql("SELECT CURRENT_VERSION()").collect()[0][0],
    }
    return info
