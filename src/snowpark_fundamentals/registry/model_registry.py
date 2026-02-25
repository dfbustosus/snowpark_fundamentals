"""Snowflake Model Registry operations.

Demonstrates how to register, version, and deploy ML models
using Snowflake's native Model Registry (snowflake.ml.registry).
"""

from __future__ import annotations

from typing import Any

from snowflake.ml.registry import Registry
from snowflake.snowpark import DataFrame, Session


def get_registry(session: Session, database: str, schema: str) -> Registry:
    """Get a reference to the Snowflake Model Registry.

    Args:
        session: Active Snowpark session.
        database: Database name for the registry.
        schema: Schema name for the registry.

    Returns:
        Registry instance.
    """
    return Registry(session=session, database_name=database, schema_name=schema)


def log_model(
    registry: Registry,
    model: Any,
    model_name: str,
    version_name: str,
    sample_input: DataFrame,
    metrics: dict[str, float] | None = None,
) -> Any:
    """Log a trained model to the Snowflake Model Registry.

    Args:
        registry: Model Registry instance.
        model: Trained model object (sklearn, xgboost, etc.).
        model_name: Name to register the model under.
        version_name: Version label (e.g., 'v1', 'v2').
        sample_input: Sample input data for schema inference.
        metrics: Optional performance metrics to store.

    Returns:
        ModelVersion reference.
    """
    log_kwargs: dict[str, Any] = {
        "model": model,
        "model_name": model_name,
        "version_name": version_name,
        "sample_input_data": sample_input,
    }
    if metrics:
        log_kwargs["metrics"] = metrics

    model_version = registry.log_model(**log_kwargs)
    return model_version


def list_models(registry: Registry) -> DataFrame:
    """List all models in the registry.

    Args:
        registry: Model Registry instance.

    Returns:
        DataFrame with model listing.
    """
    result: DataFrame = registry.show_models()
    return result


def load_model_and_predict(
    registry: Registry,
    model_name: str,
    version_name: str,
    input_df: DataFrame,
) -> DataFrame:
    """Load a model version from the registry and run predictions.

    Args:
        registry: Model Registry instance.
        model_name: Registered model name.
        version_name: Version to load.
        input_df: Data to score.

    Returns:
        DataFrame with predictions.
    """
    model_ref = registry.get_model(model_name)
    model_version = model_ref.version(version_name)
    result: DataFrame = model_version.run(input_df, function_name="predict")
    return result


def delete_model(registry: Registry, model_name: str) -> None:
    """Delete a model and all its versions from the registry.

    Args:
        registry: Model Registry instance.
        model_name: Model to delete.
    """
    registry.delete_model(model_name)
