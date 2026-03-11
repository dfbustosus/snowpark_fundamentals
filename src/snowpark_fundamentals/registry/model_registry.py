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


# ---------------------------------------------------------------------------
# Version Management
# ---------------------------------------------------------------------------


def get_model_version(registry: Registry, model_name: str, version_name: str) -> Any:
    """Get a specific model version from the registry.

    Args:
        registry: Model Registry instance.
        model_name: Registered model name.
        version_name: Version label to retrieve.

    Returns:
        ModelVersion reference.
    """
    return registry.get_model(model_name).version(version_name)


def list_versions(registry: Registry, model_name: str) -> DataFrame:
    """List all versions of a registered model.

    Args:
        registry: Model Registry instance.
        model_name: Registered model name.

    Returns:
        DataFrame with version metadata including aliases.
    """
    result: DataFrame = registry.get_model(model_name).show_versions()
    return result


def set_default_version(registry: Registry, model_name: str, version_name: str) -> None:
    """Set the default version for a model.

    The default version is used when no version is specified in
    inference calls.

    Args:
        registry: Model Registry instance.
        model_name: Registered model name.
        version_name: Version to set as default.
    """
    model_ref = registry.get_model(model_name)
    model_ref.default = version_name


def compare_model_versions(
    registry: Registry,
    model_name: str,
    version_names: list[str],
) -> list[dict]:
    """Compare metrics across multiple model versions.

    Collects metrics from each version into a list of dicts
    for easy side-by-side comparison.

    Args:
        registry: Model Registry instance.
        model_name: Registered model name.
        version_names: List of version labels to compare.

    Returns:
        List of dicts, each containing version name and its metrics.
    """
    model_ref = registry.get_model(model_name)
    results: list[dict] = []
    for version_name in version_names:
        version = model_ref.version(version_name)
        metrics = version.show_metrics()
        results.append({"version": version_name, "metrics": metrics})
    return results


# ---------------------------------------------------------------------------
# Metrics Management
# ---------------------------------------------------------------------------


def set_model_metrics(
    registry: Registry,
    model_name: str,
    version_name: str,
    metrics: dict[str, float],
) -> None:
    """Set metrics on a model version.

    Args:
        registry: Model Registry instance.
        model_name: Registered model name.
        version_name: Version to update.
        metrics: Dict of metric name to value.
    """
    version = registry.get_model(model_name).version(version_name)
    for key, value in metrics.items():
        version.set_metric(key, value)


def get_model_metrics(
    registry: Registry,
    model_name: str,
    version_name: str,
) -> dict:
    """Get all metrics for a model version.

    Args:
        registry: Model Registry instance.
        model_name: Registered model name.
        version_name: Version to query.

    Returns:
        Dict of metric name to value.
    """
    version = registry.get_model(model_name).version(version_name)
    result: dict = version.show_metrics()
    return result


def delete_model_metric(
    registry: Registry,
    model_name: str,
    version_name: str,
    metric_name: str,
) -> None:
    """Delete a specific metric from a model version.

    Args:
        registry: Model Registry instance.
        model_name: Registered model name.
        version_name: Version to update.
        metric_name: Name of the metric to remove.
    """
    version = registry.get_model(model_name).version(version_name)
    version.delete_metric(metric_name)


# ---------------------------------------------------------------------------
# Lifecycle Management
# ---------------------------------------------------------------------------


def set_model_alias(
    registry: Registry,
    model_name: str,
    version_name: str,
    alias: str,
) -> None:
    """Set an alias on a model version (e.g., 'production', 'staging').

    Aliases provide stable references for deployment. Moving an alias
    to a new version automatically removes it from the old one.

    Args:
        registry: Model Registry instance.
        model_name: Registered model name.
        version_name: Version to tag.
        alias: Alias string (e.g., 'production').
    """
    version = registry.get_model(model_name).version(version_name)
    version.set_alias(alias)


def unset_model_alias(
    registry: Registry,
    model_name: str,
    version_name: str,
    alias: str,
) -> None:
    """Remove an alias from a model version.

    Args:
        registry: Model Registry instance.
        model_name: Registered model name.
        version_name: Version to untag.
        alias: Alias to remove.
    """
    version = registry.get_model(model_name).version(version_name)
    version.unset_alias(alias)


def get_model_by_alias(registry: Registry, model_name: str, alias: str) -> Any:
    """Retrieve a model version by its alias.

    This is the recommended pattern for production inference:
    always reference by alias ('production') rather than a
    specific version name.

    Args:
        registry: Model Registry instance.
        model_name: Registered model name.
        alias: Alias to look up (e.g., 'production').

    Returns:
        ModelVersion reference.
    """
    return registry.get_model(model_name).version(alias)


def set_model_tags(
    registry: Registry,
    model_name: str,
    tags: dict[str, str],
) -> None:
    """Set tags on a model for governance and discovery.

    Args:
        registry: Model Registry instance.
        model_name: Registered model name.
        tags: Dict of tag key to value.
    """
    model_ref = registry.get_model(model_name)
    for key, value in tags.items():
        model_ref.set_tag(key, value)


# ---------------------------------------------------------------------------
# Metadata & Advanced Inference
# ---------------------------------------------------------------------------


def set_model_comment(
    registry: Registry,
    model_name: str,
    comment: str,
    version_name: str | None = None,
) -> None:
    """Set a comment on a model or a specific version.

    Args:
        registry: Model Registry instance.
        model_name: Registered model name.
        comment: Descriptive comment.
        version_name: If provided, set on the version; otherwise on the model.
    """
    model_ref = registry.get_model(model_name)
    if version_name:
        version = model_ref.version(version_name)
        version.comment = comment
    else:
        model_ref.comment = comment


def show_model_functions(
    registry: Registry,
    model_name: str,
    version_name: str,
) -> list[str]:
    """List callable functions for a model version.

    Registered models expose functions like 'predict', 'predict_proba',
    and 'explain' depending on the model type.

    Args:
        registry: Model Registry instance.
        model_name: Registered model name.
        version_name: Version to inspect.

    Returns:
        List of function name strings.
    """
    version = registry.get_model(model_name).version(version_name)
    functions: list[str] = version.show_functions()
    return functions


def predict_proba(
    registry: Registry,
    model_name: str,
    version_name: str,
    input_df: DataFrame,
) -> DataFrame:
    """Run probability predictions using a registered model.

    Unlike predict() which returns class labels, predict_proba()
    returns probability scores for each class.

    Args:
        registry: Model Registry instance.
        model_name: Registered model name.
        version_name: Version to use.
        input_df: Data to score.

    Returns:
        DataFrame with probability columns.
    """
    version = registry.get_model(model_name).version(version_name)
    result: DataFrame = version.run(input_df, function_name="predict_proba")
    return result
