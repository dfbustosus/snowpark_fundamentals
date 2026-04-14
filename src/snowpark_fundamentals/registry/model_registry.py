"""Snowflake Model Registry operations.

Demonstrates how to register, version, and deploy ML models
using Snowflake's native Model Registry (snowflake.ml.registry).
"""

from __future__ import annotations

import json
import warnings
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
    normalized_columns = {str(col).strip('"').upper(): str(col) for col in result.columns}
    prediction_col = normalized_columns.get("PREDICTION")
    if prediction_col == "PREDICTION":
        return result
    if prediction_col is not None:
        return result.with_column_renamed(prediction_col, "PREDICTION")

    fallback_col = normalized_columns.get("OUTPUT_FEATURE_0")
    if fallback_col is not None:
        return result.with_column_renamed(fallback_col, "PREDICTION")
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

    Aliases provide stable references for deployment. If the alias
    already exists on another version, it is first removed from the
    old version before being set on the target version.

    Args:
        registry: Model Registry instance.
        model_name: Registered model name.
        version_name: Version to tag.
        alias: Alias string (e.g., 'production').
    """
    model_ref = registry.get_model(model_name)
    alias_upper = alias.upper()

    # Check existing versions for the alias and unset if on a different version
    versions_df = model_ref.show_versions()
    for _, row in versions_df.iterrows():
        aliases_raw = row.get("aliases", "[]")
        try:
            current_aliases = (
                json.loads(aliases_raw) if isinstance(aliases_raw, str) else (aliases_raw or [])
            )
        except (json.JSONDecodeError, TypeError):
            current_aliases = []
        if alias_upper in current_aliases:
            if row["name"] == version_name:
                return  # Already on target version
            old_version = model_ref.version(row["name"])
            old_version.unset_alias(alias)
            break

    target_version = model_ref.version(version_name)
    target_version.set_alias(alias)


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


def _quote_identifier(identifier: str) -> str:
    """Return a safely quoted Snowflake identifier."""
    escaped = identifier.replace('"', '""')
    return f'"{escaped}"'


def _escape_sql_string(value: str) -> str:
    """Escape a string for use in a single-quoted SQL literal."""
    return value.replace("'", "''")


def _get_registry_session(registry: Registry) -> Session:
    """Resolve the Snowpark session from a registry instance."""
    model_manager = getattr(registry, "_model_manager", None)
    model_ops = getattr(model_manager, "_model_ops", None) if model_manager else None
    session = getattr(model_ops, "_session", None)
    if session is None:
        raise RuntimeError("Unable to resolve Snowpark session from registry.")
    return session


def _is_missing_tag_error(error: Exception) -> bool:
    """Return True when the error indicates the tag does not exist."""
    message = str(error).lower()
    return "tag" in message and "does not exist" in message


def _ensure_tag_exists(session: Session, registry: Registry, tag_name: str) -> None:
    """Create a tag if missing."""
    session.sql(
        f"CREATE TAG IF NOT EXISTS {registry.location}.{_quote_identifier(str(tag_name).upper())}"
    ).collect()


def _parse_model_version_metadata(metadata_value: Any) -> dict[str, Any]:
    """Parse the metadata column returned by SHOW VERSIONS."""
    if metadata_value in (None, ""):
        return {}
    if isinstance(metadata_value, str):
        parsed = json.loads(metadata_value)
        if not isinstance(parsed, dict):
            raise ValueError(f"Model version metadata must be a dict, got {type(parsed).__name__}.")
        return parsed
    if isinstance(metadata_value, dict):
        return dict(metadata_value)
    try:
        return dict(metadata_value)
    except TypeError as exc:
        raise ValueError(
            f"Model version metadata must be convert. to dict, got {type(metadata_value).__name__}."
        ) from exc


def set_model_tags(
    registry: Registry,
    model_name: str,
    tags: dict[str, str],
    *,
    strict: bool = False,
) -> None:
    """Set tags on a model for governance and discovery.

    Tags are model-level labels shared across every version of the model.
    Use version metadata or comments for per-experiment provenance.

    Args:
        registry: Model Registry instance.
        model_name: Registered model name.
        tags: Dict of tag key to value.
        strict: If True, raise when tag creation or assignment fails. If False,
            emit a warning and continue so experiment logging is not blocked by
            optional governance metadata.
    """
    if not tags:
        return

    session = _get_registry_session(registry)
    model_ref = registry.get_model(model_name)
    for key, value in tags.items():
        try:
            model_ref.set_tag(key, str(value))
        except Exception as exc:
            if _is_missing_tag_error(exc):
                try:
                    _ensure_tag_exists(session, registry, key)
                    model_ref.set_tag(key, str(value))
                    continue
                except Exception as create_exc:
                    message = (
                        f"Skip. model tag {key!r} on {model_name}:unable to create/assign the tag "
                        f"in {registry.location}. {create_exc}"
                    )
                    if strict:
                        raise RuntimeError(message) from create_exc
                    warnings.warn(message, stacklevel=2)
                    continue

            message = f"Skipping model tag {key!r} on {model_name}: {exc}"
            if strict:
                raise RuntimeError(message) from exc
            warnings.warn(message, stacklevel=2)


def get_model_version_metadata(
    registry: Registry,
    model_name: str,
    version_name: str,
) -> dict[str, Any]:
    """Get the raw metadata stored on a model version."""
    versions_df = registry.get_model(model_name).show_versions()
    target_version = version_name.upper()
    for _, row in versions_df.iterrows():
        if str(row.get("name", "")).upper() == target_version:
            return _parse_model_version_metadata(row.get("metadata"))
    raise ValueError(f"Version {version_name!r} not found for model {model_name!r}.")


def set_model_version_metadata(
    registry: Registry,
    model_name: str,
    version_name: str,
    metadata: dict[str, Any],
) -> None:
    """Merge structured metadata into a model version.

    This preserves existing version metadata such as metrics and adds the new
    keys on top. Use this for per-experiment provenance that should stay tied
    to a specific registered version.
    """
    if not metadata:
        return

    session = _get_registry_session(registry)
    model_ref = registry.get_model(model_name)
    current_metadata = get_model_version_metadata(registry, model_name, version_name)
    current_metadata.update(metadata)
    metadata_json = json.dumps(current_metadata, sort_keys=True)
    session.sql(
        " ".join(
            [
                f"ALTER MODEL {model_ref.fully_qualified_name}",
                f"MODIFY VERSION {_quote_identifier(version_name)}",
                f"SET METADATA = '{_escape_sql_string(metadata_json)}'",
            ]
        )
    ).collect()


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
