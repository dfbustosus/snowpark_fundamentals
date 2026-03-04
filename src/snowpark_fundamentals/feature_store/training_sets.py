"""Training set generation from the Feature Store.

Demonstrates point-in-time correct training set creation and
feature retrieval for inference — the core value proposition
of a Feature Store.
"""

from __future__ import annotations

from typing import Any

from snowflake.ml.feature_store import FeatureStore, FeatureView
from snowflake.snowpark import DataFrame, Session


def create_spine_dataframe(
    session: Session,
    entity_table: str,
    entity_key: str = "CUSTOMER_ID",
    timestamp_col: str | None = None,
) -> DataFrame:
    """Create a spine DataFrame for training set generation.

    The spine defines which entity keys (and optionally timestamps)
    to look up features for. For point-in-time joins, include a
    timestamp column.

    Args:
        session: Active Snowpark session.
        entity_table: Table containing entity keys.
        entity_key: Column name for the entity key.
        timestamp_col: Optional timestamp column for point-in-time joins.

    Returns:
        Snowpark DataFrame suitable as a spine for generate_training_set().
    """
    db = (session.get_current_database() or "").replace('"', "")
    schema = (session.get_current_schema() or "").replace('"', "")
    fqn = f"{db}.{schema}.{entity_table}"

    if timestamp_col:
        spine: DataFrame = session.table(fqn).select(entity_key, timestamp_col)
    else:
        spine = session.table(fqn).select(entity_key).distinct()

    return spine


def generate_training_set(
    fs: FeatureStore,
    spine_df: DataFrame,
    features: list[FeatureView],
    name: str = "TRAINING_SET",
    spine_timestamp_col: str | None = None,
    spine_label_cols: list[str] | None = None,
) -> Any:
    """Generate a point-in-time correct training set.

    Joins the spine DataFrame with one or more registered
    FeatureViews. When spine_timestamp_col is provided (and the
    FeatureView has a timestamp_col), the join is point-in-time
    correct — preventing data leakage.

    Args:
        fs: Initialized FeatureStore instance.
        spine_df: Spine DataFrame with entity keys (and optional timestamps).
        features: List of registered FeatureView objects to include.
        name: Dataset name for the materialized training set.
        spine_timestamp_col: Column in spine_df containing the event timestamp
            for point-in-time joins.
        spine_label_cols: Columns in spine_df that are labels (not features).

    Returns:
        Training set Dataset object with joined features.
    """
    training_set: Any = fs.generate_dataset(
        name=name,
        spine_df=spine_df,
        features=features,
        spine_timestamp_col=spine_timestamp_col,
        spine_label_cols=spine_label_cols,
    )
    return training_set


def retrieve_feature_values(
    fs: FeatureStore,
    spine_df: DataFrame,
    features: list[FeatureView],
    spine_timestamp_col: str | None = None,
) -> DataFrame:
    """Retrieve feature values for inference.

    Similar to generate_training_set but returns a DataFrame
    for immediate scoring. Use this at inference time to look
    up the latest feature values for a set of entity keys.

    Args:
        fs: Initialized FeatureStore instance.
        spine_df: DataFrame with entity keys to look up.
        features: List of registered FeatureView objects.
        spine_timestamp_col: Column in spine_df for point-in-time lookups.

    Returns:
        Snowpark DataFrame with joined feature values.
    """
    result: DataFrame = fs.retrieve_feature_values(
        spine_df=spine_df,
        features=features,
        spine_timestamp_col=spine_timestamp_col,
    )
    return result
