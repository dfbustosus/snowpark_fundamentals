"""Feature View creation and registration.

Demonstrates managed and external FeatureViews in the Snowflake
Feature Store API. Managed views are computed by Snowflake on a
schedule; external views wrap pre-computed tables (e.g., dbt output).
"""

from __future__ import annotations

from typing import Any

from snowflake.ml.feature_store import Entity, FeatureStore, FeatureView
from snowflake.snowpark import DataFrame


def create_managed_feature_view(
    name: str,
    entities: list[Entity],
    feature_df: DataFrame,
    refresh_freq: str = "1 day",
    desc: str = "",
    timestamp_col: str | None = None,
) -> FeatureView:
    """Create a managed FeatureView backed by a query.

    Snowflake materializes and refreshes the features on the
    given schedule. Best for features derived from raw source
    tables where Snowflake controls the computation.

    Args:
        name: Feature view name.
        entities: List of Entity objects defining the join keys.
        feature_df: Snowpark DataFrame defining the feature query.
        refresh_freq: Refresh schedule (e.g., '1 day', '1 hour').
        desc: Human-readable description.
        timestamp_col: Column name for point-in-time correct lookups.

    Returns:
        FeatureView object (not yet registered).
    """
    fv: FeatureView = FeatureView(
        name=name,
        entities=entities,
        feature_df=feature_df,
        refresh_freq=refresh_freq,
        desc=desc,
        timestamp_col=timestamp_col,
    )
    return fv


def create_external_feature_view(
    name: str,
    entities: list[Entity],
    feature_df: DataFrame,
    desc: str = "",
    timestamp_col: str | None = None,
) -> FeatureView:
    """Create an external FeatureView wrapping a pre-computed table.

    External views have no refresh schedule (refresh_freq=None) because
    the data is managed by an external system like dbt. This is the
    recommended pattern for integrating dbt feature pipelines.

    Args:
        name: Feature view name.
        entities: List of Entity objects defining the join keys.
        feature_df: Snowpark DataFrame pointing to the pre-computed table.
        desc: Human-readable description.
        timestamp_col: Column name for point-in-time correct lookups.

    Returns:
        FeatureView object (not yet registered).
    """
    fv: FeatureView = FeatureView(
        name=name,
        entities=entities,
        feature_df=feature_df,
        refresh_freq=None,
        desc=desc,
        timestamp_col=timestamp_col,
    )
    return fv


def register_feature_view(
    fs: FeatureStore,
    feature_view: FeatureView,
    version: str = "V1",
    overwrite: bool = False,
) -> FeatureView:
    """Register a FeatureView in the Feature Store.

    Registration materializes the feature view and makes it
    available for training set generation and inference lookups.

    Args:
        fs: Initialized FeatureStore instance.
        feature_view: FeatureView to register.
        version: Version label.
        overwrite: If True, replace an existing version.

    Returns:
        Registered FeatureView reference.
    """
    registered: FeatureView = fs.register_feature_view(
        feature_view=feature_view,
        version=version,
        overwrite=overwrite,
    )
    return registered


def get_feature_view(fs: FeatureStore, name: str, version: str) -> FeatureView:
    """Retrieve a registered FeatureView by name and version.

    Args:
        fs: Initialized FeatureStore instance.
        name: Feature view name.
        version: Version label.

    Returns:
        Registered FeatureView.
    """
    return fs.get_feature_view(name, version)


def list_feature_views(fs: FeatureStore) -> Any:
    """List all feature views registered in the Feature Store.

    Args:
        fs: Initialized FeatureStore instance.

    Returns:
        DataFrame with feature view metadata.
    """
    return fs.list_feature_views()


def delete_feature_view(fs: FeatureStore, name: str, version: str) -> None:
    """Delete a feature view from the Feature Store.

    Args:
        fs: Initialized FeatureStore instance.
        name: Feature view name.
        version: Version label.
    """
    fs.delete_feature_view(name, version)
