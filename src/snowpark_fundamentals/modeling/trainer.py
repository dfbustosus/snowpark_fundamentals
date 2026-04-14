"""Model training with Snowpark ML.

Demonstrates training ML models using snowflake.ml.modeling,
which provides scikit-learn-compatible APIs that execute
distributed training inside Snowflake warehouses.
"""

from __future__ import annotations

import warnings
from typing import Any

from snowflake.ml.modeling.ensemble import RandomForestClassifier
from snowflake.ml.modeling.linear_model import LogisticRegression
from snowflake.ml.modeling.xgboost import XGBClassifier
from snowflake.snowpark import DataFrame

# Suppress Snowpark ML version-mismatch warnings between local packages
# and the Snowflake server-side conda channel. These are informational only;
# the server uses its own package versions regardless of what's installed locally.
warnings.filterwarnings("ignore", message=".*does not fit the criteria.*")
warnings.filterwarnings("ignore", message=".*is not installed in the local environment.*")

MODEL_REGISTRY = {
    "xgboost": XGBClassifier,
    "random_forest": RandomForestClassifier,
    "logistic_regression": LogisticRegression,
}


def _resolve_dataframe_columns(df: DataFrame, columns: list[str]) -> list[str]:
    """Resolve requested column names against a Snowpark DataFrame.

    Snowflake commonly normalizes unquoted identifiers to uppercase. This helper
    keeps notebook code resilient by matching requested column names
    case-insensitively against the actual DataFrame schema.
    """
    available_columns = list(df.columns)
    exact_map = {col: col for col in available_columns}
    normalized_map = {col.upper(): col for col in available_columns}

    resolved: list[str] = []
    missing: list[str] = []
    for column in columns:
        if column in exact_map:
            resolved.append(exact_map[column])
        elif column.upper() in normalized_map:
            resolved.append(normalized_map[column.upper()])
        else:
            missing.append(column)

    if missing:
        raise ValueError(
            f"Columns not found in DataFrame: {missing}. Available columns: {available_columns}"
        )

    return resolved


def train_model(
    train_df: DataFrame,
    feature_cols: list[str],
    label_col: str,
    model_type: str = "xgboost",
    output_col: str = "PREDICTION",
    model_params: dict[str, Any] | None = None,
) -> Any:
    """Train a classification model using Snowpark ML.

    Training happens inside the Snowflake warehouse, not locally.
    The API mirrors scikit-learn's fit/predict pattern.

    Args:
        train_df: Training DataFrame with features and label.
        feature_cols: List of feature column names.
        label_col: Name of the target/label column.
        model_type: One of 'xgboost', 'random_forest', 'logistic_regression'.
        output_col: Name for the prediction output column.
        model_params: Optional hyperparameters to pass to the model.

    Returns:
        Fitted model instance.

    Raises:
        ValueError: If model_type is not supported.
    """
    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unsupported model type: '{model_type}'. "
            f"Choose from: {list(MODEL_REGISTRY.keys())}"
        )

    model_cls = MODEL_REGISTRY[model_type]
    params = model_params or {}
    resolved_feature_cols = _resolve_dataframe_columns(train_df, feature_cols)
    resolved_label_col = _resolve_dataframe_columns(train_df, [label_col])[0]

    model = model_cls(
        input_cols=resolved_feature_cols,
        label_cols=[resolved_label_col],
        output_cols=[output_col],
        **params,
    )

    model.fit(train_df)
    return model


def predict(model: Any, df: DataFrame) -> DataFrame:
    """Run predictions using a trained Snowpark ML model.

    Args:
        model: Fitted Snowpark ML model.
        df: DataFrame to score.

    Returns:
        DataFrame with prediction column added.
    """
    result: DataFrame = model.predict(df)
    return result
