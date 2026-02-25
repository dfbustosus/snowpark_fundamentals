"""Model evaluation utilities for Snowpark ML.

Provides functions to evaluate binary classification models
using metrics computed on Snowpark DataFrames.
"""

from __future__ import annotations

from typing import Any

from snowflake.ml.modeling.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from snowflake.snowpark import DataFrame


def evaluate_binary_classifier(
    df: DataFrame,
    label_col: str,
    prediction_col: str,
) -> dict[str, float]:
    """Evaluate a binary classification model.

    Computes standard classification metrics using Snowpark ML's
    distributed metrics functions.

    Args:
        df: DataFrame with actual labels and predictions.
        label_col: Name of the true label column.
        prediction_col: Name of the prediction column.

    Returns:
        Dict with accuracy, precision, recall, f1_score.
    """
    acc = accuracy_score(df=df, y_true_col_names=label_col, y_pred_col_names=prediction_col)
    prec = precision_score(
        df=df, y_true_col_names=label_col, y_pred_col_names=prediction_col, average="binary"
    )
    rec = recall_score(
        df=df, y_true_col_names=label_col, y_pred_col_names=prediction_col, average="binary"
    )
    f1 = f1_score(
        df=df, y_true_col_names=label_col, y_pred_col_names=prediction_col, average="binary"
    )

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
    }


def get_confusion_matrix(
    df: DataFrame,
    label_col: str,
    prediction_col: str,
) -> Any:
    """Compute confusion matrix for a binary classifier.

    Args:
        df: DataFrame with labels and predictions.
        label_col: True label column.
        prediction_col: Prediction column.

    Returns:
        Confusion matrix as a numpy array.
    """
    return confusion_matrix(df=df, y_true_col_name=label_col, y_pred_col_name=prediction_col)


def get_feature_importance(model: Any, feature_cols: list[str]) -> list[dict[str, Any]]:
    """Extract feature importance from a tree-based model.

    Works with XGBoost, RandomForest, and other tree-based models
    that expose feature_importances_.

    Args:
        model: Fitted model with feature_importances_ attribute.
        feature_cols: List of feature column names (in training order).

    Returns:
        List of dicts with feature name and importance, sorted descending.
    """
    # XGBClassifier uses to_xgboost(), others use to_sklearn()
    if hasattr(model, "to_xgboost"):
        native_model = model.to_xgboost()
    else:
        native_model = model.to_sklearn()
    importances = native_model.feature_importances_

    result = [
        {"feature": name, "importance": round(float(imp), 4)}
        for name, imp in zip(feature_cols, importances)
    ]
    return sorted(result, key=lambda x: x["importance"], reverse=True)
