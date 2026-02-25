"""Model training, evaluation, and pipeline construction with Snowpark ML."""

from snowpark_fundamentals.modeling.evaluation import (
    evaluate_binary_classifier,
    get_feature_importance,
)
from snowpark_fundamentals.modeling.pipeline import build_ml_pipeline
from snowpark_fundamentals.modeling.trainer import train_model

__all__ = [
    "train_model",
    "evaluate_binary_classifier",
    "get_feature_importance",
    "build_ml_pipeline",
]
