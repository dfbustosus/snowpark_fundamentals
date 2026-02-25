"""Data preprocessing and feature engineering with Snowpark ML."""

from snowpark_fundamentals.preprocessing.feature_engineering import (
    create_derived_features,
    create_interaction_features,
)
from snowpark_fundamentals.preprocessing.transformers import (
    build_preprocessing_pipeline,
    encode_categorical_features,
    scale_numeric_features,
)

__all__ = [
    "scale_numeric_features",
    "encode_categorical_features",
    "build_preprocessing_pipeline",
    "create_derived_features",
    "create_interaction_features",
]
