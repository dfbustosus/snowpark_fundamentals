"""ML Pipeline construction with Snowpark ML.

Demonstrates how to build end-to-end ML pipelines using
snowflake.ml.modeling.pipeline, mirroring scikit-learn's Pipeline API.
"""

from __future__ import annotations

from typing import Any

from snowflake.ml.modeling.pipeline import Pipeline
from snowflake.ml.modeling.preprocessing import OrdinalEncoder, StandardScaler
from snowflake.ml.modeling.xgboost import XGBClassifier
from snowflake.snowpark import DataFrame


def build_ml_pipeline(
    numeric_cols: list[str],
    categorical_cols: list[str],
    label_col: str,
    output_col: str = "PREDICTION",
    model_params: dict[str, Any] | None = None,
) -> Pipeline:
    """Build a complete ML pipeline with preprocessing and model.

    Creates a Snowpark ML Pipeline that chains:
    1. StandardScaler for numeric features
    2. OrdinalEncoder for categorical features
    3. XGBClassifier for prediction

    Args:
        numeric_cols: Columns to scale.
        categorical_cols: Columns to encode.
        label_col: Target column name.
        output_col: Prediction output column name.
        model_params: Optional XGBoost hyperparameters.

    Returns:
        Configured (unfitted) Snowpark ML Pipeline.
    """
    scaled_cols = [f"{c}_SCALED" for c in numeric_cols]
    encoded_cols = [f"{c}_ENCODED" for c in categorical_cols]
    all_feature_cols = scaled_cols + encoded_cols

    params = model_params or {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
    }

    pipeline = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(
                    input_cols=numeric_cols,
                    output_cols=scaled_cols,
                ),
            ),
            (
                "encoder",
                OrdinalEncoder(
                    input_cols=categorical_cols,
                    output_cols=encoded_cols,
                ),
            ),
            (
                "classifier",
                XGBClassifier(
                    input_cols=all_feature_cols,
                    label_cols=[label_col],
                    output_cols=[output_col],
                    **params,
                ),
            ),
        ]
    )

    return pipeline


def fit_and_predict(
    pipeline: Pipeline,
    train_df: DataFrame,
    test_df: DataFrame,
) -> tuple[Pipeline, DataFrame]:
    """Fit the pipeline on training data and predict on test data.

    Args:
        pipeline: Unfitted Snowpark ML Pipeline.
        train_df: Training DataFrame.
        test_df: Test DataFrame.

    Returns:
        Tuple of (fitted pipeline, predictions DataFrame).
    """
    pipeline.fit(train_df)
    predictions = pipeline.predict(test_df)
    return pipeline, predictions
