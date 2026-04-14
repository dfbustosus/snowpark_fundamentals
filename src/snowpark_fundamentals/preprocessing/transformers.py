"""Snowpark ML preprocessing transformers.

Demonstrates how to use snowflake.ml.modeling.preprocessing
for distributed data transformation that runs entirely in Snowflake.
"""

from __future__ import annotations

from snowflake.ml.modeling.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)
from snowflake.snowpark import DataFrame
from snowflake.snowpark import functions as F
from snowflake.snowpark.types import DoubleType


def scale_numeric_features(
    df: DataFrame,
    input_cols: list[str],
    output_cols: list[str] | None = None,
    method: str = "standard",
) -> tuple[DataFrame, StandardScaler | MinMaxScaler]:
    """Scale numeric features using Snowpark ML scalers.

    All computation runs distributed inside Snowflake's warehouse,
    not on the client machine. Output columns are cast to DoubleType
    to avoid Decimal precision warnings during model training.

    Args:
        df: Input Snowpark DataFrame.
        input_cols: Numeric columns to scale.
        output_cols: Output column names. Defaults to INPUT_COL_SCALED.
        method: 'standard' (zero mean, unit variance) or 'minmax' (0 to 1).

    Returns:
        Tuple of (transformed DataFrame, fitted scaler).
    """
    if output_cols is None:
        output_cols = [f"{col}_SCALED" for col in input_cols]

    scaler_cls = StandardScaler if method == "standard" else MinMaxScaler
    scaler = scaler_cls(input_cols=input_cols, output_cols=output_cols)

    transformed_df = scaler.fit(df).transform(df)

    # Cast scaled columns from Decimal to Double to prevent precision warnings
    for col in output_cols:
        transformed_df = transformed_df.with_column(col, F.col(col).cast(DoubleType()))

    return transformed_df, scaler


def encode_categorical_features(
    df: DataFrame,
    input_cols: list[str],
    output_cols: list[str] | None = None,
    method: str = "onehot",
) -> tuple[DataFrame, OneHotEncoder | OrdinalEncoder]:
    """Encode categorical features using Snowpark ML encoders.

    Args:
        df: Input Snowpark DataFrame.
        input_cols: Categorical columns to encode.
        output_cols: Output column names.
        method: 'onehot' or 'ordinal'.

    Returns:
        Tuple of (transformed DataFrame, fitted encoder).
    """
    if output_cols is None:
        output_cols = [f"{col}_ENCODED" for col in input_cols]

    encoder_cls = OneHotEncoder if method == "onehot" else OrdinalEncoder
    encoder = encoder_cls(input_cols=input_cols, output_cols=output_cols)

    transformed_df = encoder.fit(df).transform(df)
    return transformed_df, encoder


def build_preprocessing_pipeline(
    df: DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    numeric_method: str = "standard",
    categorical_method: str = "ordinal",
) -> tuple[DataFrame, dict]:
    """Build a complete preprocessing pipeline for numeric + categorical features.

    Applies scaling to numeric columns and encoding to categorical columns
    in sequence. Returns the transformed DataFrame and all fitted transformers
    for later reuse on new data.

    Args:
        df: Input DataFrame.
        numeric_cols: Columns to scale.
        categorical_cols: Columns to encode.
        numeric_method: Scaling method ('standard' or 'minmax').
        categorical_method: Encoding method ('onehot' or 'ordinal').

    Returns:
        Tuple of (transformed DataFrame, dict of fitted transformers).
    """
    transformers = {}

    df_processed, scaler = scale_numeric_features(df, numeric_cols, method=numeric_method)
    transformers["scaler"] = scaler

    df_processed, encoder = encode_categorical_features(
        df_processed, categorical_cols, method=categorical_method
    )
    transformers["encoder"] = encoder

    return df_processed, transformers


def apply_preprocessing_pipeline(
    df: DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    transformers: dict,
) -> DataFrame:
    """Apply previously fitted preprocessing transformers to new data.

    Use this on validation/test/inference data so scaling and encoding are
    driven by the transformers fitted on the training split rather than being
    refit on each dataset independently.

    Args:
        df: Input DataFrame to transform.
        numeric_cols: Numeric columns used during fitting.
        categorical_cols: Categorical columns used during fitting.
        transformers: Dict returned by build_preprocessing_pipeline().

    Returns:
        Transformed DataFrame with the fitted scaler/encoder outputs added.
    """
    df_processed = df

    scaler = transformers.get("scaler")
    if scaler is not None and numeric_cols:
        df_processed = scaler.transform(df_processed)
        for col in [f"{feature}_SCALED" for feature in numeric_cols]:
            if col in df_processed.columns:
                df_processed = df_processed.with_column(col, F.col(col).cast(DoubleType()))

    encoder = transformers.get("encoder")
    if encoder is not None and categorical_cols:
        df_processed = encoder.transform(df_processed)

    return df_processed
