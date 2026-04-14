"""Feature engineering patterns for Snowpark ML.

Demonstrates how to create derived and interaction features
using Snowpark DataFrame operations, all executed server-side.
"""

from __future__ import annotations

from snowflake.snowpark import DataFrame
from snowflake.snowpark import functions as F


def create_derived_features(df: DataFrame) -> DataFrame:
    """Create derived features from the customer churn dataset.

    Demonstrates common feature engineering patterns:
    - Ratios between existing columns
    - Binning continuous variables
    - Conditional flag creation

    Args:
        df: Customer churn DataFrame with standard columns.

    Returns:
        DataFrame with additional engineered features.
    """
    result: DataFrame = (
        df.with_column(
            "CHARGES_PER_MONTH_TENURE",
            F.when(
                F.col("TENURE_MONTHS") > 0, F.col("TOTAL_CHARGES") / F.col("TENURE_MONTHS")
            ).otherwise(F.lit(0.0)),
        )
        .with_column(
            "TICKETS_PER_TENURE",
            F.when(
                F.col("TENURE_MONTHS") > 0,
                F.col("SUPPORT_TICKETS") / F.col("TENURE_MONTHS"),
            ).otherwise(F.lit(0.0)),
        )
        .with_column(
            "TENURE_BUCKET",
            F.when(F.col("TENURE_MONTHS") < 12, F.lit("NEW"))
            .when(F.col("TENURE_MONTHS") < 36, F.lit("MEDIUM"))
            .when(F.col("TENURE_MONTHS") < 60, F.lit("ESTABLISHED"))
            .otherwise(F.lit("LOYAL")),
        )
        .with_column(
            "HIGH_VALUE_FLAG",
            F.when(F.col("MONTHLY_CHARGES") > 150, F.lit(1)).otherwise(F.lit(0)),
        )
        .with_column(
            "AT_RISK_FLAG",
            F.when(
                (F.col("CONTRACT_TYPE") == "MONTH_TO_MONTH")
                & (F.col("SUPPORT_TICKETS") > 3)
                & (F.col("TENURE_MONTHS") < 24),
                F.lit(1),
            ).otherwise(F.lit(0)),
        )
    )
    return result


def create_interaction_features(
    df: DataFrame,
    col_pairs: list[tuple[str, str]],
) -> DataFrame:
    """Create interaction features (product of column pairs).

    Args:
        df: Input DataFrame.
        col_pairs: List of (col_a, col_b) tuples to multiply.

    Returns:
        DataFrame with added interaction columns.

    Notes:
        Interaction columns use ``_X_`` in their canonical name.
    """
    for col_a, col_b in col_pairs:
        interaction_expr = F.col(col_a) * F.col(col_b)
        interaction_name = f"{col_a}_X_{col_b}"
        df = df.with_column(interaction_name, interaction_expr)
    return df
