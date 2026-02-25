"""Data loading utilities for Snowpark DataFrames.

Demonstrates best practices for loading data from Snowflake tables
into Snowpark DataFrames for downstream ML processing.
"""

from __future__ import annotations

from snowflake.snowpark import DataFrame, Session
from snowflake.snowpark import functions as F


def load_table(session: Session, table_name: str) -> DataFrame:
    """Load a Snowflake table as a Snowpark DataFrame.

    Args:
        session: Active Snowpark session.
        table_name: Fully qualified table name (e.g., 'DB.SCHEMA.TABLE').

    Returns:
        Snowpark DataFrame pointing to the table.
    """
    return session.table(table_name)


def load_with_sql(session: Session, query: str) -> DataFrame:
    """Load data using a custom SQL query.

    Args:
        session: Active Snowpark session.
        query: SQL SELECT statement.

    Returns:
        Snowpark DataFrame with query results.
    """
    return session.sql(query)


def explore_dataframe(df: DataFrame) -> dict:
    """Generate a summary profile of a Snowpark DataFrame.

    Demonstrates key DataFrame exploration methods that participants
    should know for data understanding and EDA.

    Args:
        df: Snowpark DataFrame to explore.

    Returns:
        Dict with row count, column count, column names, and dtypes.
    """
    return {
        "row_count": df.count(),
        "column_count": len(df.columns),
        "columns": df.columns,
        "dtypes": {field.name: str(field.datatype) for field in df.schema.fields},
    }


def split_data(
    df: DataFrame,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[DataFrame, DataFrame]:
    """Split a DataFrame into train and test sets using random sampling.

    Uses Snowpark's native random split functionality to ensure
    the split happens server-side in Snowflake.

    Args:
        df: Input DataFrame to split.
        train_ratio: Fraction of data for training (0.0 to 1.0).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_df, test_df).
    """
    test_ratio = 1.0 - train_ratio
    train_df, test_df = df.random_split([train_ratio, test_ratio], seed=seed)
    return train_df, test_df


def add_row_index(df: DataFrame, column_name: str = "ROW_INDEX") -> DataFrame:
    """Add a monotonically increasing row index to a DataFrame.

    Args:
        df: Input DataFrame.
        column_name: Name for the index column.

    Returns:
        DataFrame with added index column.
    """
    return df.with_column(column_name, F.monotonically_increasing_id())
