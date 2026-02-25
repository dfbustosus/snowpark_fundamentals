"""Sample data generation for the tutorial.

Creates synthetic datasets directly in Snowflake using SQL,
avoiding any local data dependencies. All tables are created
in the session's current database and schema.
"""

from __future__ import annotations

from snowflake.snowpark import DataFrame, Session
from snowflake.snowpark import functions as F


def _get_fqn(session: Session, table_name: str) -> str:
    """Build a fully-qualified table name from the session context.

    Args:
        session: Active Snowpark session.
        table_name: Short table name (no database/schema prefix).

    Returns:
        Fully-qualified name like 'MLDS_D.PREDICTIONS.TABLE_NAME'.
    """
    db = (session.get_current_database() or "").replace('"', "")
    schema = (session.get_current_schema() or "").replace('"', "")
    return f"{db}.{schema}.{table_name}"


def create_customer_churn_dataset(
    session: Session,
    table_name: str = "CUSTOMER_CHURN_DATA",
    n_rows: int = 5000,
    seed: int = 42,
) -> DataFrame:
    """Create a synthetic customer churn dataset in Snowflake.

    Generates realistic customer data with features suitable for
    binary classification (churn prediction). All computation
    runs server-side in Snowflake.

    The table is created in the session's current database/schema
    (configured via .env).

    Args:
        session: Active Snowpark session.
        table_name: Name of the table to create (no db/schema prefix needed).
        n_rows: Number of rows to generate.
        seed: Random seed for reproducibility.

    Returns:
        Snowpark DataFrame with the generated data.
    """
    fqn = _get_fqn(session, table_name)

    # Each column uses a different constant seed for independent randomness.
    # RANDOM(constant) produces a different value per row in GENERATOR context;
    # the constant seed controls reproducibility, not per-row variation.
    session.sql(f"""
        CREATE OR REPLACE TABLE {fqn} AS
        WITH base AS (
            SELECT
                SEQ4() AS CUSTOMER_ID,
                UNIFORM(18, 75, RANDOM({seed})) AS AGE,
                CASE UNIFORM(1, 3, RANDOM({seed + 1}))
                    WHEN 1 THEN 'BASIC'
                    WHEN 2 THEN 'STANDARD'
                    WHEN 3 THEN 'PREMIUM'
                END AS PLAN_TYPE,
                ROUND(UNIFORM(1, 120, RANDOM({seed + 2}))::FLOAT, 0) AS TENURE_MONTHS,
                ROUND(UNIFORM(20, 300, RANDOM({seed + 3}))::FLOAT, 2) AS MONTHLY_CHARGES,
                UNIFORM(0, 10, RANDOM({seed + 4})) AS SUPPORT_TICKETS,
                UNIFORM(0, 50, RANDOM({seed + 5})) AS USAGE_HOURS_PER_WEEK,
                CASE UNIFORM(1, 3, RANDOM({seed + 6}))
                    WHEN 1 THEN 'MONTH_TO_MONTH'
                    WHEN 2 THEN 'ONE_YEAR'
                    WHEN 3 THEN 'TWO_YEAR'
                END AS CONTRACT_TYPE,
                CASE UNIFORM(1, 4, RANDOM({seed + 7}))
                    WHEN 1 THEN 'CREDIT_CARD'
                    WHEN 2 THEN 'BANK_TRANSFER'
                    WHEN 3 THEN 'ELECTRONIC_CHECK'
                    WHEN 4 THEN 'MAILED_CHECK'
                END AS PAYMENT_METHOD,
                UNIFORM(0, 5, RANDOM({seed + 8})) AS NUM_DEPENDENTS
            FROM TABLE(GENERATOR(ROWCOUNT => {n_rows}))
        )
        SELECT
            CUSTOMER_ID,
            AGE,
            PLAN_TYPE,
            TENURE_MONTHS,
            MONTHLY_CHARGES,
            SUPPORT_TICKETS,
            USAGE_HOURS_PER_WEEK,
            CONTRACT_TYPE,
            PAYMENT_METHOD,
            NUM_DEPENDENTS,
            ROUND(MONTHLY_CHARGES * TENURE_MONTHS, 2) AS TOTAL_CHARGES,
            -- Churn label: higher probability if short tenure, high charges, many tickets
            CASE
                WHEN TENURE_MONTHS < 12
                     AND SUPPORT_TICKETS > 5
                     AND CONTRACT_TYPE = 'MONTH_TO_MONTH' THEN 1
                WHEN TENURE_MONTHS < 6
                     AND MONTHLY_CHARGES > 200 THEN 1
                WHEN SUPPORT_TICKETS > 7
                     AND USAGE_HOURS_PER_WEEK < 10 THEN 1
                WHEN UNIFORM(0, 100, RANDOM({seed + 9})) < 15 THEN 1
                ELSE 0
            END AS CHURNED
        FROM base
    """).collect()

    return session.table(fqn)


def create_sample_orders_dataset(
    session: Session,
    table_name: str = "SAMPLE_ORDERS",
    n_rows: int = 10000,
    seed: int = 42,
) -> DataFrame:
    """Create a synthetic orders dataset for DataFrame demos.

    Replaces the need for SNOWFLAKE_SAMPLE_DATA - generates
    customer and order data directly in MLDS_D.

    Args:
        session: Active Snowpark session.
        table_name: Name for the orders table.
        n_rows: Number of rows to generate.
        seed: Random seed for reproducibility.

    Returns:
        Snowpark DataFrame with order data.
    """
    fqn = _get_fqn(session, table_name)

    session.sql(f"""
        CREATE OR REPLACE TABLE {fqn} AS
        SELECT
            SEQ4() AS ORDER_ID,
            UNIFORM(1, 1000, RANDOM({seed})) AS CUSTOMER_ID,
            DATEADD(DAY, -UNIFORM(1, 730, RANDOM({seed + 1})),
                    CURRENT_DATE()) AS ORDER_DATE,
            ROUND(UNIFORM(10, 5000, RANDOM({seed + 2}))::FLOAT, 2) AS ORDER_TOTAL,
            CASE UNIFORM(1, 5, RANDOM({seed + 3}))
                WHEN 1 THEN 'COMPLETED'
                WHEN 2 THEN 'SHIPPED'
                WHEN 3 THEN 'PROCESSING'
                WHEN 4 THEN 'CANCELLED'
                WHEN 5 THEN 'RETURNED'
            END AS ORDER_STATUS,
            CASE UNIFORM(1, 4, RANDOM({seed + 4}))
                WHEN 1 THEN 'ELECTRONICS'
                WHEN 2 THEN 'CLOTHING'
                WHEN 3 THEN 'HOME'
                WHEN 4 THEN 'SPORTS'
            END AS CATEGORY,
            UNIFORM(1, 10, RANDOM({seed + 5})) AS ITEM_COUNT
        FROM TABLE(GENERATOR(ROWCOUNT => {n_rows}))
    """).collect()

    return session.table(fqn)


def create_sample_customers_dataset(
    session: Session,
    table_name: str = "SAMPLE_CUSTOMERS",
    n_rows: int = 1000,
    seed: int = 42,
) -> DataFrame:
    """Create a synthetic customers dataset for DataFrame demos.

    Args:
        session: Active Snowpark session.
        table_name: Name for the customers table.
        n_rows: Number of rows to generate.
        seed: Random seed for reproducibility.

    Returns:
        Snowpark DataFrame with customer data.
    """
    fqn = _get_fqn(session, table_name)

    session.sql(f"""
        CREATE OR REPLACE TABLE {fqn} AS
        SELECT
            SEQ4() AS CUSTOMER_ID,
            'CUSTOMER_' || LPAD(SEQ4()::VARCHAR, 5, '0') AS CUSTOMER_NAME,
            UNIFORM(18, 80, RANDOM({seed})) AS AGE,
            CASE UNIFORM(1, 5, RANDOM({seed + 1}))
                WHEN 1 THEN 'AUTOMOBILE'
                WHEN 2 THEN 'BUILDING'
                WHEN 3 THEN 'FURNITURE'
                WHEN 4 THEN 'MACHINERY'
                WHEN 5 THEN 'HOUSEHOLD'
            END AS SEGMENT,
            ROUND(UNIFORM(-500, 10000, RANDOM({seed + 2}))::FLOAT, 2) AS ACCOUNT_BALANCE,
            CASE UNIFORM(1, 5, RANDOM({seed + 3}))
                WHEN 1 THEN 'US'
                WHEN 2 THEN 'UK'
                WHEN 3 THEN 'DE'
                WHEN 4 THEN 'FR'
                WHEN 5 THEN 'JP'
            END AS COUNTRY
        FROM TABLE(GENERATOR(ROWCOUNT => {n_rows}))
    """).collect()

    return session.table(fqn)


def get_dataset_summary(df: DataFrame) -> DataFrame:
    """Get summary statistics for the churn dataset.

    Args:
        df: Customer churn DataFrame.

    Returns:
        DataFrame with churn distribution and key stats.
    """
    result: DataFrame = df.group_by("CHURNED").agg(
        F.count("*").alias("COUNT"),
        F.avg("AGE").alias("AVG_AGE"),
        F.avg("TENURE_MONTHS").alias("AVG_TENURE"),
        F.avg("MONTHLY_CHARGES").alias("AVG_MONTHLY_CHARGES"),
        F.avg("SUPPORT_TICKETS").alias("AVG_SUPPORT_TICKETS"),
    )
    return result
