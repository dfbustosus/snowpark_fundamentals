"""Synthetic temporal data generation for Feature Store tutorials.

Creates datasets with time-series characteristics (transaction dates,
interaction timestamps) directly in Snowflake using SQL. All computation
runs server-side — no local data dependencies.
"""

from __future__ import annotations

from snowflake.snowpark import DataFrame, Session


def _get_fqn(session: Session, table_name: str) -> str:
    """Build a fully-qualified table name from the session context.

    Args:
        session: Active Snowpark session.
        table_name: Short table name (no database/schema prefix).

    Returns:
        Fully-qualified name like 'DB.SCHEMA.TABLE_NAME'.
    """
    db = (session.get_current_database() or "").replace('"', "")
    schema = (session.get_current_schema() or "").replace('"', "")
    return f"{db}.{schema}.{table_name}"


def create_customer_transactions_dataset(
    session: Session,
    table_name: str = "CUSTOMER_TRANSACTIONS",
    n_rows: int = 50000,
    seed: int = 42,
) -> DataFrame:
    """Create a synthetic customer transactions dataset with temporal data.

    Generates realistic order/transaction data spanning 2 years with
    customer IDs, order dates, amounts, categories, and statuses.
    Suitable for time-windowed aggregations (RFM features).

    Args:
        session: Active Snowpark session.
        table_name: Name of the table to create.
        n_rows: Number of transaction rows to generate.
        seed: Random seed for reproducibility.

    Returns:
        Snowpark DataFrame pointing to the created table.
    """
    fqn = _get_fqn(session, table_name)

    session.sql(f"""
        CREATE OR REPLACE TABLE {fqn} AS
        SELECT
            SEQ4() AS TRANSACTION_ID,
            UNIFORM(1, 2000, RANDOM({seed})) AS CUSTOMER_ID,
            DATEADD(DAY, -UNIFORM(1, 730, RANDOM({seed + 1})),
                    CURRENT_DATE()) AS ORDER_DATE,
            ROUND(UNIFORM(5, 5000, RANDOM({seed + 2}))::FLOAT, 2) AS ORDER_AMOUNT,
            CASE UNIFORM(1, 6, RANDOM({seed + 3}))
                WHEN 1 THEN 'ELECTRONICS'
                WHEN 2 THEN 'CLOTHING'
                WHEN 3 THEN 'HOME'
                WHEN 4 THEN 'SPORTS'
                WHEN 5 THEN 'FOOD'
                WHEN 6 THEN 'TRAVEL'
            END AS CATEGORY,
            CASE UNIFORM(1, 4, RANDOM({seed + 4}))
                WHEN 1 THEN 'COMPLETED'
                WHEN 2 THEN 'COMPLETED'
                WHEN 3 THEN 'COMPLETED'
                WHEN 4 THEN 'CANCELLED'
            END AS ORDER_STATUS,
            UNIFORM(1, 10, RANDOM({seed + 5})) AS ITEM_COUNT,
            CASE UNIFORM(1, 3, RANDOM({seed + 6}))
                WHEN 1 THEN 'WEB'
                WHEN 2 THEN 'MOBILE'
                WHEN 3 THEN 'IN_STORE'
            END AS CHANNEL
        FROM TABLE(GENERATOR(ROWCOUNT => {n_rows}))
    """).collect()

    return session.table(fqn)


def create_customer_interactions_dataset(
    session: Session,
    table_name: str = "CUSTOMER_INTERACTIONS",
    n_rows: int = 100000,
    seed: int = 42,
) -> DataFrame:
    """Create a synthetic customer interactions dataset.

    Generates engagement events (page views, clicks, support tickets,
    emails) with timestamps. Suitable for behavioral feature engineering.

    Args:
        session: Active Snowpark session.
        table_name: Name of the table to create.
        n_rows: Number of interaction rows to generate.
        seed: Random seed for reproducibility.

    Returns:
        Snowpark DataFrame pointing to the created table.
    """
    fqn = _get_fqn(session, table_name)

    session.sql(f"""
        CREATE OR REPLACE TABLE {fqn} AS
        SELECT
            SEQ4() AS INTERACTION_ID,
            UNIFORM(1, 2000, RANDOM({seed})) AS CUSTOMER_ID,
            DATEADD(DAY, -UNIFORM(1, 365, RANDOM({seed + 1})),
                    CURRENT_DATE()) AS INTERACTION_DATE,
            CASE UNIFORM(1, 5, RANDOM({seed + 2}))
                WHEN 1 THEN 'PAGE_VIEW'
                WHEN 2 THEN 'CLICK'
                WHEN 3 THEN 'SUPPORT_TICKET'
                WHEN 4 THEN 'EMAIL_OPEN'
                WHEN 5 THEN 'EMAIL_CLICK'
            END AS INTERACTION_TYPE,
            CASE UNIFORM(1, 3, RANDOM({seed + 3}))
                WHEN 1 THEN 'WEB'
                WHEN 2 THEN 'MOBILE'
                WHEN 3 THEN 'EMAIL'
            END AS CHANNEL,
            UNIFORM(1, 300, RANDOM({seed + 4})) AS DURATION_SECONDS
        FROM TABLE(GENERATOR(ROWCOUNT => {n_rows}))
    """).collect()

    return session.table(fqn)


def create_rfm_features(
    session: Session,
    source_table: str = "CUSTOMER_TRANSACTIONS",
    target_table: str = "CUSTOMER_RFM_FEATURES",
) -> DataFrame:
    """Create RFM (Recency, Frequency, Monetary) features from transactions.

    Computes time-windowed aggregations at 30-day, 90-day, and 365-day
    windows — the same pattern used in production dbt feature stores.

    Args:
        session: Active Snowpark session.
        source_table: Name of the transactions source table.
        target_table: Name of the RFM features table to create.

    Returns:
        Snowpark DataFrame with RFM features per customer.
    """
    source_fqn = _get_fqn(session, source_table)
    target_fqn = _get_fqn(session, target_table)

    session.sql(f"""
        CREATE OR REPLACE TABLE {target_fqn} AS
        WITH completed_orders AS (
            SELECT * FROM {source_fqn}
            WHERE ORDER_STATUS = 'COMPLETED'
        )
        SELECT
            CUSTOMER_ID,

            -- Recency
            DATEDIFF('day', MAX(ORDER_DATE), CURRENT_DATE()) AS DAYS_SINCE_LAST_ORDER,

            -- Frequency: time-windowed order counts
            COUNT(DISTINCT CASE
                WHEN ORDER_DATE >= DATEADD('day', -30, CURRENT_DATE())
                THEN TRANSACTION_ID END) AS ORDERS_30D,
            COUNT(DISTINCT CASE
                WHEN ORDER_DATE >= DATEADD('day', -90, CURRENT_DATE())
                THEN TRANSACTION_ID END) AS ORDERS_90D,
            COUNT(DISTINCT CASE
                WHEN ORDER_DATE >= DATEADD('day', -365, CURRENT_DATE())
                THEN TRANSACTION_ID END) AS ORDERS_365D,
            COUNT(DISTINCT TRANSACTION_ID) AS ORDERS_TOTAL,

            -- Monetary: time-windowed spend
            COALESCE(SUM(CASE
                WHEN ORDER_DATE >= DATEADD('day', -30, CURRENT_DATE())
                THEN ORDER_AMOUNT END), 0) AS SPEND_30D,
            COALESCE(SUM(CASE
                WHEN ORDER_DATE >= DATEADD('day', -90, CURRENT_DATE())
                THEN ORDER_AMOUNT END), 0) AS SPEND_90D,
            COALESCE(SUM(CASE
                WHEN ORDER_DATE >= DATEADD('day', -365, CURRENT_DATE())
                THEN ORDER_AMOUNT END), 0) AS SPEND_365D,
            COALESCE(SUM(ORDER_AMOUNT), 0) AS SPEND_TOTAL,

            -- Average order value
            ROUND(AVG(ORDER_AMOUNT), 2) AS AVG_ORDER_VALUE,

            -- Item count aggregates
            SUM(ITEM_COUNT) AS TOTAL_ITEMS,
            ROUND(AVG(ITEM_COUNT), 2) AS AVG_ITEMS_PER_ORDER,

            CURRENT_TIMESTAMP() AS _FEATURE_TIMESTAMP

        FROM completed_orders
        GROUP BY CUSTOMER_ID
    """).collect()

    return session.table(target_fqn)


def create_behavioral_features(
    session: Session,
    source_table: str = "CUSTOMER_INTERACTIONS",
    target_table: str = "CUSTOMER_BEHAVIORAL_FEATURES",
) -> DataFrame:
    """Create behavioral engagement features from interactions.

    Computes engagement metrics at 30-day, 90-day, and 365-day windows,
    adapted from NCL marketing aggregate patterns.

    Args:
        session: Active Snowpark session.
        source_table: Name of the interactions source table.
        target_table: Name of the behavioral features table to create.

    Returns:
        Snowpark DataFrame with behavioral features per customer.
    """
    source_fqn = _get_fqn(session, source_table)
    target_fqn = _get_fqn(session, target_table)

    session.sql(f"""
        CREATE OR REPLACE TABLE {target_fqn} AS
        SELECT
            CUSTOMER_ID,

            -- Total interaction counts by type
            COUNT(DISTINCT CASE
                WHEN INTERACTION_TYPE = 'PAGE_VIEW' THEN INTERACTION_ID END)
                AS TOTAL_PAGE_VIEWS,
            COUNT(DISTINCT CASE
                WHEN INTERACTION_TYPE = 'CLICK' THEN INTERACTION_ID END)
                AS TOTAL_CLICKS,
            COUNT(DISTINCT CASE
                WHEN INTERACTION_TYPE = 'SUPPORT_TICKET' THEN INTERACTION_ID END)
                AS TOTAL_SUPPORT_TICKETS,
            COUNT(DISTINCT CASE
                WHEN INTERACTION_TYPE IN ('EMAIL_OPEN', 'EMAIL_CLICK')
                THEN INTERACTION_ID END)
                AS TOTAL_EMAIL_ENGAGEMENTS,

            -- 30-day window counts
            COUNT(DISTINCT CASE
                WHEN INTERACTION_DATE >= DATEADD('day', -30, CURRENT_DATE())
                THEN INTERACTION_ID END) AS INTERACTIONS_30D,
            COUNT(DISTINCT CASE
                WHEN INTERACTION_DATE >= DATEADD('day', -30, CURRENT_DATE())
                    AND INTERACTION_TYPE = 'SUPPORT_TICKET'
                THEN INTERACTION_ID END) AS SUPPORT_TICKETS_30D,

            -- 90-day window counts
            COUNT(DISTINCT CASE
                WHEN INTERACTION_DATE >= DATEADD('day', -90, CURRENT_DATE())
                THEN INTERACTION_ID END) AS INTERACTIONS_90D,

            -- Engagement recency
            DATEDIFF('day', MAX(INTERACTION_DATE), CURRENT_DATE())
                AS DAYS_SINCE_LAST_INTERACTION,

            -- Channel preference
            MODE(CHANNEL) AS PREFERRED_CHANNEL,

            -- Average session duration
            ROUND(AVG(DURATION_SECONDS), 2) AS AVG_DURATION_SECONDS,

            CURRENT_TIMESTAMP() AS _FEATURE_TIMESTAMP

        FROM {source_fqn}
        GROUP BY CUSTOMER_ID
    """).collect()

    return session.table(target_fqn)


def create_derived_features(
    session: Session,
    rfm_table: str = "CUSTOMER_RFM_FEATURES",
    behavioral_table: str = "CUSTOMER_BEHAVIORAL_FEATURES",
    target_table: str = "CUSTOMER_DERIVED_FEATURES",
) -> DataFrame:
    """Create derived features combining RFM and behavioral data.

    Computes ratios, buckets, and composite scores — adapted from
    NCL's l2_contact_derived_features pattern with safe division.

    Args:
        session: Active Snowpark session.
        rfm_table: Name of the RFM features table.
        behavioral_table: Name of the behavioral features table.
        target_table: Name of the derived features table to create.

    Returns:
        Snowpark DataFrame with derived features per customer.
    """
    rfm_fqn = _get_fqn(session, rfm_table)
    behavioral_fqn = _get_fqn(session, behavioral_table)
    target_fqn = _get_fqn(session, target_table)

    session.sql(f"""
        CREATE OR REPLACE TABLE {target_fqn} AS
        SELECT
            r.CUSTOMER_ID,

            -- RFM base features
            r.DAYS_SINCE_LAST_ORDER,
            r.ORDERS_30D,
            r.ORDERS_90D,
            r.ORDERS_365D,
            r.ORDERS_TOTAL,
            r.SPEND_TOTAL,
            r.AVG_ORDER_VALUE,

            -- Behavioral base features
            COALESCE(b.TOTAL_PAGE_VIEWS, 0) AS TOTAL_PAGE_VIEWS,
            COALESCE(b.TOTAL_CLICKS, 0) AS TOTAL_CLICKS,
            COALESCE(b.TOTAL_SUPPORT_TICKETS, 0) AS TOTAL_SUPPORT_TICKETS,
            COALESCE(b.INTERACTIONS_30D, 0) AS INTERACTIONS_30D,
            COALESCE(b.DAYS_SINCE_LAST_INTERACTION, 0) AS DAYS_SINCE_LAST_INTERACTION,
            b.PREFERRED_CHANNEL,

            -- Derived ratios (safe division)
            CASE
                WHEN r.ORDERS_TOTAL > 0
                THEN ROUND(r.SPEND_TOTAL / r.ORDERS_TOTAL, 2)
                ELSE 0
            END AS SPEND_PER_ORDER,

            CASE
                WHEN r.ORDERS_365D > 0
                THEN ROUND(r.ORDERS_90D::FLOAT / r.ORDERS_365D, 4)
                ELSE 0
            END AS ORDER_RECENCY_RATIO,

            CASE
                WHEN COALESCE(b.TOTAL_PAGE_VIEWS, 0) > 0
                THEN ROUND(
                    COALESCE(b.TOTAL_CLICKS, 0)::FLOAT
                    / b.TOTAL_PAGE_VIEWS, 4)
                ELSE 0
            END AS CLICK_THROUGH_RATE,

            -- Recency bucket (adapted from NCL pg_recency_bucket)
            CASE
                WHEN r.DAYS_SINCE_LAST_ORDER <= 30 THEN 'ACTIVE'
                WHEN r.DAYS_SINCE_LAST_ORDER <= 90 THEN 'WARM'
                WHEN r.DAYS_SINCE_LAST_ORDER <= 180 THEN 'COOLING'
                WHEN r.DAYS_SINCE_LAST_ORDER <= 365 THEN 'AT_RISK'
                ELSE 'DORMANT'
            END AS RECENCY_BUCKET,

            -- Spend bucket
            CASE
                WHEN r.SPEND_TOTAL >= 10000 THEN 'HIGH'
                WHEN r.SPEND_TOTAL >= 2000 THEN 'MEDIUM'
                WHEN r.SPEND_TOTAL > 0 THEN 'LOW'
                ELSE 'NONE'
            END AS SPEND_BUCKET,

            -- Engagement score (composite)
            ROUND(
                COALESCE(b.TOTAL_CLICKS, 0) * 2.0
                + COALESCE(b.TOTAL_PAGE_VIEWS, 0) * 0.5
                + COALESCE(b.TOTAL_EMAIL_ENGAGEMENTS, 0) * 1.5
                - COALESCE(b.TOTAL_SUPPORT_TICKETS, 0) * 3.0
            , 2) AS ENGAGEMENT_SCORE,

            CURRENT_TIMESTAMP() AS _FEATURE_TIMESTAMP

        FROM {rfm_fqn} r
        LEFT JOIN {behavioral_fqn} b ON r.CUSTOMER_ID = b.CUSTOMER_ID
    """).collect()

    return session.table(target_fqn)
