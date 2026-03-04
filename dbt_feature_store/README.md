# dbt Feature Store — Complete Guide

A production-grade dbt project for building ML features in Snowflake.
After this guide, you will be able to create your own feature tables for any model.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Project Structure](#2-project-structure)
3. [Configuration Files Explained](#3-configuration-files-explained)
4. [dbt Commands — Full Reference](#4-dbt-commands--full-reference)
5. [Architecture: Staging → Features → Marts](#5-architecture-staging--features--marts)
6. [Model Walkthrough](#6-model-walkthrough)
7. [Macros — Reusable SQL Patterns](#7-macros--reusable-sql-patterns)
8. [Schema Definitions and Testing](#8-schema-definitions-and-testing)
9. [Custom Data Quality Tests](#9-custom-data-quality-tests)
10. [How to Add Your Own Features](#10-how-to-add-your-own-features)
11. [Advanced Patterns](#11-advanced-patterns)
12. [Troubleshooting](#12-troubleshooting)
13. [References](#13-references)

---

## 1. Prerequisites

Before running any dbt command, you need two things:

**a) Source tables must exist in Snowflake.**
Run notebook 06 first (or call the Python functions directly):

```python
from snowpark_fundamentals.feature_store.feature_data import (
    create_customer_transactions_dataset,
    create_customer_interactions_dataset,
)
create_customer_transactions_dataset(session, n_rows=50000)
create_customer_interactions_dataset(session, n_rows=100000)
```

This creates `CUSTOMER_TRANSACTIONS` and `CUSTOMER_INTERACTIONS` in your configured database/schema.

**b) Environment variables must be set** (same `.env` as the notebooks):

```
SNOWFLAKE_ACCOUNT=your_account
SNOWFLAKE_USER=your_user
SNOWFLAKE_PASSWORD=your_password
SNOWFLAKE_ROLE=your_role
SNOWFLAKE_WAREHOUSE=your_warehouse
SNOWFLAKE_DATABASE=your_database
SNOWFLAKE_SCHEMA=your_schema
```

Load them before running dbt:

```bash
# Option 1: export from .env
export $(grep -v '^#' ../.env | xargs)

# Option 2: source it
set -a && source ../.env && set +a
```

---

## 2. Project Structure

```
dbt_feature_store/
|
|-- dbt_project.yml               # Project config: name, layers, materialization
|-- profiles.yml                   # Snowflake connection (reads env vars)
|-- packages.yml                   # External packages (dbt_utils)
|
|-- models/
|   |-- staging/                   # LAYER 1: Views — clean + deduplicate raw data
|   |   |-- _sources.yml           #   Declares raw source tables
|   |   |-- _staging.yml           #   Column docs + schema tests
|   |   |-- stg_customers.sql      #   Deduplicated customer dimension
|   |   |-- stg_orders.sql         #   Cleaned order history
|   |
|   |-- features/                  # LAYER 2: Tables — aggregations + derived features
|   |   |-- _features.yml          #   Column docs + schema tests
|   |   |-- fct_customer_rfm.sql   #   RFM: recency, frequency, monetary (30d/90d/365d)
|   |   |-- fct_customer_behavior.sql  # Behavioral: engagement, support, channels
|   |   |-- fct_customer_derived.sql   # Derived: ratios, buckets, composite scores
|   |
|   |-- marts/                     # LAYER 3: Tables — model-ready feature tables
|       |-- _mart_features.yml     #   Column docs + schema tests
|       |-- fct_churn_features.sql #   Wide table: all features for churn model
|
|-- macros/
|   |-- feature_helpers.sql        # deduplicate_by_keys(), safe_ratio()
|   |-- time_windows.sql           # generate_time_window_count/sum/avg(), bucket_continuous_feature()
|
|-- tests/
    |-- test_grain_uniqueness.sql  # No duplicate CUSTOMER_IDs across feature tables
    |-- test_value_ranges.sql      # No negative counts, valid ratios
```

---

## 3. Configuration Files Explained

### `dbt_project.yml` — What it Controls

```yaml
name: 'tutorial_feature_store'       # Project name (used in {{ ref() }})
profile: 'tutorial_feature_store'    # Links to profiles.yml

model-paths: ["models"]              # Where dbt looks for .sql model files
macro-paths: ["macros"]              # Where dbt looks for reusable macros
test-paths: ["tests"]                # Where dbt looks for custom tests

vars:
  time_windows: [30, 90, 365]        # Configurable time window sizes
  feature_store_schema: 'FEATURE_STORE'

models:
  tutorial_feature_store:
    staging:
      +schema: stg                   # Output schema suffix: YOUR_SCHEMA_stg
      +materialized: view            # Staging = views (cheap, always fresh)
    features:
      +schema: features              # Output schema suffix: YOUR_SCHEMA_features
      +materialized: table           # Features = tables (persisted, fast reads)
    marts:
      +schema: feature_store         # Output schema suffix: YOUR_SCHEMA_feature_store
      +materialized: table           # Marts = tables (ML-ready)
```

**Schema naming:** If your env has `SNOWFLAKE_SCHEMA=PREDICTIONS`, dbt creates:
- `PREDICTIONS_stg` — staging views
- `PREDICTIONS_features` — feature tables
- `PREDICTIONS_feature_store` — mart tables

### `profiles.yml` — Snowflake Connection

```yaml
tutorial_feature_store:
  target: dev
  outputs:
    dev:
      type: snowflake
      account: "{{ env_var('SNOWFLAKE_ACCOUNT') }}"   # Reads from environment
      user: "{{ env_var('SNOWFLAKE_USER') }}"
      password: "{{ env_var('SNOWFLAKE_PASSWORD') }}"
      role: "{{ env_var('SNOWFLAKE_ROLE') }}"
      database: "{{ env_var('SNOWFLAKE_DATABASE') }}"
      warehouse: "{{ env_var('SNOWFLAKE_WAREHOUSE') }}"
      schema: "{{ env_var('SNOWFLAKE_SCHEMA') }}"
      threads: 4                      # Parallel model execution
      query_tag: 'dbt_feature_store_tutorial'  # Shows in Snowflake query history
```

### `packages.yml` — External Dependencies

```yaml
packages:
  - package: dbt-labs/dbt_utils      # Provides accepted_range, surrogate_key, etc.
    version: ">=1.1.0"
```

---

## 4. dbt Commands — Full Reference

**All commands must be run from the `dbt_feature_store/` directory:**

```bash
cd dbt_feature_store
```

### Setup and Validation

```bash
# Install external packages (dbt_utils)
# Run this ONCE after cloning, or after changing packages.yml
dbt deps
```

```bash
# Validate your connection, config, and dependencies
# This is the FIRST thing to run if anything seems broken
dbt debug --profiles-dir .
```

Expected output:
```
  profiles.yml file [OK found and valid]
  dbt_project.yml file [OK found and valid]
  Connection test: [OK connection ok]
```

### Running Models

```bash
# Run ALL models (staging → features → marts, in dependency order)
dbt run --profiles-dir .
```

```bash
# Run a SINGLE model by name
dbt run --profiles-dir . --select stg_customers
dbt run --profiles-dir . --select fct_customer_rfm
dbt run --profiles-dir . --select fct_churn_features
```

```bash
# Run a model AND all its upstream dependencies (the + prefix)
dbt run --profiles-dir . --select +fct_churn_features
```
This runs: `stg_orders` → `fct_customer_rfm` → `fct_customer_behavior` → `fct_customer_derived` → `fct_churn_features`

```bash
# Run only models in a specific folder/layer
dbt run --profiles-dir . --select "staging.*"
dbt run --profiles-dir . --select "features.*"
dbt run --profiles-dir . --select "marts.*"
```

```bash
# Run models with a specific tag
dbt run --profiles-dir . --select tag:rfm
dbt run --profiles-dir . --select tag:behavioral
```

```bash
# Full refresh — drops and recreates tables (useful when schema changes)
dbt run --profiles-dir . --full-refresh
```

### Testing

```bash
# Run ALL tests (schema tests from YAML + custom SQL tests)
dbt test --profiles-dir .
```

```bash
# Run tests for a specific model only
dbt test --profiles-dir . --select fct_customer_rfm
dbt test --profiles-dir . --select fct_churn_features
```

```bash
# Run ONLY schema tests (unique, not_null, accepted_values, accepted_range)
dbt test --profiles-dir . --select test_type:schema
```

```bash
# Run ONLY custom SQL tests (test_grain_uniqueness, test_value_ranges)
dbt test --profiles-dir . --select test_type:data
```

### Compiled SQL Inspection

```bash
# Compile models to see the actual SQL that dbt generates (no execution)
# Output goes to target/compiled/
dbt compile --profiles-dir .
```

After compiling, inspect what the macros produce:

```bash
# See the compiled SQL for fct_customer_rfm (macros expanded)
cat target/compiled/tutorial_feature_store/models/features/fct_customer_rfm.sql
```

This is critical for debugging — you see the exact SQL that Snowflake will execute.

### Documentation

```bash
# Generate the documentation site (schema descriptions, DAG, lineage)
dbt docs generate --profiles-dir .

# Serve it locally (opens browser with interactive DAG)
dbt docs serve --profiles-dir .
```

The docs site shows:
- Interactive DAG (dependency graph) of all models
- Column-level descriptions from your YAML files
- Test coverage per model
- Source freshness information

### Cleanup

```bash
# Remove compiled artifacts (target/, dbt_packages/, logs/)
dbt clean
```

### Other Useful Commands

```bash
# List all models in the project
dbt ls --profiles-dir .

# List models with their materialization type
dbt ls --profiles-dir . --output json

# Show the DAG path for a specific model
dbt ls --profiles-dir . --select +fct_churn_features

# Retry only models that failed in the last run
dbt retry --profiles-dir .

# Print the dbt version
dbt --version
```

---

## 5. Architecture: Staging → Features → Marts

```
  SOURCE TABLES                    dbt LAYERS                        OUTPUT
  (created by notebooks)           (this project)                    (in Snowflake)
  ====================            ================                   ===============

  CUSTOMER_TRANSACTIONS   --->   staging/stg_customers     VIEW     YOUR_SCHEMA_stg
  CUSTOMER_TRANSACTIONS   --->   staging/stg_orders        VIEW     YOUR_SCHEMA_stg
                                       |
                                       v
  CUSTOMER_INTERACTIONS   --->   features/fct_customer_rfm       TABLE  YOUR_SCHEMA_features
                                 features/fct_customer_behavior  TABLE  YOUR_SCHEMA_features
                                       |
                                       v
                                 features/fct_customer_derived   TABLE  YOUR_SCHEMA_features
                                       |
                                       v
                                 marts/fct_churn_features        TABLE  YOUR_SCHEMA_feature_store
```

**Why 3 layers?**

| Layer | Materialized As | Purpose | Example |
|-------|----------------|---------|---------|
| **Staging** | View | Clean raw data. Deduplicate. Add flags. No business logic. | Remove duplicate transactions, add `IS_COMPLETED` flag |
| **Features** | Table | Compute features. Time windows. Aggregations. Ratios. | 30d/90d/365d order counts, click-through rate |
| **Marts** | Table | Select and combine features for a specific model. Wide table. | All 20 features needed for churn prediction |

**Key principle:** Each layer only references the layer above it.
- Staging references sources (`{{ source('tutorial_raw', 'customer_transactions') }}`)
- Features reference staging (`{{ ref('stg_orders') }}`)
- Marts reference features (`{{ ref('fct_customer_derived') }}`)

---

## 6. Model Walkthrough

### 6.1 Staging: `stg_customers.sql`

**Input:** `CUSTOMER_TRANSACTIONS` (raw, 50K rows, may have duplicates)
**Output:** One row per `CUSTOMER_ID` (deduplicated)

Key pattern — **QUALIFY ROW_NUMBER() for deduplication:**

```sql
select
    customer_id,
    max(order_date) as last_order_date,
    count(distinct transaction_id) as total_orders,
    sum(order_amount) as total_spend,
    ...
from source
where order_status = 'COMPLETED'
group by customer_id
-- Keep only the top row per customer (highest spend)
qualify row_number() over (
    partition by customer_id
    order by total_spend desc
) = 1
```

`QUALIFY` is Snowflake-specific — it filters window function results without a subquery.
This is the standard deduplication pattern used in production.

### 6.2 Staging: `stg_orders.sql`

**Input:** `CUSTOMER_TRANSACTIONS` (raw)
**Output:** One row per `TRANSACTION_ID` with derived boolean flags

Key pattern — **Boolean flag derivation:**

```sql
case when order_status = 'COMPLETED' then true else false end as is_completed,
case when order_status = 'CANCELLED' then true else false end as is_cancelled,
```

Downstream models filter on `is_completed = true` instead of repeating string comparisons.

### 6.3 Features: `fct_customer_rfm.sql`

**Input:** `stg_orders` (deduplicated)
**Output:** One row per `CUSTOMER_ID` with RFM features at 3 time windows

Key pattern — **Time-windowed aggregations using macros:**

```sql
-- These macro calls:
{{ generate_time_window_count('transaction_id', 'order_date', 30) }} as orders_30d,
{{ generate_time_window_count('transaction_id', 'order_date', 90) }} as orders_90d,

-- Compile to this SQL:
count(distinct case
    when order_date >= dateadd('day', -30, current_date())
    then transaction_id
end) as orders_30d,
count(distinct case
    when order_date >= dateadd('day', -90, current_date())
    then transaction_id
end) as orders_90d,
```

**Output columns (14 features):**

| Column | Type | Description |
|--------|------|-------------|
| `CUSTOMER_ID` | INT | Grain key |
| `DAYS_SINCE_LAST_ORDER` | INT | Recency |
| `ORDERS_30D` | INT | Order count, last 30 days |
| `ORDERS_90D` | INT | Order count, last 90 days |
| `ORDERS_365D` | INT | Order count, last 365 days |
| `ORDERS_TOTAL` | INT | Lifetime order count |
| `SPEND_30D` | FLOAT | Total spend, last 30 days |
| `SPEND_90D` | FLOAT | Total spend, last 90 days |
| `SPEND_365D` | FLOAT | Total spend, last 365 days |
| `SPEND_TOTAL` | FLOAT | Lifetime total spend |
| `AVG_ORDER_VALUE` | FLOAT | Average order amount |
| `TOTAL_ITEMS` | INT | Lifetime items ordered |
| `AVG_ITEMS_PER_ORDER` | FLOAT | Average items per order |
| `DISTINCT_CATEGORIES` | INT | Category diversity |

### 6.4 Features: `fct_customer_behavior.sql`

**Input:** `CUSTOMER_INTERACTIONS` (source, directly — no staging needed for clean data)
**Output:** One row per `CUSTOMER_ID` with engagement features

Key pattern — **Conditional aggregation by interaction type:**

```sql
count(distinct case
    when interaction_type = 'PAGE_VIEW' then interaction_id
end) as total_page_views,
count(distinct case
    when interaction_type = 'CLICK' then interaction_id
end) as total_clicks,
```

**Output columns (12 features):**

| Column | Type | Description |
|--------|------|-------------|
| `CUSTOMER_ID` | INT | Grain key |
| `TOTAL_INTERACTIONS` | INT | All interactions, lifetime |
| `TOTAL_PAGE_VIEWS` | INT | PAGE_VIEW count |
| `TOTAL_CLICKS` | INT | CLICK count |
| `TOTAL_SUPPORT_TICKETS` | INT | SUPPORT_TICKET count |
| `TOTAL_EMAIL_ENGAGEMENTS` | INT | EMAIL_OPEN + EMAIL_CLICK |
| `INTERACTIONS_30D` | INT | All interactions, last 30 days |
| `INTERACTIONS_90D` | INT | All interactions, last 90 days |
| `SUPPORT_TICKETS_30D` | INT | Support tickets, last 30 days |
| `DAYS_SINCE_LAST_INTERACTION` | INT | Engagement recency |
| `PREFERRED_CHANNEL` | VARCHAR | Most frequent channel |
| `AVG_DURATION_SECONDS` | FLOAT | Average session length |

### 6.5 Features: `fct_customer_derived.sql`

**Input:** `fct_customer_rfm` + `fct_customer_behavior` (LEFT JOIN)
**Output:** One row per `CUSTOMER_ID` with derived ratios, buckets, scores

Key pattern — **Safe ratio macro:**

```sql
-- This macro call:
{{ safe_ratio('r.spend_total', 'r.orders_total') }} as spend_per_order

-- Compiles to:
case
    when coalesce(r.orders_total, 0) > 0
    then round(r.spend_total::float / r.orders_total, 4)
    else 0
end as spend_per_order
```

Key pattern — **Bucketing continuous features:**

```sql
case
    when r.days_since_last_order <= 30 then 'ACTIVE'
    when r.days_since_last_order <= 90 then 'WARM'
    when r.days_since_last_order <= 180 then 'COOLING'
    when r.days_since_last_order <= 365 then 'AT_RISK'
    else 'DORMANT'
end as recency_bucket
```

Key pattern — **Composite scoring:**

```sql
round(
    coalesce(b.total_clicks, 0) * 2.0           -- clicks weighted high
    + coalesce(b.total_page_views, 0) * 0.5     -- page views weighted low
    + coalesce(b.total_email_engagements, 0) * 1.5  -- email engagement
    - coalesce(b.total_support_tickets, 0) * 3.0    -- support tickets penalize
, 2) as engagement_score
```

### 6.6 Marts: `fct_churn_features.sql`

**Input:** `fct_customer_derived`
**Output:** Wide table with exactly the 20 features needed for the churn model

This model does **no computation** — it only selects and renames columns.
This is intentional: the mart is a **contract** between data engineering and data science.

**Output columns (20 features):**

| Feature | Type | Category |
|---------|------|----------|
| `DAYS_SINCE_LAST_ORDER` | INT | RFM — Recency |
| `ORDERS_30D` | INT | RFM — Frequency |
| `ORDERS_90D` | INT | RFM — Frequency |
| `ORDERS_365D` | INT | RFM — Frequency |
| `ORDERS_TOTAL` | INT | RFM — Frequency |
| `SPEND_TOTAL` | FLOAT | RFM — Monetary |
| `AVG_ORDER_VALUE` | FLOAT | RFM — Monetary |
| `SPEND_PER_ORDER` | FLOAT | Derived — Ratio |
| `ORDER_RECENCY_RATIO` | FLOAT | Derived — Ratio |
| `TOTAL_PAGE_VIEWS` | INT | Behavioral |
| `TOTAL_CLICKS` | INT | Behavioral |
| `TOTAL_SUPPORT_TICKETS` | INT | Behavioral |
| `INTERACTIONS_30D` | INT | Behavioral |
| `DAYS_SINCE_LAST_INTERACTION` | INT | Behavioral |
| `CLICK_THROUGH_RATE` | FLOAT | Derived — Ratio |
| `ENGAGEMENT_SCORE` | FLOAT | Derived — Composite |
| `PREFERRED_CHANNEL` | VARCHAR | Behavioral — Categorical |
| `RECENCY_BUCKET` | VARCHAR | Derived — Categorical |
| `SPEND_BUCKET` | VARCHAR | Derived — Categorical |
| `_FEATURE_TIMESTAMP` | TIMESTAMP | Metadata |

---

## 7. Macros — Reusable SQL Patterns

Macros live in `macros/` and are called with `{{ macro_name(args) }}` in any model.

### `macros/time_windows.sql`

**`generate_time_window_count(column, date_column, days)`**

Counts distinct values within a rolling time window.

```sql
-- Usage:
{{ generate_time_window_count('transaction_id', 'order_date', 30) }} as orders_30d

-- Compiles to:
count(distinct case
    when order_date >= dateadd('day', -30, current_date())
    then transaction_id
end) as orders_30d
```

**`generate_time_window_sum(column, date_column, days)`**

Sums values within a rolling time window, defaulting to 0.

```sql
-- Usage:
{{ generate_time_window_sum('order_amount', 'order_date', 90) }} as spend_90d

-- Compiles to:
coalesce(sum(case
    when order_date >= dateadd('day', -90, current_date())
    then order_amount
end), 0) as spend_90d
```

**`generate_time_window_avg(column, date_column, days)`**

Averages values within a rolling time window.

```sql
-- Usage:
{{ generate_time_window_avg('order_amount', 'order_date', 365) }} as avg_order_1yr

-- Compiles to:
round(avg(case
    when order_date >= dateadd('day', -365, current_date())
    then order_amount
end), 2) as avg_order_1yr
```

**`calculate_days_since(date_column)`**

```sql
-- Usage:
{{ calculate_days_since('last_order_date') }} as days_since_last_order

-- Compiles to:
datediff('day', last_order_date, current_date()) as days_since_last_order
```

**`bucket_continuous_feature(column, thresholds, labels)`**

```sql
-- Usage:
{{ bucket_continuous_feature('spend_total', [0, 2000, 10000], ['NONE', 'LOW', 'MEDIUM', 'HIGH']) }}

-- Compiles to:
case
    when spend_total <= 0 then 'NONE'
    when spend_total <= 2000 then 'LOW'
    when spend_total <= 10000 then 'MEDIUM'
    else 'HIGH'
end
```

### `macros/feature_helpers.sql`

**`deduplicate_by_keys(relation, partition_keys, order_col, order_dir)`**

```sql
-- Usage:
select * from source
{{ deduplicate_by_keys('source', ['customer_id', 'brand_cd'], 'updated_at', 'desc') }}

-- Compiles to:
select * from source
qualify row_number() over (
    partition by customer_id, brand_cd
    order by updated_at desc nulls last
) = 1
```

**`safe_ratio(numerator, denominator, decimal_places)`**

```sql
-- Usage:
{{ safe_ratio('total_clicks', 'total_page_views') }} as ctr

-- Compiles to:
case
    when coalesce(total_page_views, 0) > 0
    then round(total_clicks::float / total_page_views, 4)
    else 0
end as ctr
```

---

## 8. Schema Definitions and Testing

Every layer has a YAML file that defines column descriptions and schema tests.

### How Schema Tests Work

In `_staging.yml`:

```yaml
models:
  - name: stg_orders
    columns:
      - name: TRANSACTION_ID
        tests:
          - unique           # No duplicates allowed
          - not_null         # No NULLs allowed
      - name: CUSTOMER_ID
        tests:
          - not_null
```

In `_features.yml`:

```yaml
models:
  - name: fct_customer_rfm
    columns:
      - name: CUSTOMER_ID
        tests:
          - unique
          - not_null
      - name: ORDERS_30D
        tests:
          - dbt_utils.accepted_range:    # From dbt_utils package
              min_value: 0               # Order counts can't be negative
      - name: RECENCY_BUCKET
        tests:
          - accepted_values:
              values: ['ACTIVE', 'WARM', 'COOLING', 'AT_RISK', 'DORMANT']
```

When you run `dbt test`, these generate SQL like:

```sql
-- unique test:
select count(*) from (
    select TRANSACTION_ID from YOUR_SCHEMA_stg.stg_orders
    group by TRANSACTION_ID having count(*) > 1
)

-- accepted_range test:
select count(*) from YOUR_SCHEMA_features.fct_customer_rfm
where ORDERS_30D < 0
```

The test **passes** if the query returns 0 rows.

---

## 9. Custom Data Quality Tests

Beyond schema tests, custom SQL tests live in `tests/`.

### `test_grain_uniqueness.sql`

Validates that every feature table has exactly one row per `CUSTOMER_ID`:

```sql
-- PASS: returns 0 rows (no duplicates)
select customer_id, count(*) as row_count
from {{ ref('fct_churn_features') }}
group by customer_id
having count(*) > 1
```

### `test_value_ranges.sql`

Validates that numeric features don't have impossible values:

```sql
-- PASS: returns 0 rows (no negative counts)
select *
from (
    select
        sum(case when orders_30d < 0 then 1 else 0 end) as invalid_orders,
        sum(case when spend_total < 0 then 1 else 0 end) as invalid_spend
    from {{ ref('fct_customer_rfm') }}
)
where invalid_orders > 0 or invalid_spend > 0
```

---

## 10. How to Add Your Own Features

### Step-by-Step: Adding a New Feature to an Existing Model

**Example:** Add `AVG_SPEND_90D` to `fct_customer_rfm.sql`.

1. Edit `models/features/fct_customer_rfm.sql` — add the column:
   ```sql
   {{ generate_time_window_avg('order_amount', 'order_date', 90) }} as avg_spend_90d,
   ```

2. Document it in `models/features/_features.yml`:
   ```yaml
   - name: AVG_SPEND_90D
     description: "Average order amount in the last 90 days"
   ```

3. Compile to verify the SQL:
   ```bash
   dbt compile --profiles-dir . --select fct_customer_rfm
   cat target/compiled/tutorial_feature_store/models/features/fct_customer_rfm.sql
   ```

4. Run and test:
   ```bash
   dbt run --profiles-dir . --select fct_customer_rfm
   dbt test --profiles-dir . --select fct_customer_rfm
   ```

### Step-by-Step: Adding a Completely New Feature Table

**Example:** Create `fct_customer_channel_preference` for channel-level features.

1. Create `models/features/fct_customer_channel_preference.sql`:

   ```sql
   {{ config(materialized='table', tags=['features', 'channel']) }}

   with orders as (
       select * from {{ ref('stg_orders') }}
       where is_completed = true
   ),

   aggregated as (
       select
           customer_id,
           count(distinct case when channel = 'WEB' then transaction_id end) as web_orders,
           count(distinct case when channel = 'MOBILE' then transaction_id end) as mobile_orders,
           count(distinct case when channel = 'IN_STORE' then transaction_id end) as store_orders,
           {{ safe_ratio(
               "count(distinct case when channel = 'MOBILE' then transaction_id end)",
               "count(distinct transaction_id)"
           ) }} as mobile_share,
           mode(channel) as primary_channel,
           current_timestamp() as _feature_timestamp
       from orders
       group by customer_id
   )

   select * from aggregated
   ```

2. Add schema in `models/features/_features.yml`:

   ```yaml
   - name: fct_customer_channel_preference
     description: "Channel preference features per customer"
     columns:
       - name: CUSTOMER_ID
         tests:
           - unique
           - not_null
       - name: MOBILE_SHARE
         description: "Fraction of orders placed via mobile"
   ```

3. If the mart needs it, add a LEFT JOIN in `fct_churn_features.sql`:

   ```sql
   with derived_features as (
       select * from {{ ref('fct_customer_derived') }}
   ),
   channel_features as (
       select * from {{ ref('fct_customer_channel_preference') }}
   ),
   final as (
       select
           d.*,
           c.mobile_share,
           c.primary_channel
       from derived_features d
       left join channel_features c on d.customer_id = c.customer_id
   )
   select * from final
   ```

4. Run the full chain:
   ```bash
   dbt run --profiles-dir . --select +fct_churn_features
   dbt test --profiles-dir .
   ```

### Step-by-Step: Creating a New Macro

**Example:** A macro for rolling distinct counts.

Create `macros/rolling_counts.sql`:

```sql
{% macro rolling_distinct_count(column, date_column, days_list) %}
{# Generate multiple time-windowed distinct counts in one call. #}
{% for days in days_list %}
count(distinct case
    when {{ date_column }} >= dateadd('day', -{{ days }}, current_date())
    then {{ column }}
end) as {{ column }}_{{ days }}d{% if not loop.last %},{% endif %}
{% endfor %}
{% endmacro %}
```

Usage:

```sql
{{ rolling_distinct_count('transaction_id', 'order_date', [7, 30, 90, 365]) }}
-- Generates: transaction_id_7d, transaction_id_30d, transaction_id_90d, transaction_id_365d
```

---

## 11. Advanced Patterns

### Incremental Materialization

For large tables that grow over time, avoid full rebuilds:

```sql
{{ config(
    materialized='incremental',
    incremental_strategy='merge',
    unique_key='customer_id',
) }}

select ...
from {{ ref('stg_orders') }}
{% if is_incremental() %}
    where order_date > (select max(order_date) from {{ this }})
{% endif %}
```

First run: full table creation. Subsequent runs: only process new rows.

### Tagging and Selecting by Tag

```sql
{{ config(tags=['features', 'rfm', 'priority_high']) }}
```

```bash
# Run only high-priority features
dbt run --profiles-dir . --select tag:priority_high

# Run everything EXCEPT staging
dbt run --profiles-dir . --exclude tag:staging
```

### Variable Overrides

```bash
# Override time_windows variable at runtime
dbt run --profiles-dir . --vars '{"time_windows": [7, 14, 30, 60, 90]}'
```

### Generating the Compiled DAG

```bash
dbt docs generate --profiles-dir .
dbt docs serve --profiles-dir .
```

This opens a browser with an interactive dependency graph showing:
- Which models depend on which
- Column-level lineage
- Test coverage

---

## 12. Troubleshooting

| Problem | Command | Fix |
|---------|---------|-----|
| Connection fails | `dbt debug --profiles-dir .` | Check env vars are exported |
| Source table not found | `dbt run --profiles-dir .` | Run notebook 06 first to create source tables |
| Schema tests fail | `dbt test --profiles-dir . --select model_name` | Inspect the failing assertion SQL in `target/run/` |
| Macro not found | `dbt compile --profiles-dir .` | Run `dbt deps` to install packages |
| Stale results | `dbt run --profiles-dir . --full-refresh` | Drops and recreates all tables |
| Want to see compiled SQL | `dbt compile --profiles-dir .` | Check `target/compiled/` |
| Need to start fresh | `dbt clean && dbt deps` | Removes all artifacts and reinstalls |

### Reading Logs

```bash
# Detailed logs for the last run
cat logs/dbt.log

# Or run with debug-level logging
dbt run --profiles-dir . --debug
```

---

## 13. References

### dbt Documentation
- [dbt Core Docs](https://docs.getdbt.com/docs/introduction)
- [dbt-snowflake Adapter](https://docs.getdbt.com/docs/core/connect-data-platform/snowflake-setup)
- [Model Materialization](https://docs.getdbt.com/docs/build/materializations)
- [Jinja + Macros](https://docs.getdbt.com/docs/build/jinja-macros)
- [Tests](https://docs.getdbt.com/docs/build/data-tests)
- [dbt_utils Package](https://hub.getdbt.com/dbt-labs/dbt_utils/latest/)
- [Incremental Models](https://docs.getdbt.com/docs/build/incremental-models)

### Snowflake-Specific
- [QUALIFY Clause](https://docs.snowflake.com/en/sql-reference/constructs/qualify)
- [DATEADD Function](https://docs.snowflake.com/en/sql-reference/functions/dateadd)
- [GENERATOR Function](https://docs.snowflake.com/en/sql-reference/functions/generator)
- [Snowflake Feature Store](https://docs.snowflake.com/en/developer-guide/snowflake-ml/feature-store/overview)

### Feature Engineering Concepts
- [RFM Analysis](https://en.wikipedia.org/wiki/RFM_(market_research))
- [Feature Store: A Centralized Feature Platform](https://www.featurestore.org/)
- [Point-in-Time Correctness](https://docs.snowflake.com/en/developer-guide/snowflake-ml/feature-store/overview#point-in-time-lookups)
