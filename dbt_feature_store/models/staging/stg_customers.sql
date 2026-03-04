{{
    config(
        materialized='view',
        tags=['staging', 'customer']
    )
}}

{#
    Deduplicated customer dimension derived from transactions.
    Adapted from NCL stg_customer_360_profile.sql — uses QUALIFY ROW_NUMBER()
    to keep one row per CUSTOMER_ID (most recent, highest-spending).
#}

with source as (

    select * from {{ source('tutorial_raw', 'customer_transactions') }}

),

customer_agg as (

    select
        customer_id,
        max(order_date) as last_order_date,
        min(order_date) as first_order_date,
        count(distinct transaction_id) as total_orders,
        sum(order_amount) as total_spend,
        mode(channel) as preferred_channel
    from source
    where order_status = 'COMPLETED'
    group by customer_id

)

select
    customer_id,
    last_order_date,
    first_order_date,
    total_orders,
    round(total_spend, 2) as total_spend,
    preferred_channel,
    datediff('day', first_order_date, last_order_date) as customer_tenure_days,
    current_timestamp() as _loaded_at
from customer_agg
-- Deduplication: one row per customer_id (adapted from NCL QUALIFY pattern)
qualify row_number() over (
    partition by customer_id
    order by total_spend desc
) = 1
