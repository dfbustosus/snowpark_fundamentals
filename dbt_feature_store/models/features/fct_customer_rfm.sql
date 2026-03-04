{{
    config(
        materialized='table',
        tags=['features', 'rfm']
    )
}}

{#
    RFM (Recency, Frequency, Monetary) features with time-windowed aggregations.
    Adapted from NCL l1_booking_aggregates.sql — uses 30d/90d/365d windows
    instead of NCL's 30d/45d/90d.

    Grain: One row per CUSTOMER_ID
#}

with orders as (

    select * from {{ ref('stg_orders') }}
    where is_completed = true

),

aggregated as (

    select
        customer_id,

        -- Recency
        datediff('day', max(order_date), current_date()) as days_since_last_order,

        -- Frequency: time-windowed order counts
        {{ generate_time_window_count('transaction_id', 'order_date', 30) }} as orders_30d,
        {{ generate_time_window_count('transaction_id', 'order_date', 90) }} as orders_90d,
        {{ generate_time_window_count('transaction_id', 'order_date', 365) }} as orders_365d,
        count(distinct transaction_id) as orders_total,

        -- Monetary: time-windowed spend
        {{ generate_time_window_sum('order_amount', 'order_date', 30) }} as spend_30d,
        {{ generate_time_window_sum('order_amount', 'order_date', 90) }} as spend_90d,
        {{ generate_time_window_sum('order_amount', 'order_date', 365) }} as spend_365d,
        coalesce(sum(order_amount), 0) as spend_total,

        -- Average order value
        round(avg(order_amount), 2) as avg_order_value,

        -- Item count aggregates
        sum(item_count) as total_items,
        round(avg(item_count), 2) as avg_items_per_order,

        -- Category diversity
        count(distinct category) as distinct_categories,

        current_timestamp() as _feature_timestamp

    from orders
    group by customer_id

)

select * from aggregated
