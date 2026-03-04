{{
    config(
        materialized='table',
        tags=['features', 'derived']
    )
}}

{#
    Derived features combining RFM and behavioral data.
    Adapted from NCL l2_contact_derived_features.sql — computes ratios,
    buckets, and composite scores with safe division.

    Grain: One row per CUSTOMER_ID
#}

with rfm as (

    select * from {{ ref('fct_customer_rfm') }}

),

behavior as (

    select * from {{ ref('fct_customer_behavior') }}

),

derived as (

    select
        r.customer_id,

        -- RFM base features
        r.days_since_last_order,
        r.orders_30d,
        r.orders_90d,
        r.orders_365d,
        r.orders_total,
        r.spend_total,
        r.avg_order_value,

        -- Behavioral base features
        coalesce(b.total_page_views, 0) as total_page_views,
        coalesce(b.total_clicks, 0) as total_clicks,
        coalesce(b.total_support_tickets, 0) as total_support_tickets,
        coalesce(b.interactions_30d, 0) as interactions_30d,
        coalesce(b.days_since_last_interaction, 0) as days_since_last_interaction,
        b.preferred_channel,

        -- Derived ratios (safe division)
        {{ safe_ratio('r.spend_total', 'r.orders_total') }} as spend_per_order,
        {{ safe_ratio('r.orders_90d', 'r.orders_365d') }} as order_recency_ratio,
        {{ safe_ratio('b.total_clicks', 'b.total_page_views') }} as click_through_rate,

        -- Recency bucket (adapted from NCL pg_recency_bucket)
        case
            when r.days_since_last_order <= 30 then 'ACTIVE'
            when r.days_since_last_order <= 90 then 'WARM'
            when r.days_since_last_order <= 180 then 'COOLING'
            when r.days_since_last_order <= 365 then 'AT_RISK'
            else 'DORMANT'
        end as recency_bucket,

        -- Spend bucket
        case
            when r.spend_total >= 10000 then 'HIGH'
            when r.spend_total >= 2000 then 'MEDIUM'
            when r.spend_total > 0 then 'LOW'
            else 'NONE'
        end as spend_bucket,

        -- Engagement score (composite)
        round(
            coalesce(b.total_clicks, 0) * 2.0
            + coalesce(b.total_page_views, 0) * 0.5
            + coalesce(b.total_email_engagements, 0) * 1.5
            - coalesce(b.total_support_tickets, 0) * 3.0
        , 2) as engagement_score,

        current_timestamp() as _feature_timestamp

    from rfm r
    left join behavior b on r.customer_id = b.customer_id

)

select * from derived
