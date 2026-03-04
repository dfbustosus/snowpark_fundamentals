{{
    config(
        materialized='table',
        tags=['mart', 'feature_store', 'churn']
    )
}}

{#
    Model-ready churn prediction feature table.
    Adapted from NCL mart pattern (fct_oci_contact_model_features.sql).
    Combines derived features into a single, wide table ready for ML.

    Grain: One row per CUSTOMER_ID
#}

with derived_features as (

    select * from {{ ref('fct_customer_derived') }}

),

final as (

    select
        -- Entity key
        customer_id,

        -- RFM features
        days_since_last_order,
        orders_30d,
        orders_90d,
        orders_365d,
        orders_total,
        spend_total,
        avg_order_value,
        spend_per_order,
        order_recency_ratio,

        -- Behavioral features
        total_page_views,
        total_clicks,
        total_support_tickets,
        interactions_30d,
        days_since_last_interaction,
        click_through_rate,
        engagement_score,

        -- Categorical features
        preferred_channel,
        recency_bucket,
        spend_bucket,

        current_timestamp() as _feature_timestamp

    from derived_features

)

select * from final
