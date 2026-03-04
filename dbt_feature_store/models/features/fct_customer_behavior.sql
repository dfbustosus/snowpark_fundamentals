{{
    config(
        materialized='table',
        tags=['features', 'behavioral']
    )
}}

{#
    Behavioral engagement features from customer interactions.
    Adapted from NCL l1_marketing_dm_aggregates.sql and l1_marketing_em_aggregates.sql.

    Grain: One row per CUSTOMER_ID
#}

with interactions as (

    select * from {{ source('tutorial_raw', 'customer_interactions') }}
    where customer_id is not null

),

aggregated as (

    select
        customer_id,

        -- Total interaction counts by type
        count(distinct interaction_id) as total_interactions,
        count(distinct case
            when interaction_type = 'PAGE_VIEW' then interaction_id
        end) as total_page_views,
        count(distinct case
            when interaction_type = 'CLICK' then interaction_id
        end) as total_clicks,
        count(distinct case
            when interaction_type = 'SUPPORT_TICKET' then interaction_id
        end) as total_support_tickets,
        count(distinct case
            when interaction_type in ('EMAIL_OPEN', 'EMAIL_CLICK') then interaction_id
        end) as total_email_engagements,

        -- Time-windowed counts (30d, 90d)
        {{ generate_time_window_count('interaction_id', 'interaction_date', 30) }}
            as interactions_30d,
        {{ generate_time_window_count('interaction_id', 'interaction_date', 90) }}
            as interactions_90d,

        -- Support tickets in 30d window
        count(distinct case
            when interaction_date >= dateadd('day', -30, current_date())
                and interaction_type = 'SUPPORT_TICKET'
            then interaction_id
        end) as support_tickets_30d,

        -- Engagement recency
        datediff('day', max(interaction_date), current_date())
            as days_since_last_interaction,

        -- Channel preference
        mode(channel) as preferred_channel,

        -- Average session duration
        round(avg(duration_seconds), 2) as avg_duration_seconds,

        current_timestamp() as _feature_timestamp

    from interactions
    group by customer_id

)

select * from aggregated
