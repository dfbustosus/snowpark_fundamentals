{{
    config(
        materialized='view',
        tags=['staging', 'orders']
    )
}}

{#
    Cleaned and deduplicated order history.
    Deduplication via QUALIFY ROW_NUMBER() — one row per TRANSACTION_ID.
#}

with source as (

    select * from {{ source('tutorial_raw', 'customer_transactions') }}

),

cleaned as (

    select
        transaction_id,
        customer_id,
        order_date,
        round(order_amount, 2) as order_amount,
        category,
        order_status,
        item_count,
        channel,
        case
            when order_status = 'COMPLETED' then true
            else false
        end as is_completed,
        case
            when order_status = 'CANCELLED' then true
            else false
        end as is_cancelled,
        current_timestamp() as _loaded_at
    from source
    where transaction_id is not null
    -- Deduplication: one row per transaction_id
    qualify row_number() over (
        partition by transaction_id
        order by order_date desc
    ) = 1

)

select * from cleaned
