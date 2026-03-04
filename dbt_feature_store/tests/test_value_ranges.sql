{#
    Test: Value Ranges for Numeric Features
    Adapted from NCL test_value_ranges.sql.
    Checks for clearly invalid values (negative counts, out-of-range ratios).
    PASS: query returns 0 rows (no invalid values)
#}

with rfm_checks as (

    select
        'fct_customer_rfm' as model_name,
        sum(case when orders_30d < 0 then 1 else 0 end) as invalid_orders_30d,
        sum(case when orders_90d < 0 then 1 else 0 end) as invalid_orders_90d,
        sum(case when orders_365d < 0 then 1 else 0 end) as invalid_orders_365d,
        sum(case when spend_total < 0 then 1 else 0 end) as invalid_spend,
        count(*) as total_rows
    from {{ ref('fct_customer_rfm') }}

),

derived_checks as (

    select
        'fct_customer_derived' as model_name,
        sum(case when spend_per_order < 0 then 1 else 0 end) as invalid_spend_ratio,
        sum(case when click_through_rate < 0 then 1 else 0 end) as invalid_ctr,
        sum(case when click_through_rate > 1 then 1 else 0 end) as suspicious_ctr,
        0 as placeholder1,
        count(*) as total_rows
    from {{ ref('fct_customer_derived') }}

)

-- Return rows only if invalid values exist
select * from rfm_checks
where invalid_orders_30d > 0
    or invalid_orders_90d > 0
    or invalid_orders_365d > 0
    or invalid_spend > 0

union all

select * from derived_checks
where invalid_spend_ratio > 0
    or invalid_ctr > 0
