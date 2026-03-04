{#
    Test: Grain Uniqueness for Feature Tables
    Adapted from NCL test_frequency_grain.sql.
    Ensures feature tables have exactly one row per CUSTOMER_ID.
    PASS: query returns 0 rows (no duplicates)
#}

with churn_duplicates as (

    select
        'fct_churn_features' as model_name,
        customer_id,
        count(*) as row_count
    from {{ ref('fct_churn_features') }}
    group by 1, 2
    having count(*) > 1

),

rfm_duplicates as (

    select
        'fct_customer_rfm' as model_name,
        customer_id,
        count(*) as row_count
    from {{ ref('fct_customer_rfm') }}
    group by 1, 2
    having count(*) > 1

),

derived_duplicates as (

    select
        'fct_customer_derived' as model_name,
        customer_id,
        count(*) as row_count
    from {{ ref('fct_customer_derived') }}
    group by 1, 2
    having count(*) > 1

)

select * from churn_duplicates
union all
select * from rfm_duplicates
union all
select * from derived_duplicates
