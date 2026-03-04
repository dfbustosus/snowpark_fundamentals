{#
    Time-windowed aggregation helper macros.
    Adapted from NCL time_windows.sql.
#}


{% macro generate_time_window_count(column, date_column, days) %}
{#
    Count distinct values within a time window.
    Adapted from NCL generate_time_window_columns() macro.

    Usage:
        {{ generate_time_window_count('transaction_id', 'order_date', 30) }} as orders_30d
#}
count(distinct case
    when {{ date_column }} >= dateadd('day', -{{ days }}, current_date())
    then {{ column }}
end)
{% endmacro %}


{% macro generate_time_window_sum(column, date_column, days) %}
{#
    Sum values within a time window.

    Usage:
        {{ generate_time_window_sum('order_amount', 'order_date', 30) }} as spend_30d
#}
coalesce(sum(case
    when {{ date_column }} >= dateadd('day', -{{ days }}, current_date())
    then {{ column }}
end), 0)
{% endmacro %}


{% macro generate_time_window_avg(column, date_column, days) %}
{#
    Average values within a time window.

    Usage:
        {{ generate_time_window_avg('order_amount', 'order_date', 90) }} as avg_order_90d
#}
round(avg(case
    when {{ date_column }} >= dateadd('day', -{{ days }}, current_date())
    then {{ column }}
end), 2)
{% endmacro %}


{% macro calculate_days_since(date_column) %}
{#
    Calculate days since a given date.
    Adapted from NCL calculate_days_since().

    Usage:
        {{ calculate_days_since('last_order_date') }} as days_since_last_order
#}
datediff('day', {{ date_column }}, current_date())
{% endmacro %}


{% macro bucket_continuous_feature(column, thresholds, labels) %}
{#
    Bucket a continuous feature into categorical bins.
    Adapted from NCL bucket_continuous_feature().

    Usage:
        {{ bucket_continuous_feature('spend_total', [0, 2000, 10000], ['NONE', 'LOW', 'MEDIUM', 'HIGH']) }}
#}
case
    {% for i in range(thresholds | length) %}
    {% if loop.first %}
    when {{ column }} <= {{ thresholds[i] }} then '{{ labels[i] }}'
    {% else %}
    when {{ column }} <= {{ thresholds[i] }} then '{{ labels[i] }}'
    {% endif %}
    {% endfor %}
    else '{{ labels[-1] }}'
end
{% endmacro %}
