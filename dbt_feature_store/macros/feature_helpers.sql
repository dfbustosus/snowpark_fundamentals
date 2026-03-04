{#
    Feature engineering helper macros.
    Adapted from NCL feature_store_helpers.sql.
#}


{% macro deduplicate_by_keys(relation, partition_keys, order_col, order_dir='desc') %}
{#
    Generate a QUALIFY clause for deduplication using ROW_NUMBER().
    Adapted from NCL deduplicate_by_keys / safe_deduplicate_staging.

    Usage:
        select * from source
        {{ deduplicate_by_keys('source', ['customer_id', 'brand_cd'], 'updated_at', 'desc') }}
#}
qualify row_number() over (
    partition by {{ partition_keys | join(', ') }}
    order by {{ order_col }} {{ order_dir }} nulls last
) = 1
{% endmacro %}


{% macro safe_ratio(numerator, denominator, decimal_places=4) %}
{#
    Compute a ratio with safe division (avoid divide-by-zero).
    Adapted from NCL l2_contact_derived_features.sql pattern.

    Usage:
        {{ safe_ratio('total_clicks', 'total_page_views') }} as click_through_rate
#}
case
    when coalesce({{ denominator }}, 0) > 0
    then round({{ numerator }}::float / {{ denominator }}, {{ decimal_places }})
    else 0
end
{% endmacro %}
