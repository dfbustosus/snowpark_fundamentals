"""Distributed model training utilities for Snowflake Compute Pools.

Provides validation and comparison utilities for running ML training
on Compute Pools via ML Jobs (@remote decorator). Requires
snowflake-ml-python >= 1.31.0 and access to Snowflake Compute Pools.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from snowflake.snowpark import Session


def check_compute_pool_status(session: Session, pool_name: str) -> dict[str, Any]:
    """Check the status of a Snowflake Compute Pool.

    Args:
        session: Active Snowpark session.
        pool_name: Name of the Compute Pool to check.

    Returns:
        Dict with pool name, state, instance_family, min/max nodes, and auto_resume.
    """
    rows = session.sql(f"SHOW COMPUTE POOLS LIKE '{pool_name}'").collect()
    if not rows:
        return {"name": pool_name, "state": "NOT_FOUND"}

    row = rows[0]
    return {
        "name": row["name"],
        "state": row["state"],
        "instance_family": row["instance_family"],
        "min_nodes": row["min_nodes"],
        "max_nodes": row["max_nodes"],
        "auto_resume": row["auto_resume"],
    }


def ensure_ml_stage(session: Session, stage_name: str) -> str:
    """Create an internal stage for ML Job artifacts if it does not exist.

    The @remote decorator requires a stage to upload serialized function
    code and dependencies.

    Args:
        session: Active Snowpark session.
        stage_name: Stage name (without @ prefix).

    Returns:
        Fully qualified stage reference (e.g., '@ML_JOBS_STAGE').
    """
    clean_name = stage_name.lstrip("@")
    session.sql(f"CREATE STAGE IF NOT EXISTS {clean_name}").collect()
    return f"@{clean_name}"


def validate_distributed_prerequisites(
    session: Session,
    compute_pool: str,
    stage_name: str,
) -> dict[str, Any]:
    """Validate all prerequisites for distributed training on a Compute Pool.

    Checks pool status and ensures the artifact stage exists.

    Args:
        session: Active Snowpark session.
        compute_pool: Name of the Compute Pool.
        stage_name: Stage name for ML Job artifacts.

    Returns:
        Dict with pool_status, stage_ref, and ready flag.
    """
    pool_status = check_compute_pool_status(session, compute_pool)
    stage_ref = ensure_ml_stage(session, stage_name)

    pool_ready = pool_status["state"] in ("ACTIVE", "IDLE", "STARTING", "SUSPENDED")
    return {
        "pool_status": pool_status,
        "stage_ref": stage_ref,
        "ready": pool_ready and pool_status["state"] != "NOT_FOUND",
    }


def compare_training_results(results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Build a comparison DataFrame from training result dicts.

    Args:
        results: Dict mapping approach name to result dict.
            Each result dict should have keys: time, f1, method.

    Returns:
        DataFrame with columns: Approach, Method, Time (s), F1.
    """
    rows = []
    for name, r in results.items():
        rows.append(
            {
                "Approach": name,
                "Method": r.get("method", name),
                "Time (s)": f"{r['time']:.1f}",
                "F1": f"{r['f1']:.4f}",
            }
        )
    return pd.DataFrame(rows)
