"""Tests for distributed training utilities."""

from __future__ import annotations

from unittest.mock import MagicMock

from snowpark_fundamentals.modeling.distributed import (
    check_compute_pool_status,
    compare_training_results,
    ensure_ml_stage,
    validate_distributed_prerequisites,
)


class TestCheckComputePoolStatus:
    def test_pool_found(self, mock_session):
        row = MagicMock()
        row.__getitem__ = lambda self, key: {
            "name": "MY_POOL",
            "state": "ACTIVE",
            "instance_family": "CPU_X64_S",
            "min_nodes": 1,
            "max_nodes": 10,
            "auto_resume": "true",
        }[key]
        mock_session.sql.return_value.collect.return_value = [row]

        result = check_compute_pool_status(mock_session, "MY_POOL")

        assert result["name"] == "MY_POOL"
        assert result["state"] == "ACTIVE"
        assert result["instance_family"] == "CPU_X64_S"
        assert result["max_nodes"] == 10

    def test_pool_not_found(self, mock_session):
        mock_session.sql.return_value.collect.return_value = []

        result = check_compute_pool_status(mock_session, "MISSING_POOL")

        assert result["state"] == "NOT_FOUND"


class TestEnsureMlStage:
    def test_creates_stage(self, mock_session):
        mock_session.sql.return_value.collect.return_value = []

        result = ensure_ml_stage(mock_session, "ML_JOBS_STAGE")

        mock_session.sql.assert_called_with("CREATE STAGE IF NOT EXISTS ML_JOBS_STAGE")
        assert result == "@ML_JOBS_STAGE"

    def test_strips_at_prefix(self, mock_session):
        mock_session.sql.return_value.collect.return_value = []

        result = ensure_ml_stage(mock_session, "@MY_STAGE")

        mock_session.sql.assert_called_with("CREATE STAGE IF NOT EXISTS MY_STAGE")
        assert result == "@MY_STAGE"


class TestValidateDistributedPrerequisites:
    def test_ready_when_pool_active(self, mock_session):
        row = MagicMock()
        row.__getitem__ = lambda self, key: {
            "name": "MY_POOL",
            "state": "ACTIVE",
            "instance_family": "CPU_X64_S",
            "min_nodes": 1,
            "max_nodes": 10,
            "auto_resume": "true",
        }[key]
        mock_session.sql.return_value.collect.return_value = [row]

        result = validate_distributed_prerequisites(mock_session, "MY_POOL", "ML_STAGE")

        assert result["ready"] is True
        assert result["stage_ref"] == "@ML_STAGE"

    def test_not_ready_when_pool_missing(self, mock_session):
        mock_session.sql.return_value.collect.return_value = []

        result = validate_distributed_prerequisites(mock_session, "MISSING", "ML_STAGE")

        assert result["ready"] is False


class TestCompareTrainingResults:
    def test_builds_dataframe(self):
        results = {
            "warehouse": {"time": 10.5, "f1": 0.85, "method": "Warehouse XGBoost"},
            "ml_job": {"time": 15.2, "f1": 0.86, "method": "@remote on CPU pool"},
        }

        df = compare_training_results(results)

        assert len(df) == 2
        assert list(df.columns) == ["Approach", "Method", "Time (s)", "F1"]
        assert df.iloc[0]["Approach"] == "warehouse"
        assert df.iloc[1]["Method"] == "@remote on CPU pool"

    def test_single_result(self):
        results = {"baseline": {"time": 5.0, "f1": 0.80, "method": "baseline"}}

        df = compare_training_results(results)

        assert len(df) == 1
