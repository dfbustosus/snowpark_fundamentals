"""Tests for model registry module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd

from snowpark_fundamentals.registry.model_registry import (
    compare_model_versions,
    delete_model,
    delete_model_metric,
    get_model_by_alias,
    get_model_metrics,
    get_model_version,
    list_models,
    list_versions,
    load_model_and_predict,
    log_model,
    predict_proba,
    set_default_version,
    set_model_alias,
    set_model_comment,
    set_model_metrics,
    set_model_tags,
    show_model_functions,
    unset_model_alias,
)


class TestGetRegistry:
    @patch("snowpark_fundamentals.registry.model_registry.Registry")
    def test_creates_registry(self, mock_registry_cls, mock_session):
        from snowpark_fundamentals.registry.model_registry import get_registry

        get_registry(mock_session, "MY_DB", "MY_SCHEMA")
        mock_registry_cls.assert_called_once_with(
            session=mock_session, database_name="MY_DB", schema_name="MY_SCHEMA"
        )


class TestLogModel:
    def test_log_model_with_metrics(self):
        mock_registry = MagicMock()
        mock_model = MagicMock()
        mock_sample = MagicMock()
        metrics = {"accuracy": 0.95}

        log_model(mock_registry, mock_model, "test_model", "v1", mock_sample, metrics)

        mock_registry.log_model.assert_called_once()
        call_kwargs = mock_registry.log_model.call_args[1]
        assert call_kwargs["model_name"] == "test_model"
        assert call_kwargs["version_name"] == "v1"
        assert call_kwargs["metrics"] == metrics

    def test_log_model_without_metrics(self):
        mock_registry = MagicMock()
        mock_model = MagicMock()
        mock_sample = MagicMock()

        log_model(mock_registry, mock_model, "test_model", "v1", mock_sample)

        call_kwargs = mock_registry.log_model.call_args[1]
        assert "metrics" not in call_kwargs


class TestListModels:
    def test_list_models(self):
        mock_registry = MagicMock()
        mock_registry.show_models.return_value = "model_list"

        result = list_models(mock_registry)
        assert result == "model_list"


class TestLoadModelAndPredict:
    def test_load_and_predict(self):
        mock_registry = MagicMock()
        mock_input = MagicMock()
        mock_predictions = MagicMock()

        mock_version = MagicMock()
        mock_version.run.return_value = mock_predictions
        mock_registry.get_model.return_value.version.return_value = mock_version

        result = load_model_and_predict(mock_registry, "my_model", "v1", mock_input)

        mock_registry.get_model.assert_called_once_with("my_model")
        mock_version.run.assert_called_once_with(mock_input, function_name="predict")
        assert result is mock_predictions


class TestDeleteModel:
    def test_delete_model(self):
        mock_registry = MagicMock()
        delete_model(mock_registry, "test_model")
        mock_registry.delete_model.assert_called_once_with("test_model")


# ---------------------------------------------------------------------------
# Week 3: Version Management
# ---------------------------------------------------------------------------


class TestGetModelVersion:
    def test_get_model_version(self):
        mock_registry = MagicMock()
        mock_version = MagicMock()
        mock_registry.get_model.return_value.version.return_value = mock_version

        result = get_model_version(mock_registry, "my_model", "V1")

        mock_registry.get_model.assert_called_once_with("my_model")
        mock_registry.get_model.return_value.version.assert_called_once_with("V1")
        assert result is mock_version


class TestListVersions:
    def test_list_versions(self):
        mock_registry = MagicMock()
        mock_versions_df = MagicMock()
        mock_registry.get_model.return_value.show_versions.return_value = mock_versions_df

        result = list_versions(mock_registry, "my_model")

        mock_registry.get_model.assert_called_once_with("my_model")
        mock_registry.get_model.return_value.show_versions.assert_called_once()
        assert result is mock_versions_df


class TestSetDefaultVersion:
    def test_set_default_version(self):
        mock_registry = MagicMock()
        mock_model = MagicMock()
        mock_registry.get_model.return_value = mock_model

        set_default_version(mock_registry, "my_model", "V2")

        mock_registry.get_model.assert_called_once_with("my_model")
        assert mock_model.default == "V2"


class TestCompareModelVersions:
    def test_compare_multiple_versions(self):
        mock_registry = MagicMock()
        mock_model = MagicMock()
        mock_registry.get_model.return_value = mock_model

        v1 = MagicMock()
        v1.show_metrics.return_value = {"accuracy": 0.85}
        v2 = MagicMock()
        v2.show_metrics.return_value = {"accuracy": 0.90}
        mock_model.version.side_effect = [v1, v2]

        result = compare_model_versions(mock_registry, "my_model", ["V1", "V2"])

        assert len(result) == 2
        assert result[0] == {"version": "V1", "metrics": {"accuracy": 0.85}}
        assert result[1] == {"version": "V2", "metrics": {"accuracy": 0.90}}

    def test_compare_empty_metrics(self):
        mock_registry = MagicMock()
        mock_model = MagicMock()
        mock_registry.get_model.return_value = mock_model

        v1 = MagicMock()
        v1.show_metrics.return_value = {}
        mock_model.version.return_value = v1

        result = compare_model_versions(mock_registry, "my_model", ["V1"])

        assert result == [{"version": "V1", "metrics": {}}]


# ---------------------------------------------------------------------------
# Week 3: Metrics Management
# ---------------------------------------------------------------------------


class TestSetModelMetrics:
    def test_set_multiple_metrics(self):
        mock_registry = MagicMock()
        mock_version = MagicMock()
        mock_registry.get_model.return_value.version.return_value = mock_version

        set_model_metrics(mock_registry, "my_model", "V1", {"accuracy": 0.9, "f1": 0.85})

        assert mock_version.set_metric.call_count == 2
        mock_version.set_metric.assert_any_call("accuracy", 0.9)
        mock_version.set_metric.assert_any_call("f1", 0.85)

    def test_set_single_metric(self):
        mock_registry = MagicMock()
        mock_version = MagicMock()
        mock_registry.get_model.return_value.version.return_value = mock_version

        set_model_metrics(mock_registry, "my_model", "V1", {"accuracy": 0.9})

        mock_version.set_metric.assert_called_once_with("accuracy", 0.9)


class TestGetModelMetrics:
    def test_get_model_metrics(self):
        mock_registry = MagicMock()
        mock_version = MagicMock()
        mock_version.show_metrics.return_value = {"accuracy": 0.9, "f1": 0.85}
        mock_registry.get_model.return_value.version.return_value = mock_version

        result = get_model_metrics(mock_registry, "my_model", "V1")

        assert result == {"accuracy": 0.9, "f1": 0.85}


class TestDeleteModelMetric:
    def test_delete_model_metric(self):
        mock_registry = MagicMock()
        mock_version = MagicMock()
        mock_registry.get_model.return_value.version.return_value = mock_version

        delete_model_metric(mock_registry, "my_model", "V1", "accuracy")

        mock_version.delete_metric.assert_called_once_with("accuracy")


# ---------------------------------------------------------------------------
# Week 3: Lifecycle Management
# ---------------------------------------------------------------------------


class TestSetModelAlias:
    def test_set_alias_no_conflict(self):
        mock_registry = MagicMock()
        mock_model = MagicMock()
        mock_registry.get_model.return_value = mock_model
        mock_model.show_versions.return_value = pd.DataFrame(
            {"name": ["V1", "V2"], "aliases": ['["FIRST"]', '["DEFAULT","LAST"]']}
        )
        mock_version = MagicMock()
        mock_model.version.return_value = mock_version

        set_model_alias(mock_registry, "my_model", "V1", "production")

        mock_version.set_alias.assert_called_once_with("production")

    def test_set_alias_moves_from_other_version(self):
        mock_registry = MagicMock()
        mock_model = MagicMock()
        mock_registry.get_model.return_value = mock_model
        mock_model.show_versions.return_value = pd.DataFrame(
            {"name": ["V1", "V2"], "aliases": ['["PRODUCTION","FIRST"]', '["DEFAULT","LAST"]']}
        )
        old_version = MagicMock()
        new_version = MagicMock()
        mock_model.version.side_effect = lambda v: old_version if v == "V1" else new_version

        set_model_alias(mock_registry, "my_model", "V2", "production")

        old_version.unset_alias.assert_called_once_with("production")
        new_version.set_alias.assert_called_once_with("production")

    def test_set_alias_already_on_target_is_noop(self):
        mock_registry = MagicMock()
        mock_model = MagicMock()
        mock_registry.get_model.return_value = mock_model
        mock_model.show_versions.return_value = pd.DataFrame(
            {"name": ["V1", "V2"], "aliases": ['["PRODUCTION","FIRST"]', '["DEFAULT","LAST"]']}
        )

        set_model_alias(mock_registry, "my_model", "V1", "production")

        mock_model.version.assert_not_called()


class TestUnsetModelAlias:
    def test_unset_model_alias(self):
        mock_registry = MagicMock()
        mock_version = MagicMock()
        mock_registry.get_model.return_value.version.return_value = mock_version

        unset_model_alias(mock_registry, "my_model", "V1", "production")

        mock_version.unset_alias.assert_called_once_with("production")


class TestGetModelByAlias:
    def test_get_model_by_alias(self):
        mock_registry = MagicMock()
        mock_version = MagicMock()
        mock_registry.get_model.return_value.version.return_value = mock_version

        result = get_model_by_alias(mock_registry, "my_model", "production")

        mock_registry.get_model.assert_called_once_with("my_model")
        mock_registry.get_model.return_value.version.assert_called_once_with("production")
        assert result is mock_version


class TestSetModelTags:
    def test_set_multiple_tags(self):
        mock_registry = MagicMock()
        mock_model = MagicMock()
        mock_registry.get_model.return_value = mock_model

        set_model_tags(mock_registry, "my_model", {"team": "ml-platform", "project": "churn"})

        assert mock_model.set_tag.call_count == 2
        mock_model.set_tag.assert_any_call("team", "ml-platform")
        mock_model.set_tag.assert_any_call("project", "churn")

    def test_set_single_tag(self):
        mock_registry = MagicMock()
        mock_model = MagicMock()
        mock_registry.get_model.return_value = mock_model

        set_model_tags(mock_registry, "my_model", {"team": "ml-platform"})

        mock_model.set_tag.assert_called_once_with("team", "ml-platform")


# ---------------------------------------------------------------------------
# Week 3: Metadata & Advanced Inference
# ---------------------------------------------------------------------------


class TestSetModelComment:
    def test_set_model_level_comment(self):
        mock_registry = MagicMock()
        mock_model = MagicMock()
        mock_registry.get_model.return_value = mock_model

        set_model_comment(mock_registry, "my_model", "Churn prediction model")

        assert mock_model.comment == "Churn prediction model"

    def test_set_version_level_comment(self):
        mock_registry = MagicMock()
        mock_version = MagicMock()
        mock_registry.get_model.return_value.version.return_value = mock_version

        set_model_comment(mock_registry, "my_model", "Tuned version", version_name="V2")

        assert mock_version.comment == "Tuned version"


class TestShowModelFunctions:
    def test_show_model_functions(self):
        mock_registry = MagicMock()
        mock_version = MagicMock()
        mock_version.show_functions.return_value = ["predict", "predict_proba", "explain"]
        mock_registry.get_model.return_value.version.return_value = mock_version

        result = show_model_functions(mock_registry, "my_model", "V1")

        mock_version.show_functions.assert_called_once()
        assert result == ["predict", "predict_proba", "explain"]


class TestPredictProba:
    def test_predict_proba(self):
        mock_registry = MagicMock()
        mock_version = MagicMock()
        mock_input = MagicMock()
        mock_result = MagicMock()
        mock_version.run.return_value = mock_result
        mock_registry.get_model.return_value.version.return_value = mock_version

        result = predict_proba(mock_registry, "my_model", "V1", mock_input)

        mock_version.run.assert_called_once_with(mock_input, function_name="predict_proba")
        assert result is mock_result
