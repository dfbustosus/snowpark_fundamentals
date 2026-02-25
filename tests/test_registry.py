"""Tests for model registry module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from snowpark_fundamentals.registry.model_registry import (
    delete_model,
    list_models,
    load_model_and_predict,
    log_model,
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
