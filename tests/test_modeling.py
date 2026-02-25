"""Tests for modeling module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from snowpark_fundamentals.modeling.trainer import MODEL_REGISTRY, predict, train_model


class TestTrainModel:
    def test_train_xgboost(self, mock_dataframe):
        mock_cls = MagicMock()
        mock_model = MagicMock()
        mock_cls.return_value = mock_model

        with patch.dict(MODEL_REGISTRY, {"xgboost": mock_cls}):
            result = train_model(
                mock_dataframe,
                feature_cols=["A", "B"],
                label_col="LABEL",
                model_type="xgboost",
            )

        mock_cls.assert_called_once()
        mock_model.fit.assert_called_once_with(mock_dataframe)
        assert result is mock_model

    def test_train_random_forest(self, mock_dataframe):
        mock_cls = MagicMock()
        mock_model = MagicMock()
        mock_cls.return_value = mock_model

        with patch.dict(MODEL_REGISTRY, {"random_forest": mock_cls}):
            train_model(
                mock_dataframe,
                feature_cols=["A"],
                label_col="LABEL",
                model_type="random_forest",
            )
        mock_cls.assert_called_once()
        mock_model.fit.assert_called_once()

    def test_invalid_model_type_raises(self, mock_dataframe):
        with pytest.raises(ValueError, match="Unsupported model type"):
            train_model(
                mock_dataframe,
                feature_cols=["A"],
                label_col="LABEL",
                model_type="invalid_model",
            )

    def test_train_with_custom_params(self, mock_dataframe):
        mock_cls = MagicMock()
        mock_model = MagicMock()
        mock_cls.return_value = mock_model

        with patch.dict(MODEL_REGISTRY, {"xgboost": mock_cls}):
            train_model(
                mock_dataframe,
                feature_cols=["A"],
                label_col="LABEL",
                model_params={"n_estimators": 200, "max_depth": 10},
            )

        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs["n_estimators"] == 200
        assert call_kwargs["max_depth"] == 10


class TestPredict:
    def test_predict_calls_model(self, mock_dataframe):
        mock_model = MagicMock()
        mock_model.predict.return_value = mock_dataframe

        result = predict(mock_model, mock_dataframe)
        mock_model.predict.assert_called_once_with(mock_dataframe)
        assert result is mock_dataframe


class TestEvaluation:
    @patch("snowpark_fundamentals.modeling.evaluation.f1_score", return_value=0.85)
    @patch("snowpark_fundamentals.modeling.evaluation.recall_score", return_value=0.80)
    @patch("snowpark_fundamentals.modeling.evaluation.precision_score", return_value=0.90)
    @patch("snowpark_fundamentals.modeling.evaluation.accuracy_score", return_value=0.92)
    def test_evaluate_binary_classifier(
        self, mock_acc, mock_prec, mock_rec, mock_f1, mock_dataframe
    ):
        from snowpark_fundamentals.modeling.evaluation import evaluate_binary_classifier

        result = evaluate_binary_classifier(mock_dataframe, "LABEL", "PRED")

        assert result["accuracy"] == 0.92
        assert result["precision"] == 0.90
        assert result["recall"] == 0.80
        assert result["f1_score"] == 0.85

    def test_feature_importance(self):
        from snowpark_fundamentals.modeling.evaluation import get_feature_importance

        mock_model = MagicMock()
        mock_native = MagicMock()
        mock_native.feature_importances_ = [0.5, 0.3, 0.2]
        mock_model.to_xgboost.return_value = mock_native

        result = get_feature_importance(mock_model, ["A", "B", "C"])

        assert len(result) == 3
        assert result[0]["feature"] == "A"
        assert result[0]["importance"] == 0.5
        assert result[1]["feature"] == "B"
