"""Tests for hyperparameter tuning module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from snowpark_fundamentals.modeling.trainer import MODEL_REGISTRY
from snowpark_fundamentals.modeling.tuning import (
    get_best_model_params,
    get_search_results,
    grid_search_cv,
    randomized_search_cv,
)


class TestGridSearchCV:
    @patch("snowpark_fundamentals.modeling.tuning.GridSearchCV")
    def test_grid_search_default_params(self, mock_gscv_cls, mock_dataframe):
        mock_gscv = MagicMock()
        mock_gscv_cls.return_value = mock_gscv
        mock_model_cls = MagicMock()

        with patch.dict(MODEL_REGISTRY, {"xgboost": mock_model_cls}):
            result = grid_search_cv(
                mock_dataframe,
                feature_cols=["A", "B"],
                label_col="LABEL",
                model_type="xgboost",
            )

        mock_gscv_cls.assert_called_once()
        mock_gscv.fit.assert_called_once_with(mock_dataframe)
        assert result is mock_gscv

    @patch("snowpark_fundamentals.modeling.tuning.GridSearchCV")
    def test_grid_search_custom_params(self, mock_gscv_cls, mock_dataframe):
        mock_gscv = MagicMock()
        mock_gscv_cls.return_value = mock_gscv
        mock_model_cls = MagicMock()

        param_grid = {"n_estimators": [100, 200], "max_depth": [4, 6, 8]}

        with patch.dict(MODEL_REGISTRY, {"xgboost": mock_model_cls}):
            grid_search_cv(
                mock_dataframe,
                feature_cols=["A"],
                label_col="LABEL",
                param_grid=param_grid,
                cv=3,
                scoring="f1",
            )

        call_kwargs = mock_gscv_cls.call_args[1]
        assert call_kwargs["param_grid"] == param_grid
        assert call_kwargs["cv"] == 3
        assert call_kwargs["scoring"] == "f1"

    def test_grid_search_invalid_model_type(self, mock_dataframe):
        with pytest.raises(ValueError, match="Unsupported model type"):
            grid_search_cv(
                mock_dataframe,
                feature_cols=["A"],
                label_col="LABEL",
                model_type="invalid_model",
            )


class TestRandomizedSearchCV:
    @patch("snowpark_fundamentals.modeling.tuning.RandomizedSearchCV")
    def test_randomized_search_default_params(self, mock_rscv_cls, mock_dataframe):
        mock_rscv = MagicMock()
        mock_rscv_cls.return_value = mock_rscv
        mock_model_cls = MagicMock()

        with patch.dict(MODEL_REGISTRY, {"xgboost": mock_model_cls}):
            result = randomized_search_cv(
                mock_dataframe,
                feature_cols=["A", "B"],
                label_col="LABEL",
                model_type="xgboost",
            )

        mock_rscv_cls.assert_called_once()
        mock_rscv.fit.assert_called_once_with(mock_dataframe)
        assert result is mock_rscv

    @patch("snowpark_fundamentals.modeling.tuning.RandomizedSearchCV")
    def test_randomized_search_custom_params(self, mock_rscv_cls, mock_dataframe):
        mock_rscv = MagicMock()
        mock_rscv_cls.return_value = mock_rscv
        mock_model_cls = MagicMock()

        param_distributions = {
            "n_estimators": [50, 100, 200, 300],
            "max_depth": [3, 4, 5, 6, 8],
        }

        with patch.dict(MODEL_REGISTRY, {"xgboost": mock_model_cls}):
            randomized_search_cv(
                mock_dataframe,
                feature_cols=["A"],
                label_col="LABEL",
                param_distributions=param_distributions,
                n_iter=15,
                cv=3,
                random_state=123,
            )

        call_kwargs = mock_rscv_cls.call_args[1]
        assert call_kwargs["param_distributions"] == param_distributions
        assert call_kwargs["n_iter"] == 15
        assert call_kwargs["cv"] == 3
        assert call_kwargs["random_state"] == 123

    def test_randomized_search_invalid_model_type(self, mock_dataframe):
        with pytest.raises(ValueError, match="Unsupported model type"):
            randomized_search_cv(
                mock_dataframe,
                feature_cols=["A"],
                label_col="LABEL",
                model_type="invalid_model",
            )


class TestGetSearchResults:
    def test_results_sorted_by_rank(self):
        mock_sklearn = MagicMock()
        mock_sklearn.cv_results_ = {
            "params": [
                {"n_estimators": 100, "max_depth": 4},
                {"n_estimators": 200, "max_depth": 6},
                {"n_estimators": 100, "max_depth": 6},
            ],
            "mean_test_score": [0.85, 0.92, 0.88],
            "std_test_score": [0.02, 0.01, 0.03],
            "rank_test_score": [3, 1, 2],
        }
        mock_search = MagicMock()
        mock_search.to_sklearn.return_value = mock_sklearn

        results = get_search_results(mock_search)

        assert len(results) == 3
        assert results[0]["rank_test_score"] == 1
        assert results[0]["mean_test_score"] == 0.92
        assert results[1]["rank_test_score"] == 2
        assert results[2]["rank_test_score"] == 3

    def test_results_contain_expected_keys(self):
        mock_sklearn = MagicMock()
        mock_sklearn.cv_results_ = {
            "params": [{"max_depth": 4}],
            "mean_test_score": [0.90],
            "std_test_score": [0.015],
            "rank_test_score": [1],
        }
        mock_search = MagicMock()
        mock_search.to_sklearn.return_value = mock_sklearn

        results = get_search_results(mock_search)

        assert len(results) == 1
        expected_keys = {"params", "mean_test_score", "std_test_score", "rank_test_score"}
        assert set(results[0].keys()) == expected_keys


class TestGetBestModelParams:
    def test_extracts_best_params(self):
        mock_sklearn = MagicMock()
        mock_sklearn.best_params_ = {"n_estimators": 200, "max_depth": 6}
        mock_sklearn.best_score_ = 0.92
        mock_sklearn.cv_results_ = {"mean_test_score": [0.85, 0.92, 0.88]}
        mock_search = MagicMock()
        mock_search.to_sklearn.return_value = mock_sklearn

        result = get_best_model_params(mock_search)

        assert result["best_params"] == {"n_estimators": 200, "max_depth": 6}
        assert result["best_score"] == 0.92
        assert result["n_candidates_evaluated"] == 3

    def test_single_candidate(self):
        mock_sklearn = MagicMock()
        mock_sklearn.best_params_ = {"max_depth": 4}
        mock_sklearn.best_score_ = 0.88
        mock_sklearn.cv_results_ = {"mean_test_score": [0.88]}
        mock_search = MagicMock()
        mock_search.to_sklearn.return_value = mock_sklearn

        result = get_best_model_params(mock_search)

        assert result["n_candidates_evaluated"] == 1
        assert result["best_score"] == 0.88
