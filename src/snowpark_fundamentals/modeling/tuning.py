"""Hyperparameter tuning utilities for Snowpark ML.

Provides wrapper functions around Snowpark ML's GridSearchCV and
RandomizedSearchCV, which execute cross-validated hyperparameter
searches server-side inside the Snowflake warehouse.
"""

from __future__ import annotations

from typing import Any

from snowflake.ml.modeling.model_selection import GridSearchCV, RandomizedSearchCV
from snowflake.snowpark import DataFrame

from snowpark_fundamentals.modeling.trainer import MODEL_REGISTRY


def grid_search_cv(
    train_df: DataFrame,
    feature_cols: list[str],
    label_col: str,
    model_type: str = "xgboost",
    param_grid: dict[str, list] | None = None,
    cv: int = 5,
    scoring: str = "accuracy",
    output_col: str = "PREDICTION",
    n_jobs: int = -1,
) -> Any:
    """Run exhaustive grid search with cross-validation.

    Wraps Snowpark ML's GridSearchCV to search all combinations
    of the provided parameter grid, executing on the warehouse.

    Args:
        train_df: Training DataFrame with features and label.
        feature_cols: List of feature column names.
        label_col: Name of the target/label column.
        model_type: One of the keys in MODEL_REGISTRY.
        param_grid: Dict mapping parameter names to lists of values.
        cv: Number of cross-validation folds.
        scoring: Scoring metric (e.g., 'accuracy', 'f1').
        output_col: Name for the prediction output column.
        n_jobs: Number of parallel jobs (-1 for all available).

    Returns:
        Fitted GridSearchCV object.

    Raises:
        ValueError: If model_type is not supported.
    """
    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unsupported model type: '{model_type}'. "
            f"Choose from: {list(MODEL_REGISTRY.keys())}"
        )

    model_cls = MODEL_REGISTRY[model_type]
    param_grid = param_grid or {}

    search = GridSearchCV(
        estimator=model_cls(),
        param_grid=param_grid,
        input_cols=feature_cols,
        label_cols=[label_col],
        output_cols=[output_col],
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
    )

    search.fit(train_df)
    return search


def randomized_search_cv(
    train_df: DataFrame,
    feature_cols: list[str],
    label_col: str,
    model_type: str = "xgboost",
    param_distributions: dict[str, list] | None = None,
    n_iter: int = 10,
    cv: int = 5,
    scoring: str = "accuracy",
    output_col: str = "PREDICTION",
    n_jobs: int = -1,
    random_state: int = 42,
) -> Any:
    """Run randomized search with cross-validation.

    Wraps Snowpark ML's RandomizedSearchCV to sample n_iter
    parameter combinations from the provided distributions,
    executing on the warehouse.

    Args:
        train_df: Training DataFrame with features and label.
        feature_cols: List of feature column names.
        label_col: Name of the target/label column.
        model_type: One of the keys in MODEL_REGISTRY.
        param_distributions: Dict mapping parameter names to lists/distributions.
        n_iter: Number of parameter combinations to sample.
        cv: Number of cross-validation folds.
        scoring: Scoring metric (e.g., 'accuracy', 'f1').
        output_col: Name for the prediction output column.
        n_jobs: Number of parallel jobs (-1 for all available).
        random_state: Random seed for reproducibility.

    Returns:
        Fitted RandomizedSearchCV object.

    Raises:
        ValueError: If model_type is not supported.
    """
    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unsupported model type: '{model_type}'. "
            f"Choose from: {list(MODEL_REGISTRY.keys())}"
        )

    model_cls = MODEL_REGISTRY[model_type]
    param_distributions = param_distributions or {}

    search = RandomizedSearchCV(
        estimator=model_cls(),
        param_distributions=param_distributions,
        input_cols=feature_cols,
        label_cols=[label_col],
        output_cols=[output_col],
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        random_state=random_state,
    )

    search.fit(train_df)
    return search


def _get_sklearn_object(search_cv: Any) -> Any:
    """Get the underlying sklearn search object from a Snowpark ML wrapper.

    Snowpark ML >= 1.31 no longer forwards best_params_, best_score_,
    cv_results_ via __getattr__. Access them through to_sklearn().
    """
    if hasattr(search_cv, "to_sklearn"):
        return search_cv.to_sklearn()
    return search_cv


def get_search_results(search_cv: Any) -> list[dict[str, Any]]:
    """Extract cross-validation results from a fitted search object.

    Args:
        search_cv: Fitted GridSearchCV or RandomizedSearchCV object.

    Returns:
        List of dicts with params, mean_test_score, std_test_score,
        and rank_test_score — sorted by rank ascending.
    """
    sk = _get_sklearn_object(search_cv)
    cv_results = sk.cv_results_
    n_candidates = len(cv_results["mean_test_score"])

    results = []
    for i in range(n_candidates):
        results.append(
            {
                "params": cv_results["params"][i],
                "mean_test_score": float(cv_results["mean_test_score"][i]),
                "std_test_score": float(cv_results["std_test_score"][i]),
                "rank_test_score": int(cv_results["rank_test_score"][i]),
            }
        )

    return sorted(results, key=lambda x: x["rank_test_score"])


def get_best_model_params(search_cv: Any) -> dict[str, Any]:
    """Extract best parameters and score from a fitted search object.

    Args:
        search_cv: Fitted GridSearchCV or RandomizedSearchCV object.

    Returns:
        Dict with best_params, best_score, and n_candidates_evaluated.
    """
    sk = _get_sklearn_object(search_cv)
    return {
        "best_params": sk.best_params_,
        "best_score": float(sk.best_score_),
        "n_candidates_evaluated": len(sk.cv_results_["mean_test_score"]),
    }
