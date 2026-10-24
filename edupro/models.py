"""
models.py
=========
Preprocessing pipelines, the model zoo, evaluation and feature importance.

Design notes
------------
* Every model is wrapped in a scikit-learn Pipeline that contains its own imputation,
  scaling and one-hot encoding. Fitting the encoder inside the cross-validation loop is
  what stops information from the validation fold leaking into the training fold.
* Enrolment and revenue are both right-skewed count-like targets, so each model family is
  fitted twice - once on the raw target, once on log1p(target) via TransformedTargetRegressor -
  and the better cross-validated fit is kept. Which one won is recorded in the results.
* A DummyRegressor that always predicts the training mean is included. Without it an R2 of
  0.4 is hard to interpret; against it, the improvement is explicit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config


# ======================================================================================
# Preprocessing
# ======================================================================================
def build_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    """Median-impute + standardise numerics; most-frequent-impute + one-hot categoricals."""
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=2)),
    ])
    return ColumnTransformer(
        [("num", numeric_pipe, numeric), ("cat", categorical_pipe, categorical)],
        remainder="drop",
        verbose_feature_names_out=False,
    )


# ======================================================================================
# Model zoo
# ======================================================================================
def model_zoo() -> dict[str, tuple[object, dict]]:
    """{name: (estimator, param_grid)} - baselines first, then the ensembles."""
    rs = config.RANDOM_STATE
    return {
        "Baseline (mean)": (DummyRegressor(strategy="mean"), {}),
        "LinearRegression": (LinearRegression(), {}),
        "Ridge": (Ridge(random_state=None), {f"model__{k}": v
                                             for k, v in config.MODEL_GRID["Ridge"].items()}),
        "Lasso": (Lasso(max_iter=20000, random_state=rs),
                  {f"model__{k}": v for k, v in config.MODEL_GRID["Lasso"].items()}),
        "RandomForest": (RandomForestRegressor(random_state=rs, n_jobs=-1),
                         {f"model__{k}": v for k, v in config.MODEL_GRID["RandomForest"].items()}),
        "GradientBoosting": (GradientBoostingRegressor(random_state=rs),
                             {f"model__{k}": v
                              for k, v in config.MODEL_GRID["GradientBoosting"].items()}),
    }


def _wrap(estimator, numeric, categorical, log_target: bool):
    """Preprocessor + estimator, optionally with a log1p/expm1 target transform."""
    pipe = Pipeline([
        ("prep", build_preprocessor(numeric, categorical)),
        ("model", estimator),
    ])
    if not log_target:
        return pipe
    # TransformedTargetRegressor renames the inner steps, so grids are re-prefixed below.
    return TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1)


def _reprefix(grid: dict, log_target: bool) -> dict:
    return {f"regressor__{k}": v for k, v in grid.items()} if log_target else grid


# ======================================================================================
# Metrics
# ======================================================================================
def score(y_true, y_pred) -> dict[str, float]:
    """MAE, RMSE and R2 in one dict."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "MAE": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "RMSE": round(float(root_mean_squared_error(y_true, y_pred)), 4),
        "R2": round(float(r2_score(y_true, y_pred)), 4),
    }


# ======================================================================================
# Training
# ======================================================================================
def train_target(
    frame: pd.DataFrame,
    target: str,
    numeric: list[str],
    categorical: list[str],
    label: str = "",
) -> dict:
    """
    Fit every model in the zoo for one target and return a full results bundle.

    Returns
    -------
    dict with keys:
        results     - per-model CV and hold-out metrics
        best_name   - winner by hold-out R2
        best_model  - the fitted pipeline (refit on the full training split)
        split       - the train/test indices and the test-set predictions
    """
    features = numeric + categorical
    X = frame[features].copy()
    y = frame[target].astype(float).copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )
    cv = KFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)

    results: dict[str, dict] = {}
    fitted: dict[str, object] = {}

    for name, (estimator, grid) in model_zoo().items():
        # The dummy gains nothing from a log target, so only try the transform where it
        # can plausibly help.
        transforms = [False] if name == "Baseline (mean)" else [False, True]
        best_cv, best_est, best_params, best_log = -np.inf, None, {}, False

        for log_target in transforms:
            base = _wrap(estimator, numeric, categorical, log_target)
            params = _reprefix(grid, log_target)
            try:
                if params:
                    search = GridSearchCV(base, params, cv=cv, scoring="r2", n_jobs=-1)
                    search.fit(X_train, y_train)
                    cv_score, est, chosen = search.best_score_, search.best_estimator_, search.best_params_
                else:
                    from sklearn.model_selection import cross_val_score
                    cv_score = float(np.mean(cross_val_score(base, X_train, y_train, cv=cv, scoring="r2")))
                    est = base.fit(X_train, y_train)
                    chosen = {}
            except Exception as exc:                      # keep the run alive, report it
                results.setdefault(name, {})["error"] = f"{type(exc).__name__}: {exc}"
                continue

            if cv_score > best_cv:
                best_cv, best_est, best_params, best_log = cv_score, est, chosen, log_target

        if best_est is None:
            continue

        preds = np.clip(best_est.predict(X_test), 0, None)   # demand cannot be negative
        results[name] = {
            "cv_r2": round(float(best_cv), 4),
            "log_target": best_log,
            "best_params": {k.split("__")[-1]: v for k, v in best_params.items()},
            **score(y_test, preds),
        }
        fitted[name] = best_est

    # Select on the CROSS-VALIDATED score, never on the hold-out score. Picking the model
    # that happens to score best on the test set turns that test set into a selection set,
    # and the number you then report is optimistic. Ranking by cv_r2 keeps the hold-out
    # metrics below an honest estimate of out-of-sample performance.
    ranked = sorted(
        (n for n in results if "cv_r2" in results[n]),
        key=lambda n: results[n]["cv_r2"],
        reverse=True,
    )
    best_name = ranked[0] if ranked else None
    best_model = fitted.get(best_name)

    test_preds = None
    if best_model is not None:
        test_preds = np.clip(best_model.predict(X_test), 0, None)

    return {
        "label": label or target,
        "target": target,
        "n_rows": int(len(frame)),
        "n_features": len(features),
        "features": {"numeric": numeric, "categorical": categorical},
        "results": results,
        "best_name": best_name,
        "best_model": best_model,
        "split": {
            "train_index": X_train.index.tolist(),
            "test_index": X_test.index.tolist(),
            "y_test": y_test.tolist(),
            "y_pred": test_preds.tolist() if test_preds is not None else [],
        },
    }


# ======================================================================================
# Feature importance
# ======================================================================================
def permutation_table(
    model, X: pd.DataFrame, y: pd.Series, n_repeats: int = 20
) -> pd.DataFrame:
    """
    Permutation importance on the original (pre-encoding) columns.

    This is the importance measure to trust. Impurity-based importances inflate
    high-cardinality and continuous features; permutation importance measures the actual
    drop in hold-out R2 when a column is shuffled.
    """
    res = permutation_importance(
        model, X, y, n_repeats=n_repeats,
        random_state=config.RANDOM_STATE, scoring="r2", n_jobs=-1,
    )
    return (
        pd.DataFrame({
            "Feature": X.columns,
            "Importance": res.importances_mean.round(5),
            "Std": res.importances_std.round(5),
        })
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )


def _inner_pipeline(model):
    """Unwrap a TransformedTargetRegressor to reach the Pipeline inside."""
    return model.regressor_ if isinstance(model, TransformedTargetRegressor) else model


def native_importance_table(model) -> pd.DataFrame | None:
    """
    Impurity importances (trees) or standardised coefficients (linear models), reported on
    the encoded feature space. Useful for reading the *direction* of a linear effect,
    which permutation importance cannot give you.
    """
    pipe = _inner_pipeline(model)
    if not isinstance(pipe, Pipeline):
        return None
    try:
        names = pipe.named_steps["prep"].get_feature_names_out()
    except Exception:
        return None

    est = pipe.named_steps["model"]
    if hasattr(est, "feature_importances_"):
        values, kind = est.feature_importances_, "impurity"
    elif hasattr(est, "coef_"):
        values, kind = np.ravel(est.coef_), "coefficient"
    else:
        return None

    df = pd.DataFrame({"Feature": names, "Value": np.round(values, 5), "Kind": kind})
    df["AbsValue"] = df["Value"].abs()
    return df.sort_values("AbsValue", ascending=False).reset_index(drop=True)
