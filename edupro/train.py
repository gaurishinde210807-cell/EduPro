"""
train.py
========
End-to-end pipeline. Run this once and everything downstream (the Streamlit app, the
research paper, the executive summary) has what it needs.

    python train.py

Artefacts written to outputs/:
    course_master.csv        one row per course, features + targets + predictions
    temporal_frame.csv       history-window features and forecast-window targets
    category_forecast.csv    category-level actuals vs predictions
    metrics.json             every model's CV and hold-out MAE / RMSE / R2
    importance_*.csv         permutation and native feature importances
    models/*.joblib          the fitted winning pipelines
    figures/*.png            EDA charts
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import config
import data_prep
import diagnostics
import eda
import features
import models


class NumpyEncoder(json.JSONEncoder):
    """
    json.dump chokes on numpy scalars; this teaches it the conversions.

    np.bool_ is the one that bites: it is NOT a subclass of Python's bool (unlike np.int64
    against int on some platforms), so a `bool(...)` that actually returned np.bool_ sails
    through every type check and then fails at write time. `np.generic` catches every numpy
    scalar in one clause, so new ones cannot reintroduce the same bug.
    """

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):          # covers bool_, int64, float64, str_, ...
            return obj.item()
        if isinstance(obj, (pd.Timestamp, pd.Period)):
            return str(obj)
        if isinstance(obj, (set, frozenset)):
            return sorted(obj)
        if pd.isna(obj) if np.isscalar(obj) else False:
            return None
        return super().default(obj)


def banner(text: str) -> None:
    print(f"\n{'=' * 78}\n  {text}\n{'=' * 78}")


def _naive_price_benchmark(frame: pd.DataFrame, bundle: dict) -> dict | None:
    """
    Score the rule 'revenue = mean enrolments x price' on the same hold-out courses.

    When revenue is arithmetically price x enrolments, the mean is the wrong benchmark for
    a revenue model: a rule with no model at all already explains most of the variance.
    Adding it to the comparison forces the learned models to be judged on what they add
    over it, rather than on how far they beat a straw man.
    """
    if bundle["target"] not in ("CourseRevenue", "FutureRevenue"):
        return None
    if not {"CoursePrice", "EnrollmentCount"}.issubset(frame.columns):
        return None

    mean_enrol = float(frame.loc[bundle["split"]["train_index"], "EnrollmentCount"].mean())
    y_true = np.asarray(bundle["split"]["y_test"], dtype=float)
    y_pred = mean_enrol * frame.loc[bundle["split"]["test_index"], "CoursePrice"].astype(float).to_numpy()

    out = models.score(y_true, y_pred)
    out["cv_r2"] = float("nan")
    out["log_target"] = False
    out["best_params"] = {}
    out["rule"] = f"Revenue = {mean_enrol:.1f} x CoursePrice"
    return out


def main(workbook: str | Path | None = None) -> dict:
    warnings.filterwarnings("ignore", category=UserWarning)
    started = time.time()
    report: dict = {"generated_at": pd.Timestamp.now().isoformat()}

    # ==================================================================================
    # 1. Load and prepare
    # ==================================================================================
    banner("1. Loading and preparing data")
    data = data_prep.load_workbook(workbook)
    print("Sheets found      :", data.quality["sheets_found"])
    for table, shape in data.quality["raw_shapes"].items():
        print(f"  {table:14s} {shape}")
    if "date_range" in data.quality:
        print("Transaction dates :", data.quality["date_range"])
    if data.quality.get("orphan_transactions"):
        print(f"Orphan transactions: {data.quality['orphan_transactions']} "
              f"({data.quality['orphan_transaction_pct']}%) reference an unknown CourseID")
    report["data_quality"] = data.quality

    cross = data_prep.build_cross_sectional(data)
    cross = features.engineer(cross)
    print(f"\nCross-sectional frame: {cross.shape[0]} courses x {cross.shape[1]} columns")
    print(f"Courses with zero enrolments: {(cross['EnrollmentCount'] == 0).sum()}")

    # ==================================================================================
    # 2. Exploratory analysis
    # ==================================================================================
    banner("2. Exploratory data analysis")
    eda_summary = eda.run_eda(cross, data.transactions)
    report["eda"] = eda_summary
    print(f"Wrote {len(eda_summary['figures'])} figures to {config.FIGURE_DIR}")
    print(f"Total revenue observed: {eda_summary['revenue']['total']:,.2f}")
    print(f"Top-decile courses hold {eda_summary['revenue']['top_decile_share_pct']}% of revenue")

    # ==================================================================================
    # 3. Signal diagnostics - BEFORE any model is fitted
    # ==================================================================================
    banner("3. Signal diagnostics - is there anything here to learn?")
    numeric, categorical = features.feature_spec(cross, include_history=False)
    numeric, dropped = features.drop_correlated(cross, numeric)
    if dropped:
        print("Pruned for collinearity (|r| > "
              f"{config.CORRELATION_THRESHOLD}): "
              + ", ".join(f"{b} (r={r} with {a})" for a, b, r in dropped))
    report["dropped_correlated"] = dropped

    diag = diagnostics.run_diagnostics(
        cross, numeric, categorical, targets=("EnrollmentCount", "CourseRevenue")
    )
    diagnostics.print_report(diag)
    report["diagnostics"] = diag

    # ==================================================================================
    # 4. Cold-start models (launch decisions)
    # ==================================================================================
    banner("4. Cold-start models - attributes only, for new course launches")
    print(f"Features: {len(numeric)} numeric + {len(categorical)} categorical")

    cold: dict[str, dict] = {}
    for target in config.TARGETS:
        print(f"\n--- target: {target} ---")
        bundle = models.train_target(cross, target, numeric, categorical,
                                     label=f"cold_start_{target}")
        cold[target] = bundle
        naive = _naive_price_benchmark(cross, bundle)
        if naive:
            bundle["results"]["Naive: mean enrolments x price"] = naive
        for name, res in bundle["results"].items():
            if "R2" in res:
                print(f"  {name:20s} CV R2={res['cv_r2']:+.3f}  "
                      f"test R2={res['R2']:+.3f}  MAE={res['MAE']:.2f}  RMSE={res['RMSE']:.2f}"
                      f"{'  [log target]' if res['log_target'] else ''}")
        print(f"  -> best: {bundle['best_name']}")

    # ==================================================================================
    # 4. Temporal forecasting models (existing courses)
    # ==================================================================================
    banner("5. Forecasting models - history window in, forecast window out")
    temporal, split_info = data_prep.build_temporal(data)
    temporal = features.engineer(temporal, reference=cross)
    report["temporal_split"] = split_info
    print(f"History : {split_info['history_start']} -> {split_info['cutoff']} "
          f"({split_info['history_months']} months)")
    print(f"Forecast: {split_info['cutoff']} -> {split_info['forecast_end']} "
          f"({split_info['forecast_months']} months)")

    forecast: dict[str, dict] = {}
    if not split_info["usable"]:
        print(f"\nSKIPPED: only {split_info['history_months']} months of history, and "
              f"config.MIN_HISTORY_MONTHS is {config.MIN_HISTORY_MONTHS}. "
              "Lower FORECAST_WINDOW_MONTHS to build this model.")
    else:
        t_numeric, t_categorical = features.feature_spec(temporal, include_history=True)
        t_numeric, t_dropped = features.drop_correlated(temporal, t_numeric)
        if t_dropped:
            print("Pruned for collinearity: "
                  + ", ".join(f"{b} (r={r} with {a})" for a, b, r in t_dropped))
        print(f"Features: {len(t_numeric)} numeric + {len(t_categorical)} categorical")

        for target in ["FutureEnrollments", "FutureRevenue"]:
            print(f"\n--- target: {target} ---")
            bundle = models.train_target(temporal, target, t_numeric, t_categorical,
                                         label=f"forecast_{target}")
            forecast[target] = bundle
            naive = _naive_price_benchmark(temporal, bundle)
            if naive:
                bundle["results"]["Naive: mean enrolments x price"] = naive
            for name, res in bundle["results"].items():
                if "R2" in res:
                    print(f"  {name:20s} CV R2={res['cv_r2']:+.3f}  "
                          f"test R2={res['R2']:+.3f}  MAE={res['MAE']:.2f}  RMSE={res['RMSE']:.2f}"
                          f"{'  [log target]' if res['log_target'] else ''}")
            print(f"  -> best: {bundle['best_name']}")

    # ==================================================================================
    # 6. Feature importance
    # ==================================================================================
    banner("6. Feature importance")
    importance_files: dict[str, str] = {}
    for tag, bundle in [(f"cold_{t}", cold[t]) for t in cold] + \
                       [(f"forecast_{t}", forecast[t]) for t in forecast]:
        model = bundle["best_model"]
        if model is None:
            continue
        source = cross if tag.startswith("cold") else temporal
        feats = bundle["features"]["numeric"] + bundle["features"]["categorical"]
        X_test = source.loc[bundle["split"]["test_index"], feats]
        y_test = pd.Series(bundle["split"]["y_test"], index=X_test.index)

        perm = models.permutation_table(model, X_test, y_test)
        path = config.TABLE_DIR / f"importance_permutation_{tag}.csv"
        perm.to_csv(path, index=False)
        importance_files[f"permutation_{tag}"] = str(path)
        bundle["permutation_importance"] = perm.to_dict(orient="records")

        print(f"\n{tag} - top drivers ({bundle['best_name']}):")
        for _, row in perm.head(8).iterrows():
            bar = "#" * max(0, int(round(row["Importance"] * 40)))
            print(f"  {row['Feature']:26s} {row['Importance']:+.4f}  {bar}")

        native = models.native_importance_table(model)
        if native is not None:
            npath = config.TABLE_DIR / f"importance_native_{tag}.csv"
            native.to_csv(npath, index=False)
            importance_files[f"native_{tag}"] = str(npath)
            bundle["native_importance"] = native.head(40).to_dict(orient="records")

    report["importance_files"] = importance_files

    # ==================================================================================
    # 7. Predictions and category roll-up
    # ==================================================================================
    banner("7. Predictions and category-level revenue")
    enroll_model = cold["EnrollmentCount"]["best_model"]
    revenue_model = cold["CourseRevenue"]["best_model"]
    feats = cold["EnrollmentCount"]["features"]["numeric"] + \
        cold["EnrollmentCount"]["features"]["categorical"]

    cross["PredEnrollments"] = np.clip(enroll_model.predict(cross[feats]), 0, None).round(2)
    cross["PredRevenue"] = np.clip(revenue_model.predict(cross[feats]), 0, None).round(2)
    # Decomposition check: revenue should be roughly price x enrolments.
    cross["PredRevenueViaPrice"] = (cross["PredEnrollments"] * cross["CoursePrice"] *
                                    cross["RealisedPriceRatio"]).round(2)
    cross["IsTestSet"] = cross.index.isin(cold["CourseRevenue"]["split"]["test_index"])

    # Category evaluation on hold-out courses only - the honest measurement.
    test = cross[cross["IsTestSet"]]
    cat_eval = (
        test.groupby("CourseCategory")
        .agg(Courses=("CourseID", "nunique"),
             ActualRevenue=("CourseRevenue", "sum"),
             PredictedRevenue=("PredRevenue", "sum"),
             ActualEnrollments=("EnrollmentCount", "sum"),
             PredictedEnrollments=("PredEnrollments", "sum"))
        .reset_index()
    )
    cat_eval["RevenueError"] = cat_eval["PredictedRevenue"] - cat_eval["ActualRevenue"]
    cat_eval["RevenueErrorPct"] = (
        100 * cat_eval["RevenueError"] / cat_eval["ActualRevenue"].replace(0, np.nan)
    ).round(2)
    cat_eval = cat_eval.sort_values("ActualRevenue", ascending=False)

    category_metrics = models.score(cat_eval["ActualRevenue"], cat_eval["PredictedRevenue"])
    report["category_revenue_metrics"] = category_metrics
    print(f"Category-level revenue (hold-out courses, {len(cat_eval)} categories): "
          f"MAE={category_metrics['MAE']:,.2f}  RMSE={category_metrics['RMSE']:,.2f}  "
          f"R2={category_metrics['R2']:+.3f}")

    cat_eval.to_csv(config.OUTPUT_DIR / "category_forecast.csv", index=False)
    report["category_table"] = cat_eval.to_dict(orient="records")

    full_cat = data_prep.category_revenue(cross)
    full_cat.to_csv(config.TABLE_DIR / "category_summary.csv", index=False)

    if forecast:
        fe = forecast["FutureEnrollments"]
        fr = forecast["FutureRevenue"]
        tfeats = fe["features"]["numeric"] + fe["features"]["categorical"]
        temporal["PredFutureEnrollments"] = np.clip(
            fe["best_model"].predict(temporal[tfeats]), 0, None).round(2)
        temporal["PredFutureRevenue"] = np.clip(
            fr["best_model"].predict(temporal[tfeats]), 0, None).round(2)
        temporal.to_csv(config.OUTPUT_DIR / "temporal_frame.csv", index=False)

    cross.to_csv(config.OUTPUT_DIR / "course_master.csv", index=False)

    # ==================================================================================
    # 7. Persist models and metrics
    # ==================================================================================
    banner("8. Saving artefacts")
    bundles = {f"cold_{k}": v for k, v in cold.items()}
    bundles.update({f"forecast_{k}": v for k, v in forecast.items()})

    for tag, bundle in bundles.items():
        if bundle["best_model"] is None:
            continue
        joblib.dump(
            {
                "model": bundle["best_model"],
                "features": bundle["features"],
                "target": bundle["target"],
                "model_name": bundle["best_name"],
                "metrics": bundle["results"].get(bundle["best_name"], {}),
            },
            config.MODEL_DIR / f"{tag}.joblib",
        )
        print(f"  saved models/{tag}.joblib  ({bundle['best_name']})")

    # Reference frame the app needs to place a hypothetical course in context.
    joblib.dump(
        {
            "categories": sorted(cross["CourseCategory"].dropna().unique().tolist()),
            "types": sorted(cross["CourseType"].dropna().unique().tolist()),
            "levels": sorted(cross["CourseLevel"].dropna().unique().tolist()),
            "expertise": sorted(cross["Expertise"].dropna().unique().tolist()),
            "price_range": [float(cross["CoursePrice"].min()), float(cross["CoursePrice"].max())],
            "duration_range": [float(cross["CourseDuration"].min()),
                               float(cross["CourseDuration"].max())],
            "reference_frame": cross[
                ["CourseID", "CourseCategory", "CoursePrice", "CourseDuration",
                 "CourseRating", "TeacherID" if "TeacherID" in cross else "CourseID"]
            ].copy(),
        },
        config.MODEL_DIR / "reference.joblib",
    )

    # Strip unserialisable objects before writing the metrics file.
    report["models"] = {
        tag: {k: v for k, v in bundle.items() if k not in ("best_model",)}
        for tag, bundle in bundles.items()
    }
    report["runtime_seconds"] = round(time.time() - started, 1)

    with open(config.OUTPUT_DIR / "metrics.json", "w") as fh:
        json.dump(report, fh, indent=2, cls=NumpyEncoder)
    print(f"  saved metrics.json")
    print(f"\nPipeline finished in {report['runtime_seconds']}s")
    return report


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else None)
