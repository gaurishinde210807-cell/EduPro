"""
app.py
======
EduPro course demand and revenue forecasting dashboard.

    streamlit run app.py

Requires `python train.py` to have been run first, since the app loads the fitted
pipelines and metrics from outputs/.
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import config
import features

st.set_page_config(page_title="EduPro Demand & Revenue Forecasting",
                   page_icon="📈", layout="wide")

ACCENT = "#4C72B0"
POSITIVE = "#55A868"
WARN = "#C44E52"


# ======================================================================================
# Loading
# ======================================================================================
@st.cache_data(show_spinner=False)
def load_outputs():
    metrics_path = config.OUTPUT_DIR / "metrics.json"
    master_path = config.OUTPUT_DIR / "course_master.csv"
    if not metrics_path.exists() or not master_path.exists():
        return None, None, None
    with open(metrics_path) as fh:
        metrics = json.load(fh)
    master = pd.read_csv(master_path)
    cat_path = config.OUTPUT_DIR / "category_forecast.csv"
    category = pd.read_csv(cat_path) if cat_path.exists() else None
    return metrics, master, category


@st.cache_resource(show_spinner=False)
def load_models():
    bundles = {}
    for path in sorted(config.MODEL_DIR.glob("*.joblib")):
        if path.stem == "reference":
            continue
        bundles[path.stem] = joblib.load(path)
    reference = config.MODEL_DIR / "reference.joblib"
    ref = joblib.load(reference) if reference.exists() else None
    return bundles, ref


metrics, master, category_df = load_outputs()

if metrics is None:
    st.title("EduPro Demand & Revenue Forecasting")
    st.error("No trained artefacts found.")
    st.markdown(
        "Run the pipeline first:\n\n"
        "```bash\n"
        "pip install -r requirements.txt\n"
        "# place 'EduPro Online Platform.xlsx' in ./data/\n"
        "python train.py\n"
        "python generate_report.py\n"
        "```"
    )
    st.stop()

model_bundles, reference = load_models()

# --------------------------------------------------------------------------------------
# Signal verdict banner
# --------------------------------------------------------------------------------------
# If the diagnostics found no learnable demand signal, every prediction this app renders is
# noise. Showing the sliders without saying so would make the dashboard actively misleading:
# a confident number in a clean interface is persuasive regardless of whether it means
# anything, so the warning has to appear on every page, not just in the write-up.
_diag = metrics.get("diagnostics", {})
_verdict = _diag.get("verdict", {})
NO_SIGNAL = _verdict and not _verdict.get("demand_signal_present", True)

if NO_SIGNAL:
    _disp = _diag.get("dispersion", {})
    _taut = _diag.get("tautology", {})
    st.error(
        f"**No demand signal in this dataset.** Enrolments are statistically "
        f"indistinguishable from random allocation (index of dispersion "
        f"{_disp.get('index_of_dispersion')}, where 1.00 is pure chance; "
        f"p = {_diag.get('uniformity', {}).get('p_value')}), and no course attribute predicts "
        f"them after correcting for multiple testing. **Enrolment predictions below are not "
        f"reliable and should not drive decisions.** Revenue predictions are arithmetic: "
        f"revenue = price × enrolments exactly, and the no-model rule "
        f"`{_taut.get('naive_rule', 'price × mean enrolments')}` reaches "
        f"R² = {_taut.get('naive_rule_r2')}. See `research_paper.md` section 4."
    )

st.sidebar.title("EduPro Analytics")
page = st.sidebar.radio(
    "Module",
    ["Overview", "Course demand predictor", "Revenue forecast",
     "Feature importance", "Category comparison", "Model performance"],
)
st.sidebar.caption(
    f"Trained {metrics['generated_at'][:10]} · "
    f"{metrics['eda']['n_courses']} courses · "
    f"{metrics['eda']['n_transactions']:,} transactions"
)


def kpi_row(items: list[tuple[str, str, str | None]]) -> None:
    cols = st.columns(len(items))
    for col, (label, value, help_text) in zip(cols, items):
        col.metric(label, value, help=help_text)


# ======================================================================================
# 1. Overview
# ======================================================================================
if page == "Overview":
    st.title("Platform overview")
    eda = metrics["eda"]

    kpi_row([
        ("Courses", f"{eda['n_courses']:,}", None),
        ("Transactions", f"{eda['n_transactions']:,}", None),
        ("Total revenue", f"{eda['revenue']['total']:,.0f}", None),
        ("Mean enrolments / course", f"{eda['enrollments']['mean']:,.1f}",
         f"Median {eda['enrollments']['median']:.0f}"),
        ("Top-decile revenue share", f"{eda['revenue']['top_decile_share_pct']}%",
         "Share of revenue held by the top 10% of courses"),
    ])

    st.markdown("---")
    left, right = st.columns([3, 2])

    with left:
        st.subheader("Monthly revenue")
        series = eda.get("monthly_series")
        if series:
            trend = pd.DataFrame(
                {"Month": pd.to_datetime(list(series.keys())),
                 "Revenue": list(series.values())}
            )
            fig = px.area(trend, x="Month", y="Revenue")
            fig.update_traces(line_color=ACCENT, fillcolor="rgba(76,114,176,0.18)")
            fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No dated transactions available.")

    with right:
        st.subheader("Revenue concentration")
        rev = master["CourseRevenue"].sort_values(ascending=False).reset_index(drop=True)
        cum = 100 * rev.cumsum() / max(rev.sum(), 1)
        pct_courses = 100 * (np.arange(len(rev)) + 1) / len(rev)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=pct_courses, y=cum, mode="lines",
                                 line=dict(color=ACCENT, width=3), name="Actual"))
        fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode="lines",
                                 line=dict(color="grey", dash="dash"), name="Even split"))
        fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0),
                          xaxis_title="% of courses (ranked)",
                          yaxis_title="Cumulative % of revenue")
        st.plotly_chart(fig, width="stretch")

    st.markdown("---")
    st.subheader("Top courses by revenue")
    cols = [c for c in ["CourseID", "CourseName", "CourseCategory", "CourseLevel",
                        "CoursePrice", "CourseRating", "EnrollmentCount",
                        "CourseRevenue", "PredRevenue"] if c in master.columns]
    st.dataframe(
        master.nlargest(15, "CourseRevenue")[cols].reset_index(drop=True),
        width="stretch", hide_index=True,
    )


# ======================================================================================
# 2. Course demand predictor
# ======================================================================================
elif page == "Course demand predictor":
    st.title("Course demand predictor")
    st.caption(
        "Estimates demand for a course that does not exist yet, using only attributes known "
        "at commissioning time. This is the cold-start model — it deliberately has no access "
        "to trading history."
    )

    enroll_bundle = model_bundles.get("cold_EnrollmentCount")
    revenue_bundle = model_bundles.get("cold_CourseRevenue")
    if not enroll_bundle or not revenue_bundle:
        st.error("Cold-start models not found. Run `python train.py`.")
        st.stop()

    categories = reference["categories"] if reference else sorted(master["CourseCategory"].unique())
    types = reference["types"] if reference else sorted(master["CourseType"].unique())
    levels = reference["levels"] if reference else sorted(master["CourseLevel"].unique())
    expertise_options = reference["expertise"] if reference else categories

    st.subheader("Course design")
    c1, c2, c3 = st.columns(3)
    with c1:
        cat = st.selectbox("Category", categories)
        ctype = st.selectbox("Type", types)
    with c2:
        level = st.selectbox("Level", levels)
        price = st.slider(
            "Price", float(master["CoursePrice"].min()), float(master["CoursePrice"].max()),
            float(master["CoursePrice"].median()), step=50.0,
        )
    with c3:
        duration = st.slider(
            "Duration (hours)", float(master["CourseDuration"].min()),
            float(master["CourseDuration"].max()), float(master["CourseDuration"].median()),
            step=1.0,
        )
        rating = st.slider("Expected course rating", 1.0, 5.0,
                           float(round(master["CourseRating"].median(), 1)), step=0.1)

    st.subheader("Instructor")
    i1, i2, i3 = st.columns(3)
    with i1:
        expertise = st.selectbox("Expertise", expertise_options)
    with i2:
        experience = st.slider("Years of experience", 0, 30,
                               int(master["YearsOfExperience"].median()))
    with i3:
        teacher_rating = st.slider("Instructor rating", 1.0, 5.0,
                                   float(round(master["TeacherRating"].median(), 1)), step=0.1)

    match = features.expertise_match(expertise, cat)
    st.caption(f"Expertise–category match score: **{match:.2f}** "
               f"({'exact match' if match >= 0.999 else 'partial or unrelated'})")

    def predict(price_value: float) -> tuple[float, float]:
        row = pd.DataFrame([{
            "CourseID": "NEW", "CourseCategory": cat, "CourseType": ctype,
            "CourseLevel": level, "CoursePrice": price_value, "CourseDuration": duration,
            "CourseRating": rating, "RatingMissing": 0, "Expertise": expertise,
            "YearsOfExperience": experience, "TeacherRating": teacher_rating,
            "TeacherID": "NEW",
        }])
        engineered = features.engineer(row, reference=master)
        e_feats = enroll_bundle["features"]["numeric"] + enroll_bundle["features"]["categorical"]
        r_feats = revenue_bundle["features"]["numeric"] + revenue_bundle["features"]["categorical"]
        e = float(np.clip(enroll_bundle["model"].predict(engineered[e_feats])[0], 0, None))
        r = float(np.clip(revenue_bundle["model"].predict(engineered[r_feats])[0], 0, None))
        return e, r

    pred_enroll, pred_revenue = predict(price)
    mae = enroll_bundle["metrics"].get("MAE", 0.0)
    rev_mae = revenue_bundle["metrics"].get("MAE", 0.0)

    st.markdown("---")
    kpi_row([
        ("Predicted enrolments", f"{pred_enroll:,.0f}",
         f"Typical error ±{mae:,.0f} enrolments"),
        ("Predicted revenue", f"{pred_revenue:,.0f}", f"Typical error ±{rev_mae:,.0f}"),
        ("Revenue via price × enrolments", f"{pred_enroll * price:,.0f}",
         "Decomposition check on the direct revenue prediction"),
        ("Category median price", f"{master.loc[master['CourseCategory'] == cat, 'CoursePrice'].median():,.0f}",
         None),
    ])

    if NO_SIGNAL:
        st.error(
            f"**These numbers are not predictions.** The enrolment model scores "
            f"R² = {enroll_bundle['metrics'].get('R2', 0):+.3f} on unseen courses, i.e. no "
            f"better than always guessing the catalogue average of "
            f"{master['EnrollmentCount'].mean():,.0f}. Moving the sliders will change the "
            "output, but the changes reflect noise the model fitted, not demand. This page "
            "is retained to demonstrate the interface, not to support decisions."
        )
    else:
        st.warning(
            f"These are point estimates from a model with a mean absolute error of "
            f"{mae:,.0f} enrolments (R² = {enroll_bundle['metrics'].get('R2', 0):+.2f}). "
            "Use them to rank options against each other, not as a committed forecast for a "
            "single course."
        )

    st.markdown("---")
    st.subheader("Price sensitivity for this configuration")
    grid = np.linspace(float(master["CoursePrice"].min()),
                       float(master["CoursePrice"].max()), 28)
    curve = pd.DataFrame(
        [(p, *predict(p)) for p in grid],
        columns=["Price", "PredEnrollments", "PredRevenue"],
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=curve["Price"], y=curve["PredEnrollments"], name="Enrolments",
                             line=dict(color=ACCENT, width=3)))
    fig.add_trace(go.Scatter(x=curve["Price"], y=curve["PredRevenue"], name="Revenue",
                             yaxis="y2", line=dict(color=POSITIVE, width=3)))
    fig.add_vline(x=price, line_dash="dash", line_color=WARN,
                  annotation_text="selected price")
    fig.update_layout(
        height=380, margin=dict(l=0, r=0, t=10, b=0),
        xaxis_title="Price",
        yaxis=dict(title="Predicted enrolments", color=ACCENT),
        yaxis2=dict(title="Predicted revenue", overlaying="y", side="right", color=POSITIVE),
        legend=dict(orientation="h", y=1.12),
    )
    st.plotly_chart(fig, width="stretch")

    best_price = curve.loc[curve["PredRevenue"].idxmax(), "Price"]
    st.info(
        f"Revenue is maximised at a price of about **{best_price:,.0f}** for this "
        "configuration. This curve is the model's fitted relationship across the catalogue, "
        "not an experiment: prices in the training data were not assigned at random, so it "
        "shows what similar courses did, not what this course would do if repriced."
    )


# ======================================================================================
# 3. Revenue forecast
# ======================================================================================
elif page == "Revenue forecast":
    st.title("Revenue forecast")

    tab1, tab2 = st.tabs(["Category level", "Course level"])

    with tab1:
        if category_df is None:
            st.info("No category forecast file found.")
        else:
            st.caption("Actual vs predicted revenue on hold-out courses, aggregated by "
                       "category. This is the level at which the forecasts are accurate "
                       "enough for budgeting.")
            cm = metrics.get("category_revenue_metrics", {})
            kpi_row([
                ("Category R²", f"{cm.get('R2', 0):+.3f}", None),
                ("MAE", f"{cm.get('MAE', 0):,.0f}", None),
                ("RMSE", f"{cm.get('RMSE', 0):,.0f}", None),
            ])

            long = category_df.melt(
                id_vars="CourseCategory",
                value_vars=["ActualRevenue", "PredictedRevenue"],
                var_name="Series", value_name="Revenue",
            )
            fig = px.bar(long, x="CourseCategory", y="Revenue", color="Series",
                         barmode="group",
                         color_discrete_map={"ActualRevenue": ACCENT,
                                             "PredictedRevenue": POSITIVE})
            fig.update_layout(height=400, margin=dict(l=0, r=0, t=10, b=0), xaxis_title="")
            st.plotly_chart(fig, width="stretch")

            st.dataframe(category_df, width="stretch", hide_index=True)

    with tab2:
        st.caption("Predicted vs actual revenue for every course. Points on the diagonal "
                   "are perfectly predicted.")
        plot_df = master.copy()
        plot_df["Set"] = np.where(plot_df.get("IsTestSet", False), "Hold-out", "Training")
        fig = px.scatter(
            plot_df, x="CourseRevenue", y="PredRevenue", color="Set",
            hover_data=[c for c in ["CourseID", "CourseCategory", "CoursePrice"]
                        if c in plot_df.columns],
            color_discrete_map={"Hold-out": WARN, "Training": ACCENT}, opacity=0.7,
        )
        lim = float(max(plot_df["CourseRevenue"].max(), plot_df["PredRevenue"].max()))
        fig.add_trace(go.Scatter(x=[0, lim], y=[0, lim], mode="lines",
                                 line=dict(color="grey", dash="dash"),
                                 name="Perfect prediction"))
        fig.update_layout(height=460, margin=dict(l=0, r=0, t=10, b=0),
                          xaxis_title="Actual revenue", yaxis_title="Predicted revenue")
        st.plotly_chart(fig, width="stretch")

        temporal_path = config.OUTPUT_DIR / "temporal_frame.csv"
        if temporal_path.exists():
            st.markdown("---")
            st.subheader("Next-period forecast for existing courses")
            split = metrics.get("temporal_split", {})
            st.caption(
                f"Trained on history to {split.get('cutoff', 'n/a')}, forecasting the "
                f"following {split.get('forecast_months', 'n/a')} months."
            )
            temporal = pd.read_csv(temporal_path)
            show = [c for c in ["CourseID", "CourseCategory", "CoursePrice",
                                "HistEnrollments", "PredFutureEnrollments",
                                "FutureEnrollments", "PredFutureRevenue", "FutureRevenue"]
                    if c in temporal.columns]
            st.dataframe(
                temporal.nlargest(20, "PredFutureRevenue")[show].reset_index(drop=True),
                width="stretch", hide_index=True,
            )


# ======================================================================================
# 4. Feature importance
# ======================================================================================
elif page == "Feature importance":
    st.title("Feature importance explorer")
    st.caption(
        "Permutation importance: the fall in hold-out R² when a feature's values are "
        "shuffled. Preferred over tree impurity importance, which inflates continuous and "
        "high-cardinality features whether or not they predict anything."
    )

    available = {k: v for k, v in metrics["models"].items()
                 if v.get("permutation_importance")}
    if not available:
        st.info("No importance tables found. Run `python train.py`.")
        st.stop()

    labels = {
        "cold_EnrollmentCount": "New course — enrolments",
        "cold_CourseRevenue": "New course — revenue",
        "forecast_FutureEnrollments": "Existing courses — future enrolments",
        "forecast_FutureRevenue": "Existing courses — future revenue",
    }
    choice = st.selectbox("Model", list(available.keys()),
                          format_func=lambda k: labels.get(k, k))
    bundle = available[choice]

    st.caption(f"Selected model: **{bundle['best_name']}** · "
               f"test R² {bundle['results'][bundle['best_name']]['R2']:+.3f}")

    imp = pd.DataFrame(bundle["permutation_importance"])
    top_n = st.slider("Features shown", 5, min(30, len(imp)), min(12, len(imp)))
    top = imp.head(top_n).sort_values("Importance")

    fig = px.bar(top, x="Importance", y="Feature", orientation="h", error_x="Std")
    fig.update_traces(marker_color=ACCENT)
    fig.update_layout(height=32 * top_n + 90, margin=dict(l=0, r=0, t=10, b=0),
                      xaxis_title="Drop in R² when shuffled", yaxis_title="")
    st.plotly_chart(fig, width="stretch")

    with st.expander("Model coefficients / impurity importances (encoded features)"):
        if bundle.get("native_importance"):
            st.dataframe(pd.DataFrame(bundle["native_importance"]),
                         width="stretch", hide_index=True)
            st.caption("For linear models, the sign of the coefficient gives the direction "
                       "of the effect on the standardised scale. Permutation importance "
                       "above is the better guide to magnitude.")
        else:
            st.info("Not available for this model.")

    st.markdown("---")
    st.subheader("Marginal correlations with enrolments")
    corr = pd.Series(metrics["eda"]["correlation_with_enrollments"]).sort_values()
    fig = px.bar(x=corr.values, y=corr.index, orientation="h")
    fig.update_traces(marker_color=[WARN if v < 0 else POSITIVE for v in corr.values])
    fig.update_layout(height=30 * len(corr) + 90, margin=dict(l=0, r=0, t=10, b=0),
                      xaxis_title="Pearson r", yaxis_title="")
    st.plotly_chart(fig, width="stretch")
    st.caption("Marginal correlations, not causal effects. Price is not assigned at random, "
               "so its correlation absorbs the influence of everything that varies with it.")


# ======================================================================================
# 5. Category comparison
# ======================================================================================
elif page == "Category comparison":
    st.title("Category-level demand comparison")

    cat = (
        master.groupby("CourseCategory")
        .agg(Courses=("CourseID", "nunique"),
             Enrollments=("EnrollmentCount", "sum"),
             Revenue=("CourseRevenue", "sum"),
             MedianPrice=("CoursePrice", "median"),
             MeanRating=("CourseRating", "mean"))
        .reset_index()
    )
    cat["RevenuePerCourse"] = cat["Revenue"] / cat["Courses"]
    cat["EnrollmentsPerCourse"] = cat["Enrollments"] / cat["Courses"]
    cat["RevenueShare%"] = (100 * cat["Revenue"] / cat["Revenue"].sum()).round(1)
    cat = cat.sort_values("Revenue", ascending=False)

    metric = st.radio(
        "Rank by", ["Revenue", "RevenuePerCourse", "Enrollments", "EnrollmentsPerCourse"],
        horizontal=True,
    )
    ranked = cat.sort_values(metric, ascending=True)
    fig = px.bar(ranked, x=metric, y="CourseCategory", orientation="h",
                 color=metric, color_continuous_scale="Blues")
    fig.update_layout(height=42 * len(cat) + 100, margin=dict(l=0, r=0, t=10, b=0),
                      yaxis_title="", coloraxis_showscale=False)
    st.plotly_chart(fig, width="stretch")

    if metric == "Revenue":
        st.info(
            "Total revenue rewards a category for having many courses in it. "
            "**RevenuePerCourse** is the better guide to whether the next course "
            "commissioned in that category will pay for itself."
        )

    st.markdown("---")
    st.subheader("Catalogue depth against productivity")
    fig = px.scatter(
        cat, x="Courses", y="RevenuePerCourse", size="Revenue", color="MedianPrice",
        text="CourseCategory", color_continuous_scale="Viridis", size_max=48,
    )
    fig.update_traces(textposition="top center")
    fig.add_hline(y=float(cat["RevenuePerCourse"].median()), line_dash="dash",
                  line_color="grey", annotation_text="median")
    fig.update_layout(height=460, margin=dict(l=0, r=0, t=10, b=0),
                      xaxis_title="Courses in catalogue",
                      yaxis_title="Revenue per course")
    st.plotly_chart(fig, width="stretch")
    st.caption("Upper-left: productive categories that are under-served — the strongest "
               "commissioning candidates. Lower-right: saturated categories.")

    st.dataframe(cat.round(2), width="stretch", hide_index=True)


# ======================================================================================
# 6. Model performance
# ======================================================================================
elif page == "Model performance":
    st.title("Model performance")
    st.caption(
        "The winning model for each target was chosen on cross-validated R², never on the "
        "hold-out score, so the test metrics below are an honest out-of-sample estimate."
    )

    labels = {
        "cold_EnrollmentCount": "New course — enrolments",
        "cold_CourseRevenue": "New course — revenue",
        "forecast_FutureEnrollments": "Existing courses — future enrolments",
        "forecast_FutureRevenue": "Existing courses — future revenue",
    }

    for tag, bundle in metrics["models"].items():
        st.subheader(labels.get(tag, tag))
        rows = []
        for name, res in bundle["results"].items():
            if "R2" not in res:
                continue
            rows.append({
                "Model": name + ("  ← selected" if name == bundle["best_name"] else ""),
                "CV R²": res["cv_r2"], "Test MAE": res["MAE"],
                "Test RMSE": res["RMSE"], "Test R²": res["R2"],
                "Target transform": "log1p" if res["log_target"] else "none",
            })
        table = pd.DataFrame(rows)
        st.dataframe(table, width="stretch", hide_index=True)

        fig = px.bar(table, x="Model", y="Test R²")
        fig.update_traces(marker_color=[POSITIVE if v > 0 else WARN
                                        for v in table["Test R²"]])
        fig.add_hline(y=0, line_color="grey")
        fig.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0), xaxis_title="")
        st.plotly_chart(fig, width="stretch")

        pred = bundle.get("split", {})
        if pred.get("y_test") and pred.get("y_pred"):
            resid = pd.DataFrame({"Actual": pred["y_test"], "Predicted": pred["y_pred"]})
            resid["Residual"] = resid["Predicted"] - resid["Actual"]
            with st.expander("Residual diagnostics"):
                c1, c2 = st.columns(2)
                f1 = px.scatter(resid, x="Actual", y="Predicted", opacity=0.7)
                lim = float(max(resid["Actual"].max(), resid["Predicted"].max()))
                f1.add_trace(go.Scatter(x=[0, lim], y=[0, lim], mode="lines",
                                        line=dict(color="grey", dash="dash"),
                                        showlegend=False))
                f1.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0))
                c1.plotly_chart(f1, width="stretch")

                f2 = px.histogram(resid, x="Residual", nbins=25)
                f2.update_traces(marker_color=ACCENT)
                f2.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0))
                c2.plotly_chart(f2, width="stretch")
                st.caption("A residual histogram centred on zero with no fan-shaped pattern "
                           "against the actuals is what you want. Systematic under-prediction "
                           "of high values is the usual failure mode on skewed targets.")
        st.markdown("---")

    st.subheader("Data quality log")
    st.json(metrics["data_quality"], expanded=False)
