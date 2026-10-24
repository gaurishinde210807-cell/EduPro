"""
eda.py
======
Exploratory analysis. Writes PNG figures to outputs/figures/ and returns a dictionary of
summary statistics that the report generator and the Streamlit app both read.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import config

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({"figure.dpi": 120, "savefig.bbox": "tight", "font.size": 9})


def _save(fig, name: str) -> str:
    path = config.FIGURE_DIR / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def run_eda(course_df: pd.DataFrame, transactions: pd.DataFrame) -> dict:
    """Produce the EDA figure set and return summary statistics."""
    figures: dict[str, str] = {}
    summary: dict = {}

    # ---------------------------------------------------------------------------------
    # 1. Target distributions
    # ---------------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    sns.histplot(course_df["EnrollmentCount"], bins=30, ax=axes[0], color="#4C72B0")
    axes[0].set_title("Enrolments per course")
    axes[0].set_xlabel("Enrolments")
    sns.histplot(course_df["CourseRevenue"], bins=30, ax=axes[1], color="#55A868")
    axes[1].set_title("Revenue per course")
    axes[1].set_xlabel("Revenue")
    figures["target_distributions"] = _save(fig, "01_target_distributions")

    summary["enrollments"] = {
        "mean": round(float(course_df["EnrollmentCount"].mean()), 2),
        "median": float(course_df["EnrollmentCount"].median()),
        "std": round(float(course_df["EnrollmentCount"].std()), 2),
        "min": int(course_df["EnrollmentCount"].min()),
        "max": int(course_df["EnrollmentCount"].max()),
        "skew": round(float(course_df["EnrollmentCount"].skew()), 3),
        "zero_enrolment_courses": int((course_df["EnrollmentCount"] == 0).sum()),
    }
    summary["revenue"] = {
        "total": round(float(course_df["CourseRevenue"].sum()), 2),
        "mean": round(float(course_df["CourseRevenue"].mean()), 2),
        "median": round(float(course_df["CourseRevenue"].median()), 2),
        "std": round(float(course_df["CourseRevenue"].std()), 2),
        "skew": round(float(course_df["CourseRevenue"].skew()), 3),
    }

    # Concentration: what share of revenue comes from the top decile of courses?
    rev_sorted = course_df["CourseRevenue"].sort_values(ascending=False)
    top_decile = max(1, int(np.ceil(0.10 * len(rev_sorted))))
    total_rev = float(rev_sorted.sum()) or 1.0
    summary["revenue"]["top_decile_share_pct"] = round(
        100 * float(rev_sorted.head(top_decile).sum()) / total_rev, 1
    )

    # ---------------------------------------------------------------------------------
    # 2. Category performance
    # ---------------------------------------------------------------------------------
    cat = (
        course_df.groupby("CourseCategory")
        .agg(Courses=("CourseID", "nunique"),
             Enrollments=("EnrollmentCount", "sum"),
             Revenue=("CourseRevenue", "sum"))
        .sort_values("Revenue", ascending=False)
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, max(3.2, 0.32 * len(cat))))
    cat["Revenue"].plot(kind="barh", ax=axes[0], color="#4C72B0")
    axes[0].invert_yaxis()
    axes[0].set_title("Total revenue by category")
    axes[0].set_xlabel("Revenue")
    axes[0].set_ylabel("")
    (cat["Revenue"] / cat["Courses"]).plot(kind="barh", ax=axes[1], color="#C44E52")
    axes[1].invert_yaxis()
    axes[1].set_title("Revenue per course by category")
    axes[1].set_xlabel("Revenue per course")
    axes[1].set_ylabel("")
    figures["category_performance"] = _save(fig, "02_category_performance")
    summary["categories"] = cat.reset_index().to_dict(orient="records")

    # ---------------------------------------------------------------------------------
    # 3. Price sensitivity
    # ---------------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    sns.regplot(data=course_df, x="CoursePrice", y="EnrollmentCount", ax=axes[0],
                scatter_kws={"alpha": 0.35, "s": 18}, line_kws={"color": "#C44E52"})
    axes[0].set_title("Price vs enrolments")
    sns.regplot(data=course_df, x="CoursePrice", y="CourseRevenue", ax=axes[1],
                scatter_kws={"alpha": 0.35, "s": 18}, line_kws={"color": "#C44E52"})
    axes[1].set_title("Price vs revenue")
    figures["price_sensitivity"] = _save(fig, "03_price_sensitivity")

    # ---------------------------------------------------------------------------------
    # 4. Quality signals
    # ---------------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    sns.regplot(data=course_df, x="CourseRating", y="EnrollmentCount", ax=axes[0],
                scatter_kws={"alpha": 0.35, "s": 18}, line_kws={"color": "#C44E52"})
    axes[0].set_title("Course rating vs enrolments")
    sns.regplot(data=course_df, x="TeacherRating", y="EnrollmentCount", ax=axes[1],
                scatter_kws={"alpha": 0.35, "s": 18}, line_kws={"color": "#C44E52"})
    axes[1].set_title("Instructor rating vs enrolments")
    sns.regplot(data=course_df, x="YearsOfExperience", y="EnrollmentCount", ax=axes[2],
                scatter_kws={"alpha": 0.35, "s": 18}, line_kws={"color": "#C44E52"})
    axes[2].set_title("Instructor experience vs enrolments")
    figures["quality_signals"] = _save(fig, "04_quality_signals")

    # ---------------------------------------------------------------------------------
    # 5. Level and type
    # ---------------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    sns.boxplot(data=course_df, x="CourseLevel", y="EnrollmentCount", ax=axes[0])
    axes[0].set_title("Enrolments by course level")
    axes[0].tick_params(axis="x", rotation=20)
    sns.boxplot(data=course_df, x="CourseType", y="EnrollmentCount", ax=axes[1])
    axes[1].set_title("Enrolments by course type")
    axes[1].tick_params(axis="x", rotation=20)
    figures["level_and_type"] = _save(fig, "05_level_and_type")

    # ---------------------------------------------------------------------------------
    # 6. Correlation heatmap
    # ---------------------------------------------------------------------------------
    numeric_cols = [c for c in ["CoursePrice", "CourseDuration", "CourseRating",
                                "LevelEncoded", "YearsOfExperience", "TeacherRating",
                                "ExpertiseMatch", "PricePerHour",
                                "EnrollmentCount", "CourseRevenue"]
                    if c in course_df.columns]
    corr = course_df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(7.5, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                ax=ax, annot_kws={"size": 7}, cbar_kws={"shrink": 0.8})
    ax.set_title("Correlation matrix")
    figures["correlation"] = _save(fig, "06_correlation")

    summary["correlation_with_enrollments"] = (
        corr["EnrollmentCount"].drop("EnrollmentCount").round(3).sort_values(ascending=False).to_dict()
    )
    summary["correlation_with_revenue"] = (
        corr["CourseRevenue"].drop("CourseRevenue").round(3).sort_values(ascending=False).to_dict()
    )

    # ---------------------------------------------------------------------------------
    # 7. Revenue over time
    # ---------------------------------------------------------------------------------
    if "TransactionDate" in transactions.columns and len(transactions):
        monthly = (
            transactions.set_index("TransactionDate")
            .resample("ME")
            .agg(Revenue=("Amount", "sum"), Enrollments=("Amount", "size"))
        )
        fig, ax1 = plt.subplots(figsize=(11, 3.4))
        ax1.plot(monthly.index, monthly["Revenue"], marker="o", ms=3, color="#4C72B0", label="Revenue")
        ax1.set_ylabel("Revenue", color="#4C72B0")
        ax2 = ax1.twinx()
        ax2.plot(monthly.index, monthly["Enrollments"], marker="s", ms=3,
                 color="#DD8452", label="Enrolments")
        ax2.set_ylabel("Enrolments", color="#DD8452")
        ax2.grid(False)
        ax1.set_title("Monthly platform revenue and enrolments")
        figures["monthly_trend"] = _save(fig, "07_monthly_trend")

        summary["monthly"] = {
            "months_observed": int(len(monthly)),
            "first_month": str(monthly.index.min().date()),
            "last_month": str(monthly.index.max().date()),
            "mean_monthly_revenue": round(float(monthly["Revenue"].mean()), 2),
            "mean_monthly_enrollments": round(float(monthly["Enrollments"].mean()), 2),
        }
        summary["monthly_series"] = {
            str(idx.date()): round(float(val), 2) for idx, val in monthly["Revenue"].items()
        }

    summary["figures"] = figures
    summary["n_courses"] = int(course_df["CourseID"].nunique())
    summary["n_transactions"] = int(len(transactions))
    summary["n_categories"] = int(course_df["CourseCategory"].nunique())
    return summary
