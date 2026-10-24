"""
make_sample_data.py
===================
Generates a SYNTHETIC workbook with the same four sheets and column names as
'EduPro Online Platform.xlsx'.

This exists purely so the pipeline can be smoke-tested before the real file is in place.
Nothing produced from this data is a finding about EduPro. Delete data/SAMPLE_*.xlsx once
you have the real workbook.

    python make_sample_data.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config

RNG = np.random.default_rng(7)

CATEGORIES = ["Data Science", "Web Development", "Cyber Security", "Business",
              "Design", "Marketing", "Cloud Computing", "Mobile Development"]
TYPES = ["Self-Paced", "Live", "Hybrid"]
LEVELS = ["Beginner", "Intermediate", "Advanced"]

# Latent category appeal, so the models have real structure to recover.
CATEGORY_PULL = dict(zip(CATEGORIES, [1.35, 1.25, 1.10, 0.80, 0.85, 0.90, 1.15, 0.95]))
TYPE_PULL = {"Self-Paced": 1.10, "Live": 0.85, "Hybrid": 1.00}
LEVEL_PULL = {"Beginner": 1.25, "Intermediate": 1.00, "Advanced": 0.70}


def build(n_teachers=45, n_courses=180, n_users=1200, months=24) -> dict[str, pd.DataFrame]:
    # ---- Teachers ---------------------------------------------------------------------
    teachers = pd.DataFrame({
        "TeacherID": [f"T{i:04d}" for i in range(1, n_teachers + 1)],
        "TeacherName": [f"teacher_{i}" for i in range(1, n_teachers + 1)],
        "Expertise": RNG.choice(CATEGORIES, n_teachers),
        "YearsOfExperience": RNG.integers(1, 22, n_teachers),
        "TeacherRating": np.round(np.clip(RNG.normal(4.1, 0.45, n_teachers), 2.5, 5.0), 1),
    })

    # ---- Courses ----------------------------------------------------------------------
    cat = RNG.choice(CATEGORIES, n_courses)
    price = np.round(np.clip(RNG.gamma(3.0, 900, n_courses), 499, 14999), -1)
    courses = pd.DataFrame({
        "CourseID": [f"C{i:04d}" for i in range(1, n_courses + 1)],
        "CourseName": [f"Course {i}" for i in range(1, n_courses + 1)],
        "CourseCategory": cat,
        "CourseType": RNG.choice(TYPES, n_courses, p=[0.55, 0.25, 0.20]),
        "CourseLevel": RNG.choice(LEVELS, n_courses, p=[0.4, 0.4, 0.2]),
        "CoursePrice": price,
        "CourseDuration": np.round(np.clip(RNG.normal(28, 12, n_courses), 4, 90), 1),
        "CourseRating": np.round(np.clip(RNG.normal(4.0, 0.55, n_courses), 1.0, 5.0), 1),
        "TeacherID": RNG.choice(teachers["TeacherID"], n_courses),
    })
    # A realistic sprinkle of missing ratings on newer courses.
    missing = RNG.choice(n_courses, size=int(0.07 * n_courses), replace=False)
    courses.loc[missing, "CourseRating"] = np.nan

    merged = courses.merge(teachers, on="TeacherID", how="left")

    # ---- Latent demand ------------------------------------------------------------------
    price_effect = (merged["CoursePrice"] / merged["CoursePrice"].median()) ** -0.75
    rating_effect = 1 + 0.35 * (merged["CourseRating"].fillna(4.0) - 4.0)
    teacher_effect = 1 + 0.22 * (merged["TeacherRating"] - 4.0)
    exp_effect = 1 + 0.012 * merged["YearsOfExperience"]
    match_effect = np.where(merged["Expertise"] == merged["CourseCategory"], 1.30, 1.0)
    cat_effect = merged["CourseCategory"].map(CATEGORY_PULL).to_numpy()
    type_effect = merged["CourseType"].map(TYPE_PULL).to_numpy()
    level_effect = merged["CourseLevel"].map(LEVEL_PULL).to_numpy()

    lam = (26 * price_effect * rating_effect * teacher_effect * exp_effect *
           match_effect * cat_effect * type_effect * level_effect)
    lam = np.clip(lam * RNG.lognormal(0, 0.35, n_courses), 0.4, None)
    enrolments = RNG.poisson(lam)

    # ---- Users ----------------------------------------------------------------------
    users = pd.DataFrame({
        "UserID": [f"U{i:05d}" for i in range(1, n_users + 1)],
        "UserName": [f"user{i}" for i in range(1, n_users + 1)],
        "Age": RNG.integers(15, 36, n_users),
        "Gender": RNG.choice(["Male", "Female"], n_users),
        "Email": [f"user{i}@example.com" for i in range(1, n_users + 1)],
    })

    # ---- Transactions -----------------------------------------------------------------
    end = pd.Timestamp("2026-06-30")
    start = end - pd.DateOffset(months=months)
    span_days = (end - start).days

    rows = []
    tid = 1
    for i, course_id in enumerate(courses["CourseID"]):
        n = int(enrolments[i])
        if n == 0:
            continue
        # Courses launch at different points; growth trends upward over the window.
        launch = RNG.integers(0, max(1, span_days - 60))
        offsets = np.clip(launch + RNG.beta(1.6, 1.6, n) * (span_days - launch), 0, span_days)
        list_price = courses.loc[i, "CoursePrice"]
        for off in offsets:
            discount = RNG.choice([1.0, 0.9, 0.75, 0.5], p=[0.62, 0.20, 0.13, 0.05])
            rows.append({
                "TransactionID": f"TX{tid:07d}",
                "UserID": RNG.choice(users["UserID"]),
                "CourseID": course_id,
                "TransactionDate": start + pd.Timedelta(days=float(off)),
                "Amount": round(list_price * discount, 2),
            })
            tid += 1

    transactions = pd.DataFrame(rows).sort_values("TransactionDate").reset_index(drop=True)
    return {"Users": users, "Teachers": teachers, "Courses": courses,
            "Transactions": transactions}


def main() -> str:
    sheets = build()
    path = config.DATA_DIR / "SAMPLE_EduPro Online Platform.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name, index=False)
    print(f"Wrote synthetic workbook -> {path}")
    for name, df in sheets.items():
        print(f"  {name:14s} {df.shape}")
    return str(path)


if __name__ == "__main__":
    main()
