"""
data_prep.py
============
Loads the EduPro workbook, standardises it, merges Courses <-> Teachers <-> Transactions,
and aggregates the transaction ledger up to course level.

Two framings are produced, because they answer two different business questions:

1. CROSS-SECTIONAL ("cold start")
   One row per course. Targets are total enrolments and total revenue across the whole
   ledger. Features are course and instructor attributes only. This is the model you use
   when deciding whether to *launch* a course that does not exist yet, so it deliberately
   has no historical-performance features.

2. TEMPORAL ("forecast")
   The ledger is cut at a date. Features are built from transactions strictly BEFORE the
   cut; targets are enrolments and revenue strictly AFTER it. This is the model you use to
   forecast next quarter for courses already on the platform, and the split is what keeps
   "past enrolment count" from leaking the answer.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

import config


# ======================================================================================
# Column / sheet name resolution
# ======================================================================================
def _norm(name: str) -> str:
    """Lowercase a name and strip everything that is not a letter or digit."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _match_sheet(available: list[str], aliases: list[str]) -> str | None:
    """Find the sheet whose normalised name best matches one of `aliases`."""
    norm_map = {_norm(s): s for s in available}
    for alias in aliases:                      # exact normalised match wins
        if alias in norm_map:
            return norm_map[alias]
    for alias in aliases:                      # then substring match
        for norm, original in norm_map.items():
            if alias in norm or norm in alias:
                return original
    return None


def canonicalise(df: pd.DataFrame, table: str) -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Rename the columns of `df` to the canonical names declared in config.COLUMN_ALIASES.

    Returns the renamed frame and a {canonical: original} map so the data-quality report
    can show exactly what was matched to what.
    """
    aliases = config.COLUMN_ALIASES[table]
    norm_map = {_norm(c): c for c in df.columns}
    resolved: dict[str, str] = {}
    used: set[str] = set()

    # Pass 1 - the column is already spelled (near enough to) the canonical name.
    for canonical in aliases:
        key = _norm(canonical)
        if key in norm_map and norm_map[key] not in used:
            resolved[canonical] = norm_map[key]
            used.add(norm_map[key])

    # Pass 2 - fall back to the alias list, first unclaimed exact match wins.
    for canonical, alias_list in aliases.items():
        if canonical in resolved:
            continue
        for alias in alias_list:
            if alias in norm_map and norm_map[alias] not in used:
                resolved[canonical] = norm_map[alias]
                used.add(norm_map[alias])
                break

    # Pass 3 - substring fallback, so "Duration (hours)" still resolves to CourseDuration.
    # Only the alias-inside-column direction is allowed (not the reverse, which matches far
    # too eagerly) and only for aliases of four characters or more, so short tokens like
    # "id" cannot claim an unrelated column. Anything matched here is flagged, because a
    # fuzzy match is exactly the kind of thing worth eyeballing before trusting a result.
    fuzzy: dict[str, str] = {}
    for canonical, alias_list in aliases.items():
        if canonical in resolved:
            continue
        for alias in alias_list:
            if len(alias) < 4:
                continue
            hits = [orig for norm, orig in norm_map.items()
                    if alias in norm and orig not in used]
            if hits:
                pick = min(hits, key=len)          # closest in length to the alias
                resolved[canonical] = pick
                fuzzy[canonical] = pick
                used.add(pick)
                break

    if fuzzy:
        warnings.warn(
            f"[{table}] resolved by fuzzy match, please verify: "
            + ", ".join(f"{k} <- '{v}'" for k, v in fuzzy.items()),
            RuntimeWarning,
        )

    out = df.rename(columns={v: k for k, v in resolved.items()}).copy()
    return out, resolved


# ======================================================================================
# Loading
# ======================================================================================
@dataclass
class EduProData:
    """Container for the cleaned tables plus a record of what happened during loading."""
    courses: pd.DataFrame
    teachers: pd.DataFrame
    transactions: pd.DataFrame
    users: pd.DataFrame
    quality: dict = field(default_factory=dict)


def load_workbook(path: str | Path | None = None) -> EduProData:
    """Read every sheet of the workbook and standardise it."""
    path = Path(path) if path else config.WORKBOOK_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find the workbook at {path}.\n"
            f"Download 'EduPro Online Platform.xlsx' from the project's Google Drive link "
            f"and save it to {config.DATA_DIR}/"
        )

    book = pd.read_excel(path, sheet_name=None, engine="openpyxl")
    available = list(book.keys())
    quality: dict = {"workbook": str(path), "sheets_found": available, "column_mapping": {}}

    tables: dict[str, pd.DataFrame] = {}
    for logical, aliases in config.SHEET_ALIASES.items():
        sheet = _match_sheet(available, aliases)
        if sheet is None:
            if logical == "users":                # users are optional for this analysis
                tables[logical] = pd.DataFrame()
                continue
            raise KeyError(
                f"No sheet matching '{logical}' in the workbook. Sheets present: {available}"
            )
        df, mapping = canonicalise(book[sheet], logical)
        df.columns = [str(c).strip() for c in df.columns]
        tables[logical] = df
        quality["column_mapping"][logical] = {"sheet": sheet, "resolved": mapping}

    data = EduProData(
        courses=tables["courses"],
        teachers=tables["teachers"],
        transactions=tables["transactions"],
        users=tables["users"],
        quality=quality,
    )
    _clean_inplace(data)
    return data


def _to_numeric(series: pd.Series) -> pd.Series:
    """Coerce a column to numeric, tolerating currency symbols and thousands separators."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    cleaned = (
        series.astype(str)
        .str.replace(r"[^\d.\-]", "", regex=True)
        .replace({"": np.nan, "-": np.nan, ".": np.nan})
    )
    return pd.to_numeric(cleaned, errors="coerce")


def _clean_inplace(data: EduProData) -> None:
    """Type coercion, whitespace trimming, de-duplication and a missingness audit."""
    q = data.quality
    q["raw_shapes"] = {
        "courses": data.courses.shape,
        "teachers": data.teachers.shape,
        "transactions": data.transactions.shape,
        "users": data.users.shape,
    }

    # ---- text columns -----------------------------------------------------------------
    for df in (data.courses, data.teachers, data.transactions, data.users):
        for col in df.select_dtypes(include=["object", "str"]).columns:
            df[col] = df[col].astype(str).str.strip().replace({"nan": np.nan, "": np.nan})

    # ---- numeric columns --------------------------------------------------------------
    for col in ["CoursePrice", "CourseDuration", "CourseRating"]:
        if col in data.courses:
            data.courses[col] = _to_numeric(data.courses[col])
    for col in ["YearsOfExperience", "TeacherRating"]:
        if col in data.teachers:
            data.teachers[col] = _to_numeric(data.teachers[col])
    if "Amount" in data.transactions:
        data.transactions["Amount"] = _to_numeric(data.transactions["Amount"])

    # ---- dates ------------------------------------------------------------------------
    if "TransactionDate" in data.transactions:
        data.transactions["TransactionDate"] = pd.to_datetime(
            data.transactions["TransactionDate"], errors="coerce"
        )

    # ---- de-duplicate -----------------------------------------------------------------
    q["duplicates_dropped"] = {}
    if "CourseID" in data.courses:
        before = len(data.courses)
        data.courses.drop_duplicates(subset="CourseID", keep="first", inplace=True)
        q["duplicates_dropped"]["courses"] = before - len(data.courses)
    if "TeacherID" in data.teachers:
        before = len(data.teachers)
        data.teachers.drop_duplicates(subset="TeacherID", keep="first", inplace=True)
        q["duplicates_dropped"]["teachers"] = before - len(data.teachers)
    if "TransactionID" in data.transactions:
        before = len(data.transactions)
        data.transactions.drop_duplicates(subset="TransactionID", keep="first", inplace=True)
        q["duplicates_dropped"]["transactions"] = before - len(data.transactions)

    # ---- drop unusable transaction rows ------------------------------------------------
    tx = data.transactions
    if {"CourseID", "TransactionDate"}.issubset(tx.columns):
        before = len(tx)
        tx.dropna(subset=["CourseID", "TransactionDate"], inplace=True)
        q["transactions_dropped_missing_key"] = before - len(tx)

    # Transactions pointing at a course that is not in the catalogue are counted and kept
    # out of the modelling frame, but reported, because a large number would signal a
    # broken join rather than a data-quality nuisance.
    if "CourseID" in tx.columns and "CourseID" in data.courses.columns:
        orphans = ~tx["CourseID"].isin(set(data.courses["CourseID"]))
        q["orphan_transactions"] = int(orphans.sum())
        q["orphan_transaction_pct"] = round(100 * float(orphans.mean()), 2) if len(tx) else 0.0

    # ---- missingness audit --------------------------------------------------------------
    q["missing_pct"] = {
        name: (df.isna().mean() * 100).round(2).to_dict()
        for name, df in [
            ("courses", data.courses),
            ("teachers", data.teachers),
            ("transactions", data.transactions),
        ]
        if len(df)
    }

    if "TransactionDate" in tx.columns and len(tx):
        q["date_range"] = {
            "min": str(tx["TransactionDate"].min().date()),
            "max": str(tx["TransactionDate"].max().date()),
        }


# ======================================================================================
# Aggregation
# ======================================================================================
def aggregate_transactions(
    tx: pd.DataFrame,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    prefix: str = "",
) -> pd.DataFrame:
    """
    Roll the transaction ledger up to one row per CourseID over the window [start, end).

    `prefix` lets the same function produce history features ("Hist") and future targets
    without column-name collisions.
    """
    window = tx
    if start is not None:
        window = window[window["TransactionDate"] >= start]
    if end is not None:
        window = window[window["TransactionDate"] < end]

    if window.empty:
        return pd.DataFrame(columns=["CourseID"]).set_index("CourseID")

    agg = window.groupby("CourseID").agg(
        Enrollments=("CourseID", "size"),
        Revenue=("Amount", "sum"),
        AvgTicket=("Amount", "mean"),
        MedianTicket=("Amount", "median"),
        FirstSale=("TransactionDate", "min"),
        LastSale=("TransactionDate", "max"),
    )
    if "UserID" in window.columns:
        agg["UniqueLearners"] = window.groupby("CourseID")["UserID"].nunique()

    # Months on sale inside the window, so that a course listed for one month is not
    # compared like-for-like against one listed for two years.
    span_days = (agg["LastSale"] - agg["FirstSale"]).dt.days.clip(lower=0)
    agg["ActiveMonths"] = (span_days / 30.44 + 1).round(2)
    agg["EnrollmentsPerMonth"] = agg["Enrollments"] / agg["ActiveMonths"]
    agg["RevenuePerEnrollment"] = agg["Revenue"] / agg["Enrollments"].replace(0, np.nan)

    if prefix:
        agg = agg.add_prefix(prefix)
    return agg


def _impute_ratings(courses: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing course ratings with the category median (global median as backstop) and
    keep an explicit indicator column, because 'no rating yet' is itself informative -
    it usually means the course is new or has very few enrolments.
    """
    df = courses.copy()
    if "CourseRating" not in df:
        df["CourseRating"] = np.nan
    df["RatingMissing"] = df["CourseRating"].isna().astype(int)
    if "CourseCategory" in df:
        df["CourseRating"] = df.groupby("CourseCategory")["CourseRating"].transform(
            lambda s: s.fillna(s.median())
        )
    df["CourseRating"] = df["CourseRating"].fillna(df["CourseRating"].median())
    return df


def _teacher_via_transactions(
    courses: pd.DataFrame, teachers: pd.DataFrame, tx: pd.DataFrame
) -> tuple[pd.DataFrame | None, dict]:
    """
    Derive course-level instructor attributes when the only Course<->Teacher link is on the
    transaction ledger.

    A transaction-level TeacherID does not necessarily mean "this teacher owns this course".
    If many teachers appear against the same course, the teacher is attached to the sale (or
    to the learner), not to the course, and averaging their attributes produces a column that
    is nearly constant across the catalogue and therefore carries no information.

    That case is detected and reported rather than silently modelled, because a feature that
    looks populated but has no variance is worse than a missing one: it survives every
    quality check and quietly contributes nothing.
    """
    if "TeacherID" not in tx.columns or "TeacherID" not in teachers.columns:
        return None, {}

    per_course = tx.groupby("CourseID")["TeacherID"].nunique()
    info = {
        "link_source": "transactions",
        "teachers_per_course_median": float(per_course.median()),
        "teachers_per_course_max": int(per_course.max()),
    }

    cols = [c for c in ["TeacherID", "TeacherRating", "YearsOfExperience", "Expertise"]
            if c in teachers.columns]
    joined = tx[["CourseID", "TeacherID"]].merge(teachers[cols], on="TeacherID", how="left")

    agg: dict[str, pd.Series] = {}
    if "TeacherRating" in joined:
        agg["TeacherRating"] = joined.groupby("CourseID")["TeacherRating"].mean()
    if "YearsOfExperience" in joined:
        agg["YearsOfExperience"] = joined.groupby("CourseID")["YearsOfExperience"].mean()
    if "Expertise" in joined:
        # Modal expertise among the teachers transacting on this course.
        agg["Expertise"] = joined.groupby("CourseID")["Expertise"].agg(
            lambda s: s.mode().iloc[0] if not s.mode().empty else "Unknown"
        )
        # Share of this course's transactions handled by a matching-expertise teacher.
        cat = courses.set_index("CourseID")["CourseCategory"]
        joined["_cat"] = joined["CourseID"].map(cat)
        agg["ExpertiseMatchRate"] = (
            joined.assign(_m=(joined["Expertise"] == joined["_cat"]).astype(float))
            .groupby("CourseID")["_m"].mean()
        )

    derived = pd.DataFrame(agg).reset_index()

    # How much of the underlying teacher-level variation survives aggregation?
    if "TeacherRating" in derived and "TeacherRating" in teachers:
        raw_sd = float(teachers["TeacherRating"].std())
        agg_sd = float(derived["TeacherRating"].std())
        info["teacher_rating_sd_raw"] = round(raw_sd, 3)
        info["teacher_rating_sd_aggregated"] = round(agg_sd, 3)
        info["variation_retained_pct"] = round(100 * agg_sd / raw_sd, 1) if raw_sd else 0.0

    info["usable"] = per_course.median() <= 2
    if not info["usable"]:
        warnings.warn(
            f"TeacherID on the Transactions sheet is not a course-ownership link: the median "
            f"course has {per_course.median():.0f} distinct teachers against it. Instructor "
            f"attributes have been aggregated per course, but only "
            f"{info.get('variation_retained_pct', 0)}% of the teacher-level variation "
            f"survives averaging, so these features are expected to carry little signal.",
            RuntimeWarning,
        )
    return derived, info


def merge_course_teacher(data: EduProData) -> pd.DataFrame:
    """
    Join the course catalogue to instructor attributes.

    The link may live on the Courses sheet, on the Teachers sheet, or only on the
    transaction ledger. All three are attempted in that order of preference.
    """
    courses = _impute_ratings(data.courses)
    teachers = data.teachers.copy()
    link_info: dict = {}

    teacher_cols = [c for c in ["TeacherID", "Expertise", "YearsOfExperience", "TeacherRating"]
                    if c in teachers.columns]

    if "TeacherID" in courses.columns and "TeacherID" in teachers.columns:
        merged = courses.merge(teachers[teacher_cols], on="TeacherID", how="left")
        link_info = {"link_source": "courses.TeacherID", "usable": True}
    elif "CourseID" in teachers.columns:
        link = teachers[["CourseID"] + teacher_cols].drop_duplicates(subset="CourseID")
        merged = courses.merge(link, on="CourseID", how="left")
        link_info = {"link_source": "teachers.CourseID", "usable": True}
    else:
        derived, link_info = _teacher_via_transactions(courses, teachers, data.transactions)
        if derived is not None:
            merged = courses.merge(derived, on="CourseID", how="left")
        else:
            warnings.warn(
                "No Course<->Teacher link anywhere in the workbook. Instructor features will "
                "be filled with medians and will carry no signal.",
                RuntimeWarning,
            )
            merged = courses.copy()
            for col in ["Expertise", "YearsOfExperience", "TeacherRating"]:
                merged[col] = np.nan
            link_info = {"link_source": "none", "usable": False}

    data.quality["teacher_link"] = link_info

    for col in ["YearsOfExperience", "TeacherRating"]:
        if col not in merged:
            merged[col] = np.nan
        merged[f"{col}Missing"] = merged[col].isna().astype(int)
        merged[col] = merged[col].fillna(merged[col].median())
    if "Expertise" not in merged:
        merged["Expertise"] = np.nan
    merged["Expertise"] = merged["Expertise"].fillna("Unknown")

    for col in ["CourseCategory", "CourseType", "CourseLevel"]:
        if col not in merged:
            merged[col] = "Unknown"
        merged[col] = merged[col].fillna("Unknown")

    for col in ["CoursePrice", "CourseDuration"]:
        if col not in merged:
            merged[col] = np.nan
        merged[col] = merged[col].fillna(merged[col].median())

    return merged


def build_cross_sectional(data: EduProData) -> pd.DataFrame:
    """One row per course, with lifetime enrolment and revenue as the targets."""
    base = merge_course_teacher(data)
    totals = aggregate_transactions(data.transactions)
    out = base.merge(totals, on="CourseID", how="left")

    # A course with no transactions has zero demand, not missing demand.
    out["EnrollmentCount"] = out["Enrollments"].fillna(0).astype(int)
    out["CourseRevenue"] = out["Revenue"].fillna(0.0)
    out["RevenuePerEnrollment"] = out["RevenuePerEnrollment"].fillna(0.0)
    out["ActiveMonths"] = out["ActiveMonths"].fillna(0.0)
    out["AvgTicket"] = out["AvgTicket"].fillna(out["CoursePrice"])

    # Realised price vs list price exposes discounting, which matters for revenue.
    out["RealisedPriceRatio"] = (out["AvgTicket"] / out["CoursePrice"].replace(0, np.nan)).fillna(1.0)
    return out


def choose_cutoff(tx: pd.DataFrame, months: int | None = None) -> pd.Timestamp:
    """Cut date that leaves `months` of ledger in the forecast window."""
    months = months or config.FORECAST_WINDOW_MONTHS
    last = tx["TransactionDate"].max()
    return (last - pd.DateOffset(months=months)).normalize()


def build_temporal(data: EduProData, cutoff: pd.Timestamp | None = None) -> tuple[pd.DataFrame, dict]:
    """
    Build the forecasting frame: history-window features, forecast-window targets.

    Returns the frame and a dict describing the split, for the write-up.
    """
    tx = data.transactions
    cutoff = cutoff or choose_cutoff(tx)
    first = tx["TransactionDate"].min()
    last = tx["TransactionDate"].max()
    history_months = (cutoff - first).days / 30.44

    info = {
        "cutoff": str(cutoff.date()),
        "history_start": str(first.date()),
        "history_months": round(history_months, 1),
        "forecast_end": str(last.date()),
        "forecast_months": round((last - cutoff).days / 30.44, 1),
        "usable": history_months >= config.MIN_HISTORY_MONTHS,
    }

    base = merge_course_teacher(data)
    hist = aggregate_transactions(tx, end=cutoff, prefix="Hist")
    future = aggregate_transactions(tx, start=cutoff)

    frame = base.merge(hist, on="CourseID", how="left").merge(
        future[["Enrollments", "Revenue"]].rename(
            columns={"Enrollments": "FutureEnrollments", "Revenue": "FutureRevenue"}
        ),
        on="CourseID",
        how="left",
    )

    for col in ["HistEnrollments", "HistRevenue", "HistActiveMonths",
                "HistEnrollmentsPerMonth", "HistRevenuePerEnrollment"]:
        if col in frame:
            frame[col] = frame[col].fillna(0.0)
    if "HistAvgTicket" in frame:
        frame["HistAvgTicket"] = frame["HistAvgTicket"].fillna(frame["CoursePrice"])

    # Days since the last sale before the cutoff - a strong recency signal.
    if "HistLastSale" in frame:
        frame["HistDaysSinceLastSale"] = (cutoff - frame["HistLastSale"]).dt.days
        frame["HistDaysSinceLastSale"] = frame["HistDaysSinceLastSale"].fillna(
            frame["HistDaysSinceLastSale"].max()
        )
        frame["IsNewCourse"] = frame["HistEnrollments"].eq(0).astype(int)

    frame["FutureEnrollments"] = frame["FutureEnrollments"].fillna(0).astype(int)
    frame["FutureRevenue"] = frame["FutureRevenue"].fillna(0.0)

    frame = frame.drop(columns=[c for c in ["HistFirstSale", "HistLastSale"] if c in frame])
    return frame, info


def category_revenue(df: pd.DataFrame, revenue_col: str = "CourseRevenue") -> pd.DataFrame:
    """Aggregate course-level revenue to category level."""
    out = (
        df.groupby("CourseCategory")
        .agg(
            Courses=("CourseID", "nunique"),
            Enrollments=("EnrollmentCount", "sum") if "EnrollmentCount" in df else ("CourseID", "size"),
            Revenue=(revenue_col, "sum"),
        )
        .reset_index()
        .sort_values("Revenue", ascending=False)
    )
    out["RevenueShare"] = 100 * out["Revenue"] / out["Revenue"].sum()
    out["RevenuePerCourse"] = out["Revenue"] / out["Courses"]
    return out


if __name__ == "__main__":
    d = load_workbook()
    print("Sheets:", d.quality["sheets_found"])
    print("Shapes:", d.quality["raw_shapes"])
    cs = build_cross_sectional(d)
    print(f"\nCross-sectional frame: {cs.shape}")
    print(cs[["CourseID", "CourseCategory", "CoursePrice", "EnrollmentCount", "CourseRevenue"]].head())
