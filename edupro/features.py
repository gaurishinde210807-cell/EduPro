"""
features.py
===========
Turns the merged course table into a modelling matrix.

Feature families, mapped to the project brief:

  Course        - price bands, duration buckets, rating tiers, ordinal level encoding
  Instructor    - experience buckets, teacher rating, expertise/category match score
  Historical    - past enrolments, past average revenue, revenue per enrolment
                  (temporal frame only; computed strictly before the cutoff date)
  Context       - price relative to the category median, price per hour, catalogue depth

Every function here is deterministic and stateless, so the same code path is used at
training time and inside the Streamlit app when a user invents a hypothetical course.
"""

from __future__ import annotations

from difflib import SequenceMatcher

import numpy as np
import pandas as pd

import config


# ======================================================================================
# Small helpers
# ======================================================================================
def _safe_qcut(series: pd.Series, labels: list[str]) -> pd.Series:
    """
    Quantile-bin a series, degrading gracefully when there are not enough distinct values.

    Real catalogues often price everything at a handful of round numbers, which makes a
    plain qcut raise. Falling back to equal-width bins keeps the pipeline running.
    """
    s = pd.to_numeric(series, errors="coerce")
    try:
        binned = pd.qcut(s, q=len(labels), labels=labels, duplicates="drop")
        if binned.cat.categories.size == len(labels):
            return binned.astype(str)
    except (ValueError, IndexError):
        pass
    try:
        return pd.cut(s, bins=len(labels), labels=labels).astype(str)
    except (ValueError, IndexError):
        return pd.Series([labels[len(labels) // 2]] * len(s), index=s.index)


def expertise_match(expertise: str, category: str) -> float:
    """
    Score how well an instructor's stated expertise matches the course category, 0-1.

    Exact match scores 1.0. Otherwise we take the better of a token-overlap (Jaccard)
    score and a character-level similarity ratio, so "Data Science" against "Data
    Analytics" scores well while "Music" against "Networking" does not.
    """
    if not isinstance(expertise, str) or not isinstance(category, str):
        return 0.0
    a, b = expertise.strip().lower(), category.strip().lower()
    if not a or not b or a == "unknown" or b == "unknown":
        return 0.0
    if a == b:
        return 1.0

    ta, tb = set(a.replace("&", " ").split()), set(b.replace("&", " ").split())
    jaccard = len(ta & tb) / len(ta | tb) if (ta | tb) else 0.0
    ratio = SequenceMatcher(None, a, b).ratio()
    return float(round(max(jaccard, ratio), 4))


def encode_level(level) -> float:
    """Ordinal encoding for course level: Beginner=1, Intermediate=2, Advanced=3."""
    if not isinstance(level, str):
        return 2.0
    return float(config.LEVEL_ORDER.get(level.strip().lower(), 2))


# ======================================================================================
# Main feature builder
# ======================================================================================
def engineer(df: pd.DataFrame, reference: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Add engineered columns to a merged course frame.

    `reference` supplies the population statistics (category medians, bin edges) used to
    place a row in context. At training time it is the frame itself; at prediction time
    inside the app it is the training frame, so a single hypothetical course is scored
    against the real catalogue rather than against itself.
    """
    out = df.copy()
    ref = reference if reference is not None else out

    # ---- Course features ---------------------------------------------------------------
    out["PriceBand"] = _safe_qcut(out["CoursePrice"], config.PRICE_BANDS)
    out["DurationBucket"] = _safe_qcut(out["CourseDuration"], config.DURATION_BUCKETS)
    out["RatingTier"] = pd.cut(
        out["CourseRating"],
        bins=config.RATING_TIER_EDGES,
        labels=config.RATING_TIERS,
        include_lowest=True,
    ).astype(str)
    out["LevelEncoded"] = out["CourseLevel"].map(encode_level)

    # ---- Instructor features -----------------------------------------------------------
    out["ExperienceBucket"] = pd.cut(
        out["YearsOfExperience"],
        bins=config.EXPERIENCE_EDGES,
        labels=config.EXPERIENCE_BUCKETS,
    ).astype(str)

    # When the teacher link comes from the transaction ledger, data_prep supplies the share
    # of a course's sales handled by a matching-expertise teacher. That is strictly more
    # informative than comparing a single modal expertise label, so prefer it.
    if "ExpertiseMatchRate" in out.columns:
        out["ExpertiseMatch"] = out["ExpertiseMatchRate"].fillna(0.0)
    else:
        out["ExpertiseMatch"] = [
            expertise_match(e, c) for e, c in zip(out["Expertise"], out["CourseCategory"])
        ]
    out["ExactExpertiseMatch"] = (out["ExpertiseMatch"] >= 0.999).astype(int)

    # Combined signal: a highly-rated instructor teaching inside their specialism.
    out["InstructorQualityScore"] = (
        out["TeacherRating"].fillna(0) * (0.5 + 0.5 * out["ExpertiseMatch"])
    )

    # ---- Monetisation ------------------------------------------------------------------
    # A catalogue that mixes free and paid courses has a hard structural break at price
    # zero: free courses generate exactly zero revenue whatever their demand. Modelling
    # that as a smooth price effect is wrong, so the break is given its own feature.
    out["IsFree"] = (out["CoursePrice"].fillna(0) <= 0).astype(int)

    # ---- Context features --------------------------------------------------------------
    cat_price = ref.groupby("CourseCategory")["CoursePrice"].median()
    global_price = float(ref["CoursePrice"].median())
    out["CategoryMedianPrice"] = out["CourseCategory"].map(cat_price).fillna(global_price)
    out["PriceVsCategory"] = out["CoursePrice"] / out["CategoryMedianPrice"].replace(0, np.nan)
    out["PriceVsCategory"] = out["PriceVsCategory"].fillna(1.0)

    out["PricePerHour"] = out["CoursePrice"] / out["CourseDuration"].replace(0, np.nan)
    out["PricePerHour"] = out["PricePerHour"].fillna(out["PricePerHour"].median())

    cat_size = ref.groupby("CourseCategory")["CourseID"].nunique()
    out["CategoryCourseCount"] = out["CourseCategory"].map(cat_size).fillna(1).astype(float)

    if "TeacherID" in ref.columns and "TeacherID" in out.columns:
        portfolio = ref.groupby("TeacherID")["CourseID"].nunique()
        out["TeacherPortfolioSize"] = out["TeacherID"].map(portfolio).fillna(1).astype(float)
    else:
        out["TeacherPortfolioSize"] = 1.0

    # Interaction terms the tree models can find on their own but the linear ones cannot.
    out["RatingXExperience"] = out["CourseRating"] * out["YearsOfExperience"]
    out["PriceXRating"] = out["CoursePrice"] * out["CourseRating"]

    return out


# ======================================================================================
# Feature specifications
# ======================================================================================
COLD_START_NUMERIC = [
    "CoursePrice", "CourseDuration", "CourseRating", "LevelEncoded",
    "YearsOfExperience", "TeacherRating", "ExpertiseMatch", "ExactExpertiseMatch",
    "InstructorQualityScore", "PriceVsCategory", "PricePerHour",
    "CategoryMedianPrice", "CategoryCourseCount", "TeacherPortfolioSize", "IsFree",
    "RatingXExperience", "PriceXRating", "RatingMissing",
]

COLD_START_CATEGORICAL = [
    "CourseCategory", "CourseType", "CourseLevel",
    "PriceBand", "DurationBucket", "RatingTier", "ExperienceBucket",
]

HISTORY_NUMERIC = [
    "HistEnrollments", "HistRevenue", "HistAvgTicket",
    "HistEnrollmentsPerMonth", "HistRevenuePerEnrollment",
    "HistActiveMonths", "HistDaysSinceLastSale", "IsNewCourse",
]


def feature_spec(frame: pd.DataFrame, include_history: bool = False) -> tuple[list[str], list[str]]:
    """Return (numeric, categorical) feature names that actually exist in `frame`."""
    numeric = [c for c in COLD_START_NUMERIC if c in frame.columns]
    if include_history:
        numeric += [c for c in HISTORY_NUMERIC if c in frame.columns]
    categorical = [c for c in COLD_START_CATEGORICAL if c in frame.columns]
    return numeric, categorical


# Headline business levers. These are the features the report has to be able to talk
# about - price sensitivity, rating influence, instructor experience - so they are never
# the one discarded when a correlated pair is pruned. Derived and interaction terms go
# first instead.
PROTECTED_FEATURES = [
    "CoursePrice", "CourseDuration", "CourseRating", "LevelEncoded",
    "YearsOfExperience", "TeacherRating", "ExpertiseMatch",
    "HistEnrollments", "HistRevenue",
]


def drop_correlated(
    frame: pd.DataFrame, numeric: list[str], threshold: float | None = None
) -> tuple[list[str], list[tuple[str, str, float]]]:
    """
    Remove one of every pair of numeric features correlated above `threshold`.

    When a pair is too correlated, the member that is NOT in PROTECTED_FEATURES is the one
    dropped. If both are protected, both are kept and the pair is reported instead - losing
    a headline lever to save a little collinearity is a bad trade when the whole point of
    the analysis is to say how price and rating drive demand.

    Returns (surviving features, [(kept, dropped, r), ...]).
    """
    threshold = threshold or config.CORRELATION_THRESHOLD
    cols = [c for c in numeric if c in frame.columns]
    if len(cols) < 2:
        return cols, []

    corr = frame[cols].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    dropped: list[tuple[str, str, float]] = []
    keep = list(cols)
    for col in upper.columns:
        if col not in keep:
            continue
        partners = [p for p in upper.index[upper[col] > threshold].tolist()
                    if p in keep and p != col]
        for partner in partners:
            col_protected = col in PROTECTED_FEATURES
            partner_protected = partner in PROTECTED_FEATURES
            if col_protected and partner_protected:
                continue                        # keep both, note it in the report
            victim = partner if not partner_protected else col
            survivor = col if victim == partner else partner
            if victim in keep:
                keep.remove(victim)
                dropped.append((survivor, victim, round(float(corr.loc[partner, col]), 4)))
            if victim == col:
                break                           # `col` is gone, stop pairing against it
    return keep, dropped
