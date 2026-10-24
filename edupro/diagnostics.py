"""
diagnostics.py
==============
Tests whether the targets actually contain learnable structure, before any model is fitted.

This runs first for a reason. A regression pipeline will always return a number, and a
model fitted to noise still reports an R². If enrolment counts are indistinguishable from
random allocation, then every "demand driver" the pipeline surfaces afterwards is an
artefact, and the responsible thing is to say so rather than to dress it up.

Four checks:

1. DISPERSION       Are enrolment counts more variable than pure counting noise?
                    If 10,000 transactions were sprinkled at random over 60 courses, each
                    course would still get a different number purely by chance. The variance
                    of that null process is known, so observed variance can be compared
                    against it. Index of dispersion (variance / mean) near 1.0 means the
                    spread is entirely explained by chance.

2. UNIFORMITY       A chi-square goodness-of-fit test against equal enrolment rates. A high
                    p-value means the data give no evidence that any course is more popular
                    than any other.

3. ATTRIBUTE SIGNAL Correlation (numeric) and one-way ANOVA (categorical) between each
                    course attribute and the target, with a Bonferroni correction applied.
                    Testing eight attributes at p < 0.05 produces a false positive roughly
                    one time in three by chance alone, so an uncorrected p-value is not
                    evidence of anything.

4. TAUTOLOGY        Is revenue simply price x enrolments? If so, a revenue model's R² is a
                    restatement of price, not a prediction, and it must be scored against
                    that naive rule rather than against the mean.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def dispersion_test(counts: pd.Series) -> dict:
    """Compare observed count variance against the Poisson null."""
    obs = np.asarray(counts, dtype=float)
    mean, var = float(obs.mean()), float(obs.var(ddof=1))
    index = var / mean if mean > 0 else float("nan")

    # Under a Poisson null, (n-1) * var / mean follows chi-square with n-1 df.
    n = len(obs)
    stat = (n - 1) * index
    p_over = float(1 - stats.chi2.cdf(stat, n - 1))     # p for over-dispersion

    return {
        "n_courses": n,
        "mean": round(mean, 2),
        "variance": round(var, 2),
        "sd": round(float(obs.std(ddof=1)), 2),
        "index_of_dispersion": round(index, 3),
        "p_overdispersed": round(p_over, 4),
        "interpretation": (
            "variance exceeds counting noise - real differences in course popularity exist"
            if p_over < 0.05 else
            "variance is fully consistent with chance - no evidence that any course is "
            "more popular than any other"
        ),
    }


def uniformity_test(counts: pd.Series) -> dict:
    """Chi-square goodness-of-fit against equal enrolment rates across courses."""
    obs = np.asarray(counts, dtype=float)
    chi2, p = stats.chisquare(obs)
    return {
        "chi2": round(float(chi2), 2),
        "df": len(obs) - 1,
        "p_value": round(float(p), 4),
        "rejects_uniform": bool(p < 0.05),
        "interpretation": (
            "enrolments are not uniformly distributed - some courses genuinely attract more"
            if p < 0.05 else
            "cannot reject uniform allocation: the enrolment pattern is statistically "
            "indistinguishable from transactions assigned to courses at random"
        ),
    }


def attribute_signal(
    frame: pd.DataFrame,
    target: str,
    numeric: list[str],
    categorical: list[str],
) -> dict:
    """Per-attribute association with the target, Bonferroni-corrected."""
    tests: list[dict] = []

    for col in numeric:
        if col not in frame or frame[col].nunique() < 3:
            continue
        r, p = stats.pearsonr(frame[col].astype(float), frame[target].astype(float))
        tests.append({"feature": col, "test": "pearson", "statistic": round(float(r), 3),
                      "p_value": round(float(p), 4)})

    for col in categorical:
        if col not in frame or frame[col].nunique() < 2:
            continue
        groups = [g[target].astype(float).values for _, g in frame.groupby(col)
                  if len(g) > 1]
        if len(groups) < 2:
            continue
        f, p = stats.f_oneway(*groups)
        tests.append({"feature": col, "test": "anova", "statistic": round(float(f), 3),
                      "p_value": round(float(p), 4)})

    n_tests = max(len(tests), 1)
    threshold = 0.05 / n_tests
    for t in tests:
        t["significant_raw"] = bool(t["p_value"] < 0.05)
        t["significant_corrected"] = bool(t["p_value"] < threshold)

    survivors = [t["feature"] for t in tests if t["significant_corrected"]]
    raw_only = [t["feature"] for t in tests if t["significant_raw"]
                and not t["significant_corrected"]]

    return {
        "target": target,
        "n_tests": n_tests,
        "bonferroni_threshold": round(threshold, 5),
        "tests": sorted(tests, key=lambda t: t["p_value"]),
        "survivors": survivors,
        "significant_before_correction_only": raw_only,
        "interpretation": (
            f"{len(survivors)} attribute(s) associated with {target} after correction: "
            f"{', '.join(survivors)}"
            if survivors else
            f"no attribute is associated with {target} once corrected for testing "
            f"{n_tests} hypotheses"
        ),
    }


def tautology_check(frame: pd.DataFrame) -> dict:
    """
    Is revenue a deterministic function of price and enrolments?

    If it is, the meaningful benchmark for a revenue model is not the mean but the naive
    rule 'revenue = average enrolments x price', which requires no model at all.
    """
    if not {"CourseRevenue", "CoursePrice", "EnrollmentCount"}.issubset(frame.columns):
        return {}

    revenue = frame["CourseRevenue"].astype(float)
    price = frame["CoursePrice"].astype(float)
    enrol = frame["EnrollmentCount"].astype(float)

    exact = bool(np.allclose(revenue, price * enrol, rtol=1e-6, atol=1e-6))
    mean_enrol = float(enrol.mean())
    naive = mean_enrol * price
    ss_res = float(((revenue - naive) ** 2).sum())
    ss_tot = float(((revenue - revenue.mean()) ** 2).sum())
    naive_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    zero_price = int((price == 0).sum())

    return {
        "revenue_is_exactly_price_times_enrolments": exact,
        "mean_enrollments": round(mean_enrol, 1),
        "naive_rule": f"Revenue = {mean_enrol:.1f} x CoursePrice",
        "naive_rule_r2": round(naive_r2, 4),
        "naive_rule_mae": round(float((revenue - naive).abs().mean()), 2),
        "zero_price_courses": zero_price,
        "zero_price_share_pct": round(100 * zero_price / len(frame), 1),
        "interpretation": (
            f"revenue is arithmetically determined by price and enrolments, and a rule with "
            f"no model at all reaches R² = {naive_r2:.4f}. Any learned model must be scored "
            f"against this, not against the mean."
            if exact else
            "revenue is not an exact function of list price and enrolments, so discounting "
            "or partial payment is present and a revenue model has something to learn."
        ),
    }


def run_diagnostics(
    frame: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
    targets: tuple[str, ...] = ("EnrollmentCount",),
) -> dict:
    """Run every check and return one report dict."""
    out: dict = {}
    counts = frame["EnrollmentCount"]

    out["dispersion"] = dispersion_test(counts)
    out["uniformity"] = uniformity_test(counts)
    out["attribute_signal"] = {
        target: attribute_signal(frame, target, numeric, categorical)
        for target in targets if target in frame.columns
    }
    out["tautology"] = tautology_check(frame)

    # Headline verdict, used by the report generator to decide how to frame everything.
    no_dispersion = out["dispersion"]["p_overdispersed"] >= 0.05
    no_uniformity_break = not out["uniformity"]["rejects_uniform"]
    no_attributes = not out["attribute_signal"].get("EnrollmentCount", {}).get("survivors")

    out["verdict"] = {
        "demand_signal_present": not (no_dispersion and no_uniformity_break and no_attributes),
        "summary": (
            "NO DEMAND SIGNAL. Enrolment counts are consistent with random allocation, and "
            "no course attribute predicts them after correcting for multiple testing. Any "
            "positive R² a model reports for enrolment on this data is fitted noise."
            if (no_dispersion and no_uniformity_break and no_attributes) else
            "Demand signal detected. Course attributes carry information about enrolment."
        ),
    }
    return out


def print_report(diag: dict) -> None:
    """Console summary."""
    d, u = diag["dispersion"], diag["uniformity"]
    print(f"  Dispersion : mean={d['mean']} var={d['variance']} "
          f"index={d['index_of_dispersion']} (1.00 = pure chance)")
    print(f"               {d['interpretation']}")
    print(f"  Uniformity : chi2={u['chi2']} df={u['df']} p={u['p_value']}")
    print(f"               {u['interpretation']}")

    for target, sig in diag["attribute_signal"].items():
        print(f"  Attributes -> {target}: {sig['n_tests']} tests, "
              f"Bonferroni p < {sig['bonferroni_threshold']}")
        for t in sig["tests"][:5]:
            mark = "*" if t["significant_corrected"] else (
                "~" if t["significant_raw"] else " ")
            print(f"      {mark} {t['feature']:20s} {t['test']:8s} "
                  f"stat={t['statistic']:+8.3f}  p={t['p_value']:.4f}")
        print(f"               {sig['interpretation']}")

    if diag.get("tautology"):
        t = diag["tautology"]
        print(f"  Revenue    : {t['interpretation']}")

    print(f"\n  VERDICT: {diag['verdict']['summary']}")
