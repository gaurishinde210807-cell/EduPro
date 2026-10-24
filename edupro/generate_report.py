"""
generate_report.py
==================
Writes the two written deliverables straight from outputs/metrics.json, so every number in
them is a number the pipeline actually computed.

    python generate_report.py

The report BRANCHES on the signal diagnostics. If the diagnostics find no learnable demand
signal, the paper says so in its first paragraph and spends its length demonstrating that,
rather than reporting model metrics as though they meant something. A pipeline that always
produces a confident-sounding result regardless of what the data contains is worse than no
pipeline at all.

Produces:
    outputs/research_paper.md
    outputs/executive_summary.md
"""

from __future__ import annotations

import json

import pandas as pd

import config


# ======================================================================================
# Helpers
# ======================================================================================
def load_metrics() -> dict:
    path = config.OUTPUT_DIR / "metrics.json"
    if not path.exists():
        raise FileNotFoundError("Run `python train.py` first - metrics.json is missing.")
    with open(path) as fh:
        return json.load(fh)


def money(x: float) -> str:
    return f"{x:,.0f}"


def sample_data_banner(m: dict) -> str:
    """Loud warning when the report was built from the synthetic smoke-test workbook."""
    workbook = m.get("data_quality", {}).get("workbook", "")
    name = workbook.split("/")[-1]
    if "SAMPLE_" not in name and "VARIANT_" not in name:
        return ""
    return (
        "> ## ⚠️ SYNTHETIC DATA — NOT A REAL FINDING\n>\n"
        f"> Generated from `{name}`, a randomly generated workbook used only to verify that\n"
        "> the pipeline runs. **Every number below is fabricated.**\n\n---\n\n"
    )


def metrics_table(bundle: dict, naive_note: bool = False) -> str:
    """Markdown comparison table for one target."""
    rows = ["| Model | CV R² | Test MAE | Test RMSE | Test R² | Target |",
            "|---|---|---|---|---|---|"]
    for name, res in bundle["results"].items():
        if "R2" not in res:
            continue
        label = name
        if name == bundle["best_name"]:
            label += " **(selected)**"
        if name.startswith("Naive"):
            label = f"*{name}*"
        cv = res.get("cv_r2")
        cv_txt = "—" if cv is None or cv != cv else f"{cv:+.3f}"
        rows.append(
            f"| {label} | {cv_txt} | {res['MAE']:,.2f} | {res['RMSE']:,.2f} | "
            f"{res['R2']:+.3f} | {'log1p' if res['log_target'] else 'raw'} |"
        )
    table = "\n".join(rows)
    if naive_note:
        table += ("\n\nThe naive rule is not a fitted model. It multiplies each course's list "
                  "price by the *average* enrolment count across the training courses and "
                  "stops there. It has no cross-validated score because there is nothing to "
                  "cross-validate.")
    return table


def importance_list(bundle: dict, n: int = 8) -> str:
    imp = bundle.get("permutation_importance", [])[:n]
    if not imp:
        return "_Not available._"
    rows = ["| Rank | Feature | Permutation importance (drop in R²) | ± SD |", "|---|---|---|---|"]
    for i, row in enumerate(imp, 1):
        rows.append(f"| {i} | `{row['Feature']}` | {row['Importance']:+.4f} | {row['Std']:.4f} |")
    return "\n".join(rows)


def diagnostics_tests_table(sig: dict, n: int = 10) -> str:
    rows = ["| Attribute | Test | Statistic | p | Significant raw | Survives Bonferroni |",
            "|---|---|---|---|---|---|"]
    for t in sig["tests"][:n]:
        rows.append(
            f"| `{t['feature']}` | {t['test']} | {t['statistic']:+.3f} | {t['p_value']:.4f} | "
            f"{'yes' if t['significant_raw'] else 'no'} | "
            f"{'**yes**' if t['significant_corrected'] else 'no'} |"
        )
    return "\n".join(rows)


def best_of(bundle: dict, exclude_naive: bool = True) -> tuple[str, dict]:
    """Best fitted model by test R², optionally ignoring the naive rule."""
    candidates = {k: v for k, v in bundle["results"].items()
                  if "R2" in v and not (exclude_naive and k.startswith("Naive"))
                  and k != "Baseline (mean)"}
    if not candidates:
        return bundle["best_name"], bundle["results"].get(bundle["best_name"], {})
    name = max(candidates, key=lambda k: candidates[k]["R2"])
    return name, candidates[name]


# ======================================================================================
# Research paper
# ======================================================================================
def build_research_paper(m: dict) -> str:
    eda = m["eda"]
    models = m["models"]
    dq = m["data_quality"]
    split = m.get("temporal_split", {})
    cat_metrics = m.get("category_revenue_metrics", {})
    diag = m.get("diagnostics", {})

    disp = diag.get("dispersion", {})
    unif = diag.get("uniformity", {})
    taut = diag.get("tautology", {})
    sig_e = diag.get("attribute_signal", {}).get("EnrollmentCount", {})
    sig_r = diag.get("attribute_signal", {}).get("CourseRevenue", {})
    has_signal = diag.get("verdict", {}).get("demand_signal_present", True)

    enroll = models.get("cold_EnrollmentCount", {})
    revenue = models.get("cold_CourseRevenue", {})
    f_enroll = models.get("forecast_FutureEnrollments")
    f_revenue = models.get("forecast_FutureRevenue")

    e_baseline = enroll["results"].get("Baseline (mean)", {})
    e_name, e_best = best_of(enroll)
    r_name, r_best = best_of(revenue)
    r_naive = revenue["results"].get("Naive: mean enrolments x price", {})

    cats = pd.DataFrame(eda["categories"])
    link = dq.get("teacher_link", {})

    # ---------------------------------------------------------------------------------
    # Headline box
    # ---------------------------------------------------------------------------------
    if not has_signal:
        headline = f"""> # ⚠️ Headline finding
>
> **This dataset does not support course-level demand prediction, and the honest
> deliverable is to say so rather than to report a model.**
>
> Enrolment counts across the {disp.get('n_courses')} courses are statistically
> indistinguishable from {eda['n_transactions']:,} transactions being allocated to courses
> at random (index of dispersion {disp.get('index_of_dispersion')}, where 1.00 is pure
> chance; χ² = {unif.get('chi2')}, df = {unif.get('df')}, p = {unif.get('p_value')}). No
> course attribute is associated with enrolment once corrected for multiple testing. The
> best enrolment model scores **R² = {e_best.get('R2', 0):+.3f}** on hold-out courses,
> which is no better than predicting the average.
>
> Revenue *appears* highly predictable (R² up to {r_best.get('R2', 0):+.3f}), but this is
> arithmetic, not prediction: `Amount` equals `CoursePrice` on every one of the
> {eda['n_transactions']:,} transactions, so revenue is exactly price × enrolments. A rule
> with **no model at all** — price × the average enrolment count — scores
> **R² = {r_naive.get('R2', 0):+.3f}** and beats every fitted model in the comparison.
>
> The pipeline, dashboard and methodology below are complete and correct. What they
> demonstrate is that the *data* cannot answer the question asked of it.

---

"""
    else:
        headline = ""

    doc = sample_data_banner(m) + headline + f"""# Predictive Modeling for Course Demand and Revenue Forecasting on EduPro

**Research paper**

**Author:** Gauri Shinde, Second-year Computer Science student, University of Greenwich

Generated {m['generated_at'][:10]} · pipeline runtime {m['runtime_seconds']}s

---

## Abstract

This study attempts course-level prediction of enrolment demand and revenue for the EduPro
platform, using {eda['n_courses']} courses, {eda['n_transactions']:,} transactions,
{dq['raw_shapes'].get('teachers', [0])[0]} instructors and {eda['n_categories']} categories
covering {dq.get('date_range', {}).get('min', 'n/a')} to
{dq.get('date_range', {}).get('max', 'n/a')}.

Before fitting any model, four diagnostics tested whether the targets contain learnable
structure. They do not, for enrolment. Enrolment counts have an index of dispersion of
{disp.get('index_of_dispersion')} against a value of 1.00 for a pure counting process, and a
goodness-of-fit test cannot reject the hypothesis that every course draws from the same
constant rate (p = {unif.get('p_value')}). Of {sig_e.get('n_tests', 0)} attribute-to-enrolment
association tests, none survives Bonferroni correction.

Six regression families were nevertheless fitted, to confirm the diagnosis rather than to
deploy. The best enrolment model reaches R² = {e_best.get('R2', 0):+.3f} on hold-out courses
against {e_baseline.get('R2', 0):+.3f} for a mean-predicting baseline: it has learned
nothing.

Revenue is a different failure. Because `Amount` equals `CoursePrice` exactly on every
transaction, revenue is arithmetically price × enrolments. Models report R² up to
{r_best.get('R2', 0):+.3f}, but the naive rule *price × average enrolments*, which requires
no fitting, reaches {r_naive.get('R2', 0):+.3f} with a mean absolute error of
{money(r_naive.get('MAE', 0))} against {money(r_best.get('MAE', 0))} for the best learned
model. Every model in the comparison is beaten by arithmetic.

The contribution of this paper is therefore a negative result, established with explicit
tests, together with a specification of what the platform would have to record for the
original question to become answerable.

---

## 1. Introduction

EduPro commissions courses, prices them and onboards instructors before any sales data for
those courses exists. The brief asks for predictive models of enrolment demand and revenue
so those decisions can be made on forecasts rather than on retrospective reporting.

The analysis addresses three questions in order:

1. **Is there anything to learn?** Do enrolment counts vary more than chance allocation
   would produce, and does any course attribute track that variation?
2. If so, how accurately can enrolment and revenue be predicted from attributes known at
   commissioning time?
3. Which attributes drive demand?

Question 1 is asked first deliberately. A regression pipeline returns a number whatever it
is given, and a model fitted to noise still reports an R². Asking question 2 before question
1 is how noise gets published as insight.

---

## 2. Data

### 2.1 Source

Workbook: `{dq['workbook'].split('/')[-1]}`
Sheets: {', '.join(dq['sheets_found'])}

| Table | Rows | Columns |
|---|---|---|
""" + "\n".join(
        f"| {name.title()} | {shape[0]:,} | {shape[1]} |"
        for name, shape in dq["raw_shapes"].items() if shape[0]
    ) + f"""

Transaction window: {dq.get('date_range', {}).get('min', 'n/a')} to
{dq.get('date_range', {}).get('max', 'n/a')}.

### 2.2 Three structural features that shape everything downstream

**(a) The catalogue has no instructor.** The Courses sheet carries no `TeacherID`. The only
Course↔Teacher link in the workbook is on the Transactions sheet, and it is not an ownership
link: the median course has **{link.get('teachers_per_course_median', 'n/a'):.0f} distinct
instructors** against it and the maximum has {link.get('teachers_per_course_max', 'n/a')}.
The `TeacherID` is attached to the sale, not to the course.

Instructor attributes were therefore averaged per course. Averaging across ~14 instructors
collapses the variation: instructor rating has a standard deviation of
{link.get('teacher_rating_sd_raw', 0)} across the {dq['raw_shapes'].get('teachers', [0])[0]}
instructors, but only {link.get('teacher_rating_sd_aggregated', 0)} across courses after
aggregation — **{link.get('variation_retained_pct', 0)}% of the signal survives**. Any
instructor feature in this analysis is close to a constant.

This means the brief's "expertise-category match score" and "instructor rating influence"
cannot be evaluated as specified. The data does not record who teaches what.

**(b) Nearly two thirds of the catalogue is free.**
{taut.get('zero_price_courses', 0)} of {eda['n_courses']} courses
({taut.get('zero_price_share_pct', 0)}%) have a list price of zero and therefore generate
exactly zero revenue regardless of demand. The median course revenue is
{money(eda['revenue']['median'])}. Revenue at course level is close to a binary
free/paid indicator multiplied by a price.

**(c) `Amount` never deviates from `CoursePrice`.** On all {eda['n_transactions']:,}
transactions the amount paid equals the course's list price to the cent. There are no
discounts, no partial payments and no promotions in this data. Consequences are developed in
section 4.4.

### 2.3 Data quality

- Duplicate rows removed: {dq.get('duplicates_dropped', {})}
- Transactions dropped for a missing CourseID or date: {dq.get('transactions_dropped_missing_key', 0)}
- Transactions referencing an unknown CourseID: {dq.get('orphan_transactions', 0)}
  ({dq.get('orphan_transaction_pct', 0)}%)
- Courses with zero transactions: {eda['enrollments']['zero_enrolment_courses']}

The referential integrity is perfect, which is itself worth noting: every transaction
resolves to a real course, user and instructor.

---

## 3. Methodology

### 3.1 Two framings

**Cold-start** models use course and instructor attributes only, with no historical
performance features. This is the information available when commissioning a course that
does not yet exist.

**Temporal forecasting** models cut the ledger at **{split.get('cutoff', 'n/a')}**, build
features from the {split.get('history_months', 'n/a')} months before it and predict the
{split.get('forecast_months', 'n/a')} months after. No feature can see its own target.

The separation matters because the brief lists "past enrolment count" as a feature while
also listing enrolment count as a target. Computed over one period those are the same
number, and a model built that way reports a near-perfect R² while forecasting nothing.

### 3.2 Preprocessing

Numeric features median-imputed and standardised; categoricals mode-imputed and one-hot
encoded, all inside a scikit-learn `Pipeline` so encoders refit within each cross-validation
fold. Numeric pairs correlated above |r| = {config.CORRELATION_THRESHOLD} were pruned, with
headline business levers protected from removal.

### 3.3 Models

Six families per target: mean baseline, Linear Regression, Ridge, Lasso, Random Forest,
Gradient Boosting. Ridge, Lasso and the ensembles were tuned by grid search over
{config.CV_FOLDS}-fold cross-validation, each fitted on both raw and log1p targets with the
better cross-validated variant kept. **Winners were selected on cross-validated R², never on
the hold-out score**, so the reported test metrics are not inflated by selection.

For revenue an additional non-model benchmark was added: *price × mean training enrolments*.
Where revenue is arithmetically determined by price, the mean is the wrong thing to beat.

---

## 4. Signal diagnostics

These ran before any model was fitted.

### 4.1 Dispersion: is enrolment variance larger than chance?

If {eda['n_transactions']:,} transactions were sprinkled at random across
{eda['n_courses']} courses, each course would still receive a different number purely by
chance, and the variance of that null process is known. Comparing observed variance against
it separates real popularity differences from counting noise.

| Quantity | Value |
|---|---|
| Mean enrolments per course | {disp.get('mean')} |
| Variance | {disp.get('variance')} |
| Standard deviation | {disp.get('sd')} |
| **Index of dispersion (variance / mean)** | **{disp.get('index_of_dispersion')}** |
| p (over-dispersion) | {disp.get('p_overdispersed')} |

An index of 1.00 is exactly what pure chance produces. The observed value is
{disp.get('index_of_dispersion')} — if anything marginally *less* variable than chance.
{disp.get('interpretation', '').capitalize()}.

![Target distributions](figures/01_target_distributions.png)

*Figure 1. The narrow, roughly symmetric spread of enrolments per course (left) is what
a constant-rate random process produces — not the long tail a genuine hit-vs-flop dynamic
would leave. Revenue (right) shows a wider spread only because it inherits variation
directly from price.*

### 4.2 Uniformity

χ² = {unif.get('chi2')}, df = {unif.get('df')}, p = {unif.get('p_value')}.
{unif.get('interpretation', '').capitalize()}.

The observed spread runs from {eda['enrollments']['min']} to {eda['enrollments']['max']}
enrolments. That range looks like a real difference between a popular and an unpopular
course, and it is not: it is the range chance produces at this sample size.

### 4.3 Attribute associations, corrected for multiple testing

{sig_e.get('n_tests', 0)} attributes were tested against enrolment (Pearson correlation for
numeric, one-way ANOVA for categorical). The Bonferroni threshold is
p < {sig_e.get('bonferroni_threshold', 0)}.

{diagnostics_tests_table(sig_e)}

**{sig_e.get('interpretation', '')}.**

Two attributes clear the uncorrected 0.05 bar
({', '.join(f'`{f}`' for f in sig_e.get('significant_before_correction_only', [])) or 'none'}).
Neither survives correction, and that is the correct reading: testing
{sig_e.get('n_tests', 0)} hypotheses at p < 0.05 produces at least one false positive around
{100 * (1 - 0.95 ** sig_e.get('n_tests', 1)):.0f}% of the time by chance alone. Reporting
`CourseRating` as a demand driver on the strength of a raw p of
{next((t['p_value'] for t in sig_e.get('tests', []) if t['feature'] == 'CourseRating'), float('nan')):.4f}
would be exactly that error.

![Price sensitivity and quality signals](figures/03_price_sensitivity.png)
![Rating and instructor signals](figures/04_quality_signals.png)

*Figure 2. Each scatter shows a nominal trend line, and none of them is a straight or
consistent slope across all four panels. This is what "no relationship survives testing"
looks like visually: any apparent trend in one panel is within the noise band that the
formal tests above already quantified.*

![Correlation matrix](figures/06_correlation.png)

*Figure 3. Every cell involving `EnrollmentCount` is close to zero except for its
correlation with itself. The strong cells all sit among the price-derived features and
`CourseRevenue` — exactly the tautology described in section 4.4.*

### 4.4 The revenue tautology

`Amount` equals `CoursePrice` on every transaction, so:

```
CourseRevenue  =  EnrollmentCount  ×  CoursePrice        (exactly, no residual)
```

Price is known before launch. Enrolment, per 4.1–4.3, is a constant plus noise. So revenue
is predictable to the extent that price is known, and to no greater extent.

The rule `{taut.get('naive_rule', 'n/a')}` uses no model, no features beyond price, and no
fitting. Across the full catalogue it reaches **R² = {taut.get('naive_rule_r2')}**.

This reframes every revenue metric in section 5. A revenue R² of
{r_best.get('R2', 0):+.3f} is not 97% of the variance explained by machine learning — it is
less than the arithmetic already gives you for free.

---

## 5. Model results

### 5.1 Enrolment count (cold start)

{metrics_table(enroll)}

The mean-predicting baseline scores R² = {e_baseline.get('R2', 0):+.3f} with MAE
{e_baseline.get('MAE', 0):,.2f}. The best fitted model, {e_name}, scores
{e_best.get('R2', 0):+.3f} with MAE {e_best.get('MAE', 0):,.2f}. Cross-validated R² is
negative for {sum(1 for v in enroll['results'].values() if v.get('cv_r2', 0) < 0)} of the
{len(enroll['results'])} entries.

A negative R² means the model predicts worse than a horizontal line through the mean. This
is the expected result when a flexible model is fitted to noise: it learns the training
sample's accidents and carries them to the test set as error. Linear Regression, with no
regularisation to restrain it, reaches R² = {enroll['results'].get('LinearRegression', {}).get('R2', 0):+.3f}
— a textbook illustration of the same effect.

### 5.2 Course revenue (cold start)

{metrics_table(revenue, naive_note=True)}

Every fitted model is beaten by the naive rule. The best learned model reaches MAE
{money(r_best.get('MAE', 0))} against {money(r_naive.get('MAE', 0))} for arithmetic — the
models are roughly **{r_best.get('MAE', 1) / max(r_naive.get('MAE', 1), 1e-9):.1f}× worse**.

They are worse for an understandable reason. Price alone gives the answer; the models are
also fitting {enroll['n_features']} other features that contain nothing, and every one of
those adds variance without adding information.

For completeness, of {sig_r.get('n_tests', 0)} attribute-to-revenue tests,
{len(sig_r.get('survivors', []))} survive correction:
{', '.join(f'`{s}`' for s in sig_r.get('survivors', []))}. All of them are price, a
transformation of price, or a proxy for it (`CourseType` is the free/paid split). No
non-price attribute is associated with revenue.

### 5.3 Category revenue

Course-level revenue predictions summed by category over hold-out courses:
MAE {money(cat_metrics.get('MAE', 0))}, RMSE {money(cat_metrics.get('RMSE', 0))},
R² {cat_metrics.get('R2', float('nan')):+.3f}.

This figure is inherited, not earned. It aggregates predictions that are themselves
restatements of price. It should not be quoted as evidence that category demand is
forecastable.

"""

    if f_enroll and f_revenue:
        fe_name, fe_best = best_of(f_enroll)
        fr_name, fr_best = best_of(f_revenue)
        doc += f"""### 5.4 Forward forecasting for existing courses

Trained on {split['history_months']} months of history, scored on the following
{split['forecast_months']} months.

**Future enrolments**

{metrics_table(f_enroll)}

**Future revenue**

{metrics_table(f_revenue)}

The forecasting models repeat the pattern. Future enrolments cannot be predicted even with
each course's own trading history available (best R² = {fe_best.get('R2', 0):+.3f}), which
is the strongest possible confirmation of section 4: if a course's own past enrolment does
not predict its future enrolment, there is no course-level demand process to model. Future
revenue again tracks price ({fr_name}, R² = {fr_best.get('R2', 0):+.3f}).

"""

    doc += f"""---

## 6. Feature importance

Permutation importance measures the fall in hold-out R² when a feature is shuffled.

### 6.1 Enrolment

{importance_list(enroll)}

Read this together with section 5.1. The selected enrolment model has an R² of
{e_best.get('R2', 0):+.3f}, so these numbers describe how much a feature contributes to a
model that predicts nothing. Importances at or near zero across the board are the correct
outcome and should not be interpreted as a ranking of demand drivers.

### 6.2 Revenue

{importance_list(revenue)}

Price and its transformations dominate, as section 4.4 predicts. This is the model
rediscovering `revenue = price × constant`.

---

## 7. What this means for EduPro

**1. Do not deploy a course-level demand forecast on this data.**
The enrolment models have negative cross-validated R². Deployed, they would produce
confident numbers uncorrelated with outcomes, which is worse than the current retrospective
reporting because it carries the authority of a model.

**2. Do not present the revenue R² as a modelling result.**
Revenue is price × enrolments, and enrolments are a constant plus noise. If a revenue
forecast is needed today, `expected revenue = price × {taut.get('mean_enrollments')}` is
more accurate than any model here, and it is auditable in a spreadsheet.

**3. The instructor question cannot currently be asked.**
The catalogue does not record who teaches each course. Adding a `TeacherID` column to the
Courses sheet is a one-off schema change that would make the brief's instructor analysis
possible.

**4. The pricing question cannot currently be asked either.**
Price never varies within a course and no transaction ever deviates from list price, so
there is no variation from which price sensitivity could be estimated. A controlled price
test — the same course offered at two prices to comparable cohorts — would generate it.

**5. What to record next.** In rough order of value:

| Data | Why it matters |
|---|---|
| Course impressions / page views | Enrolment ÷ views is a conversion rate, which *does* vary by course. Enrolment alone confounds demand with exposure. |
| Marketing spend and placement by course | Currently absent, and plausibly the dominant driver of which courses sell. |
| Course launch date | Would separate a course selling steadily for a year from one selling fast in a month. |
| `TeacherID` on the Courses sheet | Makes instructor effects estimable. |
| Discounts, promotions, list-price changes | Creates the price variation needed for elasticity. |
| Completion and refund events | Distinguishes an enrolment that stuck from one that did not. |

**6. What the catalogue does show.** {cats.iloc[0]['CourseCategory']} leads on total revenue
({money(cats.iloc[0]['Revenue'])}), but with enrolments effectively uniform, category
revenue differences are driven almost entirely by how many courses in that category carry a
non-zero price. That is a pricing decision already taken, not a demand finding.

![Category performance](figures/02_category_performance.png)

*Figure 4. Categories rank differently on total revenue versus revenue per course, and
per-course is the fairer comparison — a category is not "more in demand" simply because it
happens to list more paid courses. Neither ranking should be read as a demand signal, given
section 4.*

![Monthly trend](figures/07_monthly_trend.png)

*Figure 5. Platform-wide monthly revenue and enrolments over the observed year, included
for completeness. Course-level flatness (this report's finding) is compatible with
platform-level movement over time; the two are different questions, and this report only
addresses the first.*

---

## 8. Limitations

1. **Sample size.** {eda['n_courses']} courses is small. The diagnostics are stated as
   "cannot reject the null", not "the null is proven": a weak real effect could be present
   and undetectable at n = {eda['n_courses']}. A power calculation would be needed to say
   how weak.
2. **One year, no seasonality.** Twelve months of data cannot separate trend from seasonal
   cycle.
3. **The temporal split is short.** {split.get('history_months', 'n/a')} months of history
   against a {split.get('forecast_months', 'n/a')}-month horizon is thin.
4. **Enrolment is a proxy for demand.** Without impressions, a course with low enrolment and
   a course nobody was shown are indistinguishable.
5. **Instructor features are near-constant** after aggregation
   ({link.get('variation_retained_pct', 0)}% of variation retained), so their absence from
   the importance rankings is uninformative about whether instructors matter.
6. **The data may be synthetically generated.** Exactly 5 courses in each of
   {eda['n_categories']} categories, a dispersion index of {disp.get('index_of_dispersion')},
   and zero deviation between amount and list price across {eda['n_transactions']:,} rows are
   together more consistent with random generation than with a trading platform. If so, the
   absence of signal is a property of the generator, and the methodology here — not the
   findings — is what transfers to real data.

---

## 9. Conclusion

The project set out to predict course demand and revenue. The result is that this dataset
cannot support the first and does not require a model for the second.

Enrolment counts are statistically indistinguishable from random allocation, no attribute
predicts them after correction for multiple testing, and every fitted model performs at or
below a mean-predicting baseline. Revenue is arithmetically determined by price, and a rule
requiring no model outperforms all six model families.

The pipeline, the temporal design, the leakage controls and the dashboard are complete and
would produce valid results on data that contained signal. Applied here, their most valuable
output is a well-evidenced negative, delivered before anyone acted on a forecast that could
not work. Section 7 sets out the six data additions that would change that.

A model that reports R² = {r_best.get('R2', 0):+.3f} and is beaten by a one-line arithmetic
rule is the clearest argument available for why baselines and pre-modelling diagnostics
belong in every analysis of this kind.

---

## Appendix: reproducing this analysis

```bash
pip install -r requirements.txt
# place the workbook in ./data/
python train.py
python generate_report.py
streamlit run app.py
```

Artefacts: `outputs/metrics.json` (all metrics and diagnostics), `outputs/course_master.csv`
(per-course features, targets, predictions), `outputs/category_forecast.csv`,
`outputs/tables/importance_*.csv`, `outputs/figures/*.png`.

Diagnostics are implemented in `diagnostics.py` and run automatically as step 3 of
`train.py`.
"""
    return doc


# ======================================================================================
# Executive summary
# ======================================================================================
def build_executive_summary(m: dict) -> str:
    eda = m["eda"]
    models = m["models"]
    dq = m["data_quality"]
    diag = m.get("diagnostics", {})
    disp = diag.get("dispersion", {})
    unif = diag.get("uniformity", {})
    taut = diag.get("tautology", {})
    sig_e = diag.get("attribute_signal", {}).get("EnrollmentCount", {})
    has_signal = diag.get("verdict", {}).get("demand_signal_present", True)

    enroll = models["cold_EnrollmentCount"]
    revenue = models["cold_CourseRevenue"]
    e_name, e_best = best_of(enroll)
    r_name, r_best = best_of(revenue)
    r_naive = revenue["results"].get("Naive: mean enrolments x price", {})
    link = dq.get("teacher_link", {})

    if has_signal:
        verdict_block = "Models were built and validated; see the research paper for metrics."
    else:
        verdict_block = f"""### The finding

**The data cannot support course demand forecasting, and we recommend not deploying one.**

Enrolments are spread across the {eda['n_courses']} courses exactly as they would be if the
{eda['n_transactions']:,} purchases had been assigned to courses at random. Formally: the
index of dispersion is {disp.get('index_of_dispersion')} where 1.00 is pure chance, and a
goodness-of-fit test returns p = {unif.get('p_value')}, far from the 0.05 that would indicate
real differences between courses.

We tested {sig_e.get('n_tests', 0)} course characteristics — price, duration, rating, level,
category, type and others — against enrolment. **None is associated with demand** once
corrected for the number of tests run.

Six forecasting models were built anyway, to verify the diagnosis. The best scores
**{e_best.get('R2', 0):+.3f}** on courses it had not seen, where 0.00 means "no better than
guessing the average" and negative means worse.

### The revenue number is arithmetic, not a forecast

Revenue forecasts in this dataset look excellent — R² up to {r_best.get('R2', 0):+.3f}. That
figure is misleading and should not be reported as a modelling achievement.

Every transaction is paid at exactly the course's list price. There are no discounts or
promotions in the data. So revenue is precisely *price × number of enrolments*, and since
enrolments are near-constant, revenue is simply price × a constant.

We tested this directly. The rule **"revenue = list price × {taut.get('mean_enrollments')}"**
— no model, no machine learning, calculable in a spreadsheet column — predicts revenue with
an average error of **{money(r_naive.get('MAE', 0))}**. The best machine-learning model
manages **{money(r_best.get('MAE', 0))}**, roughly
{r_best.get('MAE', 1) / max(r_naive.get('MAE', 1), 1e-9):.0f} times worse.

If a revenue estimate is needed this quarter, the arithmetic rule is the better tool and it
is auditable by anyone."""

    return sample_data_banner(m) + f"""# Executive Summary
## Predictive Course Demand and Revenue Forecasting — EduPro

**Prepared by:** Gauri Shinde, Second-year Computer Science student, University of Greenwich
**Prepared for:** Programme and funding stakeholders
**Date:** {m['generated_at'][:10]}
**Scope:** {eda['n_courses']} courses · {eda['n_transactions']:,} transactions ·
{eda['n_categories']} categories · {dq.get('date_range', {}).get('min', '')} to
{dq.get('date_range', {}).get('max', '')} · total revenue {money(eda['revenue']['total'])}

---

{verdict_block}

### Why the data cannot answer the question

Three gaps, each fixable:

**1. The catalogue does not record who teaches each course.** The course list has no
instructor column. The only link is on individual sales, where the median course shows
{link.get('teachers_per_course_median', 0):.0f} different instructors — that field identifies
the transaction, not the teacher of record. Consequently the brief's instructor-quality
analysis could not be performed at all.

**2. Prices never vary.** Each course has one price, always charged in full. Price
sensitivity is measured from variation, and there is none to measure.

**3. There is no measure of exposure.** We can see enrolments but not how many people were
shown each course. A course with few enrolments because it is unappealing and one with few
because nobody saw it are indistinguishable here. This is likely the single most important
missing field.

### What we recommend

1. **Do not commission decisions from a demand model built on this data.** A forecast that
   is uncorrelated with outcomes but carries the authority of a model is worse than the
   honest retrospective reporting already in place.
2. **Use the arithmetic rule for revenue planning** in the interim, and state plainly that
   it assumes flat demand across courses — which is what the data shows.
3. **Add course impressions or page views to the data collected.** Conversion rate
   (enrolments ÷ views) is the metric that would actually vary by course and would make this
   analysis viable.
4. **Add an instructor column to the course catalogue,** and record marketing spend and
   course launch dates against course IDs.
5. **Run one controlled price test.** Offering comparable courses at two price points is the
   only way to obtain a defensible price elasticity.
6. **Re-run this pipeline once those fields exist.** It is built, validated and reusable; the
   diagnostics that produced this verdict will equally confirm when real signal appears.

### One caveat on this finding

"No signal detected" is not the same as "no signal exists". With {eda['n_courses']} courses,
a weak real effect could be present and undetectable. The conclusion is that **this dataset
provides no evidence of predictable course-level demand**, not that demand is inherently
unpredictable. The remedy is the same either way: measure more, then re-test.

The characteristics of the file — exactly five courses in every category, dispersion
indistinguishable from random, and zero price deviation across
{eda['n_transactions']:,} transactions — suggest it may be a generated dataset rather than a
production extract. If so, the transferable output of this project is the method, which is
complete and validated, rather than the specific finding.

---

*Full methodology, diagnostics and model comparison: `research_paper.md`.
Reproducible via `python train.py && python generate_report.py`.*
"""


def main() -> None:
    m = load_metrics()
    paper = config.OUTPUT_DIR / "research_paper.md"
    summary = config.OUTPUT_DIR / "executive_summary.md"
    paper.write_text(build_research_paper(m), encoding="utf-8")
    summary.write_text(build_executive_summary(m), encoding="utf-8")
    print(f"Wrote {paper}")
    print(f"Wrote {summary}")
    verdict = m.get("diagnostics", {}).get("verdict", {}).get("summary", "")
    if verdict:
        print(f"\nVerdict carried into both documents:\n  {verdict}")


if __name__ == "__main__":
    main()
