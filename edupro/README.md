# Predictive Modeling for Course Demand and Revenue Forecasting — EduPro

**Author:** Gauri Shinde, Second-year Computer Science, University of Greenwich

Predicts course-level enrolment demand and revenue for an online learning platform, and
aggregates revenue forecasts to category level to support commissioning, pricing and
instructor-onboarding decisions.

---

## Live dashboard

Deployed on Streamlit Community Cloud from this repository (`app.py`, branch `main`).
When deploying, set Python to 3.12 under Advanced settings so the pinned packages in
`requirements.txt` install cleanly. The app reads the trained models and results already
saved in `outputs/`, so it does not retrain on startup.

## Quick start

```bash
# Python 3.12 recommended
pip install -r requirements.txt

# 1. Put the workbook in place
#    data/EduPro Online Platform.xlsx

# 2. Run the pipeline (~4 min on 180 courses; scales with catalogue size)
python train.py

# 3. Generate the written deliverables
python generate_report.py

# 3b. (Optional) Render them as PDF
python make_pdfs.py

# 4. Launch the dashboard
streamlit run app.py
```

`train.py` accepts an explicit path if your file is named differently:

```bash
python train.py "data/whatever the file is called.xlsx"
```

### No workbook yet?

A synthetic workbook with identical structure is included so you can verify the pipeline
runs before the real data arrives:

```bash
python make_sample_data.py
python train.py "data/SAMPLE_EduPro Online Platform.xlsx"
python generate_report.py
```

Reports generated from sample data carry a **SYNTHETIC DATA** banner at the top so they
can't be handed in by mistake. Delete `data/SAMPLE_*.xlsx` once the real file is in place.

---

## Result on the supplied EduPro dataset

Running this pipeline on `EduPro Online Platform.xlsx` produced a **negative result**, and
the deliverables report it as such:

- Enrolments across the 60 courses are statistically indistinguishable from random
  allocation (index of dispersion 0.941 against 1.00 for pure chance; chi-square p = 0.605).
- No course attribute is associated with enrolment after Bonferroni correction over 17 tests.
- The best enrolment model scores R² = +0.009 on hold-out courses; most score negative.
- Revenue looks highly predictable (R² up to 0.971) but `Amount` equals `CoursePrice` on all
  10,000 transactions, so revenue is exactly price x enrolments. The no-model rule
  `revenue = price x 166.7` scores R² = 0.989 and beats every fitted model.

The pipeline is correct; the dataset cannot answer the question. `diagnostics.py` runs these
checks as step 3 of `train.py`, before any model is fitted, and both written deliverables
branch on the verdict. The dashboard shows a red banner on every page when the verdict is
"no signal", so the interface cannot be mistaken for a working forecaster.

See `outputs/research_paper.md` section 4 and section 7 (what EduPro would need to record for
the original question to become answerable).

---

## What the models do, and why there are two of them

The brief asks for enrolment count and revenue as prediction targets, and also asks for
"past enrolment count" and "past average revenue" as features. Computed over the same
period, those are the same number: a model given past enrolments to predict total
enrolments scores a near-perfect R² and forecasts nothing. This project separates the two
decisions that were being conflated.

### Cold-start model — *should we launch this course?*

Features: course and instructor attributes only (price, duration, level, category, type,
rating, instructor experience/rating/expertise match). **No historical performance
features.** This is the information actually available when commissioning a course that
does not exist yet, so it is the only honest model for that decision. It powers the
dashboard's what-if sliders.

### Temporal forecast model — *what will this course do next quarter?*

The transaction ledger is cut at a date. Features aggregate transactions strictly *before*
the cutoff; targets aggregate transactions strictly *after* it. No feature can see its own
target. Default forecast window is 3 months (`config.FORECAST_WINDOW_MONTHS`).

This model outperforms the cold-start model, which is expected — a course's own recent
trading history predicts its next quarter better than its attributes do. The gap between
the two is a direct measure of the uncertainty a genuinely new course carries.

---

## Methodology notes

**Model selection is on cross-validated R², never on the hold-out score.** Picking the
model that scores best on the test set turns that set into a selection set and inflates the
number you then report. The hold-out metrics are therefore a fair out-of-sample estimate
for a model chosen without reference to them.

**A mean-predicting baseline is always fitted.** Without it, an R² of 0.4 is hard to
interpret; against it, the improvement is explicit. A model that can't beat it has found
nothing.

**Encoders are fitted inside each CV fold** via scikit-learn `Pipeline`, not once on the
full dataset. Fitting them outside the loop is a common and quiet source of optimistic
results.

**Both targets are right-skewed**, so each model family is fitted twice — raw target and
`log1p(target)` — and the better cross-validated variant is kept. Which one won is recorded
in `metrics.json`.

**Permutation importance is reported, not tree impurity importance.** Impurity importance
inflates continuous and high-cardinality features whether or not they predict anything;
permutation importance measures the actual fall in hold-out R² when a column is shuffled.

**Headline business levers are protected from collinearity pruning.** Price, rating,
duration and experience are never the feature discarded when a correlated pair is pruned —
derived and interaction terms go first — so the importance analysis stays interpretable.

**Courses with zero transactions are kept**, with an enrolment count of zero. Dropping them
would train only on courses that succeeded and bias every prediction upward.

**Missing ratings are imputed with the category median and flagged** with a `RatingMissing`
indicator. An absent rating usually means the course is new or barely enrolled, which is
informative about demand; discarding the flag throws that signal away.

---

## Interpreting the output

Category-level forecasts are substantially more accurate than single-course forecasts,
because per-course errors are largely independent and partially cancel when summed.

- **Budget** on category revenue forecasts.
- **Prioritise** a shortlist using course-level predictions.
- **Do not** underwrite a revenue commitment for one course on a course-level prediction.

The models describe association, not causation. Prices in this data were not assigned at
random, so the price–demand relationship shows what similar courses did, not what a given
course would do if repriced. The dashboard's price-sensitivity curve carries this caveat
inline. A controlled price test on a small set of comparable courses is the single
highest-value data investment available.

---

## File map

| File | Purpose |
|---|---|
| `config.py` | Paths, column aliases, bin edges, hyperparameter grids. Change settings here, not in the analysis code. |
| `data_prep.py` | Sheet/column resolution, cleaning, merging, transaction aggregation, temporal split. |
| `features.py` | Price bands, duration buckets, rating tiers, level encoding, experience buckets, expertise-match score, collinearity pruning. |
| `models.py` | Preprocessing pipelines, model zoo, metrics, permutation importance. |
| `diagnostics.py` | Dispersion, uniformity, corrected attribute-association and revenue-tautology checks. Runs before any model is fitted. |
| `eda.py` | Seven EDA figures plus the summary statistics the report reads. |
| `train.py` | Orchestrates everything. **This is the entry point.** |
| `generate_report.py` | Writes the research paper and executive summary from `metrics.json` — every number is computed, none hand-typed. |
| `app.py` | Streamlit dashboard (6 modules). |
| `make_sample_data.py` | Synthetic workbook generator for smoke-testing. |
| `make_pdfs.py` | Renders the two Markdown deliverables as styled PDFs (requires `pandoc`/`wkhtmltopdf`, both preinstalled). |
| `test_app.py` | Runs every dashboard page headlessly and fails on any exception. |

### Artefacts written to `outputs/`

| Path | Contents |
|---|---|
| `research_paper.md` / `.pdf` | Full write-up: EDA, methodology, model comparison, insights, limitations. |
| `executive_summary.md` / `.pdf` | Two-page non-technical summary for stakeholders. |
| `metrics.json` | Every model's CV and hold-out MAE/RMSE/R², data-quality log, split definition. |
| `course_master.csv` | One row per course: features, targets, predictions, train/test flag. |
| `temporal_frame.csv` | History-window features and forecast-window targets and predictions. |
| `category_forecast.csv` | Category actual vs predicted revenue, with error percentages. |
| `tables/importance_*.csv` | Permutation and native importances per model. |
| `figures/*.png` | EDA charts. |
| `models/*.joblib` | Fitted pipelines, loaded by the dashboard. |

---

## Dashboard modules

1. **Overview** — KPIs, monthly revenue trend, revenue concentration curve, top courses.
2. **Course demand predictor** — set price, duration, level, category, type, instructor
   experience/rating/expertise; see predicted enrolments and revenue plus a
   price-sensitivity curve with the revenue-maximising price marked.
3. **Revenue forecast** — category actual vs predicted, course-level scatter against the
   perfect-prediction diagonal, next-period forecast table for existing courses.
4. **Feature importance** — permutation importance with error bars, model coefficients,
   marginal correlations.
5. **Category comparison** — rank by total or per-course revenue/enrolments; catalogue
   depth against productivity to spot under-served categories.
6. **Model performance** — full model comparison tables and residual diagnostics.

---

## Troubleshooting

**`FileNotFoundError` on startup** — the workbook isn't in `data/`. Check the filename
matches `config.WORKBOOK_NAME`, or pass the path as an argument to `train.py`.

**`RuntimeWarning: resolved by fuzzy match`** — a column name didn't match exactly and was
resolved by substring. Check the mapping printed at step 1 of `train.py` and add the exact
spelling to `config.COLUMN_ALIASES` if it guessed wrong.

**`No CourseID/TeacherID link between the Courses and Teachers sheets`** — instructor
features will be median-filled and carry no signal. The link must exist on one sheet or the
other.

**Forecasting models skipped** — there is less than `config.MIN_HISTORY_MONTHS` of history
before the cutoff. Lower `FORECAST_WINDOW_MONTHS` or `MIN_HISTORY_MONTHS` in `config.py`.

**Dashboard says no artefacts found** — run `python train.py` first.
