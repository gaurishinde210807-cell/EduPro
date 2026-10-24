# Executive Summary
## Predictive Course Demand and Revenue Forecasting — EduPro

**Prepared by:** Gauri Shinde, Second-year Computer Science student, University of Greenwich
**Prepared for:** Programme and funding stakeholders
**Date:** 2026-10-24
**Scope:** 60 courses · 10,000 transactions ·
12 categories · 2025-01-01 to
2025-12-30 · total revenue 911,323

---

### The finding

**The data cannot support course demand forecasting, and we recommend not deploying one.**

Enrolments are spread across the 60 courses exactly as they would be if the
10,000 purchases had been assigned to courses at random. Formally: the
index of dispersion is 0.941 where 1.00 is pure chance, and a
goodness-of-fit test returns p = 0.6046, far from the 0.05 that would indicate
real differences between courses.

We tested 17 course characteristics — price, duration, rating, level,
category, type and others — against enrolment. **None is associated with demand** once
corrected for the number of tests run.

Six forecasting models were built anyway, to verify the diagnosis. The best scores
**+0.009** on courses it had not seen, where 0.00 means "no better than
guessing the average" and negative means worse.

### The revenue number is arithmetic, not a forecast

Revenue forecasts in this dataset look excellent — R² up to +0.970. That
figure is misleading and should not be reported as a modelling achievement.

Every transaction is paid at exactly the course's list price. There are no discounts or
promotions in the data. So revenue is precisely *price × number of enrolments*, and since
enrolments are near-constant, revenue is simply price × a constant.

We tested this directly. The rule **"revenue = list price × 166.7"**
— no model, no machine learning, calculable in a spreadsheet column — predicts revenue with
an average error of **1,724**. The best machine-learning model
manages **3,017**, roughly
2 times worse.

If a revenue estimate is needed this quarter, the arithmetic rule is the better tool and it
is auditable by anyone.

### Why the data cannot answer the question

Three gaps, each fixable:

**1. The catalogue does not record who teaches each course.** The course list has no
instructor column. The only link is on individual sales, where the median course shows
14 different instructors — that field identifies
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

"No signal detected" is not the same as "no signal exists". With 60 courses,
a weak real effect could be present and undetectable. The conclusion is that **this dataset
provides no evidence of predictable course-level demand**, not that demand is inherently
unpredictable. The remedy is the same either way: measure more, then re-test.

The characteristics of the file — exactly five courses in every category, dispersion
indistinguishable from random, and zero price deviation across
10,000 transactions — suggest it may be a generated dataset rather than a
production extract. If so, the transferable output of this project is the method, which is
complete and validated, rather than the specific finding.

---

*Full methodology, diagnostics and model comparison: `research_paper.md`.
Reproducible via `python train.py && python generate_report.py`.*
