# Data dictionary

Every file that carries data, and what its columns mean. Paths are for `v3/`;
`V1/` and `v2/` hold earlier versions of the same shapes.

---

## The 13 categories

Defined once in `v3/base-model_tf_idf/pipeline/config.py` as `CATEGORIES`. **The order below is the
column order** in the gold standard and in every prediction matrix.

| # | Column | Meaning |
|---|---|---|
| 1 | `programming` | programming or scripting languages |
| 2 | `sql` | database querying with SQL |
| 3 | `visualisation_bi` | data visualisation and business intelligence tools |
| 4 | `reporting` | producing reports and management information |
| 5 | `excel` | spreadsheet software such as Excel |
| 6 | `statistics` | statistical analysis and forecasting |
| 7 | `machine_learning` | machine learning and predictive modelling |
| 8 | `data_cleaning` | data cleaning and data quality |
| 9 | `etl` | data engineering, ETL pipelines and data warehousing |
| 10 | `data_modelling` | data modelling and schema design |
| 11 | `cloud` | cloud computing platforms |
| 12 | `stakeholder_comm` | stakeholder communication and presenting findings |
| 13 | `ethics_governance` | data governance, privacy and GDPR compliance |

Values are always `1` (skill required/mentioned) or `0` (not). A posting may carry
any number of categories, including none — this is a multi-label problem, not
multi-class. Boundary rules and edge cases are in
`manual_work/annotation_guidelines_v2_0.docx`.

---

## `manual_work/uk_analyst_corpus_v4_clean.csv`

The working corpus. One row per job posting.

| Column | Type | Notes |
|---|---|---|
| `posting_id` | str | Primary key. Joins to every other file here. |
| `job_link` | str | Original posting URL. Also the join key back to the raw source data. |
| `job_title` | str | As advertised. |
| `company` | str | Hiring organisation (sometimes an agency, not the end employer). |
| `job_location` | str | Free text, not normalised. |
| `first_seen` | date | When the posting first appeared in the source dataset. |
| `search_country` | str | Always United Kingdom after filtering. |
| `job_level` | str | Source dataset's own seniority guess. Noisy — the gold standard's `seniority` is the trustworthy one. |
| `job_summary` | str | **The posting text. This is the only field any extraction method reads.** |
| `role_family_provisional` | str | Rule-based role grouping from the title. "Provisional" because it is unreviewed; the gold standard's `role_family` is the human-checked version. |

`manual_work/gold_standard_sample_300_v3.csv` and
`manual_work/pilot_sample_20_v2.csv` have identical columns — they are the 300-row
annotation sample and the 20-row pilot sample drawn from this corpus.

---

## `manual_work/gold_standard_annotation_workbook_v2.xlsx`

The manual annotations. **This is the ground truth** — read-only, produced by hand.

### Sheet `Annotation` — 300 rows, 21 columns

| Column | Type | Notes |
|---|---|---|
| `posting_id` | str | Joins to the corpus. |
| `job_title` | str | Carried over for the annotator's context. |
| `role_family_provisional` | str | The rule-based guess shown to the annotator. |
| `programming` … `ethics_governance` | 0/1 | **The 13 label columns, in taxonomy order.** |
| `other` | 0/1 | A relevant skill outside the 13 categories was present. |
| `seniority` | str | Human-assigned seniority. |
| `role_family` | str | Human-confirmed role family. **Used as the stratification key for the dev/test split** — not the `_provisional` column. |
| `other_skills_verbatim` | str | Free text: skills noted but unclassifiable, kept for taxonomy revision. |
| `notes` | str | Annotator's reasoning on hard cases. |

`load_gold()` reads this sheet and keeps only rows where all 13 label columns are
non-null.

### Other sheets

- `Posting_Texts` — the posting text alongside each row, so annotation happens
  without leaving the workbook.
- `Quick_Reference` — condensed category rules for the annotator.

`manual_work/pilot_annotation_completed.xlsx` is the same structure for the 20-row
pilot round, which was used to refine the guidelines before the main annotation.

---

## `base-model_tf_idf/results/tfidf_predictions_corpus.csv`

Predicted labels for the **whole corpus** from the better-performing variant.

| Column | Type | Notes |
|---|---|---|
| `posting_id` | str | Joins to the corpus. |
| `programming` … `ethics_governance` | 0/1 | Predictions, in taxonomy order. |

The pipeline also writes a per-variant `tfidf_predictions_gold_*.csv`, which adds a
`split` column (`dev` / `test`); those are committed alongside this file.

---

## `base-model_tf_idf/results/tfidf_A_cosine_test.csv`, `tfidf_B_weighted_hit_test.csv`

Scored results on the test split. One row per category, then aggregate rows.

| Column | Notes |
|---|---|
| `category` | One of the 13, or `MACRO AVG`, `MICRO AVG`, `SUBSET ACCURACY`, `HAMMING ACCURACY`. |
| `precision`, `recall`, `f1` | Standard. Empty for the two accuracy rows. |
| `support` | True positives available — how many test postings actually carry this label. |
| `predicted` | How many the method predicted. Compare against `support` to see over- or under-firing at a glance. |

For the accuracy rows the value sits in the `f1` column: `SUBSET ACCURACY` is exact
match (all 13 labels correct for a posting), `HAMMING ACCURACY` is per-label
correctness across all 2,600 posting×category cells.

---

## Splits

`make_split()` in `evaluate.py`: seeded at `RANDOM_SEED = 42`, stratified on
`role_family`, one third dev / two thirds test — 100 dev, 200 test. Thresholds are
tuned on dev only and applied unchanged to test. Changing the seed or the
stratification key makes new results incomparable with everything already committed.

---

## Not in this repository

`filter_corpus.py` builds the corpus from `linkedin_job_postings.csv` and
`job_summary.csv`. Those raw source files are **not** included, so that script
cannot be re-run from a clean clone — the cleaned corpus is provided instead.
