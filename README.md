# Skill Extraction from UK Analyst Job Postings

MSc project. Building and evaluating automated methods for extracting a 13-category
skill taxonomy from UK data/business analyst job adverts, measured against a manually
annotated gold standard.

This repository holds three iterations of the work. **`v3/` is current** — `V1/` and
`v2/` are kept for provenance and to show how the annotation scheme and corpus
evolved.

---

## The task

Each job posting is labelled with any number of 13 skill categories (a multi-label
problem). The taxonomy is derived from ESCO and O\*NET, then corpus-validated:

| | | |
|---|---|---|
| `programming` | `sql` | `visualisation_bi` |
| `reporting` | `excel` | `statistics` |
| `machine_learning` | `data_cleaning` | `etl` |
| `data_modelling` | `cloud` | `stakeholder_comm` |
| `ethics_governance` | | |

Category definitions and boundary rules live in
`v3/manual_work/annotation_guidelines_v2_0.docx`; the label space itself is defined
once in `v3/base-model_tf_idf/pipeline/config.py` so that every method and the gold
standard share identical columns.

---

## Where to look

| | |
|---|---|
| [`docs/DATA_DICTIONARY.md`](docs/DATA_DICTIONARY.md) | Every column in every data file, and what the split is |
| [`AGENTS.md`](AGENTS.md) | Working context and ground rules — read this before changing anything |
| `v3/manual_work/annotation_guidelines_v2_0.docx` | Category definitions and boundary rules |
| `v3/base-model_tf_idf/pipeline/config.py` | The taxonomy and lexicons, defined once for all methods |

---

## Repository layout

### `v3/` — current

**`v3/manual_work/`** — corpus construction and manual annotation

| File | What it is |
|---|---|
| `filter_corpus.py` | Builds the corpus: filters postings to UK analyst-adjacent titles, joins in the job summaries |
| `uk_analyst_corpus_v4_clean.csv` | The cleaned working corpus |
| `gold_standard_sample_300_v3.csv` | 300-posting sample drawn for manual annotation |
| `gold_standard_annotation_workbook_v2.xlsx` | The completed gold standard annotations |
| `pilot_sample_20_v2.csv` / `pilot_annotation_completed.xlsx` | 20-posting pilot round, used to refine the scheme before the main annotation |
| `annotation_guidelines_v2_0.docx` | Annotation guidelines, incl. homonym and boundary rules |
| `annotation_scheme_v3.xlsx` | The coding scheme |
| `postings_reader.html` | Standalone browser viewer for reading postings during annotation |

**`v3/base-model_tf_idf/`** — Method 1 of 4: TF-IDF baseline

| File | What it is |
|---|---|
| `pipeline/config.py` | Shared taxonomy, Tier 2 lexicons, negative patterns |
| `pipeline/tfidf_baseline.py` | The baseline, in two variants (below) |
| `pipeline/evaluate.py` | Splitting, per-category threshold tuning, scoring |
| `results/tfidf_A_cosine_test.csv` | Test-split results, variant A |
| `results/tfidf_B_weighted_hit_test.csv` | Test-split results, variant B |
| `results/tfidf_predictions_corpus.csv` | Predictions across the full corpus from the better variant |

### `v2/`, `V1/` — earlier iterations

Same shape, earlier corpus and annotation-scheme versions. Superseded.

---

## The baseline

Two variants, both scored per category with thresholds tuned on the dev split only and
applied unchanged to test, so test scores are not fitted to the evaluation data.

**A — cosine.** Postings and categories are both represented as TF-IDF vectors, each
category as a pseudo-document built from its lexicon, and assigned by cosine
similarity. This transfers the mapping approach of Attwood & Williams (2023) — who
mapped listings to CyBOK knowledge areas — onto this taxonomy.

**B — weighted hit.** Sum of TF-IDF weights of a category's lexicon terms present in
the posting, normalised. Stricter, and additionally applies negative-pattern
suppression for known homonyms (e.g. "excellent" matching *excel*, "reports to"
matching *reporting*, "drug pipeline" matching *etl*).

### Results, test split (n = 200)

| Variant | Macro-F1 | Micro-F1 | Subset acc. | Hamming acc. |
|---|---|---|---|---|
| A — cosine | 0.759 | 0.811 | 0.260 | 0.889 |
| **B — weighted hit** | **0.937** | **0.962** | **0.755** | **0.980** |

Variant B wins clearly. Cosine similarity over short pseudo-documents over-fires —
its precision on `excel` (0.54) and `ethics_governance` (0.47) drags the macro
average down, whereas targeted lexical matching with homonym suppression holds
precision above 0.9 on most categories. `cloud` is the weakest category under B
(F1 0.727), which is the expected consequence of platform names such as *databricks*
and *snowflake* being genuinely shared with `etl`.

Per-category precision, recall, F1 and support are in the two `*_test.csv` files.

---

## Running it

```bash
pip install -r requirements.txt

cd v3/base-model_tf_idf/pipeline
python tfidf_baseline.py \
  --corpus ../../manual_work/uk_analyst_corpus_v4_clean.csv \
  --gold   ../../manual_work/gold_standard_annotation_workbook_v2.xlsx \
  --outdir ../results
```

Writes per-variant dev/test reports, corpus-wide predictions and a
`tfidf_summary.json` with the tuned thresholds and timings into
`v3/base-model_tf_idf/results/`.

---

## Data provenance

The corpus is derived from a public dataset of LinkedIn job postings, filtered to UK
analyst-adjacent roles (see `filter_corpus.py` for the exact inclusion and exclusion
patterns). Postings are retained with their original `job_link`. The raw source files
are not included, so `filter_corpus.py` cannot be re-run from a clean clone — the
cleaned corpus is the starting point.

## Licence

Code and documentation: **MIT** ([LICENSE](LICENSE)). The job-posting data is
third-party content and is **not** licensed by this repository — see
[DATA_NOTICE.md](DATA_NOTICE.md).

## Status

Method 1 of 4 complete. Three further extraction methods are planned against the same
label space and the same gold standard, for direct comparison with this baseline.

## Reference

Attwood, S. & Williams, A. (2023) — TF-IDF mapping of job listings to CyBOK knowledge
areas, the approach adapted in variant A.
