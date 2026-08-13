# AGENTS.md

Context for AI coding agents working in this repository. Humans: start with `README.md`.

## What this is

An MSc research project, not a product. It extracts a **13-category skill taxonomy**
from UK data/business analyst job adverts and measures the extraction against a
**manually annotated gold standard of 300 postings**. Four extraction methods are
planned. Method 1, the **TF-IDF baseline**, is complete in `v3/base-model_tf_idf/`;
zero-shot, LLM prompting and BERTopic have all been run and their results committed
under `v3/models_v1/` and `v3/models_v2/`, but see `README.md` for which of those
numbers are the reported ones.

The point of the repo is *comparability between methods*. Anything that makes two
methods less directly comparable is a regression, even if it improves a score.

## Repository shape

```
V1/  v2/     earlier iterations — provenance only, DO NOT EDIT
v3/           current work
  manual_work/        corpus construction + human annotation
  base-model_tf_idf/  method 1: TF-IDF baseline (pipeline/ + results/)
  models_v1/          first pass at methods 2-3: zero-shot, LLM prompting
  models_v2/          revised zero-shot and LLM prompting, plus BERTopic
```

`V1/` and `v2/` are frozen historical snapshots. Never edit, refactor, "fix" or
delete them, and never import from them. If you notice a bug there, mention it —
don't patch it.

## Ground rules

**The label space is defined once.** `config.py` holds `CATEGORIES`, and its
**order is load-bearing** — it defines the column order of every prediction matrix
and must match the gold-standard workbook columns. Never reorder, rename, insert or
drop a category without updating the gold workbook and every results file in
lockstep. Each method directory currently carries a byte-identical copy of
`config.py` (they are run standalone, including on Colab); treat
`v3/base-model_tf_idf/pipeline/config.py` as the source and propagate any change to
every copy, or the methods stop being comparable.

**Never tune on test.** Thresholds are selected on the dev split only
(`tune_thresholds` in `evaluate.py`) and applied unchanged to test. Any change that
lets test data influence threshold selection, feature fitting or model choice
invalidates the reported numbers. This is the single most important constraint here.

**The split must stay reproducible.** `make_split()` is seeded (`RANDOM_SEED = 42`)
and stratified on `role_family`. Changing the seed, the fraction or the
stratification key silently makes new results incomparable with every result already
committed. Don't.

**Gold standard is human-produced.** `gold_standard_annotation_workbook_v2.xlsx` is
the output of manual annotation work against documented guidelines. It is data, not
a generated artefact — never rewrite it programmatically, and never "correct" labels
to improve a score.

**Results are committed.** Every `results/` directory is tracked — the numbers
behind each method live in the repo next to the code that produced them, so a
reader can check a claim without re-running anything. Regenerating a results file
is therefore a reviewable change: overwrite one only when you intend to change the
numbers reported in `README.md` too, and say so in the commit message.

## How the baseline works

Two variants, both scoring every (posting, category) pair then thresholding:

- **A — cosine.** TF-IDF vectors for postings and for category pseudo-documents
  built from `LEXICONS`; cosine similarity between them. Adapts Attwood & Williams
  (2023), who mapped listings to CyBOK knowledge areas.
- **B — weighted hit.** Sum of TF-IDF weights of a category's lexicon terms found in
  the posting, normalised, plus `NEGATIVE_PATTERNS` suppression for homonyms
  ("excellent" → *excel*, "reports to" → *reporting*, "drug pipeline" → *etl*).

B wins substantially (test macro-F1 **0.937** vs **0.759**). If you touch the
lexicons or negative patterns, re-run and expect these numbers to move — update
`README.md` if they do.

## Adding method 2, 3 or 4

Follow the shape of `tfidf_baseline.py` so results stay comparable:

1. Import `CATEGORIES` (and `CATEGORY_LABELS` for anything prompt- or
   hypothesis-based) from `config.py`. Do not redefine the taxonomy.
2. Use `load_gold`, `make_split`, `tune_thresholds`, `evaluate` and `format_report`
   from `evaluate.py` unchanged.
3. Produce a score matrix of shape `(n_postings, 13)` in `CATEGORIES` order, then
   threshold it — don't emit hard labels directly, or thresholds can't be tuned
   comparably.
4. Write `<method>_<variant>_dev.csv`, `<method>_<variant>_test.csv` and a
   `<method>_summary.json` with thresholds, timings and split sizes, matching the
   baseline's output contract.
5. Take `--corpus`, `--gold` and `--outdir` as arguments. No hardcoded paths.

## Conventions

- British spelling in category names and labels (`visualisation_bi`,
  `data_modelling`). The lexicons deliberately carry both spellings as *match terms*
  — that's intentional, not an inconsistency to clean up.
- Lexicon terms are lowercase and matched case-insensitively as whole words.
- Plain functions, `argparse`, `pathlib`. No framework, no class hierarchy, no
  config system. Match that — this code is read by examiners.
- Docstrings explain *why* a choice was made (and cite sources where relevant).
  Keep that habit; it's doing real work in a dissertation context.

## Environment note

Python is **not currently installed** on the development machine (`python` resolves
to the Windows Store stub). Nothing here has been executed in this environment —
don't claim a script runs unless you actually ran it. Setup:

```bash
pip install -r requirements.txt
```

## Data handling

The corpus derives from a public LinkedIn job-postings dataset, filtered by
`filter_corpus.py`. `filter_corpus.py` expects `linkedin_job_postings.csv` and
`job_summary.csv` in the working directory — **these source files are not in the
repo**, so that script can't be re-run from a clean clone. The postings carry
company names and original `job_link` URLs. Treat as third-party content: don't
redistribute it further, and don't paste posting text into external services.
