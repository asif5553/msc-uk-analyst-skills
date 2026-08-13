"""
Evaluation harness for the extraction methods.

Provides:
  - load_gold()          : gold-standard label matrix from the annotation workbook
  - make_split()         : reproducible dev/test split, stratified by role family
  - evaluate()           : per-category and aggregate precision / recall / F1
  - tune_thresholds()    : per-category threshold selection on the dev split only
  - lexicon_presence()   : raw lexicon-match matrix (the "does the term appear" baseline)
  - hard_subset_mask()   : posting-category pairs where lexicon presence != gold label
  - evaluate_hard()      : accuracy on that subset, split by error type

Two design points matter for interpreting results.

1. Dev/test split. Any method whose decision threshold is tuned must be tuned on dev
   and reported on test, otherwise the reported score is optimistically biased.

2. Hard subset. The gold standard follows an explicit-mention rule, so its labels
   agree with raw lexicon presence on ~98% of posting-category pairs. Headline F1 is
   therefore dominated by cases any string matcher solves, and cannot separate
   methods. The hard subset isolates the pairs where lexicon presence and the gold
   label disagree - homonyms, team descriptions, recruiter footers, qualification
   lists, and mentions whose surface form is absent from the lexicon. These are the
   cases requiring interpretation rather than matching, and they are where semantic
   methods can demonstrate value over the lexical baseline.
"""

import re

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from config import CATEGORIES, LEXICONS, NEGATIVE_PATTERNS

RANDOM_SEED = 42


def load_gold(workbook_path):
    """Return (dataframe, label matrix) for the annotated postings."""
    g = pd.read_excel(workbook_path, sheet_name="Annotation")
    g = g[g[CATEGORIES].notna().all(axis=1)].reset_index(drop=True)
    y = g[CATEGORIES].astype(int).values
    return g, y


def make_split(gold_df, dev_frac=1 / 3, seed=RANDOM_SEED):
    """Stratified dev/test split on role family. Returns boolean masks."""
    rng = np.random.default_rng(seed)
    is_dev = np.zeros(len(gold_df), dtype=bool)
    for _, idx in gold_df.groupby("role_family").groups.items():
        idx = np.array(list(idx))
        rng.shuffle(idx)
        n_dev = max(1, int(round(len(idx) * dev_frac)))
        is_dev[idx[:n_dev]] = True
    return is_dev, ~is_dev


def evaluate(y_true, y_pred, categories=CATEGORIES):
    """Per-category and aggregate metrics as a tidy dataframe."""
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0, labels=range(len(categories))
    )
    rows = [
        {"category": c, "precision": p[i], "recall": r[i], "f1": f[i],
         "support": int(y_true[:, i].sum()), "predicted": int(y_pred[:, i].sum())}
        for i, c in enumerate(categories)
    ]
    for name, avg in (("MACRO AVG", "macro"), ("MICRO AVG", "micro")):
        pp, rr, ff, _ = precision_recall_fscore_support(
            y_true, y_pred, average=avg, zero_division=0
        )
        rows.append({"category": name, "precision": pp, "recall": rr, "f1": ff,
                     "support": int(y_true.sum()), "predicted": int(y_pred.sum())})
    rows.append({"category": "SUBSET ACCURACY", "precision": np.nan, "recall": np.nan,
                 "f1": float((y_true == y_pred).all(axis=1).mean()),
                 "support": len(y_true), "predicted": len(y_true)})
    rows.append({"category": "HAMMING ACCURACY", "precision": np.nan, "recall": np.nan,
                 "f1": float((y_true == y_pred).mean()),
                 "support": y_true.size, "predicted": y_true.size})
    return pd.DataFrame(rows)


def tune_thresholds(scores, y_true, grid=None):
    """Pick the per-category threshold maximising F1 on the given (dev) data."""
    if grid is None:
        grid = np.linspace(0.0, 1.0, 201)
    thresholds = np.zeros(scores.shape[1])
    for i in range(scores.shape[1]):
        best_f, best_t = -1.0, 0.5
        for t in grid:
            pred = (scores[:, i] >= t).astype(int)
            _, _, f, _ = precision_recall_fscore_support(
                y_true[:, i], pred, average="binary", zero_division=0
            )
            if f > best_f:
                best_f, best_t = f, t
        thresholds[i] = best_t
    return thresholds


# --------------------------------------------------------------------------
# Lexicon baseline and hard subset
# --------------------------------------------------------------------------

def lexicon_presence(texts, apply_negatives=True):
    """
    Binary matrix: does any lexicon term for each category appear in each text?

    This is the reference point against which the hard subset is defined. It uses the
    lexicons and homonym-suppression patterns exactly as published in annotation
    guidelines v2.0 - no tuning, no fitting, no data-dependent choices.
    """
    compiled = {
        cat: [re.compile(rf"\b{re.escape(t)}\b", re.IGNORECASE) for t in LEXICONS[cat]]
        for cat in CATEGORIES
    }
    negatives = {
        cat: [re.compile(p, re.IGNORECASE) for p in NEGATIVE_PATTERNS.get(cat, [])]
        for cat in CATEGORIES
    }
    M = np.zeros((len(texts), len(CATEGORIES)), dtype=int)
    for i, text in enumerate(texts):
        text = text or ""
        for j, cat in enumerate(CATEGORIES):
            cleaned = text
            if apply_negatives:
                for neg in negatives[cat]:
                    cleaned = neg.sub(" ", cleaned)
            M[i, j] = int(any(p.search(cleaned) for p in compiled[cat]))
    return M


def hard_subset_mask(y_gold, lex):
    """
    Boolean matrix marking posting-category pairs where the gold label disagrees with
    raw lexicon presence. These are the cases requiring interpretation.
    """
    return y_gold != lex


def evaluate_hard(y_true, y_pred, lex, categories=CATEGORIES):
    """
    Accuracy on the hard subset, broken down by the two error types the lexicon makes.

      lexicon false positives : term present, gold says 0 (homonym, team description,
                                recruiter footer, qualification list)
      lexicon false negatives : term absent, gold says 1 (skill named in a surface
                                form outside the lexicon)

    Reported as accuracy rather than F1: the subset is small and class-degenerate
    within each error type, so precision/recall are not informative here.
    """
    hard = hard_subset_mask(y_true, lex)
    fp = hard & (lex == 1)
    fn = hard & (lex == 0)
    rows = []
    for name, mask in (("hard subset (all)", hard),
                       ("  lexicon false positives", fp),
                       ("  lexicon false negatives", fn)):
        n = int(mask.sum())
        acc = float((y_pred[mask] == y_true[mask]).mean()) if n else np.nan
        rows.append({"subset": name, "n_pairs": n, "accuracy": acc})
    for j, c in enumerate(categories):
        m = hard[:, j]
        n = int(m.sum())
        if n:
            rows.append({"subset": f"    {c}", "n_pairs": n,
                         "accuracy": float((y_pred[m, j] == y_true[m, j]).mean())})
    easy = ~hard
    rows.append({"subset": "easy subset (lexicon agrees with gold)",
                 "n_pairs": int(easy.sum()),
                 "accuracy": float((y_pred[easy] == y_true[easy]).mean())})
    return pd.DataFrame(rows)


def format_report(df, title):
    """Readable console table for evaluate() output."""
    lines = [f"\n{'=' * 72}", title, "=" * 72,
             f"{'category':22s} {'prec':>7s} {'rec':>7s} {'F1':>7s} {'n_true':>7s} {'n_pred':>7s}"]
    for _, r in df.iterrows():
        if r["category"] in ("MACRO AVG", "MICRO AVG"):
            lines.append("-" * 72)
        pv = "  n/a  " if pd.isna(r["precision"]) else f"{r['precision']:7.3f}"
        rv = "  n/a  " if pd.isna(r["recall"]) else f"{r['recall']:7.3f}"
        lines.append(f"{r['category']:22s} {pv} {rv} {r['f1']:7.3f} "
                     f"{r['support']:7d} {r['predicted']:7d}")
    return "\n".join(lines)


def format_hard_report(df, title):
    """Readable console table for evaluate_hard() output."""
    lines = [f"\n{'-' * 72}", title, "-" * 72,
             f"{'subset':42s} {'n_pairs':>8s} {'accuracy':>9s}"]
    for _, r in df.iterrows():
        acc = "     n/a" if pd.isna(r["accuracy"]) else f"{r['accuracy']:9.3f}"
        lines.append(f"{r['subset']:42s} {r['n_pairs']:8d} {acc}")
    return "\n".join(lines)
