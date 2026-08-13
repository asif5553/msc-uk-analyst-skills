"""
Method 1 of 4: TF-IDF baseline.

Adapts the mapping approach of Attwood & Williams (2023), who represented both job
listings and CyBOK knowledge areas as TF-IDF vectors and assigned listings to areas by
cosine similarity. Here the target label space is the 13-category skill taxonomy rather
than CyBOK, and each category is represented as a pseudo-document built from its
Tier 2 lexicon.

Three reference points are produced. Only the first is an independent method.

  A. cosine        TF-IDF cosine similarity between posting and category
                   pseudo-document, thresholded. This is the Attwood & Williams
                   method transferred to this taxonomy, and is the baseline that
                   should be compared against the transformer and LLM methods.

  B. weighted-hit  Sum of TF-IDF weights of the category's lexicon terms present in
                   the posting. NOT AN INDEPENDENT METHOD: its term lists are the same
                   lists that define the annotation guidelines, so it partly
                   reconstructs the labelling procedure rather than solving the task.
                   Reported as an upper bound on what lexical matching achieves.

  C. lexicon       Unweighted, untuned lexicon presence, exactly as published in the
                   guidelines. Used to define the hard subset (see evaluate.py) and
                   to quantify how much of the task is solved by string matching.

Thresholds for A and B are tuned per category on the dev split only and applied
unchanged to the test split, so reported test scores are not fitted to the evaluation
data. Variant C has no tuning at all.

Usage:  python tfidf_baseline.py --corpus <csv> --gold <xlsx> --outdir <dir>
"""

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import CATEGORIES, LEXICONS, NEGATIVE_PATTERNS
from evaluate import (load_gold, make_split, tune_thresholds, evaluate, format_report,
                      lexicon_presence, evaluate_hard, format_hard_report)


def build_category_documents():
    """One pseudo-document per category, from its lexicon terms."""
    return [" ".join(LEXICONS[c]) for c in CATEGORIES]


def mask_negative_patterns(text, category):
    """Blank out known false-positive spans before lexical matching."""
    for pat in NEGATIVE_PATTERNS.get(category, []):
        text = re.sub(pat, " ", text, flags=re.IGNORECASE)
    return text


def fit_vectoriser(corpus_texts):
    vec = TfidfVectorizer(
        sublinear_tf=True, stop_words="english", ngram_range=(1, 2),
        min_df=2, max_df=0.9, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9+#/.\-]+\b",
    )
    vec.fit(corpus_texts)
    return vec


def cosine_scores(vec, target_texts):
    """Variant A: cosine similarity against category pseudo-documents."""
    X = vec.transform(target_texts)
    C = vec.transform(build_category_documents())
    S = cosine_similarity(X, C)
    denom = S.max(axis=0, keepdims=True)
    denom[denom == 0] = 1.0
    return S / denom


def weighted_hit_scores(vec, target_texts):
    """Variant B: TF-IDF-weighted lexicon hits, with homonym suppression."""
    vocab, idf = vec.vocabulary_, vec.idf_
    S = np.zeros((len(target_texts), len(CATEGORIES)))
    for j, cat in enumerate(CATEGORIES):
        terms = [t.lower() for t in LEXICONS[cat]]
        weights = np.array([idf[vocab[t]] if t in vocab else 1.0 for t in terms])
        weights = weights / weights.sum()
        patterns = [(re.compile(rf"\b{re.escape(t)}\b", re.IGNORECASE), w)
                    for t, w in zip(terms, weights)]
        for i, text in enumerate(target_texts):
            cleaned = mask_negative_patterns(text, cat)
            S[i, j] = sum(w for pat, w in patterns if pat.search(cleaned))
    denom = S.max(axis=0, keepdims=True)
    denom[denom == 0] = 1.0
    return S / denom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    corpus = pd.read_csv(args.corpus)
    gold_df, y = load_gold(args.gold)
    text_by_id = dict(zip(corpus["posting_id"], corpus["job_summary"].fillna("")))
    gold_texts = [text_by_id[p] for p in gold_df["posting_id"]]
    corpus_texts = corpus["job_summary"].fillna("").tolist()

    dev, test = make_split(gold_df)
    print(f"corpus {len(corpus)} postings | gold {len(gold_df)} "
          f"(dev {dev.sum()}, test {test.sum()})")

    vec = fit_vectoriser(corpus_texts)

    timings, results, preds = {}, {}, {}

    t0 = time.time(); S_cos = cosine_scores(vec, gold_texts); timings["A_cosine_sec"] = time.time() - t0
    t0 = time.time(); S_hit = weighted_hit_scores(vec, gold_texts); timings["B_weighted_hit_sec"] = time.time() - t0
    t0 = time.time(); lex = lexicon_presence(gold_texts); timings["C_lexicon_sec"] = time.time() - t0

    # how much of the task is solved by string matching alone
    agreement = float((lex == y).mean())
    n_hard = int((lex != y).sum())
    print(f"\nlexicon presence agrees with gold on {agreement:.1%} of pairs "
          f"({n_hard} hard pairs of {y.size})")

    for name, S in (("A_cosine", S_cos), ("B_weighted_hit", S_hit)):
        thr = tune_thresholds(S[dev], y[dev])
        pred = (S >= thr).astype(int)
        rep_test = evaluate(y[test], pred[test])
        rep_dev = evaluate(y[dev], pred[dev])
        hard_test = evaluate_hard(y[test], pred[test], lex[test])
        print(format_report(rep_test, f"TF-IDF {name} - TEST split (n={test.sum()})"))
        print(format_hard_report(hard_test, f"TF-IDF {name} - hard subset (TEST)"))
        rep_test.to_csv(outdir / f"tfidf_{name}_test.csv", index=False)
        rep_dev.to_csv(outdir / f"tfidf_{name}_dev.csv", index=False)
        hard_test.to_csv(outdir / f"tfidf_{name}_hard_test.csv", index=False)
        results[name] = {
            "thresholds": dict(zip(CATEGORIES, thr.round(4).tolist())),
            "test_macro_f1": float(rep_test.loc[rep_test.category == "MACRO AVG", "f1"].iloc[0]),
            "test_micro_f1": float(rep_test.loc[rep_test.category == "MICRO AVG", "f1"].iloc[0]),
            "dev_macro_f1": float(rep_dev.loc[rep_dev.category == "MACRO AVG", "f1"].iloc[0]),
            "hard_accuracy": float(hard_test.loc[hard_test.subset == "hard subset (all)", "accuracy"].iloc[0]),
        }
        preds[name] = pred

    # variant C: untuned lexicon presence
    rep_lex = evaluate(y[test], lex[test])
    hard_lex = evaluate_hard(y[test], lex[test], lex[test])
    print(format_report(rep_lex, "TF-IDF C_lexicon (untuned presence) - TEST split"))
    rep_lex.to_csv(outdir / "tfidf_C_lexicon_test.csv", index=False)
    results["C_lexicon"] = {
        "thresholds": None,
        "test_macro_f1": float(rep_lex.loc[rep_lex.category == "MACRO AVG", "f1"].iloc[0]),
        "test_micro_f1": float(rep_lex.loc[rep_lex.category == "MICRO AVG", "f1"].iloc[0]),
        "hard_accuracy": 0.0,  # by construction the lexicon is wrong on every hard pair
    }
    preds["C_lexicon"] = lex

    # A is the reportable baseline; B and C are reference points, not competitors
    headline = "A_cosine"
    S_full = cosine_scores(vec, corpus_texts)
    thr_head = np.array([results[headline]["thresholds"][c] for c in CATEGORIES])
    full_pred = pd.DataFrame((S_full >= thr_head).astype(int), columns=CATEGORIES)
    full_pred.insert(0, "posting_id", corpus["posting_id"].values)
    full_pred.to_csv(outdir / "tfidf_predictions_corpus.csv", index=False)

    for name, pred in preds.items():
        out = pd.DataFrame(pred, columns=CATEGORIES)
        out.insert(0, "posting_id", gold_df["posting_id"].values)
        out.insert(1, "split", np.where(dev, "dev", "test"))
        out.to_csv(outdir / f"tfidf_predictions_gold_{name}.csv", index=False)

    # hard-subset pairs, exported for qualitative error analysis in the write-up
    hard_pairs = []
    for i in range(len(gold_df)):
        for j, c in enumerate(CATEGORIES):
            if lex[i, j] != y[i, j]:
                hard_pairs.append({
                    "posting_id": gold_df.iloc[i]["posting_id"],
                    "job_title": gold_df.iloc[i]["job_title"],
                    "split": "dev" if dev[i] else "test",
                    "category": c,
                    "gold": int(y[i, j]),
                    "lexicon": int(lex[i, j]),
                    "error_type": "lexicon_false_positive" if lex[i, j] == 1 else "lexicon_false_negative",
                    "tfidf_cosine": int(preds["A_cosine"][i, j]),
                    "note": gold_df.iloc[i].get("notes", ""),
                })
    pd.DataFrame(hard_pairs).to_csv(outdir / "hard_subset_pairs.csv", index=False)

    summary = {
        "method": "tfidf",
        "headline_variant": headline,
        "lexicon_gold_agreement": agreement,
        "n_hard_pairs": n_hard,
        "results": results,
        "timings_sec": timings,
        "n_corpus": len(corpus), "n_gold": len(gold_df),
        "n_dev": int(dev.sum()), "n_test": int(test.sum()),
        "note": ("Variant B shares its term lists with the annotation guidelines and is "
                 "reported as a lexical upper bound, not as an independent method. "
                 "Variant A is the reportable Attwood & Williams baseline."),
    }
    (outdir / "tfidf_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nheadline (A_cosine) test macro-F1 {results['A_cosine']['test_macro_f1']:.3f} | "
          f"lexical ceiling (B) {results['B_weighted_hit']['test_macro_f1']:.3f} | "
          f"untuned lexicon (C) {results['C_lexicon']['test_macro_f1']:.3f}")


if __name__ == "__main__":
    main()
