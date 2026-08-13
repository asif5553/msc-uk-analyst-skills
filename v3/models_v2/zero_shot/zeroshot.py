"""
Method 2 of 4: transformer-based zero-shot classification.

Uses a natural language inference (NLI) model as a zero-shot classifier, following the
entailment-based approach evaluated comparatively by Kyritsis et al. (2024). Each skill
category is expressed as a natural-language hypothesis; the model scores how strongly
the posting entails it. No training data and no lexicon is used at inference time,
which makes this the first genuinely independent method in the pipeline: unlike the
lexical variants of the TF-IDF baseline, it shares nothing with the annotation
guidelines and can therefore be fairly measured on the hard subset.

Long postings: the corpus averages ~3,000 characters and NLI models cap at 512 tokens,
so postings are split into overlapping word-level chunks and each chunk is scored
independently. The posting-level score is the maximum across chunks, on the reasoning
that a skill mentioned anywhere in the posting counts as mentioned - matching the
annotation rule. Naive truncation would silently discard skills named late in a
posting, which annotation experience showed to be common.

Thresholds are tuned per category on the dev split only.

Usage (GPU strongly recommended):
    python zeroshot.py --corpus <csv> --gold <xlsx> --outdir <dir> [--model NAME]
    python zeroshot.py ... --full-corpus     # also predict all 820 postings
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import CATEGORIES, CATEGORY_LABELS
from evaluate import (load_gold, make_split, tune_thresholds, evaluate, format_report,
                      lexicon_presence, evaluate_hard, format_hard_report)

DEFAULT_MODEL = "facebook/bart-large-mnli"
HYPOTHESIS_TEMPLATE = "This job posting requires {}."

# Word-level chunking. 350 words is comfortably inside a 512-token budget once the
# hypothesis and special tokens are added; 50 words of overlap stops a skill mention
# being split across a boundary and lost.
CHUNK_WORDS = 350
CHUNK_OVERLAP = 50


def chunk_text(text, size=CHUNK_WORDS, overlap=CHUNK_OVERLAP):
    """Split into overlapping word-level chunks. Always returns at least one chunk."""
    words = (text or "").split()
    if not words:
        return [""]
    if len(words) <= size:
        return [" ".join(words)]
    step = size - overlap
    return [" ".join(words[i:i + size]) for i in range(0, len(words), step)
            if words[i:i + size]]


class ZeroShotScorer:
    """Entailment scorer over a fixed set of hypotheses."""

    def __init__(self, model_name=DEFAULT_MODEL, device=None, batch_size=16):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device).eval()
        self.batch_size = batch_size
        # NLI heads are ordered [contradiction, neutral, entailment] for these models;
        # read the index from the config rather than assuming.
        labels = {v.lower(): k for k, v in self.model.config.id2label.items()}
        self.entail_idx = labels.get("entailment", 2)
        self.contra_idx = labels.get("contradiction", 0)
        self.hypotheses = [HYPOTHESIS_TEMPLATE.format(CATEGORY_LABELS[c])
                           for c in CATEGORIES]

    @torch.no_grad()
    def _score_pairs(self, premises, hypotheses):
        """P(entailment) for each (premise, hypothesis) pair, contradiction-normalised."""
        out = []
        for i in range(0, len(premises), self.batch_size):
            enc = self.tokenizer(
                premises[i:i + self.batch_size], hypotheses[i:i + self.batch_size],
                return_tensors="pt", truncation=True, max_length=512, padding=True,
            ).to(self.device)
            logits = self.model(**enc).logits
            # standard zero-shot normalisation: softmax over {contradiction, entailment}
            pair = logits[:, [self.contra_idx, self.entail_idx]]
            probs = torch.softmax(pair, dim=1)[:, 1]
            out.extend(probs.cpu().numpy().tolist())
        return np.array(out)

    def score_documents(self, texts, verbose=True):
        """Return an (n_docs, n_categories) matrix of max-over-chunk entailment scores."""
        S = np.zeros((len(texts), len(CATEGORIES)))
        t0 = time.time()
        for d, text in enumerate(texts):
            chunks = chunk_text(text)
            premises, hyps = [], []
            for ch in chunks:
                for h in self.hypotheses:
                    premises.append(ch)
                    hyps.append(h)
            scores = self._score_pairs(premises, hyps)
            scores = scores.reshape(len(chunks), len(CATEGORIES))
            S[d] = scores.max(axis=0)
            if verbose and (d + 1) % 25 == 0:
                el = time.time() - t0
                print(f"  scored {d + 1}/{len(texts)} postings "
                      f"({el:.0f}s elapsed, {el / (d + 1):.2f}s/posting)", flush=True)
        return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--full-corpus", action="store_true",
                    help="also score all corpus postings (slow; needed for the market analysis)")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    corpus = pd.read_csv(args.corpus)
    gold_df, y = load_gold(args.gold)
    text_by_id = dict(zip(corpus["posting_id"], corpus["job_summary"].fillna("")))
    gold_texts = [text_by_id[p] for p in gold_df["posting_id"]]

    dev, test = make_split(gold_df)
    lex = lexicon_presence(gold_texts)
    print(f"corpus {len(corpus)} | gold {len(gold_df)} (dev {dev.sum()}, test {test.sum()})")
    print(f"model: {args.model}")

    scorer = ZeroShotScorer(args.model, batch_size=args.batch_size)
    print(f"device: {scorer.device}")

    t0 = time.time()
    S = scorer.score_documents(gold_texts)
    t_gold = time.time() - t0
    print(f"scored {len(gold_texts)} postings in {t_gold:.0f}s "
          f"({t_gold / len(gold_texts):.2f}s/posting)")
    np.save(outdir / "zeroshot_scores_gold.npy", S)

    thr = tune_thresholds(S[dev], y[dev])
    pred = (S >= thr).astype(int)

    rep_test = evaluate(y[test], pred[test])
    rep_dev = evaluate(y[dev], pred[dev])
    hard_test = evaluate_hard(y[test], pred[test], lex[test])
    print(format_report(rep_test, f"Zero-shot - TEST split (n={test.sum()})"))
    print(format_hard_report(hard_test, "Zero-shot - hard subset (TEST)"))

    rep_test.to_csv(outdir / "zeroshot_test.csv", index=False)
    rep_dev.to_csv(outdir / "zeroshot_dev.csv", index=False)
    hard_test.to_csv(outdir / "zeroshot_hard_test.csv", index=False)

    out = pd.DataFrame(pred, columns=CATEGORIES)
    out.insert(0, "posting_id", gold_df["posting_id"].values)
    out.insert(1, "split", np.where(dev, "dev", "test"))
    out.to_csv(outdir / "zeroshot_predictions_gold.csv", index=False)

    summary = {
        "method": "zeroshot",
        "model": args.model,
        "hypothesis_template": HYPOTHESIS_TEMPLATE,
        "chunk_words": CHUNK_WORDS, "chunk_overlap": CHUNK_OVERLAP,
        "device": scorer.device,
        "thresholds": dict(zip(CATEGORIES, thr.round(4).tolist())),
        "test_macro_f1": float(rep_test.loc[rep_test.category == "MACRO AVG", "f1"].iloc[0]),
        "test_micro_f1": float(rep_test.loc[rep_test.category == "MICRO AVG", "f1"].iloc[0]),
        "dev_macro_f1": float(rep_dev.loc[rep_dev.category == "MACRO AVG", "f1"].iloc[0]),
        "hard_accuracy": float(hard_test.loc[hard_test.subset == "hard subset (all)", "accuracy"].iloc[0]),
        "seconds_gold": t_gold,
        "seconds_per_posting": t_gold / len(gold_texts),
        "n_gold": len(gold_df), "n_dev": int(dev.sum()), "n_test": int(test.sum()),
    }

    if args.full_corpus:
        print(f"\nscoring all {len(corpus)} corpus postings...")
        t0 = time.time()
        S_full = scorer.score_documents(corpus["job_summary"].fillna("").tolist())
        t_full = time.time() - t0
        np.save(outdir / "zeroshot_scores_corpus.npy", S_full)
        full_pred = pd.DataFrame((S_full >= thr).astype(int), columns=CATEGORIES)
        full_pred.insert(0, "posting_id", corpus["posting_id"].values)
        full_pred.to_csv(outdir / "zeroshot_predictions_corpus.csv", index=False)
        summary["seconds_corpus"] = t_full
        print(f"corpus scored in {t_full:.0f}s")

    (outdir / "zeroshot_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\ntest macro-F1 {summary['test_macro_f1']:.3f} | "
          f"hard-subset accuracy {summary['hard_accuracy']:.3f}")
    print("compare: TF-IDF cosine macro-F1 0.759 / hard 0.396; "
          "lexical ceiling macro-F1 0.937 / hard 0.000")


if __name__ == "__main__":
    main()
