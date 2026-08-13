"""
Method 4 of 4: BERTopic (unsupervised topic modelling).

Implements the pipeline of Grootendorst (2022) - transformer embeddings, UMAP
dimensionality reduction, HDBSCAN clustering, c-TF-IDF topic representation - and
adapts it to the same 13-category label space as the other three methods so that the
comparison is on equal terms.

Two design decisions carry the method, and both should be stated in the write-up.

  Segments, not postings, are the clustering unit. HDBSCAN assigns each document to
  exactly one cluster, so clustering whole postings would yield at most one skill
  category per posting. The gold standard averages ~3.5 categories per posting, which
  would cap recall near 0.28 for reasons of output shape rather than method quality.
  Postings are therefore split into their constituent lines and sentences - the unit a
  skill mention actually occupies - and topic assignments are aggregated back up to the
  posting. Grootendorst's own discussion notes the one-topic-per-document assumption as
  a limitation; segmenting is the standard response to it.

  Topics are mapped to categories by embedding similarity, not by lexicon overlap.
  Each topic is represented by its representative documents (real corpus sentences,
  which sentence encoders handle far better than c-TF-IDF keyword bags); their mean
  similarity to the 13 CATEGORY_LABELS phrases gives the topic's affinity to each
  category. Mapping by lexicon overlap would reintroduce the circularity that makes
  the weighted-hit TF-IDF variant a lexical ceiling rather than an independent method,
  and would invalidate the hard-subset comparison. No lexicon term list is used
  anywhere in this file.

The model is fit on all corpus segments. This is unsupervised - no gold labels are seen
during fitting - so including the annotated postings in the fit introduces no label
leakage. Only the topic-to-category thresholds are fitted, and those are tuned on the
dev split alone and applied unchanged to test, as in the other three methods.

Usage:
    python bertopic_model.py --corpus <csv> --gold <xlsx> --outdir <dir>
    python bertopic_model.py ... --min-topic-size 30 --reduce-outliers
    python bertopic_model.py ... --no-eval          # descriptive topics only
"""

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from config import CATEGORIES, CATEGORY_LABELS
from evaluate import (load_gold, make_split, tune_thresholds, evaluate, format_report,
                      lexicon_presence, evaluate_hard, format_hard_report)

RANDOM_SEED = 42
DEFAULT_EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"

# Segmentation. Lines shorter than MIN_SEG_WORDS are headers, single tokens or list
# fragments and produce noise topics; lines longer than MAX_SEG_WORDS are unsplit
# paragraphs and are broken at sentence boundaries so a segment holds one claim.
MIN_SEG_WORDS = 4
MAX_SEG_WORDS = 60
TOPIC_TOP_TERMS = 10


# --------------------------------------------------------------------------
# Segmentation
# --------------------------------------------------------------------------

def split_segments(text):
    """Split a posting into line- and sentence-level segments."""
    out = []
    for line in re.split(r"[\n\r]+", text or ""):
        line = re.sub(r"^\s*[\u2022\u2023\u25aa\u25cf*\-\u2013\u2014]+\s*", "", line).strip()
        if not line:
            continue
        words = line.split()
        if len(words) < MIN_SEG_WORDS:
            continue
        if len(words) <= MAX_SEG_WORDS:
            out.append(line)
            continue
        # long paragraph: break at sentence boundaries, then hard-wrap any remainder
        buf = []
        for sent in re.split(r"(?<=[.;!?])\s+", line):
            sw = sent.split()
            if not sw:
                continue
            if len(sw) > MAX_SEG_WORDS:
                for i in range(0, len(sw), MAX_SEG_WORDS):
                    piece = sw[i:i + MAX_SEG_WORDS]
                    if len(piece) >= MIN_SEG_WORDS:
                        out.append(" ".join(piece))
            elif len(sw) >= MIN_SEG_WORDS:
                out.append(sent.strip())
            else:
                buf.append(sent.strip())
        if buf:
            merged = " ".join(buf)
            if len(merged.split()) >= MIN_SEG_WORDS:
                out.append(merged)
    return out


def build_segment_table(posting_ids, texts):
    """Long-format table: one row per segment, carrying its posting_id."""
    rows = []
    for pid, text in zip(posting_ids, texts):
        for k, seg in enumerate(split_segments(text)):
            rows.append({"posting_id": pid, "segment_index": k, "segment": seg})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Topic model
# --------------------------------------------------------------------------

def load_encoder(name):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(name)


def embed_segments(encoder, segments, cache_path=None, batch_size=64):
    """Embed once and cache: refitting the topic model should not re-embed."""
    if cache_path is not None and Path(cache_path).exists():
        emb = np.load(cache_path)
        if len(emb) == len(segments):
            print(f"  loaded cached embeddings {emb.shape}")
            return emb
        print("  cached embeddings do not match segment count; re-embedding")
    emb = encoder.encode(segments, batch_size=batch_size, show_progress_bar=True,
                         convert_to_numpy=True)
    if cache_path is not None:
        np.save(cache_path, emb)
    return emb


def fit_topic_model(segments, embeddings, min_topic_size, nr_topics=None, seed=RANDOM_SEED):
    """Grootendorst's four-step pipeline with the seeds and guards pinned down."""
    from bertopic import BERTopic
    from bertopic.vectorizers import ClassTfidfTransformer
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP

    n = len(segments)
    if n < 50:
        raise ValueError(f"only {n} segments; too few to cluster meaningfully")
    # guard: UMAP fails or degenerates when n_neighbors approaches the sample size
    n_neighbors = max(2, min(15, n - 1))
    min_topic_size = max(2, min(min_topic_size, n // 10))

    umap_model = UMAP(n_neighbors=n_neighbors, n_components=5, min_dist=0.0,
                      metric="cosine", random_state=seed)
    hdbscan_model = HDBSCAN(min_cluster_size=min_topic_size, metric="euclidean",
                            cluster_selection_method="eom", prediction_data=True)
    # min_df guards against topic labels built from hapax terms in small corpora
    vectorizer_model = CountVectorizer(stop_words="english", ngram_range=(1, 2),
                                       min_df=max(2, n // 2000))

    topic_model = BERTopic(
        embedding_model=None,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ClassTfidfTransformer(reduce_frequent_words=True),
        nr_topics=nr_topics,
        calculate_probabilities=False,
        verbose=True,
    )
    topics, _ = topic_model.fit_transform(segments, embeddings=embeddings)
    return topic_model, np.asarray(topics)


def topic_term_strings(topic_model, topic_ids, top_n=TOPIC_TOP_TERMS):
    """Natural-language rendering of each topic from its c-TF-IDF terms."""
    out = {}
    for t in topic_ids:
        words = topic_model.get_topic(t) or []
        out[t] = ", ".join(w for w, _ in words[:top_n] if w)
    return out


def map_topics_to_categories(encoder, topic_model, term_strings, topic_ids):
    """
    Map each topic to the 13 categories by embedding similarity between the topic's
    REPRESENTATIVE DOCUMENTS and the category label phrases.

    Earlier versions embedded the comma-separated c-TF-IDF keyword bag. Sentence
    encoders represent keyword bags poorly - a topic of "communication, written,
    verbal, interpersonal" scored 0.14 against 'stakeholder communication and
    presenting findings' and mapped to nothing. Representative documents are real
    sentences from the corpus, which is what the encoder was trained on. Affinity for
    a topic is the mean similarity of its (up to 3) representative documents to each
    label; mean rather than max because a topic is coherent by construction and max
    would let one atypical exemplar set the mapping.

    Uses CATEGORY_LABELS only - the same phrases the zero-shot method uses as
    hypotheses - so the mapping remains independent of the annotation lexicons.
    """
    label_texts = [CATEGORY_LABELS[c] for c in CATEGORIES]

    def unit(a):
        n = np.linalg.norm(a, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return a / n

    label_emb = unit(encoder.encode(label_texts, convert_to_numpy=True))

    affinity = np.zeros((len(topic_ids), len(CATEGORIES)))
    n_fallback, n_zeroed = 0, 0
    for i, t in enumerate(topic_ids):
        try:
            docs = topic_model.get_representative_docs(t) or []
        except (KeyError, TypeError):
            docs = []
        docs = [d for d in docs if d and d.strip()][:3]
        if docs:
            demb = unit(encoder.encode(docs, convert_to_numpy=True))
            affinity[i] = (demb @ label_emb.T).mean(axis=0)
        elif term_strings.get(t):
            # fallback: the old keyword-bag mapping, better than nothing
            n_fallback += 1
            temb = unit(encoder.encode([term_strings[t]], convert_to_numpy=True))
            affinity[i] = (temb @ label_emb.T)[0]
        else:
            # no interpretable content at all: exclude from mapping entirely
            n_zeroed += 1
    if n_fallback:
        print(f"  {n_fallback} topic(s) had no representative docs; "
              f"fell back to c-TF-IDF term mapping")
    if n_zeroed:
        print(f"  {n_zeroed} topic(s) had no usable content; excluded from mapping")
    return affinity


def posting_scores(seg_df, topics, affinity, topic_index, posting_ids):
    """
    Aggregate segment topics up to posting level.

    A posting's score for a category is the maximum affinity over its segments'
    topics. Max rather than mean because the annotation rule is that a skill mentioned
    anywhere in the posting counts - the same aggregation the zero-shot method uses
    over its chunks. Outlier segments (topic -1) contribute nothing.
    """
    pos = {p: i for i, p in enumerate(posting_ids)}
    S = np.zeros((len(posting_ids), len(CATEGORIES)))
    for pid, t in zip(seg_df["posting_id"].values, topics):
        if t == -1 or pid not in pos:
            continue
        row = affinity[topic_index[t]]
        i = pos[pid]
        np.maximum(S[i], row, out=S[i])
    return S


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--embedder", default=DEFAULT_EMBEDDER)
    ap.add_argument("--min-topic-size", type=int, default=30)
    ap.add_argument("--nr-topics", default=None,
                    help="'auto' or an integer to reduce topics after fitting")
    ap.add_argument("--reduce-outliers", action="store_true",
                    help="reassign topic -1 segments to their nearest topic")
    ap.add_argument("--no-eval", action="store_true",
                    help="fit and export topics only, skip gold-standard evaluation")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    nr_topics = int(args.nr_topics) if (args.nr_topics or "").isdigit() else args.nr_topics

    corpus = pd.read_csv(args.corpus)
    corpus_ids = corpus["posting_id"].tolist()
    corpus_texts = corpus["job_summary"].fillna("").tolist()

    seg_df = build_segment_table(corpus_ids, corpus_texts)
    per_posting = seg_df.groupby("posting_id").size()
    print(f"corpus {len(corpus)} postings -> {len(seg_df)} segments "
          f"(median {per_posting.median():.0f}/posting, "
          f"{(per_posting.reindex(corpus_ids).fillna(0) == 0).sum()} postings with none)")

    encoder = load_encoder(args.embedder)
    t0 = time.time()
    emb = embed_segments(encoder, seg_df["segment"].tolist(),
                         cache_path=outdir / "bertopic_segment_embeddings.npy")
    t_embed = time.time() - t0

    t0 = time.time()
    topic_model, topics = fit_topic_model(seg_df["segment"].tolist(), emb,
                                          args.min_topic_size, nr_topics)
    t_fit = time.time() - t0

    outlier_share = float((topics == -1).mean())
    print(f"fitted in {t_fit:.0f}s | {len(set(topics)) - (1 if -1 in topics else 0)} topics "
          f"| outliers {outlier_share:.1%}")
    if outlier_share > 0.4:
        print("  WARNING: >40% of segments are outliers. Lower --min-topic-size, or "
              "pass --reduce-outliers, before reading anything into the topics.")

    outlier_share_pre = outlier_share
    if args.reduce_outliers and (topics == -1).any():
        topics = np.asarray(topic_model.reduce_outliers(
            seg_df["segment"].tolist(), topics.tolist(),
            strategy="embeddings", embeddings=emb))
        topic_model.update_topics(seg_df["segment"].tolist(), topics=topics.tolist())
        outlier_share = float((topics == -1).mean())
        print(f"  outliers reduced to {outlier_share:.1%}")

    seg_df["topic"] = topics
    topic_ids = sorted(t for t in set(topics.tolist()) if t != -1)
    topic_index = {t: i for i, t in enumerate(topic_ids)}
    term_strings = topic_term_strings(topic_model, topic_ids)
    affinity = map_topics_to_categories(encoder, topic_model, term_strings, topic_ids)

    # topic table: sizes, terms, and the category each topic maps onto
    info = topic_model.get_topic_info().set_index("Topic")
    best = affinity.argmax(axis=1)
    topics_out = pd.DataFrame({
        "topic": topic_ids,
        "n_segments": [int((topics == t).sum()) for t in topic_ids],
        "n_postings": [int(seg_df.loc[seg_df.topic == t, "posting_id"].nunique())
                       for t in topic_ids],
        "top_terms": [term_strings[t] for t in topic_ids],
        "mapped_category": [CATEGORIES[b] for b in best],
        "affinity": affinity.max(axis=1).round(4),
        "runner_up": [CATEGORIES[i] for i in affinity.argsort(axis=1)[:, -2]],
        "bertopic_name": [info.loc[t, "Name"] if t in info.index else "" for t in topic_ids],
    }).sort_values("n_segments", ascending=False)
    topics_out.to_csv(outdir / "bertopic_topics.csv", index=False)

    aff_out = pd.DataFrame(affinity.round(4), columns=CATEGORIES)
    aff_out.insert(0, "topic", topic_ids)
    aff_out.to_csv(outdir / "bertopic_topic_category_affinity.csv", index=False)
    seg_df.to_csv(outdir / "bertopic_segment_topics.csv", index=False)

    print("\nlargest topics:")
    for _, r in topics_out.head(12).iterrows():
        print(f"  {r.topic:>3}  n={r.n_segments:>5}  -> {r.mapped_category:<18} "
              f"({r.affinity:.2f})  {r.top_terms[:60]}")

    # taxonomy coverage: topics whose best category match is weak are candidate
    # themes the 13-category scheme does not cover. This is the descriptive
    # contribution BERTopic makes that the three top-down methods cannot.
    weak = topics_out[topics_out.affinity < topics_out.affinity.quantile(0.25)]
    coverage = {
        "n_topics": len(topic_ids),
        "outlier_share_pre_reduction": outlier_share_pre,
        "median_best_affinity": float(topics_out.affinity.median()),
        "categories_never_mapped": [c for c in CATEGORIES
                                    if c not in set(topics_out.mapped_category)],
        "weakly_mapped_topics": weak[["topic", "n_segments", "mapped_category",
                                      "affinity", "top_terms"]].to_dict("records"),
    }
    (outdir / "bertopic_coverage.json").write_text(json.dumps(coverage, indent=2))

    summary = {
        "method": "bertopic",
        "embedder": args.embedder,
        "min_topic_size": args.min_topic_size,
        "nr_topics": nr_topics,
        "reduce_outliers": bool(args.reduce_outliers),
        "segment_unit": "line/sentence",
        "n_corpus": len(corpus), "n_segments": len(seg_df),
        "n_topics": len(topic_ids),
        "outlier_share_pre_reduction": outlier_share_pre,
        "outlier_share_post_reduction": outlier_share,
        "seconds_embed": t_embed, "seconds_fit": t_fit,
        "random_seed": RANDOM_SEED,
    }

    if not args.no_eval:
        gold_df, y = load_gold(args.gold)
        gold_ids = gold_df["posting_id"].tolist()
        text_by_id = dict(zip(corpus_ids, corpus_texts))
        gold_texts = [text_by_id[p] for p in gold_ids]
        dev, test = make_split(gold_df)
        lex = lexicon_presence(gold_texts)

        S_gold = posting_scores(seg_df, topics, affinity, topic_index, gold_ids)
        thr = tune_thresholds(S_gold[dev], y[dev])
        pred = (S_gold >= thr).astype(int)

        rep_test = evaluate(y[test], pred[test])
        rep_dev = evaluate(y[dev], pred[dev])
        hard_test = evaluate_hard(y[test], pred[test], lex[test])
        print(format_report(rep_test, f"BERTopic - TEST split (n={test.sum()})"))
        print(format_hard_report(hard_test, "BERTopic - hard subset (TEST)"))

        rep_test.to_csv(outdir / "bertopic_test.csv", index=False)
        rep_dev.to_csv(outdir / "bertopic_dev.csv", index=False)
        hard_test.to_csv(outdir / "bertopic_hard_test.csv", index=False)

        out = pd.DataFrame(pred, columns=CATEGORIES)
        out.insert(0, "posting_id", gold_ids)
        out.insert(1, "split", np.where(dev, "dev", "test"))
        out.to_csv(outdir / "bertopic_predictions_gold.csv", index=False)

        S_corpus = posting_scores(seg_df, topics, affinity, topic_index, corpus_ids)
        np.save(outdir / "bertopic_scores_corpus.npy", S_corpus)
        full = pd.DataFrame((S_corpus >= thr).astype(int), columns=CATEGORIES)
        full.insert(0, "posting_id", corpus_ids)
        full.to_csv(outdir / "bertopic_predictions_corpus.csv", index=False)

        summary.update({
            "thresholds": dict(zip(CATEGORIES, thr.round(4).tolist())),
            "test_macro_f1": float(rep_test.loc[rep_test.category == "MACRO AVG", "f1"].iloc[0]),
            "test_micro_f1": float(rep_test.loc[rep_test.category == "MICRO AVG", "f1"].iloc[0]),
            "dev_macro_f1": float(rep_dev.loc[rep_dev.category == "MACRO AVG", "f1"].iloc[0]),
            "hard_fp_accuracy": float(hard_test.loc[
                hard_test.subset.str.strip() == "lexicon false positives", "accuracy"].iloc[0]),
            "hard_fn_accuracy": float(hard_test.loc[
                hard_test.subset.str.strip() == "lexicon false negatives", "accuracy"].iloc[0]),
            "n_gold": len(gold_df), "n_dev": int(dev.sum()), "n_test": int(test.sum()),
        })
        print(f"\ntest macro-F1 {summary['test_macro_f1']:.3f} | "
              f"hard FP {summary['hard_fp_accuracy']:.3f} / "
              f"FN {summary['hard_fn_accuracy']:.3f}")
        print("compare: TF-IDF cosine 0.759 (FP 0.167 / FN 0.586) | "
              "zero-shot 0.684 (0.542 / 0.483) | LLM 0.610 (0.833 / 0.138)")

    (outdir / "bertopic_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
