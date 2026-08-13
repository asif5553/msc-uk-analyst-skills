"""
Method 3 of 4: LLM-based structured prompting.

Follows the protocol of Wiest et al. (2024), whose LLM-AIx pipeline extracts structured
fields from unstructured clinical text using locally-hosted, privacy-preserving large
language models with schema-constrained JSON output. The same design is applied here to
job postings: the model receives the posting and a task specification, and returns a
JSON object with one binary value per skill category.

Design decisions, and why:

  Local by default. Wiest et al.'s central claim concerns privacy-preserving local
  inference, so an open-weights model running on the researcher's own hardware is the
  faithful reproduction. It is also reproducible (fixed weights, temperature 0) and
  free. A hosted-API backend is provided for comparison.

  Rules in the prompt, lexicons NOT in the prompt. The prompt states the annotation
  rules a human annotator worked from - explicit mention only, ignore company and team
  descriptions, ignore qualification lists, ignore recruiter footers - but never
  supplies the lexicon term lists. Supplying them would reproduce the circularity that
  makes the lexical TF-IDF variant an upper bound rather than a method, and would make
  the hard-subset result meaningless.

  Two prompt variants (--prompt minimal|full) support an ablation on how much task
  specification the model needs, which is the "structured prompting" variable the
  research question asks about.

  No chunking. Modern context windows exceed the longest posting in the corpus
  (~3,300 tokens), so the full text is passed intact, unlike the zero-shot method.

Runs are checkpointed after every posting, so an interrupted Colab session resumes
rather than restarting.

Usage:
    python llm_prompting.py --corpus <csv> --gold <xlsx> --outdir <dir>
    python llm_prompting.py ... --backend api --model claude-sonnet-4-5
    python llm_prompting.py ... --full-corpus
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from config import CATEGORIES, CATEGORY_LABELS
from evaluate import (load_gold, make_split, evaluate, format_report,
                      lexicon_presence, evaluate_hard, format_hard_report)

DEFAULT_LOCAL_MODEL = "Qwen/Qwen2.5-7B-Instruct"
MAX_CHARS = 18000  # defensive truncation; longest corpus posting is ~17,200 chars

CATEGORY_DEFINITIONS = {
    "programming": "a general-purpose programming or scripting language, or macro development",
    "sql": "querying or working with data held in databases",
    "visualisation_bi": "building charts, dashboards or visual representations of data, or a named visualisation/BI tool",
    "reporting": "producing, maintaining or distributing reports or management information",
    "excel": "spreadsheet software",
    "statistics": "statistical methods, forecasting, or named statistical software",
    "machine_learning": "building or applying machine learning, predictive or AI models",
    "data_cleaning": "cleansing, validating or quality-assuring raw data",
    "etl": "data pipelines, integration, warehousing or big-data tooling",
    "data_modelling": "designing the structure of data: schemas, dimensional models, entity relationships",
    "cloud": "a named cloud computing platform or service",
    "stakeholder_comm": "communicating with stakeholders or non-technical audiences, or presenting findings",
    "ethics_governance": "data protection, privacy, governance, or data-context regulatory compliance",
}

RULES = """Apply these rules strictly:
1. EXPLICIT MENTIONS ONLY. Tag a category 1 only if the posting explicitly names the skill or a tool belonging to it. Never infer a skill from the role type, seniority, or industry. "You will work with large datasets" names no skill and receives no tags.
2. Desirable and "nice to have" skills COUNT. The posting is signalling demand.
3. IGNORE company, team and department descriptions. "The team performs regulatory reporting" or "our Data Science team harnesses big data" describes the organisation, not the candidate.
4. IGNORE qualification and degree lists. "2:1 in Maths or Statistics" is an entry requirement, not a skill mention.
5. IGNORE recruitment-agency boilerplate: keyword footers listing technologies, agency privacy policies, and agency self-promotion.
6. Distinguish organisational structure from the reporting skill. "Reporting to the Head of Data" and "direct reports" are not reporting.
7. Watch for words that resemble a skill but are not: "excellent" is not Excel; "clinical coding" is not programming; a "demand pipeline" or "drug pipeline" is not a data pipeline; Salesforce "Sales Cloud" is not a cloud platform; "financial modelling" and "business modelling" are not data modelling.
8. A lexicon word naming a colleague or another team is not a skill: "the Data Warehouse Manager", "working with statisticians"."""


def build_prompt(posting_text, variant="full"):
    cats = "\n".join(f'  "{c}": {CATEGORY_DEFINITIONS[c]}' for c in CATEGORIES)
    rules = f"\n{RULES}\n" if variant == "full" else "\n"
    return f"""You are annotating UK job postings to identify which technical and professional skills each posting requires.

For the posting below, decide for each of these 13 categories whether the posting explicitly mentions that skill:

{cats}
{rules}
Return ONLY a JSON object with all 13 keys and values 0 or 1. No explanation, no markdown fences.

JOB POSTING:
\"\"\"
{posting_text[:MAX_CHARS]}
\"\"\"

JSON:"""


def parse_response(text):
    """Extract the JSON object and coerce to a 13-vector. Returns None if unusable."""
    if not text:
        return None
    text = re.sub(r"```(?:json)?", " ", text)
    m = re.search(r"\{.*?\}", text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    out = np.zeros(len(CATEGORIES), dtype=int)
    for i, c in enumerate(CATEGORIES):
        v = obj.get(c, 0)
        if isinstance(v, str):
            v = v.strip().lower()
            v = 1 if v in ("1", "true", "yes") else 0
        out[i] = 1 if v in (1, True) else 0
    return out


# --------------------------------------------------------------------------
# Backends
# --------------------------------------------------------------------------

class LocalBackend:
    """Open-weights model via transformers. 4-bit quantised so a 7B fits a T4."""

    def __init__(self, model_name=DEFAULT_LOCAL_MODEL, load_in_4bit=True):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        kwargs = {"device_map": "auto", "dtype": torch.float16}
        if load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                )
            except Exception as e:  # noqa: BLE001
                print(f"  4-bit unavailable ({e}); loading in fp16")
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        self.model.eval()

    def generate(self, prompt, max_new_tokens=200):
        import torch
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        enc = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(out[0][enc.input_ids.shape[1]:],
                                     skip_special_tokens=True)


class AnthropicBackend:
    def __init__(self, model_name="claude-sonnet-4-5"):
        import anthropic
        self.name = model_name
        self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    def generate(self, prompt, max_new_tokens=200):
        r = self.client.messages.create(
            model=self.name, max_tokens=max_new_tokens, temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in r.content if b.type == "text")


class OpenAIBackend:
    def __init__(self, model_name="gpt-4o-mini"):
        from openai import OpenAI
        self.name = model_name
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def generate(self, prompt, max_new_tokens=200):
        r = self.client.chat.completions.create(
            model=self.name, max_tokens=max_new_tokens, temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        return r.choices[0].message.content


def make_backend(kind, model):
    if kind == "local":
        return LocalBackend(model or DEFAULT_LOCAL_MODEL)
    if kind == "anthropic":
        return AnthropicBackend(model or "claude-sonnet-4-5")
    if kind == "openai":
        return OpenAIBackend(model or "gpt-4o-mini")
    raise ValueError(f"unknown backend: {kind}")


# --------------------------------------------------------------------------

def run_extraction(backend, ids, texts, cache_path, variant, verbose=True):
    """Score postings, checkpointing after each so an interrupted run resumes."""
    cache = {}
    if cache_path.exists():
        cache = {r["posting_id"]: r for r in
                 (json.loads(l) for l in cache_path.read_text().splitlines() if l.strip())}
        if verbose and cache:
            print(f"  resuming: {len(cache)} postings already scored")

    n_fail, t0 = 0, time.time()
    with cache_path.open("a") as fh:
        for i, (pid, text) in enumerate(zip(ids, texts)):
            if pid in cache:
                continue
            prompt = build_prompt(text, variant)
            raw, vec = "", None
            for attempt in range(2):  # one retry on unparseable output
                try:
                    raw = backend.generate(prompt)
                    vec = parse_response(raw)
                    if vec is not None:
                        break
                except Exception as e:  # noqa: BLE001
                    raw = f"ERROR: {e}"
                    time.sleep(2)
            if vec is None:
                vec = np.zeros(len(CATEGORIES), dtype=int)
                n_fail += 1
            rec = {"posting_id": pid, "labels": vec.tolist(), "raw": raw[:500]}
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            cache[pid] = rec
            if verbose and (i + 1) % 10 == 0:
                el = time.time() - t0
                done = i + 1
                print(f"  {done}/{len(ids)} ({el:.0f}s, {el / max(done,1):.1f}s/posting, "
                      f"{n_fail} parse failures)", flush=True)

    M = np.array([cache[p]["labels"] for p in ids])
    return M, n_fail, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--backend", default="local", choices=["local", "anthropic", "openai"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--prompt", default="full", choices=["minimal", "full"])
    ap.add_argument("--full-corpus", action="store_true")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.backend}_{args.prompt}"

    corpus = pd.read_csv(args.corpus)
    gold_df, y = load_gold(args.gold)
    text_by_id = dict(zip(corpus["posting_id"], corpus["job_summary"].fillna("")))
    gold_ids = gold_df["posting_id"].tolist()
    gold_texts = [text_by_id[p] for p in gold_ids]

    dev, test = make_split(gold_df)
    lex = lexicon_presence(gold_texts)
    print(f"corpus {len(corpus)} | gold {len(gold_df)} (dev {dev.sum()}, test {test.sum()})")
    print(f"backend={args.backend} prompt={args.prompt}")

    backend = make_backend(args.backend, args.model)
    print(f"model: {backend.name}")

    pred, n_fail, secs = run_extraction(
        backend, gold_ids, gold_texts, outdir / f"llm_{tag}_raw_gold.jsonl", args.prompt
    )
    print(f"scored {len(gold_ids)} postings in {secs:.0f}s "
          f"({secs / max(len(gold_ids),1):.1f}s/posting, {n_fail} parse failures)")

    # No thresholds to tune: the model emits binary labels directly. Both splits are
    # reported, but the test split remains the comparable figure across methods.
    rep_test = evaluate(y[test], pred[test])
    rep_dev = evaluate(y[dev], pred[dev])
    hard_test = evaluate_hard(y[test], pred[test], lex[test])
    print(format_report(rep_test, f"LLM {tag} - TEST split (n={test.sum()})"))
    print(format_hard_report(hard_test, f"LLM {tag} - hard subset (TEST)"))

    rep_test.to_csv(outdir / f"llm_{tag}_test.csv", index=False)
    rep_dev.to_csv(outdir / f"llm_{tag}_dev.csv", index=False)
    hard_test.to_csv(outdir / f"llm_{tag}_hard_test.csv", index=False)

    out = pd.DataFrame(pred, columns=CATEGORIES)
    out.insert(0, "posting_id", gold_ids)
    out.insert(1, "split", np.where(dev, "dev", "test"))
    out.to_csv(outdir / f"llm_{tag}_predictions_gold.csv", index=False)

    summary = {
        "method": "llm_structured_prompting",
        "backend": args.backend, "model": backend.name, "prompt_variant": args.prompt,
        "lexicons_in_prompt": False,
        "test_macro_f1": float(rep_test.loc[rep_test.category == "MACRO AVG", "f1"].iloc[0]),
        "test_micro_f1": float(rep_test.loc[rep_test.category == "MICRO AVG", "f1"].iloc[0]),
        "dev_macro_f1": float(rep_dev.loc[rep_dev.category == "MACRO AVG", "f1"].iloc[0]),
        "hard_accuracy": float(hard_test.loc[hard_test.subset == "hard subset (all)", "accuracy"].iloc[0]),
        "parse_failures": n_fail,
        "seconds_gold": secs, "seconds_per_posting": secs / max(len(gold_ids), 1),
        "n_gold": len(gold_df), "n_dev": int(dev.sum()), "n_test": int(test.sum()),
    }

    if args.full_corpus:
        print(f"\nscoring all {len(corpus)} corpus postings...")
        full_pred, n_fail_f, secs_f = run_extraction(
            backend, corpus["posting_id"].tolist(),
            corpus["job_summary"].fillna("").tolist(),
            outdir / f"llm_{tag}_raw_corpus.jsonl", args.prompt,
        )
        fp = pd.DataFrame(full_pred, columns=CATEGORIES)
        fp.insert(0, "posting_id", corpus["posting_id"].values)
        fp.to_csv(outdir / f"llm_{tag}_predictions_corpus.csv", index=False)
        summary["seconds_corpus"] = secs_f
        summary["parse_failures_corpus"] = n_fail_f

    (outdir / f"llm_{tag}_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\ntest macro-F1 {summary['test_macro_f1']:.3f} | "
          f"hard-subset accuracy {summary['hard_accuracy']:.3f}")
    print("compare: TF-IDF cosine 0.759 / hard 0.396 | zero-shot 0.684 / hard 0.509 | "
          "lexical ceiling 0.937 / hard 0.000")


if __name__ == "__main__":
    main()
