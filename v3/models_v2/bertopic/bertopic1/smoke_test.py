"""Smoke test: runs bertopic_model end to end on synthetic fixtures.

Substitutes a deterministic hashing encoder for SentenceTransformer so the test needs
no GPU and no model download. Everything else - segmentation, BERTopic fit, topic
mapping, posting aggregation, threshold tuning, evaluation wiring - is the real code.
"""
import sys, json, random
from pathlib import Path
import numpy as np
import pandas as pd

sys.argv = ["x"]
import bertopic_model as bm
from config import CATEGORIES, CATEGORY_LABELS

random.seed(0); np.random.seed(0)

# ---- deterministic stub encoder -------------------------------------------
class StubEncoder:
    """Bag-of-hashed-words embedding: same text -> same vector, similar text -> similar."""
    D = 96
    def _vec(self, text):
        v = np.zeros(self.D)
        for w in str(text).lower().split():
            v[hash(w) % self.D] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v
    def encode(self, texts, **kw):
        if isinstance(texts, str):
            texts = [texts]
        return np.vstack([self._vec(t) for t in texts]).astype(np.float32)

bm.load_encoder = lambda name: StubEncoder()

# ---- synthetic corpus ------------------------------------------------------
THEMES = {
    "programming": "write python scripts and develop code in the analytics stack",
    "sql": "query the sql server database and write complex sql joins",
    "visualisation_bi": "build power bi dashboards and tableau visualisation reports",
    "reporting": "produce weekly management information reporting packs for the board",
    "excel": "advanced excel spreadsheet modelling with pivot tables and vlookups",
    "statistics": "apply statistical analysis regression and forecasting techniques",
    "machine_learning": "develop machine learning predictive models for the business",
    "data_cleaning": "cleanse validate and quality assure raw data feeds",
    "etl": "maintain etl data pipelines and the data warehouse ingestion layer",
    "data_modelling": "design dimensional data models schemas and star schema structures",
    "cloud": "work with azure aws cloud platform services and snowflake",
    "stakeholder_comm": "present findings and communicate with non technical stakeholders",
    "ethics_governance": "ensure gdpr data protection and data governance compliance",
}
FILLER = ["we offer a competitive salary and generous holiday allowance",
          "the successful candidate will join a friendly and growing team",
          "our company has been established for over twenty five years",
          "please apply through the link below with your current cv"]

rows, gold_rows = [], []
role_families = ["data_analyst", "business_analyst", "finance_analyst"]
for i in range(240):
    pid = f"P{i:04d}"
    k = random.randint(2, 5)
    cats = random.sample(list(THEMES), k)
    lines = [f"{random.choice(role_families).replace('_',' ').title()} vacancy in Manchester"]
    for c in cats:
        # repeat each theme sentence a few times across the corpus so topics can form
        lines.append(THEMES[c] + " " + random.choice(["", "on a daily basis",
                                                      "as part of this role"]))
    lines += random.sample(FILLER, 2)
    random.shuffle(lines)
    rows.append({"posting_id": pid, "job_summary": "\n".join(lines),
                 "job_title": "Analyst", "role_family_provisional": random.choice(role_families)})
    gold_rows.append({"posting_id": pid, "job_title": "Analyst",
                      "role_family": random.choice(role_families),
                      **{c: int(c in cats) for c in CATEGORIES}})

out = Path("smoke_out"); out.mkdir(exist_ok=True)
pd.DataFrame(rows).to_csv(out / "corpus.csv", index=False)
with pd.ExcelWriter(out / "gold.xlsx") as xw:
    pd.DataFrame(gold_rows).to_excel(xw, sheet_name="Annotation", index=False)

# ---- run -------------------------------------------------------------------
sys.argv = ["bertopic_model.py", "--corpus", str(out / "corpus.csv"),
            "--gold", str(out / "gold.xlsx"), "--outdir", str(out / "results"),
            "--min-topic-size", "15", "--reduce-outliers"]
bm.main()

# ---- assertions ------------------------------------------------------------
r = out / "results"
expected = ["bertopic_topics.csv", "bertopic_topic_category_affinity.csv",
            "bertopic_segment_topics.csv", "bertopic_coverage.json",
            "bertopic_test.csv", "bertopic_dev.csv", "bertopic_hard_test.csv",
            "bertopic_predictions_gold.csv", "bertopic_predictions_corpus.csv",
            "bertopic_scores_corpus.npy", "bertopic_summary.json"]
missing = [f for f in expected if not (r / f).exists()]
assert not missing, f"missing outputs: {missing}"

pred = pd.read_csv(r / "bertopic_predictions_gold.csv")
assert list(pred.columns) == ["posting_id", "split", *CATEGORIES], pred.columns.tolist()
assert set(pred.split) == {"dev", "test"}
assert pred[CATEGORIES].isin([0, 1]).all().all()
labels_per_posting = pred[CATEGORIES].sum(axis=1)
assert labels_per_posting.max() > 1, "multi-label output failed: max 1 label per posting"
corp = pd.read_csv(r / "bertopic_predictions_corpus.csv")
assert len(corp) == 240 and list(corp.columns) == ["posting_id", *CATEGORIES]
s = json.loads((r / "bertopic_summary.json").read_text())
assert 0.0 <= s["test_macro_f1"] <= 1.0

print("\n" + "=" * 60)
print("SMOKE TEST PASSED")
print(f"  segments             {s['n_segments']}")
print(f"  topics               {s['n_topics']}  (outliers {s['outlier_share']:.1%})")
print(f"  labels per posting   mean {labels_per_posting.mean():.2f}, "
      f"max {labels_per_posting.max()}")
print(f"  test macro-F1        {s['test_macro_f1']:.3f}  (stub encoder - not meaningful)")
print("=" * 60)
