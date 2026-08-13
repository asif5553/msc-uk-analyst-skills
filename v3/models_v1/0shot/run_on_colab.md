# Running the pipeline on Google Colab

## Zero-shot (method 2) — needs a GPU

**1. Set the runtime to GPU**
Runtime → Change runtime type → Hardware accelerator: **T4 GPU** → Save.

**2. Upload the files**
```python
from google.colab import files
files.upload()
# select: config.py, evaluate.py, zeroshot.py,
#         uk_analyst_corpus_v4_clean.csv,
#         gold_standard_annotation_workbook_v2.xlsx
```

**3. Install / check dependencies**
Colab already has torch and transformers, but pin a recent transformers to be safe:
```python
!pip install -q "transformers>=4.40" openpyxl
import torch; print("GPU:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
```

**4. Run on the 300 gold postings**
```python
!python zeroshot.py \
  --corpus uk_analyst_corpus_v4_clean.csv \
  --gold gold_standard_annotation_workbook_v2.xlsx \
  --outdir results
```
Roughly 8,200 forward passes (300 postings x ~2.1 chunks x 13 hypotheses).
Expect ~5-12 minutes on a T4. The model downloads once (~1.6 GB) on first run.

**5. Once you are happy with the result, score the whole corpus**
Needed later for the comparative market analysis.
```python
!python zeroshot.py \
  --corpus uk_analyst_corpus_v4_clean.csv \
  --gold gold_standard_annotation_workbook_v2.xlsx \
  --outdir results --full-corpus
```
~22,100 forward passes; roughly 20-35 minutes on a T4.

**6. Download the results**
```python
!zip -r zeroshot_results.zip results
files.download('zeroshot_results.zip')
```

## Notes

- If you hit a CUDA out-of-memory error, lower the batch size: `--batch-size 8`.
- To try a different model: `--model MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`
  (smaller and faster than bart-large-mnli; worth reporting both if time allows,
  since model choice is exactly what Kyritsis et al. compare).
- Raw scores are saved as .npy files, so thresholds can be re-tuned later without
  re-running the model.
- Uploaded files are lost when the runtime restarts — re-upload after any restart.

## TF-IDF (method 1) — CPU is fine

```python
!python tfidf_baseline.py \
  --corpus uk_analyst_corpus_v4_clean.csv \
  --gold gold_standard_annotation_workbook_v2.xlsx \
  --outdir results
```
Runs in under 10 seconds. Needs config.py and evaluate.py in the same folder.
