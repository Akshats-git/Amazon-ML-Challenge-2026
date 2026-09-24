# Amazon ML Challenge 2026

This is the team workspace for the 72-hour hackathon, running **25 Sep 00:00 → 27 Sep 23:59 IST**.
Start with **[PLAYBOOK.md](PLAYBOOK.md)**, which covers the timeline, the first-hour checklist, roles, the compute plan and the AWS checklist.

## Setup

```bash
# needs uv (https://docs.astral.sh/uv/) - Python 3.12, CUDA 12.6 PyTorch wheels
uv sync
uv run python scripts/check_env.py --models   # should end with "ALL GOOD"
```

Open `notebooks/*.py` in VS Code and use "Run Cell". The interpreter is `.venv`, and there's also a Jupyter kernel named **Python (amlc)**.

On a SageMaker notebook instance or Kaggle, clone or upload the repo and run `pip install -e . --no-deps` into an env that already
has torch. If needed, set `AMLC_ROOT` to the repo path.

## Layout

```
Context/                  challenge info (official details, prep notes, AWS guide)
PLAYBOOK.md               how we run the 72 hours
docs/approach_template.md 1-2 page approach doc skeleton (submission requirement)
src/amlc/
  config.py    paths (data/raw, data/features, oof/, submissions/ …), seed, n_folds
  metrics.py   smape, mape_score (2023), entity_f1 (2024), rmse, f1 … + registry
  cv.py        make_folds (stratified/grouped) + train_gbdt_cv (lgb/xgb/cat, OOF + test preds)
  embed.py     cached text (sentence-transformers) and image (CLIP/SigLIP) embeddings
  images.py    parallel, resumable image downloader + corrupt-file finder
  submit.py    write_submission: validates against the official sample before writing
  utils.py     seeding, timer, reduce_mem, log_experiment -> experiments.csv
notebooks/01_eda_baseline.py  EDA → metric sanity → folds → TF-IDF baseline → embeddings+LGBM → blend
scripts/
  check_env.py             stack smoke test
  make_submission_zip.sh   final code zip (excludes data/models)
  aws_check.sh             verify the challenge AWS profile, quotas, running endpoints
data/ models/ oof/ submissions/   git-ignored
```

## Conventions

- Dataset → `data/raw/`. Never commit data. Check the rules before putting it anywhere outside AWS.
- One set of folds for the whole team: `oof/folds.npy`.
- Every experiment saves `oof/<name>_oof.npy` and `oof/<name>_test.npy` and calls `log_experiment(...)`.
- Submissions go only through `amlc.submit.write_submission`.
