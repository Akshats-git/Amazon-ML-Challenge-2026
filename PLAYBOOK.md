# Amazon ML Challenge 2026 — Team Playbook

## Key times (IST)

| When | What |
|---|---|
| **Fri 25 Sep, 00:00** | Problem statement + dataset released. The 72 hours start. |
| **Sun 27 Sep, 00:00** (48 h mark) | The top 500 teams get +$100 AWS credits, so have a strong leaderboard score by then. |
| **Sun 27 Sep, 20:00** | Our code freeze. Pick the final submission and finish the approach doc. |
| **Sun 27 Sep, 22:00** | Everything uploaded: final CSV, 1–2 page approach doc (PDF), code zip. |
| Sun 27 Sep, 23:59 | Official deadline. Don't rely on it, because portals slow down in the last hour. |
| 2 Oct / 7 Oct | Top-50 results / Grand Finale (top 10 present to Amazon scientists) |

---

## First 60 minutes (checklist)

- [ ] Download the dataset and problem PDF, then unzip into `data/raw/`. Read the rules **together**:
  - exact metric formula and edge cases
  - submission format and **daily submission cap**
  - allowed external data or models (in 2025: no external price lookup, models ≤ 8B params, MIT/Apache licence only)
  - whether the data may be uploaded outside AWS (Kaggle, Colab, Drive)
- [ ] Put the official metric into `src/amlc/metrics.py`. Check it against any sample or example they give.
- [ ] Fill the CONFIG cell in `notebooks/01_eda_baseline.py` and run it top to bottom.
- [ ] **If there are image URLs, start the download immediately in a second terminal.** It takes hours:
      `uv run python -m amlc.images data/raw/train.csv image_link data/images/train` (same for test)
- [ ] Each teammate: request the SageMaker GPU quota (see AWS section). It can take hours to approve.
- [ ] Target: first valid leaderboard submission within ~3 hours.

## Roles (2–4 people)

| Role | Owns |
|---|---|
| **Pipeline and CV owner** (team leader) | Folds, metric, submission validity, the blend, final picks, portal uploads |
| **Text modeller** | TF-IDF, text embeddings, parsing structured values out of text (quantities, units, brand, pack size…) |
| **Image / multimodal modeller** | Image download, CLIP/SigLIP embeddings, OCR or VLM if needed, cloud GPU runs |
| **Features and doc** | Hand-crafted features, error analysis on OOF predictions, running the approach doc from day 1 |

Everyone uses the same `oof/folds.npy`, saves `oof/<exp>_oof.npy` + `oof/<exp>_test.npy`, and logs every run
with `log_experiment(...)` in `experiments.csv`. Together these give us the ensemble, and the approach doc for free.

## 72-hour plan

| Block | Goal |
|---|---|
| **Day 1 (Fri)** | EDA, metric, CV, TF-IDF baseline on the leaderboard, images downloading, first embeddings + GBDT |
| **Day 2 (Sat)** | Feature engineering from error analysis, stronger models (fine-tuned small transformer on a cloud GPU, image features), first blend. **Top-500 by Sat midnight.** |
| **Day 3 (Sun)** | Seeds/folds averaging, final blend weights fitted on OOF, pick final submission, write doc, zip code. Freeze at 20:00. |

Sleep in shifts. A tired team makes submission-format mistakes.

---

## Compute plan

| Where | What it's good for |
|---|---|
| **Laptop**: GTX 1650 4 GB, 12 cores, 13 GB RAM | TF-IDF + linear, LightGBM/XGBoost/CatBoost (XGB/Cat on GPU), fp16 inference with small encoders (bge-small, LAION CLIP ViT-B/32, SigLIP-base). **Close Chrome before heavy runs.** At setup time only ~1.8 GB of RAM was free, and Chrome was using most of the rest. |
| **SageMaker notebook `ml.t3.medium`** (free tier 250 h) | Only 2 vCPU and 4 GB RAM, so it's weaker than the laptop. Use it for AWS-side data handling, not training. |
| **SageMaker/EC2 GPU** (paid from the $200 credits) | `ml.g4dn.xlarge` (T4 16 GB, roughly $0.7/h) or `ml.g5.xlarge` (A10G 24 GB, roughly $1.4/h). Use them for fine-tuning transformers, big embedding runs and VLM/OCR over images. **Stop instances when idle.** |
| Kaggle / Colab free GPUs | Only if the rules allow the data outside AWS. The Kaggle CLI is already set up on this laptop. |

Pay for the embedding or feature extraction **once**, on the biggest GPU, then share the `.npy` files through the team S3 bucket.

### Network (measured on the IIT Bhilai WiFi, 24 Sep)

| Source | Speed |
|---|---|
| General internet (Cloudflare) | ~12 MB/s |
| Hugging Face models | ~20 MB/s |
| `pypi.nvidia.com` | ~10 MB/s |
| Aliyun PyPI mirror | ~2.8 MB/s |
| **PyPI (files.pythonhosted.org)** | **~0.5 MB/s**, throttled |
| Amazon image CDN (`m.media-amazon.com`) | ~0.3 s per image. Estimate for 75k images at 64 threads: ~15–20 min. |

- Installing a new package quickly: `uv pip install --index-url https://mirrors.aliyun.com/pypi/simple <pkg>`.
  For small packages, plain `uv add <pkg>` is fine.
- **Still time the real image links in the first hour.** Download 500 images and extrapolate. If they're slow, do the image
  download and embedding on SageMaker/EC2 in us-east-1 and only pull the `.npy` features back.

## AWS checklist (each teammate, on their own Free Tier account)

> ⚠️ The default AWS CLI credentials on this laptop are the **HireLens** IAM user (`hirelens-deployer`, account 5332…).
> It has no SageMaker permissions. **Do not use it for the challenge.** Add the challenge account as a named profile instead:
> `aws configure --profile amlc` → then `export AWS_PROFILE=amlc` (or `bash scripts/aws_check.sh`).

1. [ ] Free Tier account in **us-east-1**. Note the 12-digit account ID and share it with teammates.
2. [ ] Billing → **Budgets**: a $20 monthly cost budget with email alerts at 50/80/100%. On the new Free Tier this is also one of the
       starter activities that unlock the second $100 of credits. Also do the other starter activities
       (launch EC2, create an RDS database, …) **and delete those resources straight after**.
3. [ ] Service Quotas → SageMaker → request `ml.g4dn.xlarge for notebook instance usage` and `… for training job usage`
       (and/or `ml.g5.xlarge`) = 1. **New accounts usually start at 0.**
4. [ ] SageMaker AI → Notebook instances → `ml-challenge-notebook`, `ml.t3.medium`, new IAM role. **Stop it (don't delete it) when idle.**
5. [ ] One teammate creates the team bucket `amlc26-<team>` and adds the cross-account bucket policy
       from `Context/aws-builder-center-free-tier-instructions.md`. Keep everyone in us-east-1 to avoid transfer costs.
6. [ ] Redeem any credit codes (top-500 bonus): Billing → Credits → Redeem.
7. [ ] Check Billing → Free Tier daily. **Never leave a GPU instance or an endpoint running.**

---

## Previous editions (from memory, verify details)

| Year | Task | Input | Metric | What worked |
|---|---|---|---|---|
| 2023 | Product **length** prediction | title, description, bullets, product type | `max(0, 100·(1−MAPE))` | Text features, per-product-type stats, log target |
| 2024 | **Entity value extraction** from product images (e.g. `item_weight → "10.0 gram"`) | image URL + entity name | exact-match F1 | OCR + small VLMs, strict unit normalisation, sanity checker for format |
| 2025 | **Product price** prediction | `catalog_content` text + image URL | SMAPE | Text + image embeddings → GBDT/NN on log price, parsing pack size/quantity, ensembling |

The likely pattern: messy product catalogue text plus image links, a custom metric, and an exact submission format with a sample file.
The kit in `src/amlc/` already covers each of these.

## Rules of thumb

- **Trust CV, not the public leaderboard.** The private leaderboard uses the full test set. Choose blend weights on OOF only.
- Fix the folds once (`oof/folds.npy`). Group near-duplicates (same product or text) into the same fold.
- Match the target transform to the metric: log1p for SMAPE/MAPE on skewed positive targets. For SMAPE it can also help to shrink predictions slightly after back-transforming, so tune that on OOF.
- Read 50 raw rows by hand before modelling. The structure is hidden in the text.
- Every submission goes through `write_submission`, which refuses malformed files.
- Save OOF + test predictions for **every** model, even weak ones. Blends often use them.
- Record ideas that *didn't* work too. The approach doc and the finale Q&A reward that.

## Command cheat-sheet

```bash
uv run python scripts/check_env.py --models   # full stack smoke test
uv run jupyter lab                            # or open notebooks/*.py in VS Code and run cells
uv run python -m amlc.images <csv> <url_col> <out_dir>
bash scripts/make_submission_zip.sh <team>    # final code zip
bash scripts/aws_check.sh                     # verify the challenge AWS profile
nvidia-smi -l 2                               # watch GPU memory while training
```
