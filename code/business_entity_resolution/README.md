# Business Entity Resolution: reproduction guide

Team SteinsGate. This folder rebuilds the two submission files from the official train and test TSVs:

- `output/matching_results.tsv` (the file we uploaded to the leaderboard, public score 0.989507)
- `output/candidate_pairs.tsv` (our blocking candidates, 128,969,685 pairs)

The pipeline uses no external data. Every statistic, dictionary and model is learned from the provided files.
The only pretrained weights are `FacebookAI/xlm-roberta-base` and `FacebookAI/xlm-roberta-large` (MIT license).
We fine-tune them on the training data only.

## 1. What you need

- Python 3.12 and `pip install -r requirements.txt` (or `bash code/business_entity_resolution/src/setup_box.sh` on a fresh box).
- **CPU part.** Linux with 12 or more threads.
  - Runs x1 and x2 fit in 13 GB of RAM (we ran them on a 6-core laptop).
  - Runs x3 and x4 train on all queries and need about 64 GB of RAM. We used a 32 vCPU / 128 GB VM.
  - At least 35 GB of free disk. The part-1 caches take about 16 GB and the stage-3 tables about 10 GB more.
- **GPU part.** One CUDA GPU with 40 GB of memory for the cross-encoders. We used one NVIDIA A100 40 GB.

## 2. Data layout

Run every command from one directory (the run directory) that holds the data:

```
data/raw/train/train_source1.tsv  train_source2.tsv  train_source3.tsv  train_ground_truth.tsv
data/raw/test/test_source1.tsv    test_source2.tsv   test_source3.tsv
```

The official kit calls the data folder `dataset`. Place its files under `data/raw` as shown.
Part 1 also accepts another folder through `DATA_DIR`, but the stage-3 scripts read `data/raw` and `work` only.
Caches go to `work/` and outputs go to `output/`.

## 3. How to run

There are two scripts. Run them in order from the run directory.

```bash
# Part 1 (CPU): normalization, blocking, features, stage-1 and stage-2 models (runs x1 to x4).
# It writes output/ (stage-2 blend), output_x3/ and all models and scores under work/.
bash code/business_entity_resolution/src/run_all.sh

# Part 2 (CPU + GPU): stage 3, cross-encoders, France edits, India/US rescue, final files.
# It ends by copying the final matching_results.tsv and candidate_pairs.tsv into output/.
bash code/business_entity_resolution/src/stage3/run_stage3.sh
```

On a 13 GB laptop start part 1 with `N_JOBS=8 BLOCK_THREADS=4`. Use `START=<step>` to resume `run_all.sh` from a step.

## 4. Part 1 in detail

`src/run_all.sh` runs these commands. `R="python code/business_entity_resolution/src/run.py"`.

```bash
$R env                    # versions, cores, RAM
$R check1                 # metric unit tests (0.714 / 1.0 / 0.05585)
$R check2                 # row counts and ground-truth facts (no cross-country pairs, at most one S1 per query)
rm -f work/dicts/*.json && $R norm   # normalization, first pass (no dictionaries yet)
$R dicts                  # learn transliteration dictionaries from train true pairs
$R norm                   # normalization again, with the dictionaries
$R pools                  # two validation pools per country (md5 half split, all unmatched queries, test-like density)
$R block                  # TF-IDF blocking for P0, P1 and test (top 10 per query) and a recall report
$R feats                  # 50 stage-1 pair features
$R check8                 # feature sanity checks
$R xfeats                 # 14 name-edit and unmatched-number features
$R nfeats                 # 16 number-relation and number-oracle-lexicon features

export USE_XFEATS=1
# run x1: 15% of each pool's queries per model
export TAG=x1 TRAIN_QUERY_FRAC=0.15
$R train1 && $R train2 && $R tune       # stage-1 cross-fit, stage-2 cross-fit, (T1, T2) on pooled out-of-fold scores
OUTPUT_DIR=output_x1 $R predict --check-ids
# run x2: 30% of the queries, float16 training matrix
export TAG=x2 TRAIN_QUERY_FRAC=0.30 X16=1 WRITE_CANDIDATES=0
$R train1 && $R train2 && $R tune
OUTPUT_DIR=output_x2 $R predict --check-ids
OUTPUT_DIR=output_x1x2 $R blend --tags x1 x2 --check-ids   # x1+x2 blend, thresholds (0.52, 0.74); used for France
# run x3: + the 16 number features, all queries of each pool (about 58.5M pairs per model)
export TAG=x3 USE_NFEATS=1 TRAIN_QUERY_FRAC=1.0 MASK_LEX_FRAC=0.5 X16=0
$R train1 && $R train2 && $R tune
OUTPUT_DIR=output_x3 $R predict --check-ids
# run x4: same data and features; 255 leaves, min_child 200, feature_fraction 0.6, lambda_l2 5, seed 43
export TAG=x4 LGB1_PARAMS='{"num_leaves": 255, "min_child_samples": 200, "feature_fraction": 0.6, "lambda_l2": 5.0, "seed": 43}' \
       LGB2_PARAMS='{"seed": 43}'
$R train1 && $R train2 && $R tune
OUTPUT_DIR=output_x4 $R predict --check-ids
unset LGB1_PARAMS LGB2_PARAMS
# stage-2 result: mean of x3 and x4 for US and India, thresholds (0.56, 0.74).
# France is not in train. It keeps the x1+x2 blend and its thresholds, which scored better there on the leaderboard.
WRITE_CANDIDATES=1 OUTPUT_DIR=output BLEND_TAGS_UNSEEN=x1,x2 $R blend --tags x3 x4 --check-ids
```

### Settings (environment variables)

| variable | default | meaning |
|---|---|---|
| `DATA_DIR` | `data/raw` | folder with `train/` and `test/` |
| `WORK_DIR` | `work` | caches: `norm/ dicts/ pools/ cands/ feats/ models/ oof/` |
| `OUTPUT_DIR` | `output` | where the two submission files are written |
| `N_JOBS` | all cores | worker processes and threads |
| `BLOCK_THREADS` | min(N_JOBS, 4) | threads for the sparse top-n product (3 or 4 is fastest) |
| `TRAIN_QUERY_FRAC` | `0.35` | share of each pool's queries used to fit a model |
| `DEV_FRAC` | `0` | above 0 means dev mode: only this share of queries, written under `WORK_DIR/dev` |
| `TAG` | `main` | name of a model run; models, scores and thresholds go to `WORK_DIR/{models,oof}/TAG` |
| `USE_XFEATS` | `0` | 1 adds the 14 name-edit features |
| `USE_NFEATS` | `0` | 1 adds the 16 number features |
| `MASK_LEX_FRAC` | `0.3` | share of training queries whose lexicon features are blanked (dropout for unseen countries) |
| `X16` | `0` | 1 holds the training matrix as float16 and feeds LightGBM through `lgb.Sequence` |
| `WRITE_CANDIDATES` | `1` | 0 skips `candidate_pairs.tsv` (it is the same for every run) |
| `LGB1_PARAMS`, `LGB2_PARAMS` | `{}` | JSON overrides of the stage-1 and stage-2 LightGBM parameters |
| `CAPS` | `1` | at most 5 Source-2 and 6 Source-3 links per Source-1 record (highest scores kept) |
| `BLEND_TAGS_UNSEEN` | unset | `blend`: runs used for test countries absent from train (for example `x1,x2`) |
| `T_UNSEEN` | unset | `T1,T2` for test countries absent from train |
| `FR_NORM`, `FR_IMPUTE`, `FR_RESCUE` | `0` | old France rules (R07 to R09). They lowered the leaderboard score and are off in the final. |

## 5. Part 2 in detail

The scripts are in `src/stage3/`. `run_stage3.sh` runs them in this order. Tables go to `work/stage3/` (`S3_DIR`) and final-day tables to `work/stage3/r20/` (`R20_DIR`).

1. **Argmax tables** (`tables.py`, `tables_top2.py`). One row per query with its best Source-1 candidate under the stage-2 blend. The top-2 tables add the runner-up candidate. US and India use x3+x4. France and the US/India pool copies used for the France model use x1+x2.
2. **Stage-3 features** (`top2.py build`, `stage3.py build`):
   - consensus with the other queries that choose the same Source-1 record (count, sum of scores, shared house number, name or address, number relation to their most common number);
   - raw-text marks that normalization removes (tripled letters, accents, house-number suffixes, repeated words, brackets);
   - the scores of runs x1 to x4 and the features of the competing row.
3. **Cross-encoder inputs** (`build_ce.py`, `build_ce2.py`, `ce_rows_top2.py`). Raw "name | address" pairs of the uncertain best rows (0.003 < p < 0.997) plus 4% of the other rows, and the plausible runner-up rows (p >= 0.01).
4. **Cross-encoders** (`ce_train.py`, GPU). Each model is trained on one pool and scores the other pool and test:
   - `FacebookAI/xlm-roberta-base`: 1 epoch, batch 128, learning rate 3e-5;
   - `FacebookAI/xlm-roberta-large`: 1 epoch, batch 64, learning rate 1e-5; it also scores the runner-up rows.
5. **Top-2 re-rank** (`add_ce.py`, `top2.py fit`). A LightGBM on each query's best and runner-up rows, cross-fitted between the pools on US and India together. The higher-scoring row is the link candidate. Thresholds (0.58, 0.78). Out-of-fold macro F0.5 0.99165 (US 0.99206, India 0.99106).
6. **France edits** (`fr_classes.py`, `fr_rules.py a,c,d,e`). The x1+x2 France decisions with four class-level edits:
   - (a) drop same-address links with a swapped category word that no other record of the entity carries;
   - (c) add same-address acronym pairs;
   - (d) drop links whose house number is a generator shift from both the entity and its other records;
   - (e) add x3's same-address links that differ by a common French noise word when the large cross-encoder agrees (> 0.72).
7. **Write** (`compose.py`). US and India from the re-rank, France from the edited pairs. This is output `s8b` (leaderboard 0.987777).
8. **Final-day steps (R20)**:
   - **a. Rescue candidates for India and US** (`r20_block2.py`, `r20_block3.py`, `r20_rescue_feats.py`, `r20_rescue_train.py`). For unlinked queries that have an address: an address-heavy TF-IDF pass (name 0.1, address 0.8, characters 0.1; top 10 new pairs) and a name-twin search (same core name, ranked by shared rare address tokens; top 10 new pairs). A LightGBM with 28 features scores them.
   - **b. Rescue cross-encoder inputs** (`r20_build_rce.py`). The top 4 candidates of each query whose best rescue score is at least 0.03.
   - **c. Rescue cross-encoder** (`r20_rce_train.py`, GPU). xlm-roberta-large, 3 passes over one pool's candidates. It scores the other pool and test.
   - **d. Rescue v4** (`r20_rescue4.py`). The 28 features plus the cross-encoder score, its rank, gap and margin within the query. It links the best new candidate when the score reaches a threshold tuned on out-of-fold scores (0.70 for India and 0.85 for US in the submitted run). Out-of-fold macro F0.5: India 0.99107 to 0.99262, US 0.99210 to 0.99237.
   - **e. France precision removals** (`r20_fr_table.py`, `r20_cells.py`, `r20_initl.py`, `r20_fr_transfer.py`, `r20_fr_veto.py`, `r20_merge_fr.py`, `stage3.py fit` with `FPFX=f12ce TEST_C=France OUT=pfr`, `r20_fr_precision.py`). A France stage-3 model is trained on the US/India pools with France-safe features. We drop s8b France links that this model drops and the large cross-encoder rejects (< 0.3), except the classes the leaderboard showed to be true. A cross-encoder address veto is added. This removes 3,164 links. The structural-transfer package of `r20_fr_transfer.py` is built only as an input to the veto. It is not applied.
   - **Patch** (`r20_patch_fr.py`, `r20_patch_add.py`). France removals, then India and US rescue links. The result is `output_r20/r20l/` (leaderboard 0.989472).
   - **f. France additions (final)** (`r20_fr_final.py`, `r20_patch_fr.py`). Add unlinked France pairs where the France stage-3 model and the large cross-encoder both score at least 0.95, outside the category, noise-word and generator-word classes, generator shifts and the transfer cells. Remove 74 more links of the step-e pattern. Result: +2,371 and -74 France links in `output_r20/final/` (leaderboard 0.989507).
   - **g. Candidate file** (`r20_regen_cands.py`). `candidate_pairs.tsv` is the original top-10 candidates plus every pair the rescue model scored: 128,969,685 pairs. It checks that every match is a candidate.
   - **h.** Copy both files into `output/`.

## 6. Checks

- `predict` and `blend` run the official validator on `matching_results.tsv` with `--check-ids`.
- `candidate_pairs.tsv` is stream-checked with the same rules: one row per Source-1 record in file order, only S2 and S3 ids, no duplicates, and every match inside its row's candidates. (The official validator's candidate check needs about 10 GB of RAM for 129M ids.)
- **Rerun check (28 September 2026).** We reran every CPU step of part 2 on the VM from the stored part-1 outputs and cross-encoder scores.
  - The candidate file, the stage-3 feature tables and the France tables came out byte for byte the same.
  - The top-2 re-rank fit gave the same thresholds and out-of-fold score. Its scores differ by less than 1e-8.
  - The France model fit and the France removals and additions came out exactly the same.
  - The India and US rescue models are not bit-exact. The summed IDF features differ by about 1e-5 between runs, and the holdout draw uses an unordered `unique()`. The rerun reached the same out-of-fold gain (India +0.00154 vs +0.00156, US +0.00027 vs +0.00028). 97% of the India additions and 99% of the US additions were the same. The final file differed from the uploaded one in 1,077 of 5,838,881 links (0.02%).
  - With the stored rescue outputs, steps 8e to 8f rebuild the uploaded R20l and R20p files byte for byte.
  - GPU fine-tuning of the cross-encoders is not bit-exact either. We did not rerun it.
- LightGBM results depend on the thread count. Keep the thread counts set in the scripts (16 in `stage3.py` and `top2.py`).

## 7. Runtime

- 6-core / 13 GB laptop: normalization and dictionaries about 20 min, blocking about 1.6 h, features about 45 min, side features about 26 min, runs x1 and x2 about 2 to 3.5 h each.
- 32 vCPU / 128 GB VM: number features 19 min, run x3 about 2.5 h, run x4 about 2.1 h, blend about 3 min, stage-3 tables and fits about 20 min.
- One A100 40 GB: base cross-encoder about 35 min, large about 110 min, rescue cross-encoder about 21 min per fold (both folds in parallel, scoring included).

## 8. Code map

| path | role |
|---|---|
| `src/run.py` | command-line entry point for part 1 |
| `src/ber/normalize.py`, `src/ber/translit.py` | normalization and transliteration dictionaries |
| `src/ber/pools.py`, `src/ber/blocking.py` | validation pools and TF-IDF blocking |
| `src/ber/features.py`, `src/ber/xfeats.py`, `src/ber/nfeats.py` | 50 base, 14 edit and 16 number features |
| `src/ber/model.py`, `src/ber/context.py` | two-stage LightGBM and stage-2 context |
| `src/ber/decide.py`, `src/ber/metrics.py` | per-entity decision rule, caps, macro F0.5 |
| `src/ber/submit.py`, `src/validate_submission.py` | writing, validation and packaging |
| `src/ber/rescue.py` | old France same-address rescue (off) |
| `src/run_all.sh`, `src/setup_box.sh` | part 1 driver and one-time environment setup |
| `src/stage3/` | stage 3, cross-encoders, France edits and final-day steps (`run_stage3.sh`) |
| `src/stage3/r20_rescue_apply.py`, `src/stage3/r20_candchk.py` | helpers not called by `run_stage3.sh`: the older v2 rescue decision and a France candidate check |
