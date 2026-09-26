# Business Entity Resolution: reproduction guide

This regenerates `output/matching_results.tsv` and `output/candidate_pairs.tsv` from the train and test TSVs.
The pipeline uses no external data: every statistic (IDF, transliteration dictionaries, models) is learned from the provided files.

## Environment

- Python 3.12, `pip install -r requirements.txt`
- Full run: any Linux box with 12+ threads and 13+ GB RAM and ~16 GB free disk (it ran end to end on a 6-core / 13 GB laptop,
  about 4–5 h; a 16 vCPU / 64 GB box is faster). Every step streams: feature files hold 2M pairs each, predictions are
  float32 `.npy` arrays aligned with them, and no step loads a whole 40M-pair partition as a data frame.

## Layout the commands expect

Run every command from the directory that holds the data. Override paths with env vars:

| env | default | meaning |
|---|---|---|
| `DATA_DIR` | `data/raw` | holds `train/train_source{1,2,3}.tsv`, `train/train_ground_truth.tsv`, `test/test_source{1,2,3}.tsv` (the official kit uses `dataset`) |
| `WORK_DIR` | `work` | caches: `norm/ dicts/ pools/ cands/ feats/ models/ oof/` |
| `OUTPUT_DIR` | `output` | the two submission files |
| `N_JOBS` | all cores | worker processes / threads (8 on a 13 GB laptop) |
| `BLOCK_THREADS` | min(N_JOBS, 4) | threads of the sparse top-n matmul (cache-bound: 3–4 beat 8 on a 6-core laptop) |
| `TRAIN_QUERY_FRAC` | `0.35` | share of each pool's queries used to fit a model (0.15 on a 13 GB laptop) |
| `DEV_FRAC` | `0` | >0 = dev mode: full half-country S1 index, only this fraction of queries; writes under `WORK_DIR/dev` |
| `TAG` | `main` | namespace of a model run: models, OOF predictions and thresholds go to `WORK_DIR/{models,oof}/TAG` |
| `USE_XFEATS` | `0` | 1 = add the 14 name-edit / unmatched-number side features (`xfeats` step) to both stages |
| `MASK_LEX_FRAC` | `0.3` | share of training queries whose lexicon features are blanked (feature dropout for unseen countries) |
| `X16` | `0` | 1 = hold the training matrix as float16 and feed LightGBM through `lgb.Sequence` (half the RAM) |
| `WRITE_CANDIDATES` | `1` | 0 = skip `candidate_pairs.tsv` (it is identical for every run) |

## Commands (in order)

```bash
R="python code/business_entity_resolution/src/run.py"
$R env                    # Step 0: versions, cores, RAM
$R check1                 # Step 1: metric unit tests (0.714 / 1.0 / 0.05585)
$R check2                 # Step 2: row counts, GT facts (no cross-country pairs, <=1 S1 per query)
rm -f work/dicts/*.json && $R norm   # Step 3: normalization, first pass (no dictionaries)
$R dicts                  # Step 4: learn transliteration dictionaries from train GT
$R norm                   # Step 3 again, with the dictionaries applied
$R pools                  # Step 5: md5 half-split validation pools with all distractors (test-like density)
$R block                  # Step 6: TF-IDF blocking for P0, P1 and test (top-10 per query); recall report
$R feats                  # Step 8: 50 stage-1 pair features for P0, P1 and test
$R check8                 # Step 8 sanity checks
$R xfeats                 # 14 side features: cross-fitted name-edit lexicon, edit position/frequency, unmatched numbers

export USE_XFEATS=1
# run x1: 15% of each pool's queries per model
export TAG=x1 TRAIN_QUERY_FRAC=0.15
$R train1 && $R train2 && $R tune       # stage-1 cross-fit, stage-2 cross-fit, (T1, T2) on pooled OOF
OUTPUT_DIR=output_x1 $R predict --check-ids
# run x2: 30% of the queries, float16 training matrix
export TAG=x2 TRAIN_QUERY_FRAC=0.30 X16=1 WRITE_CANDIDATES=0
$R train1 && $R train2 && $R tune
OUTPUT_DIR=output_x2 $R predict --check-ids
# final: mean of the two runs' stage-2 scores, (T1, T2) re-tuned on the averaged OOF
WRITE_CANDIDATES=1 OUTPUT_DIR=output $R blend --tags x1 x2 --check-ids
```

Optional: `$R loco` (leave-one-country-out check). The quick path that produced the first leaderboard file (tier 0) is `train1 --tier0` then `predict --tier0`: stage 1 trained on P0 only, single threshold 0.76, 50 base features.

Validation: `predict`/`blend` run the official validator on `matching_results.tsv` with `--check-ids`, and stream-check `candidate_pairs.tsv` with the same rules (one row per S1, S2-/S3- ids, no duplicates, matches ⊆ candidates). The stdlib validator's own candidate check builds Python sets of ~100M ids and needs ~10 GB of RAM.

Runtime on a 6-core / 13 GB laptop: normalization + dictionaries ~20 min, blocking ~1.6 h, features ~45 min, side features ~26 min, each model run ~2–3 h (scoring ~61 min per 99.7M test pairs with two stage-1 and two stage-2 models).

## Method in one paragraph

Names and addresses are normalized the same way for every source and country (unidecode, OCR digit fixes, doubled-letter collapse, legal-form canonicalization, alias/web/id-tag handling, plus name and address transliteration dictionaries learned from train pairs). Blocking runs per country: a sparse TF-IDF over name words, address uni+bigrams and squashed-name char 4-grams (absolute DF cap 10k), with the top 10 S1 records per S2/S3 query by cosine. The candidate file is exactly this set. A stage-1 LightGBM scores each pair from about 50 string, number and blocking features. A stage-2 LightGBM adds context from out-of-fold stage-1 scores (the query's margin, and competition among the queries assigned to the same S1). The unmatched test records are generated from real S1 records (house number shifted, one word from a small per-country list appended), so 14 side features describe the name edit and the number mismatch. The main one is a log-odds lexicon of the words the query adds or drops, learned on the pools (cross-fitted, counted per query, blanked for 30% of training queries so unseen countries such as France fall back on the label-free edit shape). In
addition, a word that dominates a partition's single appended edits (>= 0.5% of its top-1 pairs, appended last in >= 90%)
but is not a train generator word gets the median log-odds of the train generator words. This is label-free, and in test
it fires only for France's generator words (groupe, holding, participations, ...). Each query links to its argmax S1 only. Per S1, the first link is kept if p >= T1 and later links if p >= T2, with thresholds tuned on test-density out-of-fold pools. The final scores average two runs trained on 15% and 30% of the queries.
