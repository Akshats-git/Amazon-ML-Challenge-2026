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
| `USE_NFEATS` | `0` | 1 = add the 16 number-relation / number-oracle-lexicon features (`nfeats` step) to both stages |
| `LGB1_PARAMS`, `LGB2_PARAMS` | `{}` | JSON overrides of the stage-1 / stage-2 LightGBM parameters of a run |
| `FR_NORM`, `FR_NORM_ADDR`, `FR_NORM_NAME` | `0` | France-only address / name format rules (R08); off in the final |
| `FR_IMPUTE` | `0` | R07 lexicon imputation for vocabularies unseen in train; off in the final |
| `FR_RESCUE`, `RESCUE_P`, `RESCUE_JAC` | `0`, 0.90, 0.8 | R09 same-address rescue for countries absent from train; off in the final |
| `CAPS` | `1` | enforce the generator's per-source caps (<= 5 S2 and <= 6 S3 links per S1, highest p kept) |
| `BLEND_TAGS_UNSEEN` | unset | `blend`: members used for test countries absent from train (e.g. `x1,x2`), with that blend's own thresholds |
| `T_UNSEEN` | unset | `T1,T2` for test countries absent from train (default: the OOF-tuned thresholds) |

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
$R nfeats                 # 16 side features: signed number relations (generator shift set) + number-oracle lexicon

export USE_XFEATS=1
# run x1: 15% of each pool's queries per model
export TAG=x1 TRAIN_QUERY_FRAC=0.15
$R train1 && $R train2 && $R tune       # stage-1 cross-fit, stage-2 cross-fit, (T1, T2) on pooled OOF
OUTPUT_DIR=output_x1 $R predict --check-ids
# run x2: 30% of the queries, float16 training matrix
export TAG=x2 TRAIN_QUERY_FRAC=0.30 X16=1 WRITE_CANDIDATES=0
$R train1 && $R train2 && $R tune
OUTPUT_DIR=output_x2 $R predict --check-ids
$R blend --tags x1 x2      # x1+x2 blend thresholds (0.52, 0.74); the R06 leaderboard file
# run x3: + the 16 number features, ALL of each pool's queries (~58.5M pairs per model: ~64 GB RAM, ~2.5 h on 32 vCPU)
export TAG=x3 USE_NFEATS=1 TRAIN_QUERY_FRAC=1.0 MASK_LEX_FRAC=0.5 X16=0
$R train1 && $R train2 && $R tune
OUTPUT_DIR=output_x3 $R predict --check-ids
# run x4: same data and features, 255 leaves / min_child 200 / feature_fraction 0.6 / lambda_l2 5, seed 43
export TAG=x4 LGB1_PARAMS='{"num_leaves": 255, "min_child_samples": 200, "feature_fraction": 0.6, "lambda_l2": 5.0, "seed": 43}' \
       LGB2_PARAMS='{"seed": 43}'
$R train1 && $R train2 && $R tune
OUTPUT_DIR=output_x4 $R predict --check-ids
unset LGB1_PARAMS LGB2_PARAMS
# final: mean of the x3 and x4 stage-2 scores, (T1, T2) re-tuned on the averaged OOF (0.56, 0.74). France (absent from
# train) keeps the x1+x2 blend and its thresholds (0.52, 0.74): every France-specific change, including x3's own France
# predictions, scored lower on the leaderboard
WRITE_CANDIDATES=1 OUTPUT_DIR=output BLEND_TAGS_UNSEEN=x1,x2 $R blend --tags x3 x4 --check-ids
```

### Stage 3: top-2 re-rank with cross-encoders, and France edits (after the final `blend` above)

Scripts live in `src/stage3/`; `src/stage3/run_stage3.sh` runs the whole stage in order. Run it from the data directory. Tables go to `work/stage3/` (override with `S3_DIR`). It also needs x3's own file (`OUTPUT_DIR=output_x3 $R predict` above). Steps:

1. **Tables.**
   - `tables.py`: argmax tables, one row per query with its argmax S1 under the stage-2 blend (`BLEND=x3,x4` for US/India, `BLEND=x1,x2` for test France).
   - `tables_top2.py`: the argmax plus the runner-up candidate.

   Both carry the p of every run, the runner-up p, normalized and raw text, and raw-text perturbation counts.
2. **Features** (`top2.py build`), per candidate row:
   - consensus with the other queries whose argmax is the same S1: confident count, Σp, rank, shared number/name/address, number relation to the consensus number, token support;
   - raw-text perturbations (doubled letters, house-number suffixes, repeated words…), acronym flag, name collisions;
   - logits of x1–x4;
   - the competing row's S1 statistics.
3. **Cross-encoder datasets.**
   - `build_ce.py`: raw `name | address` pairs of the uncertain argmax rows (0.003 < p < 0.997) plus 4% of the rest on the pools.
   - `build_ce2.py` + `ce_rows_top2.py`: the plausible runner-up rows (p ≥ 0.01), appended to the scoring files.
4. **Cross-encoders** (GPU, `ce_train.py`). Each member fine-tunes one model per pool on the argmax rows (cross-fit) and scores the other pool and test.
   - `FacebookAI/xlm-roberta-base` (MIT, 1 epoch, bs 128, lr 3e-5);
   - `FacebookAI/xlm-roberta-large` (MIT, 1 epoch, bs 64, lr 1e-5; also scores the runner-up rows).
5. **Fit.** `add_ce.py` attaches the out-of-fold logits (test = mean of both folds). `top2.py fit` trains LightGBM cross-fitted P0 ↔ P1 on US+India; per query the higher-scoring row is the link candidate; (T1, T2) = (0.58, 0.78) tuned on the pooled out-of-fold scores. OOF macro F0.5 **0.99165** (US 0.99206, India 0.99106).
6. **France** (`fr_classes.py`, `fr_rules.py a,c,d,e`). The x1+x2 France decisions of the blend output, with four class-level edits backed by leaderboard evidence:
   - (a) drop equal-address category-swap links whose new word no other record of the S1 carries;
   - (c) add equal-address acronyms;
   - (d) drop +GEN-shifted links that are +GEN against the consensus too;
   - (e) add x3's equal-address true-noise-word links that the large cross-encoder backs (> 0.72).
7. **Write** (`compose.py output output`, `P3=p8b`, `FR_PAIRS=…/fr_pairs_acde.parquet`), then validate as above. Every link is a candidate pair (an argmax or runner-up row).
8. **R20 (final day)**, `run_stage3.sh` step 8 (work dir `R20_DIR`, default `$S3_DIR/r20`).
   - **Blocking rescue for India and US** (`r20_block2.py`, `r20_block3.py`, `r20_rescue_feats.py`, `r20_rescue_train.py`). For queries left unlinked that have an address, two retrievals propose new pairs:
     - an address-heavy TF-IDF pass (name 0.1 / address 0.8 / char 0.1, top 10);
     - a name-twin expansion (same core name, ranked by IDF-weighted shared address tokens, top 10).

     A v2 LightGBM rescue scorer (28 vocabulary-free features, cross-fitted P0 ↔ P1) ranks them.
   - **Rescue cross-encoder** (`r20_build_rce.py`, `r20_rce_train.py`, GPU):
     - input: for target queries whose best v2 score is ≥ 0.03, the top 4 candidates;
     - model: xlm-roberta-large, fine-tuned for 3 passes on one pool's candidates (raw "name | address" of both records);
     - scoring: the model scores the other pool and test.
   - **Rescue v4** (`r20_rescue4.py`, `r20_patch_add.py`): the v2 features plus the cross-encoder logit and its within-query rank, gap to the best and margin over the runner-up. It links the best new candidate when its score is ≥ τ.
     - India: τ 0.7, out-of-fold 0.99107 → 0.99262.
     - US: τ 0.85, out-of-fold 0.99210 → 0.99237.
   - **France precision removals** (`r20_fr_table.py`, `r20_fr_veto.py`, `r20_merge_fr.py`, `stage3.py fit` with `FPFX=f12ce TEST_C=France OUT=pfr`, `r20_fr_precision.py`, `r20_patch_fr.py`).
     - Removes s8b France links that a France stage-3 re-scorer drops **and** the large cross-encoder rejects (< 0.3).
     - Keeps the leaderboard-backed classes: true-noise words, acronyms, coined or concatenated names, and edit-added links.
     - Adds the cross-encoder address veto.
     - Result: −3,164 links. Final leaderboard 0.989472.
   - **Not in the final:** the France structural transfer (`r20_fr_transfer.py`) scored 0.987654 on the leaderboard against 0.987777 without it.
   - `r20_regen_cands.py` writes `candidate_pairs.tsv` = original candidates ∪ every rescue-scored pair (128,969,685 pairs) and stream-checks that matches ⊆ candidates.

Optional: `$R loco` (leave-one-country-out check). The quick path that produced the first leaderboard file (tier 0) is `train1 --tier0` then `predict --tier0`: stage 1 trained on P0 only, single threshold 0.76, 50 base features.

Validation: `predict`/`blend` run the official validator on `matching_results.tsv` with `--check-ids`, and stream-check `candidate_pairs.tsv` with the same rules (one row per S1, S2-/S3- ids, no duplicates, matches ⊆ candidates). The stdlib validator's own candidate check builds Python sets of ~100M ids and needs ~10 GB of RAM.

Runtime on a 6-core / 13 GB laptop: normalization + dictionaries ~20 min, blocking ~1.6 h, features ~45 min, side features ~26 min, runs x1/x2 ~2–3.5 h each (scoring ~61 min per 99.7M test pairs with two stage-1 and two stage-2 models). On a 32 vCPU / 128 GB VM: number features 19 min, run x3 ~2.5 h (stage 1 61 min with two 3,000-round models, stage 2 36 min, test scoring 43 min), run x4 ~2.1 h, blend ~3 min.

## Method in one paragraph

Names and addresses are normalized the same way for every source and country (unidecode, OCR digit fixes, doubled-letter collapse, legal-form canonicalization, alias/web/id-tag handling, plus name and address transliteration dictionaries learned from train pairs). Blocking runs per country: a sparse TF-IDF over name words, address uni+bigrams and squashed-name char 4-grams (absolute DF cap 10k), with the top 10 S1 records per S2/S3 query by cosine. The candidate file is exactly this set. A stage-1 LightGBM scores each pair from 80 features, and a stage-2 LightGBM adds context from out-of-fold stage-1 scores (the query's margin, and competition among the queries assigned to the same S1). The unmatched test records are generated from real S1 records: the house number is shifted upward by one of {1, 2, 3, 4, 5, 7, 9, 11, 13, 21} and the name may get a word from a small per-country list appended, a word swapped or the legal form changed. So beyond the 50 string/number/blocking features there are 14 edit features (a cross-fitted log-odds lexicon of the words the query adds or drops, counted per query and blanked for half the training queries; edit position and label-free frequency; unmatched-number offsets) and 16 number features: signed relations of the numbers one address has and the other lacks (shift in the generator set, -1/-2, digit edit, transposition, length difference, same street) and a label-free number-oracle lexicon, computed per partition: the log-odds of a name token appearing in single-upward-shift pairs versus equal-number pairs on the same street, which finds each country's generator words without labels (France's too). Each query links to its argmax S1 only. Per S1, the first link is kept if p >= T1 and later links if p >= T2 (thresholds tuned on test-density out-of-fold pools), at most 5 S2 and 6 S3 links per S1. The stage-2 scores average two full-data runs (x3, x4) for the countries seen in training; a stage-3 LightGBM then re-scores each query's argmax candidate with the consensus of the other records claiming the same S1, raw-text perturbations, model disagreement and a fine-tuned multilingual cross-encoder (xlm-roberta, MIT) reading both records' raw text; France keeps the x1+x2 blend, whose predictions scored better there on the leaderboard. France-specific rules (lexicon imputation, normalization rules, a same-address rescue) are implemented behind flags but off: each lowered the leaderboard.
