# Business Entity Resolution: reproduction guide

This regenerates `output/matching_results.tsv` and `output/candidate_pairs.tsv` from the train and test TSVs.
The pipeline uses no external data: every statistic (IDF, transliteration dictionaries, models) is learned from the provided files.

## Environment

- Python 3.12, `pip install -r requirements.txt`
- Full run: 16 vCPU / 64 GB RAM (e.g. SageMaker `ml.m5.4xlarge`), about 6–8 h end to end. A laptop only handles the dev mode (below).

## Layout the commands expect

Run every command from the directory that holds the data. Override paths with env vars:

| env | default | meaning |
|---|---|---|
| `DATA_DIR` | `data/raw` | holds `train/train_source{1,2,3}.tsv`, `train/train_ground_truth.tsv`, `test/test_source{1,2,3}.tsv` (the official kit uses `dataset`) |
| `WORK_DIR` | `work` | caches: `norm/ dicts/ pools/ cands/ feats/ models/ oof/` |
| `OUTPUT_DIR` | `output` | the two submission files |
| `N_JOBS` | all cores | worker processes / threads |
| `DEV_FRAC` | `0` | >0 = dev mode: full half-country S1 index, only this fraction of queries; writes under `WORK_DIR/dev` |

## Commands (in order)

```bash
R="python code/business_entity_resolution/src/run.py"
$R env                    # Step 0: versions, cores, RAM
$R check1                 # Step 1: metric unit tests (0.714 / 1.0 / 0.05585)
$R check2                 # Step 2: row counts, GT facts (no cross-country pairs, <=1 S1 per query)
$R norm                   # Step 3: normalization (first pass, no dictionaries)
$R dicts                  # Step 4: learn transliteration dictionaries from train GT
$R norm                   # Step 3 again, with the dictionaries applied
$R pools                  # Step 5: md5 half-split validation pools with all distractors (test-like density)
$R block                  # Step 6: TF-IDF blocking for P0, P1 and test (top-10 per query); recall report
$R feats                  # Step 8: stage-1 pair features for P0, P1 and test
$R check8                 # Step 8 sanity checks
$R train1                 # Step 10: stage-1 LightGBM cross-fit (P0 <-> P1), out-of-fold p1
$R train2                 # Step 10: stage-2 context model cross-fit, out-of-fold p2
$R tune                   # Step 11: (T1, T2) grid on pooled OOF -> work/models/thresholds.json
$R loco                   # Step 12: leave-one-country-out check (optional for reproduction)
$R predict                # Step 13: test inference, writes output/*.tsv, runs the validator
```

Quick path (the tier-0 submission): `train1 --tier0` then `predict --tier0` (stage 1 trained on P0 only, single threshold 0.76).

Validator (stdlib only): `python3 code/business_entity_resolution/src/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir $DATA_DIR/test --check-ids`

## Method in one paragraph

Names and addresses are normalized the same way for every source and country (unidecode, OCR digit fixes, doubled-letter collapse, legal-form canonicalization, alias/web/id-tag handling, plus name and address transliteration dictionaries learned from train pairs). Blocking runs per country: a sparse TF-IDF over name words, address uni+bigrams and squashed-name char 4-grams (absolute DF cap 10k), with the top 10 S1 records per S2/S3 query by cosine. The candidate file is exactly this set. A stage-1 LightGBM scores each pair from about 50 string, number and blocking features. A stage-2 LightGBM adds context from out-of-fold stage-1 scores (the query's margin, and competition among the queries assigned to the same S1). Each query links to its argmax S1 only. Per S1, the first link is kept if p >= T1 and later links if p >= T2, with thresholds tuned on test-density out-of-fold pools.
