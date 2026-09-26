# Run Log: Amazon ML Challenge 2026 (Business Entity Resolution)

Metric: **macro F0.5 per S1 entity** (weights precision over recall). CV = OOF on a train sample, LB = public leaderboard.
Newest entries go at the **bottom**, and each gets a short ID (`R01`, `R02` …) that we can reference later.
`experiments.csv` holds the machine-written row for each run. This file records the reasoning behind each run, the results and what we learned.

## Scoreboard

| ID | Time (IST) | Name | CV F0.5 | Pair recall | LB score | LB rank | Status |
|----|-----------|------|---------|-------------|----------|---------|--------|
| R01 | 25 Sep 02:52 | dev_smoke | 0.98784 | 0.9909 | – | – | smoke test only (2% sample) |
| R02 | 25 Sep 12:40 | eda_redteam (baseline, test-like pools) | 0.98303 | 0.9898 | – | – | analysis only; not submitted |
| R03 | 25 Sep 14:32 | ber_v1 dev (new pipeline, DEV_FRAC=0.1) | 0.98287* | R@10 US 0.9910 / IN 0.9841 | – | – | dev only; *optimistic, only comparable with other dev runs |
| R03c | 26 Sep 00:45 | laptop streaming rewrite: regression on dev + mini | 0.98292* | = R03 | – | – | code check only; outputs byte-identical to R03 path |
| R04 | 26 Sep 02:28 | ber_v1 FULL scale on laptop: tier-0 (stage 1, P0 model, T=0.76) | 0.97623 (P1 OOF) | R@10 US 0.9909 / IN 0.9842 | pending | pending | `output_tier0/` validated; first LB file |
| R05 | 26 Sep 04:52 | + 14 edit/number side features (xfeats), stages 1+2, tuned (T1,T2)=(0.48,0.76) | **0.98645** (pooled OOF) | same as R04 | pending | pending | `output_r05/` (predict running) |
| R06 | 26 Sep 09:09 | R05 with 2× training data (TRAIN_QUERY_FRAC 0.30, float16 + lgb.Sequence), tuned (0.50,0.76); + blend x1+x2 (0.52,0.74) | **0.98705** / blend 0.98706 | same | pending | pending | `output_r06/`, `output_blend/` validated |
| R07 | 26 Sep 09:36 | R05+R06 blend + France lexicon imputation (test-only, label-free) | **0.98706** (= blend) | same | **0.979881** | **146** | `output_blend_fr/` submitted 26 Sep ~10:00; **CV→LB gap −0.0072** |

**Current best on LB:** **R07 `output_blend_fr/` = 0.979881 public, rank 146** (26 Sep ~10:00 IST; the top-500 cutoff at Sun 00:00 is met). This is the first and only upload so far. Other validated files not uploaded yet: `output_blend/` (0.98706, no France fix), `output_r06/` (0.98705), `output_r05/` (0.98645), `output_tier0/` (0.97623).

---

## R01 · dev_smoke · 25 Sep 02:52
- **Approach:** normalise name/address → TF-IDF blocking (top_k=5, min_sim=0.3, name_weight=0.5) → pair features → LightGBM (OOF) → per-S1 threshold decision (thr=0.70).
- **Data:** train frac=0.02 (density-preserving sample).
- **Results:** OOF macro F0.5 = 0.98784, blocking pair recall = 0.9909.
- **LB:** not submitted.
- **Takeaways:** pipeline runs end to end. The 2% sample makes this CV optimistic, so re-check it on a larger frac before trusting it.
- **Next:** run dev at frac=0.05, then do the first full test submission.

## R02 · eda_redteam · 25 Sep 12:40
- **Approach:** no pipeline change. Ran the R01 baseline (char3 TF-IDF blocking k=5, LightGBM, per-query argmax + one global threshold) on disjoint 3% S1 pools: A (train distractor density, 1.21/S1), B1 (train density) and B2 (test density, 2.31/S1). Also ran blocking-recall studies on the FULL per-country train S1 index with 0.5% query samples.
- **Data:** laptop. The 3% S1 pools come from the full train set, with distractors sampled at 1x or 1.9x.
- **Results:**
  - A OOF F0.5 0.98558 @thr 0.74. B1 0.98632. **B2 (test density) 0.98303 @thr 0.76**, and using A's threshold costs only 0.0002. Blocking pair recall 0.9898, oracle F 0.9969.
  - Loss at test density is 0.0170: singletons 0.0022, n=1 0.0024, n=2-3 0.0067, n>=4 0.0056. Removing all FPs would add +0.0083. Recovering every model FN that blocking kept would add +0.0056.
  - FPs: 84% come from queries with no true candidate (distractors); 16% picked the wrong S1. Model FNs: 78% had the correct argmax but fell under the threshold (median p 0.555).
  - LOCO: eval US 0.98684 in-country vs 0.98239 when trained on India (-0.45pp). Eval India 0.97793 vs 0.96806 when trained on US (-0.99pp).
  - Blocking on the full index, R@10: unpruned char3 US 98.83 / India 97.50 at ~20 ms/query, about 60 h for test, so rejected. Sparse [name word | addr word uni+bigram | squashed-name char4] with DF cap 10k: US 99.05 / India 96.81 at ~0.7 ms/query. India + train-learned translit dict (669 entries): R@1 96.29, R@10 98.08.
  - Data facts: every S2/S3 matches <=1 S1 (0 violations); 0 cross-country pairs; singletons 5.59%; test has ~2.3 distractors per S1 vs 1.22 in train (volume and noise-marker estimates agree).
- **LB:** not submitted.
- **Takeaways:** precision on distractors is the main lever. Transliteration fixes most India misses. The char3 blocking cannot run at test scale.
- **Next:** swap blocking to the capped sparse design, add the translit dict, build the test-like 2-fold harness, then make the first full test submission.

## R03 · ber_v1 dev (new `code/business_entity_resolution` pipeline) · 25 Sep 14:32
- **Approach:** rebuilt from scratch following the implementation brief:
  - **Normalization (7.1):** web/alias/id-tag flags, OCR digit fix, doubled-letter collapse, legal-form canonicalization, alias parts. The `nonlatin` flag ignores U+2000–U+20CF, because France's `’` was triggering it.
  - **Transliteration dicts** learned from train GT: name 694 entries, address 30.
  - **Pools:** md5 halves plus ALL distractors, 2.43 distractors per S1.
  - **Blocking:** name words + address uni/bigrams + name_sq char-4, absolute DF cap 10k, weights 0.25/0.5/0.25, top-10 per query.
  - **Features:** 50 stage-1 features (rapidfuzz via `cpdist`, soft Monge-Elkan, numbers, legal, context).
  - **Models:** stage-1 LightGBM P0↔P1 cross-fit, then stage-2 on OOF p1 context. Argmax per query plus a (T1, T2) policy.
- **Data:** laptop, full train. DEV_FRAC=0.1: the full half-country S1 index but only 10% of queries (matches of 10% of S1 + 10% of distractors). Each evaluated S1 therefore sees about 10× less distractor pressure than on test, so F and thresholds are optimistic. Compare dev runs only with other dev runs.
- **Results:**
  - Step checks: metric 0.714 / 1.0 / 0.05585 ✓. Row counts and GT facts ✓. Translit held-out shared-core-token rate 0.309 → 0.999, test non-Latin coverage 0.955 ✓. Pools disjoint, US half 661.8k ✓.
  - Blocking (dev pools): US R@1 0.976, R@10 0.991, oracle F 0.997, 0.52–0.57 ms/q. India R@1 0.968, R@10 0.984, oracle F 0.995, 0.47–0.51 ms/q. 10 candidates per query.
  - Features: positive rate 5.8%. Positives' median address token_set is 94.5. No constant features. About 60–120k pairs/s on 8 laptop workers.
  - Tier-0 (stage 1, P0→P1): F@0.76 0.97579.
  - Full cross-fit stage 1 OOF: F@0.76 0.97577, best single threshold 0.98174 @0.40.
  - Stage 2 OOF: F@0.76 0.97747. Tuned (T1 0.40, T2 0.70): **0.98287** (US 0.98600, India 0.97818).
  - Buckets: n=0 0.997, n=1 ~0.87 (the weakest bucket), n=2–3 0.983, n≥4 0.992.
  - Loss decomposition: F_noFP 0.98608, F_allFN_blocked_in 0.99305. Of 18.2k FNs, 9.0k were blocked out, 7.8k had the right argmax but fell below threshold, and 1.4k lost to another S1. 87% of FPs came from queries with no true candidate.
  - Stage-1 top gain: blk_score, blk_rank, blk_gap, s1_n_top1, addr_me_q2s. Stage-2 top gain: p1, pair_rank_in_q.
- **LB:** not submitted.
- **Takeaways:** the pipeline works end to end, and every Step 1–11 check passes on dev. Thresholds hit the grid floor because of dev's low distractor density, so they are not meaningful; the full test-density pools will set them. Blocking misses are now about half of all FN. The laptop crashed once (OOM from 12 workers × a 4M-entry memo; the memo is now capped at 500k).
- **Next:** full-size pools, test blocking and test features on the 16 vCPU / 64 GB box, then the Tier-0 LB submission and the full cross-fit.

## R03b · test-path smoke (no score) · 25 Sep 15:19
- **Approach:** the R03 dev models ran on a 3% random mini test set (`work/mini`, 52k S1 / 299k queries): norm → block → feats → predict (stage 1+2, and tier-0) → write → validator.
- **Results:** validator **PASS** with `--check-ids`. Matches ⊆ candidates. Random links re-scored from raw text have a median name token_set of 93.8. Output bytes are exact: header correct, empty rows end at the tab, no quoting.
- **Next:** full run on SageMaker (`run_all.sh`).

## R03c · laptop streaming rewrite, regression checks (no new score) · 26 Sep 00:45
- **Approach:** SageMaker quota is still 0, so the full run moves to the laptop (6 cores/12 threads, 13 GB RAM, 13 GB free disk). The brief assumed 64 GB, so every whole-partition step was rewritten to stream:
  - **Blocking:** own thread count `BLOCK_THREADS` (default 4). The sparse top-n matmul is cache-bound: 0.135 ms/q at 3–4 threads vs 0.26 at 8 and 0.47 at 12 (per 200k P0 queries). The full blocking estimate drops from ~3.2 h to ~1.3 h.
  - **Features:** context features are per-query / per-S1 numpy arrays gathered chunk by chunk (no 40M-row polars joins). Each partition loads only its own rows. `y` comes from a q→S1 lookup. Files hold 2M pairs each, worker chunks 150k pairs.
  - **Feature storage:** floats are rounded to 10 mantissa bits (rel. err ≤ 5e-4; LightGBM bins to 255 anyway) and written with zstd-15: 47.1 → 34.4 B/pair, so ~7.9 GB instead of ~10.8 GB for 230M pairs.
  - **Models:** training matrices are preallocated and filled file by file, holdout rows last (train/holdout are views). `TRAIN_QUERY_FRAC` has an env override. Predictions are float32 `.npy` arrays aligned with the feature files. The stage-2 context is rebuilt in memory from p1 (`Ctx2`, numpy) and never stored, which saves ~2.3 GB of disk.
  - **Decisions and outputs:** argmax per query is a numpy reduction. Tuning and loss decomposition run on argmax rows only. `candidate_pairs.tsv` is written from sorted int32 keys, 50k S1 at a time. The official validator needs ~10 GB for 100M candidate ids, so it now checks `matching_results.tsv` (`--check-ids`), and `check_candidates()` streams the candidate file with the same rules plus matches ⊆ candidates. `check8` streams too.
  - **Determinism fix:** soft Monge-Elkan truncates leftovers to the top-8 IDF tokens, and ties were broken by set order, which depends on `PYTHONHASHSEED`. Ties are now broken by the token itself.
- **Data:** laptop, dev pools (`work/dev`, DEV_FRAC=0.1) and the 3% mini test (`work/mini`).
- **Results:**
  - Features rebuilt for dev P0 India (2.6M pairs) are identical to R03's after rounding: keys, `y` and all 18 context features. Only 21 rows differ in `addr_me_q2s/s2q/unm_idf_q`, from the tie-break fix.
  - `Ctx2` matches the old polars stage-2 context on 9 of 11 features. `pair_rank_in_q` differs on 18/2 rows (tie order). `pair_rank_in_s` differs on 0.14% of rows because tied argmax queries of the same S1 now share a rank.
  - Mini test with the R03 models: tier-0 `matching_results.tsv` byte-identical. Full two-stage `matching_results.tsv` and `candidate_pairs.tsv` byte-identical. Validator PASS, streaming candidate check OK.
  - Dev retrain with the new training code: stage 1 identical to R03 (best_iter 463/416, OOF F@0.76 0.97577, single 0.98174 @0.40). Stage 2 F@0.76 0.97778, tuned **0.98292** @ (0.40, 0.76), vs R03's 0.98287 @ (0.40, 0.70). Peak RSS 3.2 GB, 8.5 min.
- **Takeaways:** the rewrite is behaviour-preserving, and the laptop can run the full pipeline.
- **Next:** full-scale blocking → feats → tier-0 LB (R04).

## R04 · ber_v1 full scale on the laptop (tier-0 first, then full cross-fit) · 26 Sep (in progress)
- **Approach:** R03 pipeline at full scale with the R03c streaming code. Settings: `N_JOBS=8 BLOCK_THREADS=4 TRAIN_QUERY_FRAC=0.15`. **Deviation from the brief:** each model trains on 15% of its pool's queries (≈10M pairs), not 35%, because of the 13 GB of RAM. Threshold grid widened to T1 ∈ [0.10, 0.90], T2 ∈ [0.30, 0.96].
- **Data:** laptop, full train pools (P0/P1: half the S1 + all distractors per country) and the full test set.
- **Results (blocking, top-10, BLOCK_THREADS=4):**

  | partition | S1 index | queries | pairs | ms/query | R@1 | R@10 | oracle F |
  |---|---|---|---|---|---|---|---|
  | P0 India | 440,953 | 2,602,238 | 26,022,097 | 0.165 | 0.96799 | 0.98424 | 0.99483 |
  | P0 US | 661,784 | 3,897,915 | 38,978,986 | 0.282* | 0.97577 | 0.99092 | 0.99717 |
  | P1 India | 442,235 | 2,604,611 | 26,045,844 | 0.170 | 0.96805 | 0.98417 | 0.99481 |
  | P1 US | 661,849 | 3,897,309 | 38,972,928 | 0.229 | 0.97579 | 0.99079 | 0.99717 |
  | test France | 259,452 | 1,434,993 | 14,349,568 | 0.101 | – | – | – |
  | test India | 809,986 | 4,717,565 | 47,175,328 | 0.357 | – | – | – |
  | test US | 663,106 | 3,817,031 | 38,170,203 | 0.236 | – | – | – |

  Test total **99,695,099 pairs** (guard 110M ✓). The P0+test blocking took 73 min wall (peak RSS 8.6 GB), P1 23 min (8.9 GB). Both pools pass the gates (US R@10 ≥ 0.990, India ≥ 0.978).
- **Features (P0 + test, 165M pairs):** 31 min wall, 8 workers, 58–98k pairs/s (India slower than US), peak parent RSS 6.0 GB. 33.3 B/pair on disk (P0 2.2 GB, test 3.3 GB). `check8` P0: 65,001,083 pairs, positive rate 0.0581, positives' median addr token_set 94.6, no constant features, **PASS**.
- **Data insight (full-scale P0/test, feeds R05):** unmatched queries are generated hard negatives. The generator copies an S1 record, shifts the house number by a few (524→537, 3931→3934, some units kept) and appends one word from a small per-country list:
  - US: partners / holdings / group, plus ~20 location words (north, downtown, westgate, …).
  - India: public / enterprises / industries / ventures / exports / overseas / infratech / holdings / group.
  - France (test only): groupe / developement / france / participations / international / holding / distribution.

  In P0 US top-1 pairs with exactly one extra name token, "holdings" and "group" have 0 true vs ~59k distractor pairs each, while "the" has 29.8k true vs 5.5k. String similarity cannot see this, so R05 adds cross-fitted token log-odds, position (appended vs prepended) and label-free frequency features, plus unmatched-number features (house-number shift while unit numbers still match).

  \*slowed by a concurrent dev test. The gates pass: US R@10 0.9909 ≥ 0.990, India 0.9842 ≥ 0.978. Recall matches dev (same S1 index), as expected.
- **Tier-0 (stage 1 only, P0 model, single T = 0.76):**
  - Fit: 9,744,798 rows (15% of P0 queries, pos 0.0581), best_iter 1435, holdout logloss 0.00525, 7 min, peak RSS 3.2 GB.
  - Top gain: blk_score 0.378, blk_rank 0.304, blk_gap 0.081, addr_me_q2s 0.043, first_num_exact 0.037, num_soft_any 0.022, name_unm_idf_q 0.018.
  - Test: `output_tier0/matching_results.tsv` has 1,732,544 rows and 5,720,747 links. Validator `--check-ids` **PASS**. Random-link median raw-name token_set 96.4.
  - Label-free monitors per country: links/S1 France 3.289, India 3.288, US 3.325. Empty share 5.9% / 6.1% / 5.9% (train singletons 5.6%). Queries linked 59.5% / 56.5% / 57.8%. France behaves like US/India, so there's no sign of the unseen country being over- or under-linked.
  - Prediction is slow: LightGBM scores ~61k pairs/s for a 1435-tree model on 8 threads, so 27 min for the 99.7M test pairs.
  - **Abandoned:** `lleaves` (LLVM-compiled trees) needed llvmlite ≤ 0.43, and compiling this model used > 6 GB. The OOM killer took the benchmark (not the chain). Uninstalled.
- **LB (tier-0):** pending: score and rank to be reported after upload.
- **Tier-0 CV (OOF on P1, full test density):**
  - F@0.76 **0.97623**; best single threshold 0.97662 at 0.68. Precision 0.992, recall 0.950.
  - Buckets: n=0 0.9757 (loss 0.0014, FPs on singletons), n=1 0.9239, 2–3 0.9756, 4+ 0.9828.
  - Loss decomposition: F_noFP 0.98257, F_allFN_blocked_in 0.98995. FP 29,435 (27,385 from queries with no true candidate). FN 190,051: 45,317 blocked out, 116,582 argmax right but below threshold, 28,152 lost to another S1.
  - P1 scoring took 18 min for 65M pairs.
- **R05 prep: side-feature A/B** (same 2% of P0 queries: 1.30M rows, 10% query holdout, 150 rounds, stage 1):

  | features | holdout logloss |
  |---|---|
  | 50 base features (R04) | 0.00805 |
  | + 14 xfeats, pair-level lexicon | 0.00556 (a leak, see below) |
  | + 14 xfeats, **per-query lexicon (≥ 50 queries)** | **0.00569** |

  The leak: distractor queries sit in both pools, so a token seen in only a few distractors carried their labels across. Counting each query once per token and requiring ≥ 50 queries removes it, and the gain holds (−29% logloss). Top xfeats by gain: nx_num_unm_affix 0.079 (3rd overall), nx_extra_lo_max 0.028, nx_num_unm_absdiff 0.014, nx_extra_lo_min 0.009.
- **Full stage-1 cross-fit (P0 → P1 and P1 → P0 OOF over both pools, 2.2M S1):**
  - P1 model: 9,747,692 rows, best_iter 1345, holdout logloss 0.00537, 7.4 min.
  - OOF F@0.76 **0.97613**; best single threshold **0.97668 at 0.66**.
  - Loss decomposition: F_noFP 0.98205, F_allFN_blocked_in 0.99036. FP 87,139 (83,076 from queries with no true candidate). FN 389,861: 90,189 blocked out, 243,140 argmax right but below threshold, 56,532 lost to another S1.
  - Scoring 65M pool pairs takes ~17 min per model.
- **Decision:** stop R04 after the stage-1 cross-fit (done 03:46) and start R05 immediately. The R04 stage-2/tune ablation can run later.

## R05 · R04 + name-edit / unmatched-number side features (xfeats), full cross-fit · 26 Sep (in progress)
- **Approach:** the R04 pipeline (tag `x1`) with 14 side features from `ber/xfeats.py`, stored in `work/xfeats/` row-aligned with the feature files:
  - a cross-fitted per-query log-odds lexicon of extra and missing name tokens (≥ 50 queries per token);
  - label-free edit shape: extra token appended or prepended, missing token last, and the token's in-partition frequency as the single edit of top-1 pairs;
  - unmatched-number count, minimum offset and prefix/suffix relation.

  Lexicon features are blanked for 30% of training queries (feature dropout) so the model also learns the label-free route France needs. Stages 1 + 2, (T1, T2) tuned on pooled OOF. Same settings as R04 (TRAIN_QUERY_FRAC 0.15, lr 0.05). The xfeats pass took 26 min (P0+P1+test, 1.5 GB).
- **Data:** laptop, full pools + test.
- **Results:**
  - Stage-1 P0 model: 9,744,798 rows, **best_iter 799 (R04: 1435), holdout logloss 0.00417 (R04: 0.00525, −21%)** even with 30% of holdout queries blanked. 4.7 min.
  - Top gain: blk_rank 0.428, blk_score 0.251, addr_me_q2s 0.059, **nx_num_unm_affix 0.053**, blk_gap 0.050, **nx_num_unm_absdiff 0.046**, name_unm_idf_q 0.010, nx_extra_lo_max 0.008.
  - **Early read, stage-1 OOF on P1 (P0 model), directly comparable with R04 tier-0:**
    - F@0.76 **0.98451 vs 0.97623 (+0.0083)**; best single threshold **0.98460 at 0.70** vs 0.97662 at 0.68.
    - FP 16,272 vs 29,435 (−45%). FN 135,663 vs 190,051: below threshold 62,549 vs 116,582 (−46%), blocked out 45,317 (unchanged), lost to another S1 27,797 vs 28,152.
    - F_noFP 0.98814, F_allFN_blocked_in 0.99260.
    - P1 scoring took 9.5 min (R04: 17.7 min), thanks to fewer trees.
  - Blocking misses are now a third of the FNs (45k of 136k), so blocking recall becomes the next big lever after the model.
  - **Blocked-out true pairs (P1):** mostly unrecoverable.
    - US: 21,082 (0.92%), 77% with an empty query address; S1 holds same-name businesses in other states ("Wildlife Committee LLC" NY vs MO), so the name alone cannot pick one.
    - India: 24,235 (1.58%), 44% empty address; the rest are mostly fully Devanagari/Bengali names or brand names with partial addresses.
    - Conclusion: raising K or adding a second blocking key would buy little, so model quality stays the priority.
  - **Stage-1 cross-fit, pooled OOF (P0+P1, 2.2M S1):**
    - P1 model: best_iter 1048, holdout logloss 0.00413, 6 min.
    - F@0.76 **0.98455** (R04 0.97613, **+0.0084**); best single threshold **0.98462 at 0.70** (R04 0.97668 at 0.66).
    - FP 50,239 (R04 87,139, −42%). FN 269,726: below threshold 123,683 (R04 243,140, −49%), blocked out 90,189, lost to another S1 55,854.
    - F_noFP 0.98817, F_allFN_blocked_in 0.99262. train1 took 36 min wall, peak RSS 4.5 GB.
  - **Stage 2:** models best_iter 816/804, holdout logloss 0.00334/0.00335 (stage 1: 0.00417/0.00413). Top gain p1 0.756, pair_rank_in_q 0.111, blk_score 0.075, q_margin 0.033. OOF p2 F@0.76 **0.98619** (p1 0.98455, +0.0016); best single threshold 0.98626 at 0.72. train2 took 26 min.
  - **Tuned (T1, T2) = (0.48, 0.76): OOF F0.5 = 0.98645** (US 0.98799, India 0.98412). Precision 0.99609, recall 0.96898.
    - Buckets: n=0 0.9804 (loss 0.0011), n=1 0.9570 (0.0023), 2–3 0.9868 (0.0054), 4+ 0.9902 (0.0047).
    - Loss decomposition: F_noFP 0.99012, F_allFN_blocked_in 0.99257. FP 39,652 (36,749 from queries with no true candidate). FN 236,972: 90,189 blocked out, 94,726 below threshold, 52,057 lost to another S1.
  - **vs R04 tier-0 CV (0.97623): +0.0102.**
  - **Decision-policy checks on R05 OOF (no gain, policy is saturated):** per-country (T1, T2) 0.98645 (India own (0.56, 0.76), US own (0.48, 0.78)) = pooled. Expected-F rule best 0.98647. Three thresholds (T3 for rank ≥ 3) best (0.48, 0.70, 0.78) = 0.98645. The pooled grid is flat near the optimum, so the pooled thresholds stand (and suit France).
  - **Error profile (R05 OOF, tuned):** 32.2k of the 39.7k FPs are rank ≥ 3 links. Many are address-less queries matched on the name alone, or same-name S1 in other cities. Below-threshold FNs have p quartiles 0.28 / 0.48 / 0.64: OCR name damage, true matches with a house number off by one, and true matches that append words (LLC, Enterprises) like the generator does.
  - **Test (05:53):** `output_r05/matching_results.tsv` has 1,732,544 rows and **5,811,054 links**, 5.8% empty. Validator `--check-ids` **PASS**. `candidate_pairs.tsv` has 99,695,099 pairs and passes the streaming check (matches ⊆ candidates). Links/S1 France 3.306, India 3.346, US 3.383, empty share 5.8% each. Random-link median raw-name token_set 95.7. Predict took 61 min (two stage-1 + two stage-2 models on 99.7M pairs), peak RSS 5.4 GB.
- **LB (R05):** pending: to be uploaded; score and rank to be reported.
- **Takeaways:** the generator-aware side features are the biggest single gain so far (+0.0084 at stage 1). Stage 2 adds +0.0016, and the policy is saturated.
- **Next:** R06 = R05 with 2× training data (float16 matrix + lgb.Sequence, TRAIN_QUERY_FRAC 0.30), then an R05+R06 blend.

## R06 · R05 with 2× training data (+ R05/R06 blend) · 26 Sep (in progress)
- **Approach:** R05 features and settings, but TRAIN_QUERY_FRAC 0.30 (≈19.5M rows per model). The matrix is held as float16 (features are stored at float16 precision) and fed to LightGBM through `lgb.Sequence` in float64 batches. A small-scale check matched float32 (holdout logloss 0.00589 vs 0.00593). Tag `x2`, no candidate file (identical to R05's). Afterwards `blend --tags x1 x2` averages the final p2 of R05 and R06 and re-tunes (T1, T2) on the averaged OOF.
- **Data:** laptop, full pools + test.
- **Results:**
  - Stage-1 P0 model: 19,509,868 rows, best_iter 1410, holdout logloss **0.00394** (R05: 0.00417 on a smaller holdout draw), 14 min.
  - Stage-1 P1 model: 19,515,192 rows, best_iter 1469, holdout logloss 0.00394 (R05: 0.00413), 14.7 min.
  - **Stage-1 pooled OOF: F@0.76 0.98515** (R05 0.98455, **+0.0006**); best single threshold 0.98521 at 0.72. FP 44,638 (R05 50,239, −11%). FN 264,012: below threshold 118,532 (R05 123,683), blocked out 90,189, lost to another S1 55,291. F_noFP 0.98848, F_allFN_blocked_in 0.99291. train1 took 65 min.
  - Stage 2: models best_iter 1215/876, holdout logloss **0.00312/0.00315** (R05 0.00334/0.00335). OOF p2 best single threshold 0.98688 at 0.72 (R05 0.98626). train2 took 42 min.
  - **Tuned (T1, T2) = (0.50, 0.76): OOF F0.5 = 0.98705** (R05 0.98645, **+0.0006**). US 0.98848, India 0.98490.
    - Buckets: n=0 0.9830, n=1 0.9579.
    - FP 36,140 (R05 39,652). FN 230,657: below threshold 89,509, blocked out 90,189, lost to another S1 50,959.
    - F_noFP 0.99033, F_allFN_blocked_in 0.99295.
  - **Test (09:05):** `output_r06/matching_results.tsv` has 1,732,544 rows and 5,812,180 links. Validator `--check-ids` **PASS** (candidate file identical to R05's, so not rewritten). Links/S1 France 3.306, India 3.346, US 3.384; empty share 5.8%. Random-link median raw-name token_set 96.3. Test scoring took 83 min (larger models: 1410/1469 + 1215/876 trees).
- **LB (R06):** pending.
- **Blend `blend --tags x1 x2` (09:09):**
  - OOF: x1 alone 0.98645, x2 alone 0.98705; **mean of the p2s 0.98706 at re-tuned (0.52, 0.74)**, i.e. +0.00001 over x2 alone (noise level).
  - FP 36,833. FN 230,460: 89,533 below threshold, 90,189 blocked out, 50,738 lost to another S1.
  - Test (old France features): `output_blend/` has 5,815,887 links, links/S1 France 3.309, India 3.349, US 3.386. Validator **PASS**. Took 3.5 min.
- **Takeaways:** 2× data gives +0.0006. Blending with the weaker 15% run adds nothing, so a third blend member (R08) is not worth the disk risk and is dropped. The final base is the blend (marginally ahead, and averaging adds some robustness); x2 alone is equivalent.

## R07 · France lexicon imputation (label-free), no retraining · 26 Sep (queued after R06)
- **Finding (label-free, R05 test predictions):** among top-1 test pairs whose query name appends one generator word, the share that gets linked is:
  - US 0.3% (1,935 of 575k);
  - India ~0% (4 of 521k);
  - **France 2.1% (4,080 of 192k)**.

  The linked France pairs are 97% rank ≥ 3 with p median 0.974. The house number matches **exactly**, and the only difference is an appended French generator word ("… France", "… Groupe", "… Développement"). US/India catch these through the lexicon (holdings/group ≈ 9), but France's words are unseen or learned as non-generator words in train (france 1.02, distribution 5.41, holding 6.66, international 8.52, groupe/developement/participations absent). The generator words cover ~36% of France's estimated distractors (US 38%), so these links are most likely FPs, worth ~+0.0005 overall F if removed.
- **Approach:** in each partition, a token that is ≥ 0.5% of top-1 pairs as their single appended edit (appended last in ≥ 90%) has the generator profile. If it lacks that profile in the train pools, its lexicon value is raised to the median of the train generator-profile words (8 tokens, median **9.16**). Dry run:
  - test France: raised to 9.16 for distribution, international, developement, participations, groupe, holding, france;
  - test US, test India, P0, P1: **unchanged**. The models, OOF and thresholds therefore stay valid; only test France is re-scored.
- **Plan:** after R06, rebuild the test xfeats, re-score test France with the x1 and x2 models, then `blend --tags x1 x2` into `output_blend_fr/`. `output_blend/` (from R06, old France features) serves as the LB before/after comparison.
- **Results (09:36):**
  - Test xfeats rebuilt in 3 min. Only France changed: raised to 9.16 for international, participations, developement, distribution, holding, groupe, france.
  - Test France re-scored with x1 and x2 (stages 1+2) in 21 min, then `blend --tags x1 x2`: OOF **0.98706** at (0.52, 0.74), unchanged since the pools are untouched.
  - `output_blend_fr/matching_results.tsv` has 1,732,544 rows and **5,794,198 links**. Validator `--check-ids` **PASS**. Links/S1 France **3.2255** (was 3.3091), India 3.3487, US 3.3855; France empty share 6.03% (was 5.82%).
  - Diff vs `output_blend/`: **22,401 France links removed, 712 added**; US/India untouched.
    - All but 406 removed links have a French generator word among the query's extra tokens: 4,063 plain appends, plus the generator word **replacing** a name word at an identical address ("Mam Club SARL" vs "Mam SARL France", "Boos Theatre SASU" vs "Groupe Boos SASU").
    - The remaining 406 are stage-2 knock-on effects. Added links are queries that become an S1's first link once a distractor is gone.
  - Consistency check (label-free): assuming the same ~2.3 distractors per S1 in every country, queries/S1 imply true pairs/S1 of US 3.46, India 3.52, France 3.23. US lands on its expected link count; France moves from well above its expectation (3.31) to 3.23.
- **LB (R07):** **0.979881 public, rank 146** (26 Sep ~10:00 IST; team of 4, IIT Bhilai). The public LB is scored on a subset of test; the private LB uses the rest. **CV→LB gap −0.0072** (OOF 0.98706). That is far beyond sampling noise, so the gap is systematic (see R07b). `output_blend/` vs `output_blend_fr/` would measure the France fix directly.
- **Takeaways:** France's generator vocabulary was invisible to the train lexicon, and a label-free profile (frequency and position of single appended edits) recovers it without labels or retraining.
- **Next:** final package = `output_blend_fr/` matching + R05 candidate file; doc; zip.

## R07b · LB-gap diagnosis (analysis only, no new file) · 26 Sep 10:50
- **Approach:** label-free comparison of the final blend (x1+x2 p2, T = (0.52, 0.74)) on test against the OOF pools:
  - argmax/decision statistics, plus a plug-in self-estimated F (p treated as calibrated);
  - generator-word appends per S1;
  - an OOF simulation of test's distractor density;
  - a sample of uncertain France test pairs.
  Scripts are in the session scratchpad; logs are `work/logs/diag_lb.log`, `sim_density.log` and `diag_nums.log`.
- **Data:** laptop, existing p2 arrays, xfeats and freq tables. No retraining.
- **Results:**

  | partition | links/S1 | uncertain queries/S1 (p 0.1–0.9) | self-est. F | actual F |
  |---|---|---|---|---|
  | OOF US (P0 / P1) | 3.373 | 0.121 / 0.123 | 0.9934 / 0.9931 | 0.98846 / 0.98849 |
  | OOF India (P0 / P1) | 3.365 / 3.359 | 0.143 / 0.146 | 0.9918 / 0.9915 | 0.98489 / 0.98497 |
  | test US | 3.386 | 0.172 | 0.99165 | – |
  | test India | 3.349 | 0.184 | 0.99014 | – |
  | **test France** | 3.226 | **0.389** | **0.98233** | – |

  - **Hard-distractor density differs, but that costs little.** Generator-word appends at top-1 per S1: US pools 0.508 vs test 0.732 (×1.44), India 0.355 vs 0.556 (×1.57). Each pool holds all distractors, but half of them were generated from S1s in the other pool, so they are easy there. Test therefore has ~1.5× more near-copy distractors per S1. Simulated on OOF (argmax rows of queries without a true candidate duplicated with prob. r, ranks recomputed):
    - r = 0.5: F 0.98706 → 0.98640 at (0.52, 0.74); re-tuned (0.52, 0.78) gives 0.98644;
    - r = 0.75: 0.98609, re-tuned 0.98616.

    So density explains only ~0.0007 of the gap, and moving the thresholds is worth +0.00004. Keep them.
  - **US/India on test look like OOF.** Self-est. F is only −0.0015 vs the pools. Links with a house number shifted by 1–20: test 1.57% / 1.89%, pools 1.43% / 1.81%.
  - **France is the outlier.** It has 3.2× US's uncertain queries per S1, self-est. F 0.009 below US, and self-recall 0.963 vs 0.983. If US/India land near their estimates (~0.986 / ~0.983), France must be **≈ 0.95** (range 0.946–0.951) to give 0.97988, which is ~0.005 of the 0.0072 gap.
  - **What France's uncertain pairs look like** (28 sampled, p 0.2–0.8):
    - S1 addresses end with the region (nouvele aquitaine, hauts de france, pays de la loire), while queries end with the department (gironde, nord, pas de calais) or nothing;
    - queries abbreviate street types (r = rue, al/ale = allée) and saint (st herblain);
    - spaced legal forms ("s a", "s n c") are not joined into sa/snc;
    - one French generic word is swapped (club/union/comite/centre/culturel).

    Ruled out: number-shift distractors (France has *fewer* shifted-number links, 0.61%) and leading zeros (already stripped in `addr_nums`: 0.0000 everywhere).
- **LB:** n/a (analysis).
- **Takeaways:** the gap is mostly France, an unseen country whose address and legal formats the train-fit normalization never saw, plus ~0.001 from test's higher hard-distractor density. OOF stays a valid guide for US/India changes, but France changes can only be judged on the LB. More US/India modelling is worth ≤ 0.0006 (2× data gave +0.0006).
- **Next:** R08 = France-only normalization:
  - drop region/department names from addresses;
  - expand French street-type abbreviations and st → saint;
  - join spaced single-letter legal forms.

  Recompute test France features on the **same** candidate pairs (`candidate_pairs.tsv` stays valid), re-score with x1/x2, blend and upload. The pools and OOF are unchanged.
