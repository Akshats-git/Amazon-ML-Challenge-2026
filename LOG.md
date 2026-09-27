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
| R04 | 26 Sep 02:28 | ber_v1 FULL scale on laptop: tier-0 (stage 1, P0 model, T=0.76) | 0.97623 (P1 OOF) | R@10 US 0.9909 / IN 0.9842 | not uploaded | – | `output_tier0/` validated; superseded before upload |
| R05 | 26 Sep 04:52 | + 14 edit/number side features (xfeats), stages 1+2, tuned (T1,T2)=(0.48,0.76) | **0.98645** (pooled OOF) | same as R04 | not uploaded | – | `output_r05/` validated; x1 run, part of the R06 blend |
| R06 | 26 Sep 09:09 | R05 with 2× training data (TRAIN_QUERY_FRAC 0.30, float16 + lgb.Sequence), tuned (0.50,0.76); + blend x1+x2 (0.52,0.74) | **0.98705** / blend 0.98706 | same | **0.981021** (`output_blend/`) | **301** | `output_blend/` submitted 26 Sep ~17:30; **best LB**; beats R07 by +0.00114 |
| R07 | 26 Sep 09:36 | R05+R06 blend + France lexicon imputation (test-only, label-free) | **0.98706** (= blend) | same | **0.979881** | **146** | `output_blend_fr/` submitted 26 Sep ~10:00; **CV→LB gap −0.0072** |
| R08 | 26 Sep 15:15 | R07 + France-only normalization (street types, regions/departments, spaced legal forms, et→and) | 0.98706 (= R07; France not in CV) | same (no re-blocking) | 0.979 (portal shows 3 dp; below R07 0.980) | team rank 299 (best still R07) | `output_r08/` submitted 26 Sep 16:38; **did not beat R07** |
| R09 | 26 Sep 20:05 | R06 France reproduced exactly (VM) + label-free same-address rescue (unseen countries) + per-source caps | 0.98706 (US/India unchanged; rescue −0.0012 if applied to US/India OOF, so it is off there) | same | **0.976531** | > 500 for this file (team rank still from R06) | `output_r09/` uploaded 26 Sep ~20:10; **−0.0045 vs R06: rescue rejected** |
| R10a | 26 Sep 20:24 | + 16 number-relation / number-oracle-lexicon features (nfeats); stage-1 parameter A/B on 10% of P0 | holdout log-loss 0.004202 → **0.003854** (−8.3%) | – | – | – | analysis; x2 params kept (255/511 leaves worse) |
| R10c | 26 Sep 21:31 | France address-only normalization (R08c) on R06 (laptop) | 0.98706 (France not in CV) | same | not submitted | – | extra links fall in the R09-type cells; rejected |
| R10h | 26 Sep 22:55 | **x3 (+16 nfeats, all queries) for US/India** + R06's France rows (hybrid) | **0.98838** (x3; US 0.98977, India 0.98630) | same | **0.982401** | **320** | `output_r10h/` uploaded 26 Sep ~23:10; **new best LB, +0.00138 vs R06** |
| R10d | 27 Sep 01:05 | x4 (255-leaf variant, all queries) + blend x3+x4; hybrid `output_hyb34` = x3+x4 US/India + R06 France | **0.98848** (x3+x4; x4 alone 0.98840) | same | **0.982556** | **434** | `output_hyb34/` uploaded 27 Sep (#2 of the day); **new best LB; the final file** |
| R10x3 | 27 Sep 01:30 | full x3 file: x3 for all countries (= R10h with x3's France rows) | 0.98838 | same | **0.982123** | not reported (team rank from R10h) | `output_x3/` uploaded 27 Sep ~01:30; **−0.00028 vs R10h: x3's France is worse than R06's** |
| R11 | 27 Sep 01:50 | lexicon-dropout A/B for US/India (10% of P0, unmasked holdout): 0.5 / 0.3 / 0 | log-loss 0.003804 / 0.003813 / 0.003804 | – | – | – | analysis; no difference, no retrain |

**Current best on LB:** **R10d `output_hyb34/` = 0.982556 public, rank 434** (uploaded 27 Sep, #2 of the day; x3+x4 blend for US/India + R06's x1+x2 France). This is the **final file**. Uploads so far: R07 0.979881, R08 0.979, R06 `output_blend/` 0.981021, R09 0.976531, R10h 0.982401 (rank 320), R10x3 `output_x3/` 0.982123, R10d `output_hyb34/` 0.982556 (rank 434; the rank falls while the score rises because the field keeps improving). Every France change (R07 imputation, R08 normalization, R09 rescue, x3's France predictions) lowered the LB; the US/India model gains transferred (x3: +0.00138 LB for +0.0013 OOF; x3+x4: +0.00016 for +0.0001). 3 uploads left on 27 Sep.

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
- **LB (tier-0):** never uploaded (superseded by R05/R06 before an upload slot was used).
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
- **LB (R05):** never uploaded on its own (its x1 run is part of the R06 blend, which was uploaded).
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
- **LB (R06):** `output_blend/` (x1+x2 blend, no France changes) submitted 26 Sep ~17:30 IST: **0.981021 public, rank 301** (26 Sep ~19:30; the field keeps improving, so rank fell from 299 despite our better score). This is **+0.00114 over R07** (0.979881), so R07's France lexicon imputation *hurt*: its 22.4k removed France links were mostly true matches. R08 (0.979, built on R07) hurt as well.
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
- **Post-LB (26 Sep 17:30):** **wrong.** `output_blend/` (without this fix) scored 0.981021 vs 0.979881, so the appended French words (france, groupe, développement, …) mark *true* variants more often than distractors. The label-free "generator profile" is not evidence of a distractor in France.
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
  - **France distractors use the same perturbation as US/India** (`work/logs/diag_fr.log`). Share of blocking top-1 pairs with a house number shifted by 1–20:
    - test: France 0.272, India 0.271, US 0.297;
    - pools: distractor queries 0.51 (India) / 0.38 (US), true queries 0.02.

    So the generator shifts numbers in France too, and the model already rejects those.
  - **How common each France format issue is** (share of records, S1 vs S2/S3). These are what hurt France:

    | issue | S1 | S2/S3 |
    |---|---|---|
    | `r` for `rue` | 0% | **25%** (S1 `rue` 66%, S2/S3 38%) |
    | bd/av/pl/ch/imp/rte/al | 2.5% | 10% |
    | `st`/`ste` for saint(e) | 0.1% | 1.8% |
    | spaced legal forms (`s a`, `s n c`, …) | 0% | 5.2% |
    | `et` in names (S1 writes `and`) | 0% | 1% |

    Address endings differ as well:
    - S1 addresses end with the **region**: hauts de france 88k, nouvele aquitaine 74k, pays de la loire 63k (of 259k).
    - S2 addresses end with a **department**, a region or the city. Departments: gironde 65k, nord 65k, loire atlantique 55k, pas de calais 12k.
- **LB:** n/a (analysis).
- **Takeaways:** the gap is mostly France, an unseen country whose address and legal formats the train-fit normalization never saw, plus ~0.001 from test's higher hard-distractor density. OOF stays a valid guide for US/India changes, but France changes can only be judged on the LB. More US/India modelling is worth ≤ 0.0006 (2× data gave +0.0006).
- **Next:** R08 = France-only normalization:
  - drop region/department names from addresses;
  - expand French street-type abbreviations and st → saint;
  - join spaced single-letter legal forms.

  Recompute test France features on the **same** candidate pairs (`candidate_pairs.tsv` stays valid), re-score with x1/x2, blend and upload. The pools and OOF are unchanged.

## R08 · France-only normalization (test-only, no retraining) · 26 Sep 15:15
- **Approach:** France format rules in `ber/normalize.py` (`fr_name`, `fr_addr`), gated on `country == "France"` and `FR_NORM` (env, default 1; `FR_NORM=0` = R07). Mappings were mined from confident France top-1 pairs (blend p ≥ 0.9; `work/logs/diag_fr_maps.log`, q token → S1 token, purity ≥ 0.97):
  - **Address:**
    - drop the regions/departments `hauts de france`, `nouvele aquitaine`, `pays de la loire`, `loire atlantique`, `pas de calais`, `gironde`, `nord` at any position, on both sides. Queries often put the department **first**; S1 has nord/gironde in < 200 rows;
    - drop the number prefixes `no`/`ndeg` (N°); `ndeg12` → `12`;
    - expand abbreviations to S1's tokens: r→rue, av/ave→avenue, bd/blvd→boulevard, pl→place, ch/chem→chemin, imp→impase, rte→route, al→ale, crs→cours, q→quai, res→residence, psg/pas→pasage, apt/ap→apartement, st→saint, ste→sainte; `b`/`t` after a number → bis/ter;
    - strip leading zeros from number tokens.
  - **Name:** join runs of ≥ 2 single letters (`s a r l` → `sarl`), then map et→and, compagnie→cie, frs→freres, st→saint, center→centre, cb/clb→club, svc→service, farmacie→pharmacie.
  - **Rebuilt, all on the same candidate pairs (no re-blocking):**
    - `norm --splits test` (74 s). US/India rows verified byte-identical by row hashes; row order and IDs unchanged. France rows changed: S1 259,452 (all), S2 579,565, S3 608,557;
    - test France features (113 s), then test xfeats (3 min; same 7 France words raised to 9.16);
    - re-scored x1/x2 stages 1+2 (21 min), then `blend --tags x1 x2` into `output_r08/` (4 min).

    Chain: `work/logs/chain_r08.sh`. R07 France p2 backup: `work/bak_r07/`.
- **Data:** laptop; test only; models, pools, OOF and thresholds unchanged.
- **Results:**
  - **Text agreement on confident France pairs** (816k, `diag_fr_check.log`):
    - address exact 0.112 → **0.634**, same token set 0.159 → 0.775, mean Jaccard 0.638 → **0.913**;
    - name exact 0.476 → 0.527, Jaccard 0.761 → 0.797;
    - the remaining address diffs are generator typos (aenue, jen, sait), as in US/India.
  - **OOF** 0.98706 at (0.52, 0.74) (unchanged). `output_r08/matching_results.tsv`: 5,797,008 links. Validator `--check-ids` **PASS**; matches ⊆ candidates.
  - **Links/S1:** France **3.2364** (R07 3.2255), empty share 5.95% (6.03%); India 3.3487 and US 3.3855 unchanged.
  - **Diff vs R07:** France −10,218 / +13,028 links; US/India identical.
  - **diag_lb (test France), R07 → R08:**
    - uncertain share (p 0.1–0.9) 0.0816 → **0.0925** (per S1 0.389 → 0.450);
    - selfR 0.9634 → 0.9566;
    - self-est. F 0.98233 → **0.98033**, i.e. *worse* on paper.
  - **Why the self-estimate fell** (`diag_fr_andfils.log`, `diag_fr_shift_r08.log`):
    - **`et`→`and` exposes the "& Sons / & Associates" distractor pattern.** In the train lexicon, an extra query token `and` is true in 126 of 111,322 cases; `asociates` 7 of 62k, `sons` 1 of 17.5k. In France, 31k argmax queries have an extra `and` vs their S1, mostly `… et fils` / `… et associés` appended at an identical address. Under R07, `et` was unseen and 23% of these had p ≥ 0.9. Now 0.7% do, and links at p ≥ 0.52 fall from 9,393 to 5,837. That is the intended effect, but the plug-in self-F counts the new doubt as lost recall.
    - **The other 1.40M queries:** p ≥ 0.52 rises from 846,099 to **857,379** (+11k), and p ≥ 0.9 from 0.576 to 0.582. Uncertainty rises slightly (0.0666 → 0.0725), mostly previously rejected shifted-number or swapped-word distractors that now sit at p 0.1–0.5, below T1.
- **LB:** submitted 26 Sep 16:38 IST: **0.979** (the submissions page shows 3 decimals; R07 shows 0.980). Our best is still R07 0.979881. Team rank fell 146 → **299** by 17:00, mostly because other teams improved; the top 3 are now 0.990621 / 0.989475 / 0.988842. The portal never showed R08's 6-decimal score.
- **Post-LB:** R08 did not help and probably hurt slightly (≥ −0.0001 to −0.001 at 3 dp). Suspects, in order:
  1. `et`→`and` flips ~3.5k "et fils / et associés" links, which may be true variants in France rather than distractors;
  2. removing the region/department changes address similarity for shifted-number distractors (+25k queries moved from p < 0.1 to 0.1–0.9).

  Isolate with an address-only variant before any further France work.
- **Takeaways:** normalization closes most of France's format gap: address Jaccard 0.64 → 0.91 on confident pairs. The label-free self-F criterion cannot judge R08, because it rewards the old model's confident errors on "et fils" appends, so the LB has to decide.
- **Next:** upload R08. If it helps, iterate: map the French generator suffixes to the train vocabulary (fils→sons, asocies→asociates), check remaining name appends (`services`, `cie`), then optionally re-block France.

## R08b · where the remaining loss is (analysis only, no new file) · 26 Sep 17:15
- **Approach:** label-based OOF analysis of the x1+x2 blend (P0/P1) to decide the next steps after R08 missed. Logs: `work/logs/diag_ties.log`, `diag_fn_samples.log`, `diag_nonlatin.log`, `diag_fn_buckets.log`, `diag_order.log`.
- **Data:** laptop, existing OOF arrays and norm tables.
- **Results:**
  - **OOF loss is recall-bound:** P 0.9965, R 0.9698. FNs: 230k = 90k blocked out (39%), 90k correct argmax below threshold (39%), 51k other S1 won (22%). FPs: 37k.
  - **Ties are not a factor:** none at p ≥ 0.52.
  - **Empty query address dominates the errors** (P0):
    - US: 45% of below-threshold FNs and 38% of FPs, vs 3.2% of TPs; same name + empty address is 26% of FNs and 24% of FPs;
    - India: 37% of FNs and 18% of FPs.

    These are same-name S1s the model cannot choose between without an address.
  - **Other FN types:**
    - name fully replaced by a coined word (`synex`, `umbranex`, `korvantage`) at the same address: ~11% of FNs;
    - true pairs with a shifted/mistyped house number: nums equal in only 5% of US FNs.
  - **Translit dict bug:** `work/dicts/addr.json` maps `no`, `4`, `c`, `d`, `urban`, … → `patna` (and `h` → `flat`) for non-Latin records. It does little harm, though: non-Latin India queries have *higher* recall at T1 (0.980) than Latin ones (0.967). Not worth a retrain.
  - **No ordering leak:** IDs are random, and rank corr(s1_row, q_row) over true pairs is 0.0.
- **LB:** n/a. Top 3 at 17:00 are 0.9906 / 0.9895 / 0.9888. Even our US/India CV (0.9885 / 0.9849) is below them, so the leaders are ahead on all countries, not only France.
- **Takeaways:** the big lever is recall for US/India too: blocking misses and empty-address/same-name ambiguity. France alone cannot close a 0.011 gap.
- **Next:** see the brainstorm in chat (26 Sep 17:15): R08c address-only ablation, a query↔query (S2↔S3) sibling signal for empty-address queries, and larger/extra blocking.

## R08c · research: generator mechanics + France recall deficit (analysis only, no new file) · 26 Sep 19:10
- **Approach:** label-based OOF analysis (P0/P1) plus label-free test analysis, using the house-number shift as a distractor oracle. Also: web research (entity-resolution state of the art, Foursquare Location Matching winners, other public repos for this challenge). Logs: `work/logs/diag_signed_shift.log`, `diag_shift_errors.log`, `diag_ambiguity.log`, `diag_err_samples.log`, `diag_shift_test.log`, `diag_gen_joint.log`, `diag_num_align.log`, `diag_fr_words.log`, `diag_fr_proxy.log`, `diag_linkrate.log`, `diag_fr_eqother.log`, `diag_fr_swaps.log`, `diag_fr_samestreet.log`, `diag_oof_samestreet.log`, `diag_fr_sibling.log`, `diag_gt_structure.log`.
- **Data:** laptop; existing OOF/test arrays, norm tables and the three uploaded matching files. (A 7 GB analysis rebooted the laptop at ~18:20; the reruns were memory-lean, one country per process.)
- **Results:**
  - **Distractor generator:** copy an S1 record and shift its house number **upward by one of {1, 2, 3, 4, 5, 7, 9, 11, 13, 21}**, each ~equally likely (~42–56k each in US top-1 pairs; +6/+8/+10/+12 only ~1k). Optionally also perturb the name: append a word from a per-country list, swap a word (US: industry words), or change the legal form. **France uses the identical shift set** (+GEN share of top-1 pairs: France 0.274, US 0.264).
  - **True-match number noise** is symmetric (±1, ±2, ±10, ±20), plus dropped digits and digit edits. So the *sign* separates them, and our features only use |diff|:
    - US top-1 pairs, P(true): −1/−2 ≈ 0.80, +GEN ≈ 0.025;
    - numbers equal + "+1 word": **97.8% true**; numbers equal + same name core: 99.9%.
  - **Where OOF errors sit (argmax level, P0+P1):**
    - empty query address: US 14.4k FN / 9.6k FP, India 7.8k / 4.1k (≈ 45% of errors; mostly names shared by 2+ S1s, i.e. ambiguous);
    - +GEN-aligned numbers: 7.8k FP;
    - −1/−2 and −3…−21 aligned: 6k FN;
    - digit edits: 3.2k FN / 3.6k FP.

    The number-relation features could reach roughly +0.0005 CV.
  - **GT structure:** US and India have identical generator statistics: per S1, S2 matches mean 1.67 (max 5), S3 matches mean 1.79 (max 6), 5.58% singletons, 80.5% with both sources. R06 output violates the caps for only 74 S1s.
  - **France, label-free:**
    - The France generator words (+GEN pairs) are développement, participations, groupe, holding, international, distribution, france (~27k each).
    - Of these, **france / groupe / développement are also true-match noise** (≈1.4k each in equal-number pairs, like US "partners"); participations / holding / international / distribution are pure generator words. R07 raised all seven, which removed true links.
    - A proxy on the uploaded files matches the LB order: R06 → R07 removed 3.7k "equal + 1 word" and 16.2k "equal + edited name" France links (≈ 92–98% true in train); R08 added 280 +GEN links.
  - **France recall deficit.** For same street/city + identical house number, R06 link rates:

    | name relation | US | India | France |
    |---|---|---|---|
    | single word swapped | 93% | 97% | **68%** |
    | other edits, shared word | 99% | 99% | **77%** |
    | no shared word | 96% | 95% | **84%** |

    OOF US/India true rates in the same cells are 95–99%, and the model matches them. The unlinked France swaps are organisation words (club↔comité↔amicale↔école↔sportive…) and "& Fils / & Associés". The evidence says they are **true noise, not sibling organisations**:
    - these words appear at equal numbers at ~25% of their +GEN rate, while pure generator words appear there at ≈ 0;
    - their exact names repeat across sources only 0.34% of the time (linked true variants: 0.78%);
    - S1 same-address sibling rates are equal for linked (15.0%) and unlinked (16.6%) pairs.

    Expected France true matches ≈ 3.46/S1 (same generator) ≈ 898k, vs R06's 858.6k links, so France is missing ≈ 40–48k true links (recall ≈ 0.947). This fits France ≈ 0.962 back-solved from the R06 LB.
  - **Web research:**
    - other teams' public repos for this challenge report ≤ 0.976 validation (inverted-index blocking, 32-feature LightGBM);
    - Foursquare Location Matching winners: staged LightGBM (+ XLM-R cross-encoders for 1st place; 7th place used none) with FP-weighted training and graph post-processing;
    - entity-resolution literature: fine-tuned small LMs (Ditto/AnyMatch) lead on generic benchmarks, but our remaining errors are numeric/structural, and the GTX 1650 cannot score millions of pairs with a cross-encoder in the time left.
- **LB:** n/a (analysis). The challenge allows 5 uploads/day.
- **Takeaways:**
  - France's gap is **recall on same-address variants whose name edits look like US/India distractor edits**; precision on distractors is already excellent (+GEN link rate ≤ 1.6%).
  - The fix is to trust the number mechanics (equal house number + same street ⇒ not a generator distractor), not new name rules.
  - For US/India the remaining lever is number-relation features plus more data; the model class (LightGBM) is right.
- **Next:** the plan is in `docs/handoff_R09.md` (26 Sep 19:30): JarvisLabs 32-vCPU VM (₹1000 budget); tonight R06 France reproduction + label-free France same-address rescue (uploads #1/#2); x3 with number-relation + number-oracle-lexicon features on 50–60% of the data, launched as soon as the features exist (steps chained back-to-back on the VM, no overnight waiting); package by Sun 17:00.

## R09 · R06 France reproduced on the JarvisLabs VM + same-address rescue (unseen countries) · 26 Sep 20:05
- **Setup (Step A):** JarvisLabs CPU VM `517575` (32 vCPU AMD EPYC 9555, 125 GB RAM, 94 GB disk, IN1, ₹64.28/h; user `ubuntu`, ssh alias `jlber`). Laptop watchdog `work/vm/watchdog.sh` (every 10 min: pause below ₹120 balance; pause after ~40 min idle). Repo + `work/` (16.3 GB) uploaded in ~9 min (~30 MB/s). Env: uv + pinned requirements (+ `libgomp1`), identical versions to the laptop; `env`/`check1`/`check2` PASS. **Reproducibility:** x2 stage-1 re-score of test France file 0 on the VM vs the laptop array: max |diff| 3.0e-8.
- **Approach:** new flags (all env): `FR_NORM` now defaults to 0 and splits into `FR_NORM_ADDR` / `FR_NORM_NAME`; `FR_IMPUTE` (default 0) gates the R07 lexicon imputation; `FR_RESCUE` (default 1), `CAPS` (default 1). Chain `work/logs/chain_r09a.sh` on the VM: `FR_NORM=0` test norm (US/India rows byte-identical to before), test France feats (33 s), xfeats (no imputation fired), x1/x2 France re-score (5.7 min), blend.
  - **Rescue** (`ber/rescue.py`), countries absent from train only: on the query's argmax pair, if the S1 has ≥ 1 number, the number multisets are equal and the alphabetic address-token Jaccard is ≥ 0.8 (addresses compared under the R08 France address rules, whatever the features use), and no other candidate S1 of the query also satisfies this (sibling guard), set p = max(p, 0.90). **Caps:** ≤ 5 S2 / ≤ 6 S3 links per S1 (highest p kept).
- **Data:** VM; test only (models, pools, OOF unchanged).
- **Results:**
  - **R06 reproduced exactly:** `FR_RESCUE=0 CAPS=0` blend = `output_blend/` (0 links removed, 0 added; 5,815,887 links; France 3.3091 links/S1). OOF 0.98706 at (0.52, 0.74).
  - **R09 = R06 + rescue + caps:** 5,866,412 links (+50,684 France; caps removed 347 France / 30 India / 28 US). France argmax same-address rows 635,412 (sibling-guarded 2,462), rescued 632,950, **raised 61,418** (p quartiles 0.011 / 0.126 / 0.579). France links/S1 3.309 → **3.504**, empty share 5.82% → 4.95%. Validator `--check-ids` PASS; candidate check OK (matches ⊆ candidates).
  - Label-free France cells (`work/diag/fr_cells.py`, R08 address rules): same street + equal numbers, link rate R06 → R09: 1 swap 0.677 → 0.945, other 0.817 → 0.938, +1w 0.990 → 0.999; **all +GEN cells unchanged** (≤ 0.016). Added links: 1 swap 25.1k (club↔comité↔amicale↔école↔sportive… swaps), no shared word 14.3k (coined names), other 11.1k; 2.3k go to S1s that were empty in R06.
  - **Evidence for** (label-free NOL counts on test France rank-1 pairs, `ber/nfeats.py`): pure generator words appear at equal numbers only ~2% as often as at +GEN shifts (holding 515 vs 26,109), organisation words ~50% as often (club 3,932 vs 7,596), so equal-number org swaps are mostly true noise, not the distractor edit. The R06 → R07 LB drop came from removing links in exactly these cells.
  - **Evidence against:** France links/S1 3.50 is now above US 3.39 / India 3.35 although France has fewer queries per S1 (5.53 vs 5.76 / 5.82), and its empty share (4.95%) is below the generator's 5.58% singleton rate. On US/India OOF the same rule *lowers* pooled F 0.98706 → 0.98583: there the firing rows the model rejects (p < 0.1) are 99.5% distractors that kept their house number (~0.5–0.8% of firing rows; `work/logs/diag_oof_rescue.log`). So the rule is only right if France's rejections are vocabulary misfires, which the LB must decide.
- **LB:** **0.976531** (26 Sep ~20:15; this file alone ranks > 500, exact rank not shown; team rank still set by R06). **−0.00449 vs R06.**
- **Takeaways:**
  - **The rescue is wrong; the "France recall deficit" of R08c is not real on these pairs.** Back-solving (ΔF ≈ 0.21·ΔR − 0.79·ΔP on France's 15% of S1) gives only ~25% true among the 50.7k added links, about what the model's own p said (raised rows: median p 0.13). So France's model is roughly calibrated on same-address pairs and its rejections were right: France's generator makes many distractors that keep the house number and swap the organisation word or replace the name. The singleton/empty-share and links/S1 monitors were the right warning.
  - The flags reproduce R06 exactly, and the VM reproduces laptop scores.
- **Next:** `FR_RESCUE` default is now 0 (the code stays for the record). No more France rules; France changes only through the model (x3's label-free nfeats/NOL) and the normalization variant R08c (address only, no rescue).

## R10a · number-relation + number-oracle-lexicon features (nfeats) and the x3 parameter A/B · 26 Sep 20:24
- **Approach:** new module `ber/nfeats.py`, 16 label-free features stored row-aligned in `work/nfeats/` (gated by `USE_NFEATS=1`; x1/x2 still load and score unchanged because scoring now takes the column list from each booster's own feature names).
  - **Number relations (12):** q_only / s_only = numbers one address has and the other lacks; every (q_only, s_only) pair is compared as q − s. `nr_all_equal`, `nr_n_shared`, `nr_gen_pos` (q − s ∈ GEN = {1,2,3,4,5,7,9,11,13,21}), `nr_neg_small` (−1/−2), `nr_neg_gen` (−GEN), `nr_signed_min`, `nr_digit_edit` (Levenshtein 1), `nr_transposed`, `nr_len_diff`, `nr_street_jac` (alphabetic address tokens), `nr_eq_same_street`, `nr_gen_same_street`. Unit-tested on hand pairs (823→844 = +21 GEN; 3984→3982 = −2; 12↔21 transposed and +9; dropped digit = Lev 1, len −1).
  - **Number-oracle lexicon (4):** per partition (P0, P1, each test country), on blocking rank-1 pairs: D = single +GEN shift on the same street (distractor proxy), T = equal numbers on the same street (true proxy); nol(w) = log((cD+1)/N_D) − log((cT+1)/N_T) per kind (extra/missing name_core token), NaN unless cD + cT ≥ 20; features `nol_extra_max/min`, `nol_missing_max/min`. Replaces the R07 imputation (label-free, no cross-fitting needed).
- **Data:** VM (32 vCPU); P0, P1, test = 230M pairs in 18.7 min (~200k pairs/s; laptop: test France 14.3M pairs in 53 s).
- **Results:**
  - NOL mirrors the supervised lexicon without labels: US top extra eastgate 8.4, riverside 7.8, holdings 7.3, group 7.3; India pharmacy/general/stores/overseas/exports ≈ 7; bottom dba −8, formerly −8, aka −7 everywhere. Test France (R06 norm): international / participations / distribution 4.7, holding 4.6, développement / groupe / france 2.2–2.3; with the R08 address rules (laptop check) organisation words (club, école, amicale, comité …) come out at +1.3–1.4 and fils / "and" / services at −0.4 to −0.7.
  - **A/B, stage 1, 10% of P0's queries** (5.84M train + 0.65M holdout rows, lexicon dropout 0.3, lr 0.05, early stopping; `work/logs/ab_params.py`):

    | features | params (leaves / min_child / ff / l2) | best_iter | holdout logloss | s/iter |
    |---|---|---|---|---|
    | 64 (x2) | 127 / 100 / 0.8 / 1 | 688 | 0.004202 | 0.069 |
    | **80 (+ nfeats)** | **127 / 100 / 0.8 / 1** | 563 | **0.003854 (−8.3%)** | 0.078 |
    | 80 | 255 / 100 / 0.8 / 1 | 360 | 0.003883 | 0.111 |
    | 80 | 255 / 200 / 0.6 / 5 | 421 | 0.003881 | 0.121 |
    | 80 | 511 / 200 / 0.8 / 5 | 321 | 0.003929 | 0.153 |
    | 80 | 255 / 50 / 0.6 / 0 | 352 | 0.003942 | 0.113 |

    Top gain with nfeats: blk_rank 0.519, blk_score 0.199, **nr_len_diff 0.066** (3rd), nx_num_unm_affix 0.035, addr_me_q2s 0.029, blk_gap 0.027, **nr_gen_pos 0.020**, **nol_extra_max 0.014**.
- **Takeaways:** the sign/shift-set number features are worth more than 2× data was (R06: −5.5% logloss); bigger trees do not help at this size. At 0.078 s/iter on 5.8M rows the VM can train on **all** queries.
- **Next:** x3 = 80 features, x2 params, `TRAIN_QUERY_FRAC=1.0`, `MASK_LEX_FRAC=0.5` (lean on NOL where the supervised lexicon is blind), picked automatically (`work/logs/pick_x3.py`) and launched at 20:24 (`chain_r10b.sh`: train1 → train2 → tune → predict → blends x2+x3, x1+x2+x3).

## R10 diag · France thresholds, label-free (analysis only) · 26 Sep 20:35
- **Approach:** after R09 showed France's model is roughly calibrated, test whether France wants other thresholds: plug-in expected macro F0.5 per (T1, T2) grid, treating p (x1+x2 blend) as calibrated (`work/diag/plugin_T.py`); on the OOF pools the plug-in optimum is compared with the label-based one.
- **Results:** plug-in optimum (0.46, 0.74) on all four pools (label-based optima (0.42–0.52, 0.70–0.76); surfaces flat within 2e-5); **test France (0.46, 0.72)**, India (0.46, 0.74), US (0.46, 0.72). Plug-in F: France 0.98268 (vs 0.98262 at the used (0.52, 0.74)), India 0.99016, US 0.99166 (the plug-in level is ~0.007 optimistic on the pools).
- **Takeaways:** no label-free case for France-specific thresholds; the `T_UNSEEN` probe is dropped (the flag stays).

## R10b · x3 = + nfeats, all queries (VM) · 26 Sep 20:24 → 23:00
- **Approach:** tag `x3`: 80 stage-1 features (50 base + 14 xfeats + 16 nfeats), x2's LightGBM params (A/B winner), **`TRAIN_QUERY_FRAC=1.0`** (every query of the pool: ~58.5M train + 6.5M holdout pairs per model, float32, 19.4 GB), `MASK_LEX_FRAC=0.5`; stages 1 + 2 cross-fitted, (T1, T2) tuned on pooled OOF; test France = R06 normalization; no rescue; caps on. Chain `work/logs/chain_r10b.sh` (auto-launched by `chain_r10b_auto.sh` from the A/B).
- **Data:** VM (32 vCPU / 125 GB).
- **Results so far:**
  - Stage-1 P0 model: best_iter **3000 (the round cap; still improving, 2500 → 3000: −0.25%)**, holdout log-loss **0.00323** (x2: 0.00394, −18%), 20 min. P1 model: best_iter 2987, 0.00327, 18 min. Top gain: blk_rank 0.52, blk_score 0.20, nr_len_diff / nx_num_unm_absdiff 0.066–0.069, nx_num_unm_affix 0.035, nr_gen_pos 0.020, nol_extra_max 0.014–0.015.
  - **Stage-1 pooled OOF: F@0.76 0.98673, best single threshold 0.98681 @ 0.70** (x2 stage 1: 0.98515 / 0.98521; **+0.0016**). FP 34,233 (x2: 44,638, −23%); FN 244,754: 90,189 blocked out, 100,233 below threshold, 54,332 other S1 won. F_noFP 0.98944, F_allFN_blocked_in 0.99353. train1 took 60.5 min (scoring 65M pairs with a 3000-tree model ≈ 8.7 min).
  - Stage 2: best_iter 1987 / 1997 (cap 2000), holdout log-loss **0.00266 / 0.00270** (x2: 0.00312 / 0.00315, −15%), 11 min per fit; train2 took 36 min. OOF p2 F@0.76 0.98816; best single threshold 0.98822 @ 0.70.
  - **Tuned (T1, T2) = (0.48, 0.74): OOF macro F0.5 = 0.98838** (x1+x2 blend 0.98706: **+0.0013**; gate +0.0003 passed). **US 0.98977** (x2 0.98848), **India 0.98630** (x2 0.98490). Precision 0.99705, recall 0.97193.
    - Buckets: n=0 0.9853, n=1 0.9637, 2–3 0.9885, 4+ 0.9914.
    - FP 30,173 (blend 36,833, −18%; 27,949 from queries without a true candidate). FN 214,425 (blend 230,460): 90,189 blocked out, **74,267 below threshold (blend 89,533, −17%)**, 49,969 other S1 won. F_noFP 0.99115, F_allFN_blocked_in 0.99347.
- **Test (x3, `output_x3/`, 23:00):** 5,831,939 links; links/S1 France 3.3627 (R06 3.3091), India 3.3494, US 3.3878; France empty share 5.56%. Validator PASS. Test scoring took 43 min (100M pairs, 2 × 3000-tree stage-1 + 2 stage-2 models). **France cells (`fr_cells.py`):** links on +GEN-shifted pairs fall in every cell (1 swap 0.0035 → 0.0022, other 0.0075 → 0.0043, same 0.0156 → 0.0132), but equal-number same-street 1 swap 0.677 → 0.729 and other 0.817 → 0.865 (+~11k links, the cells where R09's additions were mostly false), so x3's France is an open LB question.
- **Blends (OOF, re-tuned):** x2+x3 **0.98818** at (0.46, 0.72); x1+x2+x3 0.98788; both **below x3 alone (0.98838)**, so the weaker runs dilute x3.
- **Upload #2 = `output_r10h/`** (`work/diag/hybrid.py`: x3's decisions for US/India, R06's France rows unchanged; the pipeline equivalent is `BLEND_TAGS_UNSEEN=x1,x2 blend --tags x3` → `output_hyb3/`). vs R06: US −6,588 / +8,117 links, India −11,953 / +12,573, France identical; 5,818,036 links. Validator `--check-ids` PASS, candidate check OK. It isolates the US/India gain of x3 on the LB (expected ≈ +0.0011 if CV transfers).
- **LB:** **0.982401, rank 320** (26 Sep ~23:15). **+0.00138 vs R06 (0.981021)**, a little more than the OOF gain predicted (US/India are 85% of S1: +0.0013 × 0.85 ≈ +0.0011), so US/India OOF transfers to the LB.
- **Takeaways:** the generator-aware number features + all the data are the biggest single step since R05. The public LB rank still slides (301 → 320) because the field improves.
- **Next (27 Sep, 5 uploads):** full x3 (x3's France rows) to measure x3's France effect; x4 (255 leaves) and its blend with x3 by CV; then choose France by LB and US/India by CV.

## R10c · France address-only normalization "R08c" (laptop, not submitted) · 26 Sep 21:31
- **Approach:** R06 France with the R08 **address** rules only (`FR_NORM_ADDR=1 FR_NORM_NAME=0`), no imputation, no rescue, caps on; built in the parallel work dir `work_fr8c/` (symlinks to the shared artefacts, so `work/` keeps R06 France). `work/logs/chain_r10c.sh` on the laptop (6 workers, 34 min).
- **Results:** France addresses changed for 259k S1 / 568k S2 / 598k S3 rows, names and US/India rows unchanged. `output_r10c/`: 5,823,022 links (France +9,871 / −2,678 vs R06; links/S1 3.309 → 3.337, empty share 5.82% → 5.70%). Validator PASS. France cells (`fr_cells.py`): the gains sit in the equal-number same-street **1 swap** (0.677 → 0.705) and **other** (0.817 → 0.835) cells, exactly where R09's added links were ~75% false, and the +GEN cells rise slightly (0.0035 → 0.0039, 0.0075 → 0.0080).
- **LB:** not submitted.
- **Takeaways:** better address matching mostly helps France's same-address distractors look like matches; do not use it.

## R10d · x4 (255-leaf variant, all queries) + blends; hybrids; VM paused · 27 Sep 01:05
- **Approach:** tag `x4` = x3's 80 features and data (`TRAIN_QUERY_FRAC=1.0`, `MASK_LEX_FRAC=0.5`) with the A/B runner-up params: 255 leaves, min_child 200, feature_fraction 0.6, λ2 5, seed 43 (stage 2: x2 params, seed 43). Then OOF-tuned blends. Hybrids (`BLEND_TAGS_UNSEEN=x1,x2`: R06's France members and thresholds): `output_hyb23`, `output_hyb3` (VM), `output_hyb34` (laptop). Chain `work/logs/chain_r10d.sh`.
- **Data:** VM (32 vCPU), 2 h 7 min for x4 (stage 1 51 min, stage 2 30 min, tune 3 min, test 37 min); hyb34 on the laptop (5 min).
- **Results:**
  - x4 stage 1: best_iter **2098 / 1824 (early stopped)**, holdout log-loss 0.00323 / 0.00327 (= x3). Stage 2: best_iter 1997 / 1932, 0.00265 / 0.00270. **OOF 0.98840 at (0.50, 0.74)** (x3 0.98838). FP 30,479, FN 214,246.
  - **Blends (OOF, re-tuned):** **x3+x4 0.98848 at (0.56, 0.74)** (best; +0.0001 over either alone); x2+x3+x4 0.98843 at (0.50, 0.72). Test links/S1 for x3+x4: France 3.3578, India 3.3488, US 3.3869.
  - `output_hyb3` (pipeline twin of R10h) = R10h minus 17 capped France links. `output_hyb34` (x3+x4 US/India + R06 France): 5,816,895 links; vs R10h US −1,635 / +1,048, India −3,158 / +2,621, France −17 (caps). Validator PASS, candidate check OK.
  - All models, stage-2 arrays (x1–x4) and output files mirrored to the laptop (`work/vm/pull.sh`); the laptop's test norm / France artefacts are the R06 versions again (checksums equal to the VM's). `work_fr8c/` deleted (1.5 GB). **VM paused at 01:10 (balance ₹621.84).**
- **LB:** `output_hyb34/` = **0.982556, rank 434** (27 Sep, #2 of the day, after R10x3 showed x3's France is worse than R06's). **+0.000155 vs R10h** (x3 alone for US/India), in line with the +0.0001 OOF gain of the blend.
- **Takeaways:** x3+x4 for US/India (by CV) and x1+x2 for France (by LB) is the best combination found; it becomes the final file.

## R10x3 · full x3 upload: x3's France vs R06's France (LB test) · 27 Sep 01:30
- **Approach:** upload `output_x3/` (x3 for every country, thresholds (0.48, 0.74), caps). Its US/India rows equal R10h's, so the difference to R10h is **only France** (x3: −5,239 / +19,142 links vs R06 France; fewer links on +GEN distractor pairs, ~11k more equal-address links).
- **LB:** **0.982123** (27 Sep ~01:30; rank not reported, the team rank was set by R10h). **−0.00028 vs R10h.**
- **Takeaways:** x3's France predictions are worse than R06's (−0.0019 on France's 15% of S1). This agrees with R09: in France, equal house number + same street + a name swap is mostly a distractor, and x3 learned from US/India that such pairs are true. **France keeps the x1+x2 (R06) predictions** (`BLEND_TAGS_UNSEEN=x1,x2`); US/India use the best-CV model x3+x4.
- **Next:** upload `output_hyb34/` (x3+x4 US/India + R06 France) as the final candidate; package.

## R11 diag · lexicon dropout for US/India (laptop A/B, negative) · 27 Sep 01:50
- **Approach:** x3/x4 blank the supervised lexicon for 50% of training queries to help France, but France now uses x1+x2. Does the dropout cost US/India? A/B on 10% of P0 with x3's features and params, **holdout rows never masked** (as at test; new opt-in `MASK_HOLDOUT=0`), dropout 0.5 / 0.3 / 0 (`work/logs/ab_mask.sh`). P0 nfeats regenerated on the laptop for it (6.5 min, NOL identical to the VM's).
- **Results:** holdout log-loss 0.003804 (0.5) / 0.003813 (0.3) / 0.003804 (0): no difference (NOL carries the same signal without labels).
- **Takeaways:** no x5 retrain. Also, the paused VM could not be resumed at 01:35 ("CPUs not available", IN1 has no 32/16-vCPU capacity; IN2 has), which does not matter now: the laptop holds every artefact of the final.

## R12 · final choice, reproducibility and package dry run · 27 Sep 02:00 (LB 09:15)
- **Final file: `output_hyb34/matching_results.tsv`** = x3+x4 stage-2 average for US and India (thresholds (0.56, 0.74), tuned on the pooled OOF), the x1+x2 average for France (its own thresholds (0.52, 0.74)), per-source caps on, no France rules. **OOF 0.98848; public LB 0.982556, rank 434.** sha256 `74974c99…`.
- **Why:** US/India questions decided by CV (x3+x4 0.98848 > x4 0.98840 ≈ x3 0.98838 > x2+x3+x4 0.98843 > x2+x3 0.98818). France decided by the LB (R06's x1+x2 France beat R07/R08/R09 and x3's France).
- **Reproduction:** the code defaults (`FR_NORM=0`, `FR_IMPUTE=0`, `FR_RESCUE=0`, `CAPS=1`, `MASK_HOLDOUT=1`) plus the documented runs reproduce it: `README.md` / `run_all.sh` now build x1, x2, the x1+x2 blend, x3 (`USE_NFEATS=1 TRAIN_QUERY_FRAC=1.0 MASK_LEX_FRAC=0.5`), x4 (+ `LGB1_PARAMS`/`LGB2_PARAMS`) and the final `BLEND_TAGS_UNSEEN=x1,x2 blend --tags x3 x4`, which is exactly the command that produced `output_hyb34` on the laptop.
- **Package dry run (01:56):** `final_package.sh` with CHOSEN=output_hyb34 → validator `--check-ids` PASS, candidate check OK (1,732,544 rows, 99,695,099 ids, matches within candidates), zip 559.6 MB with `output/` (both TSVs), `code/business_entity_resolution/` (all modules incl. `nfeats.py`, `rescue.py`), `Documentation_template.md`; the zipped matching file is byte-identical to `output_hyb34`. Dry-run zip deleted. The script now refuses to package while a placeholder remains (only the doc's final-LB number is left).
- **Docs:** `docs/Documentation_BER.md` rewritten for the final (team SteinsGate; generator mechanics with the GEN shift set; nfeats + NOL; results table with LB column; "what did not help" incl. the R07/R08/R09 France rules and x3's France). `README.md` (new flags, nfeats step, x3/x4 runs, final blend, method paragraph) and `run_all.sh` updated.
- **VM:** paused since 01:10 (balance ₹621.84; storage only). A resume at 01:35 failed for lack of 32-vCPU capacity in IN1; nothing further needs it. All artefacts of the final are on the laptop (models x1–x4, stage-2 arrays, test nfeats, R06-France test artefacts; checksums verified).
- **Pending:** fill the final LB into the doc, build `SteinsGate_submission.zip` (`CHOSEN=output_hyb34 TEAM=SteinsGate bash work/logs/final_package.sh`), user uploads the zip + final matching file; destroy the VM only after the user approves.
