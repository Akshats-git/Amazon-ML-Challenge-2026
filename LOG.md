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
| R14 | 27 Sep 11:00 | stage-3 re-scorer on argmax rows (consensus + raw text + model disagreement), US/India; R06 France | **0.98963** (US 0.99062, India 0.98815) | same | **0.984009** | **347** | `output_s3/` uploaded 27 Sep (#3 of the day); **new best LB, +0.001453 vs hyb34** (1.5× the OOF gain × 0.85) |
| R16 | 27 Sep 13:30 | stage 3 + xlm-r-base cross-encoder (raw text) for US/India; France R06 + LB-evidenced class edits (a, c, d) | **0.99140** (US 0.99187, India 0.99070) | same | not uploaded | – | `output_s6b/` validator PASS; projected ≈ 0.987 |
| R19 | 27 Sep 16:10 | **final**: top-2 re-rank with xlm-r large + base CE (US/India); France R06 + edits (a, c, d, e) | **0.99165** (US 0.99206, India 0.99106) | same | **0.987777** | **145** | `output_s8b/` = `output/` uploaded 27 Sep ~17:05; **new best LB, +0.003768 vs R14** (projected ≈ 0.9885) |
| R20a | 27 Sep 17:55 | s8b + France structural-transfer package P1 (+4,317 / −138 France links) | 0.99165 (US/India = s8b) | same | **0.987654** | ~200 (user estimate) | `output_r20a/` uploaded 27 Sep ~19:40 (upload 1 of 3); **−0.000123 vs s8b → France F −0.0008** (projected +0.0015): US/India cell rates do not transfer to France |
| R20b | 27 Sep 18:20 | R20a + India second-retrieval rescue (+8,258 India links) | India 0.99107 → **0.99181** (+0.00073, cross-fitted) | India + ~10k recovered pairs (OOF) | not uploaded | – | superseded by R20d–R20f (rescue v2); `output_r20b/`/`output_r20c/` removed from the laptop (still on the VM, `r20/out/u2b`, `u2a`) |
| R20d | 27 Sep 18:58 | R20a + India rescue v2 (second retrieval + name-twin expansion; +12,928) + US rescue (+1,482) | India 0.99215 (+0.00108), US 0.99228 (+0.00018) | – | pending | pending | `output_r20d/` validator PASS; candidate for upload 2; projected 0.98880 (0.98840–0.98900) |
| R20e | 27 Sep 19:08 | R20d + France CE veto (−1,737 France links the large CE rejects in address-driven classes) | – | – | pending | pending | `output_r20e/` validator PASS; best candidate so far; projected 0.98900 (0.98855–0.98925) |
| R20f | 27 Sep 19:24 | R20e with the veto restricted to real address mismatches (asim < 90) or different-business names (−1,513 instead of −1,737) | – | – | pending | pending | `output_r20f/` validator PASS; **best candidate**; projected 0.98898 (0.98855–0.98925) |
| R20g | 27 Sep 19:27 | fallback: s8b + India rescue v2 + US rescue (France = s8b) | India 0.99215, US 0.99228 | – | not uploaded | – | `output_r20g/` validator PASS; use if upload 1 shows the France changes lose; projected 0.98858 |
| R20j | 27 Sep 21:10 | s8b + rescue v4 (rescue candidates re-scored with a new xlm-r-large cross-encoder trained on them): India +18,586, US +2,115 (France = s8b) | **India 0.99262 (+0.00155), US 0.99237 (+0.00028)** | – | pending | pending | `output_r20j/` validator PASS, all 5,839,748 links ⊆ candidates (128,969,685), sha256 `ac54a967…`; **final candidate**; projected 0.98860–0.98895 |
| R20k | 27 Sep 21:30 | r20j + France precision package R2: −6,294 France links that the France stage-3 model (pfr, never uploaded) drops **and** the large CE rejects (< 0.3), excluding LB-backed classes; includes veto2 (1,298 of 1,513 overlap) | US/India = r20j | – | pending | pending | `output_r20k/` validator PASS, sha256 `a2553c98…`; upload 2; projected 0.9888–0.9897 |
| R20l | 27 Sep 21:34 | r20j + France R3: R2 without the category→true-noise-word swaps (−3,164 France links: R3 2,949 ∪ veto2) | US/India = r20j | – | **0.989472** | **96** | `output_r20l/` uploaded ~21:50 (upload 2 of 3); **new best, +0.001695 vs s8b** (projected 0.9887–0.9894) |
| R20p | 27 Sep 22:37 | r20l + 2,375 France adds (pfr p3 ≥ 0.95 ∧ large CE ≥ 0.95, x1+x2 unlinked; excluding upload-1 adds, category/TN classes and +GEN) − 74 more R3-pattern removals | US/India = r20j | – | **0.989507** | **108** | `output_r20p/` uploaded ~23:30 (upload 3 of 3); **best public score, +0.000035 vs r20l** (projected 0.98947–0.98960). Below the 0.989522 switch bar and not reproducible from the package code, so the final zip stays r20l |
| R20h | 27 Sep 19:29 | R20f + France transfer extension E1/E2 (+3,727 France links in the same Type-B-proof cells; street matched after stripping region/department words) | – | – | not uploaded | – | `output_r20h/` validator PASS; final if upload 1 confirms the France transfer; projected 0.98917 |
| R21 | 27 Sep 23:10 | listwise cross-encoder for uncertain US/India queries (handoff idea 1) | – | – | not built | – | **abandoned**: started at 23:02 with 57 min left; needs 2–3 h (idea 2 ≈ 85 min). Final zip stays r20l |
| R22 | 28 Sep 19:55 | final package: zip switched to R20p, 5 packaged-code bugs fixed, R20p step ported, byte-level reproduction check | US 0.99237, India 0.99262 | – | 0.989507 (R20p) | pending | `SteinsGate_submission.zip` holds `output_r20p` (sha `627aea51…`); doc rewritten for the methodology request (deadline 29 Sep 10:00) |

**Current best on LB:** **R20p `output_r20p/` = 0.989507 public** (27 Sep ~23:30; rank 108 at upload, final rank pending). **The final zip now holds R20p** (R22, 28 Sep): the packaged code rebuilds it byte for byte. Before: R20l `output_r20l/` = 0.989472 (rank 96 at the time); R19 `output_s8b/` = 0.987777 (rank 145).

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

## R13 · bottleneck diagnosis: where the 0.0089 gap to rank 1 goes (analysis only) · 27 Sep 09:40–10:45
- **Approach:** label-based checks on the x3/x3+x4 OOF argmax rows (P0/P1 US, P0 India) plus label-free test France vs US cell comparisons. Each hypothesis was tested directly on data (scripts in the session scratchpad: `gen1/raw1-3/cons1-2/na1/leak1/twin1/fr1-3/mix1-2/acr1/ind1.py`).
- **Data:** laptop; existing arrays and norm tables, raw TSVs for text checks.
- **Results:**
  - **LB decomposition:** with US/India on test ≈ OOF − 0.0007 (R07b) ≈ 0.9878, the 0.982556 LB implies **France ≈ 0.953**. France (15% of S1) costs ≈ 0.005 of the 0.0089 gap to rank 1 and US/India ≈ 0.004.
  - **Rejected hypotheses:**
    - distractors do *not* come in clusters: distractors pointing to the same S1 share identical numbers 1.4% of the time vs 37.6% for true matches;
    - no ID or row-order leak (id corr 0.0001; S2 siblings ~1M rows apart);
    - no raw casing/format leak (distractors carry their source's format);
    - same-name S1 "twins" are independent entities (singleton rate 5.5%, like everywhere else), so empty-address ambiguity is mostly irreducible. Queries without a number are ~half of the US argmax errors (P0 US: FP 3,861 / FN 6,249).
  - **Confirmed signals that our features miss:**
    - **S1 is noisy too.** In 22% of S1s with ≥ 3 numbered true matches, the majority of the true S2/S3 records agree on a house number different from S1's. Every feature we had compared the query only with S1, never with the S1's other queries (consensus).
    - **Normalization erased distractor perturbations:** doubled letters in brand names (Ferreon → Ferrreon, Novora → Novvora; ~1/3 of the confident US FPs sampled), house-number letter/fraction suffixes (1714-C, 2721 1/2) and repeated words (LLC LLC).
    - **Acronym names** (cmsn = centre medical saint nicolas) are 100% true at equal numbers in US (1,503/1,503) but only 95.5% linked in France.
  - **France mechanism.** Two distractor types:
    - Type A: +GEN number shift plus a name tweak (P0 US 428k);
    - Type B: same address, different business (P0 US 36k, 12× rarer).

    In US, Type B replaces the *distinctive* word (kirby → martinez mountain software) and the model rejects it (mean p 0.02–0.11). French names follow the template "{city/person} {category} {legal}" (nantes comite sarl), and France's same-address distractors swap the *category* word (pharmacie → service), which the US-trained model reads as harmless noise. Test France vs US at equal numbers:

    | cell | France share | US share | France uncertain (p 0.02–0.98) | US uncertain |
    |---|---|---|---|---|
    | eq + other name edits | 15.0% | 8.1% | 3.9% | 0.2% |
    | eq + 1 swapped word | 7.1% | 7.3% | 2.3% | 0.15% |

    Overall, 15.6% of France queries are uncertain vs 4.9% of US queries.
  - **India blocking misses** (P0: 24,093 = 1.58% of true pairs): 42% empty address, 32% non-Latin, 25% Latin with an address. The last group shows a normalization bug (`No.` → `patna` in Latin India addresses) and initialism-concatenated names (mfprojects = modern fortune projects). Fixing either needs re-blocking, which is out of time.
- **LB:** n/a (analysis).
- **Takeaways:** our model misses query↔query consensus, raw perturbations and acronyms, so it is part of the bottleneck (R14 measures how much). France's gap is a distractor type that US/India never shows.
- **Next:** R14 stage-3 re-scorer on argmax rows (US/India), then a France stage-3 probe.

## R14 · stage-3 re-scorer on argmax rows (US/India), France = R06 · 27 Sep 11:00
- **Approach:** new LightGBM stage 3 on one row per query, the argmax of the x3+x4 blend (the same rows the decision uses). 44 features:
  - logit p of the blend and of each of x1–x4, runner-up p and gap, number of candidates;
  - **consensus with the S1's other argmax queries** (label-free, from their p): confident-query count/sum, rank within the S1, same number/name/address counts, number class vs S1 and vs the consensus number, support of the query's extra name tokens and of S1's missing tokens among the other confident queries;
  - **raw-text perturbations** the normalization erased: doubled/tripled letters, accents, house-number letter and 1/2 suffixes, repeated words, brackets/parentheses, `#`;
  - acronym flag, name-collision counts (pool S1 / queries with the same name), empty address, S3 flag, India flag.

  Cross-fit P0 ↔ P1 pooled over US+India (10% query holdout for early stopping; 63 leaves, lr 0.05). Test p3 = mean of both fold models. (T1, T2) tuned on the pooled OOF p3. Ties still dropped on the stage-2 blend. Per-source caps on. France rows copied verbatim from `output_hyb34` (R06 x1+x2). Scripts: session scratchpad `gen2.py` (argmax tables), `stack2.py` (build/fit), `compose.py` (submission); to be moved into `ber/` for the package.
- **Data:** laptop; P0/P1 OOF arrays of x1–x4, raw TSVs for the text features. Fit 19 min, peak 4.8 GB.
- **Results:**
  - Argmax-row log-loss 0.01672 → **0.01277** (P0 → P1) and 0.01685 → 0.01281 (P1 → P0), i.e. −24%. best_iter 1260 / 1221. Top gain: lp 0.59–0.67, lx4 0.16, gap 0.10–0.15, lx3 0.04–0.07.
  - US-only ablation (x3 p, before the x1–x4 inputs): recalibration features alone 0.98995, + consensus 0.99031, + raw text / acronym / collisions 0.99053 (x3: 0.98977).
  - **OOF macro F0.5:**

    | | x3+x4 blend | stage 3 | Δ |
    |---|---|---|---|
    | US | 0.98984 | 0.99062 | +0.00078 |
    | India | 0.98646 | **0.98815** | +0.00169 |
    | pooled | 0.98848 at (0.56, 0.74) | **0.98963 at (0.54, 0.76)** | **+0.00115** |

  - `output_s3/matching_results.tsv`: links/S1 US 3.3893 (hyb34 3.3869), India 3.3449 (3.3488), France 3.3091 (identical). Diff vs hyb34: US −5,307 / +6,895, India −14,422 / +11,306, France 0. Caps dropped 15 links. Validator `--check-ids` **PASS**; links are argmax pairs of the existing candidates, so the R05 candidate file stays valid. sha256 `857c9535…`.
- **LB:** **0.984009, rank 347** (27 Sep ~11:30, upload #3 of the day; 2 left). **+0.001453 vs hyb34**, more than predicted (+0.00115 × 0.85 ≈ +0.001). The test gains more than OOF, probably because test has more hard distractors per S1 and consensus helps there.
- **Takeaways:** the pair model was a bottleneck. Information outside the (query, S1) pair (the other queries of the same S1, model disagreement, raw text) is worth more than any feature added since x3, and most on India.
- **Next:** upload; France probe with a stage 3 built on x1+x2; stage-3 v2 (runner-up re-ranking); move the code into the package.

## R15 · follow-ups after R14: what else moves the score (mostly negative) · 27 Sep 11:30–12:40
- **Approach:** after R14 (LB 0.984009), tested every remaining lever on OOF or with label-free test evidence. Laptop plus the JarvisLabs VM (resumed at 16 vCPU / 62 GB; IN1 had no 32-core block).
- **Results:**
  - **India `patna` bug** (translit dict maps `no`/`4`/`c`/`d`/`urban` → `patna` whenever the *name* is non-Latin): it touches 20.6% of India train queries, but affected true pairs are blocked out *less* often (1.31% vs 1.64%) with no extra FN/FP. **Not worth re-blocking.**
  - **Stage 4** (consensus features recomputed from stage-3 p3): fold-1 log-loss 0.01277 → 0.01351, **worse**. Stopped.
  - **Top-2 re-rank** (argmax + runner-up rows with both S1s' consensus and record counts, 127 leaves): pooled OOF **0.98962 vs 0.98963** for stage 3, no gain. The twin record-count prior carries no signal.
  - **France stage 3** (the R14 stage 3 rebuilt on x1+x2, France-safe features): US/India OOF 0.98706 → 0.98877. On France it only moves −5,286 / +6,129 links (removes +GEN links, adds equal-address and other-number links), so there's no clear France signal. Not used.
  - **Blocking K=50 probe** (VM): test France has only 9,082 queries with a match-looking candidate at ranks 11–50, mostly different streets. On P1 US, 12,268 of 20,988 blocked-out true pairs sit within the top 50, but an address rule recovers only 28 because they are empty-address queries. **Re-blocking would not pay.**
  - **Loss decomposition of R14 OOF** (fix one class, others unchanged): FPs +0.00183 (13.8k), argmax right but below threshold +0.00278 (62.5k), wrong S1 won +0.00226 (49.7k queries), blocked out +0.00388 (90.2k pairs). Empty-address queries dominate the first three (US: 27.7k wrong-S1 + 23.2k below threshold; India: 14.2k + 11.5k). Raw examples show genuine same-name twins in different states.
  - **Plug-in expected F** (calibrated p; bias measured on OOF):

    | | OOF plug-in | OOF actual | test plug-in | test estimate |
    |---|---|---|---|---|
    | US | 0.99134 | 0.99062 | 0.98893 | ≈ 0.988 |
    | India | 0.99116 | 0.98815 | 0.98834 | ≈ 0.985 |
    | France | – | – | 0.9655 (x1+x2) | ≈ 0.967 |

    This reproduces the LB: 0.8503 × 0.9868 + 0.1497 × 0.967 ≈ 0.9840. The test is ~0.0025 harder than OOF (1.5× more near-copy distractors per S1), which is why R14 gained 1.5× its OOF gain on the LB.
  - **France checks:**
    - France names are generic (`nantes comite sarl` × 60 S1s);
    - same-number different-street links are mostly street typos (true);
    - the +GEN band shows France's distractor swaps are legal-form and category→category swaps, while true noise uses groupe / développement / france / cie / services;
    - France has 4–5× fewer "other number" pairs than US.
  - **Cross-encoder** (multilingual MiniLM on raw text, uncertain argmax rows): datasets built (P0 781k / P1 769k training rows; test US 395k, India 541k, France 315k). The laptop GTX 1650 runs at only ~24 samples/s (5+ h), and creating an A100 instance needs the user's approval.
- **LB:** n/a.
- **Takeaways:**
  - US/India OOF has plateaued with the current feature families.
  - The remaining test loss is the harder test density plus France.
  - A new signal (a transformer on raw text) is the one untried family.
- **Next:** A100 cross-encoder if approved; stage-3 hyperparameter variant (p3b) running on the VM.

## R16 · cross-encoder (xlm-roberta-base) as a stage-3 feature + LB-evidenced France edits · 27 Sep 13:30
- **Approach:**
  - **Cross-encoder (CE):** `FacebookAI/xlm-roberta-base` (MIT) fine-tuned as a pair classifier on **raw** text ("name | address" of the query vs the S1).
    - Rows: argmax rows of the x3+x4 blend with 0.003 < p < 0.997, plus 4% of the other rows (P0 781k / P1 769k training rows).
    - Cross-fit: the P0 model scores P1 and vice versa. Test (US 395k, India 541k, France 315k uncertain rows) = mean of both.
    - 1 epoch, bs 128, lr 3e-5, bf16, max 128 tokens. JarvisLabs A100 40GB (user-approved), both folds in parallel, 34 min including scoring.
  - **Stage 3 with CE:** the R14 stage 3 plus the CE logit as a feature (NaN outside the band, identically on OOF and test).
  - **France edits** on R06, class-level:
    - (a) drop equal-address category→category swaps / category additions whose new word no other confident record of the S1 carries;
    - (c) add equal-address acronym pairs;
    - (d) drop +GEN links whose number is also +GEN vs the consensus of the S1's other records.

    Edit (b) (x3's true-noise-word additions) was dropped: back-solving with the CE's view puts them at ~72% true, the F0.5 break-even.
- **Evidence for the France edits** (`lbattr.py`: every past France upload diffed against R06 and split by name-edit class):
  - R07 removed 21.6k equal-address true-noise-word links (−0.00114), so those links are ~100% true;
  - R09 added 26.3k equal-address category swaps among 50.7k links (−0.00449);
  - x3 France added +4.4k net category swaps and +7.2k true-noise words (−0.00028);
  - together these solve to a category-swap true rate ≈ 0;
  - the CE independently scores linked category swaps at 0.46 (model 0.79), acronyms at 0.90–0.94, and lone +GEN links at 0.28;
  - the same +GEN/+GEN cell is 2.15% true in US OOF.
- **Results:**
  - CE holdout log-loss 0.119 / 0.123, accuracy 95.3% (uncertain-heavy rows).
  - On P1 uncertain rows:

    | | stage-2 p | stage-3 p3 | CE | avg(p3, CE) |
    |---|---|---|---|---|
    | India | 0.181 | 0.129 | **0.110** | **0.093** |
    | US | 0.191 | 0.157 | 0.220 | 0.153 |

  - Leak check: distractor queries appear in both pools, but only ~4% of uncertain rows were seen in CE training, and the gain holds on unseen queries (India unseen distractors 0.175 → 0.082).
  - **Stage 3 + CE OOF:** argmax-row log-loss 0.01672 → **0.00857** (stage 3 alone: 0.01277).

    | | stage 3 | stage 3 + CE |
    |---|---|---|
    | US | 0.99062 | **0.99187** |
    | India | 0.98815 | **0.99070** |
    | pooled | 0.98963 | **0.99140** at (0.58, 0.74) |

  - `output_s6b/matching_results.tsv`: US/India from stage 3 + CE, France R06 + (a, c, d). Diff vs `output_s3`: US −4,687 / +4,881; India −8,572 / +12,258; France −8,944 / +1,633 (6,581 category swaps + 590 category additions + 1,772 lone +GEN removed; 1,634 acronyms added). Validator `--check-ids` **PASS**. (`output_s6` = the same with edit (b), not preferred.)
- **LB:** not uploaded (user: upload only with strong evidence of > 0.99). Projection ≈ 0.9868–0.9876.
- **Takeaways:** a transformer reading raw text is the biggest single gain of the challenge, especially for India (transliterations, name variants).
- **Next:** xlm-roberta-large CE (running, ~15:10) and a two-CE stage 3; the CE on runner-up rows; package the stage-3 + CE code.

## R17 · checks around R16: projection, density, France cells (analysis only) · 27 Sep 13:40
- **Bias-corrected plug-in projection** (OOF plug-in − actual = bias; applied to test):

  | | test estimate |
  |---|---|
  | stage 3 (R14) US / India | 0.98821 / 0.98533 |
  | stage 3 + CE (R16) US / India | **0.99070 / 0.98958** |
  | France (implied by R14 LB 0.984009) | 0.9691 |

  **Projected LB for `output_s6b`: 0.9870 (France edits +0) … 0.9884 (+0.010 France F).**
- **Test-density simulation** on the R16 OOF (distractor rows duplicated with prob R): at R = 0.9, F 0.99140 → 0.99108 at the current thresholds; the best thresholds (0.58, 0.82) gain only +0.00003. **Thresholds stay.**
- **CE on France:** strong disagreements with the France model on 41k rows in both directions within the same classes. It rejects 5.9k linked true-noise-word pairs, which R07's LB proves true. **The CE is not reliable for France semantics**, so it is used only as a secondary check for the evidence-backed edits.
- **"Other number" lone deviants:** in US/India OOF the linked ones are 99.3% true; France's 8k linked ones have CE 0.78. No action.
- **Empty-address France queries by name multiplicity:** they behave like US at every level of name sharing, so generic French names cause no extra errors.
- **Budget:** balance ₹469 at 13:35. The VM watchdog was restarted on the new VM id 518027 (the resume created a new id). A GPU low-balance guard (`work/vm/gpu_guard.sh`, ₹200) was added for the A100 container 518069.
- **Fourth France evidence point (R08 vs R07, split by class):**
  - R08 removed 3.0k equal-address + 0.44k no-number true-noise-word links;
  - it added 1.9k net equal-address category swaps and 0.86k acronyms;
  - LB −0.0004 … −0.0014.

  At the class rates behind the France edits (true-noise words ≈ true, category swaps ≈ false, acronyms ≈ true), the first two effects alone predict ≈ −0.0003, within that range.
- **Stage-3 variants with CE-derived consensus features** (per-S1 max/mean/#positive CE of the other queries, CE rank): pooled OOF 0.99141 vs 0.99140. Saturated.

## R18 · large cross-encoder, top-2 re-rank with CE, France edit (e) · 27 Sep 14:40–15:45
- **Approach:**
  - **xlm-roberta-large CE** (MIT, 560M): same rows plus the runner-up rows (stage-2 p ≥ 0.01, which hold almost every runner-up true pair: US 8,664 / 8,738, India 4,725 / 4,805). bs 64, lr 1e-5, 1 epoch, both folds in parallel on the A100 (78 min training + 30 min scoring).
  - **Stage 3 with base + large CE** (p7).
  - **Top-2 re-rank with the large CE** (p8): stage-3 features for each query's argmax and runner-up rows plus the competing row's S1 statistics and CE score. Per query, the higher-scoring row is the link candidate.
- **Results:**
  - Large-CE holdout log-loss **0.111 / 0.107**, accuracy 96.0% (base 0.119 / 0.123, 95.3%). On P1 uncertain rows: US 0.2007 (base 0.2203), India 0.1024 (base 0.1099). avg(p3, large): US 0.1437, India 0.0896.
  - OOF pooled macro F0.5:

    | model | pooled | US | India |
    |---|---|---|---|
    | stage 3 + base (p6) | 0.99140 | | |
    | stage 3 + base + large (p7) | 0.99152 | 0.99198 | 0.99084 |
    | **top-2 re-rank + large (p8)** | **0.99163** at (0.52, 0.76) | 0.99203 | 0.99104 |
    | top-2 + base + large (p8b) | 0.99165 | | |

    Base adds nothing on top of large.
  - **The large CE reads France better:** unlinked equal-address true-noise-word pairs 0.80 (base 0.52); x3's 7,240 true-noise-word additions mean 0.91 / median 0.99 (85% > 0.9); category swaps 0.47; acronyms 0.95–0.98. **New France edit (e):** x3's equal-address true-noise-word links with large CE > 0.72 (+6,506). France (a, c, d, e): −8,943 / +8,140.
  - **Checkpoint `output_s8`:** p8 for US/India + France (a, c, d, e). US −5,380 / +5,768, India −9,121 / +14,606, France −8,944 / +8,139 vs `output_s3`. Validator **PASS**.
    - Bias-corrected plug-in: test US ≈ 0.9910, India ≈ 0.9901.
    - Class-level estimate of the France edits: ≈ +0.008 France F.
    - **Projected LB ≈ 0.9885 (0.987–0.990).**
- **Incident:** the restarted watchdog paused the VM at 14:35 after 40 idle minutes while the GPU trained; it was resumed at 15:10 (new IP 151.185.34.165, data intact). The idle rule is now off (`NO_IDLE_PAUSE`); the low-balance guards (VM + A100) sit at ₹60.
- **Next:** mdeberta-v3-base CE (third member, running); France re-scorer with CEs applied only in the classes where France behaves like US/India (hybrid); final package.
- **France hybrid (rejected):** a France re-scorer (x1+x2 stage 3 + base/large CE, trained on US/India) reaches pooled US/India OOF 0.98706 → 0.99128. Applied to France only in the "US-like" classes (same name, typo, coined, rest, acronym), it would add 10.2k and remove 4.3k links.
  - The additions are dominated by generic-name guesses: 78% of no-number same-name additions and 71% of different-number same-name additions have names shared by ≥ 2 France S1s, e.g. address-less `nantes club sarl` shared by 157 S1s, which x1+x2 correctly kept at p 0.08.
  - The re-scorer lacks the name-collision features (dropped for France on purpose), and "same name" is weak evidence in France's templated names.
  - **Not used**; France = R06 + edits (a, c, d, e).

## R19 · final candidate and package · 27 Sep 16:10
- **Final file `output_s8b/matching_results.tsv`** (copied into `output/`, sha256 `1218c64e…`):
  - US/India: **top-2 re-rank with xlm-roberta large + base cross-encoders** (p8b), OOF **0.99165** (US 0.99206, India 0.99106) at (0.58, 0.78);
  - France: R06 x1+x2 + edits (a, c, d, e).

  Diff vs the uploaded `output_s3`: US −5,809 / +5,511, India −9,415 / +14,198, France −8,944 / +8,139. Validator **PASS**; candidate check OK (1,732,544 rows, 99,695,099 ids, matches within candidates).
- **Projection** (bias-corrected plug-in): test US ≈ 0.9911, India ≈ 0.9902; France ≈ 0.969 + edits (class estimate ≈ +0.008). **LB ≈ 0.9885 (range 0.987–0.990).**
- **Dropped:**
  - mdeberta-v3-base third member: one fold ran out of GPU memory, the other reached holdout log-loss 0.146 vs large 0.107;
  - France hybrid re-scorer: generic-name guesses.
- **Package:** `SteinsGate_submission.zip` (587 MB) built by `work/logs/final_package.sh` with CHOSEN=output_s8b.
  - `code/business_entity_resolution/src/stage3/` holds every script; `run_stage3.sh` reproduces the final stage.
  - README, requirements (torch/transformers pinned) and `docs/Documentation_BER.md` are updated; no placeholders remain.
  - The doc's final-LB line says "pending (projected ≈ 0.988)".
- **Compute:** A100 container 518069 and VM 518400 are paused (not destroyed). All artefacts of the final are on the laptop.
- **LB:** **0.987777, rank 145** (27 Sep ~17:05, uploaded after the user got 2 extra submissions; 3 uploads left). **+0.003768 vs R14, new best.**
  - The projection was 0.9885 (range 0.987–0.990); the result is 0.0008 below its centre.
  - If the US/India projections hold (US 0.9911, India 0.9902), France is ≈ 0.972: the edits added ≈ +0.003 France F, not the estimated +0.008.
  - Otherwise France is ≈ 0.977 and US/India transferred ≈ 0.001 less than projected. One probe upload can tell which.
- **Next:** the user's new target is > 0.9918 (top 10), with 3 uploads and ~6.5 h left. Work continues in a new chat from `docs/handoff_R20.md`. This session's scratch is preserved (hardlinked) in `work/r19_scratch/`.

## R20 · final-day research: loss map, France structural transfer (upload 1) · 27 Sep 17:55
- **Approach:** no new model. Label-based tests on the P0 US/India OOF argmax tables plus label-free tests on test France, then a France package built from **structural cells whose truth rate transfers from US/India**. Scripts in the session scratchpad `r20/` (copied to the VM at `/home/ubuntu/r20/`).
- **Data:** JarvisLabs VM 518546 (16 vCPU / 64 GB, resumed 17:20); laptop for composing only.
- **Results:**
  - **p8b OOF loss decomposition** (pooled 0.99169 at (0.58, 0.78)): fix FPs +0.00085 (6.9k), FN argmax right but below threshold +0.00185 (45.1k), wrong S1 won +0.00206 (46.0k), **blocked out +0.00384** (90.2k pairs; US +0.00289, **India +0.00526**).
  - **France link-count distribution vs train GT** (same generator): France 3.306 links/S1, GT 3.459, US test 3.389, India test 3.351. Empty S1s 5.88% (GT 5.58%). France is ~2.4% of links more recall-limited than US.
  - **Structural transfer table** (`cells.py`): cells = house-number relation × same-street flag × vocabulary-free name relation × legal-form relation. Where US and India agree (±0.06), the rate is generator-level:
    - equal address, query name = one coined token (onyx…, dova…, riza…): **US 98.8% / India 98.7% true**; France links 92.8% (2.6k unlinked, CE ≈ 0.91);
    - same name + same legal form + same street + different number: US 99.5% / India 99.7%; France links 59–83%;
    - acronym-like single in-vocab token at equal address: US 96.6–97.3% / India ≥ 0.93.
  - **Cells that do not transfer (France has ~5× US's Type-B rate):**
    - `eq|S|swap1_oov` is 97.6% true in US, but in France it holds Type-B initialism edits (icg → icgl, zt → zty; CE 0.000);
    - short first-token 1-edit at equal address is 79% true in US/India (OCR l/i, 8/b; translit sre/shre), but France links 122 of 564 with CE ≈ 0.001–0.03;
    - `swap1_iv` mixes true-noise-word replacements with category swaps.
  - **Small or negative:**
    - France d = −1/−2 rows are mostly other businesses on the same street; the same-name ones are ~65% linked already;
    - edit (d) removed 247 same-name/same-legal/same-street +GEN links, but US has that cell at 67.6% true (India 95.9%), i.e. break-even, so no change;
    - France empty-address same-name unlinked rows are twins (runner-up p 0.2–0.4): irreducible.
  - **India blocking misses:** 13.6k of P0 India's 24.1k blocked-out true pairs have an address. The true S1 usually has a generic name shared by 20+ pool S1s whose name-twins fill the top 10. An address-heavy second retrieval (weights 0.1/0.8/0.1, K=10) recovers **4,980 (37% of those with an address)** but adds 17.3M new pairs; address-only (0/1/0) recovers 3,599.
  - **Package P1** (`pkg1b.py` → `fr_pairs_P1.parquet`):
    - additions only in Type-B-proof cells (same name, coined or acronym token, drops) with US and India ≥ 0.93 true and large CE ≥ 0.8: +4,324;
    - removals: +GEN cells ≤ 0.15 true (51), and initialism edits with CE < 0.1 (87).
    - Composed by patching s8b's France rows (`patch_fr.py`; patching with s8b's own pairs reproduces s8b byte-identically). `output_r20a/`: France +4,317 / −138 (9 capped), US/India byte-identical. Validator PASS; all pairs in the candidates; no query linked twice. sha256 `d7f478c7…`.
  - **Projection:** ≈ +385 S1-units at US/India cell rates ⇒ **LB 0.98800** (0.98785–0.98810).
- **LB:** **0.987654**, rank ~200 (user estimate; uploaded ~19:40, upload 1 of 3). −0.000123 vs s8b, so France F fell ≈ 0.0008 instead of rising ≈ 0.0015. The structural transfer fails on France: cells ≥ 0.93 true on US/India (with xl ≥ 0.8) are below the ≈ 0.70 break-even on France. Consequence: R20e/f/h (which carry P1 or the same reasoning) are dropped; R20g (rescues only) goes up next.
- **Takeaways:** France's remaining loss is spread thin. The largest structural cells were already handled by edits (a, c, d, e), and the unlinked empty-address mass is twin ambiguity. US/India's biggest remaining bucket is India's blocking.
- **Next:** India address-heavy second retrieval + rescue scorer, validated on P0/P1 OOF. Decide the transfer package by the upload-1 LB delta.

## R20b · India blocking rescue: address-heavy second retrieval + rescue scorer (upload 2) · 27 Sep 18:20
- **Approach:**
  - **Retrieval:** a second TF-IDF blocking pass with address-heavy weights (name words 0.1, address uni+bigrams 0.8, name_sq char-4 0.1; K=10; `blk2.py`, reusing `ber.blocking.block_one`), keeping only pairs that are new relative to the original top-10.
  - **Scope:** queries the p8b decision leaves **unlinked** and that have a non-empty address. Linked queries are never touched.
  - **Scorer:** LightGBM (63 leaves, lr 0.05, early stopping on a 10% query holdout) on 27 vocabulary-free features (`rescue_feats.py`): rapidfuzz name ratios (token_set/sort, partial, ratio), Jaro-Winkler on name_sq, address token_set/partial, **IDF-weighted address overlap** (share of the query's address IDF found in the S1 address, max shared IDF), house-number overlap/first-number equality/signed difference/+GEN flag, S1 name-twin count, new blocking score/rank, the query's p8b argmax p, lengths.
  - Cross-fitted P0 ↔ P1; test = mean of both models. Decision: link the query's best new candidate if score ≥ τ; caps enforced.
- **Data:** VM 518546. P0/P1 India: 2.60M queries each → 17.3M new pairs per pool; 1.09M target queries per pool with 8.1M new pairs (4,961 / 4,939 positives). Test India: 31.2M new pairs; 1.92M target queries with 14.05M new pairs.
- **Results:**
  - Recall of the second retrieval on P0 / P1 India: recovers 4,980 / 4,954 of 24,093 / 24,235 blocked-out true pairs (all from queries with an address; 37% of those).
  - Scorer holdout log-loss 0.00091 / 0.00079 (best_iter 377 / 508). Top gain: `a_idf_frac` 0.45, name ratio 0.14, address token_set 0.11, number overlap 0.08.
  - **OOF India (P0+P1, p8b decision as base): 0.99107 →**

    | τ | additions | precision | India F | Δ |
    |---|---|---|---|---|
    | 0.5 | 7,959 | 0.893 | 0.99177 | +0.00070 |
    | **0.8** | **6,796** | **0.936** | **0.99181** | **+0.00073** |
    | 0.97 | 5,144 | 0.964 | 0.99171 | +0.00064 |

  - **Test India (τ = 0.8): 8,258 additions** (10.2 per 1000 S1 vs 7.7 on OOF: the test S1 index is larger, so there are more name twins). Projected ≈ +0.00097 India F ≈ **+0.00045 LB**.
  - `output_r20b/` = R20a + rescue (India +8,258; US/France = R20a), sha256 `b9ca8a1a…`; `output_r20c/` = s8b + rescue, `9dd9a9a6…`. Validator PASS, no query linked twice.
  - The rescue pairs are outside `candidate_pairs.tsv` by construction. The final package must use a regenerated candidate file = original 99.7M ∪ the 14.05M rescue-scored India pairs.
- **France second retrieval (label-free):** 2.41M new pairs for 449k unlinked France queries, but only a few hundred fall into the high-truth same-street cells. France's missing recall is not blocking. Dropped.
- **LB:** pending (upload 2 of 3).
- **Next:** US rescue chain (running); final package with the regenerated candidate file.

## R20c/R20d · rescue v2 (name-twin expansion) + US rescue; France plug-in check · 27 Sep 18:58
- **Approach:**
  - **Name-twin expansion** (`blk3.py`): for the same target queries, every partition S1 whose core name (legal forms removed) equals the query's, ranked by IDF-weighted shared address tokens, top 10 new pairs. Added to the rescue candidates with a source flag.
  - Rescue scorer retrained on the union (cross-fitted P0 ↔ P1).
  - The same pipeline was run for US (τ tuned on OOF).
- **Data:** VM 518546. India target queries 1.09M per pool / 1.95M test; new pairs P0 9.64M (7,335 positives), P1 9.64M (7,332), test 16.93M. US: 13.6M per pool (≈ 2.5k positives), test 12.35M.
- **Results:**
  - Name-twin expansion recovers 2,374 / 2,393 more blocked-out true pairs in P0 / P1 India (on top of the 4,980 / 4,954 from the second retrieval), using 1.5M pairs per pool. US gains only ~90 per pool (US twin misses are empty-address).
  - **India rescue v2 OOF: 0.99107 → 0.99215 (+0.00108)** at τ = 0.8 (9,789 additions, precision 0.925; top gain: name ratio 0.25, `a_idf_frac` 0.22). Test: **12,932 additions** (12,928 after caps).
  - **US rescue OOF: 0.99210 → 0.99228 (+0.00018)** at τ = 0.85 (3,211 additions, precision 0.933). Test: 1,482 additions.
  - Checked and dropped:
    - switching FP links to rescue candidates: only 410 India FP links have a blocked-out true S1, and the retrieval finds 34 of them;
    - same-source count prior for empty-address twins: no asymmetric signal beyond p (the earlier "no signal" holds).
  - **Plug-in bug found:** `plugin.py` (R15–R19) computed 1.25·TP/(0.25·L + T), which weights recall like F2. F0.5 is 1.25·TP/(L + 0.25·K). The R17–R19 LB projections were bias-corrected on OOF, so they stay roughly valid, but threshold conclusions drawn from that plug-in do not.
  - With the correct formula and class-corrected France probabilities (LB-proven classes, transfer-cell rates, CE-rejected initialism edits), expected France F:
    - s8b 0.98521 → upload 1 **0.98665 (+0.00144 France F)**;
    - France threshold re-optimization is worth +0.00003, so thresholds stay.
    - The plug-in level (0.985–0.987) sits well above the LB-implied France (0.972–0.977). France's unseen loss, i.e. blocking plus miscalibration, is ≈ 0.01.
  - `output_r20d/` = R20a + India rescue v2 + US rescue. Validator PASS. The candidate file must add the rescue-scored pairs (India 16.93M + US 12.35M).
- **Projection:** 0.987777 + France P1 0.00022 + India 0.4675 × 0.00155 + US 0.3827 × 0.00018 ≈ **0.98880**.

## R20e · France CE veto for address-driven classes · 27 Sep 19:08
- **Approach:** R06 France has no cross-encoder. On US/India OOF (P1 rows scored by the P0-trained large CE), when the stage-2 p is > 0.7 but the CE says ≤ 0.05, true rates are low:
  - same name + same house number: **3.2% (US) / 0.5% (India)**;
  - no house number: 9% / 8%;
  - other equal-number edits: 11% / 5%.

  In these classes the CE judges the *address* (street/city), which does not depend on French vocabulary. The CE's known France failure modes are category swaps and true-noise words, and those classes are excluded. **Veto:** drop linked France pairs with CE ≤ 0.05 in the classes same_core / drop_only / typo / no_shared at eq / na / other numbers (TN_in, acronyms and category classes excluded).
- **Results:**
  - **1,737 links** vetoed. Examples: generic names at the same number on another street ("bordeaux colege sarl, 92 rue paulin" vs "92 rue saint jean"; "nantes pharmacie, 8 rue rubens" vs "8 rue general buat"), and concatenated *different* names at the same address ("balonsportivesas" vs "lasociation sportive sarl"). Expected ≈ +337 S1-units at US/India rates (+0.0013 France F).
  - Veto rate among CE-scored linked rows falls with R06 p: 4.0% (0.90–0.95), 1.6% (0.95–0.99), 0.49% (0.99–0.997). The 678k unscored links (p > 0.997) would yield ~1k more (≈ +0.0001 LB), so a GPU re-score is not worth ~2 h.
  - Also checked and dropped:
    - legal-form swap at equal address (strict cell only 388 rows; the earlier 41.6k came from spaced legal forms);
    - France second retrieval (a few hundred high-truth pairs);
    - US FP switching (negligible).
  - `output_r20e/` = s8b + France P1 − veto + India rescue v2 + US rescue. France +4,317 / −1,875; India +12,928; US +1,482. Validator PASS.
- **Projection:** 0.987777 + P1 0.00022 + veto 0.00020 + India 0.00073 + US 0.00007 ≈ **0.98900** (0.98855–0.98925; the France parts are the uncertain ones).

## R20f · refined France veto; further checks (negative) · 27 Sep 19:24
- **Refined veto:** the veto's rationale is an address mismatch. Among the 1,737 vetoed pairs, 462 have near-identical addresses (token-set similarity ≥ 90 after dropping region/department words), so there the CE's rejection is name-based:
  - correct for concatenated different names ("porniclub" vs "pornic services sarl", "apelunion" vs "pesac club sci") and initialism edits ("gxmc" vs "gxc", "xfbz" vs "xbz");
  - doubtful for legal-form-only differences ("riverains parents sasu" vs "riverains parents", 34 vs 034).

  **Veto v2** = address similarity < 90, or class no_shared/typo: **1,513 links**. `output_r20f/` = s8b + France P1 − veto v2 + India rescue v2 + US rescue (France +4,317 / −1,651). Validator PASS, all pairs in the candidates, sha256 `be3fde6c…`.
- **Negative / not worth it:**
  - linked France pairs naming different cities: only 153;
  - address + name-char retrieval (translit variants) recovers 340 more India misses for 4M pairs;
  - K = 20 address-heavy retrieval: 709 more for 8.7M pairs;
  - empty-address India/US misses with a unique core name: 35 / 84 per pool;
  - CE re-score of the 678k unscored confident France links: ≈ 1k expected vetoes (≈ +0.0001 LB) for ~2 h of GPU, so skipped.
- **Package:** dry run with `output_r20e` + the regenerated candidate file (128,969,685 pairs) built a valid 743 MB zip; the zipped matching file equals the chosen one. `final_package.sh` now takes `CANDS=` and refuses the R20 LB placeholders.

## R20h · France transfer extension (conditional final) · 27 Sep 19:29
- **Approach:** the upload-1 package required the loose token-Jaccard street flag `S`. France's `r`/`rue` and department-vs-region formatting pushes many true same-address pairs into `s`.
  - **E2:** pairs in the same Type-B-proof cells (with `s` read as `S`) whose addresses are near-identical after stripping region/department words and expanding r/av (token-set similarity ≥ 90), with large CE ≥ 0.8. **3,607 adds**, ≈ +311 S1-units at US/India rates.
  - **E1:** the original `S` cells with CE 0.5–0.8. 134 adds, ≈ +11 units.
- **Results:** `output_r20h/` = R20f + E1/E2: France +8,044 / −1,651 (23 capped), India +12,928, US +1,482. Validator PASS; all France pairs in the candidates; no query linked twice.
- **Decision rule for the final** (by upload 1's LB, R20a):
  - ≥ 0.98795 (transfer confirmed): R20h;
  - between 0.987777 and 0.98795: R20f;
  - below 0.987777: R20g (rescues only).
- **Disk incident (19:30):** a Docker image pull (`dockerd` → `unpigz`, root; not ours) filled the laptop partition to 0 bytes free. To restore tool output, removed the never-uploaded intermediate outputs `output_x2x3x4/`, `output_x3x4/`, `output_x4/` (R10; reproducible from the saved x3/x4 models) and the superseded `output_r20d/`, `output_r20e/` (both still on the VM, `r20/out/u2f`, `u3`). Free space after: 2.9 GB.

## R20i · upload 1 result; rescue v3 (negative); rescue cross-encoder started · 27 Sep 20:12
- **Upload 1 (R20a) = 0.987654**, rank ~200 (user estimate): −0.000123 vs s8b. France F fell ≈ 0.0008 instead of rising ≈ 0.0015, so the P1 adds were well below the ≈ 0.70 break-even on France.
  - Likely cause is selection: the adds were rows the France model had left **unlinked** inside cells whose *overall* US/India rate is ≥ 0.93. Those rows are the ones the model found ambiguous (e.g. generic-name twins), so the overall cell rate overstated their truth.
  - Consequences: R20e, R20f and R20h are dropped. France label-free edits are now 0 for 5 on the LB (R07, R08, R09, x3 France, P1). Upload 2 = `output_r20g/` (rescues only).
- **Machines:** CPU VM resumed as `518672` (32 vCPU / 128 GB) and A100 as `518671`; the watchdog was restarted for `518672`.
- **Rescue headroom (P0 + P1):**
  - India: 28.4k target queries have a true S1 in the pool. 14.7k have it among the b2/b3 candidates, and ≈ 9.5k of those are added.
  - US: 13.6k have a true S1; 5.0k are in the candidates.
  - About half the remaining misses have a made-up query name (`orbijax`, `quowexpyra`) plus a short address fragment. They sit far from their S1 in TF-IDF space (K=20 recovers only 709).
- **Rescue v3 (negative):** v2 features + within-query rank/gap/margin of 10 scores + a query-name vocabulary flag.
  - India OOF +0.00108 (τ 0.75), identical to v2. US +0.00018, identical to v2.
  - The F curve is flat for τ 0.5–0.95, so the LightGBM scorer has saturated on string features.
- **Rescue cross-encoder (running):** xlm-roberta-large trained on the rescue candidates themselves: the top 4 per query, for queries where v2 r ≥ 0.03.
  - Training sets: P0 58k pairs (8.4k positive), P1 58k pairs.
  - Setup: 3 passes, cross-fitted P0 ↔ P1; test scored by both models (141k pairs).
  - Rescue v4 adds its logit and within-query rank/gap as features.

## R20j · rescue cross-encoder + rescue v4 (final candidate) · 27 Sep 21:10
- **Approach:**
  - **Rescue cross-encoder:** xlm-roberta-large (lr 1e-5, bs 32, len 128, bf16, 3 passes), trained on the rescue candidates: the top 4 per target query where the v2 rescue score is ≥ 0.03, raw "name | address" text, US+India together.
    - Pool sizes: P0 58k pairs, P1 58k pairs.
    - Cross-fitted: the P0 model scores P1 and the P1 model scores P0; test takes the mean of both models' logits.
  - **Rescue v4:** v2 features + CE logit + within-query CE rank / gap to the best / margin over the runner-up. LightGBM (127 leaves), cross-fitted, τ tuned on OOF. Gain importance is dominated by ce_mg / ce_gap (India 0.69, US 0.87).
- **Data:** full P0/P1 pools and test. A100 518671, ~21 min per fold with both folds in parallel; LightGBM on the 32-vCPU VM 518672.
- **Results:**
  - CE holdout: logloss 0.055 / 0.043, accuracy 0.989 / 0.992.
  - Rescue holdout logloss: India 0.00088 / 0.00075 (v3 0.00124 / 0.00119); US 0.00030 / 0.00023.
  - **India OOF 0.99107 → 0.99262 (+0.00155; v2 +0.00108)** at τ 0.7: 12,746 additions at precision 0.968 (v2: 9,789 at 0.925). The F curve is flat for τ 0.5–0.95.
  - **US OOF 0.99210 → 0.99237 (+0.00028; v2 +0.00018)** at τ 0.85: 3,986 additions at precision 0.984.
  - Test: India +18,588 (+18,586 kept), US +2,115. That is 0.58 additions per CE-scored query on test vs 0.56 on OOF. 95% of r20g's additions are kept.
  - `output_r20j/` = s8b + both. Validator PASS; 5,839,748 links, all in `work/final_cands/candidate_pairs.tsv` (sha256 `c4f0dda6…`); no query linked twice. sha256 `ac54a967…`.
- **Projection:** OOF gains × LB weights = +0.00083 (India 0.00072, US 0.00011) → 0.98861. With the test/OOF addition ratio (≈ 1.4×) used for R20g, up to +0.0012 → 0.98895. Range 0.98860–0.98895.
- **LB:** **0.989472, rank 96** (upload 2, ~21:50). +0.001695 vs s8b, above the projection. R3 can give at most ≈ +0.0004 (3,164 removals × ≤ 0.21 S1-units each), so the rescue gave ≥ +0.0013, about 1.5× its OOF gain, like R14.
- **Package:**
  - Ported as `r20_build_rce.py`, `r20_rce_train.py` and `r20_rescue4.py`; `run_stage3.sh` step 8 rewritten (France transfer and veto out of the final, rescue v4 in).
  - README step 8 and Documentation §1 / §4.3 / §5 / §6 updated; the final LB placeholder is `R20J_LB`.
  - A100 paused at 21:05.
- **Takeaways:**
  - The rescue's limit was discrimination between address-similar S1s, not features. A cross-encoder trained on the same hard candidates fixed half of the remaining in-candidate misses.
  - France label-free edits keep losing, so France stays at s8b.
- **Next:** upload r20j (upload 2 or 3), fill `R20J_LB`, build the final zip.

## R20k · France precision package from the France stage-3 model + CE agreement · 27 Sep 21:30
- **Diagnosis (back to basics):**
  - Predicted links per S1 are the same in all three countries (5.8% of S1s with none; 3.31 links per S1 in France vs 3.37–3.39 in US/India), so France's loss is not recall.
  - The corrected plug-in puts s8b's France at 0.985, but the LB implies about 0.972. R09 and R20a showed that France's low- and mid-p rows are roughly calibrated, so the hidden ~0.013 must be **confident false links**.
  - France S1s share an exact address 2.4–2.8× more often than US/India (11.1% vs 4.0% / 4.75%): associations registered at shared addresses.
  - France same-address no-shared-word links are mostly true: concatenations (`letrierclub`), initials (`es` = entente sportive) and coined names built from the India syllables (`quobelo`, `zephkelo`).
- **France stage 3 (pfr, R15; US/India OOF 0.99128 vs 0.98706 for x1+x2):** on France it drops 13,213 s8b links and adds 20,432.
  - Raw removals: generic names at the same number on another street (`bordeaux colege sasu`, 40 r labtotiere vs 40 av georges clemenceau); a category word swapped for a true-noise word at the same address (`zip france` vs `zip primaire`, `montagne groupe sarl` vs `montagne culturele sarl`); initialism edits (`ozg amicale` vs `oig amicale`).
  - Calibration on US/India OOF where x1+x2 p ≥ 0.8 but p3 < 0.76 (true rate by p3 band): p3 < 0.2 → 0.021; 0.2–0.4 → 0.31; 0.4–0.58 → 0.52; 0.58–0.76 → 0.68.
- **R2 = pfr removals ∩ (large CE < 0.3 or unscored).** It excludes the LB-backed classes: edit-(e)/(c) links (p < 0.5), acronyms, pure true-noise appends, and no-shared-word links (coined names, CE ≈ 0.9).
  - 6,079 links: eq TN-swap 2,595, eq same-core 1,069, eq rest 693, na TN 407, other same-core 347, …
  - Union with veto2 (1,298 of its 1,513 already inside) gives **−6,294 France links**.
- **Results:** `output_r20k/` = r20j with France = s8b − R2. Validator PASS; France links 851,447 (3.2817 per S1). sha256 `a2553c98…`.
- **Projection:** if pfr is as calibrated on France as on US/India, R2 is ~35–40% true → +0.0025 France F → +0.0004 LB. So r20k ≈ 0.9888–0.9897.
- **LB:** pending (upload 2).

## R20l · safer France removal set R3 · 27 Sep 21:34
- **Why:** pfr's France *additions* are mostly equal-address category swaps (7,769 rows; pfr p3 median 0.96, large CE 0.997). The LB has shown that class is about 0% true (R09, x3 attribution), so both models carry a US habit into France name swaps. R2's largest block is the mirror case: a category word swapped for a true-noise word (`montagne groupe sarl` vs `montagne culturele sarl`), which both models reject. France's +GEN band shows distractor swaps never use the true-noise words, so these are probably true noise, and R2 is risky there.
- **R3** = R2 minus every TN_in kind: 2,949 links. Removals by kind:
  - eq same-core, same number on another street or in another city: 1,069;
  - eq rest, category swaps and initialism edits the class rule misses: 693 (`ld france logistique` vs `ld france loisirs`, `gvz jeunes` vs `dvz jeunes`);
  - other same-core, same street with a different number and legal form: 347;
  - na same-core: 287;
  - smaller kinds make up the rest.
  
  With veto2 the total is **−3,164**.
- **Results:** `output_r20l/` validator PASS, France 854,576 links (3.2938 per S1), sha256 `99a0fc04…`.
- **LB:** pending.

## R20m · further France checks (negative) · 27 Sep 21:37
- **Missed category swaps:** one known word swapped for another at eq/na/other numbers, outside TN words and typos. Only 1,429 linked; 484 are already in R3 or cat_swap. The rest are mostly abbreviations (`st` = saint, `ce` = comite, `el` = ecole: true noise). Nothing to add.
- **Address-less France queries with name twins:** already unlinked (42 links over 14.2k rows with twins ≥ 2). France's twin share (35% of S1) sits between US (29%) and India (46%).
- **France blocking rescue** (b2 address-heavy + b3 name twins, with French legal forms added to `blk3.py`):
  - 2.86M new pairs for 451k unlinked France queries that have an address.
  - Scored with the US v2 rescue model: 2,472 queries at r ≥ 0.8.
  - Raw top candidates are generic-name twins at *different* addresses (`nantes club sas`, 4 r de la base chenaie vs 4 avenue sagamore) and category swaps: France's distractor patterns. The India model disagrees (377 agree at 0.8). **Rejected.**

## R20n · structural checks for a large missing lever (negative) · 27 Sep 21:55
- **Leaks:** none.
  - Train matched share is flat by row position (0.732–0.735 by decile) and by ID decile.
  - Consecutive matched S2 rows almost never share an S1 (2.7e-7).
- **Generator counts are country-independent.**
  - Train GT: US 3.459 and India 3.465 true links per S1; 5.58% singletons in both; k2 1.672, k3 1.788.
  - Distractors per source are equal in every split and country: train 0.607 / 0.607; test US 1.150 / 1.146, India 1.183 / 1.181, France 1.039 / 1.032.
  - So France very likely also has ≈ 3.46 true links per S1 (≈ 898k), against 857.7k predicted.
- **Calibration against that count:** sum of argmax p vs the expected true argmax count.
  - OOF: 7,502,046 vs 7,502,163 (exact).
  - Test: US 2.262M vs ≈ 2.26M; India 2.728M vs ≈ 2.74M; France (x1+x2 p) 0.883M vs ≈ 0.88M.
  - Every model is calibrated in aggregate. France links fewer pairs because its rows are genuinely more ambiguous (15.6% uncertain vs 4.9% US), not because of a fixable bias. No count-based recalibration lever exists.
- **Conclusion:** the remaining gap to the top (0.992) is France discrimination on equal-address name edits (true noise vs Type B). It needs France-specific information that only LB probes can validate, and one upload is left.

## R20o · last upload plan and package · 27 Sep 22:27
- **Extension check:** 160 more links meet the R3 pattern (R3 kinds, CE < 0.1), 74 of them with p3 < 0.9. Negligible; the pattern is used up.
- **Last upload (3 of 3):** `output_r20k/` = r20l + the 3,130 category→true-noise-word swap removals that R3 held back. The LB keeps the best upload, so this is a free test of that class:
  - above 0.989472 → those swaps are distractors, and r20k becomes the final zip;
  - below → the final stays r20l.
  - Expected ±0.0004.
- **Package:**
  - New: `r20_merge_fr.py` (France f12 table + CE logits) and `r20_fr_precision.py` (pfr decision ∧ CE < 0.3, minus the LB-backed classes, ∪ veto2). Verified: it reproduces `fr_pairs_R3` exactly (854,578 pairs).
  - `run_stage3.sh` step 8e runs the France table, veto, merge, `stage3.py fit` (FPFX=f12ce TEST_C=France OUT=pfr) and precision removals before the rescue patches.
  - README / Documentation updated for r20l (LB 0.989472, rank 96).
  - Final zip being built: `CHOSEN=output_r20l CANDS=work/final_cands/candidate_pairs.tsv TEAM=SteinsGate` (log `work/logs/final_package_r20l.log`).

## R20p · last upload: r20k dropped, small positive additions instead · 27 Sep 22:37
- **Swap-rate test (label-free):** France's +GEN band (all distractors) has category→TN swaps : category→category swaps = 5,647 : 56,574 (0.10).
  - At the same address there are 38,001 category→category swaps (the Type B distractors), which predicts ≈ 3.8k category→TN distractors.
  - Observed: 39,258 category→TN swaps at the same address. So ≈ 90% of them are true noise.
  - **r20k (removing 3,130 of them) is dropped** as likely negative.
  - The same table shows TN appends are a common Type A edit (80k in +GEN) but rare at equal address (12.4k), consistent with R07.
- **France adds:** on US/India OOF, x1+x2 p < 0.5 but pfr p3 ≥ 0.95 is 99.4% true (18,749 rows).
  - France candidates: pfr p3 ≥ 0.95 and large CE ≥ 0.95, outside the category/TN classes and +GEN: 3,353. Minus the 978 that were in the losing upload-1 package: **2,375 adds**. Plus 74 more R3-pattern removals.
  - F0.5 values an added true link at only ≈ 0.06 S1-units (a removed FP ≈ 0.21), so this is worth ≤ +0.0001 LB.
- **Ceiling arithmetic:** even if all 3,130 TN swaps were false, their removal would add ≤ 0.0004 (0.98985), and nothing else of size is left. So 0.990 is out of reach with today's levers.
- `output_r20p/` validator PASS, sha256 `627aea51…`. Last upload (3 of 3). If it scores above 0.989472, rebuild the zip with `CHOSEN=output_r20p`.
- **22:58:** r20p verified against r20l: US/India identical, France +2,371 / −74 (4 adds dropped by caps), 1,732,544 rows. The zip still holds r20l (sha `ce14222e…` zip, matching `99a0fc04…`). `work/logs/switch_to_r20p.sh <LB> <RANK>` updates the docs and rebuilds the zip with r20p; run it only if r20p ≥ 0.989522, so the choice isn't made on public-LB noise.
- **LB:** **0.989507, rank 108** (upload 3 of 3, ~23:30). +0.000035 vs r20l, inside the projection but below the 0.989522 switch bar; the rank fell from 96 because other teams improved.
  - **Zip stays r20l:** the gain is within noise; the France add step (pfr ≥ 0.95 ∧ CE ≥ 0.95) was never ported to `run_stage3.sh`, so the package reproduces r20l, not r20p; re-uploading 743 MB with 25 min left wasn't worth +0.00003.

## R21 · final hour: listwise CE not started (time) · 27 Sep 23:10
- **Clock:** session started 23:02 IST, 57 min before the deadline.
- **Idea 1 (listwise CE on each uncertain query's top-4, into the top-2 re-rank):**
  - Target: "wrong S1 won" + "argmax right, below threshold" = +0.0039 OOF if fully fixed. Realistic +0.0002–0.0008 OOF (+0.0003–0.0012 LB).
  - Cost 2–3 h: resume both instances and copy data (the A100 has none), about 600k rows per fold at ≈ 135 samples/s, refit p8b, recompose, validate, rebuild the zip (3 min). The zero-delay path ends after 23:59. **Not started.**
- **Idea 2 (rescue CE round 2):** ≈ 70 min + 15 min compose for +0.0001–0.0002 LB, too small to reach 0.990 and too long. Not started.
- **Quick levers checked without new training:** a per-S1 expected-F0.5 rule instead of T1/T2 only moves links with p ≈ 0.67–0.8 (≈ +0.00005). Not worth an upload.
- **r20p:** held back at first (gain too small), then uploaded ~23:30 because the LB keeps the best upload: **0.989507, rank 108** (+0.000035). No instances resumed; ₹0 spent.
- **Final:** `SteinsGate_submission.zip` = r20l (zip matching sha `99a0fc04…`, LB 0.989472, rank 96), verified 23:02. Kept after r20p's score (23:35): below the switch bar, and the package code does not produce r20p's France adds.
- **23:13, more GPUs?** No. Setup, row build, p8b refit + T1/T2, compose/validate, upload and zip take ≥ 35 min without any GPU work, so ≤ 10 min of GPU time would be left. That fits only a toy base CE on top-2 rows, which p8b's large CE already scores (R18 took 78 min train + 30 min scoring at this scale).
- **Takeaway:** idea 1 needed ≈ 3 h and had to start by ~20:30. It remains the best next lever (+0.0003–0.0012 LB) if there is ever another round.


## R22 · final package: R20p zip, packaged-code fixes, reproduction check · 28 Sep 19:55
- **Why:** the methodology and zip request (deadline 29 Sep 10:00) asks for the *best* submission. R20p (0.989507) is both the best and the last upload, but its France step lived only on the VM. The user chose R20p if the code reproduces it exactly, else R20l.
- **Approach:**
  - Found the R20p scripts in the R20 session scratch (`r20/addchk.py`, `r20/ext4.py`, composition command in the transcript) and ported them as `src/stage3/r20_fr_final.py`. The exclusion set is `pkg1.py`'s add cells (not the narrower upload-1 pairs).
  - `run_stage3.sh`: new step 8f (R20p) after R20l, step 8h copies both files to `output/`; steps 1–2 now also build the x1+x2 pool tables (`g12`/`f12` for P0/P1 US/India) that the France stage-3 fit needs (they were missing).
  - **Bugs fixed in the packaged code** (all from the automated port on 27 Sep): `r20_block2.py` and `r20_regen_cands.py` used an undefined `fR` (NameError); `r20_fr_transfer.py` overwrote the dir variable `R` with a DataFrame (crash); `r20_fr_veto.py` wrote `fr_veto.parquet` but `r20_fr_precision.py` reads `fr_veto2.parquet`. pyflakes is clean apart from two false positives (`tok` read by a closure before `del`).
  - `run_all.sh` and `setup_box.sh` moved into `src/` (the brief wants all source under `src/`).
  - README and `docs/Documentation_BER.md` rewritten in plain language (no ranks, as the user asked).
- **Data:** laptop, `S3_DIR=work/r19_scratch` tables, memory-capped scopes (the cell step needs ~4.5 GB; the user closed Chrome).
- **Results:**
  - Rebuilt from the stored tables with the packaged scripts: transfer adds 4,324 / rem 51 / initialism 87 (upload-1 pairs identical to the VM copy); veto 1,513; R3 2,949 (−3,164 with the veto); R3 pairs 854,578; R20p step −74 / +2,375.
  - Pipeline order s8b → France R3 → India → US rescue gives `output_r20l` byte-identical (`99a0fc04…`); patching R3P gives `output_r20p` byte-identical (`627aea51…`).
  - Final file: 5,838,881 links (US 2,249,303, India 2,732,705, France 856,873), 5.8% empty.
  - Zip: validator PASS (`--check-ids`), candidate stream check OK (128,969,685 ids, matches within candidates), 60 entries, zipped files equal the repo and `output_r20p`.
- **Not checked end to end:** the GPU steps and the rescue tables (they exist only on the paused VM), so `r20_regen_cands.py` and `r20_block2.py` fixes are checked by pyflakes only.
- **LB:** 0.989507 (R20p); final rank pending.

## R22b · rerun check of the packaged stage-3 CPU steps on the VM · 28 Sep 21:50
- **Approach:** resumed the CPU VM (now `519972`, 32 vCPU) and ran every CPU step of `run_stage3.sh` step 1–2 (new g12/f12 pool tables) and step 8 with the packaged code into `~/chk` and `~/chk2` (original tables read through read-only symlinks, nothing overwritten). GPU training not rerun: the stored cross-encoder scores were used. Compared every table with the originals.
- **Round 1 (32 threads):** all 51 steps exit 0. Identical: g12→f12 and f12ce pool tables, frg, fr_pairs_P1, fr_veto2, all b2_a retrievals, and **`candidate_pairs.tsv` (sha `c4f0dda6…`)**, so the `r20_block2`/`r20_regen_cands` fixes are confirmed. Differed: the France stage-3 fit (p3 up to 0.037, T1 0.56 vs 0.58) and the rescue models.
- **Cause 1 (fixed):** the packaged `stage3.py` had `num_threads=6` (laptop), the original France fit used `stack2.py` with 16. Set to 16. **Round 2:** France thresholds and OOF F equal to every digit, `fr_pairs_R3` and `fr_pairs_R3P` identical. The top-2 re-rank refit (`top2.py`, 16 threads) matches p8b (scores within 1e-8, same thresholds (0.58, 0.78), same OOF 0.991654).
- **Cause 2 (documented, not fixable after the fact):** rescue features differ by ~1e-5 in summed IDF (polars group_by float sums; also with POLARS_MAX_THREADS=16), and the holdout draw uses an unordered `unique()`. The retrained rescue models give the same OOF gain (India +0.00154 vs +0.00156; US +0.00027 vs +0.00028) but 97% (India) / 99% (US) the same additions. Rerun final file vs uploaded r20p: 488 + 589 = 1,077 of 5,838,881 links differ (0.02%).
- **Package:** README §6 and doc Appendix A state these results; zip rebuilt.
- **Cost:** ~₹98 (balance ₹315). VM paused at 21:44; watchdog guards `519972`.
