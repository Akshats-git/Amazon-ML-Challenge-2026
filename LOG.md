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

**Current best on LB:** none submitted yet

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
