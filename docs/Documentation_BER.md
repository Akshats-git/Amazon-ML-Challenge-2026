# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** SteinsGate  
**Team Members:** Lakshay Gupta, Akshat Gupta, Keshav Mishra, Akansh Tyagi (IIT Bhilai)  
**Submission Date:** 27 September 2026

---

## 1. Executive Summary

We link every Source-2/3 record ("query") to at most one Source-1 entity with four parts:
1. a country-agnostic normalizer (transliteration dictionaries learned from train);
2. per-country sparse TF-IDF blocking (top-10 S1 per query);
3. a two-stage LightGBM scorer;
4. a per-entity decision policy (argmax per query; per S1, first link at p ≥ T1 and later links at p ≥ T2).

The largest gains came from reverse-engineering how the unmatched records were generated. A distractor is a copy of a real S1 record whose house number is shifted **upward by one of {1, 2, 3, 4, 5, 7, 9, 11, 13, 21}** and whose name is optionally edited. Features that see this mechanism took out-of-fold macro F0.5 from 0.976 to **0.988**:
- a name-edit lexicon;
- signed number relations;
- a label-free "number-oracle" lexicon, learned per country without labels.

All thresholds and scores come from cross-fitted pools rebuilt at test distractor density. France has no training data, so every France-specific choice was judged on the public leaderboard. It preferred the earlier, simpler models' France predictions, so the final file uses the new x3+x4 blend for US and India and the x1+x2 blend for France.

---

## 2. Methodology

### 2.1 Problem Analysis

- **Structure (train GT):**
  - Every S2/S3 record matches at most one S1, and no pair crosses countries. So per-query argmax is safe and blocking runs per country.
  - Per S1: 1.67 S2 matches on average (max 5) and 1.79 S3 matches (max 6); 5.58% of S1 are singletons.
  - Train has 1.2 unmatched queries per S1. Test has ~2.3: query volume and noise-marker counts give the same figure.
- **True-match noise:**
  - OCR digit/letter swaps, doubled/dropped letters, legal-form variants, DBA aliases, web handles, ID tags.
  - Abbreviations, missing addresses, non-Latin (mainly Indian) scripts.
  - Symmetric house-number noise: ±1, ±2, ±10, ±20, dropped digits, digit edits.
  - Occasionally the whole name is replaced by a coined word at the same address.
- **Distractor generator (measured at full scale):**
  - Copy an S1 record and shift its house number *upward* by a value in GEN = {1, 2, 3, 4, 5, 7, 9, 11, 13, 21} (each ≈ equally likely).
  - Optionally also: append a word from a small per-country list (US *holdings, group, partners, downtown, eastgate…*; India *enterprises, exports, overseas, ventures…*; France *holding, participations, international, distribution, groupe, développement, france*), swap a word, or change the legal form.
  - The street and city stay the same, so these records get high blocking scores. The **sign** of the number difference separates them: in US top-1 pairs, P(true | q − s ∈ {−1, −2}) ≈ 0.80 but P(true | q − s ∈ GEN) ≈ 0.025.
  - Our first number features only saw |q − s|.
- **France appears only in test.** Nothing may depend on the country label, and every France-specific choice can only be judged on the leaderboard (see §5).

### 2.2 Solution Strategy

**Approach Type:** Blocking + two-stage gradient-boosted classifier + per-entity decision policy.  
**Core Innovation:**
- **Generator-aware features:**
  - *14 edit features:* a cross-fitted log-odds lexicon of the words the query adds or drops, the edit position, and unmatched-number offsets.
  - *16 number features:* signed number relations against the GEN shift set, plus the label-free number-oracle lexicon (NOL).
- **Test-density out-of-fold pools.** The S1 of each country are split in halves by `md5(entity_id) mod 2`. Pool P_k^c holds half k's S1, their matches and **all** unmatched queries of country c, i.e. ~2.4 unmatched queries per indexed S1. Every stage is cross-fitted P0 ↔ P1.
- **Stage-2 competition context.** Built from out-of-fold stage-1 scores: the query's runner-up margin, and the other queries competing for the same S1.

---

## 3. Candidate Generation (Blocking)

- **Normalization** is the same for every source and country:
  - unidecode, lowercase, `&` → and, OCR fixes inside alphanumeric tokens, doubled-letter collapse;
  - legal forms canonicalized and split off (`legal`, `name_core`, squashed `name_sq`); DBA names split into both parts;
  - null addresses flagged; address digit strings extracted (`addr_nums`, leading zeros stripped).
  - **Transliteration dictionaries** are learned from train true pairs with a non-Latin query (token alignment; ≥ 3 occurrences, ≥ 50% purity): 694 name and 30 address entries. On held-out S1 they raise the share of non-Latin true pairs that share a core token from 0.31 to 0.999.
- **Blocking keys:** one sparse TF-IDF matrix per country over the S1 index (sublinear tf, **absolute document-frequency cap 10,000**, blocks L2-normalized and weighted):
  - name word unigrams (0.25);
  - address word uni+bigrams (0.50);
  - character 4-grams of `name_sq` (0.25).

  Each query keeps its **top-10 S1 by cosine** (`sparse_dot_topn`).
- **Candidate pairs generated:** **99,695,099** on test (US 38.2M, India 47.2M, France 14.3M), 10 per query.
- **Keeping true matches:**
  - The DF cap keeps rare tokens and drops stop-like ones.
  - The char-4 block survives spacing and typos, and the dictionaries recover transliterations.
  - Measured on the pools: R@10 **0.991** (US) and **0.984** (India); a perfect matcher on these candidates would reach macro F0.5 0.997 / 0.995.
  - An unpruned char-3 index was only as good (R@10 0.988 / 0.975) at ~20 ms/query (≈ 60 h for test), so we rejected it.

---

## 4. Matching Model

**Features used (stage 1: 80, all country-agnostic):**
- *Base (50):*
  - blocking context: cosine, rank, gap to the query's top, candidate counts, how often the S1 is someone's top-1, name/address frequencies, out-of-vocabulary share;
  - query flags: alias, web, ID tag, non-Latin, source, missing address;
  - name: rapidfuzz ratio / token-sort / token-set / partial / Jaro-Winkler, IDF-weighted soft Monge-Elkan both ways, unmatched IDF mass, legal-form equal / conflicting;
  - address: ratio, token-set, soft Monge-Elkan, shared bigrams, maximum shared IDF;
  - numbers: exact / first / soft match, minimum absolute difference, shared long number.
- *Edit (14):*
  - For the query's **extra** name tokens (absent from the S1) and the S1's **missing** ones: the max/min of a smoothed log-odds of "not a true pair".
  - The lexicon is learned on labelled pool pairs, counted once per query (≥ 50 queries), and cross-fitted (P0 features use the P1 lexicon).
  - Also: whether the extra token is appended or prepended, its label-free frequency as the single edit of top-1 pairs, and unmatched-number counts, offset and prefix/suffix relation.
- *Number (16, R10):* q_only / s_only are the numbers one address has and the other lacks; every (q_only, s_only) pair is compared as q − s.
  - Features: all equal; count shared; # pairs with q − s ∈ GEN; ∈ {−1, −2}; ∈ −GEN; signed closest difference; Levenshtein-1 digit edit; transposition; digit-length difference.
  - Street-token Jaccard, *equal numbers & same street*, and *single +GEN shift & same street*.
  - **Number-oracle lexicon (NOL), label-free, computed per partition (each pool and each test country):**
    - On blocking rank-1 pairs, a single +GEN shift on the same street marks a distractor proxy D, and equal numbers on the same street a true proxy T.
    - For each name token w the query adds (or drops): nol(w) = log((c_D(w) + 1)/N_D) − log((c_T(w) + 1)/N_T), with ≥ 20 occurrences required.
    - Features: max/min over the pair's tokens.
    - Without labels it reproduces the supervised lexicon (US *eastgate* 8.4, *holdings* 7.3; *dba* −8.1). In France it finds that country's generator words (*international, participations, distribution, holding* ≈ 4.7) with no French training data.
- *Stage 2 (+11):*
  - per query: p1, max/second p1, margin, rank, is-argmax;
  - per S1, over the queries whose argmax it is: the count with p1 ≥ 0.5 (all / same source), Σp1, the best competitor, and the pair's rank.

**Model type:** LightGBM binary classifiers.
- Stage 1: 127 leaves; stage 2: 63 leaves.
- lr 0.05, min_child 100, feature/bagging fraction 0.8, λ2 1; early stopping on a 10% query holdout.
- The supervised lexicon is blanked for 50% of training queries (feature dropout), so the model also learns the label-free route France needs.
- Two cross-fitted models per stage, each trained on **all** of its pool's queries (58.5M pairs, 80 features) on a 32-vCPU VM. Test scores average the two pool models.
- A parameter A/B on 10% of P0 (stage-1 holdout log-loss) found 255/511 leaves no better.
- The number features cut log-loss by **8.3%** at equal data (0.004202 → 0.003854); doubling the data had given 5.5%.

**Threshold selection method:** grid search of (T1, T2) for **macro F0.5** on the pooled out-of-fold test-density predictions, T1 ∈ [0.10, 0.90], T2 ∈ [0.30, 0.96], step 0.02. The final uses (0.56, 0.74) for the x3+x4 blend (US, India) and (0.52, 0.74) for the x1+x2 blend (France).
- The optimum is flat (top points within 1e-5).
- Per-country thresholds, an expected-F rule and a third threshold each gave ≤ +0.00002.
- A label-free plug-in check puts France's optimum in the same place as US/India's.
- Links beyond the generator's caps (5 S2 / 6 S3 per S1) are dropped, lowest p first.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** out-of-fold on the test-density pools (2.2M S1): **0.98848** for the x3+x4 blend (x3 alone 0.98838: US 0.98977, India 0.98630). Public leaderboard: **FINAL_LB** for the final file (best so far 0.982401, rank 320).
- **Common false positives (wrong merges):**
  - Queries **without an address** matched on the name alone when several S1 share that name ("Wildlife Committee LLC" in NY and MO).
  - Generator records whose word edit is also common true-match noise ("… Center", "… Private Limited").
  - 93% of false positives come from queries with no true candidate.
- **Common false negatives (missed matches):**
  - **38% blocked out**: mostly address-less queries whose name belongs to several S1 (unresolvable), plus fully Devanagari/Bengali names.
  - **40% right argmax but below threshold**: OCR-damaged names, true matches with a house number off by one or two.
  - 22% another S1 won.

| run | what changed | OOF F0.5 | public LB |
|---|---|---|---|
| tier 0 | normalization + blocking + 50 features, stage 1, T = 0.76 | 0.97623 | – |
| R05 (x1) | + 14 edit features, stage 2, tuned (T1, T2); 15% of queries | 0.98645 | – |
| R06 (x2, blend x1+x2) | 30% of queries; blend | 0.98706 | **0.981021** |
| R07 | + France lexicon imputation | 0.98706 | 0.979881 |
| R08 | + France normalization rules | 0.98706 | 0.979 |
| R09 | R06 + France same-address rescue | 0.98706 | 0.976531 |
| R10 (x3) | + 16 number features, all queries (58.5M pairs per model), US/India only; France from R06 | **0.98838** (US 0.98977, India 0.98630) | **0.982401** |
| R10 (x3, all countries) | the same with x3's own France predictions | 0.98838 | 0.982123 |
| **final** | **x3+x4 blend (x4: 255-leaf variant) for US/India; x1+x2 for France** | **0.98848** | **FINAL_LB** |

### What we tried that did not help (measured)

| idea | result |
|---|---|
| **France lexicon imputation** (R07): give France's frequent appended words the train generator-word weight | −0.0011 LB. *france, groupe, développement* are also French true-match noise |
| **France normalization rules** (R08): street types, regions/departments, spaced legal forms, et → and | −0.0009 LB (on top of R07); "et fils" → "and fils" looked like the US "& Sons" distractor |
| **x3's own France predictions** (number features + NOL) | −0.00028 LB vs the x1+x2 France predictions; x3 links more equal-address name swaps, which in France are mostly distractors |
| **France same-address rescue** (R09): force-link argmax pairs with equal house numbers on the same street | −0.0045 LB; only ~25% of the 50.7k added links were true. France's generator keeps the house number far more often than US/India (−0.0012 on US/India OOF too) |
| bigger trees (255 / 511 leaves) | stage-1 log-loss +0.8% / +1.9% |
| per-country thresholds / expected-F rule / third threshold | ≤ +0.00002 OOF |
| unpruned char-3 blocking | ~60 h for test |
| per-pair (instead of per-query) lexicon counts | leaks labels (unmatched queries sit in both pools) |

---

## 6. Conclusion

- **Largest gain:** modelling the generator of the unmatched records: which words it appends, and that it shifts house numbers upward by a fixed set. Features built on that mechanism were worth more than any model or data change.
- **Validation:** test-density, cross-fitted pools made thresholds and model choices transfer to test.
- **France:** with no labels, the leaderboard was the only judge. Every France-specific change lost there: three hand-made rules, and the stronger model's own France predictions. France keeps the earlier x1+x2 predictions. Its generator differs from US/India: it often keeps the house number and swaps an organisation word, so "same address" is weaker evidence there.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/` (entry point `src/run.py`; steps in `README.md`; `run_all.sh` runs everything):

| module | role |
|---|---|
| `ber/normalize.py`, `ber/translit.py` | normalization; transliteration dictionaries learned from train |
| `ber/pools.py`, `ber/blocking.py` | test-density pools; per-country TF-IDF top-10 blocking |
| `ber/features.py`, `ber/xfeats.py`, `ber/nfeats.py` | 50 base, 14 edit, 16 number features |
| `ber/model.py`, `ber/context.py` | cross-fitted two-stage LightGBM, stage-2 context |
| `ber/decide.py`, `ber/metrics.py` | argmax + (T1, T2) policy, caps, macro F0.5, loss decomposition |
| `ber/submit.py`, `ber/rescue.py` | test inference, blends, validator; the (disabled) R09 rescue |

- **Runtime:**
  - On a 6-core / 13 GB laptop: normalization + dictionaries 20 min, blocking 1.6 h, features 45 min, edit features 26 min.
  - On a 32-vCPU / 128 GB VM: number features 19 min; run x3: stage 1 61 min (two 3,000-round models on 58.5M pairs each), stage 2 36 min, tuning 3 min, test scoring ~40 min.
- **Data:** no external data or pretrained models. Every statistic comes from the provided files.
- **Libraries:** polars, numpy, scipy, scikit-learn, sparse-dot-topn, rapidfuzz, Unidecode, LightGBM (MIT/BSD/Apache).
