# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** ⟨TBD⟩  
**Team Members:** Lakshay Gupta, Akshat Gupta, Keshav Mishra, Akansh Tyagi (IIT Bhilai)  
**Submission Date:** 27 September 2026

---

## 1. Executive Summary

We resolve every Source-2/3 record ("query") to at most one Source-1 entity with a four-part pipeline:

1. **Normalization.** One country-agnostic normalizer is used for every source. Its transliteration dictionaries are learned from the training ground truth.
2. **Blocking.** Sparse TF-IDF blocking runs per country and returns the top-10 S1 records per query.
3. **Scoring.** A 50-feature LightGBM pair scorer is followed by a second LightGBM that adds out-of-fold *competition context*: how the query's other candidates scored, and which other queries already point at the same S1.
4. **Decision.** Each query links to its argmax S1 only. Per S1, the first link is kept if p ≥ T1 and later links if p ≥ T2.

The key design choice is the validation set. We rebuilt train into two disjoint *test-density* pools (half of each country's S1 plus **all** unmatched records). Thresholds and every reported score therefore come from out-of-fold predictions that face the same distractor pressure as the test set, which has 2.3 distractors per S1 against 1.2 in train. France, which is absent from training, is handled by the same country-agnostic features.

---

## 2. Methodology

### 2.1 Problem Analysis

Facts measured on train that shaped the design:

- **One-to-one structure:** every S2/S3 record matches **at most one** S1 (0 violations), and no true pair crosses countries. Per-query argmax is therefore safe, and blocking can run per country.
- **Singletons and distractors:** 5.59% of S1 have no match, and they score 1.0 only if left empty. Train has 1.22 unmatched queries per S1. In test we estimate ~2.3 per S1: query volume and noise-marker counts give the same figure. Precision on unmatched queries is the main lever: in a small-scale study, 84% of false positives came from queries with no true candidate.
- **Noise patterns:**
  - OCR digit/letter swaps (0↔o, 1↔l, 5↔s) and doubled/dropped letters.
  - Legal-form variants (LLC/L.L.C., Pvt/Private, Ltd/Limited, SARL/SAS…).
  - DBA/"trading as" aliases, web handles (`.com`, `www.`, `@`) and ID tags (`(ID: 123)`, `#1234`).
  - Punctuation and case differences, `&`/"and", token reordering, address components in permuted order.
  - Abbreviations (Street/St, Nagar/Ngr), missing addresses (`N/A`, empty).
  - **Non-Latin script** records, mainly Indian: some queries are fully transliterated.
- **France** appears only in test, so no feature, dictionary or threshold may depend on the country label.
- **How the unmatched records are made (found at full scale):** the unmatched queries are generated hard negatives. The generator copies a real S1 record and:
  - shifts the house number by a few units (524 → 537, 3931 → 3934), sometimes keeping the unit number;
  - appends one word from a small per-country list: US *partners, holdings, group* plus ~20 location words (*north, downtown, westgate, …*); India *public, enterprises, industries, ventures, exports, overseas, infratech*; France *groupe, holding, participations, développement, …*;
  - or changes the legal form.

  The street and city are unchanged, so these records get a high blocking score (90th-percentile cosine 0.74). In pairs where the query name has exactly one extra token, *holdings* and *group* occur ~59,000 times in non-matching pairs and never in true ones, while *the* (29.8k true vs 5.5k) is typical true-match noise. String similarity cannot separate "X Holdings" from "The X".

### 2.2 Solution Strategy

**Approach Type:** Blocking + two-stage gradient-boosted classifier + per-entity decision policy.  
**Core Innovation:**
- **Edit features aimed at the generator:** a cross-fitted log-odds lexicon of the words the query name adds or drops relative to the S1 name, whether the extra word is appended (generator) or prepended (noise), a label-free frequency of that word, and house-number shift features. Together they cut stage-1 log-loss by about 29% at equal data.

- **Test-density out-of-fold pools.** Thresholds are tuned under realistic distractor pressure.
- **Stage-2 competition context.** The second model sees each query's runner-up margin and the other queries competing for the same S1. That is what lets it reject look-alike distractors that a pairwise model cannot tell apart.
- **Learned transliteration dictionaries.** They are learned from the training ground truth, not hand-made.

Validation pools: S1 of each country are split into halves by `md5(entity_id) mod 2`. Pool P_k^c holds half k's S1, the queries matched to them, and all unmatched queries of country c. That gives about 2.4 unmatched queries per indexed S1, close to test. Blocking, features and models are built per pool. Stage 1 and stage 2 are cross-fitted P0 ↔ P1, so every pool prediction is out-of-fold.

---

## 3. Candidate Generation (Blocking)

**Normalization (same for S1, S2, S3 and all countries).**

- *Name:*
  - Strip ID tags and web handles.
  - unidecode → lowercase → `&`→"and" → non-alphanumerics to spaces.
  - OCR fixes inside alphanumeric tokens, then doubled-letter collapse.
  - For non-Latin records, the learned name dictionary.
  - Legal forms canonicalized and split off (`legal`), leaving the core name (`name_core`) and a squashed core without spaces (`name_sq`).
  - A DBA name is split into both parts.
- *Address:* the same basic cleaning, the learned address dictionary for non-Latin records, null markers → missing, and extraction of the digit strings (`addr_nums`).
- *Transliteration dictionaries:* learned from train true pairs whose query is non-Latin. For names, token-by-token alignment when the token counts match; for addresses, the single residual token. An entry is kept when seen ≥ 3 times with ≥ 50% purity. Result: 694 name and 30 address entries. On held-out S1, the share of non-Latin true pairs that share a core token rose from 0.309 to 0.999.

**Blocking keys used.** One sparse TF-IDF matrix per country over the S1 index (sublinear tf, smoothed IDF, **absolute document-frequency cap of 10,000**, each block L2-normalized and weighted):

| block | tokens | weight |
|---|---|---|
| name words | unigrams of the normalized name | 0.25 |
| address | word unigrams + bigrams of the normalized address | 0.50 |
| squashed name | character 4-grams of `name_sq` | 0.25 |

Each query keeps its **top-10 S1 by cosine** (sparse top-n matrix product, `sparse_dot_topn`).

- **Candidate pairs generated:** **99,695,099** on test (US 38.2M, India 47.2M, France 14.3M): 10 per query, 9.97M queries. That is ≈ 1 candidate pair per 67,000 pairs of the per-country S1 × query cross product (6.7·10¹² pairs).
- **How we ensured true matches were not lost:**
  - The DF cap keeps rare, discriminative tokens and removes stop-like ones such as street types and city names.
  - The char-4 squashed-name block survives spacing, typos and concatenations.
  - The learned dictionaries recover transliterated names: with them, India reached R@1 0.963 / R@10 0.981 on the full-country index in our study.
  - We measured pair recall on the test-density pools (true pairs kept in the top-10):

| pool | R@1 | R@10 | oracle F0.5 (perfect matcher on candidates) |
|---|---|---|---|
| US (P0 / P1) | 0.976 / 0.976 | 0.991 / 0.991 | 0.997 / 0.997 |
| India (P0 / P1) | 0.968 / 0.968 | 0.984 / 0.984 | 0.995 / 0.995 |

  An unpruned character-3-gram TF-IDF index reached R@10 0.988 (US) / 0.975 (India) at ~20 ms/query (≈ 60 h for test), so we rejected it.

---

## 4. Matching Model

**Features used (stage 1, 50 features, all country-agnostic):**

- *Blocking context (12):* blocking cosine and rank, the query's top-1 and top-2 cosine, gap to the top, number of candidates, how often the S1 is someone's top-1, how many queries retrieved the S1, S1 name/address frequency, query name frequency, and the share of the query's name tokens unseen in the index.
- *Query flags (6):* alias, web handle, ID tag, non-Latin, source S3, missing address.
- *Name (16):*
  - rapidfuzz ratio, token-sort, token-set, partial and Jaro-Winkler on the core name, ratio on the full name, ratio/partial on the squashed name. Alias queries take the maximum over their parts.
  - **Soft Monge-Elkan in both directions**, IDF-weighted. Token similarity: exact; prefix/subsequence 0.9; Levenshtein ≥ 0.8; numeric ±2 at 0.7. Plus the IDF mass of unmatched tokens on each side and the length difference.
  - Legal form equal / conflicting / both empty.
- *Address (9):* ratio, token-set, partial-token-set, soft Monge-Elkan both ways, unmatched IDF mass, shared bigrams, maximum IDF of a shared token, token-count ratio.
- *Numbers (7):* any exact shared number, first number equal, soft number match (±2 or prefix/suffix), minimum absolute difference, both have numbers, number conflict, shared long number (≥ 5 digits, e.g. ZIP/PIN).

**Edit features (14, added in R05):**

- *Name-edit lexicon:* the query's **extra** name tokens (absent from the S1 name) and the S1's **missing** tokens (absent from the query). For each side, the maximum and minimum of a smoothed log-odds of "not a true pair", plus the counts.
  - The lexicon is learned on labelled pool pairs with a small name edit (≥ 1 shared token, ≤ 2 extra, ≤ 1 missing).
  - It counts each query at most once per token and needs ≥ 50 queries per token. Unmatched queries sit in both pools, so pair-level counts leaked their own labels.
  - It is cross-fitted: P0 features use the P1 lexicon and vice versa; test uses both.
  - During training the lexicon features are blanked for a random 30% of queries (feature dropout), so the model also learns the label-free route it needs in France, whose generator words never occur in train.
- *Lexicon imputation for unseen vocabularies (France):*
  - In each partition, a word that makes up ≥ 0.5% of the top-1 pairs as their single appended edit (appended last in ≥ 90%) has the generator profile.
  - If it lacks that profile in the train pools, its lexicon value is raised to the median log-odds of the train generator-profile words (9.16).
  - In test this fires only for France's seven generator words and leaves the pools untouched.
  - It closes a measured gap. Before the fix, a query appending a generator word to an otherwise identical record was linked in 2.1% of such top-1 pairs in France, against 0.3% in the US and ~0% in India.
  - Applied to the final model, it removes 22.4k France links and adds 0.7k; US and India are untouched. All but 406 of the removed links have a French generator word as an extra token, either appended or replacing a word at an identical address ("Mam Club SARL" vs "Mam SARL France").
  - France moves to 3.23 links per S1. That is what its query volume implies at the same distractor rate as US and India (≈ 3.23 true matches per S1), where it had been 3.31.
- *Label-free edit shape:* whether the extra token is the query's last (appended) or first (prepended) core token, and whether the missing token is the S1's last. Also the frequency of the extra token as the single edit of top-1 pairs in the same partition, computed on the test partition itself for France.
- *Unmatched numbers:* numbers present in one address but not the other (counts), their minimum absolute offset (a generator shift is small), and whether one is a prefix or suffix of the other (the dropped-digit noise of true matches).

**Stage 2 (11 additional features, computed from out-of-fold stage-1 probabilities p1 over all candidates):**

- *Per query:* p1, the query's maximum and second p1, the margin, the pair's rank in the query, and whether it is the query's argmax.
- *Per S1, among the queries whose argmax is this S1 (excluding itself):* the count with p1 ≥ 0.5 overall and from the same source, the sum of p1, the maximum competing p1, and this pair's rank.

**Model type:** LightGBM binary classifiers.

- Stage 1: 127 leaves; stage 2: 63 leaves.
- Learning rate 0.05, min_child_samples 100, feature/bagging fraction 0.8, early stopping on a 10% query holdout.
- Two models per stage, cross-fitted on the pools (train on P0 → score P1 and vice versa). Each is trained on ⟨15%⟩ of its pool's queries, ⟨≈10M⟩ pairs. On test, stage-1 and stage-2 scores are the average of the two pool models.

**Threshold selection method.** Each query links only to its highest-p S1, and ties (margin < 1e-6) get no link. Within each S1, links are ranked by p: the first is kept if p ≥ **T1** and later ones if p ≥ **T2**. We grid-search (T1, T2) directly for **macro F0.5** on the pooled out-of-fold, test-density predictions, from T1 ∈ [0.10, 0.90] and T2 ∈ [0.30, 0.96] in steps of 0.02. Selected: **T1 = 0.52, T2 = 0.74** for the final blend (single runs: (0.48, 0.76) and (0.50, 0.76)). The optimum is flat: the top-8 grid points are within 1e-5. Alternatives gave nothing on the same OOF:
- per-country thresholds: +0.00000;
- an expected-F0.5 per-entity rule: +0.00002;
- a third threshold for rank ≥ 3: +0.00000.

A single pooled pair also applies unchanged to France.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro), out-of-fold on the test-density pools (2.2M S1): 0.98706** for the final blend (single 30%-data run: 0.98705, US 0.98848, India 0.98490). Public leaderboard (a subset of test): **0.97988**, rank 146 (26 Sep). Most of the 0.007 gap to OOF is France, which has no labels, so OOF cannot measure it; label-free diagnostics put France near 0.95 against about 0.985 for US and India. The bucket table, loss decomposition and error analysis below are for the 15%-data run (0.98645, US 0.98799, India 0.98412, precision 0.996, recall 0.969); the final run has the same profile with ~9% fewer false positives.
- By number of true matches per S1:

| true matches | share of S1 | F0.5 | precision | recall |
|---|---|---|---|---|
| 0 (singleton) | 5.6% | 0.980 | – | – |
| 1 | 5.4% | 0.957 | 0.982 | 0.964 |
| 2–3 | 41.1% | 0.987 | 0.995 | 0.970 |
| 4+ | 48.0% | 0.990 | 0.998 | 0.968 |

- **Loss decomposition (OOF, tuned):**
  - With every false positive removed, F would be 0.9901.
  - With every false negative that blocking kept recovered, F would be 0.9926.
  - The 237k false negatives split into 38% blocked out, 40% right argmax but below threshold, and 22% another S1 won.
  - 93% of the 39.7k false positives come from queries with no true candidate.
- **Common false positives (wrong merges):**
  - 81% are the third-or-later link of an S1.
  - Typical cases are queries without an address matched on the name alone, and S1 that share a name with a business in another city ("Wildlife Committee LLC" in NY and in MO).
  - The rest are generator records whose appended word is also common true-match noise ("… Center", "… Private Limited").
- **Common false negatives (missed matches):**
  - *Blocked out* (0.9% of US, 1.6% of India true pairs):
    - 77% of the US misses (44% in India) are queries **without an address** whose name belongs to several S1, so they are unresolvable by design.
    - The rest are fully Devanagari/Bengali names or brand names with partial addresses.
  - *Below threshold* (p quartiles 0.28–0.64): OCR-damaged names ("OPTlMAL"), true matches whose house number is off by one (10990 vs 10991, which looks like a generator shift), and true matches that append a word the generator also uses ("… Enterprises", "… LLC").

| run | what changed | OOF F0.5 | LB |
|---|---|---|---|
| baseline | char-3 TF-IDF blocking, LightGBM, one threshold (small-scale study) | 0.983 | – |
| tier 0 | new normalization + blocking + 50 features, stage 1 only, T = 0.76 | 0.97623 (P1) | – |
| stage 1, cross-fit | both pools out of fold | 0.97613 (best single T 0.97668) | – |
| + edit features | 14 generator-aware side features, stage 1 | 0.98455 (0.98462) | – |
| + stage 2 + (T1, T2) | competition context, tuned policy (0.48, 0.76) | 0.98645 | – |
| + 2× training data | 30% of each pool's queries per model (float16 matrix), tuned (0.50, 0.76) | 0.98705 | – |
| + blend | mean of the 15% and 30% runs' stage-2 scores, re-tuned (0.52, 0.74) | **0.98706** | – |
| + France lexicon imputation (**final**) | test-only, label-free (pools unchanged): −22.4k / +0.7k France links | 0.98706 | **0.97988** |

---

### What we tried that did not help (measured)

| idea | result |
|---|---|
| unpruned char-3-gram TF-IDF blocking | R@10 0.988 / 0.975 (US/India), but ~20 ms/query ≈ 60 h for test, so rejected for the capped sparse design |
| per-country (T1, T2) thresholds | 0.98645 = pooled (India own (0.56, 0.76), US own (0.48, 0.78)) |
| expected-F0.5 per-entity link selection | 0.98647 (+0.00002) |
| third threshold for the 3rd+ link of an S1 | 0.98645 (+0.00000) |
| more candidates per query (K > 10) | not run. Most blocked-out true pairs are address-less queries whose name belongs to several S1 in other cities, so they are unrecoverable by any blocking key |
| lexicon counted per pair instead of per query | leaks labels (unmatched queries sit in both pools); a small A/B showed 0.00556 vs 0.00569 logloss, and the leak-free per-query version is used |
| LLVM-compiled tree inference (`lleaves`) to speed up scoring | compiling the 1,400-tree model needed > 6 GB of RAM, so we abandoned it |

## 6. Conclusion

- **Validation:** a test-density validation design (half the S1 of each country plus every unmatched record, cross-fitted) made thresholds and model choices transfer to the test set.
- **Largest gain:** understanding how the unmatched records were generated (shifted house numbers and a small vocabulary of appended words) was worth more than any model change. The 14 generator-aware features lifted out-of-fold macro F0.5 from 0.976 to 0.985 at stage 1.
- **Stacking and data:** stage-2 competition context, twice the training data and a two-run blend added the rest.
- **France:** a label-free lexicon imputation carries the generator-word signal to France, which has no labels.
- **Engineering:** every stage streams on a 13 GB laptop.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/` (entry point `src/run.py`; the steps are in `README.md`, and `run_all.sh` runs them all):

| module | role |
|---|---|
| `ber/normalize.py`, `ber/translit.py` | normalization; transliteration dictionaries learned from train GT |
| `ber/pools.py` | md5 half-split, test-density validation pools |
| `ber/blocking.py` | per-country sparse TF-IDF top-10 blocking and the recall report |
| `ber/features.py` | 50 stage-1 features (context arrays + forked string-feature workers) |
| `ber/model.py`, `ber/context.py` | LightGBM cross-fit, stage-2 context, OOF evaluation, LOCO |
| `ber/decide.py`, `ber/metrics.py` | argmax + (T1, T2) policy, threshold grid, macro F0.5, loss decomposition |
| `ber/submit.py` | test inference, both output files, checks and the validator |

Command order: `env check1 check2 norm dicts norm pools block feats check8 train1 train2 tune predict`.

- **Runtime** on a 6-core / 13 GB laptop (AMD Ryzen 5 5600H):
  - normalization + dictionaries ~20 min;
  - blocking 1 h 36 min;
  - stage-1 features 44 min (230M pairs, 8 workers);
  - edit features 26 min;
  - run x1 (15% of queries) 2 h: stage 1 36 min, stage 2 26 min, tuning 3 min, test scoring 61 min;
  - run x2 (30%) ~3.5 h;
  - blend ~15 min.
- **Data:** no external data or pretrained models. Every statistic (IDF, dictionaries, models) comes from the provided training files.
- **Libraries:** polars, numpy, scipy, scikit-learn, sparse-dot-topn, rapidfuzz, Unidecode, LightGBM (all MIT/BSD/Apache).

### B. Additional Results

- **Engineering for a 13 GB laptop:**
  - Every stage streams: 2M-pair feature files, and feature floats rounded to 10 mantissa bits (34 bytes per pair on disk).
  - Predictions are `.npy` arrays, and the stage-2 context is rebuilt in memory from p1.
  - The sparse top-n product runs faster with 4 threads than 8, because it is cache-bound.
- **Deviations from the original design:**
  - "emmalynn" → "emalyn" (the doubled-letter collapse is applied everywhere).
  - Typographic punctuation (U+2000–U+20CF) is not treated as non-Latin.
  - Legal forms are canonicalized inside `name_norm`.
  - Stage-2 ranks give tied competitors the same rank.
- ⟨LOCO / hand-map A/B results if run⟩
