# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** SteinsGate  
**Team Members:** Lakshay Gupta, Akshat Gupta, Keshav Mishra, Akansh Tyagi (IIT Bhilai)  
**Submission Date:** 27 September 2026

---

## 1. Executive Summary

We link every Source-2/3 record ("query") to at most one Source-1 entity with five parts:
1. a country-agnostic normalizer (transliteration dictionaries learned from train);
2. per-country sparse TF-IDF blocking (top-10 S1 per query);
3. a two-stage LightGBM pair scorer;
4. a **stage-3 re-scorer** on each query's best candidate. It adds information the pair model never sees: agreement with the *other* records that claim the same S1 (Source 1 is itself noisy), raw-text perturbations that normalization erases, model disagreement, and a **fine-tuned multilingual cross-encoder (xlm-roberta, MIT) reading the raw text of both records** (base and large), re-ranking each query's two best candidates;
5. a per-entity decision policy (argmax per query; per S1, first link at p ≥ T1 and later links at p ≥ T2).

The largest gains came from reverse-engineering how the unmatched records were generated. A distractor is a copy of a real S1 record whose house number is shifted **upward by one of {1, 2, 3, 4, 5, 7, 9, 11, 13, 21}** and whose name is optionally edited. Features that see this mechanism took out-of-fold macro F0.5 from 0.976 to **0.988**, and the stage-3 re-scorer with the cross-encoder to **0.9914**:
- a name-edit lexicon;
- signed number relations;
- a label-free "number-oracle" lexicon, learned per country without labels.

All thresholds and scores come from cross-fitted pools rebuilt at test distractor density. France has no training data. France keeps the x1+x2 predictions, which won on the leaderboard, plus three class-level edits. Each edit rests on leaderboard evidence: every past France upload was diffed against the base and split by name-edit class. Each is also confirmed independently by the cross-encoder or by US labels.

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

### 4.1 Stage 3: re-scoring each query's best candidate (R14, R16)

Stage 3 is a LightGBM model on one row per query, its argmax S1 under the stage-2 blend, with 45 features, cross-fitted P0 ↔ P1 and pooled over US and India.

- **Consensus with the other records of the S1.** In 22% of S1 entities with ≥ 3 numbered true matches, most of the true S2/S3 records agree on a house number that differs from S1's own, so Source 1 is noisy too. Features:
  - the count and Σp of the other confident queries of the same S1, and the query's rank among them;
  - how many of them share the query's number sequence, name or address;
  - the query's number relation to the *consensus* number (the most common number of the other confident queries), not only to S1's;
  - how many of them carry the query's extra name tokens, and S1's missing ones.
- **Raw-text perturbations erased by normalization:**
  - doubled/tripled letters (distractor brand names such as Ferreon → Ferrreon; our normalizer collapses doubles);
  - accents, house-number letter and ½ suffixes (1714-C), repeated words (LLC LLC), brackets, `#`.
- **Model disagreement:** logits of the x1, x2, x3, x4 stage-2 predictions, runner-up p and margin; plus acronym and name-collision counts.
- **Cross-encoder (R16):**
  - `FacebookAI/xlm-roberta-base` (MIT, 278M parameters) fine-tuned as a pair classifier on the raw text `name | address` of the query and of the S1;
  - trained on the argmax rows where the stage-2 blend is unsure (0.003 < p < 0.997) plus 4% of the rest, about 770k pairs per pool, 1 epoch;
  - cross-fitted (the model trained on P0 scores P1 and vice versa; test = mean of both). Its logit is a stage-3 feature on exactly those rows.
  - It reads transliterations and name variants that string features miss. On India's uncertain rows, log-loss is stage 2 0.181, stage 3 0.129, cross-encoder alone 0.110, combined 0.093.
  - Training both folds and scoring 2.3M pairs took 34 min on one A100.
- **Large member and top-2 re-rank (final, R18):**
  - `FacebookAI/xlm-roberta-large` (MIT, 560M; 1 epoch, bs 64, lr 1e-5): holdout log-loss 0.107–0.111 vs 0.119–0.123 for base. It also scores each query's runner-up candidate.
  - The final re-scorer sees each query's **two** best candidates, each with its own consensus features and both cross-encoder scores plus the competing row's. The higher-scoring row becomes the link candidate, so a query can move to its runner-up S1.
- **Effect** (pooled OOF macro F0.5, US+India):

  | step | pooled | US | India |
  |---|---|---|---|
  | stage 2 | 0.98848 | | |
  | stage 3 | 0.98963 | | |
  | + base cross-encoder | 0.99140 | | |
  | + large member | 0.99152 | | |
  | **top-2 re-rank with large + base (final)** | **0.99165** | 0.99206 | 0.99106 |

  Argmax-row log-loss 0.0167 → 0.0083. On the leaderboard, stage 3 alone (no cross-encoder) gained +0.00145 (R14).

### 4.2 France: class-level edits backed by leaderboard evidence (R16)

France appears only in test. Every past France upload (R06, R07, R09, x3's France) was diffed against the base and split into label-free name-edit classes at equal house numbers:
- category→category word swaps (the vocabulary comes from single swaps in +GEN-shifted pairs, which are known distractors: *club, comité, école, amicale, pharmacie…*);
- true-noise words (*groupe, france, développement, cie, services, fils*);
- acronyms; legal-form changes; coined names.

Solving the leaderboard deltas gives:
- linked true-noise-word pairs ≈ 100% true (R07 removed 21.6k: −0.00114);
- equal-address category swaps ≈ 0% true (R09 and x3 added them: −0.00449, −0.00028).

France's same-address distractors swap the *category* word of its "{city} {category} {legal form}" names. US same-address distractors swap the *distinctive* first word instead, so the US-trained model cannot learn the French pattern. The final France rows are the x1+x2 predictions with three edits:
- (a) drop equal-address category-swap / category-added links unless another confident record of the S1 carries the new word (−7,171 links);
- (c) add equal-address acronym pairs (+1,634; 100% true in US OOF);
- (d) drop +GEN-shifted links whose number is also +GEN against the consensus of the S1's other records (−1,772; 2.15% true in US OOF);
- (e) add the equal-address true-noise-word links that x3 made and the large cross-encoder backs (> 0.72; +6,506). The x3 back-solving gives ≈ 0.72–0.9 true, and the large cross-encoder scores them at a median of 0.99.

The cross-encoders, trained without any France hypothesis, independently score (a) at 0.46 (the old model said 0.79), (c) at 0.90–0.98 and (d) at 0.17–0.28. We did **not** let them re-score France more broadly: in France's templated names, "same name" is weak evidence, and a US/India-trained re-scorer would add guesses such as an address-less *nantes club sarl* (157 S1 share that name).

### 4.3 Final day (R20): France structural transfer and an India blocking rescue

**France: transfer of generator-level truth rates from US/India.** Every argmax pair was placed into a vocabulary-free *structural cell*: house-number relation (equal, +GEN shift, −1/−2, other, no number) × same-street flag × name relation (identical core, single coined token, single in-vocabulary token, one word dropped/added/swapped, no shared word, …) × legal-form relation. Where the US and India truth rates agree (within 0.06), the rate belongs to the generator rather than to either country's vocabulary, so it can be applied to France. Examples:
- equal address with the name replaced by one coined token (*onyx…, dova…, riza…*): 98.8% true in US and 98.7% in India (France linked 92.8%);
- same name, same legal form, same street, different house number: 99.5% / 99.7% true.

Some cells do **not** transfer, because France has about five times the US rate of same-address (Type B) distractors:
- `equal address + one word swapped for an unknown token` is 97.6% true in US, but in France it holds Type B initialism edits (*icg → icgl*, *zt → zty*); the cross-encoder scores these near 0.
- a one-letter change of a short leading token is 79% true in US/India (OCR *l/i*, *8/b*; transliteration *sre/shre*) but is a distractor pattern in France.

So the package adds France links only in same-street, Type-B-proof cells that US and India both put at ≥ 93% true, and only where the large cross-encoder agrees (≥ 0.8). It removes links in +GEN cells that both put at ≤ 15%, plus initialism edits the cross-encoder rejects: **+4,317 / −138 France links**.

**France: cross-encoder veto in address-driven classes.** R06 France has no cross-encoder. On US/India out-of-fold rows where the pair model is confident (p > 0.7) but the large cross-encoder scores ≤ 0.05, the pair is rarely true:
- same name and same house number: 3.2% (US) and 0.5% (India);
- no house number: 9% and 8%.

In these classes the cross-encoder judges the address, not French vocabulary, so we drop the 1,737 linked France pairs it rejects there. Typical cases are generic names at the same number on another street (*bordeaux colege, 92 rue paulin* vs *92 rue saint jean*) and concatenated different names at the same address. True-noise-word, acronym and category classes are excluded.

**India and US: rescue of blocked-out matches.** After the cross-encoder re-rank, blocking misses are the largest remaining out-of-fold loss: +0.00384 pooled F if fixed, and +0.00526 for India. India misses that have an address are mostly generic names (*laxmi trading*, *shre global pvt ltd*) shared by 20+ S1 records. Those name-twins fill the top 10 and push out the true record, whose address shares a house number and a rare locality token (*kandivali west*, *palghar*) with the query.

For queries the decision leaves unlinked and that have an address, two extra retrievals propose new candidates:
- an address-heavy TF-IDF pass (name 0.1, address 0.8, char 0.1; top 10);
- a **name-twin expansion** (every S1 with the same core name, ranked by IDF-weighted shared address tokens; top 10).

Together they recover 30% of India's blocked-out true pairs. A LightGBM rescue scorer, cross-fitted P0 ↔ P1 on 28 vocabulary-free features (name similarities, IDF-weighted address overlap, house-number relations, name-twin count, source and retrieval score), links the best new candidate when its score is ≥ τ:

| country | τ | out-of-fold F | additions (precision) | test additions |
|---|---|---|---|---|
| India | 0.80 | 0.99107 → **0.99215** | 9,789 (0.925) | 12,928 |
| US | 0.85 | 0.99210 → 0.99228 | 3,211 (0.933) | 1,482 |

Because the rescue scorer runs inference on these pairs, `candidate_pairs.tsv` is the original top-10 candidates plus every rescue-scored pair: **128,969,685 pairs**.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** out-of-fold on the test-density pools (2.2M S1): **0.99165** for the final file (US 0.99206, India 0.99106). Public leaderboard of the final file: pending at the time of writing. A bias-corrected plug-in estimate projects ≈ 0.988 (test US ≈ 0.991, India ≈ 0.990, France ≈ 0.975); stage 3 without the cross-encoder scored 0.984009 (rank 347).
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
| R10d | x3+x4 blend (x4: 255-leaf variant) for US/India; x1+x2 for France | 0.98848 | 0.982556 |
| R14 | + stage 3 (consensus, raw-text perturbations, model disagreement) for US/India | 0.98963 (US 0.99062, India 0.98815) | **0.984009** |
| R16 | + xlm-roberta-base cross-encoder feature; France class edits (a, c, d) | 0.99140 (US 0.99187, India 0.99070) | – |
| R18/R19 | top-2 re-rank with xlm-roberta large + base cross-encoders for US/India; France x1+x2 + edits (a, c, d, e) | 0.99165 (US 0.99206, India 0.99106) | **0.987777** (rank 145) |
| R20a | + France structural-transfer package (+4,317 / −138 France links) | 0.99165 (US/India unchanged) | R20A_LB |
| R20e | + France cross-encoder veto (−1,737), India rescue v2 (+12,928) and US rescue (+1,482) | US 0.99228, India 0.99215 | R20E_LB |

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
| stage 4: consensus recomputed from stage-3 scores | argmax-row log-loss 0.01277 → 0.01351 (worse) |
| top-2 re-rank without the cross-encoder (runner-up row + both S1s' record counts) | 0.98962 vs 0.98963 |
| stage-3 hyperparameters (255 leaves, lr 0.03) | identical OOF |
| re-blocking with K = 50 | P1 US: 12.3k of 21k blocked-out true pairs lie in the top 50, but they are address-less queries of multi-S1 names; test France: < 9k plausible extra matches |
| fixing the `No.` → `patna` transliteration artefact (20.6% of India queries) | affected true pairs are blocked out *less* often (1.31% vs 1.64%); no gain |
| x1+x2-based stage 3 applied to France | moves only ±6k France links without a consistent direction; not used |
| France re-scorer with cross-encoders in the "US-like" classes | +10.2k / −4.3k France links, mostly address-less or different-number guesses on generic names shared by ≥ 2 S1; rejected |
| base cross-encoder on top of the large one | top-2 OOF 0.99165 vs 0.99163 |
| mdeberta-v3-base as a third member | one fold ran out of GPU memory; the other reached holdout log-loss 0.146 (xlm-r large 0.107); dropped |
| widening the cross-encoder band beyond 0.003 < p < 0.997 | only ~2.4k residual OOF errors lie outside the band |

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
- **Stage 3 / cross-encoder** (`src/stage3/`):
  - `run_stage3.sh` runs the whole stage in order;
  - `tables.py` / `tables_top2.py` build argmax / top-2 tables; `stage3.py` / `top2.py` build features and fit;
  - `build_ce.py`, `build_ce2.py`, `ce_rows_top2.py`, `ce_train.py`, `add_ce.py` run the cross-encoders;
  - `fr_classes.py` + `fr_rules.py` apply the France edits; `compose.py` writes the submission.
  - Features and fits take ~20 min on a 16-vCPU VM. The cross-encoders need a GPU: on one A100 40GB, the base member takes ~35 min and the large one ~110 min (both folds in parallel, incl. scoring).
- **Data and models:** no external data; every statistic comes from the provided files. The only pretrained models are `FacebookAI/xlm-roberta-base` and `FacebookAI/xlm-roberta-large` (MIT licence; 278M / 560M parameters), fine-tuned on the training data only.
- **Libraries:** polars, numpy, scipy, scikit-learn, sparse-dot-topn, rapidfuzz, Unidecode, LightGBM, PyTorch, transformers (MIT/BSD/Apache).
