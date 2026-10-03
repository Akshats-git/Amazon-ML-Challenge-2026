# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** SteinsGate  
**Team Members:** Akshat Gupta, Lakshay Gupta, Keshav Mishra, Akansh Tyagi  
**Institute:** Indian Institute of Technology Bhilai  
**Submission Date:** 28 September 2026

---

## 1. Executive Summary

We link each Source 2 and Source 3 record (a "query") to at most one Source 1 record (an "entity"). Candidates come from a per-country TF-IDF index. A two-stage LightGBM model scores each candidate pair. A third LightGBM re-ranks each query's two best candidates. It uses two fine-tuned multilingual cross-encoders (xlm-roberta base and large) that read the raw text. A rescue step finds new candidates for India and US queries that stay unlinked. A per-entity threshold rule makes the final links.

Our key finding is how the unmatched records were generated. Many are copies of a real Source 1 record with the house number shifted upward by one of a fixed set of values. Features built on this pattern gave our largest single gain. The final out-of-fold macro F0.5 is 0.99237 for US and 0.99262 for India. The final file scored **0.989507** on the public leaderboard.

---

## 2. Methodology

### 2.1 Problem Analysis

**Data.**
- Train covers US and India. Test covers US, India and France. France has no training labels.
- Test has 1,732,544 Source 1 records: 663,106 US, 809,986 India and 259,452 France.
- Test has 9,969,589 Source 2 and Source 3 records.

**Structure of the ground truth.**
- Each Source 2 or Source 3 record matches at most one Source 1 record. So each query can keep only its best candidate.
- No match crosses countries. So blocking runs inside each country.
- A Source 1 record has 1.67 Source 2 matches on average (at most 5) and 1.79 Source 3 matches (at most 6).
- 5.58% of Source 1 records have no match.

**Noise in true matches.**
- OCR swaps between letters and digits. Doubled or dropped letters.
- Legal form changes (LLC, Pvt Ltd, SARL). DBA aliases. Web handles and ID tags.
- Abbreviations and missing addresses. Many India names are in Devanagari or Bengali script.
- House numbers off by 1, 2, 10 or 20 in either direction. Dropped digits.

**How the unmatched records are made.** We found this by studying the train data at full scale.
- A generator copies a real Source 1 record.
- It shifts the house number **upward** by one value from GEN = {1, 2, 3, 4, 5, 7, 9, 11, 13, 21}.
- It may also append a word from a short list per country. It may swap a word or change the legal form. Example words: US "holdings" and "group". India "enterprises" and "exports". France "holding" and "participations".
- The street and city stay the same. So these records look almost identical to the original.
- Some distractors keep the whole address and change the business name. This type is rare in US and India but common in France.
- The sign of the number difference is the key signal. In US top-1 pairs a query number that is 1 or 2 **below** the Source 1 number is true 80% of the time. A query number that is **above** it by a GEN value is true only 2.5% of the time.

**Density.** Train has about 1.2 unmatched queries per Source 1 record. Test has about 2.3. A validation set at train density gives optimistic scores and poor thresholds.

**Source 1 is noisy too.** In 22% of entities with at least 3 numbered true matches, most true matches agree on a house number that differs from the Source 1 number.

### 2.2 Solution Strategy

**Approach Type:** Hybrid. Blocking, then gradient-boosted pair classifiers, then a cross-encoder re-ranker, then a per-entity decision rule.

**Core Innovation:**
- **Generator-aware features.** Signed house-number relations and a name-edit lexicon detect the distractor pattern. A label-free "number-oracle lexicon" learns each country's generator words without labels. It also works on France.
- **Validation at test density.** We split each country's Source 1 records into two halves by md5(entity_id) mod 2. Each half (pool P0 or P1) keeps its true matches and all unmatched queries. This gives about 2.4 unmatched queries per Source 1 record, close to test. Every model is cross-fitted: the model trained on P0 scores P1 and the reverse.
- **Consensus.** We compare a query with the other records that choose the same entity, not only with the Source 1 record.
- **Cross-encoders on raw text.** They read name variants and transliterations that string features miss.

---

## 3. Candidate Generation (Blocking)

**Normalization** is the same for every source and country:
- unidecode, lowercase, "&" to "and", OCR fixes inside alphanumeric tokens, doubled letters collapsed;
- legal forms made canonical and split off; DBA names split into both parts; house numbers extracted;
- transliteration dictionaries learned from train true pairs (694 name and 30 address entries). On held-out records they raise the share of non-Latin true pairs that share a name token from 0.31 to 0.999.

- **Blocking keys used:** one sparse TF-IDF index per country over the Source 1 records. It joins three weighted blocks:
  - name words (weight 0.25);
  - address word unigrams and bigrams (weight 0.50);
  - character 4-grams of the name with spaces removed (weight 0.25).

  Term frequency is sublinear. Tokens found in more than 10,000 records are dropped. Each query keeps its 10 most similar Source 1 records by cosine similarity (`sparse_dot_topn`).

  **Rescue retrieval for India and US.** Some queries stay unlinked after the models. For those with an address we run two more searches:
  - an address-heavy TF-IDF pass (name 0.1, address 0.8, characters 0.1; top 10 new pairs);
  - a name-twin search: all Source 1 records with the same core name, ranked by shared rare address tokens (top 10 new pairs).

- **Candidate pairs generated:** **128,969,685** pairs in `candidate_pairs.tsv`.
  - 99,695,099 come from the main blocking (10 per query): 38,170,203 US, 47,175,328 India and 14,349,568 France.
  - 29,274,586 are new pairs from the rescue retrieval that our rescue model scored.
  - Every final match is inside this set. A script checks this.

- **How we ensured true matches were not lost:**
  - The three blocks cover different errors. Character 4-grams survive typos and spacing. Address n-grams find records whose name changed. The dictionaries handle transliteration.
  - The frequency cap removes very common tokens and keeps rare ones.
  - We measured recall on the pools at test density. Recall at 10 is 0.991 for US and 0.984 for India. A perfect matcher on these candidates would reach macro F0.5 0.997 for US and 0.995 for India.
  - Most blocked-out true pairs have an empty query address and a name shared by several entities. Text alone cannot resolve them.
  - In pool P0, 13.6k blocked-out India true pairs have an address. The rescue retrieval finds 7,354 of them (54%).
  - We rejected an unpruned character 3-gram index. Its recall was no better. It would take about 60 hours on test.

---

## 4. Matching Model

**Features used.** Stage 1 has 80 pair features. None uses the country name.
- **Name features:**
  - rapidfuzz ratio, token sort, token set, partial ratio and Jaro-Winkler;
  - IDF-weighted soft Monge-Elkan in both directions and the IDF mass of unmatched tokens;
  - legal form equal or in conflict;
  - 14 edit features. For each word the query adds or drops we look up a smoothed log-odds of "not a true pair". This lexicon is cross-fitted and needs at least 50 queries per word. We also use the edit position (appended or prepended) and its label-free frequency. The lexicon is blanked for part of the training queries (50% in runs x3 and x4). So the model also learns from the label-free features.
- **Address features:**
  - ratio, token set, soft Monge-Elkan, shared bigrams and the largest shared IDF;
  - house numbers: exact match, first number match, soft match, smallest absolute difference and shared long numbers;
  - 16 number features. Numbers that one address has and the other lacks are compared as query minus Source 1. Flags mark a difference in GEN, in {-1, -2} and in minus GEN. Other features: signed closest difference, one-digit edit, transposition, digit length difference, street token overlap, "same number on the same street" and "one GEN shift on the same street".
  - Number-oracle lexicon. On rank-1 pairs "one GEN shift on the same street" marks likely distractors. "Same number on the same street" marks likely true pairs. The log-odds of each added or dropped name word between these two groups is a feature. It needs no labels. It is computed separately for each pool and each test country. On France it finds the French generator words (international, participations, distribution, holding).
- **Other:**
  - blocking context: cosine, rank, gap to the query's best score, number of candidates, how often the entity is some query's top 1, token frequencies;
  - query flags: alias, web handle, ID tag, non-Latin script, source, missing address;
  - stage 2 adds 11 features from out-of-fold stage-1 scores: the query's best and second score, margin and rank, and for each entity the count and sum of confident queries that choose it.

**Model type:**
- **Stages 1 and 2: LightGBM binary classifiers.** Stage 1 uses 127 leaves, learning rate 0.05, min child 100, feature and bagging fraction 0.8 and L2 1. Stage 2 uses 63 leaves. Early stopping uses a 10% query holdout. Runs x3 and x4 train on all queries of a pool (about 58.5 million pairs per model). US and India use the mean of x3 and x4 (x4 has 255 leaves). France uses the mean of runs x1 and x2. These runs have no number features. They scored better on France on the leaderboard.
- **Stage 3: top-2 re-rank (US and India).** A LightGBM on each query's best and runner-up candidates with 59 features. It adds:
  - consensus with the other records that choose the same entity: count, sum of scores, shared number, name or address, and the number relation to their most common house number;
  - raw-text marks that normalization removes: tripled letters, accents, house-number suffixes such as "1714-C", repeated words, brackets;
  - the scores of runs x1 to x4;
  - both cross-encoder scores for the row and for the competing row.

  The higher-scoring row becomes the link candidate. So a query can move to its runner-up entity.
- **Cross-encoders.** `FacebookAI/xlm-roberta-base` (278M parameters) and `FacebookAI/xlm-roberta-large` (560M parameters), both MIT license. Each is fine-tuned for one epoch as a pair classifier on raw "name | address" text of the query and the entity. Training rows are the uncertain best rows (0.003 < p < 0.997) plus 4% of the others: 781k rows in P0 and 769k in P1. They are cross-fitted. The test score is the mean of the two fold models. The large model also scores each query's runner-up candidate. Holdout log-loss is 0.107 to 0.111 for large and 0.119 to 0.123 for base.
- **Rescue scorer (India and US).** A LightGBM with 28 features scores the rescue candidates: name similarity, IDF-weighted address overlap, house-number relations, name-twin count and retrieval score. A third cross-encoder (xlm-roberta-large, 3 passes) is trained on the top 4 rescue candidates of each query. Its score and its rank, gap and margin within the query feed a final LightGBM. We link the best new candidate when its score reaches the tuned threshold (0.70 for India, 0.85 for US).

**Threshold selection method:**
- Each query links only to its best candidate.
- For each entity the first link needs p >= T1 and later links need p >= T2.
- We grid-search (T1, T2) for macro F0.5 on pooled out-of-fold scores at test density. The final values are (0.58, 0.78) for the US and India re-ranker and (0.52, 0.74) for France.
- At most 5 Source 2 and 6 Source 3 links per entity, as in train. The lowest scores are dropped first.
- Per-country thresholds and an expected-F rule gave no gain (at most +0.00002).

### 4.1 France

France has no labels. So we judged France changes by the public leaderboard. We compared past uploads that differ only in France and split the differences by name-edit class. France keeps the x1 and x2 decisions with these changes:
- (a) Drop same-address links where a category word is swapped or added (for example "club" and "comité") and no other record of the entity carries the new word. In France this is how same-address distractors are made. Removes 7,171 links.
- (c) Add same-address acronym pairs. Adds 1,634 links.
- (d) Drop links whose house number is a GEN shift from both the entity and its other records. Removes 1,772 links.
- (e) Add same-address links that differ only by a common French noise word (groupe, france, développement) when the large cross-encoder agrees (score above 0.72). Adds 6,506 links.
- **Precision removals.** A France stage-3 model is trained on the US and India pools. It uses the x1 and x2 scores. It leaves out the name-collision features because France has many templated names such as "nantes club sarl". We drop links that it rejects and the large cross-encoder also rejects (score below 0.3). Classes the leaderboard showed to be true are kept. With a cross-encoder address veto this removes 3,164 links.
- **Final additions.** We add unlinked pairs where the France stage-3 model and the large cross-encoder both score at least 0.95. Category, noise-word and generator-word classes and GEN shifts are excluded. So are pairs in the cells of an earlier France upload that scored lower. On US and India out-of-fold rows this condition is 99.4% true. This adds 2,371 links. It also removes 74 more links of the precision-removal pattern.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** out-of-fold at test density: **US 0.99237** and **India 0.99262** for the final pipeline. France has no labels. The final file scored **0.989507** on the public leaderboard.
- **Common false positives (wrong merges):**
  - Queries with no address matched by name alone when several entities share that name (for example "Wildlife Committee LLC" in New York and in Missouri).
  - Distractors whose word edit is also common true noise (for example "... Center" or "... Private Limited").
  - About 93% of false positives (stage 2) come from queries that have no true match at all.
  - In France: generic names at the same house number on another street or in another city.
- **Common false negatives (missed matches):** before the rescue step the out-of-fold misses split into three groups:
  - 90.2k true pairs were not in the candidates. Most have an empty address and a shared name. Others have a name fully in Devanagari or Bengali script or a made-up name with a short address fragment.
  - 45.1k had the right best candidate but a score below the threshold. Typical causes are OCR damage and house numbers off by one or two.
  - 46.0k chose another entity, often a same-name twin.

**Progress by run** (OOF = pooled out-of-fold macro F0.5 on US and India):

| run | change | OOF | public LB |
|---|---|---|---|
| R04 | normalization, blocking, 50 features, stage 1 | 0.97613 | not uploaded |
| R05 | + 14 edit features, stage 2, tuned thresholds | 0.98645 | not uploaded |
| R06 | twice the training data, blend of x1 and x2 | 0.98706 | 0.981021 |
| R10 | + 16 number features, all queries (x3) for US and India | 0.98838 | 0.982401 |
| R10d | mean of x3 and x4 for US and India | 0.98848 | 0.982556 |
| R14 | + stage 3 (consensus, raw text, run disagreement) | 0.98963 | 0.984009 |
| R16 | + base cross-encoder | 0.99140 | not uploaded |
| R19 | top-2 re-rank with both cross-encoders; France edits a, c, d, e | 0.99165 | 0.987777 |
| R20l | + India and US rescue with a rescue cross-encoder; France precision removals | US 0.99237, India 0.99262 | 0.989472 |
| **R20p (final)** | + France final additions | US 0.99237, India 0.99262 | **0.989507** |

**What did not help (measured):**

| idea | result |
|---|---|
| France lexicon imputation (R07) | LB -0.0011 |
| France normalization rules (R08) | LB lower (0.979 vs 0.979881) |
| France same-address rescue (R09) | LB -0.0045 |
| x3's own France predictions | LB -0.00028 |
| transfer of US and India truth rates to France by structural cell (R20a) | LB -0.00012 |
| bigger trees (255 or 511 leaves) in stage 1 | no gain in holdout log-loss |
| per-country thresholds, expected-F rule, third threshold | at most +0.00002 OOF |
| top-2 re-rank without cross-encoders | 0.98962 vs 0.98963 |
| stage 4 with consensus from stage-3 scores | worse log-loss (0.01351 vs 0.01277) |
| mdeberta-v3-base as a third cross-encoder | holdout log-loss 0.146 vs 0.107 for xlm-roberta-large |
| more blocking candidates (top 50, or top 20 in the address pass) | few new true pairs for millions of extra pairs |

---

## 6. Conclusion

Understanding the distractor generator gave the largest gain. The sign of the house-number shift and the added name words separate distractors from true matches. Validation pools at test density made our out-of-fold scores and thresholds transfer to test for US and India. Consensus with other records and cross-encoders on raw text gave the next gains. France had no labels. Most France rules without leaderboard evidence lowered the score. So we changed France only by classes that the leaderboard or both models supported.

---

## Appendix

### A. Code Artefacts

The complete code is in `code/business_entity_resolution/`. Its `README.md` gives every command.
- **Part 1 (CPU):** `src/run_all.sh` calls `src/run.py` for normalization, blocking, features and runs x1 to x4. The core modules are in `src/ber/`.
- **Part 2 (CPU and GPU):** `src/stage3/run_stage3.sh` runs stage 3, the cross-encoders, the France edits and the rescue. It then copies `matching_results.tsv` and `candidate_pairs.tsv` into `output/`.
- **Checks:** the official validator checks the matching file. A streaming check confirms that every match is in `candidate_pairs.tsv`.
- **Rerun check:** on 28 September 2026 we reran all CPU steps of part 2. The candidate file, the feature tables, the top-2 re-rank and all France steps came out exactly the same. The India and US rescue models are not bit-exact because their inputs differ by about 1e-5 between runs. The rerun reached the same out-of-fold gain and differed from the uploaded file in 0.02% of links. With the stored rescue outputs the steps rebuild the uploaded file byte for byte.
- **Compute:** a 6-core laptop with 13 GB RAM, rented CPU VMs (up to 32 vCPU and 128 GB RAM) and one rented A100 40 GB GPU.
- **Data and models:** no external data. The only pretrained models are `FacebookAI/xlm-roberta-base` and `FacebookAI/xlm-roberta-large` (MIT license, 278M and 560M parameters). All libraries are open source (polars, numpy, scipy, scikit-learn, sparse-dot-topn, rapidfuzz, Unidecode, LightGBM, PyTorch, transformers).

### B. Additional Results

**Blocking recall on the pools (top 10 per query, test density):**

| pool | Source 1 records | queries | pairs | recall at 1 | recall at 10 | oracle F0.5 |
|---|---|---|---|---|---|---|
| P0 US | 661,784 | 3,897,915 | 38,978,986 | 0.97577 | 0.99092 | 0.99717 |
| P1 US | 661,849 | 3,897,309 | 38,972,928 | 0.97579 | 0.99079 | 0.99717 |
| P0 India | 440,953 | 2,602,238 | 26,022,097 | 0.96799 | 0.98424 | 0.99483 |
| P1 India | 442,235 | 2,604,611 | 26,045,844 | 0.96805 | 0.98417 | 0.99481 |

**Stage 3 and cross-encoder effect** (pooled OOF macro F0.5, US and India):

| step | pooled | US | India |
|---|---|---|---|
| stage 2 (mean of x3 and x4) | 0.98848 | 0.98984 | 0.98646 |
| stage 3 | 0.98963 | 0.99062 | 0.98815 |
| + base cross-encoder | 0.99140 | 0.99187 | 0.99070 |
| top-2 re-rank with large and base cross-encoders | 0.99165 | 0.99206 | 0.99106 |
| + rescue with rescue cross-encoder | | 0.99237 | 0.99262 |

**Final file:** 5,838,881 links. US 2,249,303, India 2,732,705 and France 856,873. 5.8% of entities have no link. The train rate of entities with no match is 5.58%.
