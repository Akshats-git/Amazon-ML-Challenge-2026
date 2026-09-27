# Handoff: raise the LB score (Amazon ML Challenge 2026, Business Entity Resolution)

You are picking up work in `/home/akshat/AmazonML2026`. First read these:
- `CLAUDE.md` (the logging rules);
- `LOG.md` entries R04 to R07b (R07b explains everything below);
- `code/business_entity_resolution/README.md`.

The pipeline is built and submitted. Your job:
- raise the leaderboard score before the deadline, **mainly by fixing France**;
- ship the final package.

Written 26 Sep 10:50 IST.

## Rules and constraints
- **Timeline:**
  - The deadline is **Sun 27 Sep 23:59 IST**; our code freeze is **Sun 20:00**.
  - The top-500 cutoff (Sun 00:00) is already met at rank 146. But if the portal ranks by the *latest* upload rather than the *best*, upload only files you expect to beat 0.979881 before then. Ask the user which it is.
- **Machine:** laptop only. It has 6 cores / 12 threads, 13 GB RAM (Chrome takes a lot) and **~2.3 GB free disk**. SageMaker quota is 0.
- **AWS:** never use the default AWS CLI profile (it belongs to HireLens). Use profile `amlc` if AWS is ever needed.
- **Logging:** after every run or upload, update `LOG.md` and `experiments.csv`:
  - add an entry `R<NN> · name · time IST` with Approach, Data, Results (all metrics), LB score **and rank**, Takeaways and Next;
  - add a Scoreboard row and update "Current best on LB";
  - add an `experiments.csv` row.

  Ask the user for the LB score and rank after each upload.
- **How to run steps:** `N_JOBS=8 PYTHONUNBUFFERED=1 USE_XFEATS=1 .venv/bin/python code/business_entity_resolution/src/run.py <cmd>`.
  - `USE_XFEATS=1` is required for the x1/x2 models.
  - Run in the background with logs in `work/logs/`.
  - Don't run side analyses above ~1.5 GB while a pipeline step runs; two OOM kills happened on 26 Sep. The `work/diag` scripts peak at ~2.4 GB, so run them only between steps.
- **Protect these files:**
  - Never overwrite `output_blend_fr/`, the current LB best.
  - Never overwrite `output_r05/candidate_pairs.tsv`, the candidate file for every current output. It is hardlinked as `output/candidate_pairs.tsv`.
  - New outputs go in new directories (`output_r08/`, …).
- **Git:** don't commit unless the user asks.

## Current state
- **Pipeline:**
  1. normalize (`ber/normalize.py`);
  2. sparse TF-IDF blocking, top-10 per query (`ber/blocking.py`);
  3. 50 pair features (`ber/features.py`) plus 14 generator-aware xfeats (`ber/xfeats.py`);
  4. stage-1 LightGBM, cross-fit P0↔P1;
  5. stage-2 LightGBM on the p1 context (`ber/context.py`);
  6. argmax per query, then a per-S1 policy: first link if p ≥ T1, later links if p ≥ T2 (`ber/decide.py`).
- **Final = R07:** the mean of run x1 (15% of queries) and run x2 (30%) stage-2 scores, (T1, T2) = (0.52, 0.74), plus the R07 France lexicon imputation (test-only).
- **CV:** pooled OOF over P0/P1 (md5 halves of train S1 plus all distractors) is **0.98706** (US 0.98848, India 0.98490). France has no train labels, so CV cannot see it.
- **LB:** **0.979881 public, rank 146**, from `output_blend_fr/matching_results.tsv`. Gap −0.0072.
- **Test size:**

  | country | S1 | queries | pairs |
  |---|---|---|---|
  | France | 259,452 | 1.43M | 14.3M |
  | India | 809,986 | 4.72M | 47.2M |
  | US | 663,106 | 3.82M | 38.2M |

## Diagnosis (R07b, label-free): why the LB is below CV
The table comes from `work/diag/diag_lb.py`, using the blend and T = (0.52, 0.74). "Self-est. F" is a plug-in expected F that treats p as calibrated. On CV it runs 0.005–0.007 above the actual F.

| partition | links/S1 | uncertain queries/S1 (p 0.1–0.9) | self-est. F | actual F |
|---|---|---|---|---|
| CV US / India | 3.37 / 3.36 | 0.12 / 0.14 | 0.993 / 0.992 | 0.9885 / 0.9849 |
| test US / India | 3.39 / 3.35 | 0.17 / 0.18 | 0.992 / 0.990 | – |
| **test France** | 3.23 | **0.39** | **0.982** | – |

- **Thresholds are fine.** Test has ~1.5× more hard distractors per S1 than the CV pools, because half of each pool's distractors were generated from S1s in the other pool. Simulated on OOF (`work/diag/sim_density.py`), that costs only 0.0007, and re-tuning (T1, T2) gains +0.00004. Leave the thresholds and policy alone.
- **US and India on test look like CV:** self-est. F is only −0.0015 lower.
- **France is about 0.95.** Back-solving the LB with US ≈ 0.986 and India ≈ 0.983 gives that figure.
  - Each +0.01 of France F adds ≈ +0.0015 on the LB.
  - France at US level would add ≈ +0.005.
- **France's problem is format, not distractors.**
  - France distractors shift house numbers exactly like US/India's (shifted-number share of top-1 pairs: France 0.272, India 0.271, US 0.297), and the model already rejects them.
  - What differs is how S2/S3 write French records compared with S1 (`work/logs/diag_fr.log`):

    | issue | S1 | S2/S3 |
    |---|---|---|
    | `r` for `rue` | 0% | **25%** (S1 `rue` 66%, S2/S3 38%) |
    | bd/av/pl/ch/imp/rte/al | 2.5% | 10% |
    | `st`/`ste` for saint(e) | 0.1% | 1.8% |
    | spaced legal forms (`s a`, `s n c`, `s a r l`) | 0% | 5.2% |
    | `et` in names (S1 writes `and`) | 0% | 1% |

  - **Address endings:**
    - S1 ends addresses with the **region**: `hauts de france` 88k, `nouvele aquitaine` 74k, `pays de la loire` 63k (of 259k).
    - S2/S3 end with a **department** (`gironde` 65k, `nord` 65k, `loire atlantique` 55k, `pas de calais` 12k in S2), the region, or just the city.
  - A sample of uncertain France pairs shows the same things, e.g. `29 rue louis braile bordeaux nouvele aquitaine` vs `29 r louis braile bordeaux gironde`.
  - Leading zeros are already stripped in `addr_nums`, so number features are fine.

## Plan, in priority order

### 1. R08: France-only normalization (highest value, no retraining, ~2 h)
**Goal:** make French query records look like their S1 so the US/India-trained models see familiar similarity levels.

Only France rows change. The pools, models, OOF, thresholds and the candidate file all stay valid.

1. **Add France rules to `ber/normalize.py`.**
   - Pass `country` through `normalize_frame` → `_chunk` → `norm_record`; the frame has a `country` column.
   - Gate the rules on `country == "France"` and a config flag `FR_NORM` (env, default 1), so `FR_NORM=0` reproduces R07.
   - **Name rules** (apply to the token list before `_name_post`, so legal canonicalization, `name_core` and `name_sq` see the result; apply them to alias parts too):
     - join maximal runs of ≥ 2 single-letter tokens (`s a r l` → `sarl`, `s n c` → `snc`, `s a` → `sa`);
     - map `et` → `and`.
   - **Address rules** (apply to the tokens from `addr_tokens_pre`, before the `n a` drop and before `addr_nums`):
     - Remove region/department phrases. Remove multi-token phrases anywhere: `hauts de france`, `nouvele aquitaine`, `pays de la loire`, `loire atlantique`, `pas de calais`. Remove single-token departments (`gironde`, `nord`) only as the last token.
     - Expand abbreviations to **the exact token S1 uses after normalization**. Doubled letters are collapsed, so allée → `ale` and impasse → `impase`. Examples: `r`→`rue`, `av`→`avenue`, `bd`→`boulevard`, `pl`→`place`, `ch`/`chem`→`chemin`, `imp`→`impase`, `rte`→`route`, `al`→`ale`, `st`→`saint`, `ste`→`sainte`.
   - **Confirm and extend both lists from data first:**
     - count trailing 1–3 tokens per source (as in `work/diag/diag_fr.py`) and check what remains after removal;
     - count (query token, S1 token) pairs over confident France top-1 pairs (blend p ≥ 0.9) where exactly one address token differs on each side;
     - keep mappings with high count and purity (the same idea as `ber/translit.py:_select`).
2. **Back up, then re-normalize the test split.**
   - Back up the R07 France predictions (~0.25 GB): `mkdir -p work/bak_r07 && for t in x1 x2; do cp work/oof/$t/p2_test_France.npy work/bak_r07/${t}_p2_test_France.npy; done`.
   - Record a hash of the US/India rows of the three test norm files (`io.norm_path("test", s)`).
   - Run `run.py norm --splits test`.
   - Verify that the US/India rows are byte-identical and that the row order is unchanged. Row indices are used everywhere.
   - Never re-normalize train; it has no France rows.
3. **Rebuild test France features on the existing candidate pairs** (no re-blocking, so `candidate_pairs.tsv` stays valid). Run `from ber.features import build_partition; build_partition("test", "France", "test", None)`. It takes ~4 min and rewrites the files in place.
4. **Rebuild xfeats.** First `rm work/xfeats/freq_test_France.parquet`: it's a stale cache of France name edits. Then run `run.py xfeats --parts test` (~3 min). In the log, check that the R07 France words are still raised to 9.16.
5. **Re-score test France** with x1 and x2, stage 1 then stage 2. Use the `rescore_france` snippet in `work/logs/chain_r07.sh` (~21 min).
6. **Blend into `output_r08/`.** Run `mkdir -p output_r08 && ln output_r05/candidate_pairs.tsv output_r08/candidate_pairs.tsv`, then `OUTPUT_DIR=output_r08 WRITE_CANDIDATES=0 run.py blend --tags x1 x2 --check-ids`. `validate()` then also checks matches ⊆ candidates.
7. **Check without labels before uploading.**
   - Run `.venv/bin/python work/diag/diag_lb.py`:
     - France uncertain queries/S1 should drop clearly below 0.389 (US/India test: 0.17–0.18);
     - France self-est. F should rise from 0.9823 towards ~0.99;
     - the US/India rows must be unchanged.
   - Diff links against R07 per country: `.venv/bin/python work/diag/compare_fr.py output_blend_fr/matching_results.tsv output_r08/matching_results.tsv`.
   - If uncertainty did not fall, investigate before spending an upload.
   - Otherwise ask the user to upload `output_r08/matching_results.tsv`.
8. **Log R08** in LOG.md and experiments.csv. Add the France rules to `docs/Documentation_BER.md` (normalization section, error analysis, results table). If the LB drops, go back to `output_blend_fr/`, find which rule hurt, and remember that each probe costs an upload.

### 2. France iteration 2 (if R08 helps or is neutral)
- **Look for the next issue.** Sample ~30 uncertain France test pairs (p 0.2–0.8) with their normalized text and find the next systematic difference. Candidates:
  - French articles (`du`/`de`/`la`/`le`/`l`/`d`) dropped or added in names;
  - `bis`/`ter` in addresses;
  - elisions (`l'énergie` → `lenergie`).
- **Iterate cheaply:** fix → rebuild France features → re-score x2 only → `diag_lb.py`. Repeat while France uncertainty keeps falling. One cycle takes ~20–25 min.
- **Optional: re-block test France** with the new normalization. It raises `blk_score`/`blk_rank`, the top features, for `r` queries. Steps:
  1. block ("test", "France") only, with a small snippet around `ber/blocking.py:block_one`;
  2. rebuild features and xfeats, then re-score;
  3. blend with `WRITE_CANDIDATES=1` into a new directory, which regenerates the full candidate file.

  The new candidate file needs 1.3 GB of disk. Keep the old candidate file until the final choice is made, because `output_blend_fr` and `output_r08` depend on it. `work/logs/final_package.sh` must then link the new candidate file.

### 3. France thresholds via the LB (only if the user has spare uploads)
- Add a per-country override to `ber/submit.py:blend_and_write`, e.g. env `T_FRANCE=t1,t2`.
- Probe France-only shifts, e.g. (0.42, 0.66) and (0.62, 0.82), and keep whichever the LB prefers. OOF cannot judge France.

### 4. US/India model upgrade (overnight, measurable on CV, small gain: ≤ +0.0005 CV)
- **Parameter A/B.** Tune LightGBM params (num_leaves, min_child_samples, feature_fraction, lambda_l2) on a 2% query sample by holdout logloss. Follow the "R05 prep" A/B in LOG.md: 1.3M rows, 10% query holdout, 150 rounds.
- **Bigger run.** Train a new tag `x3` with the best params and `TRAIN_QUERY_FRAC=0.40 X16=1`:
  - timing: train1 ~1.5 h, train2 ~1 h, test predict ~1.5 h;
  - before starting, check R06's peak RSS in `work/logs/*r06*.log`.
- **Disk.** A new tag needs ~1.8 GB of arrays. Delete x1's arrays first and blend x2 with x3; x1 adds only +0.00001. Test France for x3 automatically uses the R08 features.

### 5. Final package (Sun, before 20:00)
- Ask for the **team name**.
- Choose the final file by public LB, with a CV sanity check.
- Run `CHOSEN=<dir> TEAM=<name> bash work/logs/final_package.sh`, after updating the script if the candidate file changed.
- Make sure `run_all.sh` and the README reproduce the chosen output. The France rules live in `normalize.py`; the imputation is in `xfeats.py`.
- Fill the doc header (team name) and the LB column.

### Not worth the time
- More threshold or policy work: it's saturated; the per-country, expected-F and three-threshold rules all gave ≈ 0.
- Density-matched training: ≤ 0.0007 is at stake.
- More blocking recall for US/India: the misses are empty addresses, same-name businesses in other states, and Devanagari names.
- More seeds or blends: +0.00001.
- Lower priority, LB-only: extending the India translit dict from confident test pairs (test non-Latin coverage is 0.955), and France pseudo-label self-training (risk of confirmation bias).

## Useful files
- **Diagnostics** (from the previous session, in `work/diag/`):
  - `diag_lb.py`: per-partition label-free profile, ~35 s;
  - `diag_fr.py`: edit signatures and France format stats;
  - `diag_nums.py`: shifted-number links;
  - `sim_density.py`: density simulation, ~9 min;
  - `compare_fr.py`: diff two matching files per country.
- **Logs:** `work/logs/` (`*_r0*.log`, `diag_*.log`, `sim_density.log`) and the chain scripts `work/logs/chain_*.sh`, which show example command sequences.
- **Artifacts in `work/`:**
  - models: `models/` (x1/x2, stages 1–2, pools P0/P1);
  - predictions: `oof/{x1,x2}/p2_*.npy`;
  - features: `feats/{P0,P1,test}/<country>_part-*.parquet`;
  - xfeats: `xfeats/`;
  - blocking: `cands/`;
  - normalized tables: `norm/`.
- **Deletable if disk runs short:** `work/dev` (118 MB), `output_tier0/` and `output_r06/` (93 MB each).

## Ask the user
1. How many uploads per day, how many are left today, and does the LB rank the best or the latest upload?
2. The team name, for the zip and the doc. The members are Lakshay Gupta, Akshat Gupta, Keshav Mishra and Akansh Tyagi (IIT Bhilai).
3. Is it OK to commit the code changes?
