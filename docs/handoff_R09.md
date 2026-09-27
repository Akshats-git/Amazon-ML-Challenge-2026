# Handoff R09: the paid run (JarvisLabs VM): France fix, generator-aware features, bigger model

Written Sat 26 Sep 19:30 IST. You are picking up in `/home/akshat/AmazonML2026`.

The user paid ₹1,000 for a JarvisLabs CPU VM so that this run counts. The goal is the best final (private-LB) score by the **code freeze, Sun 27 Sep 20:00 IST**. The official deadline is 23:59; don't rely on it.

Work autonomously and keep the user's involvement minimal. They upload files, report the LB score and rank, and approve destroying the VM.

**Work continuously (user instruction, 26 Sep 19:40).** Heavy jobs run on the rented VM, so **don't schedule anything "for overnight"**. Start each step the moment the previous one finishes: chain VM jobs in one script so there are no idle gaps, and code the next step while the current one runs. Pause the VM only when nothing is queued, e.g. while waiting for the user.

**Read first:**
- `CLAUDE.md`: the LOG.md rules (log every run and upload, and ask for the LB score and rank).
- `LOG.md` R06–R08c. **R08c is the research this plan is built on. Don't redo it.**
- `code/business_entity_resolution/README.md`.

---

## 0. Status at handoff

**Leaderboard**
- Best: **R06 `output_blend/matching_results.tsv` = 0.981021, rank 301.**
- Other uploads: R07 `output_blend_fr/` 0.979881, R08 `output_r08/` 0.979.
- Top 3 at 17:00: 0.990621 / 0.989475 / 0.988842.
- Limit: 5 uploads/day. The LB ranks the **best** upload. **2 uploads are left on Sat 26 Sep** and 5 on Sun 27 Sep.

**CV** (pooled OOF P0+P1, x1+x2 blend, T = (0.52, 0.74)): **0.98706** (US 0.9885, India 0.9849). France has no labels.

**Team:** **SteinsGate** (Lakshay Gupta, Akshat Gupta, Keshav Mishra, Akansh Tyagi, IIT Bhilai). Git: commit only when the user explicitly asks.

**Laptop**
- 6 cores, 13 GB RAM, Chrome-heavy.
- Keep analyses **under 3 GB**, and never load train and test norm tables in one process. A 7 GB script rebooted the laptop on 26 Sep; see memory `laptop-oom-concurrency`.
- About **3.3 GB of free disk**: download only what you need from the VM.

**Artefact state (important)**
- `work/norm/test_s*.parquet`: the France rows are **FR_NORM=1** (R08 rules). Train norm is unchanged.
- `work/feats/test/France_*`, `work/xfeats/test/France*` and `work/oof/{x1,x2}/p{1,2}_test_France.npy` are **R08 versions**. R06's France scores survive only as decisions in `output_blend/`. `work/bak_r07/` holds R07 France p2.
- `ber/normalize.py`: `FR_NORM` env, default 1 (R08 address **and** name rules).
- `ber/xfeats.py:build_xfeats` applies the R07 France lexicon imputation (`impute_lexicon`) **unconditionally**.
- **Both France changes lost on the LB.** Reproducing R06 France needs `FR_NORM=0` and no imputation (add a flag).
- `output/` holds only `candidate_pairs.tsv`, a hardlink of `output_r05/candidate_pairs.tsv`. That is the candidate file of every output so far; blocking is unchanged. **Never overwrite it or `output_blend/`.** `work/logs/final_package.sh` was fixed to `rm` before `cp`.
- `work/diag/`: reusable label-free checks:
  - `top1_cells.py <country> <matching_file>`, `samestreet_cells.py <country> <matching_file>` (CSV to `work/diag/out/`);
  - `oof_samestreet.py <country>`, `fr_sibling.py`, `fr_swaps.py`, `fr_eqother.py`, `diag_lb.py`, `compare_fr.py`.

## 1. What we know (R08c, verified on data; the numbers are in LOG.md and `work/logs/diag_*.log`)

- **Generator.** US and India have identical statistics, and France almost certainly does too.
  - Per S1: S2 matches mean 1.67 (**max 5**), S3 matches mean 1.79 (**max 6**); 5.58% singletons.
  - Every S2/S3 record matches ≤ 1 S1, and matches never cross countries.
- **Distractors** copy an S1 record and **shift its house number upward by one of GEN = {1, 2, 3, 4, 5, 7, 9, 11, 13, 21}**, all about equally likely. They may also change the name: append a word from a per-country list, swap in an industry word, or change the legal form.
  - France uses the **same shift set**. Its appended words are développement, participations, groupe, holding, international, distribution and france.
- **True-match number noise is symmetric:** ±1, ±2, ±10, ±20, dropped digits, digit edits.
  - Our features only see |diff|, never the sign or the shift set.
  - In US top-1 pairs, P(true | −1/−2) ≈ 0.80 and P(true | +GEN) ≈ 0.025.
  - **Equal house number + same street ⇒ 96–99% true, whatever the name edit** (OOF US/India, and the model already matches that there).
- **US/India OOF errors (argmax level, P0+P1):**
  - ~45% are an empty query address plus a name shared by 2+ S1s: mostly unresolvable. Don't chase them.
  - Number-mechanics buckets: ~7.8k FPs with a +GEN-aligned number, ~6k FNs aligned at −1/−2 or −3…−21, and 3.2k FN / 3.6k FP digit edits.
  - Proper number features are worth roughly **+0.0005 CV**.
- **France (≈ 0.962 back-solved from the LB) is a RECALL problem.**
  - Precision is excellent: +GEN pairs are linked at ≤ 1.6%.
  - Same street + identical house number, R06 link rates (US / India / France):

    | name relation | US | India | France |
    |---|---|---|---|
    | single word swapped | 93% | 97% | **68%** |
    | other edits, shared word | 99% | 99% | **77%** |
    | no shared word | 96% | 95% | **84%** |

  - The unlinked France edits are organisation-word swaps (club↔comité↔amicale↔école↔sportive…) and "& Fils" / "& Associés". **These are true noise, not sibling organisations:**
    - their exact names repeat across sources only 0.34% of the time (true variants: 0.78%);
    - S1 same-address sibling rate: unlinked 16.6% vs linked 15.0%;
    - pure generator words have ≈ 0 equal-number occurrences.
  - Expected France true matches ≈ 3.46/S1 ≈ 898k, against R06's 858.6k links: **≈ 40–48k true France links are missing.**
  - R07 lost because it removed ~20k of these (equal-number links). R08's `et`→`and` turned "& Fils" into the US "& Sons" distractor pattern.
- **Model choice.** Keep the **two-stage LightGBM**. The errors are structural (numbers, vocabulary), which GBDT with the right features handles best.
  - Research: Foursquare Location Matching winners used GBDT stages (+ XLM-R for 1st place; 7th place used no transformers). Other teams' public repos report ≤ 0.976.
  - **Do not spend time on transformers.** The VM has no GPU.

## 2. The plan (priority order)

### Step A: JarvisLabs VM (start first; upload and install run in the background while you code)

**Already done**
- CLI `~/.local/bin/jl` v0.2.17 is installed, and the user's token is configured (`jl status` → ₹1000.00).
- SSH key `~/.ssh/jarvislabs_ed25519` exists, **not yet registered**.

**Price** (`jl cpus`): 32 vCPU / 128 GB = **₹64.28/h** (IN1 and IN2), plus ~₹1.2/h for 100 GB storage. ₹1000 ≈ 15 h.

**1. Register the key and create the VM**
```bash
JL=~/.local/bin/jl
$JL ssh-key add ~/.ssh/jarvislabs_ed25519.pub --name steinsgate-laptop
$JL create --vm --cpu --vcpus 32 --ram 128 --storage 100 --region IN1 --name steinsgate-ber --yes --json   # save machine_id
$JL ssh <id> --print-command
```
The VM user is `cloud`. Add a `Host jlber` block to `~/.ssh/config` (append; don't touch the existing entries) with `IdentityFile ~/.ssh/jarvislabs_ed25519`. **The IP can change on resume.**

**2. Start the budget watchdog immediately** (a local background loop). At zero balance JarvisLabs pauses **and permanently deletes the data**. Every 10 minutes:
- run `jl status --json`;
- if the balance is below ₹120, run `jl pause <id> --yes` and tell the user to top up.

Also pause whenever the VM will sit idle for more than 30 minutes. The laptop must stay on while the VM runs, because the watchdog lives there, so ask the user to keep it plugged in with sleep disabled.

**3. Upload from the repo root** (~6 GB on the wire; measure the speed first with a 200 MB file):
```bash
rsync -az --partial --info=progress2 -e "ssh -F ~/.ssh/config" \
  --exclude .venv --exclude .ruff_cache --exclude 'output*' --exclude work/dev --exclude work/bak_r07 --exclude work/feats \
  ./ jlber:~/AmazonML2026/
```
For `work/feats` (7.5 GB): upload it if the link does ≥ 10 MB/s. Otherwise regenerate it on the VM with `run.py feats` (~20 min at 32 vCPU). **Test France features must match whichever France variant you score.**

**4. Install and smoke-test on the VM**
```bash
bash code/business_entity_resolution/setup_box.sh      # uv + Python 3.12 + pinned requirements -> .venv-ber
.venv-ber/bin/python code/business_entity_resolution/src/run.py env && ... check1 && ... check2
```
Then a reproducibility check: re-score one P1 feature file with the x2 models and compare against the uploaded `work/oof/x2` arrays (differences should be ~1e-6).

**VM defaults:** `N_JOBS=32`, `BLOCK_THREADS=8` (blocking is unchanged anyway), no `X16`.

### Step B: France quick win on the laptop (tonight: uploads #1 and maybe #2)

1. **Flags** (all env-driven, defaults set to the final choice later):
   - `FR_IMPUTE` (default **0**): gates `impute_lexicon` in `build_xfeats`.
   - Split `FR_NORM` into `FR_NORM_ADDR` / `FR_NORM_NAME` so address-only normalization can be tested.
   - `FR_RESCUE` (default 1).
2. **Before re-normalizing**, save the FR_NORM=1 France fields that the rescue rule needs: entity_id, addr_nums, addr_norm and name_core of France S1 and queries (small) → e.g. `work/norm/test_france_fr1.parquet`. The rule compares addresses with the R08 address rules (r→rue, regions/departments dropped), **whatever normalization the model features use**.
3. **Reproduce R06 France:**
   - `FR_NORM=0` → `run.py norm --splits test` (~75 s). US/India rows must stay byte-identical.
   - Rebuild test France features: `from ber.features import build_partition; build_partition("test", "France", "test", None)` (~2 min).
   - `run.py xfeats --parts test` with `FR_IMPUTE=0` (~3 min; the log must show **no** France "lexicon raised").
   - Re-score test France with x1 and x2, stages 1+2. Use the `rescore_france` snippet in `work/logs/chain_r08.sh` (~21 min).
   - Blend with `FR_RESCUE=0` into a scratch dir and check with `work/diag/compare_fr.py` that it **equals `output_blend/`** (≈ 0 diffs). That proves we reproduced R06.
4. **Rescue rule** (post-processing in `ber/submit.py`, before `decide()` in `blend_and_write` and `predict_and_write`).
   - Apply it to countries **absent from train**; don't hard-code France (the rules require country to be an open set).
   - Condition: on the query's **argmax** row, all three must hold:
     - the S1 has ≥ 1 number and the query's number multiset equals the S1's;
     - the Jaccard of alphabetic address tokens is ≥ 0.8;
     - no other S1 among the query's candidates has the same (numbers, street tokens) key (sibling guard).
   - Action: set `p = max(p, 0.90)`, so the (T1, T2) policy links it.
   - Also enforce the per-source caps (≤ 5 S2, ≤ 6 S3 per S1) in `decide`, keeping the highest p. This is tiny but free.
   - Expected: **+30–40k France links**.
   - Check with `samestreet_cells.py France <file>`: the "equal" rows should go to ≥ 0.93, and the "+GEN" rows must not move.
   - Check `oof_samestreet.py US|India`: the rule is not needed there, because the model is already calibrated in those cells.
5. **Upload #1**, e.g. `output_r09/`: R06 + rescue. Expected **+0.0015–0.003**. Log R09, ask for the score and rank.
   - Optional **upload #2** before 23:59: "R08c" = `FR_NORM_ADDR=1`, `FR_NORM_NAME=0`, `FR_IMPUTE=0`, plus rescue. It tests whether address normalization helps on top; it needs France norm, features, xfeats and a rescore (~30 min).
   - Before uploading, check the label-free proxy (`top1_cells.py France`): links on "+GEN" pairs must not rise.

### Step C: new country-agnostic features (code on the laptop, run on the VM)

Write them to a **new module and new files** (e.g. `ber/nfeats.py` → `work/nfeats/{part}/{country}_part-*.parquet`, aligned with the feature files). Gate them behind `USE_NFEATS=1`, so tags x1/x2 (64 features) still load and score unchanged. **Keep backward compatibility**: store each tag's feature list with its models, or select the list by flag.

**(a) Number relations**, from `addr_nums` (leading zeros already stripped) and the alphabetic tokens of `addr_norm`. Let q_only and s_only be the unmatched numbers.

| feature | definition |
|---|---|
| `nr_all_equal` | both sides have numbers and the multisets are equal |
| `nr_n_shared` | count of shared numbers |
| `nr_gen_pos` | # (q_only, s_only) pairs with q−s ∈ GEN |
| `nr_neg_small` | q−s ∈ {−1, −2} |
| `nr_neg_gen` | q−s ∈ −{3, 4, 5, 7, 9, 11, 13, 21} |
| `nr_signed_min` | signed q−s of the closest pair, clipped to ±10⁴ |
| `nr_digit_edit` | Levenshtein = 1 on the digit strings |
| `nr_transposed` | same digits permuted, a ≠ b |
| `nr_len_diff` | digit-length difference of the closest pair |
| `nr_street_jac` | Jaccard of alphabetic address tokens |
| `nr_eq_same_street` | `nr_all_equal` and `nr_street_jac` ≥ 0.8 |
| `nr_gen_same_street` | exactly one q_only and one s_only number, q−s ∈ GEN, and `nr_street_jac` ≥ 0.8 |

**(b) Number-oracle lexicon (NOL): label-free, per partition, identical code for P0, P1 and test.** It replaces the R07 imputation.
- On blocking rank-1 pairs, `nr_gen_same_street` marks the distractor proxy D and `nr_eq_same_street` the true proxy T.
- For each token w in the name-core extra set (query − S1) and missing set (S1 − query), count its occurrences in D and T.
- `nol(w) = log((cD+1)/N_D) − log((cT+1)/N_T)`, per kind. Require cD + cT ≥ 20, else NaN.
- Pair features: `nol_extra_max/min`, `nol_missing_max/min`.
- Expected values: in US/India it mirrors the supervised lexicon. In France it gives pure generator words ≈ +high, organisation words mildly positive, and fils / associés / services / cie negative.
- Consider `MASK_LEX_FRAC=0.5`, so the model learns to lean on NOL and the number features where the supervised lexicon is empty (France).

Unit-test on a few hand pairs, e.g. 823→844 is +21 (GEN); 3984→3982 is −2; 12↔21 is transposed and also +9. Time a whole-partition run (vectorize with polars lists; use rapidfuzz only for Levenshtein).

### Step D: train x3 on the VM (start the moment the nfeats files exist; no waiting)

1. **Quick A/B** (≤ 40 min): 5% of queries, stage 1 only, holdout logloss (like the "R05 prep" A/B in LOG.md).
   - Configs: {leaves 127 / 255 / 511} × {min_child_samples 50 / 100 / 200} × {feature_fraction 0.6 / 0.8} × {lambda_l2 0 / 5}. Pick about 5 of these combinations, not the full grid.
   - Use the same run to measure throughput and pick `TRAIN_QUERY_FRAC`, so that stage 1 + stage 2 + tune + test prediction fit in ~5–6 h. Aim for **0.5–0.6**. x2 used 0.30; 2× data gave +0.0006 before.
2. **Run:**
```bash
TAG=x3 USE_XFEATS=1 USE_NFEATS=1 TRAIN_QUERY_FRAC=<chosen> N_JOBS=32 run.py train1
... train2
... tune
WRITE_CANDIDATES=0 OUTPUT_DIR=output_x3 run.py predict
```
   Chain all four in **one** `nohup` script on the VM, with logs in `work/logs/*_r10.log` and a done-marker file at the end. Poll the marker over ssh. The moment it appears, evaluate, then queue the next VM job right away (blend variants, France variants, a second x3 seed, or more data if time allows). Pause the VM only if nothing is queued.
   - The x1/x2 test France arrays must be the version you intend to blend: R06-reproduction or R08c.
3. **Compare on pooled OOF:** x3 alone vs `blend --tags x2 x3` (and x1 x2 x3). **Gate: x3 must beat 0.98706 by ≥ 0.0003**, otherwise keep R06 + rescue. Check that the number buckets shrank.
4. **Download to the laptop** only the matching file(s) and small logs. Validate locally with the laptop's candidate file (`submit.validate(check_ids=True)` plus `check_candidates`): the VM has no candidate file.

### Step E: Sunday: evaluate, upload, choose (5 uploads)
- Candidates:
  - best OOF model (x3 or blend) + rescue;
  - the same with the other France normalization variant;
  - a rescue strictness variant only if needed.
- Before each upload, run the label-free France checks and `diag_lb.py`.
- **Final choice:** decide US/India questions by CV, and France questions by LB, because France has no labels. Differences under ~0.0003 on the public LB are noise.

### Step F: final package (by 17:00 Sun)
- **Code defaults must reproduce the chosen file:** FR flags, `FR_IMPUTE=0`, `FR_RESCUE=1`, `USE_NFEATS=1`, the x3 parameters and blend tags. Update `run_all.sh` and `README.md` (add the nfeats step and the x3 run).
- **`docs/Documentation_BER.md`:**
  - team name SteinsGate;
  - generator mechanics (the GEN shift set), NOL, the France rescue;
  - "what failed": R07 imputation, R08 name rules;
  - results table with an LB column (R06 0.981021 …).
- Run `CHOSEN=<dir> TEAM=SteinsGate bash work/logs/final_package.sh` on the laptop. Verify the zip contents, then ask the user to upload the zip and the final matching file.
- Ask the user before **destroying** the VM (`jl destroy <id>`), once the zip is verified.

## 3. Expected outcome
- France rescue ≈ +0.0015–0.003 LB; x3 ≈ +0.0005–0.001; target **≈ 0.983–0.985**.
- The R06 + rescue file is the safety net. Never let the final be worse than the best LB file.

## 4. Timeline (IST): back-to-back, no overnight waiting

| When | What |
|---|---|
| Sat 19:45–20:30 | VM up, upload and install in the background; Step B flags and rescue code |
| 20:30–21:45 | Reproduce R06 France + rescue → **upload #1** |
| In parallel from ~20:30 | Step C code + tests. As soon as they pass: nfeats on the VM → A/B → **launch x3 immediately** (≈ 4–5 h chained on the VM) |
| While x3 runs | Prepare the France variant (R08c) → upload #2 before 23:59; write the doc/README changes |
| x3 done (≈ Sun 03:00–04:00) | Evaluate at once and queue the next VM job (blend variants, a second model/seed, or more data). Pause only if nothing is queued |
| Sun 08:00–13:00 | Uploads #3–#6 (the new day's quota) |
| 13:00–17:00 | Reproducibility, docs, package, last upload |
| 17:00–20:00 | Buffer; freeze at 20:00 |

## 5. Ask the user
- After every upload: the LB score and rank. Log them straight away.
- To keep the laptop on and plugged in while the VM runs (the watchdog runs there).
- To top up JarvisLabs if the balance drops below ~₹150 before the run is done.
- Before destroying the VM.
