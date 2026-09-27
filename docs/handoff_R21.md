# Handoff R21: push the public LB above 0.990

You are taking over the Amazon ML Challenge 2026 entity-resolution run (Business Entity Resolution, macro F0.5 per S1). Before planning anything, run `date`. The deadline is **27 Sep 2026, 23:59 IST**. The last three sessions each misjudged the time, so check the clock, never estimate it.

Read in this order:
1. this file;
2. `CLAUDE.md` (the logging rules are mandatory);
3. `LOG.md`: the Scoreboard and entries **R20–R20p**;
4. `docs/handoff_R20.md` §4–5, which covers the generator facts and the negative results.

Don't repeat anything listed as negative below or in those files unless you have a new reason.

## 1. Goal and where we are

- **Goal:** public LB **> 0.990**.
  - Best so far: **0.989472, rank 96**, from `output_r20l/`.
  - Top team: 0.992082.
  - The gap is **+0.00053**.
- **Uploads:** ask the user how many are left and whether `output_r20p/` has been uploaded yet.
  - For every upload, tell the user the file, its purpose and the expected LB, then ask for the score and rank.
  - The LB keeps the best upload.
  - The final zip must hold the file with the highest LB.
- **Pending decision (r20p):** if `output_r20p` (r20l + 2,371 France adds − 74 removals) scores **≥ 0.989522**, run `bash work/logs/switch_to_r20p.sh <LB> <RANK>`. It updates the docs and rebuilds the zip. Otherwise keep the current zip.
- **Final zip:** `SteinsGate_submission.zip` (743 MB, validated) currently holds **r20l**.
  - Rebuild: `CHOSEN=<dir> CANDS=work/final_cands/candidate_pairs.tsv TEAM=SteinsGate bash work/logs/final_package.sh`, about 3 minutes.
  - It refuses unfilled placeholders in `docs/Documentation_BER.md` and the README.
  - Any new final needs its docs lines updated first.

## 2. The metric and what moves it

- **LB = 0.3827·F_US + 0.4675·F_India + 0.1498·F_France** (test S1 shares).
- Per-S1 F0.5 = 1.25·TP / (L + 0.25·K). An S1 with no true matches and no predicted links scores 1; 5.6% of S1s are singletons.
- **Value of one link change for an S1 that already has links:**
  - removing a false link: ≈ +0.21 S1-units;
  - losing a true link: ≈ −0.09;
  - adding a true link: ≈ +0.06;
  - adding a false link: ≈ −0.19.
  - Break-even is ≈ 0.70 true for removals and ≈ 0.75 for additions.
  - **Additions barely move the score.** Only removing false links, or US/India model gains, can deliver +0.0005.
- **US/India OOF gains have transferred to the LB at 1.0–1.5×:** R14 1.5×; the rescue ≥ 1.5×.
  - So +0.0005 LB needs about +0.0004 pooled US/India OOF (for example India +0.0006 and US +0.0003) at 1.5×.
  - That is the most reliable route, because it is measurable offline.

## 3. Pipeline state

- **US/India:**
  - normalizer → TF-IDF top-10 blocking → 2-stage LightGBM → stage 3 (consensus, raw text, xlm-r base+large cross-encoders) → top-2 re-rank `p8b`;
  - → **blocking rescue v4**:
    - two extra retrievals for unlinked queries that have an address;
    - a v2 LightGBM;
    - an **xlm-r-large "rescue CE" trained on the rescue candidates themselves**;
    - a LightGBM with the CE logit and its within-query rank/gap/margin.
  - OOF India 0.99107 → **0.99262**, US 0.99210 → **0.99237**.
- **France (no labels at all; train is US+India only):**
  - x1+x2 predictions + LB-backed class edits (a, c, d, e);
  - **R3 precision removals**: links a France stage-3 model (`pfr`) drops AND the large CE rejects (< 0.3), except LB-backed classes, plus the CE veto: −3,164 links.
  - The implied France F is ≈ 0.975–0.98. The top teams are probably ≈ 0.99 there.
- **Code:**
  - package `code/business_entity_resolution/`; step 8 of `src/stage3/run_stage3.sh` holds all R20 steps;
  - VM scratch `/home/ubuntu/r20/` (scripts, `rf_*`, `rce/`, `frg.parquet`, `fr_pfr_dec.parquet`, `out/v*/`);
  - stage-3 tables `/home/ubuntu/s3/`;
  - local session scratch copies under `/tmp/claude-1000/.../scratchpad/r20/`.

## 4. Verified facts (use them; don't re-derive)

- **Generator:**
  - Type A distractor: house number +GEN {1,2,3,4,5,7,9,11,13,21} plus a name edit.
  - Type B: same address. US replaces the first word; France swaps a category word.
- **True-link counts are country-independent:** 3.46 per S1, 5.58% singletons.
  - Distractors per source are equal in every split: test US 1.15, India 1.18, France 1.04 per source per S1.
- **All models are calibrated in aggregate.** Summed argmax p ≈ expected true count, on OOF and on test for all three countries. There is no count-based recalibration lever.
- **No leaks** in row order, ID value or adjacency (R20n).
- **France label-free evidence:**
  - Distractor edits in the +GEN band: appends 170k (TN 80k), category→category swaps 56.6k, category→TN swaps 5.6k.
  - At equal address: 38.0k category→category swaps (≈ all distractors, already unlinked) and 39.3k category→TN swaps. That count implies ≈ 90% of the category→TN swaps are **true**, so don't remove them (`output_r20k` was dropped for this reason).
  - Both US/India-trained models (`pfr`, large CE) wrongly accept France category swaps. Trust them in France only on structure (address/number), never on name-swap vocabulary.
  - `pfr` on US/India OOF:
    - where x1+x2 ≥ 0.8 and p3 < 0.2, 2.1% true;
    - where x1+x2 < 0.5 and p3 ≥ 0.95, 99.4% true.
- **France additions have lost on the LB every time** (R09 −0.0045, x3 France −0.00028, R20a −0.00012). Removals backed by two models plus a structural reason worked (R20l).

## 5. Negative results from R20 (don't repeat)

- France structural cell transfer (adds);
- France blocking rescue (candidates are France's own distractors);
- rescue with relative features only (no CE);
- K=20 or translit retrieval;
- same-source twin priors;
- FP switching;
- count recalibration;
- more one-word-swap removals (the remaining ones are abbreviations, i.e. true noise);
- address-less twins (already unlinked);
- TN-swap removals (r20k).

## 6. Ideas ranked by expected LB gain per hour

Write the estimated recoverable F before building anything. Each OOF-measurable idea must show its OOF delta before you compose a file.

1. **Listwise cross-encoder for the main US/India candidates.** This targets the "wrong S1 won" (+0.00206 if fixed) and "argmax right, below threshold" (+0.00185) buckets.
   - Why it might work: the rescue showed that a CE trained on a query's *competing* candidates, with within-query `ce_gap`/`ce_mg` features, cut log-loss 30–40% where string features had saturated. `p8b`'s large CE was trained only on argmax rows and scores runner-up rows. It was never trained on the competition itself.
   - Plan:
     - For uncertain queries (argmax p5 in 0.02–0.98, US+India, each pool), take the top 4 candidates by stage-2 p. Build raw "name | address" text as in `r20_build_rce.py`.
     - Train xlm-r-large cross-fitted P0 ↔ P1 with `r20_rce_train.py` (`CE_REP` 2–3).
     - Add the CE logit + rank/gap/margin to the top-2 re-rank table. Refit `p8b` (see `run_stage3.sh`, the `top2.py` / `tables_top2.py` steps), re-tune T1/T2 on OOF, then recompose (rescue adds on top).
   - Cost:
     - measure the row count first; ≈ 150k queries × 4 per pool is plausible;
     - large CE ≈ 135 samples/s per fold on one A100, so 1 pass of 600k ≈ 75 min;
     - two A100s run both folds in parallel;
     - base xlm-r is ≈ 3× faster.
     - Total **2–3 h**.
   - Expected: +0.0002–0.0008 OOF, i.e. +0.0003–0.0012 LB. This is the only listed idea that can plausibly clear 0.990 on its own.
2. **Rescue CE round 2 (cheap, small).**
   - Widen the rescue CE set: `RMIN=0.005 TOPK=6` in `r20_build_rce.py` (12% of rescue positives are outside today's set), and add a second seed. Then rerun `r20_rescue4.py`.
   - Cost ≈ 70 min. Expected +0.0001–0.0002 LB.
3. **France-specific CE from pseudo-labels.** Labels:
   - positives: France links with x1+x2 p > 0.99 in LB-backed true classes;
   - negatives: +GEN pairs and equal-address category→category swaps.
   - Use it only to *remove* links where it strongly disagrees, and only in structural classes.
   - Cost ≈ 2 h; it cannot be validated except on the LB.
4. Anything else you find by reading raw examples of the remaining OOF errors (`decomp8.py` on the latest decision).

If time can't fit ideas 1 or 2 plus about 15 min for compose, validation and zip, say so plainly. Don't spend an upload on an unmeasured gamble unless the user asks.

## 7. Operations

- **JarvisLabs:** CLI `~/.local/bin/jl` (`jl list`, `jl resume <id> --yes`, `jl pause <id> --yes`, `jl status`).
  - Balance ≈ ₹460. The user wants it spent if that helps.
  - **Zero balance deletes instance data.**
  - Costs: A100 ≈ ₹84/h; 32-vCPU VM ≈ ₹64/h.
  - You may create a second A100 for parallel folds.
  - Ask the user before destroying any instance.
- **Instance ids and IPs change on every resume.** CPU VM (last id `518811`, paused), A100 `518671` (paused).
  - After a resume, update `~/.ssh/config`: HostName of `jlber` (user `ubuntu`) or `jlgpu` (user `root`).
  - Clear the stale key: `ssh-keygen -f ~/.ssh/known_hosts_jlber -R <ip>`.
  - Restart the laptop watchdog: `cd work/vm && ID=<new id> MIN_BAL=60 nohup setsid bash watchdog.sh &`.
  - Kill the old watchdog **by exact PID only**; `pkill -f` has killed our own shell before.
- **VM Python:** `~/AmazonML2026/.venv-ber/bin/python`.
  - VM scripts `chdir` to the repo root, so pass **absolute paths**.
  - Watch LightGBM thread oversubscription when running two jobs at once.
- **A100:** it had no data and no polars after resume (`pip install polars`). HF models are cached. The CE checkpoints are **not saved**; scoring new rows means retraining.
- **Laptop:** 13 GB RAM with Chrome running; keep analyses under 3 GB. The laptop venv is `.venv/bin/python`; system python3 has no polars.
- **Disk:** ≈ 11 GB free. A root Docker pull filled the disk once; check `df -h /` before large writes.
- **Never:**
  - overwrite `output_s8b/`, `output_blend/`, `output_s3/` or `output/candidate_pairs.tsv` in place (hardlinks);
  - use the default AWS profile (use `amlc` if AWS is needed);
  - commit unless the user asks.
- **Composition tools** (in `/home/ubuntu/r20/`; paths absolute):
  - `patch_fr.py BASE FR_PAIRS OUT` (replace France rows);
  - `patch_add.py BASE ADD OUT` (add links for unlinked queries);
  - `candchk2.py MATCH CANDS` (links ⊆ candidates; the candidate file is `out/cands_final.tsv` = `work/final_cands/candidate_pairs.tsv`, sha `c4f0dda6…`).
  - Any new pair outside the 128,969,685 candidates means regenerating the candidate file (`r20_regen_cands.py`).
  - Validator: `python3 scripts/validate_submission.py --matching <file> --candidate <dir>/_none --test-dir data/raw/test`.
- **Logging (mandatory):** after every run or upload, add a LOG.md entry, a Scoreboard row and an `experiments.csv` row, with honest numbers and failed runs included.
  - Update the Scoreboard as soon as the user reports a score.
  - Port anything that goes into the final into the package and `run_stage3.sh`, and update the README/Documentation.
