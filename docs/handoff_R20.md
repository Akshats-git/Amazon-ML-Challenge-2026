# Handoff R20: from 0.987777 (rank 145) to > 0.9918 (top 10)

Written Sun 27 Sep 16:50 IST and updated 17:15 after the `output_s8b` upload. This is the final day. You are picking up in `/home/akshat/AmazonML2026`.

## 0. The job and the upload rule

- **Where we are:**
  - Best LB: **0.987777, rank 145**, from `output_s8b`, uploaded at about 17:05.
  - Target: **> 0.9918**, which the user expects to reach the top 10. The user said "at any cost", so compute spend is fine.
  - Deadline: **23:59 IST**, about 6.5 hours from 17:15.
- **Uploads:** **3 left.** The user does the uploading; you tell them what to upload and when.
  - The public LB keeps each team's best upload, so a lower-scoring upload can't cost rank.
  - You may use up to two uploads before the final one, either as a **step-up** (a candidate projected clearly above 0.987777) or as a **designed probe** whose answer is worth more than an upload (e.g. the true effect of one France edit).
  - Make the **last upload the best candidate**, at about 23:00–23:15 to leave margin, in case the final ranking uses the last submission.
  - For every upload, tell the user the file, its purpose and the expected LB. Log the result in LOG.md, the Scoreboard and `experiments.csv`.

Work in four steps:

1. **Deep brainstorm.** Before building anything, list every mechanism that could still cost F on test, France first, then US/India. Give each one a size estimate and a data test.
2. **Deep research.** Test each hypothesis on data:
   - US/India: use the labels.
   - France: use LB evidence (including your probes) plus label-free structure.
   - Read raw examples, not just aggregates.
   - Use web search where outside knowledge helps: French business-name conventions, entity resolution, and cross-encoder training.
3. **Build.** Implement the fixes with the largest *measured* expected gain. Run them back-to-back on the paid machines, in parallel where possible.
4. **Report.** Give the user honest numbers. No excuses and no "not possible" framing, but never overclaim.

## 1. The target in numbers

Test S1 shares (from `data/raw/test/test_source1.tsv`): US **38.27%**, India **46.75%**, France **14.98%**.

**LB = 0.3827·F_US + 0.4675·F_India + 0.1498·F_France.**

| scenario | US/India weighted mean F | F_France | LB |
|---|---|---|---|
| `output_s8b`, if the US/India projections hold (US 0.9911, India 0.9902) | 0.9906 | ≈ 0.972 | **0.987777** (actual) |
| `output_s8b`, other reading: the France edits worked as estimated | ≈ 0.9897 | ≈ 0.977 | 0.987777 (actual) |
| target with France 0.990 | 0.9921 | 0.990 | 0.9918 |
| target with France 0.985 | 0.9930 | 0.985 | 0.9918 |
| target with France unchanged | 0.9953 | 0.972 | 0.9918 |

- **Sensitivities:**
  - +0.01 France F is worth +0.0015 LB.
  - +0.001 on the US/India mean is worth +0.00085 LB.
- **Gap: +0.0040.** That equals today's whole gain from R14 to `s8b` (+0.0038). It needs France at about 0.985–0.990 **and** US/India at about 0.992–0.993.
- **Biggest lever:** France. It sits about 0.02 below US/India.

## 2. State at handoff

**LB history** (public LB; the leaderboard keeps your best upload):

| upload | content | LB |
|---|---|---|
| R06 `output_blend/` | x1+x2 for all countries; its France rows are the base of every later France | 0.981021 |
| R07 | France lexicon imputation | 0.979881 |
| R08 | France normalization | 0.979 |
| R09 | France same-address rescue | 0.976531 |
| R10h `output_r10h/` | x3 US/India + R06 France | 0.982401 |
| R10x3 `output_x3/` | x3 for France too | 0.982123 |
| R10d `output_hyb34/` | x3+x4 US/India + R06 France | 0.982556 |
| R14 `output_s3/` | stage 3 US/India + R06 France | 0.984009, rank 347 |
| **R19 `output_s8b/`** | **top-2 re-rank with cross-encoders (US/India); R06 France + edits (a, c, d, e)** | **0.987777, rank 145 (best)** |

R07, R08 and R09 changed only France relative to R06, and all three lost.

**What the `s8b` result tells us:**
- The projection was 0.9885 (range 0.987–0.990). The actual score was 0.0008 below that centre; the same method had matched R14 exactly.
- There are two readings (see the table in §1):
  1. The France edits added only about +0.003 France F instead of the estimated +0.008.
  2. US/India transferred about 0.001 less than projected.
- One probe separates them. For example, upload `s8b`'s US/India rows with R06 France (the France rows of `output_s3`). Its diff to 0.987777 is the edits' effect, and its diff to R14's 0.984009 is the US/India effect.

**Current best file:** `output_s8b/`, which equals `output/matching_results.tsv` (sha256 `1218c64e…`).
- OOF: US 0.99206, India 0.99106; pooled 0.99165 at T1 0.58 / T2 0.78.
- `SteinsGate_submission.zip` is built with it. If a new final wins, rebuild the zip with that file.
- **Never overwrite `output_s8b/`.**

**Pipeline** (details in `docs/Documentation_BER.md` §3–4 and LOG.md):
1. **Normalize and block.** `ber/normalize.py`, then TF-IDF blocking to the top-10 S1s per query. The candidate file is `output/candidate_pairs.tsv`, a hardlink of `output_r05/`'s.
2. **Stages 1 and 2.** LightGBM pair models: runs x1–x4 with generator-aware xfeats and number-relation nfeats, cross-fitted P0 ↔ P1 over test-density pools.
3. **Stage 3.** LightGBM on each query's argmax row. Features:
   - consensus with the S1's other queries;
   - raw-text perturbations that normalization erased;
   - disagreement between x1 and x4;
   - acronym and name-collision features.
4. **Cross-encoders.** xlm-roberta base and large, each fine-tuned for 1 epoch on raw "name | address" pairs.
   - Training rows: uncertain argmax rows (0.003 < p < 0.997) plus 4% of the other argmax rows. The large model also gets the runner-up rows.
   - Cross-fitted; the test score is the mean of both folds.
5. **Top-2 re-rank (`p8b`).** LightGBM over each query's argmax and runner-up rows, with the competing row's features and both cross-encoder scores.
6. **Decisions.**
   - Each query keeps its top row.
   - Per S1, the first link needs p ≥ T1 and later links p ≥ T2.
   - Caps: 5 S2 and 6 S3 records per S1.
   - Stage-2 ties are dropped.
7. **France.** R06 (x1+x2) decisions plus four class edits, each backed by LB evidence:
   - **(a)** drop equal-address category swaps or additions that no other confident record of the S1 supports (−7.2k links);
   - **(c)** add equal-address acronyms (+1.6k);
   - **(d)** drop links whose number is +GEN relative to both the S1 and the consensus (−1.8k);
   - **(e)** add x3's equal-address true-noise-word links where the large cross-encoder scores > 0.72 (+6.5k).

## 3. Where everything is

**Read first:**
- `CLAUDE.md`.
- `LOG.md`: the Scoreboard and **R13–R19 in full**. This handoff is built on that research, so don't redo it.
- For older France evidence: R02, R07b, R08c and R09.
- `docs/Documentation_BER.md`, `code/business_entity_resolution/README.md` and `code/business_entity_resolution/src/stage3/run_stage3.sh`.

**Data:**
- Raw: `data/raw/{train,test}/*_source{1,2,3}.tsv` (columns `entity_id`, `business_name`, `business_address`, `country`) and `train_ground_truth.tsv`.
- Derived: normalized tables in `work/norm/`, pools in `work/pools/`, stage 1–2 predictions in `work/oof/`.

**Previous session's scratch: `work/r19_scratch/`** (8.2 GB). It is a hardlinked copy of `/tmp/claude-1000/-home-akshat-AmazonML2026/a03488fb-9f05-49db-b546-8d81719e0733/scratchpad`, which a reboot wipes. The scripts hardcode `SP='/tmp/claude-1000/...'`; if /tmp is gone, point `SP` at `work/r19_scratch`.

| files | contents |
|---|---|
| `g_{P0,P1,test}_{US,India}` | x3+x4 argmax tables |
| `g12_*` | x1+x2 argmax tables, including `g12_test_France` |
| `f_*`, `f12_*` | stage-3 features; `f12_test_France` holds France's consensus features |
| `rawf_*` | raw-text perturbation features |
| `p8b_oof_rows`, `p8b_test_rows`, `p8b_thresholds.json` | final re-rank scores (earlier `p3_`, `p6_`, `p8_` too) |
| `ceout/{xr,xl}_{fold}_{P0,P1,test}_{US,India,France}` | cross-encoder logits. `xr` = base model, `xl` = large; the fold (`P0`/`P1`) is the pool the model trained on, so pool scores come from the other fold's model |
| `ce/`, `ce2/` | cross-encoder text inputs (argmax rows / runner-up rows) |
| `fr_cls.parquet` | France class per argmax row: `num` ∈ {eq, gen, other, na} × `cls` ∈ {same_core, acronym, typo, TN_in, pure_gen_word, cat_swap, drop_only, no_shared, cat_added, rest} |
| `fr_pairs_*.parquet` | France link sets (`fr_pairs_acde` is `s8b`'s France) |
| `fr_rules.py`, `fr9.py`, `fr*.py` | France class analyses and edits |
| `plugin.py`, `plugin3.py` | projection |
| `lbattr.py`, `lbattr2.py` | LB attribution of past France uploads, by class |
| `decomp.py` | loss decomposition |
| `guards.py` | safe process list/kill |
| `vm/` | VM-side scripts and `chain*.sh` used for R14–R18 |

**Package code** (clean and env-driven through `BER_ROOT` and `S3_DIR`): `code/business_entity_resolution/src/stage3/`.

**Outputs:**
- `output_s3/`: R14.
- `output_s6b/` and `output_s8/`: earlier checkpoints.
- `output_s8b/`: best on LB.
- `output_x3/`: x3 run for all countries; France edit (e) takes its links from here.
- `output_blend/`: R06.
- `output_r05/`: source of `candidate_pairs.tsv`.
- Never overwrite `output/candidate_pairs.tsv`, `output_blend/`, `output_s3/` or `output_s8b/`.
- Every match must stay inside the candidate file. If you add candidates, regenerate the file and re-run the candidate check.

**JarvisLabs** (both instances paused):
- **CPU VM 518400** (16 vCPU / 64 GB).
  - The instance id and IP change on every resume.
  - SSH alias: `jlber` in `~/.ssh/config`. Refresh its HostName with `jl ssh <id> --print-command`.
  - `/home/ubuntu/AmazonML2026` holds the full work tree.
  - `/home/ubuntu/s3` holds the stage-3 tables, including the top-2 tables `h_`, `k_`, `kce_` and `kce2_`. Those four exist **only on the VM**.
- **A100 40GB container 518069:** the cross-encoder training setup.
  - **The cross-encoder checkpoints were not saved**; `ce_train.py` writes scores only. Scoring new rows means retraining.
  - Retraining both folds in parallel on one A100 takes about 35 min for base and 110 min for large, including scoring.
  - Add `save_pretrained` this time.
  - For parallel GPU work you may create a second A100 (₹84.24/h); the user approved spending "at any cost".

## 4. What we know (verified; the numbers are in LOG.md)

### Generator

- Every S2/S3 record matches at most one S1. There are no cross-country pairs, and 5.6% of S1s are singletons.
- Test has about 2.3 distractors per S1, against 1.22 in train. The pools use 1.9× density.
- **Type A distractor:** house number shifted *upward* by GEN ∈ {1, 2, 3, 4, 5, 7, 9, 11, 13, 21}, plus one name edit: an appended word from a per-country list, a swapped word, or a changed legal form.
- **Type B distractor:** same address, different business.
  - US: the distinctive first word is replaced.
  - France: names follow "{city/person} {category} {legal}", and Type B swaps the category word (e.g. pharmacie → service).
  - Type B is 12× rarer than Type A in US.
- **S1 is noisy.** In 22% of S1s with ≥ 3 numbered true matches, most true matches agree on a number other than S1's. That is why consensus over the S1's other queries matters.
- **France class evidence:**
  - True-noise words (groupe, france, développement, cie, services, fils, …): linked TN links are about 100% true (R07's LB loss).
  - Equal-address category swaps: about 0% true (R09 and x3 attribution).
  - Acronyms: true (US: 1,503 / 1,503 at equal numbers).
  - +GEN relative to both S1 and the consensus: 2.15% true in US OOF.
- **Rejected leaks:** no ID or row-order leak, no casing leak, and distractors don't cluster.
- **France names are generic:** `nantes comite sarl` appears in 60 S1s, and address-less `nantes club sarl` in 157.
- `normalize.py` line 43 already maps French legal forms (sarl, sas, sasu, eurl, …).

### Loss budget

**US/India OOF.** R14 decomposition, from before the cross-encoder; redo it on `p8b`. Each figure is the F gained if that bucket were fixed:

| bucket | F if fixed | count |
|---|---|---|
| blocked out | +0.00388 | 90.2k pairs |
| argmax right, below threshold | +0.00278 | 62.5k |
| wrong S1 won | +0.00226 | 49.7k |
| FPs | +0.00183 | 13.8k |

- **Blocked out:**
  - India recall@10 is about 0.984. Its misses are 42% empty address, 32% non-Latin and 25% Latin with an address; the last group includes initialism names such as mfprojects = modern fortune projects.
  - US misses are mostly empty-address queries.
- **Empty-address queries dominate the last three buckets:** same-name S1 twins in different states.

**Test vs OOF:** after the cross-encoder, the bias-corrected plug-in puts test US/India about 0.001 below OOF (0.002 if reading 2 in §2 is right).

**France:**
- About 0.969 for R06 France (implied by R14's LB), and about 0.972 for `s8b`'s France if the US/India projections hold (§2).
- 15.6% of France argmax queries are uncertain (p 0.02–0.98), against 4.9% in US.
- The uncertainty sits in two cells at equal numbers:

| cell | France share | US share | France uncertain | US uncertain |
|---|---|---|---|---|
| other name edits | 15.0% | 8.1% | 3.9% | 0.2% |
| one swapped word | 7.1% | 7.3% | 2.3% | 0.15% |

**Cross-encoders on France:**
- The **base** model contradicts LB-proven classes: it rejected 5.9k linked TN pairs.
- The **large** model reads France better:
  - unlinked equal-address TN pairs: 0.80;
  - TN additions: median 0.99;
  - category swaps: 0.47;
  - acronyms: 0.95–0.98.
- Neither is validated for France in general. Use them only where class evidence agrees.

## 5. Already tried and negative (don't repeat without a new reason)

- **Decision policy:** per-country thresholds, expected-F decoding and three thresholds all gave ≈ 0. A test-density threshold shift gave +0.00003.
- **Re-blocking:**
  - Fixing the `patna` transliteration bug changes no FN or FP.
  - K=50 blocking: US misses are empty-address, and France has few match-like candidates at ranks 11–50.
- **Model changes:**

| change | result |
|---|---|
| stage 4 (consensus recomputed from stage-3 scores) | worse |
| top-2 re-rank without the cross-encoder | 0 |
| cross-encoder-derived consensus features | +0.00001 |
| base cross-encoder on top of large | +0.00002 |
| mdeberta-v3-base cross-encoder | weaker (log-loss 0.146 vs 0.107), and out of GPU memory |
| lexicon dropout | 0 |
| 255-leaf stage 2 (x4) | ≈ x3 |
| x3 blended with x1/x2 | worse |

- **France without LB evidence:**

| attempt | LB effect or verdict |
|---|---|
| R07 lexicon imputation | −0.00114 |
| R08 normalization | −0.0004 to −0.0014 |
| R09 same-address rescue | −0.0045 |
| R10c address normalization | rejected |
| x3's France | −0.00028 |
| France stage 3 without the cross-encoder | no clear signal |
| France hybrid re-scorer | rejected: guesses on generic names (`nantes club sarl` × 157 S1s) |

## 6. Seed hypotheses (verify, extend and re-rank them; add your own)

The data is synthetic. Every regularity found so far (GEN shifts, word lists, S1 noise) is a deterministic mechanism. The closer we reconstruct the generator, the closer we get to 0.99+. Ask what the top 10 exploit that we don't.

### Probes (decide early)

**P1.** Decide which single question, answered by one upload, changes the most decisions. Candidates:
- `s8b` US/India + R06 France: splits the France edits' effect from the US/India transfer (§2);
- `s8b` without edit (e) (6.5k TN additions) or without edit (a) (7.2k category-swap removals): measures that edit's true rate;
- one new France change on top of `s8b`.

How to run probes:
- Change one thing per probe, and prefer probes that are also expected improvements.
- A probe's ΔLB is exact to 1e-6. Convert it to a class true rate with the plug-in (see `lbattr*.py`).

### France (the largest lever)

- **F1. Class-by-class audit of France's current decisions.**
  - Build the table: `fr_cls` (num × cls) × linked/unlinked × stage-2 p band × large-cross-encoder band, with counts.
  - Estimate each cell's true rate from every source of evidence:
    - the LB diffs against R06 (R07, R08, R09, x3, and now `s8b` vs R14 plus your probes; `lbattr*.py`);
    - the matching US/India cell (OOF labels);
    - large-cross-encoder scores, calibrated on US/India.
  - Choose each cell's decision by expected F0.5. Adding a link to an S1 that already has links breaks even at about 0.72 true.
  - Show which cells hold France's ≈ 0.02 gap.
- **F2. Learn France's edit vocabulary without labels.**
  - Pairs in the +GEN band are nearly all distractors (2.15% true in US), so their name edits sample France's *distractor* edits. R15 found these are legal-form swaps and category→category swaps.
  - Pairs that agree with the S1's consensus sample *true* noise.
  - Use the two word distributions as features or rules for equal-address pairs.
  - Check whether a legal-form change at an equal number is a distractor edit in US.
- **F3. A France-specific model trained on labels we can create.** Two sources:
  - synthetic France pairs from replaying the generator on the test France S1s: Type A and Type B distractors, plus true-noise edits measured on US/India with French vocabulary;
  - pseudo-labels from the LB-proven classes.

  Validate it only against the LB-evidenced classes and probes. It is risky because it encodes our assumptions, so measure before trusting it.
- **F4. Noisy S1 in France.** Look at the cells where the S1's number disagrees with the consensus (features in `f12_test_France`).

### US/India

- **U1.** Redo the loss decomposition on the `p8b` OOF (`decomp.py`) to see what is left after the cross-encoder.
- **U2. Blocking misses, the largest bucket.**
  - Add a second candidate source for India's non-Latin and initialism names, e.g. a multilingual embedding ANN or transliterated character n-grams, only for queries whose best candidate is weak.
  - Score the new pairs with the cross-encoder or a small model.
  - Add them to `candidate_pairs.tsv`.
  - Estimate the recoverable F before building.
- **U3. A stronger cross-encoder.**
  - Options: a second epoch or a second seed of the large model; all rows instead of the uncertain band; test-density negatives.
  - Large over base improved log-loss (0.123 → 0.107) but F by only +0.0001, so prefer new information over model size.
- **U4. Residual signal in hard cases.** Look for any remaining signal in empty-address twins and noisy-S1 cases: source (S2 vs S3), record counts per source, name-noise pattern.

## 7. How to project an upload

- **US/India: bias-corrected plug-in** (`plugin.py`, `plugin3.py`).
  1. Compute the expected F from calibrated p under the real decision rule on test.
  2. Subtract the OOF bias (OOF plug-in − OOF actual).

  This matched R14's LB exactly and was 0.0008 high for `s8b`.
- **France has no labels.**
  - The anchors are the LB points: 0.9691 for R06 France (R14) and ≈ 0.972 or ≈ 0.977 for `s8b`'s France, depending on the reading in §2. Say which one you use.
  - Each change is valued class by class using the true rates attributed from LB diffs.
  - This is the weakest part of the projection: every France change needs class-level evidence, and the projection must state its France assumption.
- For each upload, give the user the central value and range, and show the arithmetic.

## 8. Operating rules

**Time.** The hard deadline is 23:59 IST. Plan the final upload for 23:00–23:15, with the zip rebuilt on the final file.

**Logging** (see `CLAUDE.md`):
- Every run and every upload gets a LOG.md entry (the next ID is R20), a Scoreboard row and an `experiments.csv` row.
- Negative results get one line.
- After any upload, ask the user for the LB score and rank.

**Git and AWS.** Commit only when the user explicitly asks. Never use the default AWS profile (HireLens); use `amlc`.

**Laptop.** 13 GB RAM with Chrome running, and 5 GB of free disk.
- Keep laptop analyses under 3 GB RAM; a 7 GB script once rebooted it.
- Run heavy jobs on the VM.

**JarvisLabs** (`~/.local/bin/jl`: `status`, `list`, `resume`, `pause`, `create`, `ssh <id> --print-command`, `gpus`):
- **Prices:** A100 40GB ₹84.24/h, 16-vCPU VM ₹32.14/h (32 vCPU ₹64.28/h), and about ₹1.5/h of storage while paused.
- **Balance:** the user is topping up. Check `jl status` before paid work. **A zero balance deletes all instance data.**
- **Guards** (both running on the laptop now):
  - `work/vm/watchdog.sh` watches the VM (env `ID`, `MIN_BAL=60`). The file `work/vm/NO_IDLE_PAUSE` exists; keep it, because the idle rule once paused the VM mid-training.
  - `work/vm/gpu_guard.sh` watches the A100 (`ID=518069`, `MIN_BAL=60`).
  - After resuming the VM, restart the watchdog with the new id. Start one guard per new instance.
  - Kill the guards only by exact PID: `python work/r19_scratch/guards.py list|kill`. `pkill -f` and `pgrep -f` patterns have killed our own shell.
- **Discipline:**
  - Run paid jobs back-to-back.
  - Pause an instance the moment it goes idle.
  - Ask the user before destroying any instance.

**Compose, validate, package:**
1. Compose with `compose.py` (`P3=<tag> FR_PAIRS=<france pairs parquet>`).
2. Run the validator with a non-existent `--candidate` path, because the official validator runs out of memory on the candidate file.
3. Stream-check the candidates separately (see R14 and R19).
4. Build the zip: `CHOSEN=<dir> TEAM=SteinsGate bash work/logs/final_package.sh`.
5. If the final changes, update `run_stage3.sh`, the package README and `docs/Documentation_BER.md`, including its final-LB line.

## 9. Suggested timeline

| time | work |
|---|---|
| 17:15–18:00 | read, brainstorm, first audits (the France class table with its evidence, and the `p8b` loss decomposition); design probe 1. Resume the machines when jobs are ready for them. |
| about 18:00–18:30 | **upload 1**: a probe or a first step-up |
| 18:00–21:00 | build and run the top 2–3 levers in parallel across the GPUs, the VM and the laptop |
| about 21:00 | **upload 2**: the best candidate so far, or a second probe |
| 21:00–22:45 | final improvements, compose, project |
| 22:45–23:15 | validate, package, update LOG.md and the doc; **upload 3** = the best candidate |

## 10. Report to the user

**For each upload,** keep it short and include:
- the file path;
- its purpose (step-up or probe, and what the probe answers);
- the expected LB (central value and range).

**For the final,** add:
- per-country F;
- the evidence behind every France change;
- one clear line: **"upload `<dir>/matching_results.tsv`"**.
