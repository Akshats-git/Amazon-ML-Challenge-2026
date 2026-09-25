# CLAUDE.md

Amazon ML Challenge 2026 (Business Entity Resolution, macro F0.5). The team plan is in [PLAYBOOK.md](PLAYBOOK.md) and the layout is in [README.md](README.md).

## Run log: always keep [LOG.md](LOG.md) up to date

We use LOG.md throughout the 72 hours to decide what to try next, and we write the final approach doc from it. Keep it clean and short.

- **After every dev/test run**, append an entry at the bottom (`R<NN> · name · time IST`) with these fields:
  - **Approach:** what changed compared with the previous run: pipeline stages, key params, features, model.
  - **Data:** train frac or sample, and the machine used (laptop / SageMaker / Kaggle).
  - **Results:** *all* metrics the run printed: OOF macro F0.5, blocking pair recall, tuned threshold, the per-match-count F/precision/recall breakdown if notable, and runtime.
  - **LB:** public score **and leaderboard rank** once it's submitted. If rank isn't known yet, write `pending` and ask the user for it.
  - **Takeaways** (one or two lines) and **Next**.
- Add a matching row to the **Scoreboard** table and update **Current best on LB**.
- When the user reports an LB score or rank, update both the entry and the Scoreboard row straight away.
- Also record failed or abandoned runs with one line saying why. Negative results matter too.
- `experiments.csv` (written by `log_experiment`) stays the raw machine log. LOG.md is the curated human log. Keep them consistent.
