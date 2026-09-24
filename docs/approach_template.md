# <Team name> — Amazon ML Challenge 2026 Approach

**Team:** <names, colleges> · **Final private-LB metric:** <…> · **CV:** <…>

<!-- Hard limit: 1–2 pages as PDF. Fill it DURING the hackathon from experiments.csv, not at 23:00 on Sunday.
     The top 10 are chosen on leaderboard + this doc, and the finale Q&A with Amazon scientists starts from it. -->

## 1. Problem understanding
- Task, inputs, target, metric (one line each).
- 2–3 key observations from EDA that shaped the solution (e.g. target skew, text structure, duplicates, missing images).

## 2. Validation strategy
- Folds (k, stratification/grouping) and why. How well CV agreed with the public leaderboard.

## 3. Features
| Source | Features | Why |
|---|---|---|
| Text | … | … |
| Image | … | … |
| Engineered | … | … |

## 4. Models
| Model | Input | CV | LB |
|---|---|---|---|
| Baseline: TF-IDF + Ridge | | | |
| … | | | |
| **Final ensemble** | | | |

Training details: target transform, loss, key hyperparameters, compute used (GPU type, hours).

## 5. Ensembling and post-processing
- How the blend weights were chosen (on OOF), any post-processing and its CV gain.

## 6. What didn't work
- Short list with the measured impact. It shows rigour.

## 7. Reproducibility
- Entry points: `notebooks/…`, `scripts/…`. Runtime. External models used (name, size, licence).
