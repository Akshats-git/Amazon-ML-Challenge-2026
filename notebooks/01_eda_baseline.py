# %% [markdown]
# # 01 - EDA + first baseline
# Run cell-by-cell in VS Code ("Run Cell" above each `# %%`), kernel = `.venv` / "Python (amlc)".
# Goal for hour 1-3: understand the data, reproduce the metric, get a VALID submission on the board.
# Only edit the CONFIG cell to adapt to the real problem.

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent / "src") if Path.cwd().name == "notebooks" else "src")

import numpy as np
import pandas as pd

from amlc.config import OOF, RAW
from amlc.cv import make_folds, train_gbdt_cv
from amlc.metrics import get_metric
from amlc.submit import write_submission
from amlc.utils import log_experiment, seed_everything

seed_everything()
pd.set_option("display.max_columns", 100, "display.max_colwidth", 200)
print(sorted(p.relative_to(RAW) for p in RAW.rglob("*") if p.is_file())[:50])

# %% CONFIG - fill in from the problem statement
TRAIN_FILE = RAW / "train.csv"
TEST_FILE = RAW / "test.csv"
SAMPLE_FILE = RAW / "sample_test_out.csv"   # the official sample output file
ID_COL = "sample_id"
TARGET = "price"
TASK = "regression"                          # regression | binary | multiclass
METRIC = "smape"                             # key in amlc.metrics.METRICS (add the official one if new)
TARGET_TRANSFORM = "log1p"                   # log1p for skewed positive targets (prices, lengths), else None
TEXT_COLS = ["catalog_content"]
IMAGE_COL = "image_link"                     # None if no images
GROUP_COL = None                             # e.g. a product/group id if near-duplicates exist

metric_fn, greater_is_better = get_metric(METRIC)

# %% Load
train = pd.read_csv(TRAIN_FILE)
test = pd.read_csv(TEST_FILE)
print(train.shape, test.shape)
train.head()

# %% Structure: dtypes, nulls, uniques
summary = pd.DataFrame({
    "dtype": train.dtypes,
    "null%": train.isna().mean().round(4) * 100,
    "nunique": train.nunique(),
    "example": train.iloc[0],
})
summary

# %% Target
y = train[TARGET]
print(y.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]))
if TASK == "regression":
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(12, 3))
    y.clip(upper=y.quantile(0.99)).hist(bins=100, ax=ax[0]); ax[0].set_title(TARGET)
    np.log1p(y.clip(lower=0)).hist(bins=100, ax=ax[1]); ax[1].set_title(f"log1p({TARGET})")
    plt.show()
else:
    print(y.value_counts(normalize=True).head(30))

# %% Text columns: lengths + a few raw examples (READ THEM - structure hides in the text)
for c in TEXT_COLS:
    s = train[c].fillna("").astype(str)
    print(c, s.str.len().describe().round(0).to_dict())
for t in train[TEXT_COLS[0]].sample(3, random_state=0):
    print("-" * 100, "\n", t[:1500])

# %% Train/test drift + duplicates
for c in TEXT_COLS:
    print(c, "exact dup texts in train:", train[c].duplicated().sum(),
          "| test texts seen in train:", test[c].isin(set(train[c])).sum())
if IMAGE_COL:
    print("image urls shared train/test:", test[IMAGE_COL].isin(set(train[IMAGE_COL])).sum())

# %% Sanity-check the metric implementation on trivial predictions
const = np.full(len(y), y.median()) if TASK == "regression" else np.full(len(y), y.mode()[0])
print(f"{METRIC} of constant prediction: {metric_fn(y, const):.4f}")

# %% Folds (fixed for the whole competition - save them, everyone on the team uses the same file)
folds = make_folds(train, target=TARGET, task=TASK, group=GROUP_COL)
np.save(OOF / "folds.npy", folds)
print(np.bincount(folds))

# %% Baseline A: TF-IDF (word + char) + Ridge/Logistic - minutes on CPU, surprisingly strong on text
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge

text_tr = train[TEXT_COLS].fillna("").astype(str).agg(" ".join, axis=1)
text_te = test[TEXT_COLS].fillna("").astype(str).agg(" ".join, axis=1)
word = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=300_000, sublinear_tf=True)
char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5, max_features=300_000, sublinear_tf=True)
X_tr = hstack([word.fit_transform(text_tr), char.fit_transform(text_tr)]).tocsr()
X_te = hstack([word.transform(text_te), char.transform(text_te)]).tocsr()
print(X_tr.shape)

yt = np.log1p(y.to_numpy()) if TARGET_TRANSFORM == "log1p" else y.to_numpy()
oof_a, test_a = np.zeros(len(train)), np.zeros(len(test))
for k in range(folds.max() + 1):
    tr, va = folds != k, folds == k
    m = Ridge(alpha=2.0) if TASK == "regression" else LogisticRegression(C=4, max_iter=2000)
    m.fit(X_tr[tr], yt[tr])
    oof_a[va] = m.predict(X_tr[va])
    test_a += m.predict(X_te) / (folds.max() + 1)
if TARGET_TRANSFORM == "log1p":
    oof_a, test_a = np.expm1(oof_a), np.expm1(test_a)
score_a = metric_fn(y, oof_a)
print(f"TF-IDF Ridge CV {METRIC}: {score_a:.4f}")
np.save(OOF / "tfidf_ridge_oof.npy", oof_a); np.save(OOF / "tfidf_ridge_test.npy", test_a)
log_experiment("tfidf_ridge", score_a, "word1-2 + char3-5 tfidf, ridge a=2")

# %% Submit baseline A (get on the leaderboard early; validates the whole pipeline)
sub = pd.DataFrame({ID_COL: test[ID_COL], TARGET: test_a})
write_submission(sub, "tfidf_ridge", sample=SAMPLE_FILE, id_col=ID_COL)

# %% Baseline B: sentence embeddings (+ image embeddings) -> LightGBM
from amlc.embed import embed_texts

emb_tr = embed_texts(text_tr.tolist(), cache_name="txt_bge_small_train")
emb_te = embed_texts(text_te.tolist(), cache_name="txt_bge_small_test")
res_b = train_gbdt_cv(emb_tr, y.to_numpy(), emb_te, folds, model="lgb", task=TASK, metric=METRIC,
                      target_transform=TARGET_TRANSFORM, params={"learning_rate": 0.05})
np.save(OOF / "bge_lgb_oof.npy", res_b["oof"]); np.save(OOF / "bge_lgb_test.npy", res_b["test"])
log_experiment("bge_small_lgb", res_b["score"], "bge-small-en-v1.5 text emb -> lgb")

# %% Blend A + B on OOF (pick the weight on OOF, never on the public LB)
best = max(((w, metric_fn(y, w * oof_a + (1 - w) * res_b["oof"])) for w in np.linspace(0, 1, 21)),
           key=lambda t: t[1] if greater_is_better else -t[1])
print("best weight/score:", best)
