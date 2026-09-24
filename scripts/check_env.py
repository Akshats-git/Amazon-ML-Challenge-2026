"""Smoke-test the whole stack on synthetic data.

    uv run python scripts/check_env.py            # libs + GPU + GBDT + CV + submission checks
    uv run python scripts/check_env.py --models   # also download + run the default text/image encoders
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from amlc import metrics  # noqa: E402
from amlc.cv import make_folds, train_gbdt_cv  # noqa: E402
from amlc.submit import check_submission  # noqa: E402

ok = True


def section(name):
    print(f"\n=== {name} ===")


def check(name, fn):
    global ok
    try:
        out = fn()
        print(f"[OK]   {name}" + (f": {out}" if out is not None else ""))
    except Exception as e:  # keep going, report everything
        ok = False
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")


section("libraries")
for mod in ["numpy", "pandas", "polars", "sklearn", "xgboost", "lightgbm", "catboost", "optuna",
            "torch", "torchvision", "transformers", "sentence_transformers", "timm", "peft",
            "accelerate", "cv2", "PIL", "boto3"]:
    check(mod, lambda m=mod: __import__(m).__version__)

section("GPU")
import torch  # noqa: E402

check("cuda available", lambda: torch.cuda.is_available() or (_ for _ in ()).throw(RuntimeError("no CUDA")))
if torch.cuda.is_available():
    check("device", lambda: f"{torch.cuda.get_device_name(0)}, "
          f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB, sm_{''.join(map(str, torch.cuda.get_device_capability(0)))}")
    check("arch in build", lambda: torch.cuda.get_arch_list())
    check("fp16 matmul", lambda: float((torch.randn(1024, 1024, device="cuda", dtype=torch.half) @
                                        torch.randn(1024, 1024, device="cuda", dtype=torch.half)).float().abs().mean()))

section("metrics")
check("smape", lambda: round(metrics.smape([100, 200, 0], [110, 180, 0]), 4))
check("entity_f1", lambda: round(metrics.entity_f1(["10 gram", "", "5 volt", ""], ["10 gram", "2 cm", "", ""]), 4))

section("CV + GBDT on synthetic regression")
rng = np.random.default_rng(0)
X = pd.DataFrame(rng.normal(size=(2000, 20)), columns=[f"f{i}" for i in range(20)])
y = np.exp(X["f0"] + 0.5 * X["f1"] + rng.normal(scale=0.1, size=len(X))) * 10
Xte = X.iloc[:300]
folds = make_folds(pd.DataFrame({"y": y}), target="y", task="regression", n_splits=3)
check("folds balanced", lambda: np.bincount(folds).tolist())
check("lightgbm cpu", lambda: round(train_gbdt_cv(X, y, Xte, folds, "lgb", metric="smape",
                                                  target_transform="log1p", params={"n_estimators": 300})["score"], 3))
check("xgboost gpu", lambda: round(train_gbdt_cv(X, y, Xte, folds, "xgb", metric="smape", target_transform="log1p",
                                                 params={"n_estimators": 300}, use_gpu=True)["score"], 3))
check("catboost gpu", lambda: round(train_gbdt_cv(X, y, Xte, folds, "cat", metric="smape", target_transform="log1p",
                                                  params={"iterations": 300}, use_gpu=True)["score"], 3))
yc = (X["f0"] > 0).astype(int).to_numpy()
fc = make_folds(pd.DataFrame({"y": yc}), target="y", task="binary", n_splits=3)
check("lightgbm binary", lambda: round(train_gbdt_cv(X, yc, Xte, fc, "lgb", task="binary", metric="accuracy",
                                                     params={"n_estimators": 200})["score"], 3))

section("submission validator")
sample = pd.DataFrame({"sample_id": [3, 1, 2], "price": [0.0, 0.0, 0.0]})
check("reorders valid sub", lambda: check_submission(
    pd.DataFrame({"sample_id": [1, 2, 3], "price": [1.0, 2.0, 3.0]}), sample, "sample_id")["sample_id"].tolist())


def _rejects_bad():
    try:
        check_submission(pd.DataFrame({"sample_id": [1, 2, 4], "price": [1, None, 3]}), sample, "sample_id")
    except ValueError as e:
        return f"rejected ({str(e).count('  - ')} problems)"
    raise AssertionError("bad submission was accepted")


check("rejects bad sub", _rejects_bad)

if "--models" in sys.argv:
    section("encoders (downloads to ~/.cache/huggingface)")
    from PIL import Image

    from amlc.embed import embed_images, embed_texts

    check("text emb", lambda: embed_texts(["Apple iPhone 15, 128 GB, Black", "Cotton T-shirt, pack of 3"]).shape)
    tmp = Path(tempfile.mkdtemp())
    Image.new("RGB", (300, 300), (200, 30, 30)).save(tmp / "a.jpg")
    check("image emb", lambda: embed_images([tmp / "a.jpg", tmp / "missing.jpg"], num_workers=0).shape)

print("\nALL GOOD" if ok else "\nSOME CHECKS FAILED - see above")
sys.exit(0 if ok else 1)
