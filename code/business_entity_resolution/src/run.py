"""CLI for the entity-resolution pipeline. Run from the directory that holds data/, work/ and output/:

    python code/business_entity_resolution/src/run.py <command> [options]

Commands (in build order): env check1 check2 norm dicts pools block handmap feats train1 train2 tune loco predict package
"""
import argparse
import os
import platform
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ber import config as C  # noqa: E402


def cmd_env(a):
    import importlib

    for m in ["polars", "numpy", "scipy", "sklearn", "sparse_dot_topn", "rapidfuzz", "unidecode", "lightgbm"]:
        mod = importlib.import_module(m)
        print(f"{m:16s} {getattr(mod, '__version__', '?')}")
    print(f"python {platform.python_version()}  nproc {os.cpu_count()}  N_JOBS {C.N_JOBS}")
    try:
        with open("/proc/meminfo") as f:
            mem = {l.split(":")[0]: int(l.split()[1]) for l in f}
        print(f"RAM total {mem['MemTotal'] / 2**20:.1f} GB  available {mem['MemAvailable'] / 2**20:.1f} GB")
    except OSError:
        pass
    print(f"DATA_DIR={C.DATA_DIR} WORK_DIR={C.WORK_DIR} OUTPUT_DIR={C.OUTPUT_DIR} DEV_FRAC={C.DEV_FRAC}")


def cmd_check1(a):
    from ber.checks import step1
    step1()


def cmd_check2(a):
    from ber.checks import step2
    step2()


def cmd_check8(a):
    from ber.checks import step8
    step8()


def cmd_norm(a):
    from ber.normalize import normalize_all
    normalize_all(splits=a.splits)


def cmd_dicts(a):
    from ber.translit import learn_and_check
    learn_and_check()


def cmd_pools(a):
    from ber.pools import build_pools
    build_pools()


def cmd_block(a):
    from ber.blocking import block_partitions
    block_partitions(a.parts)


def cmd_handmap(a):
    from ber.blocking import handmap_ab
    handmap_ab()


def cmd_feats(a):
    from ber.features import build_features
    build_features(a.parts)


def cmd_train1(a):
    from ber.model import train_stage1
    train_stage1(tier0=a.tier0, reuse=a.reuse)


def cmd_train2(a):
    from ber.model import train_stage2
    train_stage2()


def cmd_tune(a):
    from ber.decide import tune
    tune()


def cmd_loco(a):
    from ber.model import loco
    loco()


def cmd_predict(a):
    from ber.submit import predict_and_write
    predict_and_write(tier0=a.tier0, check_ids=a.check_ids)


def cmd_package(a):
    from ber.submit import package
    package(a.team)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ["env", "check1", "check2", "check8", "dicts", "pools", "handmap", "train2", "tune", "loco"]:
        sub.add_parser(name)
    s = sub.add_parser("norm")
    s.add_argument("--splits", nargs="+", default=list(C.SPLITS))
    for name in ["block", "feats"]:
        s = sub.add_parser(name)
        s.add_argument("--parts", nargs="+", default=["P0", "P1", "test"], help="partitions: P0 P1 test")
    for name in ["train1", "predict"]:
        s = sub.add_parser(name)
        s.add_argument("--tier0", action="store_true", help="stage-1 only, P0 model, single threshold")
        s.add_argument("--reuse", action="store_true", help="train1: keep an existing model+OOF instead of refitting")
        s.add_argument("--check-ids", action="store_true", help="predict: validator --check-ids (needs a few GB RAM)")
    s = sub.add_parser("package")
    s.add_argument("--team", default=os.environ.get("TEAM_NAME", "team"))
    a = p.parse_args()
    t0 = time.time()
    globals()[f"cmd_{a.cmd}"](a)
    print(f"[{a.cmd}] done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
