"""Benchmark runner: trains each (model, dataset) combination and logs timing.

Usage:
    python run_benchmarks.py                     # run default grid
    python run_benchmarks.py --models SASRec BSARec --datasets Beauty
    python run_benchmarks.py --dry_run           # print commands only
"""
import argparse
import itertools
import os
import subprocess
import sys
import time

DEFAULT_MODELS = [
    "BSARec", "SASRec", "BERT4Rec", "Caser",
    "GRU4Rec", "FMLPRec", "DuoRec", "FEARec",
    "LRURec", "Mamba4Rec", "ICSRec", "ICLRec",
    "SIGMA",
]
DEFAULT_DATASETS = [
    "LastFM", "Beauty", "ML-1M",
    "Sports_and_Outdoors", "Toys_and_Games", "Yelp",
]

# Paper-recommended hyperparameters per (model, dataset).
# Sources:
#   - BSARec README (Beauty)
#   - ICLRec/src/scripts/run_*.sh
#   - SIGMA/model/config.yaml
# Datasets without a script in the original repo (LastFM) reuse the
# closest-match settings (typically Beauty).
README_HPARAMS = {
    ("BSARec", "Beauty"): {
        "lr": "0.0005", "alpha": "0.7", "c": "5", "num_attention_heads": "1",
    },
    # ICLRec — from ICLRec/src/scripts/run_*.sh
    ("ICLRec", "Beauty"): {
        "cf_weight": "0.1",
        "contrast_type": "Hybrid",
        "num_intent_clusters": "256",
        "seq_representation_type": "mean",
        "warm_up_epoches": "0",
        "intent_cf_weight": "0.1",
        "num_hidden_layers": "1",
    },
    ("ICLRec", "ML-1M"): {
        "cf_weight": "0.0",
        "contrast_type": "IntentCL",
        "num_intent_clusters": "256",
        "seq_representation_type": "mean",
        "warm_up_epoches": "0",
        "intent_cf_weight": "0.1",
        "num_hidden_layers": "2",
        "max_seq_length": "200",
    },
    ("ICLRec", "LastFM"): {  # no script in repo; reuse Beauty
        "cf_weight": "0.1",
        "contrast_type": "Hybrid",
        "num_intent_clusters": "256",
        "seq_representation_type": "mean",
        "warm_up_epoches": "0",
        "intent_cf_weight": "0.1",
        "num_hidden_layers": "1",
    },
    ("ICLRec", "Sports_and_Outdoors"): {
        "cf_weight": "0.1",
        "contrast_type": "Hybrid",
        "num_intent_clusters": "256",
        "seq_representation_type": "mean",
        "warm_up_epoches": "0",
        "intent_cf_weight": "0.1",
        "num_hidden_layers": "2",
    },
    ("ICLRec", "Toys_and_Games"): {
        "cf_weight": "0.1",
        "contrast_type": "Hybrid",
        "num_intent_clusters": "256",
        "seq_representation_type": "mean",
        "warm_up_epoches": "0",
        "intent_cf_weight": "0.1",
        "num_hidden_layers": "3",
    },
    ("ICLRec", "Yelp"): {
        "cf_weight": "0.1",
        "contrast_type": "Hybrid",
        "num_intent_clusters": "256",
        "seq_representation_type": "mean",
        "warm_up_epoches": "0",
        "intent_cf_weight": "0.1",
        "num_hidden_layers": "2",
    },
    # SIGMA — from SIMGA/model/config.yaml (same config across datasets)
    **{
        ("SIGMA", _ds): {
            "num_hidden_layers": "1",
            "hidden_dropout_prob": "0.2",
            "attention_probs_dropout_prob": "0.2",
            "d_state": "32", "d_conv": "4", "expand": "2",
        }
        for _ds in [
            "LastFM", "Beauty", "ML-1M",
            "Sports_and_Outdoors", "Toys_and_Games", "Yelp",
        ]
    },
}


def build_cmd(model, dataset, args):
    train_name = f"{model}_{dataset}_bench"
    cmd = [
        sys.executable, "main.py",
        "--model_type", model,
        "--data_name", dataset,
        "--train_name", train_name,
        "--epochs", str(args.epochs),
        "--patience", str(args.patience),
        "--seed", str(args.seed),
        "--gpu_id", args.gpu_id,
    ]
    if args.no_cuda:
        cmd.append("--no_cuda")
    for k, v in README_HPARAMS.get((model, dataset), {}).items():
        if isinstance(v, bool):
            if v:
                cmd.append(f"--{k}")
        else:
            cmd.extend([f"--{k}", str(v)])
    return train_name, cmd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu_id", type=str, default="0")
    p.add_argument("--no_cuda", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip runs whose log already exists in output/")
    args = p.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs("output", exist_ok=True)

    combos = list(itertools.product(args.models, args.datasets))
    print(f"[runner] planned {len(combos)} runs: {combos}")

    summary = []
    total_start = time.perf_counter()
    for i, (model, dataset) in enumerate(combos, 1):
        train_name, cmd = build_cmd(model, dataset, args)
        log_path = os.path.join("output", f"{train_name}.log")

        if args.skip_existing and os.path.exists(log_path):
            print(f"[runner] ({i}/{len(combos)}) SKIP {train_name} (log exists)")
            continue

        print(f"\n[runner] ({i}/{len(combos)}) RUN {train_name}")
        print(f"[runner] cmd: {' '.join(cmd)}")
        if args.dry_run:
            continue

        t0 = time.perf_counter()
        rc = subprocess.call(cmd)
        elapsed = time.perf_counter() - t0
        status = "ok" if rc == 0 else f"fail(rc={rc})"
        summary.append((train_name, status, elapsed))
        print(f"[runner] ({i}/{len(combos)}) DONE {train_name} {status} {elapsed:.1f}s")

    total_elapsed = time.perf_counter() - total_start
    print("\n========= runner summary =========")
    for name, status, t in summary:
        print(f"  {name:40s} {status:10s} {t:8.1f}s")
    print(f"[runner] total wall time: {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()
