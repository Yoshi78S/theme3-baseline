"""Parse benchmark logs in output/*.log and emit a CSV summary.

For each `<model>_<dataset>_bench.log` (or any log matching --pattern):
  - pulls final Test Score line (HR@K, NDCG@K)
  - pulls TIMING markers (n_params, wall_train, epochs_run, train_epoch_mean,
    valid_epoch_mean, test time)

Usage:
    python aggregate_results.py
    python aggregate_results.py --pattern "*_bench.log" --out results.csv
"""
import argparse
import csv
import glob
import os
import re


TIMING_RE = re.compile(r"TIMING\s+(\S+)\s+([\d.]+)s?")
TIMING_INT_RE = re.compile(r"TIMING\s+(n_params|epochs_run)\s+(\d+)")
# Final test line comes right after "---------------Test Score---------------"
SCORE_RE = re.compile(
    r"'Epoch':\s*\d+,\s*"
    r"'HR@5':\s*'([\d.]+)',\s*'NDCG@5':\s*'([\d.]+)',\s*"
    r"'HR@10':\s*'([\d.]+)',\s*'NDCG@10':\s*'([\d.]+)',\s*"
    r"'HR@20':\s*'([\d.]+)',\s*'NDCG@20':\s*'([\d.]+)'"
)


def parse_log(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        text = f.read()

    timings = {}
    for m in TIMING_INT_RE.finditer(text):
        timings[m.group(1)] = int(m.group(2))
    for m in TIMING_RE.finditer(text):
        key, val = m.group(1), m.group(2)
        if key in timings:
            continue
        try:
            timings[key] = float(val)
        except ValueError:
            pass

    # Take the LAST test-score line in the file (the final test eval).
    score = None
    last_test_idx = text.rfind("Test Score")
    tail = text[last_test_idx:] if last_test_idx >= 0 else text
    matches = SCORE_RE.findall(tail)
    if matches:
        hr5, ndcg5, hr10, ndcg10, hr20, ndcg20 = matches[-1]
        score = {
            "HR@5": float(hr5), "NDCG@5": float(ndcg5),
            "HR@10": float(hr10), "NDCG@10": float(ndcg10),
            "HR@20": float(hr20), "NDCG@20": float(ndcg20),
        }
    return timings, score


def infer_model_dataset(name):
    # Expected: <Model>_<Dataset>_bench  (Dataset may contain underscores)
    stem = name[:-len("_bench")] if name.endswith("_bench") else name
    if "_" not in stem:
        return stem, ""
    model, dataset = stem.split("_", 1)
    return model, dataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log_dir", default="output")
    p.add_argument("--pattern", default="*_bench.log")
    p.add_argument("--out", default="benchmark_results.csv")
    args = p.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    paths = sorted(glob.glob(os.path.join(args.log_dir, args.pattern)))
    if not paths:
        print(f"[aggregate] no logs matching {args.pattern} in {args.log_dir}")
        return

    fieldnames = [
        "model", "dataset", "log",
        "n_params", "epochs_run", "wall_train_sec",
        "train_epoch_mean_sec", "train_epoch_sum_sec",
        "valid_epoch_mean_sec", "test_sec",
        "HR@5", "NDCG@5", "HR@10", "NDCG@10", "HR@20", "NDCG@20",
    ]
    rows = []
    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        model, dataset = infer_model_dataset(name)
        timings, score = parse_log(path)
        row = {
            "model": model,
            "dataset": dataset,
            "log": os.path.basename(path),
            "n_params": timings.get("n_params"),
            "epochs_run": timings.get("epochs_run"),
            "wall_train_sec": timings.get("wall_train"),
            "train_epoch_mean_sec": timings.get("train_epoch_mean"),
            "train_epoch_sum_sec": timings.get("train_epoch_sum"),
            "valid_epoch_mean_sec": timings.get("valid_epoch_mean"),
            "test_sec": timings.get("test"),
        }
        if score:
            row.update(score)
        rows.append(row)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"[aggregate] wrote {len(rows)} rows to {args.out}")

    for r in rows:
        hr10 = r.get("HR@10")
        tmean = r.get("train_epoch_mean_sec")
        print(f"  {r['model']:10s} {r['dataset']:20s} "
              f"HR@10={hr10 if hr10 is not None else 'NA':>7} "
              f"train_ep={tmean if tmean is not None else 'NA':>7}s "
              f"params={r['n_params']}")


if __name__ == "__main__":
    main()
