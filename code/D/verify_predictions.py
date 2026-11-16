"""从本包的 results 目录重算四组准确率。"""

import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SEEDS = (42, 43, 44, 45, 46)
FILES = {
    "cnn_baseline": ROOT / "results" / "cnn" / "predictions_cnn_baseline_seed{seed}.csv",
    "cnn_eda": ROOT / "results" / "cnn" / "predictions_cnn_eda_seed{seed}.csv",
    "rnn_baseline": ROOT / "results" / "rnn" / "predictions_rnn_baseline_seed{seed}.csv",
    "rnn_eda": ROOT / "results" / "rnn" / "predictions_rnn_eda_seed{seed}.csv",
}


def accuracy(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    correct = sum(row["gold_label"] == row["pred_label"] for row in rows)
    return correct / len(rows), len(rows)


def summarize(pattern):
    per_seed = {}
    for seed in SEEDS:
        acc, n = accuracy(Path(str(pattern).format(seed=seed)))
        per_seed[str(seed)] = {"accuracy": acc, "n": n}
    values = [item["accuracy"] for item in per_seed.values()]
    return {
        "n": per_seed["42"]["n"],
        "mean": statistics.mean(values),
        "std": statistics.stdev(values),
        "per_seed": per_seed,
    }


def main():
    report = {name: summarize(pattern) for name, pattern in FILES.items()}
    out = ROOT / "results" / "verification.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for name, item in report.items():
        print(f"{name}: {item['mean'] * 100:.2f} ± {item['std'] * 100:.2f}  n={item['n']}")
    print("wrote", out)


if __name__ == "__main__":
    main()
