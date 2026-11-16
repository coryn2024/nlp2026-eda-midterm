"""Independently verify exported CNN predictions, logs and paired settings.

Uses only Python's standard library; does not run a model or re-evaluate test.
"""

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent


def check(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve_a_module_dir(protocol):
    """Allow a relocated A delivery; the caller still checks every input hash."""
    a_dir = Path(protocol["a_module_dir"])
    if not a_dir.is_absolute():
        a_dir = HERE / a_dir
    if (a_dir / "data_utils.py").is_file():
        return a_dir.resolve()
    current_config = read_json(HERE / "config.json")
    relocated = Path(current_config["a_module_dir"])
    if not relocated.is_absolute():
        relocated = HERE / relocated
    check((relocated / "data_utils.py").is_file(), "A's data interface was not found at the recorded or current configured path.")
    return relocated.resolve()


def score(rows):
    matrix = [[0, 0], [0, 0]]
    for row in rows:
        matrix[int(row["gold_label"])][int(row["pred_label"])] += 1
    f1s = []
    for label in (0, 1):
        true_positive = matrix[label][label]
        denominator = sum(matrix[label]) + sum(row[label] for row in matrix)
        f1s.append(2 * true_positive / denominator if denominator else 0.0)
    return {"n": len(rows), "accuracy": (matrix[0][0] + matrix[1][1]) / len(rows),
            "macro_f1": sum(f1s) / 2, "confusion_matrix_labels_0_1": matrix}


def verify(results_dir):
    protocol = read_json(results_dir / "protocol.json")
    audit = read_json(results_dir / "data_audit.json")
    environment = read_json(results_dir / "environment.json")
    summary = read_json(results_dir / "summary.json")
    completion = read_json(results_dir / "completion.json")
    check(completion["status"] == "complete", "Training did not complete.")
    seeds = protocol["seeds"]
    check(completion["runs"] == len(seeds) * 2, "Missing completed runs.")
    a_dir = resolve_a_module_dir(protocol)
    data_dir = a_dir / "data"
    for name, expected in audit["file_sha256"].items():
        check(sha256(data_dir / name) == expected, f"Data changed: {name}")
    check(sha256(a_dir / "data_utils.py") == audit["data_utils_sha256"], "A's interface changed.")
    check(sha256(results_dir / "embedding.npy") == audit["embedding_sha256"], "Embedding changed.")
    source_files_used = {}
    for name, expected in environment["source_sha256"].items():
        snapshot = results_dir / "source_snapshot" / name
        source = snapshot if snapshot.is_file() else HERE / name
        check(sha256(source) == expected, f"Training source changed: {name}")
        source_files_used[name] = str(source.resolve())
    with (data_dir / "sst2_dev_872.tsv").open(encoding="utf-8", newline="") as stream:
        reference = list(csv.DictReader(stream, delimiter="\t"))
    clean_ids = set(audit["clean_evaluation_ids"])
    configs, verified, row_by_run = {}, [], {}
    for seed in seeds:
        for variant in ("baseline", "eda"):
            run_dir = results_dir / f"cnn_{variant}_seed{seed}"
            metrics = read_json(run_dir / "metrics.json")
            config = read_json(run_dir / "config.json")
            configs[(seed, variant)] = config
            check(config["seed"] == seed and config["variant"] == variant, "Run identity mismatch.")
            for key, value in protocol.items():
                if key not in ("frozen_at_before_training", "config_sha256"):
                    check(config[key] == value, f"Protocol mismatch: {seed}/{variant}/{key}")
            predictions_path = results_dir / metrics["predictions_file"]
            check(sha256(predictions_path) == metrics["predictions_sha256"], "Prediction checksum mismatch.")
            with predictions_path.open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                check(reader.fieldnames == ["id", "text", "gold_label", "pred_label", "prob_0", "prob_1"],
                      "Prediction columns do not match the shared format.")
                rows = list(reader)
            check(len(rows) == len(reference) == 872, "Expected exactly 872 predictions.")
            for index, (row, ref) in enumerate(zip(rows, reference)):
                check(int(row["id"]) == index, "Prediction IDs are not sequential.")
                check(row["text"] == ref["sentence"] and row["gold_label"] == ref["label"],
                      f"Prediction/reference mismatch at ID {index}.")
                p0, p1 = float(row["prob_0"]), float(row["prob_1"])
                check(math.isfinite(p0) and math.isfinite(p1), "Non-finite probability.")
                check(0 <= p0 <= 1 and 0 <= p1 <= 1 and abs(p0 + p1 - 1) < 1e-6,
                      "Invalid probability distribution.")
                check(int(row["pred_label"]) == int(p1 > p0), "Prediction is not probability argmax.")
            calculated = {
                "evaluation_full": score(rows),
                "evaluation_without_detected_overlap": score([r for r in rows if int(r["id"]) in clean_ids]),
            }
            for scope, values in calculated.items():
                for key, value in values.items():
                    actual = metrics[scope][key]
                    if isinstance(value, float):
                        check(abs(actual - value) < 1e-12, f"Metric mismatch: {scope}/{key}")
                    else:
                        check(actual == value, f"Metric mismatch: {scope}/{key}")
            with (run_dir / "history.csv").open(encoding="utf-8", newline="") as stream:
                reader = csv.DictReader(stream)
                check(not any("test" in name for name in reader.fieldnames), "Test metrics logged during training.")
                history = list(reader)
            check(len(history) == metrics["epochs_trained"], "Incorrect epoch count.")
            check([int(r["epoch"]) for r in history] == list(range(1, len(history) + 1)), "Missing epochs.")
            best_row = min(history, key=lambda r: float(r["val_loss"]))
            check(int(best_row["epoch"]) == metrics["best_epoch"], "Wrong best checkpoint epoch.")
            check(abs(float(best_row["val_loss"]) - metrics["best_val_loss"]) < 1e-12, "Wrong best validation loss.")
            check((run_dir / "best_model.pt").is_file(), "Checkpoint missing.")
            check(metrics["optimizer_steps"] == len(history) * config["optimizer_steps_per_epoch"],
                  "Incorrect optimizer step count.")
            check(metrics["initial_model_sha256"] == config["initial_model_sha256"], "Initial hash mismatch.")
            if metrics["stopped_early"]:
                check(int(history[-1]["stale_epochs"]) == config["patience"], "Invalid early stopping.")
            else:
                check(len(history) == config["max_epochs"], "Unexpected training termination.")
            row_by_run[(seed, variant)] = rows
            verified.append({"seed": seed, "variant": variant, **calculated})
    for seed in seeds:
        baseline, augmented = configs[(seed, "baseline")], configs[(seed, "eda")]
        check(baseline["initial_model_sha256"] == augmented["initial_model_sha256"], "Initial weights differ.")
        allowed = {"variant", "train_samples", "optimizer_steps_per_epoch", "started_at"}
        check({k:v for k,v in baseline.items() if k not in allowed} ==
              {k:v for k,v in augmented.items() if k not in allowed}, "Paired settings differ.")
    for variant in ("baseline", "eda"):
        for scope in ("evaluation_full", "evaluation_without_detected_overlap"):
            for metric in ("accuracy", "macro_f1"):
                values = [r[scope][metric] for r in verified if r["variant"] == variant]
                recorded = summary["variants"][variant][scope][metric]
                check(abs(statistics.mean(values) - recorded["mean"]) < 1e-12, "Summary mean mismatch.")
                std = statistics.stdev(values) if len(values) > 1 else 0.0
                check(abs(std - recorded["std"]) < 1e-12, "Summary std mismatch.")
    report = {"status": "passed", "verified_runs": len(verified),
              "a_module_dir_used": str(a_dir), "source_files_used": source_files_used,
              "predictions_per_run": 872, "clean_subset_size": len(clean_ids),
              "checks": ["data and source checksums", "every prediction text, ID and true label",
                         "probability normalization and argmax", "independent accuracy, macro-F1 and confusion matrix",
                         "best validation checkpoint and early stopping", "identical paired initialization and settings",
                         "summary means and sample standard deviations"], "runs": verified}
    (results_dir / "verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: {len(verified)} runs, {len(verified)*872} predictions; metrics and paired settings verified.")
    return row_by_run


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    verify(args.results.resolve())
