"""Run the fixed B-module CNN comparison without selecting on test scores."""

import argparse
import csv
import gc
import hashlib
import importlib.util
import json
import math
import os
import platform
import random
import re
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Must be set before initializing CUDA for deterministic matrix multiplication.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import sklearn
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from cnn_model import TextCNN, state_sha256

HERE = Path(__file__).resolve().parent
LOCAL_TZ = timezone(timedelta(hours=8))
VARIANTS = ("baseline", "eda")
DATA_NAMES = (
    "sst2_train_500.txt", "sst2_train_500_eda.txt", "train_400.txt",
    "train_eda_final.txt", "val_100.txt", "sst2_dev_872.tsv",
)


def now():
    return datetime.now(LOCAL_TZ).isoformat(timespec="seconds")


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_path(base, value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def seed_everything(seed, deterministic):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(deterministic)


def eda_clean(text):
    """Match official get_only_chars(), with safe empty-string handling."""
    text = text.replace("\u2019", "").replace("'", "").replace("-", " ")
    text = text.replace("\t", " ").replace("\n", " ").lower()
    text = "".join(char if char in "abcdefghijklmnopqrstuvwxyz " else " " for char in text)
    return re.sub(" +", " ", text).strip()


def read_label_first(path):
    return [tuple(row.split("\t", 1)) for row in
            Path(path).read_text(encoding="utf-8").splitlines() if row.strip()]


def audit_data(data_dir):
    original = read_label_first(data_dir / "sst2_train_500.txt")
    full_eda = read_label_first(data_dir / "sst2_train_500_eda.txt")
    baseline = read_label_first(data_dir / "train_400.txt")
    augmented = read_label_first(data_dir / "train_eda_final.txt")
    validation = read_label_first(data_dir / "val_100.txt")
    if tuple(map(len, (original, full_eda, baseline, augmented, validation))) != (
        500, 8500, 400, 6800, 100
    ):
        raise ValueError("Unexpected data counts; check A's delivery.")
    if Counter(original) != Counter(baseline + validation):
        raise ValueError("Baseline and validation do not partition original data.")
    if set(original) != set(baseline + validation) or len(set(original)) != 500:
        raise ValueError("Duplicate original rows need a source-ID based audit.")
    if {eda_clean(t) for _, t in baseline} & {eda_clean(t) for _, t in validation}:
        raise ValueError("Training and validation overlap after EDA normalization.")
    if not all(
        all(label == source[0] for label, _ in full_eda[i * 17:(i + 1) * 17])
        and eda_clean(full_eda[i * 17 + 16][1]) == eda_clean(source[1])
        for i, source in enumerate(original)
    ):
        raise ValueError("Augmentation blocks do not match their original examples.")
    baseline_set = set(baseline)
    expected = [row for i, source in enumerate(original) if source in baseline_set
                for row in full_eda[i * 17:(i + 1) * 17]]
    if augmented != expected:
        raise ValueError("EDA training contains unexpected source blocks.")
    original_ids = {row: i for i, row in enumerate(original)}
    test_rows = (data_dir / "sst2_dev_872.tsv").read_text(encoding="utf-8").splitlines()
    if test_rows[0].split("\t") != ["sentence", "label"]:
        raise ValueError("Unexpected evaluation header.")
    test = [(label, text) for text, label in (row.rsplit("\t", 1) for row in test_rows[1:])]
    if len(test) != 872:
        raise ValueError("Expected 872 evaluation examples.")
    split_rows = {"baseline_train": baseline, "eda_train": augmented, "validation": validation}
    lookups = {name: {eda_clean(t) for _, t in rows} for name, rows in split_rows.items()}
    whitespace_lookups = {
        name: {" ".join(t.lower().split()) for _, t in rows} for name, rows in split_rows.items()
    }
    overlaps = []
    for test_id, (label, text) in enumerate(test):
        matches = [name for name, texts in lookups.items() if eda_clean(text) in texts]
        if matches:
            overlaps.append({
                "id": test_id, "text": text, "gold_label": int(label),
                "matching_splits": matches,
                "whitespace_exact_splits": [name for name, texts in whitespace_lookups.items()
                                             if " ".join(text.lower().split()) in texts],
            })
    excluded = {row["id"] for row in overlaps}
    clean_ids = [i for i in range(len(test)) if i not in excluded]
    report = {
        "created_at": now(),
        "sample_counts": {"original": 500, "full_eda": 8500, "baseline_train": 400,
                          "eda_train": 6800, "validation": 100, "evaluation": 872,
                          "evaluation_without_detected_overlap": len(clean_ids)},
        "label_counts": {name: dict(Counter(label for label, _ in rows))
                         for name, rows in {**split_rows, "evaluation": test}.items()},
        "validation_is_disjoint_from_training": True,
        "augmented_training_source_blocks_verified": True,
        "baseline_source_ids": [original_ids[row] for row in baseline],
        "validation_source_ids": [original_ids[row] for row in validation],
        "normalization": "official EDA get_only_chars: lowercase, remove apostrophes, ASCII letters, collapse spaces",
        "overlapping_evaluation_rows": overlaps,
        "clean_evaluation_ids": clean_ids,
        "file_sha256": {name: sha256(data_dir / name) for name in DATA_NAMES},
        "limitations": [
            "Full evaluation includes text overlap with training/validation; also report the fixed non-overlap subset.",
            "Normalization only detects matching cleaned text, not paraphrases or shared sentence fragments.",
            "The shared vocabulary includes EDA training words for both variants, as required by A's interface.",
            "The delivered EDA corpus is fixed across model seeds; augmentation randomness is not resampled.",
            "Epochs contain 13 baseline or 213 EDA optimizer steps at batch_size=32; update budgets differ.",
        ],
    }
    return report


def load_data(config, config_dir, output_dir):
    a_dir = resolve_path(config_dir, config["a_module_dir"])
    glove = resolve_path(config_dir, os.environ.get("GLOVE_PATH", config["glove_path"]))
    if not glove.is_file():
        raise FileNotFoundError(f"GloVe file not found: {glove}")
    data_dir = a_dir / "data"
    audit = audit_data(data_dir)
    write_json(output_dir / "data_audit.json", audit)
    spec = importlib.util.spec_from_file_location("a_data_utils", a_dir / "data_utils.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.MAX_LEN != config["max_len"]:
        raise ValueError("Use A's MAX_LEN consistently for all models.")
    np.random.seed(config["embedding_seed"])
    print(f"Loading A data interface and GloVe: {glove}", flush=True)
    started = time.perf_counter()
    data = module.load_splits(data_dir=str(data_dir), glove_path=str(glove))
    expected = {"baseline_train_x": (400, 50), "train_x": (6800, 50),
                "val_x": (100, 50), "test_x": (872, 50)}
    for key, shape in expected.items():
        if data[key].shape != shape:
            raise ValueError(f"Unexpected {key} shape: {data[key].shape}")
    if data["embedding"].shape != (len(data["vocab"]), config["embedding_dim"]):
        raise ValueError("Unexpected embedding dimensions.")
    if not np.isfinite(data["embedding"]).all():
        raise ValueError("Embedding contains non-finite numbers.")
    write_json(output_dir / "vocab.json", data["vocab"])
    np.save(output_dir / "embedding.npy", data["embedding"])
    audit["vocab_size"] = len(data["vocab"])
    audit["embedding_seed"] = config["embedding_seed"]
    audit["embedding_sha256"] = sha256(output_dir / "embedding.npy")
    audit["data_utils_sha256"] = sha256(a_dir / "data_utils.py")
    # Capture a checksum of the large local resource as provenance, too.
    audit["glove"] = {"path": str(glove), "bytes": glove.stat().st_size, "sha256": sha256(glove)}
    audit["load_seconds"] = time.perf_counter() - started
    write_json(output_dir / "data_audit.json", audit)
    print(f"Data ready: vocab={len(data['vocab'])}, clean evaluation={len(audit['clean_evaluation_ids'])}", flush=True)
    return data, audit


def make_loader(x, y, batch_size, shuffle=False, seed=42):
    return DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
        batch_size=batch_size, shuffle=shuffle, num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    total_loss, labels, probabilities = 0.0, [], []
    for x, y in loader:
        logits = model(x.to(device))
        total_loss += nn.functional.cross_entropy(logits, y.to(device), reduction="sum").item()
        labels.append(y.numpy())
        probabilities.append(logits.softmax(dim=1).cpu().numpy())
    labels, probabilities = np.concatenate(labels), np.concatenate(probabilities)
    predicted = probabilities.argmax(axis=1)
    return total_loss / len(labels), labels, predicted, probabilities


def classification_metrics(gold, predicted):
    return {
        "n": len(gold), "accuracy": float(accuracy_score(gold, predicted)),
        "macro_f1": float(f1_score(gold, predicted, average="macro", labels=[0, 1], zero_division=0)),
        "confusion_matrix_labels_0_1": confusion_matrix(gold, predicted, labels=[0, 1]).tolist(),
    }


def save_predictions(path, texts, gold, predicted, probabilities):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "text", "gold_label", "pred_label", "prob_0", "prob_1"])
        writer.writerows((i, text, int(gold[i]), int(predicted[i]),
                          format(float(probabilities[i, 0]), ".9g"),
                          format(float(probabilities[i, 1]), ".9g"))
                         for i, text in enumerate(texts))


def run_one(variant, seed, config, data, audit, output_dir, device):
    run_dir = output_dir / f"cnn_{variant}_seed{seed}"
    run_dir.mkdir()
    seed_everything(seed, config["deterministic"])
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    model = TextCNN(data["embedding"], config).to(device)
    initial_hash = state_sha256(model)
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad), lr=config["learning_rate"],
        betas=tuple(config["adam_betas"]), eps=config["adam_eps"],
        weight_decay=config["weight_decay"],
    )
    prefix = "baseline_train" if variant == "baseline" else "train"
    loader = make_loader(data[f"{prefix}_x"], data[f"{prefix}_y"], config["batch_size"], True, seed)
    validation_loader = make_loader(data["val_x"], data["val_y"], config["batch_size"])
    run_config = {
        **config, "variant": variant, "seed": seed, "train_samples": len(loader.dataset),
        "optimizer_steps_per_epoch": len(loader), "vocab_size": len(data["vocab"]),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "initial_model_sha256": initial_hash, "embedding_sha256": audit["embedding_sha256"],
        "early_stopping_monitor": "validation_cross_entropy", "restore_best_weights": True,
        "test_evaluation_policy": "one pass after selecting the lowest-validation-loss checkpoint",
        "actual_device": str(device), "started_at": now(),
    }
    write_json(run_dir / "config.json", run_config)
    best_loss, best_epoch, stale, optimizer_steps = math.inf, 0, 0, 0
    history = []
    started = time.perf_counter()
    checkpoint_path = run_dir / "best_model.pt"
    print(f"START cnn_{variant}_seed{seed}; train={len(loader.dataset)}", flush=True)
    fields = ["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy",
              "val_macro_f1", "improved", "stale_epochs", "optimizer_steps", "seconds"]
    with (run_dir / "history.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, config["max_epochs"] + 1):
            epoch_started = time.perf_counter()
            model.train()
            loss_sum, correct, samples = 0.0, 0, 0
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                loss = nn.functional.cross_entropy(logits, y)
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite loss; stop instead of emitting invalid results.")
                loss.backward()
                optimizer.step()
                optimizer_steps += 1
                loss_sum += loss.item() * len(y)
                correct += (logits.argmax(dim=1) == y).sum().item()
                samples += len(y)
            val_loss, val_gold, val_pred, _ = evaluate(model, validation_loader, device)
            val_metrics = classification_metrics(val_gold, val_pred)
            improved = val_loss < best_loss - config["min_delta"]
            if improved:
                best_loss, best_epoch, stale = val_loss, epoch, 0
                torch.save({"model_state_dict": model.state_dict(), "config": run_config,
                            "best_epoch": best_epoch, "best_val_loss": best_loss}, checkpoint_path)
            else:
                stale += 1
            row = {
                "epoch": epoch, "train_loss": loss_sum / samples, "train_accuracy": correct / samples,
                "val_loss": val_loss, "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"], "improved": improved,
                "stale_epochs": stale, "optimizer_steps": optimizer_steps,
                "seconds": time.perf_counter() - epoch_started,
            }
            writer.writerow(row)
            stream.flush()
            history.append(row)
            print(f"{variant} seed={seed} epoch={epoch:02d} train_loss={row['train_loss']:.4f} "
                  f"val_loss={val_loss:.4f} val_acc={val_metrics['accuracy']:.3f} "
                  f"best={best_epoch} patience={stale}/{config['patience']}", flush=True)
            if stale >= config["patience"]:
                break
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    # This is the only evaluation pass over test_x/test_y for this trained run.
    test_loss, gold, predicted, probabilities = evaluate(
        model, make_loader(data["test_x"], data["test_y"], config["batch_size"]), device
    )
    clean_ids = np.array(audit["clean_evaluation_ids"], dtype=np.int64)
    predictions_path = output_dir / f"predictions_cnn_{variant}_seed{seed}.csv"
    save_predictions(predictions_path, data["test_texts"], gold, predicted, probabilities)
    metrics = {
        "variant": variant, "seed": seed, "evaluation_full": classification_metrics(gold, predicted),
        "evaluation_without_detected_overlap": classification_metrics(gold[clean_ids], predicted[clean_ids]),
        "test_loss": test_loss, "best_epoch": best_epoch, "best_val_loss": best_loss,
        "epochs_trained": len(history), "stopped_early": stale >= config["patience"],
        "optimizer_steps": optimizer_steps, "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None,
        "peak_cuda_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20 if device.type == "cuda" else None,
        "initial_model_sha256": initial_hash, "predictions_file": predictions_path.name,
        "predictions_sha256": sha256(predictions_path), "finished_at": now(),
    }
    write_json(run_dir / "metrics.json", metrics)
    print(f"DONE cnn_{variant}_seed{seed}: accuracy={metrics['evaluation_full']['accuracy']:.4f}; "
          f"macro_f1={metrics['evaluation_full']['macro_f1']:.4f}", flush=True)
    del model, optimizer, checkpoint
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metrics


def summarize(results, output_dir, audit):
    summary = {"created_at": now(), "seed_count": len(results) // 2,
               "accuracy_unit": "fraction; report tables use percent",
               "std_definition": "sample standard deviation across model seeds, ddof=1",
               "full_evaluation_n": 872, "non_overlap_evaluation_n": len(audit["clean_evaluation_ids"]),
               "variants": {}, "paired_deltas_eda_minus_baseline": {}}
    rows = []
    for result in results:
        rows.append({
            "variant": result["variant"], "seed": result["seed"],
            "accuracy": result["evaluation_full"]["accuracy"],
            "macro_f1": result["evaluation_full"]["macro_f1"],
            "non_overlap_accuracy": result["evaluation_without_detected_overlap"]["accuracy"],
            "non_overlap_macro_f1": result["evaluation_without_detected_overlap"]["macro_f1"],
            "best_epoch": result["best_epoch"], "epochs_trained": result["epochs_trained"],
            "elapsed_seconds": result["elapsed_seconds"],
            "peak_cuda_allocated_mib": result["peak_cuda_allocated_mib"],
        })
    with (output_dir / "results_by_seed.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for variant in VARIANTS:
        selected = [result for result in results if result["variant"] == variant]
        summary["variants"][variant] = {}
        for scope in ("evaluation_full", "evaluation_without_detected_overlap"):
            summary["variants"][variant][scope] = {}
            for metric in ("accuracy", "macro_f1"):
                values = [result[scope][metric] for result in selected]
                summary["variants"][variant][scope][metric] = {
                    "mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "per_seed": {str(result["seed"]): result[scope][metric] for result in selected},
                }
    for scope in ("evaluation_full", "evaluation_without_detected_overlap"):
        summary["paired_deltas_eda_minus_baseline"][scope] = {}
        for metric in ("accuracy", "macro_f1"):
            paired = []
            for seed in sorted({result["seed"] for result in results}):
                pair = {result["variant"]: result for result in results if result["seed"] == seed}
                if pair["baseline"]["initial_model_sha256"] != pair["eda"]["initial_model_sha256"]:
                    raise ValueError("Paired models have different initial weights.")
                paired.append(pair["eda"][scope][metric] - pair["baseline"][scope][metric])
            summary["paired_deltas_eda_minus_baseline"][scope][metric] = {
                "mean": float(np.mean(paired)), "std": float(np.std(paired, ddof=1)) if len(paired) > 1 else 0.0,
                "per_seed": paired,
            }
    write_json(output_dir / "summary.json", summary)
    return summary


def plot_history(results, output_dir):
    os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    seeds = sorted({result["seed"] for result in results})
    fig, axes = plt.subplots(len(seeds), 2, figsize=(11, 3.2 * len(seeds)), squeeze=False)
    for index, seed in enumerate(seeds):
        for variant in VARIANTS:
            path = output_dir / f"cnn_{variant}_seed{seed}" / "history.csv"
            with path.open(encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            epochs = [int(row["epoch"]) for row in rows]
            axes[index, 0].plot(epochs, [float(row["val_loss"]) for row in rows], label=variant)
            axes[index, 1].plot(epochs, [float(row["val_accuracy"]) * 100 for row in rows], label=variant)
        for column, ylabel in enumerate(("Validation cross entropy", "Validation accuracy (%)")):
            axes[index, column].set(title=f"Seed {seed}", xlabel="Epoch", ylabel=ylabel)
            axes[index, column].legend()
            axes[index, column].grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "learning_curves.png", dpi=180)
    fig.savefig(output_dir / "learning_curves.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--preflight", action="store_true", help="Check real inputs and one gradient step; never evaluate test.")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if args.device:
        config["device"] = args.device
    if len(set(config["seeds"])) != len(config["seeds"]) or not config["seeds"]:
        raise ValueError("Provide distinct model seeds.")
    output_dir = args.output_dir or HERE / "results" / datetime.now(LOCAL_TZ).strftime("%Y%m%d_%H%M%S")
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Use a new output directory to preserve existing evidence: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "protocol.json", {**config, "frozen_at_before_training": now(),
                                            "config_sha256": sha256(config_path)})
    torch.set_num_threads(config["num_threads"])
    device_name = config["device"]
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    device = torch.device(device_name)
    environment = {
        "created_at": now(), "python": sys.version, "python_executable": sys.executable,
        "platform": platform.platform(), "numpy": np.__version__, "torch": torch.__version__,
        "scikit_learn": sklearn.__version__, "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(), "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "gpu_total_memory_mib": torch.cuda.get_device_properties(device).total_memory / 2**20 if device.type == "cuda" else None,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "source_sha256": {name: sha256(HERE / name) for name in ("cnn_model.py", "train_cnn.py", "config.json")},
    }
    author_repo = resolve_path(config_path.parent, config["author_repo"])
    environment["author_reference_sha256"] = {
        name: sha256(author_repo / name) for name in ("experiments/methods.py", "code/eda.py", "code/augment.py")
    }
    write_json(output_dir / "environment.json", environment)
    print(f"Output: {output_dir}\nDevice: {device}; GPU: {environment['gpu']}", flush=True)
    try:
        data, audit = load_data(config, config_path.parent, output_dir)
        if args.preflight:
            seed_everything(config["seeds"][0], config["deterministic"])
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            model = TextCNN(data["embedding"], config).to(device)
            optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=config["learning_rate"])
            x = torch.from_numpy(data["baseline_train_x"][:config["batch_size"]]).to(device)
            y = torch.from_numpy(data["baseline_train_y"][:config["batch_size"]]).to(device)
            logits = model(x)
            loss = nn.functional.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            if logits.shape != (len(y), 2) or not torch.isfinite(loss):
                raise ValueError("Preflight failed.")
            result = {"status": "passed", "batch_size": len(y), "loss": loss.item(),
                      "test_evaluation_calls": 0, "device": str(device),
                      "total_parameters": sum(p.numel() for p in model.parameters()),
                      "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                      "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None}
            write_json(output_dir / "preflight.json", result)
            print(json.dumps(result), flush=True)
            return
        results = []
        for seed in config["seeds"]:
            for variant in VARIANTS:
                results.append(run_one(variant, seed, config, data, audit, output_dir, device))
        summarize(results, output_dir, audit)
        plot_history(results, output_dir)
        write_json(output_dir / "completion.json", {"status": "complete", "runs": len(results), "finished_at": now()})
        from verify_results import verify
        verify(output_dir)
        print(f"Completed {len(results)} runs. Summary: {output_dir / 'summary.json'}", flush=True)
    except Exception:
        (output_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
