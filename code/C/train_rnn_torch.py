"""
C 模块：PyTorch RNN / RNN + EDA 主实验
=========================================================
环境：
    Python 3.11.12
    PyTorch 2.7.0+cu126
    NumPy 2.2.6
    scikit-learn 1.7.0（A 的 data_utils 依赖）

使用：
    1. 将本文件放在项目根目录，与 data_utils.py 同级
    2. 确保 data/ 和 GloVe 文件路径正确
    3. PowerShell（可选，GloVe 不在 A 默认路径时）：
         $env:GLOVE_PATH="D:\\glove\\glove.6B.300d.txt"
    4. 运行：
         python train_rnn_torch.py

主实验：
    RNN Baseline：400 条
    RNN + EDA：6800 条
    验证集：100 条（Early Stopping）
    测试集：872 条（每次训练结束后最终评测一次）
    Seeds：42, 43, 44, 45, 46

关键约束：
    - 通过 A 的 data_utils.load_splits() 读取数据
    - 先 np.random.seed(42)，再 load_splits()，固定 OOV 初始化
    - 两组共用 A 的词表、embedding、验证集和测试集
    - GloVe embedding 冻结，不参与梯度更新
    - Adam lr=0.001，batch_size=32
    - Normal(0, 0.05)，bias=0
    - Early Stopping 监控 val_loss，patience=3，恢复最佳权重
"""

from __future__ import annotations

import os
import sys
import csv
import json
import copy
import random
import platform
from pathlib import Path
from typing import Any

# 必须在导入 data_utils 之前固定 NumPy 随机种子。
# A 的 load_glove() 会为 OOV 词生成随机向量。
import numpy as np

# 避免部分环境下线程数过多导致运行不稳定。
os.environ.setdefault("OMP_NUM_THREADS", "1")
# CUDA 确定性计算配置；必须在导入 torch 前设置。
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ------------------------------------------------------------
# 1. 固定实验配置
# ------------------------------------------------------------

SEEDS = [42, 43, 44, 45, 46]

MAX_LEN = 50
EMBEDDING_DIM = 300

BATCH_SIZE = 32
LEARNING_RATE = 0.001
MAX_EPOCHS = 30
PATIENCE = 3

LSTM_1_UNITS = 64
LSTM_2_UNITS = 32
DROPOUT_RATE = 0.5
DENSE_UNITS = 20

INIT_MEAN = 0.0
INIT_STD = 0.05

# 主实验必须冻结 GloVe。
FREEZE_EMBEDDING = True

# A 的接口使用这个种子生成 OOV 向量。
DATA_INITIALIZATION_SEED = 42

# ------------------------------------------------------------
# 2. 项目目录
# ------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent
os.chdir(ROOT_DIR)

RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# ------------------------------------------------------------
# 3. 导入 A 的接口
# ------------------------------------------------------------

from data_utils import load_splits


# ------------------------------------------------------------
# 4. 随机种子与设备
# ------------------------------------------------------------

def set_model_seed(seed: int) -> None:
    """固定每次模型训练的随机种子。"""
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # 尽量使用确定性算法，便于复现实验。
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------
# 5. 检查并读取 A 的数据
# ------------------------------------------------------------

def load_and_check_data() -> dict[str, Any]:
    print("\n" + "=" * 65)
    print("读取 A 模块数据")
    print("=" * 65)

    # 必须在 load_splits() 之前设置。
    # 这样 A 的 np.random.normal(0, 0.1, ...) 对 OOV 的初始化固定。
    np.random.seed(DATA_INITIALIZATION_SEED)
    random.seed(DATA_INITIALIZATION_SEED)

    data = load_splits()

    expected_shapes = {
        "baseline_train_x": (400, MAX_LEN),
        "baseline_train_y": (400,),
        "train_x": (6800, MAX_LEN),
        "train_y": (6800,),
        "val_x": (100, MAX_LEN),
        "val_y": (100,),
        "test_x": (872, MAX_LEN),
        "test_y": (872,),
        "embedding": (3748, EMBEDDING_DIM),
    }

    for key, expected_shape in expected_shapes.items():
        actual_shape = tuple(data[key].shape)
        print(f"{key:20s}: {actual_shape}")

        if actual_shape != expected_shape:
            raise ValueError(
                f"{key} shape 不匹配："
                f"期望 {expected_shape}，实际 {actual_shape}。"
                "请检查 A 交付的数据和 GloVe 文件，不要自行重新划分。"
            )

    # 检查标签。
    for key in [
        "baseline_train_y",
        "train_y",
        "val_y",
        "test_y",
    ]:
        unique_labels = set(np.unique(data[key]).tolist())
        if not unique_labels.issubset({0, 1}):
            raise ValueError(
                f"{key} 中存在非 0/1 标签：{unique_labels}"
            )

    if len(data["test_texts"]) != 872:
        raise ValueError("test_texts 数量不是 872。")

    embedding = np.asarray(data["embedding"], dtype=np.float32)

    if not np.isfinite(embedding).all():
        raise ValueError("embedding 中出现 NaN 或 Inf。")

    # PAD=0 必须是零向量。
    if not np.allclose(embedding[0], 0.0):
        raise ValueError("A 的 PAD embedding（索引 0）不是零向量。")

    print("\n数据检查通过。")
    print("词表大小：", len(data["vocab"]))
    print("设备：", DEVICE)

    return data


# ------------------------------------------------------------
# 6. PyTorch 模型：论文 9.1 节 RNN
# ------------------------------------------------------------

class RNNClassifier(nn.Module):
    """
    网络：
        Embedding(vocab_size, 300)，使用 A 的 GloVe，冻结
        -> Bidirectional LSTM(64)
        -> Dropout(0.5)
        -> Bidirectional LSTM(32)
        -> Dropout(0.5)
        -> Linear(64, 20) + ReLU
        -> Linear(20, 1)

    forward 返回 logits。
    训练使用 BCEWithLogitsLoss；
    推理时使用 sigmoid 得到 prob_1。
    """

    def __init__(self, embedding_matrix: np.ndarray):
        super().__init__()

        embedding_tensor = torch.as_tensor(
            embedding_matrix,
            dtype=torch.float32
        )

        vocab_size, embedding_dim = embedding_tensor.shape

        if embedding_dim != EMBEDDING_DIM:
            raise ValueError(
                f"Embedding 维度必须是 {EMBEDDING_DIM}，"
                f"实际是 {embedding_dim}"
            )

        # from_pretrained + freeze=True：词向量不更新。
        self.embedding = nn.Embedding.from_pretrained(
            embeddings=embedding_tensor,
            freeze=FREEZE_EMBEDDING,
            padding_idx=0
        )

        # 第一层双向 LSTM。
        self.bilstm1 = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=LSTM_1_UNITS,
            batch_first=True,
            bidirectional=True
        )

        self.dropout1 = nn.Dropout(DROPOUT_RATE)

        # 第一层输出维度为 64 * 2 = 128。
        self.bilstm2 = nn.LSTM(
            input_size=LSTM_1_UNITS * 2,
            hidden_size=LSTM_2_UNITS,
            batch_first=True,
            bidirectional=True
        )

        self.dropout2 = nn.Dropout(DROPOUT_RATE)

        # 第二层双向输出维度为 32 * 2 = 64。
        self.fc1 = nn.Linear(LSTM_2_UNITS * 2, DENSE_UNITS)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(DENSE_UNITS, 1)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """
        所有可训练 LSTM / Linear 权重：
            Normal(mean=0, std=0.05)
        所有 bias：
            0

        GloVe embedding 使用 A 提供的矩阵，不重新初始化。
        """
        for name, parameter in self.named_parameters():
            if name.startswith("embedding."):
                continue

            if "weight" in name:
                nn.init.normal_(
                    parameter,
                    mean=INIT_MEAN,
                    std=INIT_STD
                )
            elif "bias" in name:
                nn.init.zeros_(parameter)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        token_ids: [batch, 50]
        返回 logits: [batch]

        使用 pack_padded_sequence 忽略 PAD 位置，
        避免双向 LSTM 把 padding 当作有效文本。
        """
        # 每条句子的真实长度（不包括 PAD）。
        lengths = (token_ids != 0).sum(dim=1)

        # pack_padded_sequence 要求长度至少为 1。
        lengths = lengths.clamp(min=1)

        embedded = self.embedding(token_ids)

        # 第一层 BiLSTM。
        packed_input = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        packed_output1, _ = self.bilstm1(packed_input)

        output1, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output1,
            batch_first=True,
            total_length=token_ids.size(1)
        )

        output1 = self.dropout1(output1)

        # 第二层 BiLSTM。
        packed_input2 = nn.utils.rnn.pack_padded_sequence(
            output1,
            lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        _, (hidden2, _) = self.bilstm2(packed_input2)

        # 双向 LSTM 最后一层：
        # hidden2[-2] = forward，hidden2[-1] = backward。
        representation = torch.cat(
            [hidden2[-2], hidden2[-1]],
            dim=1
        )

        representation = self.dropout2(representation)

        x = self.fc1(representation)
        x = self.relu(x)
        logits = self.fc2(x).squeeze(1)

        return logits


# ------------------------------------------------------------
# 7. DataLoader
# ------------------------------------------------------------

def make_loader(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool = False,
    seed: int | None = None
) -> DataLoader:

    x_tensor = torch.as_tensor(x, dtype=torch.long)
    y_tensor = torch.as_tensor(y, dtype=torch.float32)

    dataset = TensorDataset(x_tensor, y_tensor)

    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
        drop_last=False
    )


# ------------------------------------------------------------
# 8. 一个 epoch 的训练
# ------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module
) -> tuple[float, float]:

    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(DEVICE, non_blocking=True)
        batch_y = batch_y.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model(batch_x)
        loss = criterion(logits, batch_y)

        loss.backward()
        optimizer.step()

        batch_size = batch_y.size(0)

        total_loss += loss.item() * batch_size

        predictions = (logits >= 0).long()
        total_correct += (predictions == batch_y.long()).sum().item()
        total_samples += batch_size

    average_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return average_loss, accuracy


# ------------------------------------------------------------
# 9. 验证 / 最终测试
# ------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module
) -> tuple[float, float, np.ndarray, np.ndarray]:

    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    all_logits = []
    all_labels = []

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(DEVICE, non_blocking=True)
        batch_y = batch_y.to(DEVICE, non_blocking=True)

        logits = model(batch_x)
        loss = criterion(logits, batch_y)

        batch_size = batch_y.size(0)

        total_loss += loss.item() * batch_size

        predictions = (logits >= 0).long()
        total_correct += (predictions == batch_y.long()).sum().item()
        total_samples += batch_size

        all_logits.append(logits.detach().cpu().numpy())
        all_labels.append(batch_y.detach().cpu().numpy())

    average_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    logits_array = np.concatenate(all_logits)
    labels_array = np.concatenate(all_labels).astype(np.int64)

    return average_loss, accuracy, logits_array, labels_array


# ------------------------------------------------------------
# 10. Early Stopping：监控 val_loss，恢复最佳权重
# ------------------------------------------------------------

class EarlyStopping:
    def __init__(self, patience: int = 3):
        self.patience = patience
        self.best_loss = float("inf")
        self.best_epoch = 0
        self.bad_epochs = 0
        self.best_state = None

    def update(
        self,
        val_loss: float,
        epoch: int,
        model: nn.Module
    ) -> bool:
        """
        返回 True 表示应该停止。
        val_loss 严格下降时保存最佳权重。
        """
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.bad_epochs = 0

            self.best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            self.bad_epochs += 1

        return self.bad_epochs >= self.patience

    def restore_best_weights(self, model: nn.Module) -> None:
        if self.best_state is None:
            raise RuntimeError("没有保存到最佳模型权重。")

        model.load_state_dict(self.best_state)


# ------------------------------------------------------------
# 11. 保存训练历史和曲线
# ------------------------------------------------------------

def save_history(
    history: dict[str, list[float]],
    output_dir: Path
) -> None:

    csv_path = output_dir / "training_history.csv"

    keys = [
        "loss",
        "accuracy",
        "val_loss",
        "val_accuracy"
    ]

    epoch_count = len(history["loss"])

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["epoch"] + keys)

        for index in range(epoch_count):
            row = [index + 1]
            for key in keys:
                row.append(history[key][index])
            writer.writerow(row)

    # matplotlib 为可选依赖；没有安装时仍保存完整 CSV 日志。
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].plot(history["loss"], label="Train Loss")
        axes[0].plot(history["val_loss"], label="Validation Loss")
        axes[0].set_title("Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(history["accuracy"], label="Train Accuracy")
        axes[1].plot(history["val_accuracy"], label="Validation Accuracy")
        axes[1].set_title("Accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(output_dir / "training_curves.png", dpi=200)
        plt.close(fig)

    except ImportError:
        print("未安装 matplotlib，已保存 CSV 日志，跳过曲线图片。")


# ------------------------------------------------------------
# 12. 保存逐条预测 CSV
# ------------------------------------------------------------

def save_predictions(
    output_path: Path,
    texts: list[str],
    gold_labels: np.ndarray,
    logits: np.ndarray
) -> tuple[float, float]:

    # 单输出 Sigmoid：prob_1 = sigmoid(logit)
    # prob_0 = 1 - prob_1
    logits = np.asarray(logits).reshape(-1)
    prob_1 = 1.0 / (1.0 + np.exp(-np.clip(logits, -80, 80)))
    prob_0 = 1.0 - prob_1

    pred_labels = (prob_1 >= 0.5).astype(np.int64)
    gold_labels = np.asarray(gold_labels).astype(np.int64)

    if not (
        len(texts)
        == len(gold_labels)
        == len(pred_labels)
        == len(prob_1)
    ):
        raise ValueError("预测数量、文本数量和标签数量不一致。")

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:
        writer = csv.writer(f)

        writer.writerow([
            "id",
            "text",
            "gold_label",
            "pred_label",
            "prob_0",
            "prob_1"
        ])

        for i in range(len(texts)):
            writer.writerow([
                i,
                texts[i],
                int(gold_labels[i]),
                int(pred_labels[i]),
                f"{prob_0[i]:.6f}",
                f"{prob_1[i]:.6f}"
            ])

    accuracy = float(np.mean(pred_labels == gold_labels))

    # BCEWithLogits 对应的二分类交叉熵，使用稳定的 numpy 表达式。
    y = gold_labels.astype(np.float64)
    z = logits.astype(np.float64)
    test_loss = float(
        np.mean(np.logaddexp(0.0, z) - y * z)
    )

    return accuracy, test_loss


# ------------------------------------------------------------
# 13. 单次实验
# ------------------------------------------------------------

def run_experiment(
    experiment_name: str,
    seed: int,
    train_x: np.ndarray,
    train_y: np.ndarray,
    data: dict[str, Any]
) -> dict[str, Any]:

    print("\n" + "#" * 70)
    print(f"实验：{experiment_name} | Seed={seed}")
    print("#" * 70)

    output_dir = RESULTS_DIR / experiment_name / f"seed{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 每次实验重新初始化相同结构的模型。
    set_model_seed(seed)

    embedding_matrix = np.asarray(
        data["embedding"],
        dtype=np.float32
    )

    train_loader = make_loader(
        train_x,
        train_y,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=seed
    )

    val_loader = make_loader(
        data["val_x"],
        data["val_y"],
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    test_loader = make_loader(
        data["test_x"],
        data["test_y"],
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model = RNNClassifier(embedding_matrix).to(DEVICE)

    # 确认 Embedding 确实冻结。
    if model.embedding.weight.requires_grad:
        raise RuntimeError("Embedding 没有冻结，请检查模型配置。")

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE
    )

    early_stopping = EarlyStopping(patience=PATIENCE)

    history = {
        "loss": [],
        "accuracy": [],
        "val_loss": [],
        "val_accuracy": []
    }

    print(f"训练样本：{len(train_x)}")
    print(f"验证样本：{len(data['val_x'])}")
    print(f"测试样本：{len(data['test_x'])}")
    print(f"Embedding 冻结：{not model.embedding.weight.requires_grad}")
    print(f"设备：{DEVICE}")
    print(f"Batch size：{BATCH_SIZE}")
    print(f"Learning rate：{LEARNING_RATE}")
    print(f"Max epochs：{MAX_EPOCHS}")
    print(f"Patience：{PATIENCE}")

    # ------------------------
    # 训练与验证
    # ------------------------

    for epoch in range(1, MAX_EPOCHS + 1):

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion
        )

        val_loss, val_acc, _, _ = evaluate(
            model,
            val_loader,
            criterion
        )

        history["loss"].append(train_loss)
        history["accuracy"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        improved = val_loss < early_stopping.best_loss

        should_stop = early_stopping.update(
            val_loss,
            epoch,
            model
        )

        marker = "  * best" if improved else ""

        print(
            f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
            f"loss={train_loss:.4f} | "
            f"acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f}"
            f"{marker}"
        )

        if should_stop:
            print(
                f"Early Stopping：连续 {PATIENCE} 个 epoch "
                f"验证损失没有改善，在 epoch {epoch} 停止。"
            )
            break

    epochs_ran = len(history["loss"])
    early_stopped = epochs_ran < MAX_EPOCHS

    # ------------------------
    # 恢复最佳权重
    # ------------------------

    early_stopping.restore_best_weights(model)

    best_epoch = early_stopping.best_epoch
    best_val_loss = early_stopping.best_loss

    # 保存恢复后的最佳模型。
    checkpoint_path = output_dir / "best_model.pt"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "seed": seed,
            "experiment": experiment_name,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "embedding_trainable": False,
        },
        checkpoint_path
    )

    # ------------------------
    # 最终测试集评测
    # 训练完成后只在这里评测一次，不参与早停和调参。
    # ------------------------

    test_loss, test_accuracy, test_logits, test_labels = evaluate(
        model,
        test_loader,
        criterion
    )

    prediction_filename = (
        f"predictions_{experiment_name}_seed{seed}.csv"
    )
    prediction_path = RESULTS_DIR / prediction_filename

    # 使用同一次 test 预测结果生成逐条 CSV。
    csv_accuracy, csv_test_loss = save_predictions(
        prediction_path,
        data["test_texts"],
        test_labels,
        test_logits
    )

    if not np.isclose(test_accuracy, csv_accuracy):
        raise RuntimeError("测试准确率与预测 CSV 计算结果不一致。")

    # 保存训练历史。
    save_history(history, output_dir)

    # 保存模型结构说明。
    model_summary = {
        "architecture": [
            "Embedding(vocab_size, 300), pretrained GloVe, frozen",
            "Bidirectional(LSTM(64), return_sequences=True)",
            "Dropout(0.5)",
            "Bidirectional(LSTM(32))",
            "Dropout(0.5)",
            "Linear(64, 20)",
            "ReLU",
            "Linear(20, 1)",
            "Sigmoid at inference"
        ],
        "loss": "BCEWithLogitsLoss",
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "early_stopping": {
            "monitor": "val_loss",
            "patience": PATIENCE,
            "restore_best_weights": True
        },
        "initialization": {
            "weights": "Normal",
            "mean": INIT_MEAN,
            "std": INIT_STD,
            "bias": 0
        },
        "embedding_frozen": True,
        "seed": seed
    }

    with open(
        output_dir / "model_config.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            model_summary,
            f,
            ensure_ascii=False,
            indent=4
        )

    result = {
        "experiment": experiment_name,
        "seed": seed,
        "train_samples": int(len(train_x)),
        "epochs_ran": int(epochs_ran),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "early_stopped": bool(early_stopped),
        "test_accuracy": float(test_accuracy),
        "test_loss": float(test_loss),
        "prediction_file": str(
            prediction_path.relative_to(ROOT_DIR)
        ),
        "checkpoint_file": str(
            checkpoint_path.relative_to(ROOT_DIR)
        )
    }

    with open(
        output_dir / "experiment_result.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

    print("\n本次实验完成：")
    print(f"Best epoch：{best_epoch}")
    print(f"Best val loss：{best_val_loss:.6f}")
    print(f"Test accuracy：{test_accuracy:.4%}")
    print(f"Test loss：{test_loss:.6f}")
    print(f"Prediction：{prediction_path}")

    return result


# ------------------------------------------------------------
# 14. 保存五个种子的汇总结果
# ------------------------------------------------------------

def save_all_results(results: list[dict[str, Any]]) -> None:

    summary_path = RESULTS_DIR / "rnn_all_seeds_summary.csv"

    columns = [
        "experiment",
        "seed",
        "train_samples",
        "epochs_ran",
        "best_epoch",
        "best_val_loss",
        "early_stopped",
        "test_accuracy",
        "test_loss",
        "prediction_file",
        "checkpoint_file"
    ]

    with open(
        summary_path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(results)

    # 每组的 5 个 seed 均值和标准差。
    aggregate_path = RESULTS_DIR / "rnn_mean_std_summary.csv"

    with open(
        aggregate_path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:
        writer = csv.writer(f)
        writer.writerow([
            "experiment",
            "num_seeds",
            "mean_test_accuracy",
            "std_test_accuracy",
            "mean_test_loss",
            "std_test_loss"
        ])

        for experiment_name in ["rnn_baseline", "rnn_eda"]:
            subset = [
                item for item in results
                if item["experiment"] == experiment_name
            ]

            accuracies = np.asarray(
                [item["test_accuracy"] for item in subset],
                dtype=np.float64
            )

            losses = np.asarray(
                [item["test_loss"] for item in subset],
                dtype=np.float64
            )

            writer.writerow([
                experiment_name,
                len(subset),
                float(accuracies.mean()),
                float(accuracies.std(ddof=1)) if len(accuracies) > 1 else 0.0,
                float(losses.mean()),
                float(losses.std(ddof=1)) if len(losses) > 1 else 0.0
            ])

    print("\n" + "=" * 70)
    print("五个随机种子的汇总")
    print("=" * 70)

    for experiment_name in ["rnn_baseline", "rnn_eda"]:
        subset = [
            item for item in results
            if item["experiment"] == experiment_name
        ]

        accuracies = np.asarray(
            [item["test_accuracy"] for item in subset],
            dtype=np.float64
        )

        print(
            f"{experiment_name}: "
            f"{accuracies.mean():.4%} ± "
            f"{accuracies.std(ddof=1):.4%}"
        )

    print("\n详细结果：", summary_path)
    print("均值标准差：", aggregate_path)


# ------------------------------------------------------------
# 15. 主程序：两组 × 五个种子，共十次训练
# ------------------------------------------------------------

def main():

    print("=" * 70)
    print("C 模块：PyTorch RNN 主实验")
    print("=" * 70)

    print("Python:", platform.python_version())
    print("PyTorch:", torch.__version__)
    print("NumPy:", np.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("Device:", DEVICE)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    # 版本提示：不因补丁版本字符串差异而强制中断。
    if not torch.__version__.startswith("2.7.0"):
        print(
            "警告：当前 PyTorch 版本与约定的 2.7.0 不完全一致，"
            "请确认环境。"
        )

    # 只加载一次 A 的数据：
    # 先 np.random.seed(42)，确保 OOV 初始化固定；
    # 两组和所有模型种子共享同一 embedding 矩阵。
    data = load_and_check_data()

    all_results = []

    for seed in SEEDS:

        print("\n\n" + "=" * 70)
        print(f"随机种子 {seed}：RNN Baseline")
        print("=" * 70)

        baseline_result = run_experiment(
            experiment_name="rnn_baseline",
            seed=seed,
            train_x=data["baseline_train_x"],
            train_y=data["baseline_train_y"],
            data=data
        )

        all_results.append(baseline_result)

        print("\n\n" + "=" * 70)
        print(f"随机种子 {seed}：RNN + EDA")
        print("=" * 70)

        eda_result = run_experiment(
            experiment_name="rnn_eda",
            seed=seed,
            train_x=data["train_x"],
            train_y=data["train_y"],
            data=data
        )

        all_results.append(eda_result)

        # 每完成一个 seed，立即更新汇总表，避免中途停止后没有总记录。
        save_all_results(all_results)

    print("\n" + "=" * 70)
    print("C 模块主实验全部完成：2 组 × 5 个种子 = 10 次训练")
    print("=" * 70)

    save_all_results(all_results)

    print("\n最终请将整个 results 文件夹交给 D 同学。")


if __name__ == "__main__":
    main()
