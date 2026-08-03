"""深度学习实验：MLP、CNN 和缩放点积自注意力机制   使用 PyTorch 实现
加载一批数据
→ 前向传播
→ 计算交叉熵损失
→ loss.backward() 反向传播
→ optimizer.step() 更新参数
→ 在测试集计算准确率

"""

from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from .plotting import COLORS, configure_matplotlib, draw_confusion_matrix, save_figure

RANDOM_STATE = 42


def set_random_seed(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("指定了 CUDA，但当前 PyTorch 未检测到可用 GPU")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 多层感知机
class MLPClassifier(nn.Module):
    """Two-layer perceptron: 64 -> 128 -> 64 -> 10."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)


# 卷积神经网络
class CNNClassifier(nn.Module):
    """Small CNN adapted to the 8x8 handwritten-digits dataset."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 2 * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(self.features(x))


# 自注意力分类器
class SelfAttentionClassifier(nn.Module):
    """Classify an image by treating its eight rows as a token sequence.

    The forward pass explicitly implements
    softmax(QK^T / sqrt(d_k))V, matching the formula in assignment two.
    """

    def __init__(self, d_model: int = 32) -> None:
        super().__init__()
        self.input_projection = nn.Linear(8, d_model)
        self.position_embedding = nn.Parameter(torch.zeros(1, 8, d_model))
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.output_projection = nn.Linear(d_model, d_model)
        self.normalization = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.ReLU(),
            nn.Linear(2 * d_model, d_model),
        )
        self.feed_forward_normalization = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8 * d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )
        self.last_attention: Tensor | None = None
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        tokens = x.reshape(-1, 8, 8)
        encoded = self.input_projection(tokens) + self.position_embedding
        queries = self.query(encoded)
        keys = self.key(encoded)
        values = self.value(encoded)
        scores = queries @ keys.transpose(-2, -1) / math.sqrt(queries.shape[-1])
        attention = torch.softmax(scores, dim=-1)
        self.last_attention = attention.detach()
        context = attention @ values
        encoded = self.normalization(encoded + self.output_projection(context))
        encoded = self.feed_forward_normalization(encoded + self.feed_forward(encoded))
        return self.classifier(encoded)


@dataclass
class TrainingHistory:
    train_loss: list[float]
    test_accuracy: list[float]


def _load_data(
    batch_size: int,
) -> tuple[DataLoader, DataLoader, np.ndarray, np.ndarray]:
    digits = load_digits()
    features = (digits.images.astype(np.float32) / 16.0)[:, None, :, :]
    targets = digits.target.astype(np.int64)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        targets,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=targets,
    )
    generator = torch.Generator().manual_seed(RANDOM_STATE)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    return train_loader, test_loader, x_test, y_test


@torch.no_grad()
def _predict(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    actual: list[np.ndarray] = []
    predicted: list[np.ndarray] = []
    for features, targets in loader:
        logits = model(features.to(device))
        actual.append(targets.numpy())
        predicted.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(actual), np.concatenate(predicted)


def _train(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
) -> TrainingHistory:
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    history = TrainingHistory(train_loss=[], test_accuracy=[])

    for _ in range(epochs):
        model.train()
        running_loss = 0.0
        sample_count = 0
        for features, targets in train_loader:
            features = features.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * len(features)
            sample_count += len(features)

        actual, predicted = _predict(model, test_loader, device)
        history.train_loss.append(running_loss / sample_count)
        history.test_accuracy.append(float(accuracy_score(actual, predicted)))
    return history


def _parameter_count(model: nn.Module) -> int:
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )


def run_deep_learning_experiments(
    figures_dir: Path,
    epochs: int = 15,
    device_name: str = "auto",
) -> dict[str, dict[str, Any]]:
    """Train all three neural architectures and save learning diagnostics."""
    if epochs < 1:
        raise ValueError("epochs 必须大于等于 1")
    configure_matplotlib()
    set_random_seed()
    device = resolve_device(device_name)
    train_loader, test_loader, x_test, y_test = _load_data(batch_size=64)
    models: dict[str, nn.Module] = {
        "MLP": MLPClassifier(),
        "CNN": CNNClassifier(),
        "Self-Attention": SelfAttentionClassifier(),
    }
    learning_rates = {"MLP": 1e-3, "CNN": 2e-3, "Self-Attention": 1e-3}
    histories: dict[str, TrainingHistory] = {}
    predictions: dict[str, np.ndarray] = {}
    results: dict[str, dict[str, Any]] = {}

    for name, model in models.items():
        history = _train(
            model,
            train_loader,
            test_loader,
            device,
            epochs,
            learning_rate=learning_rates[name],
        )
        actual, predicted = _predict(model, test_loader, device)
        histories[name] = history
        predictions[name] = predicted
        results[name] = {
            "task": "deep_learning_classification",
            "accuracy": round(float(accuracy_score(actual, predicted)), 6),
            "macro_f1": round(float(f1_score(actual, predicted, average="macro")), 6),
            "parameters": _parameter_count(model),
            "epochs": epochs,
            "device": str(device),
        }

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    epoch_axis = np.arange(1, epochs + 1)
    for color, (name, history) in zip(COLORS, histories.items(), strict=False):
        axes[0].plot(epoch_axis, history.train_loss, label=name, color=color)
        axes[1].plot(epoch_axis, history.test_accuracy, label=name, color=color)
    axes[0].set_title("Deep-learning training loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[1].set_title("Test accuracy after each epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.5, 1.0)
    for ax in axes:
        ax.legend()
    save_figure(fig, figures_dir / "07_deep_learning_curves.png")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (name, predicted) in zip(axes, predictions.items(), strict=True):
        draw_confusion_matrix(
            ax,
            confusion_matrix(y_test, predicted),
            f"{name} confusion matrix",
            labels=[str(number) for number in range(10)],
        )
    save_figure(fig, figures_dir / "08_deep_learning_confusion_matrices.png")

    cnn = models["CNN"]
    sample_tensor = torch.from_numpy(x_test[:10]).to(device)
    with torch.no_grad():
        sample_predictions = cnn(sample_tensor).argmax(dim=1).cpu().numpy()
    fig, axes = plt.subplots(2, 5, figsize=(10, 4.5))
    for index, ax in enumerate(axes.flat):
        ax.imshow(x_test[index, 0], cmap="gray_r")
        color = "#15803d" if sample_predictions[index] == y_test[index] else "#dc2626"
        ax.set_title(
            f"true={y_test[index]}, pred={sample_predictions[index]}",
            color=color,
            fontsize=9,
        )
        ax.axis("off")
    fig.suptitle("CNN sample predictions", fontsize=13)
    save_figure(fig, figures_dir / "09_cnn_sample_predictions.png")

    attention_model = models["Self-Attention"]
    with torch.no_grad():
        attention_prediction = (
            attention_model(torch.from_numpy(x_test[:1]).to(device))
            .argmax(dim=1)
            .cpu()
            .item()
        )
    assert isinstance(attention_model, SelfAttentionClassifier)
    assert attention_model.last_attention is not None
    attention_matrix = attention_model.last_attention[0].cpu().numpy()
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    axes[0].imshow(x_test[0, 0], cmap="gray_r")
    axes[0].set_title(f"Input digit: true={y_test[0]}, pred={attention_prediction}")
    axes[0].axis("off")
    image = axes[1].imshow(attention_matrix, cmap="magma", vmin=0.0)
    axes[1].set_title("Self-attention weights between image rows")
    axes[1].set_xlabel("Key row")
    axes[1].set_ylabel("Query row")
    axes[1].set_xticks(range(8))
    axes[1].set_yticks(range(8))
    fig.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04)
    save_figure(fig, figures_dir / "10_self_attention_heatmap.png")
    return results
