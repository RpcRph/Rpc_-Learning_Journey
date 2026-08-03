"""Shared plotting helpers.

The Agg backend saves figures without opening a GUI window, so the same code
works in VS Code's terminal, a CI runner, or a remote server.
公共绘图模块
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

# 使用agg后端保存图形，而不打开GUI窗口，因此相同的代码可以在VS Code的终端、CI运行器或远程服务器上运行。
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure


COLORS = ["#2563eb", "#f97316", "#16a34a", "#dc2626", "#7c3aed", "#0891b2"]


def configure_matplotlib() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 180,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "figure.facecolor": "white",
        }
    )


def save_figure(fig: Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def draw_confusion_matrix(
    ax: Axes,
    matrix: np.ndarray,
    title: str,
    labels: list[str] | None = None,
) -> None:
    labels = labels or [str(index) for index in range(matrix.shape[0])]
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    threshold = matrix.max() / 2
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )
    ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
