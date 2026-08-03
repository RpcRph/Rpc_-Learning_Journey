"""Command-line entry point for all assignment-three experiments.
命令行入口点，运行作业三中的所有实验

"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import torch

from .classical import run_classical_experiments
from .deep_learning import run_deep_learning_experiments
from .plotting import COLORS, configure_matplotlib, save_figure

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _write_json(metrics: dict[str, dict[str, Any]], output_dir: Path) -> None:
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _write_csv(metrics: dict[str, dict[str, Any]], output_dir: Path) -> None:
    field_names = sorted(
        {"model"} | {key for values in metrics.values() for key in values}
    )
    with (output_dir / "metrics.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=field_names)
        writer.writeheader()
        for model, values in metrics.items():
            row = {"model": model}
            row.update(
                {
                    key: json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (list, dict))
                    else value
                    for key, value in values.items()
                }
            )
            writer.writerow(row)


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, list):
        return ", ".join(_format_value(item) for item in value)
    return str(value)


def _markdown_table(
    metrics: dict[str, dict[str, Any]], names: list[str], columns: list[str]
) -> str:
    header = "| 模型 | " + " | ".join(columns) + " |"
    divider = "|---|" + "|".join("---:" for _ in columns) + "|"
    rows = [header, divider]
    for name in names:
        values = metrics[name]
        rows.append(
            "| "
            + name
            + " | "
            + " | ".join(_format_value(values.get(column, "—")) for column in columns)
            + " |"
        )
    return "\n".join(rows)


def _write_summary_figure(
    metrics: dict[str, dict[str, Any]], figures_dir: Path
) -> None:
    configure_matplotlib()
    classical_names = [
        name
        for name, values in metrics.items()
        if values.get("task") == "classification"
    ]
    cluster_names = [
        name for name, values in metrics.items() if values.get("task") == "clustering"
    ]
    deep_names = [
        name
        for name, values in metrics.items()
        if values.get("task") == "deep_learning_classification"
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    linear_r2 = metrics["Linear Regression (NumPy)"]["r2"]
    axes[0, 0].bar(["Linear Regression"], [linear_r2], color=COLORS[0], width=0.5)
    axes[0, 0].set_ylim(0.0, 1.0)
    axes[0, 0].set_title("Regression: R-squared")
    axes[0, 0].bar_label(axes[0, 0].containers[0], fmt="%.4f")

    axes[0, 1].barh(
        [name.replace(" (NumPy)", "") for name in classical_names],
        [metrics[name]["f1"] for name in classical_names],
        color=COLORS[1],
    )
    axes[0, 1].set_xlim(0.75, 1.0)
    axes[0, 1].set_title("Classical classification: F1")

    axes[1, 0].bar(
        [name.replace(" (NumPy)", "") for name in cluster_names],
        [metrics[name]["silhouette"] for name in cluster_names],
        color=COLORS[2],
    )
    axes[1, 0].set_ylim(0.0, 1.0)
    axes[1, 0].set_title("Clustering: silhouette score")

    if deep_names:
        axes[1, 1].bar(
            deep_names,
            [metrics[name]["accuracy"] for name in deep_names],
            color=COLORS[4],
        )
        axes[1, 1].set_ylim(0.0, 1.0)
        axes[1, 1].set_title("Deep learning: test accuracy")
    else:
        axes[1, 1].text(
            0.5,
            0.5,
            "Deep learning skipped",
            ha="center",
            va="center",
            transform=axes[1, 1].transAxes,
        )
        axes[1, 1].set_axis_off()
    save_figure(fig, figures_dir / "11_result_summary.png")


def _write_report(
    metrics: dict[str, dict[str, Any]],
    output_dir: Path,
    epochs: int,
    deep_learning_ran: bool,
) -> None:
    classical_names = [
        "Logistic Regression (NumPy)",
        "Decision Tree",
        "Gaussian Naive Bayes (NumPy)",
        "SVM (RBF kernel)",
        "Random Forest",
        "Gradient Boosting",
    ]
    clustering_names = ["K-Means (NumPy)", "Hierarchical Clustering"]
    deep_names = ["MLP", "CNN", "Self-Attention"]
    best_classical = max(classical_names, key=lambda name: metrics[name]["f1"])
    deep_observation = ""
    deep_table = "本次使用 `--skip-deep-learning` 跳过了神经网络训练。"
    if deep_learning_ran:
        best_deep = max(deep_names, key=lambda name: metrics[name]["accuracy"])
        deep_observation = (
            f"- 三个深度学习模型中，当前设置下 `{best_deep}` 的测试准确率最高，"
            f"为 {metrics[best_deep]['accuracy']:.4f}。"
        )
        deep_table = _markdown_table(
            metrics,
            deep_names,
            ["accuracy", "macro_f1", "parameters", "epochs", "device"],
        )

    report = f"""# 机器学习入门代码实践：运行结果

> 本文件由 `uv run ml-practice` 自动生成。生成时间：
> {datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")}。

## 实验目标与流程

我按照第二部分理论报告中的模型顺序组织实验。每组实验都遵循“准备数据 → 划分或预处理
→ 训练模型 → 在未参与训练的数据上评价 → 使用 Matplotlib 保存图片”的流程。经典算法采用
scikit-learn 自带小型数据集或固定随机种子的合成数据，深度学习采用 8×8 手写数字数据集，
因此运行时不需要额外下载数据。

本次运行环境：

- Python：{platform.python_version()}
- 操作系统：{platform.platform()}
- NumPy：{np.__version__}
- scikit-learn：{sklearn.__version__}
- Matplotlib：{matplotlib.__version__}
- PyTorch：{torch.__version__}
- 深度学习轮数：{epochs if deep_learning_ran else "已跳过"}

## 实现范围

| 知识库部分 | 模型或方法 | 本工程中的实现 |
|---|---|---|
| Machine Learning 1：Supervised Learning | 线性回归、逻辑回归、决策树、朴素贝叶斯 | 线性/逻辑/高斯朴素贝叶斯使用 NumPy 手写，决策树使用 scikit-learn |
| Machine Learning 2：Advanced / Unsupervised | SVM、随机森林、梯度提升、K-Means、层次聚类、PCA | K-Means/PCA 使用 NumPy 手写，其余使用 scikit-learn / SciPy |
| Machine Learning 3：Optimization & Evaluation | 特征缩放、交叉验证、超参数搜索、统一指标 | `Pipeline`、`StratifiedKFold`、`GridSearchCV` 与多种评价指标 |
| Deep Learning 1/2/3 | MLP、CNN、自注意力 | 使用 PyTorch 搭建并训练；自注意力显式实现缩放点积公式 |

## 结果统计

### 线性回归

{_markdown_table(metrics, ["Linear Regression (NumPy)"], ["mse", "rmse", "mae", "r2", "iterations"])}

![线性回归拟合与收敛](results/figures/01_linear_regression.png)

### 监督学习与进阶监督学习

{_markdown_table(metrics, classical_names, ["accuracy", "precision", "recall", "f1", "roc_auc"])}

![监督分类模型对比](results/figures/02_supervised_model_comparison.png)

![决策树结构](results/figures/03_decision_tree_structure.png)

![随机森林与梯度提升](results/figures/04_ensemble_models.png)

### 优化与评价

{_markdown_table(metrics, ["SVM Cross Validation"], ["fold_f1", "mean_f1", "std_f1"])}

{_markdown_table(metrics, ["SVM Grid Search"], ["best_cv_f1", "best_C", "best_gamma"])}

![交叉验证与参数搜索](results/figures/05_cross_validation_and_tuning.png)

### 无监督学习

{_markdown_table(metrics, clustering_names, ["silhouette", "adjusted_rand_index", "inertia", "iterations"])}

{_markdown_table(metrics, ["PCA (NumPy)"], ["explained_variance_pc1", "explained_variance_pc2", "cumulative_explained_variance", "reconstruction_mse"])}

![聚类与 PCA](results/figures/06_unsupervised_learning.png)

### 深度学习

{deep_table}

{"![深度学习训练曲线](results/figures/07_deep_learning_curves.png)" if deep_learning_ran else ""}

{"![深度学习混淆矩阵](results/figures/08_deep_learning_confusion_matrices.png)" if deep_learning_ran else ""}

{"![CNN 样本预测](results/figures/09_cnn_sample_predictions.png)" if deep_learning_ran else ""}

{"![自注意力权重热力图](results/figures/10_self_attention_heatmap.png)" if deep_learning_ran else ""}

### 总览

不同任务的指标含义不同，因此总览图分面展示回归 R²、传统分类 F1、聚类轮廓系数和
深度学习准确率，不把它们错误地合并成同一排名。

![结果统计总览](results/figures/11_result_summary.png)

## 结果分析

- 线性回归的测试集 R² 为 {metrics["Linear Regression (NumPy)"]["r2"]:.4f}，
  同时损失曲线逐步下降，说明梯度下降在当前学习率下已经收敛。
- 六个传统分类模型中，`{best_classical}` 的 F1 最高，为
  {metrics[best_classical]["f1"]:.4f}。单次划分结果不等同于模型的一般排名，
  因此工程又使用分层五折交叉验证报告均值和波动。
- K-Means 与层次聚类的轮廓系数分别为
  {metrics["K-Means (NumPy)"]["silhouette"]:.4f} 和
  {metrics["Hierarchical Clustering"]["silhouette"]:.4f}；树状图额外保留了样本合并层次。
- PCA 的前两个主成分累计解释方差比例为
  {metrics["PCA (NumPy)"]["cumulative_explained_variance"]:.4f}。二维图便于观察结构，
  但降维也会损失一部分信息，重建误差对此进行了量化。
{deep_observation}

## 可复现性说明

所有随机过程都使用固定种子 42。`metrics.json` 保存结构化结果，`metrics.csv` 便于在表格软件
中继续分析；重新执行完整命令会更新本文件和全部图片。
"""
    (PROJECT_ROOT / "RESULTS.md").write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="运行作业三中的经典机器学习和深度学习实验"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="MLP、CNN、自注意力的训练轮数（默认 15）",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="PyTorch 训练设备（默认自动选择）",
    )
    parser.add_argument(
        "--skip-deep-learning",
        action="store_true",
        help="只运行经典机器学习，用于快速检查环境",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results",
        help="指标与图片输出目录",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("[1/4] 运行经典机器学习实验……")
    metrics = run_classical_experiments(figures_dir)
    if args.skip_deep_learning:
        print("[2/4] 已按参数跳过深度学习。")
    else:
        print("[2/4] 训练 MLP、CNN 和 Self-Attention……")
        metrics.update(
            run_deep_learning_experiments(
                figures_dir,
                epochs=args.epochs,
                device_name=args.device,
            )
        )
    print("[3/4] 汇总指标和统计图……")
    _write_summary_figure(metrics, figures_dir)
    _write_json(metrics, output_dir)
    _write_csv(metrics, output_dir)
    _write_report(metrics, output_dir, args.epochs, not args.skip_deep_learning)
    print("[4/4] 完成。")
    print(f"结果文档：{PROJECT_ROOT / 'RESULTS.md'}")
    print(f"图片目录：{figures_dir}")


if __name__ == "__main__":
    main()
