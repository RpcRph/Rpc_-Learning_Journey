"""经典机器学习实验和结果图"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import load_breast_cancer, load_wine, make_blobs, make_regression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree

from .from_scratch import (
    GaussianNaiveBayes,
    KMeansScratch,
    LinearRegressionGD,
    LogisticRegressionGD,
    PCAScratch,
)
from .plotting import COLORS, configure_matplotlib, save_figure

MetricMap = dict[str, dict[str, Any]]
RANDOM_STATE = 42


def _rounded(value: float) -> float:
    return round(float(value), 6)


# 线性回归   拟合收敛可视化
def run_linear_regression(figures_dir: Path) -> MetricMap:
    features, targets = make_regression(
        n_samples=240,
        n_features=1,
        noise=12.0,
        random_state=RANDOM_STATE,
    )
    x_train, x_test, y_train, y_test = train_test_split(
        features, targets, test_size=0.25, random_state=RANDOM_STATE
    )
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    model = LinearRegressionGD(learning_rate=0.08, max_iter=3_000)
    model.fit(x_train_scaled, y_train)
    predictions = model.predict(x_test_scaled)

    mse = mean_squared_error(y_test, predictions)
    metrics: MetricMap = {
        "Linear Regression (NumPy)": {
            "task": "regression",
            "mse": _rounded(mse),
            "rmse": _rounded(np.sqrt(mse)),
            "mae": _rounded(mean_absolute_error(y_test, predictions)),
            "r2": _rounded(r2_score(y_test, predictions)),
            "iterations": model.n_iter_,
        }
    }

    x_line = np.linspace(features.min(), features.max(), 300).reshape(-1, 1)
    y_line = model.predict(scaler.transform(x_line))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].scatter(x_train, y_train, s=18, alpha=0.55, label="Train samples")
    axes[0].scatter(x_test, y_test, s=22, alpha=0.75, label="Test samples")
    axes[0].plot(x_line, y_line, color=COLORS[3], linewidth=2.4, label="Fitted line")
    axes[0].set_title("Linear Regression: data and fitted line")
    axes[0].set_xlabel("Feature x")
    axes[0].set_ylabel("Target y")
    axes[0].legend()

    axes[1].plot(model.loss_history_, color=COLORS[0])
    axes[1].set_title("Gradient-descent convergence")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Mean squared error")
    axes[1].set_yscale("log")
    save_figure(fig, figures_dir / "01_linear_regression.png")
    return metrics


def _classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    positive_score: np.ndarray,
) -> dict[str, float | str]:
    return {
        "task": "classification",
        "accuracy": _rounded(accuracy_score(y_true, y_pred)),
        "precision": _rounded(precision_score(y_true, y_pred, zero_division=0)),
        "recall": _rounded(recall_score(y_true, y_pred, zero_division=0)),
        "f1": _rounded(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": _rounded(roc_auc_score(y_true, positive_score)),
    }


# 逻辑回归、决策树、朴素贝叶斯、SVM、随机森林、梯度提升
def run_supervised_and_advanced(figures_dir: Path) -> MetricMap:
    """Compare Logistic, Tree, Bayes, SVM, Random Forest and Boosting."""
    dataset = load_breast_cancer()
    x_train, x_test, y_train, y_test = train_test_split(
        dataset.data,
        dataset.target,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=dataset.target,
    )
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    logistic = LogisticRegressionGD(learning_rate=0.08, max_iter=4_000, l2=2e-3).fit(
        x_train_scaled, y_train
    )
    decision_tree = DecisionTreeClassifier(
        max_depth=4, min_samples_leaf=5, random_state=RANDOM_STATE
    ).fit(x_train, y_train)
    naive_bayes = GaussianNaiveBayes().fit(x_train, y_train)
    svm = Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", SVC(C=2.0, kernel="rbf", gamma="scale")),
        ]
    ).fit(x_train, y_train)
    random_forest = RandomForestClassifier(
        n_estimators=180,
        max_depth=7,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ).fit(x_train, y_train)
    gradient_boosting = GradientBoostingClassifier(
        n_estimators=120,
        learning_rate=0.05,
        max_depth=2,
        random_state=RANDOM_STATE,
    ).fit(x_train, y_train)

    predictions_and_scores = {
        "Logistic Regression (NumPy)": (
            logistic.predict(x_test_scaled),
            logistic.predict_proba(x_test_scaled)[:, 1],
        ),
        "Decision Tree": (
            decision_tree.predict(x_test),
            decision_tree.predict_proba(x_test)[:, 1],
        ),
        "Gaussian Naive Bayes (NumPy)": (
            naive_bayes.predict(x_test),
            naive_bayes.predict_proba(x_test)[:, 1],
        ),
        "SVM (RBF kernel)": (
            svm.predict(x_test),
            svm.decision_function(x_test),
        ),
        "Random Forest": (
            random_forest.predict(x_test),
            random_forest.predict_proba(x_test)[:, 1],
        ),
        "Gradient Boosting": (
            gradient_boosting.predict(x_test),
            gradient_boosting.predict_proba(x_test)[:, 1],
        ),
    }
    metrics: MetricMap = {
        name: _classification_metrics(y_test, prediction, score)
        for name, (prediction, score) in predictions_and_scores.items()
    }

    names = list(predictions_and_scores)
    short_names = [
        "Logistic",
        "Tree",
        "Naive Bayes",
        "SVM",
        "Random Forest",
        "Boosting",
    ]
    x_positions = np.arange(len(names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(11, 5))
    for offset, metric_name, color in zip(
        (-width, 0.0, width), ("accuracy", "f1", "roc_auc"), COLORS[:3], strict=True
    ):
        values = [metrics[name][metric_name] for name in names]
        ax.bar(
            x_positions + offset,
            values,
            width,
            label=metric_name.upper(),
            color=color,
            alpha=0.88,
        )
    ax.set_ylim(0.75, 1.01)
    ax.set_ylabel("Score")
    ax.set_title("Supervised learning model comparison")
    ax.set_xticks(x_positions, short_names, rotation=18, ha="right")
    ax.legend(ncols=3)
    save_figure(fig, figures_dir / "02_supervised_model_comparison.png")

    fig, ax = plt.subplots(figsize=(14, 7))
    plot_tree(
        decision_tree,
        ax=ax,
        feature_names=dataset.feature_names,
        class_names=list(dataset.target_names),
        filled=True,
        rounded=True,
        fontsize=7,
    )
    ax.set_title("Decision Tree structure (max_depth=4)")
    save_figure(fig, figures_dir / "03_decision_tree_structure.png")

    importance_order = np.argsort(random_forest.feature_importances_)[-10:]
    staged_accuracy = [
        accuracy_score(y_test, stage_prediction)
        for stage_prediction in gradient_boosting.staged_predict(x_test)
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].barh(
        np.asarray(dataset.feature_names)[importance_order],
        random_forest.feature_importances_[importance_order],
        color=COLORS[2],
    )
    axes[0].set_title("Random Forest: top-10 feature importance")
    axes[0].set_xlabel("Gini importance")
    axes[1].plot(
        np.arange(1, len(staged_accuracy) + 1),
        staged_accuracy,
        color=COLORS[4],
    )
    axes[1].set_title("Gradient Boosting: staged test accuracy")
    axes[1].set_xlabel("Number of weak learners")
    axes[1].set_ylabel("Accuracy")
    save_figure(fig, figures_dir / "04_ensemble_models.png")
    return metrics


# 交叉验证和网格搜索
def run_optimization_and_evaluation(figures_dir: Path) -> MetricMap:
    """Demonstrate scaling, stratified cross-validation and grid search."""
    dataset = load_breast_cancer()
    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", SVC(kernel="rbf")),
        ]
    )
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    fold_scores = cross_val_score(
        pipeline, dataset.data, dataset.target, cv=folds, scoring="f1", n_jobs=-1
    )
    search = GridSearchCV(
        pipeline,
        param_grid={
            "model__C": [0.5, 1.0, 2.0, 5.0],
            "model__gamma": ["scale", 0.01, 0.1],
        },
        scoring="f1",
        cv=folds,
        n_jobs=-1,
    )
    search.fit(dataset.data, dataset.target)

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.bar(np.arange(1, 6), fold_scores, color=COLORS[0], alpha=0.86)
    ax.axhline(
        fold_scores.mean(),
        color=COLORS[3],
        linestyle="--",
        label=f"Mean F1 = {fold_scores.mean():.4f}",
    )
    ax.set_ylim(0.85, 1.0)
    ax.set_xticks(np.arange(1, 6))
    ax.set_xlabel("Stratified CV fold")
    ax.set_ylabel("F1 score")
    ax.set_title("Optimization & evaluation: StandardScaler + SVM")
    ax.legend()
    save_figure(fig, figures_dir / "05_cross_validation_and_tuning.png")

    return {
        "SVM Cross Validation": {
            "task": "model_selection",
            "fold_f1": [_rounded(value) for value in fold_scores],
            "mean_f1": _rounded(fold_scores.mean()),
            "std_f1": _rounded(fold_scores.std()),
        },
        "SVM Grid Search": {
            "task": "model_selection",
            "best_cv_f1": _rounded(search.best_score_),
            "best_C": search.best_params_["model__C"],
            "best_gamma": search.best_params_["model__gamma"],
        },
    }


# K-Means、层次聚类和 PCA
def run_unsupervised(figures_dir: Path) -> MetricMap:
    """Run K-Means, agglomerative clustering and PCA."""
    blob_features, blob_labels = make_blobs(
        n_samples=300,
        centers=3,
        cluster_std=(0.75, 0.95, 0.65),
        random_state=RANDOM_STATE,
    )
    kmeans = KMeansScratch(n_clusters=3, random_state=RANDOM_STATE).fit(blob_features)
    hierarchical = AgglomerativeClustering(n_clusters=3, linkage="ward").fit(
        blob_features
    )

    wine = load_wine()
    wine_scaled = StandardScaler().fit_transform(wine.data)
    pca = PCAScratch(n_components=2)
    reduced = pca.fit_transform(wine_scaled)
    reconstructed = pca.inverse_transform(reduced)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    axes[0].scatter(
        blob_features[:, 0],
        blob_features[:, 1],
        c=kmeans.labels_,
        cmap="viridis",
        s=20,
        alpha=0.75,
    )
    axes[0].scatter(
        kmeans.cluster_centers_[:, 0],
        kmeans.cluster_centers_[:, 1],
        c="red",
        marker="X",
        s=180,
        edgecolor="white",
        label="Centroids",
    )
    axes[0].set_title("K-Means (NumPy)")
    axes[0].legend()

    subset = blob_features[::5]
    dendrogram(
        linkage(subset, method="ward"),
        ax=axes[1],
        no_labels=True,
        color_threshold=None,
    )
    axes[1].set_title("Hierarchical clustering dendrogram")
    axes[1].set_xlabel("Sample")
    axes[1].set_ylabel("Ward distance")

    scatter = axes[2].scatter(
        reduced[:, 0],
        reduced[:, 1],
        c=wine.target,
        cmap="plasma",
        s=28,
        alpha=0.8,
    )
    axes[2].set_title("PCA projection (NumPy)")
    axes[2].set_xlabel("Principal component 1")
    axes[2].set_ylabel("Principal component 2")
    axes[2].legend(
        scatter.legend_elements()[0],
        list(wine.target_names),
        title="Wine class",
        loc="best",
    )
    save_figure(fig, figures_dir / "06_unsupervised_learning.png")

    return {
        "K-Means (NumPy)": {
            "task": "clustering",
            "silhouette": _rounded(silhouette_score(blob_features, kmeans.labels_)),
            "adjusted_rand_index": _rounded(
                adjusted_rand_score(blob_labels, kmeans.labels_)
            ),
            "inertia": _rounded(kmeans.inertia_),
            "iterations": kmeans.n_iter_,
        },
        "Hierarchical Clustering": {
            "task": "clustering",
            "silhouette": _rounded(
                silhouette_score(blob_features, hierarchical.labels_)
            ),
            "adjusted_rand_index": _rounded(
                adjusted_rand_score(blob_labels, hierarchical.labels_)
            ),
        },
        "PCA (NumPy)": {
            "task": "dimensionality_reduction",
            "explained_variance_pc1": _rounded(pca.explained_variance_ratio_[0]),
            "explained_variance_pc2": _rounded(pca.explained_variance_ratio_[1]),
            "cumulative_explained_variance": _rounded(
                pca.explained_variance_ratio_.sum()
            ),
            "reconstruction_mse": _rounded(
                mean_squared_error(wine_scaled, reconstructed)
            ),
        },
    }


# 按顺序调用以上所有实验
def run_classical_experiments(figures_dir: Path) -> MetricMap:
    configure_matplotlib()
    results: MetricMap = {}
    results.update(run_linear_regression(figures_dir))
    results.update(run_supervised_and_advanced(figures_dir))
    results.update(run_optimization_and_evaluation(figures_dir))
    results.update(run_unsupervised(figures_dir))
    return results
