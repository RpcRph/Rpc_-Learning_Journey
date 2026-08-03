"""五个手写的机器学习算法实现，分别是：
1. 线性回归（Linear Regression）通过批量梯度下降训练。
2. 逻辑回归（Logistic Regression）通过交叉熵训练。
3. 高斯朴素贝叶斯（Gaussian Naive Bayes）使用对数概率进行数值稳定性计算。
4. 带 K-Means++ 初始化的 K-Means（K-Means with K-Means++ initialization）。
5. 基于协方差矩阵特征分解的 PCA（Principal Component Analysis via covariance eigendecomposition）。
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


# 将任何可转为数组的对象  转为   numpy 数组
def _as_2d_float(x: ArrayLike) -> NDArray[np.float64]:
    array = np.asarray(x, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError("X 必须是一维或二维数组")
    return array


# 灭有加入正则化
# 梯度下降线性回归  损失函数均方误差
class LinearRegressionGD:
    """线性回归通过批量梯度下降训练。"""

    def __init__(
        self,
        learning_rate: float = 0.05,
        # 3000次迭代
        max_iter: int = 3_000,
        tolerance: float = 1e-10,
    ) -> None:
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tolerance = tolerance

    # 训练模型
    def fit(self, x: ArrayLike, y: ArrayLike) -> "LinearRegressionGD":
        features = _as_2d_float(x)
        targets = np.asarray(y, dtype=np.float64).reshape(-1)
        if len(features) != len(targets):
            raise ValueError("X 与 y 的样本数必须一致")
        # 样本数及每个样本的特征数
        n_samples, n_features = features.shape
        # 系数一开始都为0
        self.coef_ = np.zeros(n_features, dtype=np.float64)
        # 初始化b
        self.intercept_ = 0.0
        # 损失记录初始
        self.loss_history_: list[float] = []
        previous_loss = np.inf

        for _ in range(self.max_iter):
            # 前向传播
            # y = Xw + b
            predictions = features @ self.coef_ + self.intercept_
            # e = y_hat - y
            errors = predictions - targets
            # 权重梯度
            gradient_w = (2.0 / n_samples) * (features.T @ errors)
            # 偏置梯度
            gradient_b = float(2.0 * np.mean(errors))
            # 更新
            self.coef_ -= self.learning_rate * gradient_w
            self.intercept_ -= self.learning_rate * gradient_b
            # 损失计算
            loss = float(
                np.mean((features @ self.coef_ + self.intercept_ - targets) ** 2)
            )
            # 记录
            self.loss_history_.append(loss)
            # 收敛就结束
            if abs(previous_loss - loss) < self.tolerance:
                break
            # 无则继续下一轮
            previous_loss = loss
        # 迭代次数
        self.n_iter_ = len(self.loss_history_)
        return self

    # 预测
    def predict(self, x: ArrayLike) -> NDArray[np.float64]:
        features = _as_2d_float(x)
        return features @ self.coef_ + self.intercept_


# 逻辑回归通   过交叉熵训练的二元逻辑回归
# y = h_w(x) = 1 / (1 + exp(-z))  z = w^T * x + b
class LogisticRegressionGD:
    def __init__(
        self,
        learning_rate: float = 0.1,
        max_iter: int = 2_000,
        l2: float = 1e-3,
        tolerance: float = 1e-9,
    ) -> None:
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.l2 = l2
        self.tolerance = tolerance

    # 1 / (1 + exp(-z))
    @staticmethod
    def _sigmoid(z: NDArray[np.float64]) -> NDArray[np.float64]:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500.0, 500.0)))

    def fit(self, x: ArrayLike, y: ArrayLike) -> "LogisticRegressionGD":
        features = _as_2d_float(x)
        targets = np.asarray(y, dtype=np.float64).reshape(-1)
        unique = np.unique(targets)
        if not np.array_equal(unique, np.array([0.0, 1.0])):
            raise ValueError("LogisticRegressionGD 仅接受标签 0 和 1")
        if len(features) != len(targets):
            raise ValueError("X 与 y 的样本数必须一致")

        n_samples, n_features = features.shape
        self.coef_ = np.zeros(n_features, dtype=np.float64)
        self.intercept_ = 0.0
        self.loss_history_: list[float] = []
        previous_loss = np.inf
        epsilon = 1e-12

        for _ in range(self.max_iter):
            probabilities = self._sigmoid(features @ self.coef_ + self.intercept_)
            errors = probabilities - targets
            gradient_w = (features.T @ errors) / n_samples + self.l2 * self.coef_
            gradient_b = float(np.mean(errors))
            self.coef_ -= self.learning_rate * gradient_w
            self.intercept_ -= self.learning_rate * gradient_b

            probabilities = self._sigmoid(features @ self.coef_ + self.intercept_)
            # 二元的交叉熵
            cross_entropy = -np.mean(
                targets * np.log(probabilities + epsilon)
                + (1.0 - targets) * np.log(1.0 - probabilities + epsilon)
            )
            # L2正则
            loss = float(cross_entropy + 0.5 * self.l2 * np.sum(self.coef_**2))
            self.loss_history_.append(loss)
            if abs(previous_loss - loss) < self.tolerance:
                break
            previous_loss = loss

        self.n_iter_ = len(self.loss_history_)
        return self

    def predict_proba(self, x: ArrayLike) -> NDArray[np.float64]:
        positive = self._sigmoid(_as_2d_float(x) @ self.coef_ + self.intercept_)
        # 0 和 1 的概率 按列拼接
        return np.column_stack((1.0 - positive, positive))

    def predict(self, x: ArrayLike) -> NDArray[np.int64]:
        return (self.predict_proba(x)[:, 1] >= 0.5).astype(np.int64)


# 高斯朴素贝叶斯（用对数概率进行数值稳定性计算）
class GaussianNaiveBayes:
    def __init__(self, var_smoothing: float = 1e-9) -> None:
        # 防止除零异常
        self.var_smoothing = var_smoothing

    def fit(self, x: ArrayLike, y: ArrayLike) -> "GaussianNaiveBayes":
        features = _as_2d_float(x)
        targets = np.asarray(y).reshape(-1)
        if len(features) != len(targets):
            raise ValueError("X 与 y 的样本数必须一致")

        self.classes_, counts = np.unique(targets, return_counts=True)
        # 先验概率
        self.class_prior_ = counts / counts.sum()
        """
            means = []
            for label in self.classes_:
                class_features = features[targets == label]
                class_mean = class_features.mean(axis=0)
                means.append(class_mean)
            简化写法
        """
        self.theta_ = np.vstack(
            [features[targets == label].mean(axis=0) for label in self.classes_]
        )
        # 每个特征在全部数据上的方差，取最大值
        base_variance = float(np.var(features, axis=0).max())
        epsilon = self.var_smoothing * max(base_variance, 1.0)
        # 各类方差
        self.var_ = np.vstack(
            [
                features[targets == label].var(axis=0) + epsilon
                for label in self.classes_
            ]
        )
        return self

    def _joint_log_likelihood(self, x: ArrayLike) -> NDArray[np.float64]:
        features = _as_2d_float(x)
        rows = []
        for index, _ in enumerate(self.classes_):
            log_prior = np.log(self.class_prior_[index])
            log_density = -0.5 * np.sum(
                np.log(2.0 * np.pi * self.var_[index])
                + ((features - self.theta_[index]) ** 2) / self.var_[index],
                axis=1,
            )
            rows.append(log_prior + log_density)
        return np.column_stack(rows)

    def predict(self, x: ArrayLike) -> NDArray:
        indices = np.argmax(self._joint_log_likelihood(x), axis=1)
        return self.classes_[indices]

    def predict_proba(self, x: ArrayLike) -> NDArray[np.float64]:
        log_likelihood = self._joint_log_likelihood(x)
        shifted = log_likelihood - log_likelihood.max(axis=1, keepdims=True)
        probabilities = np.exp(shifted)
        return probabilities / probabilities.sum(axis=1, keepdims=True)


# 带 K-Means++ 初始化的 K-Means
class KMeansScratch:
    """K-Means with K-Means++ initialization."""

    def __init__(
        self,
        n_clusters: int = 3,
        max_iter: int = 300,
        tolerance: float = 1e-4,
        random_state: int = 42,
    ) -> None:
        if n_clusters < 1:
            raise ValueError("n_clusters 必须为正整数")
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.random_state = random_state

    def _initialize_centers(
        self, features: NDArray[np.float64], rng: np.random.Generator
    ) -> NDArray[np.float64]:
        n_samples = len(features)
        if self.n_clusters > n_samples:
            raise ValueError("聚类数不能超过样本数")

        centers = [features[rng.integers(n_samples)].copy()]
        for _ in range(1, self.n_clusters):
            squared_distances = np.min(
                np.sum(
                    # 欧氏距离
                    (features[:, None, :] - np.asarray(centers)[None, :, :]) ** 2,
                    axis=2,
                ),
                axis=1,
            )
            total = float(squared_distances.sum())
            if total <= 0.0:
                candidate = features[rng.integers(n_samples)].copy()
            else:
                candidate = features[
                    rng.choice(n_samples, p=squared_distances / total)
                ].copy()
            centers.append(candidate)
        return np.asarray(centers)

    def fit(self, x: ArrayLike) -> "KMeansScratch":
        features = _as_2d_float(x)
        rng = np.random.default_rng(self.random_state)
        centers = self._initialize_centers(features, rng)

        for iteration in range(1, self.max_iter + 1):
            squared_distances = np.sum(
                (features[:, None, :] - centers[None, :, :]) ** 2, axis=2
            )
            # 距离最小的中心（每个样本都找）
            labels = np.argmin(squared_distances, axis=1)
            new_centers = centers.copy()
            nearest_distances = np.min(squared_distances, axis=1)
            for cluster in range(self.n_clusters):
                members = features[labels == cluster]
                if len(members):
                    new_centers[cluster] = members.mean(axis=0)
                else:
                    farthest = int(np.argmax(nearest_distances))
                    new_centers[cluster] = features[farthest]

            shift = float(np.linalg.norm(new_centers - centers))
            centers = new_centers
            if shift <= self.tolerance:
                break

        self.cluster_centers_ = centers
        self.labels_ = self.predict(features)
        self.inertia_ = float(
            np.sum((features - self.cluster_centers_[self.labels_]) ** 2)
        )
        self.n_iter_ = iteration
        return self

    def predict(self, x: ArrayLike) -> NDArray[np.int64]:
        features = _as_2d_float(x)
        squared_distances = np.sum(
            (features[:, None, :] - self.cluster_centers_[None, :, :]) ** 2, axis=2
        )
        return np.argmin(squared_distances, axis=1).astype(np.int64)


# 基于协方差矩阵特征分解的 PCA
class PCAScratch:
    def __init__(self, n_components: int = 2) -> None:
        if n_components < 1:
            raise ValueError("n_components 必须为正整数")
        self.n_components = n_components

    def fit(self, x: ArrayLike) -> "PCAScratch":
        features = _as_2d_float(x)
        if self.n_components > features.shape[1]:
            raise ValueError("主成分数不能超过特征数")

        self.mean_ = features.mean(axis=0)
        centered = features - self.mean_
        # 协方差矩阵
        covariance = np.cov(centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.maximum(eigenvalues[order], 0.0)
        eigenvectors = eigenvectors[:, order]

        self.explained_variance_ = eigenvalues[: self.n_components]
        total_variance = float(eigenvalues.sum())
        self.explained_variance_ratio_ = (
            self.explained_variance_ / total_variance
            if total_variance > 0
            else np.zeros_like(self.explained_variance_)
        )
        self.components_ = eigenvectors[:, : self.n_components].T
        return self

    def transform(self, x: ArrayLike) -> NDArray[np.float64]:
        return (_as_2d_float(x) - self.mean_) @ self.components_.T

    def fit_transform(self, x: ArrayLike) -> NDArray[np.float64]:
        return self.fit(x).transform(x)

    def inverse_transform(self, transformed: ArrayLike) -> NDArray[np.float64]:
        reduced = _as_2d_float(transformed)
        return reduced @ self.components_ + self.mean_
