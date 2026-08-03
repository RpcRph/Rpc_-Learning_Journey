import numpy as np
from sklearn.datasets import make_blobs, make_classification
from sklearn.metrics import accuracy_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler

from ml_practice.from_scratch import (
    GaussianNaiveBayes,
    KMeansScratch,
    LinearRegressionGD,
    LogisticRegressionGD,
    PCAScratch,
)


def test_linear_regression_learns_known_line() -> None:
    x = np.linspace(-2.0, 2.0, 100).reshape(-1, 1)
    y = 3.0 * x[:, 0] - 1.5
    model = LinearRegressionGD(learning_rate=0.08).fit(x, y)
    assert np.allclose(model.coef_, [3.0], atol=1e-3)
    assert np.isclose(model.intercept_, -1.5, atol=1e-3)


def test_logistic_regression_classifies_separable_data() -> None:
    x, y = make_classification(
        n_samples=240,
        n_features=4,
        n_redundant=0,
        class_sep=2.5,
        random_state=42,
    )
    x = StandardScaler().fit_transform(x)
    model = LogisticRegressionGD(max_iter=3_000).fit(x, y)
    assert accuracy_score(y, model.predict(x)) > 0.95
    assert np.allclose(model.predict_proba(x).sum(axis=1), 1.0)


def test_gaussian_naive_bayes_predicts_blobs() -> None:
    x, y = make_blobs(n_samples=200, centers=3, cluster_std=0.7, random_state=42)
    model = GaussianNaiveBayes().fit(x, y)
    assert accuracy_score(y, model.predict(x)) > 0.95
    assert np.allclose(model.predict_proba(x).sum(axis=1), 1.0)


def test_kmeans_recovers_blob_structure() -> None:
    x, y = make_blobs(n_samples=180, centers=3, cluster_std=0.6, random_state=42)
    model = KMeansScratch(n_clusters=3, random_state=42).fit(x)
    assert adjusted_rand_score(y, model.labels_) > 0.95
    assert model.inertia_ > 0


def test_pca_shapes_ratios_and_inverse_transform() -> None:
    rng = np.random.default_rng(42)
    x = rng.normal(size=(120, 5))
    x[:, 1] = 2.0 * x[:, 0] + rng.normal(scale=0.05, size=120)
    model = PCAScratch(n_components=2)
    reduced = model.fit_transform(x)
    reconstructed = model.inverse_transform(reduced)
    assert reduced.shape == (120, 2)
    assert reconstructed.shape == x.shape
    assert 0 < model.explained_variance_ratio_.sum() <= 1
