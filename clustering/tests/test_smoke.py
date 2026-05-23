"""Smoke tests for the clustering service: feature matrix + projection helpers + /cluster route."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from main import _build_feature_matrix, _build_visual_projection, app


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


def test_health_returns_ok(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_methods_lists_kmeans_dbscan_and_reductions(client: TestClient) -> None:
    response = client.get("/methods")
    body = response.json()
    assert set(body["available_methods"]) == {"kmeans", "dbscan"}
    assert set(body["available_reductions"]) == {"tsne", "pca"}


def test_build_feature_matrix_infers_numeric_columns() -> None:
    rows = [
        {"a": 1.0, "b": 2.0, "name": "first"},
        {"a": 3.0, "b": 4.0, "name": "second"},
    ]
    matrix, columns = _build_feature_matrix(rows, feature_columns=None)
    assert columns == ["a", "b"]
    assert matrix.tolist() == [[1.0, 2.0], [3.0, 4.0]]


def test_build_visual_projection_passes_through_two_features() -> None:
    matrix = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    projection, mode, labels = _build_visual_projection(matrix, ["a", "b"], random_state=0)
    assert mode == "feature_space"
    assert labels == ["a", "b"]
    assert projection.shape == (3, 2)


def test_build_visual_projection_uses_pca_when_requested() -> None:
    rng = np.random.default_rng(seed=42)
    matrix = rng.normal(size=(20, 5))
    projection, mode, labels = _build_visual_projection(
        matrix, [f"f{i}" for i in range(5)], random_state=0, reduction_method="pca"
    )
    assert mode == "pca"
    assert labels == ["PC 1", "PC 2"]
    assert projection.shape == (20, 2)


def test_cluster_endpoint_kmeans_two_features(client: TestClient) -> None:
    rng = np.random.default_rng(seed=0)
    cluster_a = rng.normal(loc=0.0, size=(8, 2))
    cluster_b = rng.normal(loc=5.0, size=(8, 2))
    rows = [
        {"a": float(point[0]), "b": float(point[1])}
        for point in np.vstack([cluster_a, cluster_b])
    ]
    response = client.post(
        "/cluster",
        json={"method": "kmeans", "dataframe": rows, "feature_columns": ["a", "b"], "k": 2},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["method_used"] == "kmeans"
    assert body["visualization_mode"] == "feature_space"
    labels = {row["cluster"] for row in body["dataframe"]}
    assert labels == {0, 1}
