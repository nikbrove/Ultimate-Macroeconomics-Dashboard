"""FastAPI service exposing unsupervised clustering on tabular input.

Wraps scikit-learn ``KMeans`` / ``DBSCAN`` and projects the feature matrix into
2D for visualization. When the input has more than two features, the projection
is built with t-SNE or PCA depending on the ``reduction_method`` field of the
request; with one or two features the original space is preserved as-is.
"""

from pathlib import Path

import numpy as np
import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from schemas import ClusterRequest, ClusterResponse

CONFIG_PATH = Path("config.yaml")
ENV_FILE_PATH = Path(".env")

CONFIG = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
load_dotenv(ENV_FILE_PATH)

VIZ_X_COL = "__viz_x"
VIZ_Y_COL = "__viz_y"

app = FastAPI(
    title="Clustering API",
    description="A lightweight API for kmeans and dbscan clustering.",
)


def _infer_numeric_columns(rows: list[dict[str, object]]) -> list[str]:
    """Return the keys whose values are numeric in *every* row of ``rows``.

    Args:
        rows: Non-empty list of row dicts (caller guarantees ``len(rows) >= 1``).

    Returns:
        Column names that are numeric across all rows. Order matches ``rows[0]``.
    """
    first_row = rows[0]
    numeric_columns: list[str] = []
    for col in first_row:
        values = [row.get(col) for row in rows]
        if all(isinstance(v, (int, float, np.integer, np.floating)) for v in values):
            numeric_columns.append(col)
    return numeric_columns


def _build_feature_matrix(
    rows: list[dict[str, object]], feature_columns: list[str] | None
) -> tuple[np.ndarray, list[str]]:
    """Cast ``rows`` to a 2D float matrix using ``feature_columns``.

    Args:
        rows: Tabular input as list of row dicts.
        feature_columns: Explicit feature list, or ``None`` to auto-detect
            numeric columns via :func:`_infer_numeric_columns`.

    Returns:
        Tuple of ``(matrix, columns_used)``.

    Raises:
        HTTPException: When no numeric features are available, a feature
            column is missing from a row, or a value isn't a number.
    """
    if feature_columns is None:
        feature_columns = _infer_numeric_columns(rows)

    if len(feature_columns) == 0:
        raise HTTPException(
            status_code=400,
            detail="No numeric features available. Provide numeric columns in 'feature_columns'.",
        )

    matrix: list[list[float]] = []
    for row_index, row in enumerate(rows):
        values: list[float] = []
        for col in feature_columns:
            if col not in row:
                raise HTTPException(
                    status_code=400,
                    detail=f"Row {row_index} is missing required feature column '{col}'.",
                )
            raw_value = row[col]
            if not isinstance(raw_value, (int, float, np.integer, np.floating)):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Column '{col}' contains a non-finite or non-numeric value at row {row_index}."
                    ),
                )
            values.append(float(raw_value))
        matrix.append(values)

    return np.asarray(matrix, dtype=float), feature_columns


def _build_visual_projection(
    feature_matrix: np.ndarray,
    feature_columns: list[str],
    random_state: int,
    reduction_method: str = "tsne",
) -> tuple[np.ndarray, str, list[str]]:
    """Project ``feature_matrix`` into 2D for plotting.

    The branch taken depends on shape: one feature is padded with a zero
    axis, two features are returned as-is, otherwise a dim-reduction is
    applied (PCA or t-SNE per ``reduction_method``). Fewer than 5 rows skip
    the reduction entirely because t-SNE needs perplexity ``< n_rows``.

    Args:
        feature_matrix: ``(n_rows, n_features)`` numeric matrix.
        feature_columns: Names of the columns in ``feature_matrix``.
        random_state: Seed forwarded to PCA / TSNE for determinism.
        reduction_method: ``"tsne"`` or ``"pca"``; only consulted when
            ``n_features > 2`` and ``n_rows >= 5``.

    Returns:
        Tuple of ``(projection, mode, axis_labels)`` where ``mode`` is one
        of ``"feature_space"``, ``"tsne"``, ``"pca"`` for downstream UIs.
    """
    n_rows, n_features = feature_matrix.shape

    if n_features == 1:
        zero_col = np.zeros((n_rows, 1), dtype=float)
        return (
            np.hstack([feature_matrix, zero_col]),
            "feature_space",
            [
                feature_columns[0],
                "Zero axis",
            ],
        )

    if n_features == 2:
        return feature_matrix, "feature_space", [feature_columns[0], feature_columns[1]]

    if n_rows < 5:
        return (
            feature_matrix[:, :2],
            "feature_space",
            [
                feature_columns[0],
                feature_columns[1],
            ],
        )

    if reduction_method == "pca":
        projection = PCA(n_components=2, random_state=random_state).fit_transform(feature_matrix)
        return projection, "pca", ["PC 1", "PC 2"]

    perplexity = min(30.0, float(n_rows - 1))
    projection = TSNE(
        n_components=2,
        random_state=random_state,
        init="random",
        learning_rate="auto",
        perplexity=perplexity,
    ).fit_transform(feature_matrix)
    return projection, "tsne", ["t-SNE 1", "t-SNE 2"]


@app.get("/")
def root() -> dict[str, str]:
    """Return a static welcome banner — used as a liveness signal."""
    return {"message": "Welcome to the Clustering API"}


@app.get("/health")
def health_check() -> dict[str, str]:
    """Return ``{"status": "ok"}`` for the Compose healthcheck."""
    return {"status": "ok"}


@app.get("/methods")
def list_methods() -> dict[str, list[str]]:
    """Expose the algorithms and dim-reduction methods supported by this service."""
    return {
        "available_methods": ["kmeans", "dbscan"],
        "available_reductions": ["tsne", "pca"],
    }


def _run_clustering(
    estimator: object,
    feature_matrix: np.ndarray,
    feature_columns: list[str],
    random_state: int,
    reduction_method: str,
) -> tuple[np.ndarray, np.ndarray, str, list[str]]:
    """Fit ``estimator`` on ``feature_matrix`` and build the 2D projection.

    Args:
        estimator: A scikit-learn estimator implementing ``fit_predict``.
        feature_matrix: Numeric input passed straight to the estimator.
        feature_columns: Original column names (used for axis labels).
        random_state: Seed for the projection step.
        reduction_method: ``"tsne"`` or ``"pca"`` — see
            :func:`_build_visual_projection`.

    Returns:
        Tuple of ``(labels, projection, projection_mode, projection_labels)``.
    """
    labels = estimator.fit_predict(feature_matrix)
    projection, projection_mode, projection_labels = _build_visual_projection(
        feature_matrix=feature_matrix,
        feature_columns=feature_columns,
        random_state=random_state,
        reduction_method=reduction_method,
    )
    return labels, projection, projection_mode, projection_labels


@app.post("/cluster", response_model=ClusterResponse)
async def cluster_dataframe(request: ClusterRequest) -> ClusterResponse:
    """Cluster the rows in ``request`` and return labels + 2D coordinates.

    Args:
        request: Payload selecting the algorithm, hyperparameters, and the
            input rows. See :class:`schemas.ClusterRequest`.

    Returns:
        ClusterResponse with the original rows augmented with ``cluster`` /
        ``__viz_x`` / ``__viz_y`` plus projection metadata.

    Raises:
        HTTPException: 400 for unknown algorithms or bad inputs; 500 for
            internal sklearn failures.
    """
    rows = request.dataframe
    feature_matrix, feature_columns = _build_feature_matrix(rows, request.feature_columns)

    if request.method == "kmeans":
        if request.k > len(rows):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"k ({request.k}) cannot be greater than the number of rows ({len(rows)})."
                ),
            )
        estimator = KMeans(
            n_clusters=request.k,
            n_init=request.n_init,
            random_state=request.random_state,
        )
    elif request.method == "dbscan":
        estimator = DBSCAN(
            eps=request.eps,
            min_samples=request.min_samples,
        )
    else:
        raise HTTPException(status_code=400, detail=f"Unknown method: {request.method}")

    try:
        labels, projection, projection_mode, projection_labels = await run_in_threadpool(
            _run_clustering,
            estimator,
            feature_matrix,
            feature_columns,
            request.random_state,
            request.reduction_method,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid clustering input: {str(e)}. Features used: {feature_columns}",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clustering failed: {str(e)}")

    output_rows = [dict(row) for row in rows]
    for row, label, point in zip(output_rows, labels, projection):
        row["cluster"] = int(label)
        row[VIZ_X_COL] = float(point[0])
        row[VIZ_Y_COL] = float(point[1])

    return ClusterResponse(
        method_used=request.method,
        dataframe=output_rows,
        visualization_mode=projection_mode,
        visualization_columns=[VIZ_X_COL, VIZ_Y_COL],
        visualization_labels=projection_labels,
    )
