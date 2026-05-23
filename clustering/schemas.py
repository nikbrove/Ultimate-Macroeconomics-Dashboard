"""Pydantic request/response schemas for the clustering FastAPI service."""

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class ClusterRequest(BaseModel):
    """Body accepted by ``POST /cluster``.

    Args:
        method: Clustering algorithm — ``kmeans`` or ``dbscan``.
        dataframe: Tabular input as a list of row dicts; must be non-empty.
        feature_columns: Optional explicit feature list; ``None`` means infer.
        k: Number of clusters (kmeans only).
        n_init: Restart count for kmeans.
        random_state: Seed for deterministic kmeans + dim-reduction.
        eps: Neighborhood radius for dbscan.
        min_samples: Minimum points per neighborhood for dbscan.
        reduction_method: ``tsne`` or ``pca``, used when ``n_features > 2``.
    """

    method: Literal["kmeans", "dbscan"] = Field(..., description="Clustering algorithm to use.")
    dataframe: list[dict[str, Any]] = Field(
        ..., description="Tabular data represented as a list of rows (JSON objects)."
    )
    feature_columns: list[str] | None = Field(
        default=None,
        description="Optional explicit list of numeric columns to use for clustering.",
    )

    k: int = Field(3, gt=0, description="Number of clusters for kmeans.")
    n_init: int = Field(10, gt=0, description="Number of kmeans initializations.")
    random_state: int = Field(42, description="Random seed for deterministic kmeans behavior.")

    eps: float = Field(0.5, gt=0.0, description="Neighborhood radius for dbscan.")
    min_samples: int = Field(5, gt=0, description="Min points per neighborhood.")

    reduction_method: Literal["tsne", "pca"] = Field(
        default="tsne",
        description=(
            "Dim-reduction method for 2D projection when n_features > 2. "
            "Ignored when n_features <= 2."
        ),
    )

    @field_validator("dataframe")
    @classmethod
    def validate_dataframe_not_empty(cls, value: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Reject empty ``dataframe`` payloads at parse time."""
        if len(value) == 0:
            raise ValueError("'dataframe' must contain at least one row.")
        return value

    @model_validator(mode="after")
    def validate_feature_columns(self) -> "ClusterRequest":
        """Cross-field check: ``feature_columns`` must refer to keys in the first row."""
        rows = self.dataframe

        if self.feature_columns is not None and len(self.feature_columns) == 0:
            raise ValueError("'feature_columns' cannot be an empty list.")

        first_row = rows[0]
        available_columns = set(first_row.keys())

        if self.feature_columns is not None:
            missing = [c for c in self.feature_columns if c not in available_columns]
            if missing:
                raise ValueError(
                    f"feature_columns contains unknown columns: {missing}. Available columns: {sorted(available_columns)}"
                )

        return self


class ClusterResponse(BaseModel):
    """Response returned by ``POST /cluster``.

    Args:
        method_used: Echo of the algorithm that was actually applied.
        dataframe: Input rows with ``cluster``, ``__viz_x``, ``__viz_y`` added.
        visualization_mode: How the 2D projection was produced.
        visualization_columns: The two row keys to use as x / y coordinates.
        visualization_labels: Human-readable axis titles.
    """

    method_used: str
    dataframe: list[dict[str, Any]]
    visualization_mode: Literal["feature_space", "tsne", "pca"] = Field(
        ..., description="Projection mode used for 2D visualization."
    )
    visualization_columns: list[str] = Field(
        ...,
        description="Two dataframe columns to use as x/y coordinates in scatter plots.",
    )
    visualization_labels: list[str] = Field(
        ..., description="Human-readable labels for visualization x/y axes."
    )
