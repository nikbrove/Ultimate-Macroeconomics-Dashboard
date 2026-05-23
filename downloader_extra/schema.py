"""ORM model + pydantic schemas for the on-demand WB indicator ingestion service."""

from pydantic import BaseModel
from sqlalchemy import Float, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Declarative base for this service's ORM models."""


class MacroIndicator(Base):
    """One ``(economy, year, indicator_id, db_id)`` cell from the World Bank.

    Mirrors the ``indicators`` table that ``downloader_general`` populates
    on first boot; this service re-fetches single indicators on demand.
    """

    __tablename__ = "indicators"

    economy: Mapped[str] = mapped_column(String, primary_key=True, nullable=False)
    year: Mapped[int] = mapped_column(Integer, primary_key=True, nullable=False)
    value: Mapped[float | None] = mapped_column(Float, nullable=True)
    indicator_id: Mapped[str] = mapped_column(String, primary_key=True, nullable=False)
    db_id: Mapped[int] = mapped_column(Integer, primary_key=True, nullable=False, index=True)


class IngestResponse(BaseModel):
    """Response from ``POST /ingest``.

    Args:
        indicator_id: WB indicator id (echo of the request).
        db_id: WB database id (echo of the request).
        rows_inserted: Number of rows actually written; ``0`` if the
            indicator was already present.
        status: ``success`` or ``already_downloaded``.
    """

    indicator_id: str
    db_id: int
    rows_inserted: int
    status: str


class IngestRequest(BaseModel):
    """Body for ``POST /ingest``.

    Args:
        indicator_id: World Bank indicator id (e.g. ``NY.GDP.MKTP.CD``).
        db_id: World Bank database id (e.g. ``2`` for WDI).
    """

    indicator_id: str
    db_id: int
