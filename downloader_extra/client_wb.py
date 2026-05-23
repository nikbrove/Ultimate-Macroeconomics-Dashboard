"""World Bank fetch + Postgres upsert for one indicator at a time.

Called by ``downloader_extra``'s ``POST /ingest`` endpoint when the agent
asks for an indicator not yet in the database. Tries the high-level
``wbgapi`` library first and falls back to the raw v2 REST API when the
former returns an empty payload (some sources need the fallback).
"""

import logging

import httpx
import polars as pl
import wbgapi as wb
from sqlalchemy import create_engine, delete
from sqlalchemy.orm import Session

from schema import MacroIndicator

logger = logging.getLogger(__name__)


def _fetch_indicator_data_via_api(indicator_id: str, db: int) -> list[dict]:
    """Fetch an indicator via the v2 REST endpoint when ``wbgapi`` returns empty.

    Pages through all results, drops aggregate ISO codes (anything that
    isn't 3 characters long) and ``null`` values, and converts the year
    field to ``int``.

    Args:
        indicator_id: World Bank indicator id.
        db: World Bank database id (the ``source`` parameter).

    Returns:
        Rows shaped as ``{"economy", "year", "value"}``. Empty list if the
        endpoint returns nothing.
    """
    rows = []
    page = 1
    with httpx.Client(timeout=30.0) as client:
        while True:
            resp = client.get(
                f"https://api.worldbank.org/v2/country/all/indicator/{indicator_id}",
                params={
                    "source": db,
                    "format": "json",
                    "per_page": 1000,
                    "page": page,
                },
            )
            resp.raise_for_status()
            payload = resp.json()
            if not isinstance(payload, list) or len(payload) < 2 or payload[1] is None:
                break
            meta, records = payload[0], payload[1]
            for r in records:
                iso3 = r.get("countryiso3code", "")
                if len(iso3) != 3:
                    continue
                if r.get("value") is None:
                    continue
                try:
                    year = int(r["date"])
                except (ValueError, TypeError):
                    continue
                rows.append({"economy": iso3, "year": year, "value": r["value"]})
            if page >= int(meta.get("pages", 1)):
                break
            page += 1
    return rows


def fetch_and_store_indicator(indicator_id: str, wb_db_id: int, sql_uri: str) -> int:
    """Fetch one WB indicator and replace any prior copy in Postgres.

    The function is idempotent: it first deletes all existing rows for the
    ``(indicator_id, db_id)`` pair, then inserts the fresh fetch in a single
    transaction.

    Args:
        indicator_id: World Bank indicator id.
        wb_db_id: World Bank database id.
        sql_uri: SQLAlchemy URI for the Postgres superuser connection.

    Returns:
        Number of rows that were inserted.

    Raises:
        ValueError: When neither the primary nor the fallback endpoint
            returns any usable data for the indicator.
    """
    try:
        records = wb.data.fetch(
            indicator_id,
            db=wb_db_id,
            skipAggs=True,
            economy="all",
            time="all",
            skipBlanks=False,
            numericTimeKeys=True,
        )
        rows = list(records)
    except Exception as exc:
        logger.warning(
            "wbgapi primary fetch failed for indicator_id=%s db=%s: %s; "
            "falling back to v2 REST endpoint",
            indicator_id,
            wb_db_id,
            exc,
        )
        rows = []

    if rows:
        df = pl.DataFrame(rows)
        economy_column = "economy"
        year_column = "time"

        df_transformed = df.select(
            [
                pl.col(economy_column).alias("economy"),
                pl.col(year_column).alias("year"),
                pl.col("value"),
            ]
        ).with_columns(
            [
                pl.lit(indicator_id).alias("indicator_id"),
                pl.lit(wb_db_id).alias("db_id"),
            ]
        )
    else:
        fallback_rows = _fetch_indicator_data_via_api(indicator_id, wb_db_id)
        if not fallback_rows:
            raise ValueError(f"No data found for indicator id: {indicator_id}")
        df_transformed = pl.DataFrame(fallback_rows).with_columns(
            [
                pl.lit(indicator_id).alias("indicator_id"),
                pl.lit(wb_db_id).alias("db_id"),
            ]
        )

    df_transformed = df_transformed.drop_nulls(subset=["economy", "year"])

    df_transformed = df_transformed.with_columns(
        [
            pl.col("year").cast(pl.Int32, strict=False),
            pl.col("value").cast(pl.Float64, strict=False),
        ]
    )

    if df_transformed.is_empty():
        raise ValueError(
            f"No non-null rows found for indicator id: {indicator_id} in db: {wb_db_id}"
        )

    df_transformed = df_transformed.unique(
        subset=["economy", "year", "indicator_id", "db_id"],
        keep="last",
        maintain_order=True,
    )

    rows_to_insert = df_transformed.to_dicts()
    engine = create_engine(sql_uri)
    try:
        with Session(engine) as session, session.begin():
            session.execute(
                delete(MacroIndicator).where(
                    MacroIndicator.indicator_id == indicator_id,
                    MacroIndicator.db_id == wb_db_id,
                )
            )
            session.add_all([MacroIndicator(**row) for row in rows_to_insert])
    finally:
        engine.dispose()

    return len(rows_to_insert)
