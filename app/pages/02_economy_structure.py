import plotly.graph_objects as go
import polars as pl
import streamlit as st

from core.page_helpers import fetch_indicator_slice
from core.plotting import apply_plotly_theme
from core.theming import get_color
from core.postgres_client import (
    get_world_bank_country_codes,
    get_world_bank_country_mapping,
)
from pages.page_utils import render_page_from_config


ECONOMY_STRUCTURE_INDICATORS = [
    ("Agriculture", "NV.AGR.TOTL.ZS", "sector_agriculture"),
    ("Manufacturing", "NV.IND.MANF.ZS", "sector_manufacturing"),
    ("Services", "NV.SRV.TOTL.ZS", "sector_services"),
]
PAGE_TITLE = "Economy Structure"


def _build_country_labels() -> tuple[list[str], dict[str, str], dict[str, str]]:
    country_options = sorted(get_world_bank_country_codes())
    country_mapping_df = get_world_bank_country_mapping()
    label_by_iso: dict[str, str] = {}
    name_by_iso: dict[str, str] = {}
    if not country_mapping_df.is_empty() and {"id", "value"}.issubset(
        set(country_mapping_df.columns)
    ):
        for row in country_mapping_df.to_dicts():
            iso = str(row.get("id", "")).strip().upper()
            name = str(row.get("value", "")).strip()
            if iso and name:
                label_by_iso[iso] = f"{name} ({iso})"
                name_by_iso[iso] = name
    return country_options, label_by_iso, name_by_iso


def _resolve_default_structure_country(country_options: list[str]) -> str | None:
    if not country_options:
        return None

    selected_trend_countries = st.session_state.get(f"{PAGE_TITLE}_countries", [])
    if selected_trend_countries:
        first_selected = str(selected_trend_countries[0]).strip().upper()
        if first_selected in country_options:
            return first_selected

    if "USA" in country_options:
        return "USA"

    return country_options[0]


def _build_economy_structure_data(
    country_code: str,
) -> tuple[pl.DataFrame, int | None]:
    latest_common_year_df: pl.DataFrame | None = None

    for sector_name, indicator_id, _ in ECONOMY_STRUCTURE_INDICATORS:
        sector_df = fetch_indicator_slice(indicator_id, country_code=country_code)
        if sector_df.is_empty():
            return pl.DataFrame(), None

        sector_year_df = sector_df.select(
            [
                pl.col("year"),
                pl.col("value").alias(sector_name),
            ]
        )

        if latest_common_year_df is None:
            latest_common_year_df = sector_year_df
        else:
            latest_common_year_df = latest_common_year_df.join(
                sector_year_df,
                on="year",
                how="inner",
            )

    if latest_common_year_df is None or latest_common_year_df.is_empty():
        return pl.DataFrame(), None

    latest_row = latest_common_year_df.sort("year").tail(1)
    latest_year = int(latest_row.get_column("year")[0])
    structure_df = pl.DataFrame(
        {
            "sector": [item[0] for item in ECONOMY_STRUCTURE_INDICATORS],
            "indicator_id": [item[1] for item in ECONOMY_STRUCTURE_INDICATORS],
            "color": [get_color(item[2]) for item in ECONOMY_STRUCTURE_INDICATORS],
            "value": [
                float(latest_row.get_column(item[0])[0])
                for item in ECONOMY_STRUCTURE_INDICATORS
            ],
        }
    ).filter(pl.col("value").is_not_null() & (pl.col("value") >= 0))

    return structure_df, latest_year


def _build_economy_structure_pie(
    structure_df: pl.DataFrame,
    country_name: str,
    year: int,
) -> go.Figure:
    fig = go.Figure(
        data=[
            go.Pie(
                labels=structure_df["sector"].to_list(),
                values=structure_df["value"].to_list(),
                sort=False,
                hole=0.35,
                marker=dict(colors=structure_df["color"].to_list()),
                texttemplate="%{label}<br>%{value:.1f}%",
                textinfo="text",
                customdata=structure_df["indicator_id"].to_list(),
                hovertemplate=(
                    "%{label}<br>%{value:.2f}% of GDP"
                    "<br>Indicator: %{customdata}<extra></extra>"
                ),
            )
        ]
    )
    fig.update_layout(
        title=f"Economic Structure of {country_name} ({year})",
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return apply_plotly_theme(fig)


def _render_economy_structure_section() -> None:
    st.subheader("Economy Structure")

    country_options, label_by_iso, name_by_iso = _build_country_labels()
    if not country_options:
        st.info("Country selector is unavailable right now.")
        return

    default_country = _resolve_default_structure_country(country_options)
    if default_country is None:
        st.info("Country selector is unavailable right now.")
        return

    selected_country = st.selectbox(
        "Country for economy structure",
        options=country_options,
        index=country_options.index(default_country),
        format_func=lambda iso: label_by_iso.get(str(iso).upper(), str(iso).upper()),
        key="economy_structure_country",
    )

    structure_df, latest_year = _build_economy_structure_data(selected_country)
    if structure_df.is_empty() or latest_year is None:
        st.info(
            "No common year with all three structure indicators is available for the selected country."
        )
        return

    country_name = name_by_iso.get(selected_country, selected_country)
    st.plotly_chart(
        _build_economy_structure_pie(
            structure_df,
            country_name=country_name,
            year=latest_year,
        ),
        width="stretch",
    )
    st.caption(
        "Uses the latest year where agriculture, manufacturing, and services values are all available. "
        "The pie is normalized across these three indicators, while slice labels show each indicator's original value as a percent of GDP."
    )

    st.divider()


def _build_sector_timeseries(country_code: str | None) -> tuple[pl.DataFrame, str | None]:
    rows: list[pl.DataFrame] = []
    for sector_name, indicator_id, _ in ECONOMY_STRUCTURE_INDICATORS:
        if country_code is None:
            slice_df = fetch_indicator_slice(indicator_id, country_code="ALL")
            if slice_df.is_empty():
                continue
            agg_df = slice_df.group_by("year").agg(pl.col("value").mean().alias(sector_name))
            slice_df = agg_df
        else:
            slice_df = fetch_indicator_slice(indicator_id, country_code=country_code)
            if slice_df.is_empty():
                continue
            slice_df = slice_df.select(["year", pl.col("value").alias(sector_name)])

        rows.append(slice_df)

    if not rows:
        return pl.DataFrame(), None

    panel = rows[0]
    for extra in rows[1:]:
        panel = panel.join(extra, on="year", how="inner")

    panel = panel.sort("year")
    if panel.is_empty():
        return pl.DataFrame(), country_code
    return panel, country_code


def _build_sector_area(
    panel: pl.DataFrame, title: str
) -> go.Figure:
    fig = go.Figure()
    for sector_name, _, color_token in ECONOMY_STRUCTURE_INDICATORS:
        if sector_name not in panel.columns:
            continue
        fig.add_trace(
            go.Scatter(
                x=panel["year"].to_list(),
                y=panel[sector_name].to_list(),
                mode="lines",
                stackgroup="one",
                name=sector_name,
                line={"color": get_color(color_token), "width": 0.5},
                hovertemplate=f"%{{x}}<br>{sector_name}: %{{y:.2f}}% of GDP<extra></extra>",
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="Year",
        yaxis_title="% of GDP (stacked)",
        margin={"l": 40, "r": 20, "t": 50, "b": 40},
    )
    return apply_plotly_theme(fig)


def _render_sector_trajectory_deep_dive() -> None:
    st.divider()
    st.subheader("Sector Trajectory (1960 → today)")
    st.caption(
        "Stacked-area view of how each economy has rebalanced agriculture, "
        "manufacturing, and services value-added over time. One small-multiple "
        "per selected country (up to 4); if no countries are selected the chart "
        "shows the global cross-country mean."
    )

    selected_codes = [
        str(code).strip().upper()
        for code in st.session_state.get(f"{PAGE_TITLE}_countries", [])
        if str(code).strip()
    ][:4]
    _, label_by_iso, name_by_iso = _build_country_labels()

    if not selected_codes:
        panel, _ = _build_sector_timeseries(None)
        if panel.is_empty():
            st.info("Sector trajectory data is unavailable.")
            return
        fig = _build_sector_area(panel, title="Global cross-country mean — sector shares of GDP")
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Showing the global cross-country mean. Select countries above to "
            "compare individual trajectories side by side."
        )
        return

    columns = st.columns(min(2, len(selected_codes)))
    for index, code in enumerate(selected_codes):
        panel, _ = _build_sector_timeseries(code)
        if panel.is_empty():
            with columns[index % len(columns)]:
                st.info(f"No sector data available for {code}.")
            continue
        country_name = name_by_iso.get(code, code)
        with columns[index % len(columns)]:
            fig = _build_sector_area(panel, title=f"{country_name} ({code})")
            st.plotly_chart(fig, width="stretch")


render_page_from_config(
    page_title=PAGE_TITLE,
    section_keys=["Structure"],
    caption=(
        "Explore the structure of national economies: agriculture, manufacturing, "
        "and services as a share of GDP."
    ),
    before_graphs_renderer=_render_economy_structure_section,
    after_graphs_renderer=_render_sector_trajectory_deep_dive,
)
