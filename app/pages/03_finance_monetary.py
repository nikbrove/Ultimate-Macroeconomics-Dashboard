import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import streamlit as st

from core.plotting import apply_plotly_theme
from core.theming import get_color, get_colorway
from core.postgres_client import (
    get_world_bank_country_mapping,
    get_world_bank_indicator,
)
from pages.page_utils import render_page_from_config


PAGE_TITLE = "Finance and Monetary"
INFLATION_INDICATOR_ID = "FP.CPI.TOTL.ZG"
REAL_RATE_INDICATOR_ID = "FR.INR.RINR"


def _prepare_indicator_slice(df: pl.DataFrame, value_col: str) -> pl.DataFrame:
    required_cols = {"year", "economy", "value"}
    if df.is_empty() or not required_cols.issubset(set(df.columns)):
        return pl.DataFrame()

    return (
        df.select(
            [
                pl.col("year").cast(pl.Int64, strict=False).alias("year"),
                pl.col("economy").cast(pl.Utf8).str.to_uppercase().alias("economy"),
                pl.col("value").cast(pl.Float64, strict=False).alias(value_col),
            ]
        )
        .filter(
            pl.col("year").is_not_null()
            & pl.col("economy").is_not_null()
            & pl.col(value_col).is_not_null()
        )
        .group_by(["year", "economy"])
        .agg(pl.col(value_col).mean().alias(value_col))
        .sort(["year", "economy"])
    )


def _render_inflation_vs_rate_overview() -> None:
    st.subheader("Inflation vs Real Rate")
    st.caption(
        "Compares CPI inflation against the real lending interest rate across "
        "countries. Quadrants separate monetary-policy stances: top-right "
        "(high inflation, positive real rates) typically signals tight policy "
        "fighting price pressures."
    )

    inflation_df = _prepare_indicator_slice(
        get_world_bank_indicator(INFLATION_INDICATOR_ID, country_code="ALL"),
        value_col="inflation_pct",
    )
    rate_df = _prepare_indicator_slice(
        get_world_bank_indicator(REAL_RATE_INDICATOR_ID, country_code="ALL"),
        value_col="real_rate_pct",
    )

    if inflation_df.is_empty() or rate_df.is_empty():
        st.info(
            "Inflation/interest scatter is unavailable because source data is empty."
        )
        st.divider()
        return

    joined_df = inflation_df.join(rate_df, on=["year", "economy"], how="inner")
    if joined_df.is_empty():
        st.info("No overlapping inflation and interest-rate values were found.")
        st.divider()
        return

    country_map = get_world_bank_country_mapping()
    if not country_map.is_empty() and {"id", "value"}.issubset(
        set(country_map.columns)
    ):
        country_map = country_map.select(
            [
                pl.col("id").cast(pl.Utf8).str.to_uppercase().alias("economy"),
                pl.col("value").cast(pl.Utf8).alias("country_name"),
            ]
        )
        joined_df = joined_df.join(country_map, on="economy", how="left")

    joined_df = joined_df.with_columns(
        pl.col("country_name").fill_null(pl.col("economy")).alias("country_name"),
    )

    year_options = (
        joined_df.select("year").unique().sort("year").get_column("year").to_list()
    )
    if not year_options:
        st.info("Inflation/interest scatter is unavailable because years are missing.")
        st.divider()
        return

    selected_year = st.select_slider(
        "Scatter year",
        options=year_options,
        value=year_options[-1],
        key="finmon_inflation_rate_year",
    )

    year_df = joined_df.filter(pl.col("year") == int(selected_year))
    if year_df.is_empty():
        st.info("No inflation/interest observations are available for this year.")
        st.divider()
        return

    selected_countries = {
        str(code).strip().upper()
        for code in st.session_state.get(f"{PAGE_TITLE}_countries", [])
        if str(code).strip()
    }
    year_df = year_df.with_columns(
        pl.when(pl.col("economy").is_in(list(selected_countries)))
        .then(pl.lit("Selected"))
        .otherwise(pl.lit("Other"))
        .alias("country_group")
    )

    plot_df = year_df.to_pandas()

    fig = px.scatter(
        plot_df,
        x="inflation_pct",
        y="real_rate_pct",
        color="country_group",
        category_orders={"country_group": ["Other", "Selected"]},
        color_discrete_map={
            "Other": get_color("reference_line"),
            "Selected": get_colorway()[0],
        },
        hover_name="country_name",
        hover_data={
            "economy": True,
            "inflation_pct": ":.2f",
            "real_rate_pct": ":.2f",
            "country_group": False,
        },
        labels={
            "inflation_pct": "CPI inflation (%)",
            "real_rate_pct": "Real interest rate (%)",
            "country_group": "",
        },
        title=f"Inflation vs Real Rate ({selected_year})",
    )

    selected_df = plot_df[plot_df["country_group"] == "Selected"]
    if not selected_df.empty:
        fig.add_trace(
            go.Scatter(
                x=selected_df["inflation_pct"],
                y=selected_df["real_rate_pct"],
                mode="text",
                text=selected_df["economy"],
                textposition="top center",
                showlegend=False,
                hoverinfo="skip",
            )
        )

    fig.add_hline(
        y=0,
        line={"color": get_color("reference_line"), "dash": "dash", "width": 1.2},
    )
    fig.add_vline(
        x=0,
        line={"color": get_color("reference_line"), "dash": "dash", "width": 1.2},
    )

    fig.update_traces(
        selector={"mode": "markers"},
        marker={"size": 9, "opacity": 0.78, "line": {"width": 0.5}},
    )
    fig = apply_plotly_theme(fig)

    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Reference lines at 0 split the plane into monetary-policy quadrants. "
        "Selected countries from the multiselect above are highlighted and labelled."
    )
    st.divider()


POPULATION_INDICATOR_ID = "SP.POP.TOTL"
HEATMAP_TOP_N = 30
HEATMAP_YEARS = 25
INFLATION_CLIP = 50.0


def _render_inflation_heatmap_deep_dive() -> None:
    st.divider()
    st.subheader("Inflation Heatmap")
    st.caption(
        f"CPI inflation (%) for the {HEATMAP_TOP_N} most populous economies "
        f"over the last {HEATMAP_YEARS} years. Colour is clipped to "
        f"±{INFLATION_CLIP:.0f}% so hyperinflation episodes don't drown out "
        "everything else. Countries from your multiselect are outlined."
    )

    inflation_df = _prepare_indicator_slice(
        get_world_bank_indicator(INFLATION_INDICATOR_ID, country_code="ALL"),
        value_col="cpi",
    )
    population_df = _prepare_indicator_slice(
        get_world_bank_indicator(POPULATION_INDICATOR_ID, country_code="ALL"),
        value_col="pop",
    )

    if inflation_df.is_empty() or population_df.is_empty():
        st.info("Inflation heatmap is unavailable - source data missing.")
        return

    latest_pop_year = int(population_df.select(pl.col("year").max()).item())
    latest_pop = (
        population_df.filter(pl.col("year") == latest_pop_year)
        .sort("pop", descending=True)
        .head(HEATMAP_TOP_N)
    )
    top_economies = latest_pop.get_column("economy").to_list()

    latest_inf_year = int(inflation_df.select(pl.col("year").max()).item())
    start_year = latest_inf_year - HEATMAP_YEARS + 1

    grid = inflation_df.filter(
        pl.col("economy").is_in(top_economies)
        & (pl.col("year") >= start_year)
        & (pl.col("year") <= latest_inf_year)
    )

    if grid.is_empty():
        st.info("No inflation observations in the heatmap window.")
        return

    country_map = get_world_bank_country_mapping()
    name_by_iso: dict[str, str] = {}
    if not country_map.is_empty() and {"id", "value"}.issubset(set(country_map.columns)):
        for row in country_map.to_dicts():
            iso = str(row.get("id", "")).strip().upper()
            name = str(row.get("value", "")).strip()
            if iso and name:
                name_by_iso[iso] = name

    ordered_econ = (
        latest_pop.get_column("economy").to_list()
    )

    years = list(range(start_year, latest_inf_year + 1))
    matrix_df = grid.pivot(
        values="cpi", index="economy", on="year", aggregate_function="mean"
    )

    z = []
    y_labels = []
    selected_iso_codes = {
        str(c).strip().upper()
        for c in st.session_state.get(f"{PAGE_TITLE}_countries", [])
        if str(c).strip()
    }
    matrix_dict = {row["economy"]: row for row in matrix_df.to_dicts()}
    for econ in ordered_econ:
        row = matrix_dict.get(econ)
        if row is None:
            continue
        z.append([row.get(str(year)) for year in years])
        country_name = name_by_iso.get(econ, econ)
        marker = " ★" if econ in selected_iso_codes else ""
        y_labels.append(f"{country_name} ({econ}){marker}")

    if not z:
        st.info("No inflation matrix could be built.")
        return

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=years,
            y=y_labels,
            zmin=-INFLATION_CLIP,
            zmax=INFLATION_CLIP,
            colorscale="RdBu_r",
            colorbar={"title": "CPI %"},
            hovertemplate="%{y}<br>%{x}: %{z:.2f}%<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"Inflation Heatmap ({start_year}–{latest_inf_year})",
        xaxis_title="Year",
        yaxis_title="",
        height=max(420, 22 * len(y_labels) + 100),
        margin={"l": 40, "r": 20, "t": 60, "b": 40},
    )
    fig.update_yaxes(autorange="reversed")
    fig = apply_plotly_theme(fig)

    st.plotly_chart(fig, width="stretch")
    st.caption(
        "★ marks countries selected in the multiselect above. Deep red = "
        "hyperinflation (>+50% clipped); deep blue = deflation (<-50% clipped). "
        "Rows are ordered by population (largest first)."
    )


render_page_from_config(
    page_title=PAGE_TITLE,
    section_keys=["Finance and Monetary", "Fiscal"],
    caption=(
        "Monitor monetary, fiscal, and financial indicators to compare policy "
        "stance and macro-financial stability across economies."
    ),
    before_graphs_renderer=_render_inflation_vs_rate_overview,
    after_graphs_renderer=_render_inflation_heatmap_deep_dive,
)
