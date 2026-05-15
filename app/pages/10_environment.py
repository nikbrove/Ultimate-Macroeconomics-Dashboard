import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import streamlit as st

from core.plotting import apply_plotly_theme
from core.theming import get_color, get_colorway
from core.postgres_client import (
    get_world_bank_country_mapping,
    get_world_bank_country_regions,
    get_world_bank_indicator,
)
from pages.page_utils import render_page_from_config


PAGE_TITLE = "Environment and Sustainability"
RENEWABLE_INDICATOR_ID = "EG.FEC.RNEW.ZS"
PM25_INDICATOR_ID = "EN.ATM.PM25.MC.M3"


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


def _render_renewable_vs_pm25_overview() -> None:
    st.subheader("Renewable Energy vs PM2.5 Air Pollution")
    st.caption(
        "Compares the share of renewables in final energy consumption with "
        "average ambient PM2.5 pollution. The expected pattern is a negative "
        "slope — cleaner energy mixes coincide with cleaner air — but the "
        "relationship is noisy because of geography, density, and industry mix."
    )

    renewable_df = _prepare_indicator_slice(
        get_world_bank_indicator(RENEWABLE_INDICATOR_ID, country_code="ALL"),
        value_col="renewable_pct",
    )
    pm25_df = _prepare_indicator_slice(
        get_world_bank_indicator(PM25_INDICATOR_ID, country_code="ALL"),
        value_col="pm25_ugm3",
    )

    if renewable_df.is_empty() or pm25_df.is_empty():
        st.info(
            "Renewable / PM2.5 scatter is unavailable because source data is empty."
        )
        st.divider()
        return

    joined_df = renewable_df.join(pm25_df, on=["year", "economy"], how="inner")
    if joined_df.is_empty():
        st.info("No overlapping renewable and PM2.5 values were found.")
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
        st.info("Renewable / PM2.5 scatter is unavailable because years are missing.")
        st.divider()
        return

    selected_year = st.select_slider(
        "Scatter year",
        options=year_options,
        value=year_options[-1],
        key="env_renewable_pm25_year",
    )

    year_df = joined_df.filter(pl.col("year") == int(selected_year))
    if year_df.is_empty():
        st.info("No renewable / PM2.5 observations are available for this year.")
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
        x="renewable_pct",
        y="pm25_ugm3",
        color="country_group",
        category_orders={"country_group": ["Other", "Selected"]},
        color_discrete_map={
            "Other": get_color("reference_line"),
            "Selected": get_colorway()[0],
        },
        hover_name="country_name",
        hover_data={
            "economy": True,
            "renewable_pct": ":.2f",
            "pm25_ugm3": ":.2f",
            "country_group": False,
        },
        labels={
            "renewable_pct": "Renewable energy (% of final consumption)",
            "pm25_ugm3": "PM2.5 (µg/m³, lower is better)",
            "country_group": "",
        },
        title=f"Renewable Energy vs PM2.5 ({selected_year})",
    )

    selected_df = plot_df[plot_df["country_group"] == "Selected"]
    if not selected_df.empty:
        fig.add_trace(
            go.Scatter(
                x=selected_df["renewable_pct"],
                y=selected_df["pm25_ugm3"],
                mode="text",
                text=selected_df["economy"],
                textposition="top center",
                showlegend=False,
                hoverinfo="skip",
            )
        )

    fig.update_traces(
        selector={"mode": "markers"},
        marker={"size": 9, "opacity": 0.78, "line": {"width": 0.5}},
    )
    fig = apply_plotly_theme(fig)

    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Lower Y is healthier air. Selected countries from the multiselect "
        "above are highlighted and labelled."
    )
    st.divider()


GHG_PC_INDICATOR_ID = "EN.GHG.ALL.PC.CE.AR5"
GHG_TOTAL_INDICATOR_ID = "EN.GHG.ALL.MT.CE.AR5"
GDP_PC_INDICATOR_ID = "NY.GDP.PCAP.CD"


def _render_kuznets_curve_deep_dive() -> None:
    st.divider()
    st.subheader("Environmental Kuznets Curve (animation)")
    st.caption(
        "X = GDP per capita (current US$, log). Y = total greenhouse-gas "
        "emissions per capita (t CO2e/capita). Bubble size = total emissions "
        "(Mt). Animation frame = year. Watch the inverted-U / sideways drift "
        "as economies decouple growth from emissions intensity."
    )

    ghg_pc = _prepare_indicator_slice(
        get_world_bank_indicator(GHG_PC_INDICATOR_ID, country_code="ALL"),
        value_col="ghg_pc",
    )
    ghg_total = _prepare_indicator_slice(
        get_world_bank_indicator(GHG_TOTAL_INDICATOR_ID, country_code="ALL"),
        value_col="ghg_total",
    )
    gdp_pc = _prepare_indicator_slice(
        get_world_bank_indicator(GDP_PC_INDICATOR_ID, country_code="ALL"),
        value_col="gdp_pc",
    )
    regions_df = get_world_bank_country_regions()

    if any(df.is_empty() for df in (ghg_pc, ghg_total, gdp_pc)) or regions_df.is_empty():
        st.info("Kuznets-curve animation is unavailable - source data missing.")
        return

    regions_df = regions_df.select(
        [
            pl.col("id").cast(pl.Utf8).str.to_uppercase().alias("economy"),
            pl.col("value").cast(pl.Utf8).alias("country_name"),
            pl.col("region").cast(pl.Utf8).alias("region"),
        ]
    )

    joined = (
        ghg_pc.join(ghg_total, on=["year", "economy"])
        .join(gdp_pc, on=["year", "economy"])
        .join(regions_df, on="economy", how="inner")
        .filter(
            (pl.col("ghg_pc") > 0)
            & (pl.col("ghg_total") > 0)
            & (pl.col("gdp_pc") > 0)
        )
        .sort(["year", "economy"])
    )
    if joined.is_empty():
        st.info("No overlapping observations for the Kuznets curve.")
        return

    plot_df = joined.to_pandas().sort_values(["year", "country_name"])
    fig = px.scatter(
        plot_df,
        x="gdp_pc",
        y="ghg_pc",
        size="ghg_total",
        color="region",
        animation_frame="year",
        animation_group="economy",
        hover_name="country_name",
        hover_data={
            "economy": True,
            "gdp_pc": ":,.0f",
            "ghg_pc": ":.2f",
            "ghg_total": ":,.1f",
            "region": True,
            "year": False,
        },
        log_x=True,
        size_max=55,
        range_x=[
            max(100.0, float(plot_df["gdp_pc"].min()) * 0.8),
            float(plot_df["gdp_pc"].max()) * 1.2,
        ],
        range_y=[
            0.0,
            float(plot_df["ghg_pc"].quantile(0.99)) * 1.1,
        ],
        labels={
            "gdp_pc": "GDP per capita (US$, log)",
            "ghg_pc": "GHG emissions per capita (t CO2e)",
            "ghg_total": "Total emissions (Mt CO2e)",
            "region": "Region",
        },
        title="Income vs Emissions intensity",
    )
    fig.update_traces(marker={"opacity": 0.7, "line": {"width": 0.5}})
    fig = apply_plotly_theme(fig)

    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Y-axis clipped at the 99th percentile so a few extreme outliers don't "
        "compress the rest of the chart. The classic Kuznets-curve hypothesis "
        "is that emissions per capita rise with income, peak, then decline as "
        "economies decarbonise."
    )


render_page_from_config(
    page_title=PAGE_TITLE,
    section_keys=["Environment and ecology"],
    caption=(
        "Track environmental pressures and sustainability signals that interact "
        "with growth, risk, and long-term development."
    ),
    before_graphs_renderer=_render_renewable_vs_pm25_overview,
    after_graphs_renderer=_render_kuznets_curve_deep_dive,
)
