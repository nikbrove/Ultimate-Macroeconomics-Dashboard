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


PAGE_TITLE = "Demography"
AGE_0_14_INDICATOR_ID = "SP.POP.0014.TO.ZS"
AGE_15_64_INDICATOR_ID = "SP.POP.1564.TO.ZS"
AGE_65_INDICATOR_ID = "SP.POP.65UP.TO.ZS"


POPULATION_INDICATOR_ID = "SP.POP.TOTL"
POPULATION_GROWTH_INDICATOR_ID = "SP.POP.GROW"
LABOR_FORCE_INDICATOR_ID = "SL.TLF.TOTL.IN"
MALE_POPULATION_INDICATOR_ID = "SP.POP.TOTL.MA.IN"
FEMALE_POPULATION_INDICATOR_ID = "SP.POP.TOTL.FE.IN"


def _prepare_indicator_slice(indicator_id: str, value_col: str) -> pl.DataFrame:
    indicator_df = get_world_bank_indicator(indicator_id, country_code="ALL")
    required_cols = {"year", "economy", "value"}
    if indicator_df.is_empty() or not required_cols.issubset(set(indicator_df.columns)):
        return pl.DataFrame()

    return (
        indicator_df.select(
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


def _render_demography_bubble() -> None:
    st.subheader("Population Bubble Explorer")
    st.caption(
        "Bubble size reflects total population. Hover includes population growth, "
        "labor force, and male/female population totals."
    )

    total_population_df = _prepare_indicator_slice(
        POPULATION_INDICATOR_ID,
        value_col="total_population",
    )
    population_growth_df = _prepare_indicator_slice(
        POPULATION_GROWTH_INDICATOR_ID,
        value_col="population_growth",
    )
    labor_force_df = _prepare_indicator_slice(
        LABOR_FORCE_INDICATOR_ID,
        value_col="labor_force",
    )
    male_population_df = _prepare_indicator_slice(
        MALE_POPULATION_INDICATOR_ID,
        value_col="male_population",
    )
    female_population_df = _prepare_indicator_slice(
        FEMALE_POPULATION_INDICATOR_ID,
        value_col="female_population",
    )

    joined_df = (
        total_population_df.join(
            population_growth_df, on=["year", "economy"], how="inner"
        )
        .join(labor_force_df, on=["year", "economy"], how="inner")
        .join(male_population_df, on=["year", "economy"], how="inner")
        .join(female_population_df, on=["year", "economy"], how="inner")
    )

    if joined_df.is_empty():
        st.info(
            "Demography bubble chart is unavailable because source data is incomplete."
        )
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
        pl.coalesce([pl.col("country_name"), pl.col("economy")]).alias("country_name")
    )

    year_options = (
        joined_df.select("year").unique().sort("year").get_column("year").to_list()
    )
    if not year_options:
        st.info("Demography bubble chart is unavailable because years are missing.")
        st.divider()
        return

    selected_year = st.select_slider(
        "Bubble chart year",
        options=year_options,
        value=year_options[-1],
        key="demography_bubble_year",
    )

    year_df = joined_df.filter(pl.col("year") == int(selected_year)).sort(
        "total_population", descending=True
    )
    if year_df.is_empty():
        st.info("No demography observations are available for this year.")
        st.divider()
        return

    plot_df = year_df.to_pandas()
    fig = px.scatter(
        plot_df,
        x="population_growth",
        y="labor_force",
        size="total_population",
        color="country_name",
        hover_name="country_name",
        hover_data={
            "economy": True,
            "population_growth": ":.2f",
            "labor_force": ":,.0f",
            "male_population": ":,.0f",
            "female_population": ":,.0f",
            "total_population": ":,.0f",
        },
        labels={
            "population_growth": "Population growth (% annual)",
            "labor_force": "Labor force (people)",
            "total_population": "Total population",
            "country_name": "Country",
        },
        size_max=55,
        title=f"Population Growth vs Labor Force ({selected_year})",
    )
    fig.update_traces(
        marker={
            "opacity": 0.78,
            "line": {"width": 0.5, "color": "rgba(255, 255, 255, 0.7)"},
        }
    )
    fig.update_layout(showlegend=False)
    fig.update_yaxes(tickformat=",.2s")
    fig = apply_plotly_theme(fig)

    st.plotly_chart(fig, width="stretch")
    st.divider()


def _render_age_structure_deep_dive() -> None:
    st.divider()
    st.subheader("Age Structure (ternary plot)")
    st.caption(
        "Every country plotted on a triangle whose corners are the three age "
        "shares — Young (0-14), Working (15-64), Elderly (65+). Latest "
        "available year. Selected countries are highlighted and labelled."
    )

    young = _prepare_indicator_slice(AGE_0_14_INDICATOR_ID, value_col="young")
    working = _prepare_indicator_slice(AGE_15_64_INDICATOR_ID, value_col="working")
    elderly = _prepare_indicator_slice(AGE_65_INDICATOR_ID, value_col="elderly")

    if young.is_empty() or working.is_empty() or elderly.is_empty():
        st.info("Age structure ternary is unavailable - source data missing.")
        return

    joined = young.join(working, on=["year", "economy"]).join(
        elderly, on=["year", "economy"]
    )
    if joined.is_empty():
        st.info("No overlapping age-structure observations.")
        return

    latest_year = int(joined.select(pl.col("year").max()).item())
    snapshot = joined.filter(pl.col("year") == latest_year)

    country_map = get_world_bank_country_mapping()
    if not country_map.is_empty() and {"id", "value"}.issubset(set(country_map.columns)):
        country_map = country_map.select(
            [
                pl.col("id").cast(pl.Utf8).str.to_uppercase().alias("economy"),
                pl.col("value").cast(pl.Utf8).alias("country_name"),
            ]
        )
        snapshot = snapshot.join(country_map, on="economy", how="left")
    snapshot = snapshot.with_columns(
        pl.col("country_name").fill_null(pl.col("economy")).alias("country_name")
    )

    selected_iso_codes = {
        str(c).strip().upper()
        for c in st.session_state.get(f"{PAGE_TITLE}_countries", [])
        if str(c).strip()
    }
    snapshot = snapshot.with_columns(
        pl.when(pl.col("economy").is_in(list(selected_iso_codes)))
        .then(pl.lit("Selected"))
        .otherwise(pl.lit("Other"))
        .alias("group")
    )

    plot_df = snapshot.to_pandas()

    fig = px.scatter_ternary(
        plot_df,
        a="young",
        b="working",
        c="elderly",
        color="group",
        category_orders={"group": ["Other", "Selected"]},
        color_discrete_map={
            "Other": get_color("reference_line"),
            "Selected": get_colorway()[0],
        },
        hover_name="country_name",
        hover_data={
            "economy": True,
            "young": ":.1f",
            "working": ":.1f",
            "elderly": ":.1f",
            "group": False,
        },
        title=f"Age structure ({latest_year})",
    )

    selected_df = plot_df[plot_df["group"] == "Selected"]
    if not selected_df.empty:
        fig.add_trace(
            go.Scatterternary(
                a=selected_df["young"],
                b=selected_df["working"],
                c=selected_df["elderly"],
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
    fig.update_layout(
        ternary={
            "aaxis": {"title": "Young 0-14 (%)"},
            "baxis": {"title": "Working 15-64 (%)"},
            "caxis": {"title": "Elderly 65+ (%)"},
        }
    )
    fig = apply_plotly_theme(fig)

    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Top vertex = young countries (high 0-14 share); bottom-left = working-"
        "age heavy; bottom-right = aged societies. The demographic transition "
        "moves countries clockwise from top down to the right."
    )


render_page_from_config(
    page_title=PAGE_TITLE,
    section_keys=["Demography"],
    caption=(
        "Explore population size, structure, and demographic dynamics to connect "
        "labor and social trends with macroeconomic outcomes."
    ),
    before_graphs_renderer=_render_demography_bubble,
    after_graphs_renderer=_render_age_structure_deep_dive,
)
