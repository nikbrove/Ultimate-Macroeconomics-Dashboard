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


PAGE_TITLE = "Health and wellbeing"
LIFE_EXP_FEMALE_ID = "SP.DYN.LE00.FE.IN"
LIFE_EXP_MALE_ID = "SP.DYN.LE00.MA.IN"
TOP_N_GAPS = 15


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


def _render_life_expectancy_gap_overview() -> None:
    st.subheader("Life Expectancy Gender Gap")
    st.caption(
        "Female minus male life expectancy at birth in years. Positive values "
        "mean women outlive men, the typical worldwide pattern."
    )

    female_df = _prepare_indicator_slice(
        get_world_bank_indicator(LIFE_EXP_FEMALE_ID, country_code="ALL"),
        value_col="life_exp_female",
    )
    male_df = _prepare_indicator_slice(
        get_world_bank_indicator(LIFE_EXP_MALE_ID, country_code="ALL"),
        value_col="life_exp_male",
    )

    if female_df.is_empty() or male_df.is_empty():
        st.info(
            "Life-expectancy gap bar is unavailable because source data is empty."
        )
        st.divider()
        return

    joined_df = female_df.join(male_df, on=["year", "economy"], how="inner")
    if joined_df.is_empty():
        st.info("No overlapping male/female life-expectancy values were found.")
        st.divider()
        return

    joined_df = joined_df.with_columns(
        (pl.col("life_exp_female") - pl.col("life_exp_male")).alias("gap_years")
    )

    year_options = (
        joined_df.select("year").unique().sort("year").get_column("year").to_list()
    )
    if not year_options:
        st.info(
            "Life-expectancy gap bar is unavailable because years are missing."
        )
        st.divider()
        return

    selected_year = st.select_slider(
        "Snapshot year",
        options=year_options,
        value=year_options[-1],
        key="health_lifeexp_gap_year",
    )

    year_df = joined_df.filter(pl.col("year") == int(selected_year))
    if year_df.is_empty():
        st.info("No life-expectancy gap observations are available for this year.")
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
        year_df = year_df.join(country_map, on="economy", how="left")

    year_df = year_df.with_columns(
        pl.col("country_name").fill_null(pl.col("economy")).alias("country_name"),
    )

    selected_iso_codes = [
        str(code).strip().upper()
        for code in st.session_state.get(f"{PAGE_TITLE}_countries", [])
        if str(code).strip()
    ]

    top_df = year_df.sort("gap_years", descending=True).head(TOP_N_GAPS)
    top_economies = set(top_df.get_column("economy").to_list())

    extra_selected_df = year_df.filter(
        pl.col("economy").is_in(selected_iso_codes)
        & ~pl.col("economy").is_in(list(top_economies))
    )

    combined_df = pl.concat([top_df, extra_selected_df], how="vertical_relaxed").sort(
        "gap_years", descending=False
    )

    if combined_df.is_empty():
        st.info("No countries to display for this year.")
        st.divider()
        return

    plot_df = combined_df.with_columns(
        pl.when(pl.col("economy").is_in(selected_iso_codes))
        .then(pl.lit("Selected"))
        .otherwise(pl.lit("Top-15"))
        .alias("bar_group")
    ).to_pandas()

    color_map = {
        "Top-15": get_color("reference_line"),
        "Selected": get_colorway()[0],
    }
    fig = go.Figure()
    for group_name, group_color in color_map.items():
        group_rows = plot_df[plot_df["bar_group"] == group_name]
        if group_rows.empty:
            continue
        fig.add_trace(
            go.Bar(
                y=group_rows["country_name"],
                x=group_rows["gap_years"],
                orientation="h",
                marker={"color": group_color},
                name=group_name,
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    f"Female - Male life expectancy: %{{x:.2f}} years<extra></extra>"
                ),
            )
        )

    fig.add_vline(
        x=0,
        line={"color": get_color("reference_line"), "dash": "dash", "width": 1.2},
    )
    fig.update_layout(
        title=f"Female minus Male Life Expectancy ({selected_year})",
        xaxis_title="Years",
        yaxis_title="",
        barmode="group",
        bargap=0.25,
        height=max(380, 28 * combined_df.height + 120),
        margin={"l": 40, "r": 40, "t": 60, "b": 40},
    )
    fig = apply_plotly_theme(fig)

    st.plotly_chart(fig, width="stretch")
    st.caption(
        f"Top {TOP_N_GAPS} widest gaps shown by default; any countries selected "
        "in the multiselect above are appended in the highlighted colour if not "
        "already in the top set."
    )
    st.divider()


FERTILITY_INDICATOR_ID = "SP.DYN.TFRT.IN"
UNDER5_INDICATOR_ID = "SH.DYN.MORT"
POPULATION_INDICATOR_ID = "SP.POP.TOTL"


def _render_health_transition_deep_dive() -> None:
    st.divider()
    st.subheader("Demographic-Health Transition (animation)")
    st.caption(
        "X = total fertility rate (births per woman). Y = under-5 mortality "
        "(per 1,000 live births, log). Size = total population. Colour = WB "
        "region. Play the animation to see the global health transition: "
        "fertility and child mortality fall together as development progresses."
    )

    fert = _prepare_indicator_slice(
        get_world_bank_indicator(FERTILITY_INDICATOR_ID, country_code="ALL"),
        value_col="tfr",
    )
    u5 = _prepare_indicator_slice(
        get_world_bank_indicator(UNDER5_INDICATOR_ID, country_code="ALL"),
        value_col="u5m",
    )
    pop = _prepare_indicator_slice(
        get_world_bank_indicator(POPULATION_INDICATOR_ID, country_code="ALL"),
        value_col="pop",
    )
    regions_df = get_world_bank_country_regions()

    if any(df.is_empty() for df in (fert, u5, pop)) or regions_df.is_empty():
        st.info("Health transition animation is unavailable.")
        return

    regions_df = regions_df.select(
        [
            pl.col("id").cast(pl.Utf8).str.to_uppercase().alias("economy"),
            pl.col("value").cast(pl.Utf8).alias("country_name"),
            pl.col("region").cast(pl.Utf8).alias("region"),
        ]
    )

    joined = (
        fert.join(u5, on=["year", "economy"])
        .join(pop, on=["year", "economy"])
        .join(regions_df, on="economy", how="inner")
        .filter((pl.col("tfr") > 0) & (pl.col("u5m") > 0) & (pl.col("pop") > 0))
        .sort(["year", "economy"])
    )
    if joined.is_empty():
        st.info("No overlapping fertility/under-5 mortality observations.")
        return

    plot_df = joined.to_pandas().sort_values(["year", "country_name"])
    fig = px.scatter(
        plot_df,
        x="tfr",
        y="u5m",
        size="pop",
        color="region",
        animation_frame="year",
        animation_group="economy",
        hover_name="country_name",
        hover_data={
            "economy": True,
            "tfr": ":.2f",
            "u5m": ":.1f",
            "pop": ":,.0f",
            "region": True,
            "year": False,
        },
        log_y=True,
        size_max=55,
        range_x=[max(0.5, float(plot_df["tfr"].min()) - 0.3), float(plot_df["tfr"].max()) + 0.3],
        range_y=[max(1.0, float(plot_df["u5m"].min()) * 0.8), float(plot_df["u5m"].max()) * 1.2],
        labels={
            "tfr": "Fertility rate (births per woman)",
            "u5m": "Under-5 mortality (per 1,000, log)",
            "pop": "Population",
            "region": "Region",
        },
        title="Fertility vs Under-5 Mortality over time",
    )
    fig.update_traces(marker={"opacity": 0.72, "line": {"width": 0.5}})
    fig = apply_plotly_theme(fig)

    st.plotly_chart(fig, width="stretch")
    st.caption(
        "The classic demographic-and-health transition: a southwestward drift "
        "from the upper-right (high fertility, high child mortality) to the "
        "lower-left over the decades."
    )


render_page_from_config(
    page_title=PAGE_TITLE,
    section_keys=["Health and wellbeing"],
    caption=(
        "Assess health outcomes and wellbeing indicators that influence human "
        "capital, resilience, and long-run economic performance."
    ),
    before_graphs_renderer=_render_life_expectancy_gap_overview,
    after_graphs_renderer=_render_health_transition_deep_dive,
)
