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


PAGE_TITLE = "Technology and Innovations"
RND_INDICATOR_ID = "GB.XPD.RSDV.GD.ZS"
HIGHTECH_INDICATOR_ID = "TX.VAL.TECH.CD"
GDP_INDICATOR_ID = "NY.GDP.MKTP.CD"


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


def _render_rd_vs_hightech_overview() -> None:
    st.subheader("R&D Intensity vs High-Tech Exports")
    st.caption(
        "Cross-country relationship between research spending and high-tech "
        "export value for the same year. Bubble size scales with GDP."
    )

    rnd_df = _prepare_indicator_slice(
        get_world_bank_indicator(RND_INDICATOR_ID, country_code="ALL"),
        value_col="rnd_pct_gdp",
    )
    hightech_df = _prepare_indicator_slice(
        get_world_bank_indicator(HIGHTECH_INDICATOR_ID, country_code="ALL"),
        value_col="hightech_usd",
    )
    gdp_df = _prepare_indicator_slice(
        get_world_bank_indicator(GDP_INDICATOR_ID, country_code="ALL"),
        value_col="gdp_usd",
    )

    if rnd_df.is_empty() or hightech_df.is_empty():
        st.info(
            "R&D vs high-tech scatter is unavailable because source data is empty."
        )
        st.divider()
        return

    joined_df = rnd_df.join(hightech_df, on=["year", "economy"], how="inner")
    if not gdp_df.is_empty():
        joined_df = joined_df.join(gdp_df, on=["year", "economy"], how="left")

    joined_df = joined_df.filter(pl.col("hightech_usd") > 0)
    if joined_df.is_empty():
        st.info("No overlapping R&D and high-tech-export values were found.")
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
    if "gdp_usd" in joined_df.columns:
        joined_df = joined_df.with_columns(
            pl.col("gdp_usd").fill_null(0.0).alias("gdp_usd"),
        )

    year_options = (
        joined_df.select("year").unique().sort("year").get_column("year").to_list()
    )
    if not year_options:
        st.info(
            "R&D vs high-tech scatter is unavailable because years are missing."
        )
        st.divider()
        return

    selected_year = st.select_slider(
        "Scatter year",
        options=year_options,
        value=year_options[-1],
        key="tech_rd_hightech_year",
    )

    year_df = joined_df.filter(pl.col("year") == int(selected_year))
    if year_df.is_empty():
        st.info("No R&D / high-tech observations are available for this year.")
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
    size_kwargs: dict = {}
    if "gdp_usd" in plot_df.columns and plot_df["gdp_usd"].fillna(0).gt(0).any():
        size_kwargs = {"size": "gdp_usd", "size_max": 45}

    fig = px.scatter(
        plot_df,
        x="rnd_pct_gdp",
        y="hightech_usd",
        color="country_group",
        category_orders={"country_group": ["Other", "Selected"]},
        color_discrete_map={
            "Other": get_color("reference_line"),
            "Selected": get_colorway()[0],
        },
        hover_name="country_name",
        hover_data={
            "economy": True,
            "rnd_pct_gdp": ":.2f",
            "hightech_usd": ":,.0f",
            "country_group": False,
        },
        labels={
            "rnd_pct_gdp": "R&D expenditure (% of GDP)",
            "hightech_usd": "High-tech exports (current US$, log)",
            "country_group": "",
        },
        title=f"R&D Intensity vs High-Tech Exports ({selected_year})",
        log_y=True,
        **size_kwargs,
    )

    selected_df = plot_df[plot_df["country_group"] == "Selected"]
    if not selected_df.empty:
        fig.add_trace(
            go.Scatter(
                x=selected_df["rnd_pct_gdp"],
                y=selected_df["hightech_usd"],
                mode="text",
                text=selected_df["economy"],
                textposition="top center",
                showlegend=False,
                hoverinfo="skip",
            )
        )

    fig.update_traces(
        selector={"mode": "markers"},
        marker={"opacity": 0.78, "line": {"width": 0.5}},
    )
    fig = apply_plotly_theme(fig)

    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Y-axis is logarithmic. Bubble size scales with nominal GDP (current "
        "US$). Selected countries from the multiselect above are highlighted "
        "and labelled."
    )
    st.divider()


INTERNET_INDICATOR_ID = "IT.NET.USER.ZS"
MOBILE_INDICATOR_ID = "IT.CEL.SETS.P2"
DIGITAL_MAX_LINES = 6


def _render_digital_adoption_deep_dive() -> None:
    st.divider()
    st.subheader("Digital Adoption — Internet & Mobile (time series)")
    st.caption(
        "Solid line = individuals using the Internet (% of population). Dashed "
        "line = mobile cellular subscriptions per 100 people. Up to "
        f"{DIGITAL_MAX_LINES} selected countries; with no selection the chart "
        "shows the global cross-country mean."
    )

    internet = _prepare_indicator_slice(
        get_world_bank_indicator(INTERNET_INDICATOR_ID, country_code="ALL"),
        value_col="internet_pct",
    )
    mobile = _prepare_indicator_slice(
        get_world_bank_indicator(MOBILE_INDICATOR_ID, country_code="ALL"),
        value_col="mobile_per_100",
    )

    if internet.is_empty() or mobile.is_empty():
        st.info("Digital adoption data is unavailable.")
        return

    selected_iso_codes = [
        str(c).strip().upper()
        for c in st.session_state.get(f"{PAGE_TITLE}_countries", [])
        if str(c).strip()
    ][:DIGITAL_MAX_LINES]

    country_map = get_world_bank_country_mapping()
    name_by_iso: dict[str, str] = {}
    if not country_map.is_empty() and {"id", "value"}.issubset(set(country_map.columns)):
        for row in country_map.to_dicts():
            iso = str(row.get("id", "")).strip().upper()
            nm = str(row.get("value", "")).strip()
            if iso and nm:
                name_by_iso[iso] = nm

    fig = go.Figure()
    palette = get_colorway()

    if selected_iso_codes:
        for index, iso in enumerate(selected_iso_codes):
            color = palette[index % len(palette)] if palette else None
            inet_country = internet.filter(pl.col("economy") == iso).sort("year")
            mob_country = mobile.filter(pl.col("economy") == iso).sort("year")
            display = name_by_iso.get(iso, iso)
            if not inet_country.is_empty():
                fig.add_trace(
                    go.Scatter(
                        x=inet_country["year"].to_list(),
                        y=inet_country["internet_pct"].to_list(),
                        mode="lines",
                        name=f"{display} - Internet",
                        line={"color": color, "width": 2},
                        hovertemplate=f"<b>{display}</b><br>%{{x}}<br>Internet users: %{{y:.1f}}%<extra></extra>",
                    )
                )
            if not mob_country.is_empty():
                fig.add_trace(
                    go.Scatter(
                        x=mob_country["year"].to_list(),
                        y=mob_country["mobile_per_100"].to_list(),
                        mode="lines",
                        name=f"{display} - Mobile",
                        line={"color": color, "dash": "dash", "width": 2},
                        hovertemplate=f"<b>{display}</b><br>%{{x}}<br>Mobile/100: %{{y:.1f}}<extra></extra>",
                    )
                )
    else:
        inet_mean = internet.group_by("year").agg(pl.col("internet_pct").mean().alias("internet_pct")).sort("year")
        mob_mean = mobile.group_by("year").agg(pl.col("mobile_per_100").mean().alias("mobile_per_100")).sort("year")
        fig.add_trace(
            go.Scatter(
                x=inet_mean["year"].to_list(),
                y=inet_mean["internet_pct"].to_list(),
                mode="lines",
                name="Global mean - Internet",
                line={"color": palette[0] if palette else None, "width": 2.5},
            )
        )
        fig.add_trace(
            go.Scatter(
                x=mob_mean["year"].to_list(),
                y=mob_mean["mobile_per_100"].to_list(),
                mode="lines",
                name="Global mean - Mobile",
                line={"color": palette[1] if palette else None, "dash": "dash", "width": 2.5},
            )
        )

    fig.update_layout(
        title="Digital adoption trajectory",
        xaxis_title="Year",
        yaxis_title="% / per 100 people",
        margin={"l": 40, "r": 20, "t": 60, "b": 40},
        legend={"orientation": "h", "y": -0.18},
    )
    fig = apply_plotly_theme(fig)

    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Mobile subscriptions often overshoot 100 per 100 people (multi-SIM). "
        "Where the dashed line stays above the solid line, mobile uptake has "
        "outpaced fixed-internet adoption — typical of leapfrogging countries."
    )


render_page_from_config(
    page_title=PAGE_TITLE,
    section_keys=["Technology and Innovations"],
    caption=(
        "Follow innovation capacity, digital adoption, and R&D-related metrics "
        "that shape long-term productivity growth."
    ),
    before_graphs_renderer=_render_rd_vs_hightech_overview,
    after_graphs_renderer=_render_digital_adoption_deep_dive,
)
