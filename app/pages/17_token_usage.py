"""Historical token usage dashboard page.

Surfaces totals, per-model breakdown, per-source split, and a daily trend
of LLM token consumption recorded in the ``token_usage`` Postgres table by
:mod:`core.token_usage_store`. Read-only view; the table itself is populated
from the chat and plot-interpretation flows.
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from core.app_logging import log_page_render
from core.plotting import apply_plotly_theme
from core.theming import get_colorway
from core.token_usage_store import (
    get_aggregates_by_day,
    get_aggregates_by_model,
    get_aggregates_by_source,
    get_totals,
)


def _fmt(value: int) -> str:
    """Pretty-print an integer with thousands separators."""
    return f"{int(value):,}"


def render_page() -> None:
    """Page entry-point: totals row, daily trend, per-model and per-source charts."""
    log_page_render("Token Usage")
    st.title("Token Usage")
    st.caption(
        "Cumulative LLM token consumption recorded by the dashboard. "
        "Includes chat completions and plot interpretation calls."
    )

    totals = get_totals()
    metric_cols = st.columns(3)
    metric_cols[0].metric("Prompt tokens", _fmt(totals["prompt_tokens"]))
    metric_cols[1].metric("Completion tokens", _fmt(totals["completion_tokens"]))
    metric_cols[2].metric("Total tokens", _fmt(totals["total_tokens"]))

    if totals["total_tokens"] == 0:
        st.info(
            "No token usage recorded yet. Run a question on the AI Agent page "
            "or interpret a plot to populate this view."
        )
        return

    palette = get_colorway()

    st.subheader("Daily trend")
    daily_df = get_aggregates_by_day()
    if daily_df.is_empty():
        st.info("No daily aggregates available.")
    else:
        trend_fig = go.Figure()
        trend_fig.add_trace(
            go.Scatter(
                x=daily_df["day"].to_list(),
                y=daily_df["prompt_tokens"].to_list(),
                mode="lines+markers",
                name="Prompt",
                line={"color": palette[0] if palette else None},
            )
        )
        trend_fig.add_trace(
            go.Scatter(
                x=daily_df["day"].to_list(),
                y=daily_df["completion_tokens"].to_list(),
                mode="lines+markers",
                name="Completion",
                line={"color": palette[1] if len(palette) > 1 else None},
            )
        )
        trend_fig.add_trace(
            go.Scatter(
                x=daily_df["day"].to_list(),
                y=daily_df["total_tokens"].to_list(),
                mode="lines+markers",
                name="Total",
                line={"color": palette[2] if len(palette) > 2 else None, "dash": "dot"},
            )
        )
        trend_fig.update_layout(
            xaxis_title="Day",
            yaxis_title="Tokens",
            legend={"orientation": "h"},
        )
        st.plotly_chart(apply_plotly_theme(trend_fig), width="stretch")

    left, right = st.columns(2)

    with left:
        st.subheader("By model")
        model_df = get_aggregates_by_model()
        if model_df.is_empty():
            st.info("No per-model rows.")
        else:
            bar_fig = go.Figure()
            bar_fig.add_trace(
                go.Bar(
                    x=model_df["model"].to_list(),
                    y=model_df["total_tokens"].to_list(),
                    marker={"color": palette[0] if palette else None},
                    name="Total tokens",
                )
            )
            bar_fig.update_layout(
                xaxis_title="Model",
                yaxis_title="Total tokens",
                showlegend=False,
            )
            st.plotly_chart(apply_plotly_theme(bar_fig), width="stretch")
            st.dataframe(model_df, hide_index=True, width="stretch")

    with right:
        st.subheader("By source")
        source_df = get_aggregates_by_source()
        if source_df.is_empty():
            st.info("No per-source rows.")
        else:
            pie_fig = go.Figure()
            pie_fig.add_trace(
                go.Pie(
                    labels=source_df["source"].to_list(),
                    values=source_df["total_tokens"].to_list(),
                    hole=0.4,
                    marker={"colors": palette},
                )
            )
            pie_fig.update_layout(showlegend=True)
            st.plotly_chart(apply_plotly_theme(pie_fig), width="stretch")
            st.dataframe(source_df, hide_index=True, width="stretch")


render_page()
