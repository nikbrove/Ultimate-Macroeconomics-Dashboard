"""System monitoring page: per-service health + per-container resource use.

Two sources are aggregated:

* :func:`core.monitoring.probe_all_services` for HTTP/TCP health probes;
* :func:`core.monitoring.get_container_stats` for CPU / memory / network
  pulled from the Docker Engine API over the mounted UNIX socket.

The page is read-only and re-runs on every Streamlit refresh — the user
triggers a refresh manually with the button.
"""

from __future__ import annotations

import plotly.graph_objects as go
import polars as pl
import streamlit as st

from core.app_logging import log_page_render
from core.monitoring import (
    docker_socket_available,
    get_container_stats,
    probe_all_services,
)
from core.plotting import apply_plotly_theme
from core.theming import get_color, get_colorway


def _status_badge(status: str) -> str:
    """Convert a probe status string into a coloured emoji label."""
    if status == "up":
        return "🟢 up"
    if status == "skipped":
        return "⚪ skipped"
    return "🔴 down"


def render_page() -> None:
    """Page entry-point: refresh button, health table, summary metrics, container stats."""
    log_page_render("Monitoring")
    st.title("System Monitoring")
    st.caption(
        "Live health probes and resource usage for every Compose container. "
        "Refreshes on demand — click the button to re-poll."
    )

    if st.button("Refresh now", type="primary", width="content"):
        st.rerun()

    st.subheader("Service health")
    health_results = probe_all_services()
    if not health_results:
        st.info("No services probed.")
    else:
        rows = [
            {
                "Service": r.service,
                "Status": _status_badge(r.status),
                "Latency (ms)": (round(r.latency_ms, 1) if r.latency_ms is not None else None),
                "Endpoint": r.url,
                "Detail": r.detail,
            }
            for r in health_results
        ]
        st.dataframe(rows, hide_index=True, width="stretch")

        up = sum(1 for r in health_results if r.status == "up")
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Services up", f"{up} / {len(health_results)}")
        latencies = [r.latency_ms for r in health_results if r.latency_ms is not None]
        if latencies:
            col_b.metric("Avg latency (ms)", round(sum(latencies) / len(latencies), 1))
            col_c.metric("Max latency (ms)", round(max(latencies), 1))

    st.divider()
    st.subheader("Container resource use")
    if not docker_socket_available():
        st.warning(
            "Docker socket not mounted into this container at "
            "`/var/run/docker.sock`. Add the bind-mount in `docker-compose.yaml` "
            "to enable container-level stats."
        )
        return

    stats_rows = get_container_stats()
    if not stats_rows:
        st.info("Docker reported no containers.")
        return

    df = pl.DataFrame(
        {
            "Container": [s.name for s in stats_rows],
            "Status": [s.status for s in stats_rows],
            "CPU %": [round(s.cpu_percent, 2) for s in stats_rows],
            "Memory (MB)": [round(s.memory_used_mb, 1) for s in stats_rows],
            "Memory limit (MB)": [round(s.memory_limit_mb, 1) for s in stats_rows],
            "Memory %": [round(s.memory_percent, 2) for s in stats_rows],
            "Net RX (MB)": [round(s.rx_mb, 2) for s in stats_rows],
            "Net TX (MB)": [round(s.tx_mb, 2) for s in stats_rows],
        }
    )
    st.dataframe(df, hide_index=True, width="stretch")

    palette = get_colorway()
    left, right = st.columns(2)

    with left:
        cpu_fig = go.Figure()
        cpu_fig.add_trace(
            go.Bar(
                x=[s.name for s in stats_rows],
                y=[s.cpu_percent for s in stats_rows],
                marker={"color": palette[0] if palette else None},
                name="CPU %",
            )
        )
        cpu_fig.update_layout(
            title="CPU % per container",
            xaxis_title="Container",
            yaxis_title="CPU %",
            showlegend=False,
        )
        st.plotly_chart(apply_plotly_theme(cpu_fig), width="stretch")

    with right:
        mem_fig = go.Figure()
        mem_fig.add_trace(
            go.Bar(
                x=[s.name for s in stats_rows],
                y=[s.memory_used_mb for s in stats_rows],
                marker={"color": palette[1] if len(palette) > 1 else None},
                name="Used MB",
            )
        )
        try:
            ref_color = get_color("reference_line")
        except KeyError:
            ref_color = None
        mem_fig.add_trace(
            go.Bar(
                x=[s.name for s in stats_rows],
                y=[max(s.memory_limit_mb - s.memory_used_mb, 0) for s in stats_rows],
                marker={"color": ref_color},
                name="Free (within limit)",
                opacity=0.4,
            )
        )
        mem_fig.update_layout(
            barmode="stack",
            title="Memory per container (MB)",
            xaxis_title="Container",
            yaxis_title="MB",
        )
        st.plotly_chart(apply_plotly_theme(mem_fig), width="stretch")


render_page()
