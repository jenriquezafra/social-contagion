"""Matplotlib visualizations for the social contagion app."""

from __future__ import annotations

from io import BytesIO

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from networkx.drawing.layout import _fruchterman_reingold
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from src.simulation import EXPOSED, INFECTED, RECOVERED, SUSCEPTIBLE, SimulationResult


STATE_COLORS = {
    SUSCEPTIBLE: "#1f77b4",
    EXPOSED: "#f59e0b",
    INFECTED: "#d62728",
    RECOVERED: "#8c8c8c",
}

STATE_COLUMNS = (
    ("S", SUSCEPTIBLE, "S(t)"),
    ("E", EXPOSED, "E(t)"),
    ("I", INFECTED, "I(t)"),
    ("R", RECOVERED, "R(t)"),
)


def compute_layout(graph: nx.Graph, random_seed: int) -> dict[int, tuple[float, float]]:
    """Use a stable force-directed layout without requiring SciPy."""
    nodes = list(graph.nodes)
    adjacency = nx.to_numpy_array(graph, nodelist=nodes, dtype=float)
    positions = _fruchterman_reingold(adjacency, seed=random_seed)
    return {
        node: (float(positions[index, 0]), float(positions[index, 1]))
        for index, node in enumerate(nodes)
    }


def plot_network_snapshot(
    graph: nx.Graph,
    state,
    pos: dict[int, tuple[float, float]],
    title: str,
    figsize: tuple[float, float] = (5, 4),
    dark: bool = False,
) -> Figure:
    """Plot one network snapshot with S/I/R colors."""
    fig, ax = plt.subplots(figsize=figsize, dpi=130)
    _draw_network(ax, graph, state, pos, title, dark=dark)
    fig.tight_layout()
    return fig


def plot_curves(
    history: pd.DataFrame,
    marker_step: int | None = None,
    figsize: tuple[float, float] = (8, 4.3),
    dark: bool = False,
) -> Figure:
    """Plot S(t), I(t), R(t) curves."""
    fig, ax = plt.subplots(figsize=figsize, dpi=130)

    _draw_curves_on_axis(ax, history, marker_step, dark, title="S(t), E(t), I(t), R(t)")
    fig.tight_layout()
    return fig


def plot_simulation_stage(
    graph: nx.Graph,
    result: SimulationResult,
    pos: dict[int, tuple[float, float]],
    step: int,
) -> Figure:
    """Create a presentation-oriented frame for the dynamic simulation."""
    history = result.history
    row = history.iloc[step]
    total_nodes = graph.number_of_nodes()

    fig = plt.figure(figsize=(12.5, 6.8), dpi=140)
    fig.patch.set_facecolor("#060913")
    grid = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.25, 1.25, 1.0],
        height_ratios=[1.0, 0.34],
        wspace=0.18,
        hspace=0.30,
    )

    network_ax = fig.add_subplot(grid[:, :2])
    _draw_network(network_ax, graph, result.states[step], pos, f"Live contagion | t={step}", dark=True)

    curve_ax = fig.add_subplot(grid[0, 2])
    _draw_curves_on_axis(curve_ax, history, step, dark=True, title="Temporal wave")

    mix_ax = fig.add_subplot(grid[1, 2])
    _draw_state_mix(mix_ax, row, total_nodes)

    fig.suptitle(
        f"{result.model.upper()} / {result.variant}  "
        f"cascade={result.metrics['final_cascade_size']:.1%}  "
        f"peak={result.metrics['peak_infected_fraction']:.1%}",
        color="#f8fafc",
        fontsize=15,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.06, top=0.88)
    return fig


def plot_sweep(
    sweep: pd.DataFrame,
    x_column: str,
    title: str,
) -> Figure:
    """Plot final cascade size for a parameter sweep."""
    fig, ax = plt.subplots(figsize=(6, 4), dpi=130)
    ax.plot(
        sweep[x_column],
        sweep["cascade_size"],
        marker="o",
        color="#222222",
        lw=2,
    )
    ax.set_xlabel(x_column)
    ax.set_ylabel("Final cascade size")
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def plot_dashboard_export(
    graph: nx.Graph,
    result: SimulationResult,
    pos: dict[int, tuple[float, float]],
) -> Figure:
    """Create a compact PNG-ready figure with snapshots and curves."""
    peak_index = int(result.metrics["peak_index"])
    fig = plt.figure(figsize=(12, 8), dpi=150)
    grid = fig.add_gridspec(2, 3, height_ratios=[1, 0.95])

    axes = [fig.add_subplot(grid[0, idx]) for idx in range(3)]
    _draw_network(axes[0], graph, result.states[0], pos, "Initial")
    _draw_network(axes[1], graph, result.states[peak_index], pos, f"Peak (t={peak_index})")
    _draw_network(axes[2], graph, result.states[-1], pos, "Final")

    curve_ax = fig.add_subplot(grid[1, :])
    history = result.history
    _draw_curves_on_axis(curve_ax, history, marker_step=None, dark=False, title="Contagion dynamics")

    fig.suptitle(
        "Social Contagion Simulation "
        f"| Cascade={result.metrics['final_cascade_size']:.1%} "
        f"| Peak={result.metrics['peak_infected_fraction']:.1%} "
        f"| Time to peak={int(result.metrics['time_to_peak'])}",
        fontsize=13,
    )
    fig.tight_layout()
    return fig


def figure_to_png_bytes(fig: Figure) -> bytes:
    """Serialize a Matplotlib figure as PNG bytes."""
    buffer = BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    buffer.seek(0)
    return buffer.getvalue()


def _draw_network(
    ax,
    graph: nx.Graph,
    state,
    pos: dict[int, tuple[float, float]],
    title: str,
    dark: bool = False,
) -> None:
    node_colors = [STATE_COLORS[int(state[node])] for node in graph.nodes]
    node_size = max(35, min(120, 3800 / max(1, graph.number_of_nodes())))
    max_degree = max((graph.degree[node] for node in graph.nodes), default=1)
    node_sizes = [
        node_size * (0.72 + 0.75 * graph.degree[node] / max(1, max_degree))
        for node in graph.nodes
    ]
    exposed_nodes = [node for node in graph.nodes if int(state[node]) == EXPOSED]
    infected_nodes = [node for node in graph.nodes if int(state[node]) == INFECTED]

    if dark:
        ax.set_facecolor("#080b12")
        ax.figure.patch.set_facecolor("#080b12")

    nx.draw_networkx_edges(
        graph,
        pos,
        ax=ax,
        alpha=0.32 if dark else 0.18,
        width=0.9,
        edge_color="#5f6b7a" if dark else "#666666",
    )
    if exposed_nodes:
        exposed_sizes = [
            node_size * (2.8 + 1.2 * graph.degree[node] / max(1, max_degree))
            for node in exposed_nodes
        ]
        nx.draw_networkx_nodes(
            graph,
            pos,
            nodelist=exposed_nodes,
            ax=ax,
            node_color=STATE_COLORS[EXPOSED],
            node_size=exposed_sizes,
            alpha=0.16,
            linewidths=0,
        )
    if infected_nodes:
        infected_sizes = [
            node_size * (4.0 + 1.6 * graph.degree[node] / max(1, max_degree))
            for node in infected_nodes
        ]
        nx.draw_networkx_nodes(
            graph,
            pos,
            nodelist=infected_nodes,
            ax=ax,
            node_color="#ff2d2d",
            node_size=infected_sizes,
            alpha=0.18,
            linewidths=0,
        )
    nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        linewidths=0.25,
        edgecolors="white",
    )

    ax.set_title(title, color="#f8fafc" if dark else "#111111", fontweight="bold")
    ax.set_axis_off()
    legend = ax.legend(handles=_legend_handles(), loc="lower left", frameon=False, fontsize=8)
    if dark:
        for text in legend.get_texts():
            text.set_color("#e5e7eb")


def _style_axis(ax, fig: Figure, dark: bool) -> None:
    if dark:
        fig.patch.set_facecolor("#080b12")
        ax.set_facecolor("#101826")
        text_color = "#e5e7eb"
        grid_color = "#526071"
        spine_color = "#526071"
    else:
        text_color = "#111111"
        grid_color = "#999999"
        spine_color = "#cccccc"

    ax.grid(alpha=0.28, color=grid_color)
    ax.tick_params(colors=text_color)
    ax.xaxis.label.set_color(text_color)
    ax.yaxis.label.set_color(text_color)
    ax.title.set_color(text_color)
    for spine in ax.spines.values():
        spine.set_color(spine_color)


def _draw_curves_on_axis(
    ax,
    history: pd.DataFrame,
    marker_step: int | None,
    dark: bool,
    title: str,
) -> None:
    for column, state_id, label in STATE_COLUMNS:
        if column in history.columns:
            alpha = 1.0 if column != "E" or history[column].max() > 0 else 0.35
            ax.plot(history["t"], history[column], color=STATE_COLORS[state_id], label=label, lw=2, alpha=alpha)

    if marker_step is not None:
        ax.axvline(marker_step, color="#f8fafc" if dark else "#222222", ls="--", lw=1.4, alpha=0.78)

    ax.set_xlabel("Step")
    ax.set_ylabel("Nodes")
    ax.set_title(title)
    _style_axis(ax, ax.figure, dark)
    legend = ax.legend(frameon=False, ncol=2 if dark else 4)
    if dark:
        for text in legend.get_texts():
            text.set_color("#e5e7eb")


def _draw_state_mix(ax, row: pd.Series, total_nodes: int) -> None:
    ax.set_facecolor("#101826")
    ax.figure.patch.set_facecolor("#060913")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()

    left = 0.0
    for label, state_id, _curve_label in STATE_COLUMNS:
        fraction = float(row[label] / total_nodes)
        if fraction <= 0:
            continue
        ax.barh(
            [0.50],
            [fraction],
            left=left,
            height=0.28,
            color=STATE_COLORS[state_id],
            edgecolor="none",
        )
        if fraction > 0.11:
            ax.text(
                left + fraction / 2,
                0.50,
                f"{label} {fraction:.0%}",
                ha="center",
                va="center",
                color="#f8fafc",
                fontsize=8,
                fontweight="bold",
            )
        left += fraction

    ax.text(0.0, 0.88, "State mix", color="#e5e7eb", fontsize=10, fontweight="bold")
    ax.text(1.0, 0.12, f"ever infected {int(row['ever_infected'])}", color="#9ca3af", fontsize=8, ha="right")


def _legend_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], marker="o", color="w", label="S", markerfacecolor=STATE_COLORS[SUSCEPTIBLE], markersize=7),
        Line2D([0], [0], marker="o", color="w", label="E", markerfacecolor=STATE_COLORS[EXPOSED], markersize=7),
        Line2D([0], [0], marker="o", color="w", label="I", markerfacecolor=STATE_COLORS[INFECTED], markersize=7),
        Line2D([0], [0], marker="o", color="w", label="R", markerfacecolor=STATE_COLORS[RECOVERED], markersize=7),
    ]
