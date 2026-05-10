"""Streamlit app for social contagion simulations on networks."""

from __future__ import annotations

import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as components

from src.networks import TOPOLOGIES, generate_network
from src.plots import (
    compute_layout,
    figure_to_png_bytes,
    plot_curves,
    plot_dashboard_export,
    plot_network_snapshot,
    plot_sweep,
)
from src.simulation import run_beta_sweep, run_simulation, run_theta_sweep
from src.web_stage import build_stage_html


BETA_SWEEP = [0.1, 0.2, 0.3, 0.4, 0.5]
THETA_SWEEP = [0.1, 0.25, 0.4]
APP_STATE_VERSION = 3


def main() -> None:
    st.set_page_config(
        page_title="Social Contagion Simulator",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    params = _sidebar_controls()
    _inject_style(params["visual_theme"])
    _show_header()

    run_clicked = st.sidebar.button("Run Simulation", type="primary", width="stretch")

    if (
        run_clicked
        or "simulation_payload" not in st.session_state
        or st.session_state.get("app_state_version") != APP_STATE_VERSION
    ):
        st.session_state["simulation_payload"] = _run_model(params)
        st.session_state["app_state_version"] = APP_STATE_VERSION

    payload = st.session_state["simulation_payload"]
    result = payload["result"]
    graph = payload["graph"]
    pos = payload["pos"]

    _show_dynamic_simulation(
        graph,
        result,
        pos,
        payload["topology"],
        params["visual_theme"],
        params["stage_height"],
        params["stage_width"],
    )
    _show_analysis_tabs(graph, result, pos, payload)


def _sidebar_controls() -> dict:
    st.sidebar.header("Appearance")
    visual_theme = st.sidebar.selectbox("Visual theme", ["Auto", "Light", "Dark"])
    stage_preset = st.sidebar.selectbox("Stage format", ["Wide", "Compact", "Tall"])
    default_height = {"Wide": 585, "Compact": 500, "Tall": 720}[stage_preset]
    stage_height = st.sidebar.slider("Stage height", 460, 820, default_height, 20)
    stage_width = st.sidebar.slider("Stage width", 70, 100, 100, 5)

    st.sidebar.header("Network")
    n_nodes = st.sidebar.slider("Number of nodes", 20, 500, 150, 10)
    average_degree = st.sidebar.slider("Average degree", 1, 30, 6, 1)
    topology = st.sidebar.selectbox("Topology", TOPOLOGIES)

    st.sidebar.header("Initial condition")
    initial_infected = st.sidebar.slider(
        "Initial infected",
        1,
        min(80, n_nodes),
        min(5, n_nodes),
        1,
    )
    seed_mode = st.sidebar.radio("Seed mode", ["random", "hubs"], horizontal=True)
    max_steps = st.sidebar.slider("Maximum steps", 5, 200, 60, 5)
    random_seed = st.sidebar.number_input("Random seed", min_value=0, value=42, step=1)

    st.sidebar.header("Model")
    model = st.sidebar.radio("Contagion model", ["simple", "threshold"], horizontal=True)
    simple_variant = st.sidebar.selectbox("Simple model type", ["SIR", "SIS", "SEIR"])
    beta = st.sidebar.slider("beta", 0.0, 1.0, 0.30, 0.01)
    sigma = st.sidebar.slider("sigma", 0.0, 1.0, 0.35, 0.01)
    theta = st.sidebar.slider("theta", 0.0, 1.0, 0.25, 0.01)
    gamma = st.sidebar.slider("gamma", 0.0, 1.0, 0.10, 0.01)
    external_noise = st.sidebar.slider("external_noise", 0.0, 0.10, 0.00, 0.001)

    return {
        "n_nodes": n_nodes,
        "average_degree": average_degree,
        "topology": topology,
        "initial_infected": initial_infected,
        "seed_mode": seed_mode,
        "max_steps": max_steps,
        "random_seed": int(random_seed),
        "model": model,
        "simple_variant": simple_variant,
        "beta": beta,
        "sigma": sigma,
        "theta": theta,
        "gamma": gamma,
        "external_noise": external_noise,
        "visual_theme": visual_theme,
        "stage_preset": stage_preset,
        "stage_height": stage_height,
        "stage_width": stage_width,
    }


def _inject_style(visual_theme: str) -> None:
    st.markdown(
        """
        <style>
        :root {
            --page-top: #ffffff;
            --page-bg: #f5f5f7;
            --page-bottom: #ffffff;
            --ink: #1d1d1f;
            --muted: #6e6e73;
            --line: rgba(0, 0, 0, 0.08);
            --card: rgba(255, 255, 255, 0.82);
            --sidebar-bg: rgba(245, 245, 247, 0.86);
            --tab-bg: rgba(255, 255, 255, 0.70);
            --tab-selected-bg: #1d1d1f;
            --tab-selected-text: #ffffff;
            --input-bg: #ffffff;
            --shadow: 0 16px 42px rgba(0, 0, 0, 0.05);
        }
        @media (prefers-color-scheme: dark) {
            :root {
                --page-top: #050506;
                --page-bg: #111113;
                --page-bottom: #050506;
                --ink: #f5f5f7;
                --muted: #a1a1a6;
                --line: rgba(255, 255, 255, 0.13);
                --card: rgba(28, 28, 30, 0.78);
                --sidebar-bg: rgba(18, 18, 20, 0.90);
                --tab-bg: rgba(28, 28, 30, 0.82);
                --tab-selected-bg: #f5f5f7;
                --tab-selected-text: #1d1d1f;
                --input-bg: rgba(255, 255, 255, 0.08);
                --shadow: 0 18px 48px rgba(0, 0, 0, 0.24);
            }
        }
        html, body, [class*="css"] {
            font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Inter", "Segoe UI", sans-serif;
        }
        .stApp {
            background:
                linear-gradient(180deg, var(--page-top) 0%, var(--page-bg) 24%, var(--page-bottom) 100%);
            color: var(--ink);
        }
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"],
        [data-testid="stSidebarContent"] {
            background: transparent;
        }
        .block-container {
            max-width: min(1600px, 96vw);
            padding-top: 2.1rem;
            padding-bottom: 4rem;
        }
        h1 {
            font-weight: 850;
            letter-spacing: -0.04em;
        }
        h3 {
            font-weight: 760;
            letter-spacing: -0.025em;
        }
        [data-testid="stSidebar"] {
            background: var(--sidebar-bg);
            border-right: 1px solid var(--line);
            backdrop-filter: blur(22px);
        }
        [data-testid="stHeader"] {
            background: var(--page-top);
        }
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] span {
            color: var(--ink);
        }
        [data-testid="stSidebar"] [data-baseweb="select"] > div,
        [data-testid="stSidebar"] input {
            background: var(--input-bg);
            border-color: var(--line);
            color: var(--ink);
        }
        [data-testid="stSidebar"] [data-baseweb="select"] * {
            color: var(--ink);
        }
        [data-testid="stMetric"] {
            background: var(--card);
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 0.7rem 0.8rem;
            box-shadow: var(--shadow);
        }
        .hero-panel {
            text-align: center;
            margin: 0 auto 1.4rem;
            padding: 0.4rem 1rem 0.2rem;
        }
        .hero-panel h1 {
            color: var(--ink);
            margin: 0;
            font-size: clamp(2.8rem, 7vw, 5.8rem);
            line-height: 0.92;
            letter-spacing: -0.065em;
        }
        .hero-panel p {
            color: var(--muted);
            margin: 0.7rem auto 0;
            max-width: 760px;
            font-size: 1.18rem;
            line-height: 1.45;
        }
        .hero-kicker {
            color: #0066cc;
            font-size: 0.8rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.55rem;
        }
        .section-label {
            color: var(--muted);
            font-size: 0.84rem;
            font-weight: 760;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin: 2.2rem 0 0.6rem;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background: var(--tab-bg);
            border: 1px solid var(--line);
            border-radius: 999px;
            padding: 6px;
            width: fit-content;
            box-shadow: var(--shadow);
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 999px;
            padding: 8px 18px;
            color: var(--ink);
        }
        .stTabs [aria-selected="true"] {
            background: var(--tab-selected-bg);
            color: var(--tab-selected-text);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    theme_override = _visual_theme_override(visual_theme)
    if theme_override:
        st.markdown(f"<style>{theme_override}</style>", unsafe_allow_html=True)


def _visual_theme_override(visual_theme: str) -> str:
    if visual_theme == "Light":
        return """
        :root, .stApp {
            --page-top: #ffffff;
            --page-bg: #f5f5f7;
            --page-bottom: #ffffff;
            --ink: #1d1d1f;
            --muted: #6e6e73;
            --line: rgba(0, 0, 0, 0.08);
            --card: rgba(255, 255, 255, 0.82);
            --sidebar-bg: rgba(245, 245, 247, 0.86);
            --tab-bg: rgba(255, 255, 255, 0.70);
            --tab-selected-bg: #1d1d1f;
            --tab-selected-text: #ffffff;
            --input-bg: #ffffff;
            --shadow: 0 16px 42px rgba(0, 0, 0, 0.05);
        }
        """
    if visual_theme == "Dark":
        return """
        :root, .stApp {
            --page-top: #050506;
            --page-bg: #111113;
            --page-bottom: #050506;
            --ink: #f5f5f7;
            --muted: #a1a1a6;
            --line: rgba(255, 255, 255, 0.13);
            --card: rgba(28, 28, 30, 0.78);
            --sidebar-bg: rgba(18, 18, 20, 0.90);
            --tab-bg: rgba(28, 28, 30, 0.82);
            --tab-selected-bg: #f5f5f7;
            --tab-selected-text: #1d1d1f;
            --input-bg: rgba(255, 255, 255, 0.08);
            --shadow: 0 18px 48px rgba(0, 0, 0, 0.24);
        }
        """
    return ""


def _show_header() -> None:
    st.markdown(
        """
        <div class="hero-panel">
            <div class="hero-kicker">Complex Systems Studio</div>
            <h1>Social Contagion Simulator</h1>
            <p>Model cascades across random, small-world and scale-free networks with a polished interactive simulation built for presentation.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _run_model(params: dict) -> dict:
    graph = generate_network(
        n_nodes=params["n_nodes"],
        average_degree=params["average_degree"],
        topology=params["topology"],
        random_seed=params["random_seed"],
    )
    pos = compute_layout(graph, params["random_seed"])

    result = run_simulation(
        graph=graph,
        model=params["model"],
        max_steps=params["max_steps"],
        initial_infected_count=params["initial_infected"],
        seed_mode=params["seed_mode"],
        random_seed=params["random_seed"],
        beta=params["beta"],
        gamma=params["gamma"],
        sigma=params["sigma"],
        theta=params["theta"],
        simple_variant=params["simple_variant"],
        external_noise=params["external_noise"],
    )

    beta_sweep = run_beta_sweep(
        graph=graph,
        beta_values=BETA_SWEEP,
        max_steps=params["max_steps"],
        initial_infected_count=params["initial_infected"],
        seed_mode=params["seed_mode"],
        random_seed=params["random_seed"],
        gamma=params["gamma"],
        sigma=params["sigma"],
        simple_variant=params["simple_variant"],
        external_noise=params["external_noise"],
    )
    theta_sweep = run_theta_sweep(
        graph=graph,
        theta_values=THETA_SWEEP,
        max_steps=params["max_steps"],
        initial_infected_count=params["initial_infected"],
        seed_mode=params["seed_mode"],
        random_seed=params["random_seed"],
        gamma=params["gamma"],
        external_noise=params["external_noise"],
    )

    return {
        "graph": graph,
        "pos": pos,
        "result": result,
        "beta_sweep": beta_sweep,
        "theta_sweep": theta_sweep,
        "topology": params["topology"],
    }


def _show_metrics(metrics: dict[str, float], graph) -> None:
    st.subheader("Control room")
    cols = st.columns(5)
    cols[0].metric("Cascade", f"{metrics['final_cascade_size']:.1%}")
    cols[1].metric("Peak I", f"{metrics['peak_infected_fraction']:.1%}")
    cols[2].metric("Peak time", f"{int(metrics['time_to_peak'])} steps")
    cols[3].metric("Active E+I", f"{metrics['final_active_fraction']:.1%}")
    cols[4].metric("Edges", f"{graph.number_of_edges():,}")


def _show_snapshots(graph, result, pos) -> None:
    st.subheader("Network snapshots")
    peak_index = int(result.metrics["peak_index"])
    snapshots = [
        ("Initial network", result.states[0]),
        (f"Network at peak, t={peak_index}", result.states[peak_index]),
        ("Final network", result.states[-1]),
    ]

    cols = st.columns(3)
    for col, (title, state) in zip(cols, snapshots):
        fig = plot_network_snapshot(graph, state, pos, title)
        col.pyplot(fig, clear_figure=True)
        plt.close(fig)


def _show_dynamic_simulation(
    graph,
    result,
    pos,
    topology: str,
    visual_theme: str,
    stage_height: int,
    stage_width: int,
) -> None:
    html = build_stage_html(graph, result, pos, topology, visual_theme)
    if stage_width >= 100:
        components.html(html, height=stage_height, scrolling=False)
        return

    side_width = (100 - stage_width) / 2
    left, center, right = st.columns([side_width, stage_width, side_width])
    with center:
        components.html(html, height=stage_height, scrolling=False)


def _show_analysis_tabs(graph, result, pos, payload) -> None:
    st.markdown('<div class="section-label">Analysis</div>', unsafe_allow_html=True)
    overview_tab, snapshots_tab, sweeps_tab, export_tab = st.tabs(
        ["Overview", "Snapshots", "Parameter sweeps", "Export"]
    )

    with overview_tab:
        _show_metrics(result.metrics, graph)
        _show_curves(result)

    with snapshots_tab:
        _show_snapshots(graph, result, pos)

    with sweeps_tab:
        _show_sweeps(payload["beta_sweep"], payload["theta_sweep"])

    with export_tab:
        _show_export_button(graph, result, pos)


def _show_curves(result) -> None:
    st.subheader("Time series")
    fig = plot_curves(result.history)
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


def _show_sweeps(beta_sweep, theta_sweep) -> None:
    st.subheader("Automatic comparison")
    cols = st.columns(2)

    fig_beta = plot_sweep(beta_sweep, "beta", "Cascade size vs beta")
    cols[0].pyplot(fig_beta, clear_figure=True)
    plt.close(fig_beta)

    fig_theta = plot_sweep(theta_sweep, "theta", "Cascade size vs theta")
    cols[1].pyplot(fig_theta, clear_figure=True)
    plt.close(fig_theta)

    with st.expander("Sweep data"):
        left, right = st.columns(2)
        left.dataframe(beta_sweep, width="stretch", hide_index=True)
        right.dataframe(theta_sweep, width="stretch", hide_index=True)


def _show_export_button(graph, result, pos) -> None:
    export_fig = plot_dashboard_export(graph, result, pos)
    png_bytes = figure_to_png_bytes(export_fig)
    plt.close(export_fig)

    st.download_button(
        label="Export PNG",
        data=png_bytes,
        file_name="social_contagion_simulation.png",
        mime="image/png",
        width="stretch",
    )


if __name__ == "__main__":
    main()
