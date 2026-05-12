"""Streamlit app for social contagion simulations on networks."""

from __future__ import annotations

from datetime import datetime, timezone

import matplotlib.pyplot as plt
import streamlit as st

from src.networks import BARABASI_ALBERT, TOPOLOGIES, generate_network
from src.plots import (
    compute_layout,
    figure_to_png_bytes,
    plot_curves,
    plot_dashboard_export,
    plot_network_snapshot,
    plot_sweep,
)
from src.simulation import (
    INFECTED,
    SUSCEPTIBLE,
    run_beta_sweep,
    run_simulation,
    run_theta_sweep,
)
from src.twitter_higgs import (
    HIGGS_TWITTER_TOPOLOGY,
    download_higgs_data,
    load_higgs_retweet_sample,
    missing_higgs_data_files,
    observed_activity_history,
)
from src.web_stage import build_stage_html


BETA_SWEEP = [0.1, 0.2, 0.3, 0.4, 0.5]
THETA_SWEEP = [0.1, 0.25, 0.4]
APP_STATE_VERSION = 9
APP_TOPOLOGIES = (*TOPOLOGIES, HIGGS_TWITTER_TOPOLOGY)


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
        try:
            st.session_state["simulation_payload"] = _run_model(params)
        except FileNotFoundError as error:
            st.error(str(error))
            st.stop()
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
    topology = st.sidebar.selectbox("Topology", APP_TOPOLOGIES)
    influencer_layer = False
    average_degree = 6
    influencer_fraction = 0.06
    influencers_receive_from_peers = False
    min_edge_weight = 1
    reverse_retweets = True

    if topology == HIGGS_TWITTER_TOPOLOGY:
        n_nodes = st.sidebar.slider("Sample users", 50, 500, 250, 25)
        min_edge_weight = st.sidebar.slider("Minimum retweet weight", 1, 10, 1, 1)
        reverse_retweets = st.sidebar.checkbox(
            "Reverse retweets into influence flow",
            value=True,
        )
        missing_files = missing_higgs_data_files()
        if missing_files:
            st.sidebar.warning("Higgs Twitter data is not available locally.")
            if st.sidebar.button("Download Higgs data", width="stretch"):
                with st.spinner("Downloading SNAP Higgs Twitter data..."):
                    download_higgs_data()
                st.rerun()
        else:
            st.sidebar.caption(
                "Using SNAP Higgs Twitter retweets and activity from July 1-7, 2012."
            )
    else:
        n_nodes = st.sidebar.slider("Number of nodes", 20, 500, 150, 10)
        average_degree = st.sidebar.slider("Average degree", 1, 30, 6, 1)
        if topology == BARABASI_ALBERT:
            influencer_layer = st.sidebar.checkbox("Enable influencer layer", value=False)
            if influencer_layer:
                influencer_fraction = st.sidebar.slider(
                    "Influencer fraction",
                    0.01,
                    0.25,
                    0.06,
                    0.01,
                )
                influencers_receive_from_peers = st.sidebar.checkbox(
                    "Peers can influence influencers",
                    value=False,
                )
        else:
            st.sidebar.caption(
                "Synthetic mode builds reproducible NetworkX graphs from sliders."
            )

    st.sidebar.header("Initial condition")
    initial_infected = st.sidebar.slider(
        "Initial infected",
        1,
        min(80, n_nodes),
        min(5, n_nodes),
        1,
    )
    max_steps = st.sidebar.slider("Maximum steps", 5, 200, 60, 5)
    random_seed = st.sidebar.number_input("Random seed", min_value=0, value=42, step=1)

    st.sidebar.header("Model")
    model = st.sidebar.radio("Contagion model", ["simple", "threshold"], horizontal=True)
    simple_variant = "SIR"
    beta = 0.30
    sigma = 0.35
    theta = 0.25

    if model == "simple":
        simple_variant = st.sidebar.selectbox("Simple model type", ["SIR", "SIS", "SEIR"])
        beta = st.sidebar.slider("beta", 0.0, 1.0, 0.30, 0.01)
        if simple_variant == "SEIR":
            sigma = st.sidebar.slider("sigma", 0.0, 1.0, 0.35, 0.01)
    else:
        theta = st.sidebar.slider("theta", 0.0, 1.0, 0.25, 0.01)

    gamma = st.sidebar.slider("gamma", 0.0, 1.0, 0.10, 0.01)
    external_noise = st.sidebar.slider("external_noise", 0.0, 0.10, 0.00, 0.001)

    return {
        "n_nodes": n_nodes,
        "average_degree": average_degree,
        "topology": topology,
        "influencer_layer": influencer_layer,
        "influencer_fraction": influencer_fraction,
        "influencers_receive_from_peers": influencers_receive_from_peers,
        "min_edge_weight": min_edge_weight,
        "reverse_retweets": reverse_retweets,
        "initial_infected": initial_infected,
        "seed_mode": "random",
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
            <p>Model cascades across synthetic networks or a real Twitter retweet sample from the SNAP Higgs rumor dataset.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def _load_higgs_sample(
    sample_size: int,
    initial_seed_count: int,
    random_seed: int,
    min_edge_weight: int,
    reverse_retweets: bool,
):
    return load_higgs_retweet_sample(
        sample_size=sample_size,
        initial_seed_count=initial_seed_count,
        random_seed=random_seed,
        min_edge_weight=min_edge_weight,
        reverse_retweets=reverse_retweets,
    )


def _run_model(params: dict) -> dict:
    initial_infected_nodes = None
    higgs_metadata = None

    if params["topology"] == HIGGS_TWITTER_TOPOLOGY:
        higgs_sample = _load_higgs_sample(
            params["n_nodes"],
            params["initial_infected"],
            params["random_seed"],
            params["min_edge_weight"],
            params["reverse_retweets"],
        )
        graph = higgs_sample.graph
        initial_infected_nodes = higgs_sample.initial_infected
        higgs_metadata = higgs_sample.metadata
    else:
        graph = generate_network(
            n_nodes=params["n_nodes"],
            average_degree=params["average_degree"],
            topology=params["topology"],
            random_seed=params["random_seed"],
            influencer_layer=params["influencer_layer"],
            influencer_fraction=params["influencer_fraction"],
            influencers_receive_from_peers=params["influencers_receive_from_peers"],
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
        initial_infected_nodes=initial_infected_nodes,
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
        initial_infected_nodes=initial_infected_nodes,
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
        initial_infected_nodes=initial_infected_nodes,
    )

    return {
        "graph": graph,
        "pos": pos,
        "result": result,
        "beta_sweep": beta_sweep,
        "theta_sweep": theta_sweep,
        "topology": params["topology"],
        "params": params.copy(),
        "higgs_metadata": higgs_metadata,
    }


def _show_metrics(metrics: dict[str, float], graph) -> None:
    st.subheader("Control room")
    cols = st.columns(5)
    cols[0].metric("Cascade", f"{metrics['final_cascade_size']:.1%}")
    cols[1].metric("Peak I", f"{metrics['peak_infected_fraction']:.1%}")
    cols[2].metric("Peak time", f"{int(metrics['time_to_peak'])} steps")
    cols[3].metric("Active E+I", f"{metrics['final_active_fraction']:.1%}")
    cols[4].metric("Edges", f"{graph.number_of_edges():,}")


def _show_higgs_summary(metadata: dict[str, object]) -> None:
    st.caption(
        "Real-data mode uses the SNAP Higgs Twitter dataset. Retweet edges are "
        "reversed by default so an edge points from the retweeted account to "
        "the account that retweeted it."
    )
    cols = st.columns(4)
    cols[0].metric("Raw retweet edges", f"{metadata['raw_retweet_edges']:,}")
    cols[1].metric("Raw activity events", f"{metadata['raw_activity_events']:,}")
    cols[2].metric("Sample users", f"{metadata['sample_nodes']:,}")
    cols[3].metric("Sample edges", f"{metadata['sample_edges']:,}")


def _show_higgs_data_tab(graph, result, payload) -> None:
    metadata = payload["higgs_metadata"]
    params = payload["params"]
    st.subheader("Twitter Higgs data")
    st.markdown(
        f"Source: [{metadata['source_name']}]({metadata['source_url']}). "
        f"{metadata['citation']}"
    )
    st.caption(
        "The raw users are anonymized by SNAP. The app keeps those anonymized "
        "IDs as node attributes and relabels the sampled graph to contiguous "
        "integer nodes for fast simulation."
    )

    source_rows = [
        {"Field": "Topology", "Value": str(metadata["topology_label"])},
        {
            "Field": "Direction",
            "Value": (
                "retweeted user -> retweeter"
                if metadata["reverse_retweets"]
                else "retweeter -> retweeted user"
            ),
        },
        {"Field": "Minimum retweet weight", "Value": str(metadata["min_edge_weight"])},
        {
            "Field": "First sample activity",
            "Value": _format_timestamp(metadata["first_timestamp"]),
        },
        {
            "Field": "Last sample activity",
            "Value": _format_timestamp(metadata["last_timestamp"]),
        },
    ]
    st.dataframe(source_rows, width="stretch", hide_index=True)

    st.markdown("#### Observed vs simulated adoption")
    observed = observed_activity_history(graph, params["max_steps"])
    comparison = result.history[["t", "ever_infected"]].merge(
        observed,
        on="t",
        how="left",
    )
    comparison["observed_adopters"] = comparison["observed_adopters"].ffill().fillna(0)
    st.line_chart(
        comparison,
        x="t",
        y=["ever_infected", "observed_adopters"],
        height=280,
    )

    st.markdown("#### Initial real seeds")
    seed_rows = []
    for node in result.initial_infected:
        data = graph.nodes[node]
        seed_rows.append(
            {
                "Graph node": int(node),
                "Anonymized Twitter ID": int(data["twitter_user_id"]),
                "Role": data.get("role", "peer"),
                "First seen": _format_timestamp(data.get("first_seen_timestamp")),
            }
        )
    st.dataframe(seed_rows, width="stretch", hide_index=True)

    with st.expander("Raw interaction counts"):
        st.dataframe(
            [
                {"Interaction": interaction, "Events": count}
                for interaction, count in metadata["interaction_counts"].items()
            ],
            width="stretch",
            hide_index=True,
        )


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
        st.iframe(html, height=stage_height)
        return

    side_width = (100 - stage_width) / 2
    left, center, right = st.columns([side_width, stage_width, side_width])
    with center:
        st.iframe(html, height=stage_height)


def _show_analysis_tabs(graph, result, pos, payload) -> None:
    st.markdown('<div class="section-label">Analysis</div>', unsafe_allow_html=True)
    if payload.get("higgs_metadata"):
        (
            overview_tab,
            snapshots_tab,
            sweeps_tab,
            topologies_tab,
            math_tab,
            data_tab,
            export_tab,
        ) = st.tabs(
            [
                "Overview",
                "Snapshots",
                "Parameter sweeps",
                "Topologies",
                "Mathematics",
                "Twitter data",
                "Export",
            ]
        )
    else:
        overview_tab, snapshots_tab, sweeps_tab, topologies_tab, math_tab, export_tab = st.tabs(
            ["Overview", "Snapshots", "Parameter sweeps", "Topologies", "Mathematics", "Export"]
        )
        data_tab = None

    with overview_tab:
        _show_metrics(result.metrics, graph)
        if payload.get("higgs_metadata"):
            _show_higgs_summary(payload["higgs_metadata"])
        _show_curves(result)

    with snapshots_tab:
        _show_snapshots(graph, result, pos)

    with sweeps_tab:
        _show_sweeps(payload["beta_sweep"], payload["theta_sweep"])

    with topologies_tab:
        _show_topology_math(graph, payload["params"])

    with math_tab:
        _show_model_math(graph, result, payload["params"])

    if data_tab is not None:
        with data_tab:
            _show_higgs_data_tab(graph, result, payload)

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


def _show_model_math(graph, result, params: dict) -> None:
    st.subheader("Model mathematics")
    st.caption(
        "The equations below are the actual update rules used by the simulator. "
        "All node transitions are synchronous: every node reads the state at t "
        "before the app writes state t+1."
    )

    _show_current_math_context(graph, result, params)

    st.markdown("#### Common notation")
    neighborhood = r"\mathcal{N}^{-}(i)" if graph.is_directed() else r"\mathcal{N}(i)"
    degree_text = "incoming degree" if graph.is_directed() else "degree"
    st.latex(
        fr"m_i(t)=\sum_{{j \in {neighborhood}}} "
        r"\mathbf{1}\{x_j(t)=I\}"
    )
    st.latex(
        fr"k_i=|{neighborhood}|,\qquad "
        r"\epsilon=\mathrm{external\_noise}"
    )
    st.markdown(
        "`m_i(t)` is the number of infected neighbors of node `i`; `k_i` is "
        f"its {degree_text}. The random draw is evaluated independently for "
        "each node at each step."
    )
    if graph.is_directed():
        st.caption(
            "For directed influence graphs, neighbors mean incoming predecessors: "
            "an edge `u -> v` lets `u` influence `v`, not the reverse."
        )

    sir_tab, sis_tab, seir_tab, threshold_tab = st.tabs(
        ["SIR", "SIS", "SEIR", "Threshold"]
    )
    with sir_tab:
        _show_simple_model_math(
            title="Simple contagion: SIR",
            transition_text=(
                "State path: `S -> I -> R`. Recovered nodes remain recovered."
            ),
            exposed=False,
            sis=False,
        )
    with sis_tab:
        _show_simple_model_math(
            title="Simple contagion: SIS",
            transition_text=(
                "State path: `S -> I -> S`. Recovery returns a node to "
                "susceptibility."
            ),
            exposed=False,
            sis=True,
        )
    with seir_tab:
        _show_simple_model_math(
            title="Simple contagion: SEIR",
            transition_text=(
                "State path: `S -> E -> I -> R`. New contagions become "
                "exposed before activation."
            ),
            exposed=True,
            sis=False,
        )
    with threshold_tab:
        _show_threshold_model_math(params)


def _show_topology_math(graph, params: dict) -> None:
    st.subheader("Topology mathematics")
    st.caption(
        "These formulas describe how the network itself is generated before "
        "contagion dynamics start."
    )

    topology_metrics = _topology_metrics(graph)
    metric_cols = st.columns(5 if graph.is_directed() else 4)
    metric_cols[0].metric("Nodes", f"{topology_metrics['nodes']:,}")
    metric_cols[1].metric("Edges", f"{topology_metrics['edges']:,}")
    metric_cols[2].metric("Density", f"{topology_metrics['density']:.3f}")
    metric_cols[3].metric(topology_metrics["degree_label"], f"{topology_metrics['mean_degree']:.2f}")
    if graph.is_directed():
        metric_cols[4].metric("Mean out-degree", f"{topology_metrics['mean_out_degree']:.2f}")

    st.markdown(f"**Current topology:** `{params['topology']}`")
    if params["influencer_layer"]:
        st.markdown("**Influencer layer:** `enabled`")
    _show_active_topology_equations(params)

    with st.expander("Equations for all topologies"):
        er_tab, ws_tab, ba_tab, sf_tab, influencer_tab, twitter_tab = st.tabs(
            [
                "Erdos-Renyi",
                "Watts-Strogatz",
                "Barabasi-Albert",
                "Scale-Free",
                "Influencer Layer",
                "Twitter Higgs",
            ]
        )
        with er_tab:
            _show_erdos_renyi_math()
        with ws_tab:
            _show_watts_strogatz_math()
        with ba_tab:
            _show_barabasi_albert_math()
        with sf_tab:
            _show_scale_free_math()
        with influencer_tab:
            _show_barabasi_influencer_layer_math()
        with twitter_tab:
            _show_twitter_higgs_math()


def _show_active_topology_equations(params: dict) -> None:
    topology = params["topology"]
    if topology == "Erdos-Renyi":
        _show_erdos_renyi_math()
    elif topology == "Watts-Strogatz":
        _show_watts_strogatz_math()
    elif topology == BARABASI_ALBERT:
        _show_barabasi_albert_math()
        if params["influencer_layer"]:
            _show_barabasi_influencer_layer_math()
    elif topology == "Scale-Free":
        _show_scale_free_math()
    elif topology == HIGGS_TWITTER_TOPOLOGY:
        _show_twitter_higgs_math()


def _show_erdos_renyi_math() -> None:
    st.markdown("**Erdos-Renyi random graph**")
    st.latex(r"G(n,p),\qquad p=\frac{\bar{k}}{n-1}")
    st.latex(r"\mathbb{E}[M]=p\binom{n}{2},\qquad \mathbb{E}[k_i]=p(n-1)")
    st.markdown(
        "Every possible undirected edge is sampled independently. It is useful "
        "as a baseline because it has no explicit high-degree mechanism or "
        "local clustering mechanism."
    )


def _show_watts_strogatz_math() -> None:
    st.markdown("**Watts-Strogatz small-world graph**")
    st.latex(r"k_{\mathrm{ring}}=\mathrm{nearest\ even}(\bar{k}),\qquad p_r=0.1")
    st.latex(r"\mathbb{E}[k_i]\approx k_{\mathrm{ring}}")
    st.markdown(
        "The graph starts as a ring lattice, then each edge is rewired with "
        "probability `p_r`. This keeps local neighborhoods but adds shortcuts."
    )


def _show_barabasi_albert_math() -> None:
    st.markdown("**Barabasi-Albert preferential attachment**")
    st.latex(r"m=\max(1,\mathrm{round}(\bar{k}/2))")
    st.latex(r"P(\mathrm{new\ edge}\rightarrow i)=\frac{k_i}{\sum_j k_j}")
    st.latex(r"\mathbb{E}[k]\approx 2m,\qquad P(k)\sim k^{-3}")
    st.markdown(
        "New nodes attach preferentially to already connected nodes, creating "
        "high-degree nodes that can become structurally important spreaders."
    )


def _show_scale_free_math() -> None:
    st.markdown("**Scale-Free graph**")
    st.latex(r"P(k)\propto k^{-\alpha}")
    st.latex(r"M_{\mathrm{target}}\approx \frac{n\bar{k}}{2}")
    st.latex(r"P(\mathrm{extra\ source}=i)=\frac{k_i+1}{\sum_j(k_j+1)}")
    st.markdown(
        "The app starts from a NetworkX scale-free graph, converts it to a "
        "simple undirected graph, then adds preferentially weighted edges until "
        "it roughly matches the requested average degree."
    )


def _show_barabasi_influencer_layer_math() -> None:
    st.markdown("**Influencer layer on Barabasi-Albert**")
    st.latex(r"F=\max(1,\mathrm{round}(fn))")
    st.latex(r"\mathcal{I}=\mathrm{top}\ F\ \mathrm{nodes\ by\ BA\ degree}")
    st.latex(r"u\rightarrow v\quad\Rightarrow\quad u\ \mathrm{can\ influence}\ v")
    st.latex(r"i,j\in\mathcal{I}\Rightarrow i\leftrightarrow j\ \mathrm{when\ linked}")
    st.latex(r"p\rightarrow i,\ i\in\mathcal{I}\ \mathrm{only\ if\ enabled}")
    st.markdown(
        "The base topology remains Barabasi-Albert. The influencer layer marks "
        "the highest-degree BA nodes as influencers, then orients each base edge "
        "according to influence direction. Influencers can influence peers and "
        "other linked influencers; peer-to-influencer influence is controlled by "
        "the sidebar checkbox."
    )


def _show_twitter_higgs_math() -> None:
    st.markdown("**Twitter Higgs retweet graph**")
    st.latex(r"A\ \mathrm{retweets}\ B\quad\Rightarrow\quad B\rightarrow A")
    st.latex(r"w_{BA}=\mathrm{number\ of\ observed\ retweets}")
    st.latex(r"\mathcal{I}=\mathrm{top\ sampled\ nodes\ by\ weighted\ outdegree}")
    st.markdown(
        "The real-data mode uses the SNAP Higgs retweet network. The observed "
        "retweet edge is reversed for information-flow simulation, so incoming "
        "predecessors are the accounts that can expose a node to the topic. "
        "Initial infected nodes come from the earliest observed Higgs activity "
        "inside the sampled graph."
    )


def _show_current_math_context(graph, result, params: dict) -> None:
    peak_step = int(result.metrics["peak_index"])
    profile = _contagion_profile(graph, result.states[peak_step], params)
    average_degree = _average_degree(graph)
    infectious_steps = _expected_steps(params["gamma"])
    exposed_steps = _expected_steps(params["sigma"])
    vulnerable_fraction = _one_neighbor_vulnerable_fraction(graph, params["theta"])
    degree_label = "Avg in-degree" if graph.is_directed() else "Average degree"

    st.markdown("#### Current run")
    cols = st.columns(5 if graph.is_directed() else 4)
    cols[0].metric(degree_label, f"{average_degree:.2f}")
    metric_offset = 0
    if graph.is_directed():
        cols[1].metric("Influencers", f"{_influencer_count(graph):,}")
        metric_offset = 1
    cols[1 + metric_offset].metric("Expected I steps", _format_steps(infectious_steps))
    cols[2 + metric_offset].metric(
        "Peak-step infection p",
        f"{profile['mean_infection_probability']:.1%}",
    )
    cols[3 + metric_offset].metric("1-neighbor vulnerable", f"{vulnerable_fraction:.1%}")

    if params["model"] == "simple":
        rough_r = (
            float("inf")
            if params["gamma"] <= 0
            else params["beta"] * average_degree / params["gamma"]
        )
        st.info(
            "For the selected simple model, a rough early-spread heuristic is "
            f"beta * average_degree / gamma = {_format_float_or_inf(rough_r)}. "
            "It is a presentation aid, not a replacement for the network simulation."
        )
    else:
        st.info(
            "For the selected threshold model, adoption is deterministic once "
            "the infected-neighbor fraction reaches theta; external noise only "
            "matters below that threshold."
        )

    parameter_rows = [
        {
            "Parameter": "topology",
            "Current value": params["topology"],
            "Meaning": "Network structure used for exposure paths",
        },
        {
            "Parameter": "model",
            "Current value": params["model"],
            "Meaning": "Simple probabilistic or threshold contagion",
        },
        {
            "Parameter": "variant",
            "Current value": result.variant,
            "Meaning": "SIR, SIS, SEIR, or threshold",
        },
        {
            "Parameter": "beta",
            "Current value": f"{params['beta']:.2f}",
            "Meaning": "Per-neighbor transmission probability in simple contagion",
        },
        {
            "Parameter": "sigma",
            "Current value": f"{params['sigma']:.2f}",
            "Meaning": "E -> I activation probability in SEIR",
        },
        {
            "Parameter": "theta",
            "Current value": f"{params['theta']:.2f}",
            "Meaning": "Required infected-neighbor fraction in threshold contagion",
        },
        {
            "Parameter": "gamma",
            "Current value": f"{params['gamma']:.2f}",
            "Meaning": "I recovery probability per step",
        },
        {
            "Parameter": "external_noise",
            "Current value": f"{params['external_noise']:.3f}",
            "Meaning": "Outside adoption probability when contagion does not trigger",
        },
        {
            "Parameter": "random_seed",
            "Current value": str(params["random_seed"]),
            "Meaning": "Controls reproducible network and random draws",
        },
    ]
    if graph.is_directed() and params["topology"] == HIGGS_TWITTER_TOPOLOGY:
        parameter_rows.insert(
            1,
            {
                "Parameter": "data_source",
                "Current value": "SNAP Higgs Twitter",
                "Meaning": "Real retweet/activity data used for the exposure graph",
            },
        )
        parameter_rows.insert(
            2,
            {
                "Parameter": "reverse_retweets",
                "Current value": str(params["reverse_retweets"]),
                "Meaning": "Whether A retweets B is modeled as B influences A",
            },
        )
        parameter_rows.insert(
            3,
            {
                "Parameter": "min_edge_weight",
                "Current value": str(params["min_edge_weight"]),
                "Meaning": "Minimum observed retweet count kept in the sample",
            },
        )
    elif graph.is_directed():
        parameter_rows.insert(
            1,
            {
                "Parameter": "influencer_fraction",
                "Current value": f"{params['influencer_fraction']:.2f}",
                "Meaning": "Fraction of BA nodes marked as influencers",
            },
        )
        parameter_rows.insert(
            2,
            {
                "Parameter": "peers_influence_influencers",
                "Current value": str(params["influencers_receive_from_peers"]),
                "Meaning": "Whether peer nodes can send influence into influencer nodes",
            },
        )
    st.dataframe(parameter_rows, width="stretch", hide_index=True)

    with st.expander("Peak-step pressure details"):
        detail_cols = st.columns(4)
        detail_cols[0].metric("Susceptible nodes", f"{profile['susceptible_count']:,}")
        detail_cols[1].metric(
            "Mean infected neighbors",
            f"{profile['mean_infected_neighbors']:.2f}",
        )
        detail_cols[2].metric(
            "Mean infected-neighbor share",
            f"{profile['mean_neighbor_fraction']:.1%}",
        )
        detail_cols[3].metric(
            "Threshold-ready S nodes",
            f"{profile['threshold_ready_fraction']:.1%}",
        )

        st.markdown(
            "These values are computed at the peak infected step. They show how "
            "much adoption pressure the current network structure creates, not "
            "just which slider values were chosen."
        )

    if params["model"] == "simple" and params["simple_variant"] == "SEIR":
        st.caption(
            "With the current sigma, expected exposed waiting time is "
            f"{_format_steps(exposed_steps)}."
        )


def _show_simple_model_math(
    title: str,
    transition_text: str,
    exposed: bool,
    sis: bool,
) -> None:
    st.markdown(f"#### {title}")
    st.markdown(transition_text)
    st.latex(r"q_i(t)=1-(1-\beta)^{m_i(t)}")

    if exposed:
        st.latex(r"P(S_i \rightarrow E)=q_i(t)+(1-q_i(t))\epsilon")
        st.latex(r"P(E_i \rightarrow I)=\sigma")
        st.latex(r"P(I_i \rightarrow R)=\gamma")
    elif sis:
        st.latex(r"P(S_i \rightarrow I)=q_i(t)+(1-q_i(t))\epsilon")
        st.latex(r"P(I_i \rightarrow S)=\gamma")
    else:
        st.latex(r"P(S_i \rightarrow I)=q_i(t)+(1-q_i(t))\epsilon")
        st.latex(r"P(I_i \rightarrow R)=\gamma")

    st.markdown(
        "The term `1 - (1 - beta)^m` means each infected neighbor gets one "
        "independent chance to transmit. If neighbor contagion fails, the "
        "external-noise draw can still create adoption."
    )


def _show_threshold_model_math(params: dict) -> None:
    st.markdown("#### Complex contagion: threshold")
    st.markdown(
        "State path: `S -> I -> R`. Adoption needs enough infected neighbors "
        "at the same time."
    )
    st.latex(
        r"P(S_i \rightarrow I)=\begin{cases}"
        r"1, & k_i>0 \ \mathrm{and}\ \frac{m_i(t)}{k_i}\geq\theta\\"
        r"\epsilon, & \mathrm{otherwise}"
        r"\end{cases}"
    )
    st.latex(r"P(I_i \rightarrow R)=\gamma")
    st.markdown(
        "For a node with degree `k`, a single infected neighbor is sufficient "
        f"only when `1 / k >= theta`. With the current theta={params['theta']:.2f}, "
        "high-degree nodes usually need several infected neighbors at once."
    )


def _contagion_profile(graph, state, params: dict) -> dict[str, float]:
    susceptible_nodes = [node for node in graph.nodes if int(state[node]) == SUSCEPTIBLE]
    if not susceptible_nodes:
        return {
            "susceptible_count": 0,
            "mean_infected_neighbors": 0.0,
            "mean_neighbor_fraction": 0.0,
            "mean_infection_probability": 0.0,
            "threshold_ready_fraction": 0.0,
        }

    infected_neighbor_counts = []
    neighbor_fractions = []
    infection_probabilities = []
    threshold_ready = 0

    for node in susceptible_nodes:
        degree = _exposure_degree(graph, node)
        infected_neighbors = sum(
            1 for neighbor in _incoming_neighbors(graph, node)
            if int(state[neighbor]) == INFECTED
        )
        infected_neighbor_counts.append(infected_neighbors)
        neighbor_fractions.append(infected_neighbors / degree if degree else 0.0)
        is_threshold_ready = (
            degree > 0 and infected_neighbors / degree >= params["theta"]
        )
        threshold_ready += int(is_threshold_ready)

        if params["model"] == "simple":
            neighbor_probability = 1.0 - (1.0 - params["beta"]) ** infected_neighbors
            infection_probability = (
                neighbor_probability
                + (1.0 - neighbor_probability) * params["external_noise"]
            )
        else:
            infection_probability = 1.0 if is_threshold_ready else params["external_noise"]

        infection_probabilities.append(infection_probability)

    susceptible_count = len(susceptible_nodes)
    return {
        "susceptible_count": susceptible_count,
        "mean_infected_neighbors": sum(infected_neighbor_counts) / susceptible_count,
        "mean_neighbor_fraction": sum(neighbor_fractions) / susceptible_count,
        "mean_infection_probability": sum(infection_probabilities) / susceptible_count,
        "threshold_ready_fraction": threshold_ready / susceptible_count,
    }


def _average_degree(graph) -> float:
    if graph.number_of_nodes() == 0:
        return 0.0
    if graph.is_directed():
        return graph.number_of_edges() / graph.number_of_nodes()
    return 2.0 * graph.number_of_edges() / graph.number_of_nodes()


def _topology_metrics(graph) -> dict[str, float | int | str]:
    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()
    if graph.is_directed():
        max_edges = n_nodes * (n_nodes - 1)
        mean_in_degree = n_edges / n_nodes if n_nodes else 0.0
        mean_out_degree = mean_in_degree
        return {
            "nodes": n_nodes,
            "edges": n_edges,
            "density": n_edges / max_edges if max_edges else 0.0,
            "degree_label": "Mean in-degree",
            "mean_degree": mean_in_degree,
            "mean_out_degree": mean_out_degree,
        }

    max_edges = n_nodes * (n_nodes - 1) / 2
    mean_degree = 2.0 * n_edges / n_nodes if n_nodes else 0.0
    return {
        "nodes": n_nodes,
        "edges": n_edges,
        "density": n_edges / max_edges if max_edges else 0.0,
        "degree_label": "Mean degree",
        "mean_degree": mean_degree,
        "mean_out_degree": 0.0,
    }


def _expected_steps(probability: float) -> float:
    if probability <= 0:
        return float("inf")
    return 1.0 / probability


def _one_neighbor_vulnerable_fraction(graph, theta: float) -> float:
    nodes = list(graph.nodes)
    if not nodes:
        return 0.0
    vulnerable = sum(
        1 for node in nodes
        if _exposure_degree(graph, node) > 0
        and 1.0 / _exposure_degree(graph, node) >= theta
    )
    return vulnerable / len(nodes)


def _incoming_neighbors(graph, node: int):
    if graph.is_directed():
        return graph.predecessors(node)
    return graph.neighbors(node)


def _exposure_degree(graph, node: int) -> int:
    if graph.is_directed():
        return int(graph.in_degree[node])
    return int(graph.degree[node])


def _influencer_count(graph) -> int:
    if not graph.is_directed():
        return 0
    return sum(
        1 for node in graph.nodes
        if graph.nodes[node].get("role") == "influencer"
    )


def _format_timestamp(timestamp) -> str:
    if timestamp is None:
        return "not observed"
    return datetime.fromtimestamp(int(timestamp), tz=timezone.utc).strftime(
        "%Y-%m-%d %H:%M UTC"
    )


def _format_steps(value: float) -> str:
    if value == float("inf"):
        return "infinite"
    return f"{value:.1f}"


def _format_float_or_inf(value: float) -> str:
    if value == float("inf"):
        return "infinite"
    return f"{value:.2f}"


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
