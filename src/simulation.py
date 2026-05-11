"""Simulation rules for simple and threshold social contagion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import networkx as nx
import numpy as np
import pandas as pd


SUSCEPTIBLE = 0
INFECTED = 1
RECOVERED = 2
EXPOSED = 3

STATE_LABELS = {
    SUSCEPTIBLE: "S",
    INFECTED: "I",
    RECOVERED: "R",
    EXPOSED: "E",
}

ModelName = Literal["simple", "threshold"]
SimpleVariant = Literal["SIR", "SIS", "SEIR"]
SeedMode = Literal["random", "hubs"]


@dataclass(frozen=True)
class SimulationResult:
    """Container returned by each simulation run."""

    states: list[np.ndarray]
    events: list[dict[str, list[dict]]]
    history: pd.DataFrame
    metrics: dict[str, float]
    initial_infected: list[int]
    model: str
    variant: str


def run_simulation(
    graph: nx.Graph,
    model: ModelName,
    max_steps: int,
    initial_infected_count: int,
    seed_mode: SeedMode,
    random_seed: int,
    beta: float = 0.3,
    gamma: float = 0.1,
    sigma: float = 0.3,
    theta: float = 0.25,
    simple_variant: SimpleVariant = "SIR",
    external_noise: float = 0.0,
) -> SimulationResult:
    """Run a synchronous contagion process on a graph."""
    rng = np.random.default_rng(random_seed)
    initial_infected = choose_initial_infected(
        graph=graph,
        n_initial=initial_infected_count,
        mode=seed_mode,
        rng=rng,
    )

    state = np.full(graph.number_of_nodes(), SUSCEPTIBLE, dtype=np.int8)
    state[initial_infected] = INFECTED

    states = [state.copy()]
    events = [_empty_step_events()]
    ever_infected = state == INFECTED
    history_rows = [_count_state_row(t=0, state=state, ever_infected=ever_infected)]

    for step in range(1, max_steps + 1):
        next_state = state.copy()
        step_events = _empty_step_events()

        for node in graph.nodes:
            if state[node] == SUSCEPTIBLE:
                infected, sources, trigger = _infection_trigger(
                    graph=graph,
                    node=node,
                    state=state,
                    model=model,
                    beta=beta,
                    theta=theta,
                    external_noise=external_noise,
                    rng=rng,
                )
                if infected:
                    new_state = _new_contagion_state(model, simple_variant)
                    next_state[node] = new_state
                    _record_transition(step_events, node, SUSCEPTIBLE, new_state, "contagion")
                    step_events["contagions"].append(
                        {
                            "target": int(node),
                            "sources": [int(source) for source in sources],
                            "trigger": trigger,
                            "state": STATE_LABELS[new_state],
                        }
                    )
            elif state[node] == EXPOSED:
                if model == "simple" and simple_variant == "SEIR" and rng.random() < sigma:
                    next_state[node] = INFECTED
                    _record_transition(step_events, node, EXPOSED, INFECTED, "activation")
            elif state[node] == INFECTED and rng.random() < gamma:
                if model == "simple" and simple_variant == "SIS":
                    next_state[node] = SUSCEPTIBLE
                else:
                    next_state[node] = RECOVERED
                _record_transition(step_events, node, INFECTED, int(next_state[node]), "recovery")

        state = next_state
        ever_infected = ever_infected | (state != SUSCEPTIBLE)
        states.append(state.copy())
        events.append(step_events)
        history_rows.append(_count_state_row(step, state, ever_infected))

    history = pd.DataFrame(history_rows)
    metrics = compute_metrics(history)

    return SimulationResult(
        states=states,
        events=events,
        history=history,
        metrics=metrics,
        initial_infected=initial_infected,
        model=model,
        variant=simple_variant if model == "simple" else "threshold",
    )


def choose_initial_infected(
    graph: nx.Graph,
    n_initial: int,
    mode: SeedMode,
    rng: np.random.Generator,
) -> list[int]:
    """Select initial infected nodes either randomly or by strongest influence."""
    n_nodes = graph.number_of_nodes()
    n_initial = max(1, min(int(n_initial), n_nodes))

    if mode == "random":
        return sorted(rng.choice(n_nodes, size=n_initial, replace=False).tolist())
    if mode == "hubs":
        ranked_nodes = sorted(
            graph.nodes,
            key=lambda node: (-_influence_degree(graph, node), node),
        )
        return sorted(ranked_nodes[:n_initial])
    raise ValueError(f"Unknown seed mode: {mode}")


def compute_metrics(history: pd.DataFrame) -> dict[str, float]:
    """Calculate summary metrics used in the dashboard."""
    n_nodes = float(history[["S", "E", "I", "R"]].iloc[0].sum())
    infected = history["I"].to_numpy()
    peak_index = int(infected.argmax())

    return {
        "final_cascade_size": float(history["ever_infected"].iloc[-1] / n_nodes),
        "peak_infected_fraction": float(infected[peak_index] / n_nodes),
        "time_to_peak": float(history["t"].iloc[peak_index]),
        "peak_index": float(peak_index),
        "final_infected_fraction": float(history["I"].iloc[-1] / n_nodes),
        "final_exposed_fraction": float(history["E"].iloc[-1] / n_nodes),
        "final_active_fraction": float((history["E"].iloc[-1] + history["I"].iloc[-1]) / n_nodes),
        "final_recovered_fraction": float(history["R"].iloc[-1] / n_nodes),
    }


def run_beta_sweep(
    graph: nx.Graph,
    beta_values: list[float],
    max_steps: int,
    initial_infected_count: int,
    seed_mode: SeedMode,
    random_seed: int,
    gamma: float,
    simple_variant: SimpleVariant,
    external_noise: float,
    sigma: float = 0.3,
) -> pd.DataFrame:
    """Run simple contagion for several beta values."""
    rows = []
    for beta in beta_values:
        result = run_simulation(
            graph=graph,
            model="simple",
            max_steps=max_steps,
            initial_infected_count=initial_infected_count,
            seed_mode=seed_mode,
            random_seed=random_seed,
            beta=beta,
            gamma=gamma,
            sigma=sigma,
            simple_variant=simple_variant,
            external_noise=external_noise,
        )
        rows.append(
            {
                "beta": beta,
                "cascade_size": result.metrics["final_cascade_size"],
                "peak_infected_fraction": result.metrics["peak_infected_fraction"],
            }
        )
    return pd.DataFrame(rows)


def run_theta_sweep(
    graph: nx.Graph,
    theta_values: list[float],
    max_steps: int,
    initial_infected_count: int,
    seed_mode: SeedMode,
    random_seed: int,
    gamma: float,
    external_noise: float,
) -> pd.DataFrame:
    """Run threshold contagion for several theta values."""
    rows = []
    for theta in theta_values:
        result = run_simulation(
            graph=graph,
            model="threshold",
            max_steps=max_steps,
            initial_infected_count=initial_infected_count,
            seed_mode=seed_mode,
            random_seed=random_seed,
            theta=theta,
            gamma=gamma,
            external_noise=external_noise,
        )
        rows.append(
            {
                "theta": theta,
                "cascade_size": result.metrics["final_cascade_size"],
                "peak_infected_fraction": result.metrics["peak_infected_fraction"],
            }
        )
    return pd.DataFrame(rows)


def _infection_trigger(
    graph: nx.Graph,
    node: int,
    state: np.ndarray,
    model: ModelName,
    beta: float,
    theta: float,
    external_noise: float,
    rng: np.random.Generator,
) -> tuple[bool, list[int], str]:
    infected_neighbors = [
        neighbor for neighbor in _incoming_neighbors(graph, node)
        if state[neighbor] == INFECTED
    ]

    if model == "simple":
        if infected_neighbors:
            infection_probability = 1.0 - (1.0 - beta) ** len(infected_neighbors)
            if rng.random() < infection_probability:
                return True, [_primary_cause(graph, infected_neighbors)], "neighbor"
        else:
            contagion = False
    elif model == "threshold":
        degree = _exposure_degree(graph, node)
        if degree > 0 and len(infected_neighbors) / degree >= theta:
            return True, _threshold_causes(graph, infected_neighbors), "threshold"
    else:
        raise ValueError(f"Unknown model: {model}")

    if external_noise <= 0:
        return False, [], "none"
    if rng.random() < external_noise:
        return True, [], "external"
    return False, [], "none"


def _new_contagion_state(model: ModelName, simple_variant: SimpleVariant) -> int:
    if model == "simple" and simple_variant == "SEIR":
        return EXPOSED
    return INFECTED


def _primary_cause(graph: nx.Graph, infected_neighbors: list[int]) -> int:
    """Choose a stable representative infected neighbor for simple contagion."""
    return sorted(
        infected_neighbors,
        key=lambda node: (-_influence_degree(graph, node), node),
    )[0]


def _threshold_causes(graph: nx.Graph, infected_neighbors: list[int]) -> list[int]:
    """Return the most connected infected neighbors for threshold contagion highlighting."""
    return sorted(
        infected_neighbors,
        key=lambda node: (-_influence_degree(graph, node), node),
    )[:5]


def _incoming_neighbors(graph: nx.Graph, node: int):
    if graph.is_directed():
        return graph.predecessors(node)
    return graph.neighbors(node)


def _exposure_degree(graph: nx.Graph, node: int) -> int:
    if graph.is_directed():
        return int(graph.in_degree[node])
    return int(graph.degree[node])


def _influence_degree(graph: nx.Graph, node: int) -> int:
    if graph.is_directed():
        return int(graph.out_degree[node])
    return int(graph.degree[node])


def _empty_step_events() -> dict[str, list[dict]]:
    return {
        "contagions": [],
        "transitions": [],
    }


def _record_transition(
    step_events: dict[str, list[dict]],
    node: int,
    old_state: int,
    new_state: int,
    kind: str,
) -> None:
    step_events["transitions"].append(
        {
            "node": int(node),
            "from": STATE_LABELS[old_state],
            "to": STATE_LABELS[new_state],
            "kind": kind,
        }
    )


def _count_state_row(t: int, state: np.ndarray, ever_infected: np.ndarray) -> dict[str, int]:
    return {
        "t": t,
        "S": int(np.sum(state == SUSCEPTIBLE)),
        "E": int(np.sum(state == EXPOSED)),
        "I": int(np.sum(state == INFECTED)),
        "R": int(np.sum(state == RECOVERED)),
        "ever_infected": int(np.sum(ever_infected)),
    }
