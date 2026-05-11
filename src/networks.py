"""Network generation helpers for the Streamlit app."""

from __future__ import annotations

import networkx as nx
import numpy as np


DIRECTED_INFLUENCERS = "Directed Influencers"

TOPOLOGIES = (
    "Erdos-Renyi",
    "Watts-Strogatz",
    "Barabasi-Albert",
    "Scale-Free",
    DIRECTED_INFLUENCERS,
)


def generate_network(
    n_nodes: int,
    average_degree: int,
    topology: str,
    random_seed: int,
    influencer_fraction: float = 0.06,
) -> nx.Graph:
    """Create a reproducible NetworkX graph with integer node labels."""
    if n_nodes < 2:
        raise ValueError("n_nodes must be at least 2")

    degree = max(1, min(int(average_degree), n_nodes - 1))

    if topology == "Erdos-Renyi":
        probability = degree / (n_nodes - 1)
        graph = nx.erdos_renyi_graph(n_nodes, probability, seed=random_seed)
    elif topology == "Watts-Strogatz":
        k_neighbors = _nearest_valid_even_degree(degree, n_nodes)
        graph = nx.watts_strogatz_graph(
            n_nodes,
            k_neighbors,
            p=0.1,
            seed=random_seed,
        )
    elif topology == "Barabasi-Albert":
        # BA graphs have average degree close to 2m.
        edges_per_new_node = max(1, min(round(degree / 2), n_nodes - 1))
        graph = nx.barabasi_albert_graph(
            n_nodes,
            edges_per_new_node,
            seed=random_seed,
        )
    elif topology == "Scale-Free":
        graph = _scale_free_graph(n_nodes, degree, random_seed)
    elif topology == DIRECTED_INFLUENCERS:
        graph = _directed_influencer_graph(
            n_nodes=n_nodes,
            degree=degree,
            random_seed=random_seed,
            influencer_fraction=influencer_fraction,
        )
    else:
        raise ValueError(f"Unknown topology: {topology}")

    return graph


def _nearest_valid_even_degree(degree: int, n_nodes: int) -> int:
    """Watts-Strogatz requires an even k with 0 < k < n."""
    k_neighbors = max(2, degree)
    if k_neighbors % 2:
        k_neighbors += 1
    if k_neighbors >= n_nodes:
        k_neighbors = n_nodes - 1
        if k_neighbors % 2:
            k_neighbors -= 1
    return max(2, k_neighbors)


def _scale_free_graph(n_nodes: int, degree: int, random_seed: int) -> nx.Graph:
    """Create an explicit scale-free graph and roughly match the requested degree."""
    rng = np.random.default_rng(random_seed)
    multigraph = nx.scale_free_graph(n_nodes, seed=random_seed)
    graph = nx.Graph()
    graph.add_nodes_from(range(n_nodes))
    graph.add_edges_from((u, v) for u, v in multigraph.edges() if u != v)

    target_edges = int(n_nodes * degree / 2)
    attempts = 0
    max_attempts = max(1000, target_edges * 20)

    while graph.number_of_edges() < target_edges and attempts < max_attempts:
        source = _weighted_node_choice(graph, rng)
        target = int(rng.integers(0, n_nodes))
        if source != target and not graph.has_edge(source, target):
            graph.add_edge(source, target)
        attempts += 1

    return graph


def _directed_influencer_graph(
    n_nodes: int,
    degree: int,
    random_seed: int,
    influencer_fraction: float,
) -> nx.DiGraph:
    """Create a directed graph where influencer nodes broadcast but do not listen."""
    rng = np.random.default_rng(random_seed)
    influencer_count = int(round(n_nodes * influencer_fraction))
    influencer_count = max(1, min(influencer_count, n_nodes - 1))

    influencer_nodes = set(
        int(node)
        for node in rng.choice(n_nodes, size=influencer_count, replace=False).tolist()
    )
    peer_nodes = [node for node in range(n_nodes) if node not in influencer_nodes]

    graph = nx.DiGraph()
    graph.add_nodes_from(range(n_nodes))
    nx.set_node_attributes(
        graph,
        {
            node: "influencer" if node in influencer_nodes else "peer"
            for node in graph.nodes
        },
        "role",
    )
    graph.graph["influencer_count"] = influencer_count
    graph.graph["influencer_fraction"] = influencer_count / n_nodes

    if not peer_nodes:
        return graph

    source_budget = max(1, min(degree, n_nodes - 1))
    influencer_budget = max(1, round(source_budget * 0.35))
    influencer_budget = min(influencer_budget, len(influencer_nodes))
    peer_budget = max(0, source_budget - influencer_budget)

    for target in peer_nodes:
        sources: list[int] = []

        if influencer_budget:
            sources.extend(
                int(node)
                for node in rng.choice(
                    list(influencer_nodes),
                    size=influencer_budget,
                    replace=False,
                ).tolist()
            )

        peer_candidates = [node for node in peer_nodes if node != target]
        peer_source_count = min(peer_budget, len(peer_candidates))
        if peer_source_count:
            sources.extend(
                int(node)
                for node in rng.choice(
                    peer_candidates,
                    size=peer_source_count,
                    replace=False,
                ).tolist()
            )

        graph.add_edges_from((source, target) for source in sources)

    for source in influencer_nodes:
        if graph.out_degree[source] == 0:
            target = int(rng.choice(peer_nodes))
            graph.add_edge(source, target)

    return graph


def _weighted_node_choice(graph: nx.Graph, rng: np.random.Generator) -> int:
    degrees = np.array([graph.degree[node] + 1 for node in graph.nodes], dtype=float)
    probabilities = degrees / degrees.sum()
    return int(rng.choice(list(graph.nodes), p=probabilities))
