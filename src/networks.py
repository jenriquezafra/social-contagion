"""Network generation helpers for the Streamlit app."""

from __future__ import annotations

import networkx as nx
import numpy as np


BARABASI_ALBERT = "Barabasi-Albert"

TOPOLOGIES = (
    "Erdos-Renyi",
    "Watts-Strogatz",
    BARABASI_ALBERT,
    "Scale-Free",
)


def generate_network(
    n_nodes: int,
    average_degree: int,
    topology: str,
    random_seed: int,
    influencer_layer: bool = False,
    influencer_fraction: float = 0.06,
    influencers_receive_from_peers: bool = False,
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
    elif topology == BARABASI_ALBERT:
        # BA graphs have average degree close to 2m.
        edges_per_new_node = max(1, min(round(degree / 2), n_nodes - 1))
        graph = nx.barabasi_albert_graph(
            n_nodes,
            edges_per_new_node,
            seed=random_seed,
        )
        if influencer_layer:
            graph = _barabasi_albert_influence_layer(
                graph=graph,
                influencer_fraction=influencer_fraction,
                influencers_receive_from_peers=influencers_receive_from_peers,
            )
    elif topology == "Scale-Free":
        graph = _scale_free_graph(n_nodes, degree, random_seed)
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


def _barabasi_albert_influence_layer(
    graph: nx.Graph,
    influencer_fraction: float,
    influencers_receive_from_peers: bool,
) -> nx.DiGraph:
    """Orient a BA graph into a directed influence graph with influencer nodes."""
    n_nodes = graph.number_of_nodes()
    influencer_count = int(round(n_nodes * influencer_fraction))
    influencer_count = max(1, min(influencer_count, n_nodes - 1))
    influencer_nodes = set(
        node for node, _degree in sorted(
            graph.degree,
            key=lambda item: (-item[1], item[0]),
        )[:influencer_count]
    )

    influence_graph = nx.DiGraph()
    influence_graph.add_nodes_from(graph.nodes)
    nx.set_node_attributes(
        influence_graph,
        {
            node: "influencer" if node in influencer_nodes else "peer"
            for node in graph.nodes
        },
        "role",
    )
    influence_graph.graph["influencer_layer"] = True
    influence_graph.graph["base_topology"] = BARABASI_ALBERT
    influence_graph.graph["influencer_count"] = influencer_count
    influence_graph.graph["influencer_fraction"] = influencer_count / n_nodes
    influence_graph.graph["influencers_receive_from_peers"] = influencers_receive_from_peers

    for source, target in graph.edges:
        source_is_influencer = source in influencer_nodes
        target_is_influencer = target in influencer_nodes

        if source_is_influencer and target_is_influencer:
            influence_graph.add_edge(source, target)
            influence_graph.add_edge(target, source)
        elif source_is_influencer:
            influence_graph.add_edge(source, target)
            if influencers_receive_from_peers:
                influence_graph.add_edge(target, source)
        elif target_is_influencer:
            influence_graph.add_edge(target, source)
            if influencers_receive_from_peers:
                influence_graph.add_edge(source, target)
        else:
            influence_graph.add_edge(source, target)
            influence_graph.add_edge(target, source)

    return influence_graph


def _weighted_node_choice(graph: nx.Graph, rng: np.random.Generator) -> int:
    degrees = np.array([graph.degree[node] + 1 for node in graph.nodes], dtype=float)
    probabilities = degrees / degrees.sum()
    return int(rng.choice(list(graph.nodes), p=probabilities))
