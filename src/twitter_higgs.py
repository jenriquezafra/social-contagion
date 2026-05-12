"""Load a presentation-sized sample from the SNAP Higgs Twitter dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import urllib.request

import networkx as nx
import numpy as np
import pandas as pd


HIGGS_TWITTER_TOPOLOGY = "Twitter Higgs retweets"
SNAP_DATASET_URL = "https://snap.stanford.edu/data/higgs-twitter.html"
HIGGS_CITATION = (
    "M. De Domenico, A. Lima, P. Mougel and M. Musolesi. "
    "The Anatomy of a Scientific Rumor. Scientific Reports 3, 2980 (2013)."
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "higgs-twitter" / "raw"
RETWEET_PATH = RAW_DIR / "higgs-retweet_network.edgelist.gz"
ACTIVITY_PATH = RAW_DIR / "higgs-activity_time.txt.gz"

RAW_DATA_URLS = {
    RETWEET_PATH: "https://snap.stanford.edu/data/higgs-retweet_network.edgelist.gz",
    ACTIVITY_PATH: "https://snap.stanford.edu/data/higgs-activity_time.txt.gz",
}


@dataclass(frozen=True)
class HiggsRetweetSample:
    """Sampled Twitter influence graph and the real early adopters used as seeds."""

    graph: nx.DiGraph
    initial_infected: list[int]
    metadata: dict[str, object]


def missing_higgs_data_files() -> list[Path]:
    """Return the required raw files that are not present locally."""
    return [path for path in RAW_DATA_URLS if not path.exists()]


def download_higgs_data() -> list[Path]:
    """Download the small raw SNAP files needed by this app."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    downloaded = []
    for path, url in RAW_DATA_URLS.items():
        with urllib.request.urlopen(url, timeout=60) as response, path.open("wb") as output:
            shutil.copyfileobj(response, output)
        downloaded.append(path)
    return downloaded


def load_higgs_retweet_sample(
    sample_size: int,
    initial_seed_count: int,
    random_seed: int,
    min_edge_weight: int = 1,
    reverse_retweets: bool = True,
    influencer_fraction: float = 0.06,
) -> HiggsRetweetSample:
    """Load a directed influence sample from real Higgs retweet activity."""
    _require_higgs_data()
    sample_size = max(20, int(sample_size))
    initial_seed_count = max(1, int(initial_seed_count))
    min_edge_weight = max(1, int(min_edge_weight))

    retweets = _read_retweet_edges()
    retweets = retweets[retweets["weight"] >= min_edge_weight]
    if retweets.empty:
        raise ValueError("No Higgs retweet edges remain after the weight filter.")

    full_graph = _build_retweet_influence_graph(retweets, reverse_retweets)
    activity = _read_activity_events()
    first_seen = _first_seen_by_user(activity)
    early_users = _early_topic_users(activity, full_graph)
    influence_ranking = _rank_by_influence(full_graph)
    selected_nodes = _select_sample_nodes(
        graph=full_graph,
        sample_size=sample_size,
        initial_seed_count=initial_seed_count,
        early_users=early_users,
        influence_ranking=influence_ranking,
        random_seed=random_seed,
        influencer_fraction=influencer_fraction,
    )
    subgraph = full_graph.subgraph(selected_nodes).copy()
    relabeled_graph, old_to_new = _relabel_and_annotate_graph(
        subgraph,
        first_seen=first_seen,
        influencer_fraction=influencer_fraction,
        reverse_retweets=reverse_retweets,
        min_edge_weight=min_edge_weight,
    )
    initial_infected = _initial_seed_nodes(
        early_users=early_users,
        influence_ranking=influence_ranking,
        old_to_new=old_to_new,
        count=initial_seed_count,
    )
    metadata = _sample_metadata(
        graph=relabeled_graph,
        activity=activity,
        retweets=retweets,
        initial_infected=initial_infected,
        reverse_retweets=reverse_retweets,
        min_edge_weight=min_edge_weight,
    )
    relabeled_graph.graph.update(metadata)

    return HiggsRetweetSample(
        graph=relabeled_graph,
        initial_infected=initial_infected,
        metadata=metadata,
    )


def observed_activity_history(graph: nx.Graph, max_steps: int) -> pd.DataFrame:
    """Return observed cumulative sample activity rescaled to simulation steps."""
    timestamps = [
        int(data["first_seen_timestamp"])
        for _node, data in graph.nodes(data=True)
        if data.get("first_seen_timestamp") is not None
    ]
    max_steps = max(1, int(max_steps))
    if not timestamps:
        return pd.DataFrame(
            {
                "t": list(range(max_steps + 1)),
                "observed_adopters": [0] * (max_steps + 1),
            }
        )

    start = min(timestamps)
    end = max(timestamps)
    if start == end:
        scaled_steps = [0 for _timestamp in timestamps]
    else:
        scaled_steps = [
            min(max_steps, int((timestamp - start) / (end - start) * max_steps))
            for timestamp in timestamps
        ]

    counts = pd.Series(scaled_steps).value_counts().sort_index()
    history = (
        counts.reindex(range(max_steps + 1), fill_value=0)
        .cumsum()
        .reset_index()
    )
    history.columns = ["t", "observed_adopters"]
    return history


def _require_higgs_data() -> None:
    missing = missing_higgs_data_files()
    if missing:
        missing_names = ", ".join(path.name for path in missing)
        raise FileNotFoundError(
            "Missing Higgs Twitter data files: "
            f"{missing_names}. Use the sidebar download button or run "
            "`python scripts/download_higgs_twitter.py`."
        )


def _read_retweet_edges() -> pd.DataFrame:
    return pd.read_csv(
        RETWEET_PATH,
        compression="gzip",
        sep=r"\s+",
        names=["retweeter", "retweeted", "weight"],
        dtype={"retweeter": "int64", "retweeted": "int64", "weight": "int64"},
    )


def _read_activity_events() -> pd.DataFrame:
    return pd.read_csv(
        ACTIVITY_PATH,
        compression="gzip",
        sep=r"\s+",
        names=["user_a", "user_b", "timestamp", "interaction"],
        dtype={
            "user_a": "int64",
            "user_b": "int64",
            "timestamp": "int64",
            "interaction": "category",
        },
    )


def _build_retweet_influence_graph(
    retweets: pd.DataFrame,
    reverse_retweets: bool,
) -> nx.DiGraph:
    if reverse_retweets:
        edges = retweets.rename(
            columns={"retweeted": "source", "retweeter": "target"}
        )
    else:
        edges = retweets.rename(
            columns={"retweeter": "source", "retweeted": "target"}
        )

    graph = nx.from_pandas_edgelist(
        edges,
        source="source",
        target="target",
        edge_attr="weight",
        create_using=nx.DiGraph(),
    )
    graph.remove_edges_from(nx.selfloop_edges(graph))
    return graph


def _first_seen_by_user(activity: pd.DataFrame) -> dict[int, int]:
    actors = activity[["user_a", "timestamp"]].rename(columns={"user_a": "user"})
    targets = activity[["user_b", "timestamp"]].rename(columns={"user_b": "user"})
    first_seen = (
        pd.concat([actors, targets], ignore_index=True)
        .groupby("user", sort=False)["timestamp"]
        .min()
    )
    return {int(user): int(timestamp) for user, timestamp in first_seen.items()}


def _early_topic_users(activity: pd.DataFrame, graph: nx.DiGraph) -> list[int]:
    early_users = []
    seen = set()
    for row in activity.sort_values("timestamp", kind="mergesort").itertuples(index=False):
        if row.interaction == "RT":
            candidates = (int(row.user_b), int(row.user_a))
        else:
            candidates = (int(row.user_a), int(row.user_b))

        for user in candidates:
            if user in graph and user not in seen:
                early_users.append(user)
                seen.add(user)
        if len(early_users) >= 10000:
            break
    return early_users


def _rank_by_influence(graph: nx.DiGraph) -> list[int]:
    weighted_out = dict(graph.out_degree(weight="weight"))
    plain_out = dict(graph.out_degree())
    return sorted(
        graph.nodes,
        key=lambda node: (-weighted_out.get(node, 0), -plain_out.get(node, 0), node),
    )


def _select_sample_nodes(
    graph: nx.DiGraph,
    sample_size: int,
    initial_seed_count: int,
    early_users: list[int],
    influence_ranking: list[int],
    random_seed: int,
    influencer_fraction: float,
) -> set[int]:
    target_size = min(sample_size, graph.number_of_nodes())
    influencer_count = max(1, round(target_size * influencer_fraction))
    early_count = max(initial_seed_count, round(target_size * 0.08), 3)

    selected = set(early_users[:early_count])
    selected.update(influence_ranking[:influencer_count])

    frontier = list(
        dict.fromkeys(
            early_users[: max(early_count, target_size // 4)]
            + influence_ranking[: max(influencer_count, target_size // 6)]
        )
    )
    weighted_out = dict(graph.out_degree(weight="weight"))
    frontier_index = 0

    while len(selected) < target_size and frontier_index < len(frontier):
        node = frontier[frontier_index]
        frontier_index += 1
        neighbors = _ranked_neighbors(graph, node, weighted_out)
        for neighbor in neighbors:
            if neighbor not in selected:
                selected.add(neighbor)
                frontier.append(neighbor)
            if len(selected) >= target_size:
                break

    if len(selected) < target_size:
        for node in influence_ranking:
            selected.add(node)
            if len(selected) >= target_size:
                break

    if len(selected) < target_size:
        rng = np.random.default_rng(random_seed)
        remaining = [node for node in graph.nodes if node not in selected]
        fill_count = min(target_size - len(selected), len(remaining))
        if fill_count:
            selected.update(rng.choice(remaining, size=fill_count, replace=False).tolist())

    return selected


def _ranked_neighbors(
    graph: nx.DiGraph,
    node: int,
    weighted_out: dict[int, float],
) -> list[int]:
    neighbors = set(graph.successors(node)) | set(graph.predecessors(node))

    def edge_weight(neighbor: int) -> float:
        forward = graph[node][neighbor].get("weight", 0) if graph.has_edge(node, neighbor) else 0
        backward = graph[neighbor][node].get("weight", 0) if graph.has_edge(neighbor, node) else 0
        return float(forward + backward)

    return sorted(
        neighbors,
        key=lambda neighbor: (-edge_weight(neighbor), -weighted_out.get(neighbor, 0), neighbor),
    )


def _relabel_and_annotate_graph(
    graph: nx.DiGraph,
    first_seen: dict[int, int],
    influencer_fraction: float,
    reverse_retweets: bool,
    min_edge_weight: int,
) -> tuple[nx.DiGraph, dict[int, int]]:
    weighted_out = dict(graph.out_degree(weight="weight"))
    missing_timestamp = 2**63 - 1
    ordered_nodes = sorted(
        graph.nodes,
        key=lambda node: (
            first_seen.get(node, missing_timestamp),
            -weighted_out.get(node, 0),
            node,
        ),
    )
    old_to_new = {old: new for new, old in enumerate(ordered_nodes)}
    relabeled = nx.relabel_nodes(graph, old_to_new, copy=True)

    id_attrs = {new: int(old) for old, new in old_to_new.items()}
    first_seen_attrs = {
        new: first_seen.get(old)
        for old, new in old_to_new.items()
    }
    nx.set_node_attributes(relabeled, id_attrs, "twitter_user_id")
    nx.set_node_attributes(relabeled, first_seen_attrs, "first_seen_timestamp")

    influencer_count = max(1, round(relabeled.number_of_nodes() * influencer_fraction))
    ranked = sorted(
        relabeled.nodes,
        key=lambda node: (
            -relabeled.out_degree(node, weight="weight"),
            -relabeled.out_degree(node),
            node,
        ),
    )
    influencers = set(ranked[:influencer_count])
    nx.set_node_attributes(
        relabeled,
        {
            node: "influencer" if node in influencers else "peer"
            for node in relabeled.nodes
        },
        "role",
    )
    relabeled.graph["influencer_count"] = influencer_count
    relabeled.graph["influencer_fraction"] = influencer_count / relabeled.number_of_nodes()
    relabeled.graph["reverse_retweets"] = reverse_retweets
    relabeled.graph["min_edge_weight"] = min_edge_weight
    return relabeled, old_to_new


def _initial_seed_nodes(
    early_users: list[int],
    influence_ranking: list[int],
    old_to_new: dict[int, int],
    count: int,
) -> list[int]:
    selected = []
    seen = set()
    for user in early_users + influence_ranking:
        if user not in old_to_new:
            continue
        node = old_to_new[user]
        if node not in seen:
            selected.append(node)
            seen.add(node)
        if len(selected) >= count:
            break
    return sorted(selected)


def _sample_metadata(
    graph: nx.DiGraph,
    activity: pd.DataFrame,
    retweets: pd.DataFrame,
    initial_infected: list[int],
    reverse_retweets: bool,
    min_edge_weight: int,
) -> dict[str, object]:
    timestamps = [
        int(data["first_seen_timestamp"])
        for _node, data in graph.nodes(data=True)
        if data.get("first_seen_timestamp") is not None
    ]
    interaction_counts = {
        str(interaction): int(count)
        for interaction, count in activity["interaction"].value_counts().sort_index().items()
    }
    initial_user_ids = tuple(
        int(graph.nodes[node]["twitter_user_id"])
        for node in initial_infected
        if node in graph
    )

    return {
        "data_source": "twitter_higgs",
        "source_name": "SNAP Higgs Twitter Dataset",
        "source_url": SNAP_DATASET_URL,
        "citation": HIGGS_CITATION,
        "topology_label": HIGGS_TWITTER_TOPOLOGY,
        "raw_retweet_edges": int(len(retweets)),
        "raw_activity_events": int(len(activity)),
        "sample_nodes": int(graph.number_of_nodes()),
        "sample_edges": int(graph.number_of_edges()),
        "initial_seed_nodes": tuple(int(node) for node in initial_infected),
        "initial_seed_user_ids": initial_user_ids,
        "first_timestamp": min(timestamps) if timestamps else None,
        "last_timestamp": max(timestamps) if timestamps else None,
        "interaction_counts": interaction_counts,
        "reverse_retweets": bool(reverse_retweets),
        "min_edge_weight": int(min_edge_weight),
    }
