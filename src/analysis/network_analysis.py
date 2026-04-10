"""Network analysis: emotion–bias bipartite network and intra-domain projections."""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / "src/review_stages/analysis_outputs"


def build_bipartite_network(
    cluster_ecbss: pd.DataFrame,
    ecbss_threshold: float = 100.0,
) -> nx.Graph:
    """Build bipartite graph: emotion clusters ↔ bias families.

    Nodes: emotion clusters (type='emotion') and bias families (type='bias')
    Edges: if |mean ECBSS| > threshold, with weight = |ECBSS| and sign attribute.
    """
    G = nx.Graph()

    # Add emotion cluster nodes
    for cluster_id in cluster_ecbss.index:
        G.add_node(f"ec_{cluster_id}", node_type="emotion", cluster_id=int(cluster_id))

    # Add bias family nodes
    for family in cluster_ecbss.columns:
        short = family.replace("_biases", "").replace("_", " ").title()[:30]
        G.add_node(f"bf_{family}", node_type="bias", family=family, label=short)

    # Add edges where |ECBSS| > threshold
    for cluster_id in cluster_ecbss.index:
        for family in cluster_ecbss.columns:
            ecbss = cluster_ecbss.loc[cluster_id, family]
            if abs(ecbss) > ecbss_threshold:
                G.add_edge(
                    f"ec_{cluster_id}",
                    f"bf_{family}",
                    weight=abs(ecbss),
                    ecbss=float(ecbss),
                    direction="amplify" if ecbss > 0 else "attenuate",
                )

    return G


def build_full_bipartite_network(
    ecbss_df: pd.DataFrame,
    cluster_labels: pd.Series,
    ecbss_threshold: float = 200.0,
) -> nx.DiGraph:
    """Build directed bipartite graph with individual emotion nodes."""
    G = nx.DiGraph()

    for emotion in ecbss_df.index:
        cluster_id = cluster_labels.get(emotion, -1)
        G.add_node(
            emotion,
            node_type="emotion",
            cluster_id=int(cluster_id),
        )

    for family in ecbss_df.columns:
        G.add_node(family, node_type="bias")

    for emotion in ecbss_df.index:
        for family in ecbss_df.columns:
            ecbss = ecbss_df.loc[emotion, family]
            if abs(ecbss) > ecbss_threshold:
                G.add_edge(
                    emotion,
                    family,
                    weight=float(abs(ecbss)),
                    ecbss=float(ecbss),
                    direction="amplify" if ecbss > 0 else "attenuate",
                )

    return G


def compute_network_metrics(G: nx.Graph) -> dict:
    """Compute centrality and connectivity metrics."""
    metrics = {}

    # Degree centrality
    dc = nx.degree_centrality(G)
    metrics["degree_centrality"] = {n: float(v) for n, v in dc.items()}

    # Betweenness centrality
    bc = nx.betweenness_centrality(G, weight="weight")
    metrics["betweenness_centrality"] = {n: float(v) for n, v in bc.items()}

    # Connected components
    if not G.is_directed():
        components = list(nx.connected_components(G))
        metrics["n_components"] = len(components)
        metrics["largest_component_size"] = max(len(c) for c in components)
    else:
        metrics["n_components"] = nx.number_weakly_connected_components(G)

    # Density
    metrics["density"] = float(nx.density(G))

    # Separate emotion and bias node rankings
    emotion_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "emotion"]
    bias_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "bias"]

    metrics["top_emotion_by_degree"] = sorted(
        [(n, dc[n]) for n in emotion_nodes if n in dc],
        key=lambda x: -x[1]
    )[:10]
    metrics["top_bias_by_degree"] = sorted(
        [(n, dc[n]) for n in bias_nodes if n in dc],
        key=lambda x: -x[1]
    )[:10]

    # Edge weight statistics
    edge_weights = [d["weight"] for _, _, d in G.edges(data=True) if "weight" in d]
    if edge_weights:
        metrics["mean_edge_weight"] = float(np.mean(edge_weights))
        metrics["max_edge_weight"] = float(np.max(edge_weights))
        metrics["n_amplifying_edges"] = sum(
            1 for _, _, d in G.edges(data=True)
            if d.get("direction") == "amplify"
        )
        metrics["n_attenuating_edges"] = sum(
            1 for _, _, d in G.edges(data=True)
            if d.get("direction") == "attenuate"
        )

    return metrics


def compute_emotion_emotion_similarity(
    ecbss_df: pd.DataFrame,
    top_n: int = 30,
) -> pd.DataFrame:
    """Build emotion–emotion similarity network based on ECBSS profiles."""
    # Cosine similarity between emotion ECBSS vectors
    mat = ecbss_df.values.astype(float)
    # Fill NaN with 0
    mat = np.nan_to_num(mat)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normed = mat / norms
    sim = normed @ normed.T  # (N_emotions × N_emotions)

    sim_df = pd.DataFrame(sim, index=ecbss_df.index, columns=ecbss_df.index)
    return sim_df


def compute_bias_bias_similarity(ecbss_df: pd.DataFrame) -> pd.DataFrame:
    """Build bias–bias similarity based on shared emotion profiles."""
    mat = ecbss_df.values.astype(float)
    mat = np.nan_to_num(mat).T  # (N_families × N_emotions)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normed = mat / norms
    sim = normed @ normed.T  # (N_families × N_families)

    sim_df = pd.DataFrame(sim, index=ecbss_df.columns, columns=ecbss_df.columns)
    return sim_df
