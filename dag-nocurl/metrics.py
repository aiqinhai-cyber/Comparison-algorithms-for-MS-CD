"""Threshold conversion and directed-graph evaluation metrics."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import networkx as nx
import numpy as np


def graph_at_threshold(
    raw_weights: np.ndarray,
    node_names: Sequence[Any],
    threshold: float,
) -> nx.DiGraph:
    """Apply abs(W) < threshold -> 0, matching the reference convention."""
    adjacency = np.asarray(raw_weights, dtype=float).copy()
    if adjacency.shape != (len(node_names), len(node_names)):
        raise ValueError("Weight matrix shape does not match node count")
    if not np.isfinite(adjacency).all():
        raise ValueError("Weight matrix contains non-finite values")
    adjacency[np.abs(adjacency) < threshold] = 0.0
    graph = nx.DiGraph()
    graph.add_nodes_from(node_names)
    rows, columns = np.nonzero(adjacency)
    graph.add_edges_from(
        (node_names[row], node_names[column])
        for row, column in zip(rows, columns, strict=True)
    )
    return graph


def compute_metrics(
    true_graph: nx.DiGraph,
    predicted_graph: nx.DiGraph,
) -> dict[str, float | int]:
    if set(true_graph.nodes()) != set(predicted_graph.nodes()):
        raise ValueError("Ground-truth and prediction node sets differ")
    if not nx.is_directed_acyclic_graph(true_graph):
        raise ValueError("Ground truth is not a DAG")
    if not nx.is_directed_acyclic_graph(predicted_graph):
        raise ValueError("Prediction is not a DAG")

    true_edges = set(true_graph.edges())
    predicted_edges = set(predicted_graph.edges())
    true_positive = len(true_edges & predicted_edges)
    false_positive = len(predicted_edges - true_edges)
    false_negative = len(true_edges - predicted_edges)
    precision = (
        true_positive / len(predicted_edges) if predicted_edges else 0.0
    )
    recall = true_positive / len(true_edges) if true_edges else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    true_skeleton = {frozenset(edge) for edge in true_edges}
    predicted_skeleton = {frozenset(edge) for edge in predicted_edges}
    extra = len(predicted_skeleton - true_skeleton)
    missing = len(true_skeleton - predicted_skeleton)
    reversed_edges = sum(
        (child, parent) in true_edges for parent, child in predicted_edges
    )
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "shd": int(extra + missing + reversed_edges),
        "TP": true_positive,
        "FP": false_positive,
        "FN": false_negative,
        "extra_skeleton_edges": extra,
        "missing_skeleton_edges": missing,
        "reversed_edges": reversed_edges,
        "true_edges": len(true_edges),
        "learned_edges": len(predicted_edges),
    }
