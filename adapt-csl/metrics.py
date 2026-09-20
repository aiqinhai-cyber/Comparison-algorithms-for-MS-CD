"""Directed graph metrics used by the Adapt-CSL benchmark."""

from __future__ import annotations

import networkx as nx


def structural_hamming_distance(
    estimated: nx.DiGraph,
    truth: nx.DiGraph,
) -> dict[str, int]:
    """Compute SHD where a reversed edge costs one."""
    estimated_edges = set(estimated.edges())
    true_edges = set(truth.edges())
    missing = sum(
        1
        for edge in true_edges
        if edge not in estimated_edges and (edge[1], edge[0]) not in estimated_edges
    )
    extra = sum(
        1
        for edge in estimated_edges
        if edge not in true_edges and (edge[1], edge[0]) not in true_edges
    )
    reversed_edges = sum(
        1 for edge in estimated_edges if (edge[1], edge[0]) in true_edges
    )
    return {
        "shd": missing + extra + reversed_edges,
        "missing": missing,
        "extra": extra,
        "reversed": reversed_edges,
    }


def directed_metrics(
    estimated: nx.DiGraph,
    truth: nx.DiGraph,
) -> dict[str, float | int]:
    estimated_edges = set(estimated.edges())
    true_edges = set(truth.edges())
    true_positive = len(estimated_edges & true_edges)
    precision = true_positive / len(estimated_edges) if estimated_edges else 0.0
    recall = true_positive / len(true_edges) if true_edges else 0.0
    f1 = (
        2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    )
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        **structural_hamming_distance(estimated, truth),
        "true_edges": len(true_edges),
        "learned_edges": len(estimated_edges),
    }
