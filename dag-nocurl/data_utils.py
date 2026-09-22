"""BIF loading, discrete ancestral sampling, encoding, and preprocessing."""

from __future__ import annotations

import logging
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
from pgmpy.readwrite import BIFReader

import config


class WarningCollector(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.messages: Counter[str] = Counter()

    def emit(self, record: logging.LogRecord) -> None:
        self.messages[record.getMessage()] += 1


def resolve_bif_files() -> dict[str, Path]:
    """Resolve every configured BIF file and report all missing files."""
    paths = {
        network: config.BIF_DIR / f"{network}.bif"
        for network in config.NETWORKS
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing BIF files:\n" + "\n".join(missing))
    return {name: path.resolve() for name, path in paths.items()}


def _ancestral_sample(model: Any, sample_size: int, seed: int) -> dict[str, Any]:
    """Sample without pgmpy's legacy np.mat-dependent sampling path."""
    rng = np.random.default_rng(seed)
    graph = nx.DiGraph(model.edges())
    graph.add_nodes_from(model.nodes())
    node_names = list(nx.topological_sort(graph))
    sampled_codes: dict[Any, np.ndarray] = {}
    state_names_by_node: dict[Any, list[Any]] = {}

    for node in node_names:
        cpd = model.get_cpds(node)
        if cpd is None:
            raise ValueError(f"Missing CPD for node {node!r}")
        variables = list(cpd.variables)
        parents = variables[1:]
        state_names = getattr(cpd, "state_names", {}) or {}
        node_states = list(state_names.get(node, range(int(cpd.cardinality[0]))))
        state_names_by_node[node] = node_states
        probabilities = np.asarray(cpd.get_values(), dtype=np.float64)

        if parents:
            parent_cards = [len(state_names[parent]) for parent in parents]
            configurations = np.zeros(sample_size, dtype=np.int64)
            for parent, cardinality in zip(parents, parent_cards, strict=True):
                configurations = (
                    configurations * cardinality + sampled_codes[parent]
                )
        else:
            configurations = np.zeros(sample_size, dtype=np.int64)

        codes = np.empty(sample_size, dtype=np.int64)
        for configuration in np.unique(configurations):
            rows = np.flatnonzero(configurations == configuration)
            probability = probabilities[:, int(configuration)].copy()
            probability = np.clip(probability, 0.0, None)
            total = probability.sum()
            if total <= 0:
                raise ValueError(f"Invalid CPD column for node {node!r}")
            probability /= total
            codes[rows] = rng.choice(
                len(node_states),
                size=len(rows),
                p=probability,
            )
        sampled_codes[node] = codes

    code_matrix = np.column_stack([sampled_codes[node] for node in node_names])
    return {
        "codes": code_matrix,
        "node_names": node_names,
        "states": {
            str(node): [str(state) for state in state_names_by_node[node]]
            for node in node_names
        },
        "cardinalities": [len(state_names_by_node[node]) for node in node_names],
    }


def load_and_sample(
    bif_path: str | Path,
    sample_size: int,
    seed: int,
) -> dict[str, Any]:
    logger = logging.getLogger("pgmpy")
    collector = WarningCollector()
    old_handlers = logger.handlers[:]
    old_propagate = logger.propagate
    old_level = logger.level
    python_messages: Counter[str] = Counter()

    if config.CAPTURE_SAMPLING_WARNINGS:
        logger.handlers = [collector]
        logger.propagate = False
        logger.setLevel(logging.WARNING)

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = BIFReader(str(bif_path)).get_model()
            sampled = _ancestral_sample(model, sample_size, seed)
        for item in caught:
            python_messages[f"{item.category.__name__}: {item.message}"] += 1
    finally:
        if config.CAPTURE_SAMPLING_WARNINGS:
            logger.handlers = old_handlers
            logger.propagate = old_propagate
            logger.setLevel(old_level)

    codes = sampled["codes"]
    cardinalities = sampled["cardinalities"]
    x = codes.astype(np.float64)
    if config.SCALE_BY_DECLARED_CARDINALITY:
        scale = np.maximum(np.asarray(cardinalities, dtype=float) - 1.0, 1.0)
        x = x / scale
    means = x.mean(axis=0)
    if config.CENTER_DATA:
        x = x - means
    if not np.isfinite(x).all():
        raise ValueError("Non-finite input data")

    true_graph = nx.DiGraph()
    true_graph.add_nodes_from(sampled["node_names"])
    true_graph.add_edges_from(model.edges())
    return {
        "X": x,
        "codes": codes,
        "node_names": sampled["node_names"],
        "true_graph": true_graph,
        "states": sampled["states"],
        "cardinalities": cardinalities,
        "means_before_centering": means,
        "constant_columns": [
            str(sampled["node_names"][index])
            for index in range(len(sampled["node_names"]))
            if np.ptp(x[:, index]) == 0
        ],
        "sampling_warnings": {
            "logging": dict(collector.messages),
            "python": dict(python_messages),
        },
    }
