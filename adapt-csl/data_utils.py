"""BIF loading and reproducible multistate discrete sampling."""

from __future__ import annotations

import logging
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from pgmpy.readwrite import BIFReader

import config


class WarningCollector(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.messages: Counter[str] = Counter()

    def emit(self, record: logging.LogRecord) -> None:
        self.messages[record.getMessage()] += 1


def resolve_bif_file(filename: str) -> Path:
    candidates = [config.BIF_DIR / filename, config.PROJECT_DIR / filename]
    for path in candidates:
        if path.is_file():
            return path.resolve()
    checked = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"BIF file not found: {filename}. Checked: {checked}")


def _sample_model(model: Any, sample_size: int, seed: int) -> pd.DataFrame:
    """Sample a discrete BN without pgmpy's NumPy-2-incompatible sampler."""
    rng = np.random.default_rng(seed)
    graph = nx.DiGraph(model.edges())
    graph.add_nodes_from(model.nodes())
    order = list(nx.topological_sort(graph))
    sampled_codes: dict[str, np.ndarray] = {}
    sampled_labels: dict[str, np.ndarray] = {}

    for node in order:
        cpd = model.get_cpds(node)
        if cpd is None:
            raise ValueError(f"Missing CPD for node {node!r}")
        variables = list(cpd.variables)
        parents = variables[1:]
        state_names = getattr(cpd, "state_names", {}) or {}
        node_states = list(state_names.get(node, range(int(cpd.cardinality[0]))))
        probabilities = np.asarray(cpd.get_values(), dtype=float)

        if parents:
            parent_cards = [len(state_names[parent]) for parent in parents]
            parent_matrix = np.column_stack(
                [sampled_codes[parent] for parent in parents]
            )
            configurations = np.zeros(sample_size, dtype=np.int64)
            for column, cardinality in zip(
                parent_matrix.T,
                parent_cards,
                strict=True,
            ):
                configurations = configurations * cardinality + column
        else:
            configurations = np.zeros(sample_size, dtype=np.int64)

        codes = np.empty(sample_size, dtype=np.int64)
        for configuration in np.unique(configurations):
            rows = np.flatnonzero(configurations == configuration)
            probability = probabilities[:, int(configuration)].astype(float, copy=True)
            probability = np.clip(probability, 0.0, None)
            total = probability.sum()
            if total <= 0:
                raise ValueError(f"Invalid CPD column for {node!r}")
            probability /= total
            codes[rows] = rng.choice(len(node_states), size=len(rows), p=probability)
        sampled_codes[node] = codes
        sampled_labels[node] = np.asarray(node_states, dtype=object)[codes]
    return pd.DataFrame({node: sampled_labels[node] for node in order})


def _encode_dataframe(
    frame: pd.DataFrame, variables: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    columns, cardinalities = [], []
    for variable in variables:
        codes, unique = pd.factorize(frame[variable], sort=True)
        if np.any(codes < 0):
            raise ValueError(f"Missing value detected in variable {variable!r}")
        columns.append(codes.astype(np.int64))
        cardinalities.append(len(unique))
    return np.column_stack(columns), np.asarray(cardinalities, dtype=np.int64)


def sample_bif(bif_path: str | Path, sample_size: int, seed: int) -> dict[str, Any]:
    logger = logging.getLogger("pgmpy")
    collector = WarningCollector()
    old_handlers, old_propagate, old_level = (
        logger.handlers[:],
        logger.propagate,
        logger.level,
    )
    caught_messages: Counter[str] = Counter()
    if config.CAPTURE_WARNINGS:
        logger.handlers = [collector]
        logger.propagate = False
        logger.setLevel(logging.WARNING)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = BIFReader(str(bif_path)).get_model()
            frame = _sample_model(model, int(sample_size), int(seed))
        for item in caught:
            caught_messages[f"{item.category.__name__}: {item.message}"] += 1
    finally:
        if config.CAPTURE_WARNINGS:
            logger.handlers, logger.propagate, logger.level = (
                old_handlers,
                old_propagate,
                old_level,
            )

    variables = sorted(str(node) for node in model.nodes())
    frame = frame.loc[:, variables]
    data, cardinalities = _encode_dataframe(frame, variables)
    truth = nx.DiGraph()
    truth.add_nodes_from(variables)
    truth.add_edges_from((str(left), str(right)) for left, right in model.edges())
    return {
        "dataframe": frame,
        "data": data,
        "cardinalities": cardinalities,
        "variables": variables,
        "true_graph": truth,
        "sampling_warnings": {
            "logging": dict(collector.messages),
            "python": dict(caught_messages),
        },
    }
