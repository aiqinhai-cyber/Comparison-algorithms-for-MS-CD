"""Small tests that do not run the full benchmark."""

from __future__ import annotations

import networkx as nx
import numpy as np

from adapt_csl import AdaptCSL
from graph_utils import apply_meek_rules, directed
from scores import g2_test


def test_multistate_g2_and_adapt_csl_output() -> None:
    rng = np.random.default_rng(7)
    sample_size = 1200
    x = rng.integers(0, 2, sample_size)
    y = x.copy()
    y[rng.random(sample_size) < 0.05] ^= 1
    z = rng.integers(0, 3, sample_size)
    data = np.column_stack([x, y, z])
    cardinalities = np.array([2, 2, 3])

    assert g2_test(data, cardinalities, 0, 1)[0] < 0.01
    learner = AdaptCSL(max_conditioning_set=2)
    graph = learner.fit(
        data,
        ["X", "Y", "Z"],
        cardinalities,
        seed=42,
    )
    assert nx.is_directed_acyclic_graph(graph)
    assert graph.number_of_edges() == 1


def test_meek_rule_r1() -> None:
    graph = nx.DiGraph()
    graph.add_nodes_from([0, 1, 2])
    graph.add_edge(0, 1)
    graph.add_edge(1, 2)
    graph.add_edge(2, 1)

    apply_meek_rules(graph)
    assert directed(graph, 1, 2)
