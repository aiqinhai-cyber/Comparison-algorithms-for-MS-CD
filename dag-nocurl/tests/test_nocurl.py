import networkx as nx
import numpy as np

from metrics import graph_at_threshold
from nocurl import NoCurl2


def test_nocurl_smoke_returns_finite_dag() -> None:
    rng = np.random.default_rng(7)
    sample_count = 120
    x0 = rng.normal(size=sample_count)
    x1 = 0.8 * x0 + rng.normal(scale=0.3, size=sample_count)
    x2 = -0.5 * x1 + rng.normal(scale=0.3, size=sample_count)
    data = np.column_stack([x0, x1, x2])
    data -= data.mean(axis=0)

    learner = NoCurl2()
    weights, loss = learner.fit_raw(data)
    graph = graph_at_threshold(weights, ["X0", "X1", "X2"], 0.3)

    assert weights.shape == (3, 3)
    assert np.isfinite(weights).all()
    assert np.isfinite(loss)
    assert set(learner.diagnostics) == {"stage1", "stage2", "weight_stage"}
    assert nx.is_directed_acyclic_graph(graph)
