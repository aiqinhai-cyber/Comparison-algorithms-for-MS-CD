import networkx as nx
import numpy as np

from metrics import compute_metrics, graph_at_threshold


def test_directed_metrics_and_shd_count_reversal_once() -> None:
    truth = nx.DiGraph([("A", "B"), ("B", "C")])
    prediction = nx.DiGraph([("B", "A"), ("B", "C"), ("A", "C")])

    metrics = compute_metrics(truth, prediction)

    assert metrics["TP"] == 1
    assert metrics["FP"] == 2
    assert metrics["FN"] == 1
    assert metrics["precision"] == 1 / 3
    assert metrics["recall"] == 1 / 2
    assert metrics["f1"] == 0.4
    assert metrics["shd"] == 2


def test_graph_threshold_uses_absolute_weights() -> None:
    weights = np.array([[0.0, -0.3], [0.0, 0.0]])
    graph = graph_at_threshold(weights, ["A", "B"], threshold=0.3)
    assert set(graph.edges()) == {("A", "B")}
