import numpy as np
import pandas as pd

from boss_discrete.bif_data import load_and_encode_samples, load_bif_metadata
from boss_discrete.bdeu import local_score_bdeu_fast
from boss_discrete.metrics import directed_metrics
from boss_discrete.search import boss_bdeu


def test_bdeu_parent_order_is_invariant():
    data = np.asarray(
        [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]] * 20,
        dtype=np.int64,
    )
    parameters = {
        "sample_prior": 10.0,
        "structure_prior": 1.0,
        "r_i_map": np.array([2, 2, 2]),
        "max_parents": 5,
        "max_parent_configurations": 1000,
    }
    left = local_score_bdeu_fast(data, 2, [0, 1], parameters)
    right = local_score_bdeu_fast(data, 2, [1, 0], parameters)
    assert np.isclose(left, right)


def test_shd_counts_reversal_once():
    true = np.array([[0, 1], [0, 0]], dtype=np.int8)
    estimated = np.array([[0, 0], [1, 0]], dtype=np.int8)
    metrics = directed_metrics(true, estimated)
    assert metrics["shd"] == 1
    assert metrics["arc_tp"] == 0
    assert metrics["arc_fp"] == 1
    assert metrics["arc_fn"] == 1


def test_boss_returns_a_dag():
    rng = np.random.default_rng(42)
    x = rng.integers(0, 2, size=200)
    noise = rng.random(200) < 0.05
    y = np.bitwise_xor(x, noise.astype(np.int64))
    data = np.column_stack([x, y])
    result = boss_bdeu(
        data,
        node_names=["X", "Y"],
        cardinalities=np.array([2, 2]),
        equivalent_sample_size=10,
        number_of_starts=1,
        seed=42,
    )
    assert result.dag_adjacency.shape == (2, 2)
    assert np.all(np.diag(result.dag_adjacency) == 0)
    assert result.dag_adjacency.sum() <= 1


def test_bif_controls_cardinality_and_column_order(tmp_path):
    bif = tmp_path / "tiny.bif"
    bif.write_text(
        """network tiny {\n}\n"
        "variable A {\n  type discrete [ 3 ] { low, mid, high };\n}\n"
        "variable B {\n  type discrete [ 2 ] { no, yes };\n}\n"
        "probability ( A ) {\n  table 0.4, 0.4, 0.2;\n}\n"
        "probability ( B | A ) {\n"
        "  (low) 0.9, 0.1;\n  (mid) 0.5, 0.5;\n  (high) 0.1, 0.9;\n}\n"
        """,
        encoding="utf-8",
    )
    samples = tmp_path / "tiny_3.csv"
    pd.DataFrame({"B": ["no", "yes", "no"], "A": ["low", "mid", "low"]}).to_csv(
        samples, index=False
    )
    metadata = load_bif_metadata(bif)
    encoded = load_and_encode_samples(samples, metadata)
    assert metadata.nodes == ["A", "B"]
    assert metadata.cardinalities.tolist() == [3, 2]
    assert encoded.shape == (3, 2)
    assert metadata.true_adjacency.tolist() == [[0, 1], [0, 0]]
