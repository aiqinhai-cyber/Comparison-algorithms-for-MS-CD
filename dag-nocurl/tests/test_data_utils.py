from pathlib import Path

import networkx as nx
import numpy as np

from data_utils import load_and_sample

TOY_BIF = """
network toy { }

variable A {
  type discrete [ 2 ] { no, yes };
}

variable B {
  type discrete [ 2 ] { low, high };
}

probability ( A ) {
  table 0.7, 0.3;
}

probability ( B | A ) {
  (no) 0.9, 0.1;
  (yes) 0.2, 0.8;
}
"""


def test_bif_sampling_is_seeded_and_numpy2_compatible(tmp_path: Path) -> None:
    path = tmp_path / "toy.bif"
    path.write_text(TOY_BIF, encoding="utf-8")

    first = load_and_sample(path, sample_size=100, seed=42)
    second = load_and_sample(path, sample_size=100, seed=42)

    assert np.array_equal(first["codes"], second["codes"])
    assert np.allclose(first["X"].mean(axis=0), 0.0)
    assert first["cardinalities"] == [2, 2]
    assert set(first["true_graph"].edges()) == {("A", "B")}
    assert nx.is_directed_acyclic_graph(first["true_graph"])
