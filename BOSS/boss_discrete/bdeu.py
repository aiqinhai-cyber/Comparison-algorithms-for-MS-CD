from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import gammaln


def local_score_bdeu_fast(
    data: np.ndarray,
    node: int,
    parents: list[int],
    parameters: dict[str, Any],
) -> float:
    """Return negative BDeu because causal-learn's GST negates local scores.

    Complete cardinalities are supplied from the BIF, so unobserved states in
    small samples are still represented correctly.
    """
    sample_prior = float(parameters["sample_prior"])
    structure_prior = float(parameters["structure_prior"])
    cardinalities = np.asarray(parameters["r_i_map"], dtype=np.int64)
    max_parents = parameters.get("max_parents")
    max_configs = parameters.get("max_parent_configurations")

    if sample_prior <= 0:
        raise ValueError("sample_prior/ESS must be positive")
    if not 0 < structure_prior < data.shape[0] - 1:
        raise ValueError("structure_prior must be in (0, n - 1)")
    if max_parents is not None and len(parents) > int(max_parents):
        return float("inf")

    parents = sorted(int(parent) for parent in parents)
    child_cardinality = int(cardinalities[node])
    parent_cardinalities = cardinalities[parents] if parents else np.array([], dtype=int)
    q_i = int(np.prod(parent_cardinalities, dtype=np.int64)) if parents else 1
    if max_configs is not None and q_i > int(max_configs):
        return float("inf")

    if parents:
        multipliers = np.ones(len(parents), dtype=np.int64)
        if len(parents) > 1:
            multipliers[1:] = np.cumprod(parent_cardinalities[:-1])
        parent_config = (data[:, parents] * multipliers).sum(axis=1)
    else:
        parent_config = np.zeros(data.shape[0], dtype=np.int64)

    counts = np.zeros((q_i, child_cardinality), dtype=np.int64)
    np.add.at(counts, (parent_config, data[:, node]), 1)
    parent_counts = counts.sum(axis=1)

    alpha_ij = sample_prior / q_i
    alpha_ijk = sample_prior / (q_i * child_cardinality)
    bdeu = np.sum(gammaln(alpha_ij) - gammaln(alpha_ij + parent_counts))
    bdeu += np.sum(gammaln(alpha_ijk + counts) - gammaln(alpha_ijk))

    # Match causal-learn's optional structure-prior convention.
    vm = data.shape[0] - 1
    edge_probability = structure_prior / vm
    bdeu += len(parents) * np.log(edge_probability)
    bdeu += (vm - len(parents)) * np.log1p(-edge_probability)

    return float(-bdeu)

