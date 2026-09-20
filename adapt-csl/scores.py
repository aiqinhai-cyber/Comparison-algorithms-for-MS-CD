"""Discrete G2 conditional-independence test and local BDeu score."""

from __future__ import annotations

from math import prod

import numpy as np
from scipy.special import gammaln
from scipy.stats import chi2


def _mixed_radix_index(values: np.ndarray, cardinalities: list[int]) -> np.ndarray:
    if values.shape[1] == 0:
        return np.zeros(values.shape[0], dtype=np.int64)
    index = np.zeros(values.shape[0], dtype=np.int64)
    for column, cardinality in zip(values.T, cardinalities, strict=True):
        index = index * int(cardinality) + column.astype(np.int64, copy=False)
    return index


def g2_test(
    data: np.ndarray,
    cardinalities: np.ndarray,
    x: int,
    y: int,
    conditioning: tuple[int, ...] = (),
) -> tuple[float, float, int]:
    """Return (p-value, G2 statistic, degrees of freedom)."""
    rx, ry = int(cardinalities[x]), int(cardinalities[y])
    if conditioning:
        z_values = data[:, conditioning]
        z_cards = [int(cardinalities[index]) for index in conditioning]
        strata = _mixed_radix_index(z_values, z_cards)
    else:
        strata = np.zeros(data.shape[0], dtype=np.int64)

    statistic = 0.0
    degrees = 0
    for state in np.unique(strata):
        mask = strata == state
        flat = data[mask, x].astype(np.int64) * ry + data[mask, y]
        table = np.bincount(flat, minlength=rx * ry).reshape(rx, ry)
        active_rows = table.sum(axis=1) > 0
        active_columns = table.sum(axis=0) > 0
        table = table[np.ix_(active_rows, active_columns)]
        if table.shape[0] < 2 or table.shape[1] < 2:
            continue
        row_sums = table.sum(axis=1, keepdims=True)
        column_sums = table.sum(axis=0, keepdims=True)
        expected = row_sums @ column_sums / table.sum()
        positive = table > 0
        statistic += float(
            2.0 * np.sum(table[positive] * np.log(table[positive] / expected[positive]))
        )
        degrees += (table.shape[0] - 1) * (table.shape[1] - 1)
    if degrees == 0:
        return 1.0, 0.0, 0
    return float(chi2.sf(statistic, degrees)), statistic, degrees


def local_bdeu_score(
    data: np.ndarray,
    cardinalities: np.ndarray,
    variable: int,
    parents: tuple[int, ...],
    equivalent_sample_size: float,
) -> float:
    """Compute the decomposable local BDeu score for multistate data."""
    parents = tuple(sorted(parents))
    r = int(cardinalities[variable])
    parent_cards = [int(cardinalities[index]) for index in parents]
    q = int(prod(parent_cards)) if parent_cards else 1
    alpha_ij = equivalent_sample_size / q
    alpha_ijk = equivalent_sample_size / (q * r)
    configurations = (
        _mixed_radix_index(data[:, parents], parent_cards)
        if parents
        else np.zeros(data.shape[0], dtype=np.int64)
    )
    score = 0.0
    # Unobserved parent configurations contribute exactly zero.
    for configuration in np.unique(configurations):
        child = data[configurations == configuration, variable]
        counts = np.bincount(child, minlength=r)
        total = int(counts.sum())
        score += float(gammaln(alpha_ij) - gammaln(alpha_ij + total))
        score += float(np.sum(gammaln(alpha_ijk + counts) - gammaln(alpha_ijk)))
    return score


class DiscreteStatistics:
    """Cached statistical queries used throughout one Adapt-CSL fit."""

    def __init__(self, data: np.ndarray, cardinalities: np.ndarray, ess: float) -> None:
        self.data = np.asarray(data, dtype=np.int64)
        self.cardinalities = np.asarray(cardinalities, dtype=np.int64)
        self.ess = float(ess)
        self.ci_test_count = 0
        self._independence_cache: dict[tuple[int, int, tuple[int, ...]], float] = {}
        self._score_cache: dict[tuple[int, tuple[int, ...]], float] = {}

    def independence(self, x: int, y: int, conditioning: tuple[int, ...]) -> float:
        conditioning = tuple(sorted(conditioning))
        key = (min(int(x), int(y)), max(int(x), int(y)), conditioning)
        if key in self._independence_cache:
            return self._independence_cache[key]
        self.ci_test_count += 1
        value = g2_test(
            self.data,
            self.cardinalities,
            int(x),
            int(y),
            conditioning,
        )[0]
        self._independence_cache[key] = value
        return value

    def score(self, variable: int, parents: tuple[int, ...]) -> float:
        parents = tuple(sorted(parents))
        key = (int(variable), parents)
        if key in self._score_cache:
            return self._score_cache[key]
        value = local_bdeu_score(
            self.data,
            self.cardinalities,
            int(variable),
            parents,
            self.ess,
        )
        self._score_cache[key] = value
        return value
