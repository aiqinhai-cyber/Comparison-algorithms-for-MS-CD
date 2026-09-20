"""Paper-aligned Python implementation of Adapt-CSL Algorithm 1."""

from __future__ import annotations

from itertools import combinations

import networkx as nx
import numpy as np

import config
from graph_utils import (
    apply_meek_rules,
    complete_to_dag,
    directed_part,
    orient,
    undirected,
)
from scores import DiscreteStatistics


class AdaptCSL:
    """Adaptive local-to-global causal structure learner for discrete data.

    Stages correspond to Algorithm 1 in Li et al. (2026): target-wise PC
    learning and separating-set recording, Theorem-3 adaptive AND/OR skeleton
    reconciliation, Theorem-4 V-structure orientation, and Meek propagation.
    """

    def __init__(
        self,
        alpha: float = config.ALPHA,
        max_conditioning_set: int | None = config.MAX_CONDITIONING_SET,
        equivalent_sample_size: float = config.BDEU_EQUIVALENT_SAMPLE_SIZE,
        score_tolerance: float = config.SCORE_TOLERANCE,
    ) -> None:
        self.alpha = float(alpha)
        self.max_conditioning_set = max_conditioning_set
        self.equivalent_sample_size = float(equivalent_sample_size)
        self.score_tolerance = float(score_tolerance)
        self.diagnostics: dict[str, int | float] = {}

    def fit(
        self,
        data: np.ndarray,
        variables: list[str],
        cardinalities: np.ndarray,
        seed: int = 0,
    ) -> nx.DiGraph:
        data = np.asarray(data, dtype=np.int64)
        cardinalities = np.asarray(cardinalities, dtype=np.int64)
        if data.ndim != 2 or data.shape[1] != len(variables):
            raise ValueError("data shape does not match variables")
        if cardinalities.shape != (len(variables),):
            raise ValueError("cardinalities shape does not match variables")
        if np.any(data < 0) or any(
            np.any(data[:, index] >= cardinalities[index])
            for index in range(len(variables))
        ):
            raise ValueError("data contain invalid categorical codes")

        statistics = DiscreteStatistics(
            data, cardinalities, self.equivalent_sample_size
        )
        local_pc, separators = self._learn_local_pc(len(variables), statistics)
        skeleton, adaptive_counts = self._adaptive_skeleton(
            local_pc, separators, statistics
        )
        pdag, collider_count = self._orient_v_structures(skeleton, statistics)
        meek_count = apply_meek_rules(pdag)
        completion_seed = seed if config.RANDOMLY_COMPLETE_UNDIRECTED else 0
        dag = complete_to_dag(pdag, completion_seed)

        mapping = {index: variable for index, variable in enumerate(variables)}
        result = nx.relabel_nodes(dag, mapping, copy=True)
        self.diagnostics = {
            "ci_tests": statistics.ci_test_count,
            "and_rule_count": adaptive_counts["and"],
            "or_rule_count": adaptive_counts["or"],
            "symmetric_pair_count": adaptive_counts["symmetric"],
            "skeleton_edges": skeleton.number_of_edges(),
            "v_structures_oriented": collider_count,
            "meek_orientations": meek_count,
            "directed_before_completion": directed_part(pdag).number_of_edges(),
            "randomly_completed_edges": sum(
                1 for x, y in pdag.edges() if x < y and undirected(pdag, x, y)
            ),
            "learned_edges": result.number_of_edges(),
        }
        return result

    def _learn_local_pc(
        self,
        number_of_variables: int,
        statistics: DiscreteStatistics,
    ) -> tuple[dict[int, set[int]], dict[tuple[int, int], tuple[int, ...]]]:
        """Target-wise order-independent PC-simple implementation of learnPC."""
        local_pc: dict[int, set[int]] = {}
        separators: dict[tuple[int, int], tuple[int, ...]] = {}
        for target in range(number_of_variables):
            candidates: set[int] = set()
            for candidate in range(number_of_variables):
                if candidate == target:
                    continue
                if statistics.independence(target, candidate, ()) < self.alpha:
                    candidates.add(candidate)
                else:
                    separators[(target, candidate)] = ()

            level = 1
            while candidates and len(candidates) > level:
                if (
                    self.max_conditioning_set is not None
                    and level > self.max_conditioning_set
                ):
                    break
                snapshot = set(candidates)
                removals: dict[int, tuple[int, ...]] = {}
                for candidate in sorted(snapshot):
                    others = sorted(snapshot - {candidate})
                    for conditioning in combinations(others, level):
                        if (
                            statistics.independence(target, candidate, conditioning)
                            >= self.alpha
                        ):
                            removals[candidate] = tuple(conditioning)
                            break
                for candidate, conditioning in removals.items():
                    candidates.discard(candidate)
                    separators[(target, candidate)] = conditioning
                level += 1
            local_pc[target] = candidates
        return local_pc, separators

    def _adaptive_skeleton(
        self,
        local_pc: dict[int, set[int]],
        separators: dict[tuple[int, int], tuple[int, ...]],
        statistics: DiscreteStatistics,
    ) -> tuple[nx.Graph, dict[str, int]]:
        """Apply Algorithm 1 lines 7-14 and Theorem 3 exactly once per pair."""
        skeleton = nx.Graph()
        skeleton.add_nodes_from(local_pc)
        counts = {"and": 0, "or": 0, "symmetric": 0}
        for left, right in combinations(sorted(local_pc), 2):
            left_has_right = right in local_pc[left]
            right_has_left = left in local_pc[right]
            if left_has_right and right_has_left:
                skeleton.add_edge(left, right)
                counts["symmetric"] += 1
                continue
            if not left_has_right and not right_has_left:
                continue

            # missing_target is the side whose local PC excluded the other;
            # its recorded separation set is the one used in Theorem 3.
            if not left_has_right:
                missing_target, other = left, right
            else:
                missing_target, other = right, left
            separator = separators.get((missing_target, other), ())
            score_without = statistics.score(missing_target, tuple(separator))
            score_with = statistics.score(
                missing_target, tuple(sorted((*separator, other)))
            )
            if score_without - score_with > self.score_tolerance:
                counts["and"] += 1
            else:
                skeleton.add_edge(left, right)
                counts["or"] += 1
        return skeleton, counts

    def _orient_v_structures(
        self,
        skeleton: nx.Graph,
        statistics: DiscreteStatistics,
    ) -> tuple[nx.DiGraph, int]:
        """Orient unshielded colliders using Theorem 4 score comparison."""
        pdag = nx.DiGraph()
        pdag.add_nodes_from(skeleton.nodes())
        for left, right in skeleton.edges():
            pdag.add_edge(left, right)
            pdag.add_edge(right, left)

        proposals: list[tuple[float, int, int, int]] = []
        for center in sorted(skeleton.nodes()):
            neighbors = sorted(skeleton.neighbors(center))
            for left, right in combinations(neighbors, 2):
                if skeleton.has_edge(left, right):
                    continue
                # Eq. (10): left->center<-right versus left->center->right.
                collider = statistics.score(center, tuple(sorted((left, right))))
                collider += statistics.score(right, ())
                chain = statistics.score(center, (left,))
                chain += statistics.score(right, (center,))
                margin = collider - chain
                if margin > self.score_tolerance:
                    proposals.append((margin, left, center, right))

        accepted = 0
        # Stronger score evidence is applied first when finite-sample proposals
        # conflict. A proposal is accepted only when both arrowheads are legal.
        for _, left, center, right in sorted(proposals, reverse=True):
            trial = pdag.copy()
            if orient(trial, left, center) and orient(trial, right, center):
                pdag = trial
                accepted += 1
        return pdag, accepted
