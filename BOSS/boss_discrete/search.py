from __future__ import annotations

import random
import time
from dataclasses import dataclass

import numpy as np
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode
from causallearn.score.LocalScoreFunctionClass import LocalScoreClass
from causallearn.search.PermutationBased.gst import GST
from causallearn.utils.DAG2CPDAG import dag2cpdag

from .bdeu import local_score_bdeu_fast


@dataclass
class BossResult:
    dag: GeneralGraph
    cpdag: GeneralGraph
    dag_adjacency: np.ndarray
    order: list[int]
    parents: dict[int, list[int]]
    score: float
    runtime_seconds: float
    starts: int


def _reversed_enumerate(values, index):
    for value in reversed(values):
        yield index, value
        index -= 1


def _better_mutation(vertex: int, order: list[int], trees: list[GST]) -> bool:
    """Best one-variable relocation used by causal-learn's BOSS."""
    old_index = order.index(vertex)
    number_of_variables = len(order)
    scores = np.zeros(number_of_variables + 1)

    prefix: list[int] = []
    accumulated = 0.0
    for index, other in enumerate(order):
        scores[index] = trees[vertex].trace(prefix) + accumulated
        if vertex != other:
            accumulated += trees[other].trace(prefix)
            prefix.append(other)

    scores[number_of_variables] = trees[vertex].trace(prefix) + accumulated
    best_index = number_of_variables

    prefix.append(vertex)
    accumulated = 0.0
    for index, other in _reversed_enumerate(order, number_of_variables - 1):
        if vertex != other:
            prefix.remove(other)
            accumulated += trees[other].trace(prefix)
        scores[index] += accumulated
        if scores[index] > scores[best_index]:
            best_index = index

    if scores[old_index] + 1e-6 > scores[best_index]:
        return False

    order.remove(vertex)
    order.insert(best_index - int(best_index > old_index), vertex)
    return True


def _run_one_start(
    score: LocalScoreClass,
    initial_order: list[int],
    rng: random.Random,
    max_sweeps: int,
) -> tuple[list[int], dict[int, list[int]], float]:
    order = list(initial_order)
    trees = [GST(vertex, score) for vertex in range(len(order))]
    variables = list(range(len(order)))

    for _ in range(max_sweeps):
        improved = False
        rng.shuffle(variables)
        for vertex in variables:
            improved |= _better_mutation(vertex, order, trees)
        if not improved:
            break
    else:
        raise RuntimeError(
            f"BOSS did not converge within max_sweeps={max_sweeps}. "
            "Increase max_sweeps and rerun."
        )

    parents = {vertex: [] for vertex in order}
    total_score = 0.0
    for index, vertex in enumerate(order):
        total_score += trees[vertex].trace(order[:index], parents[vertex])
    return order, parents, float(total_score)


def _make_graphs(
    node_names: list[str], parents: dict[int, list[int]]
) -> tuple[GeneralGraph, GeneralGraph, np.ndarray]:
    nodes = [GraphNode(name) for name in node_names]
    dag = GeneralGraph(nodes)
    adjacency = np.zeros((len(nodes), len(nodes)), dtype=np.int8)
    for child, parent_list in parents.items():
        for parent in parent_list:
            dag.add_directed_edge(nodes[parent], nodes[child])
            adjacency[parent, child] = 1
    return dag, dag2cpdag(dag), adjacency


def boss_bdeu(
    data: np.ndarray,
    node_names: list[str],
    cardinalities: np.ndarray,
    equivalent_sample_size: float = 10.0,
    structure_prior: float = 1.0,
    max_parents: int | None = 5,
    max_parent_configurations: int | None = 1_000_000,
    number_of_starts: int = 1,
    seed: int = 42,
    max_sweeps: int = 100,
) -> BossResult:
    """Run a configurable discrete BOSS search using causal-learn GSTs."""
    data = np.asarray(data, dtype=np.int64)
    cardinalities = np.asarray(cardinalities, dtype=np.int64)
    if data.ndim != 2:
        raise ValueError("data must be a 2-D array")
    if data.shape[1] != len(node_names) or data.shape[1] != len(cardinalities):
        raise ValueError("data, node_names, and cardinalities disagree")
    if number_of_starts < 1:
        raise ValueError("number_of_starts must be at least 1")

    parameters = {
        "sample_prior": equivalent_sample_size,
        "structure_prior": structure_prior,
        "r_i_map": cardinalities,
        "max_parents": max_parents,
        "max_parent_configurations": max_parent_configurations,
    }
    local_score = LocalScoreClass(
        data=data,
        local_score_fun=local_score_bdeu_fast,
        parameters=parameters,
    )

    started = time.perf_counter()
    best = None
    for start in range(number_of_starts):
        rng = random.Random(seed + start)
        initial_order = list(range(data.shape[1]))
        if start > 0:
            rng.shuffle(initial_order)
        candidate = _run_one_start(local_score, initial_order, rng, max_sweeps)
        if best is None or candidate[2] > best[2]:
            best = candidate

    assert best is not None
    order, parents, total_score = best
    dag, cpdag, adjacency = _make_graphs(node_names, parents)
    runtime = time.perf_counter() - started
    return BossResult(
        dag=dag,
        cpdag=cpdag,
        dag_adjacency=adjacency,
        order=order,
        parents=parents,
        score=total_score,
        runtime_seconds=runtime,
        starts=number_of_starts,
    )

