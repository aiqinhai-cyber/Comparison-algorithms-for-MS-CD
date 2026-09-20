"""Partially directed graph helpers and Meek orientation rules."""

from __future__ import annotations

import networkx as nx
import numpy as np


def adjacent(graph: nx.DiGraph, x: str, y: str) -> bool:
    return graph.has_edge(x, y) or graph.has_edge(y, x)


def directed(graph: nx.DiGraph, x: str, y: str) -> bool:
    return graph.has_edge(x, y) and not graph.has_edge(y, x)


def undirected(graph: nx.DiGraph, x: str, y: str) -> bool:
    return graph.has_edge(x, y) and graph.has_edge(y, x)


def directed_part(graph: nx.DiGraph) -> nx.DiGraph:
    result = nx.DiGraph()
    result.add_nodes_from(graph.nodes())
    result.add_edges_from((x, y) for x, y in graph.edges() if not graph.has_edge(y, x))
    return result


def orient(graph: nx.DiGraph, parent: str, child: str) -> bool:
    """Orient parent-child if compatible with current arrows and acyclicity."""
    if not adjacent(graph, parent, child) or directed(graph, child, parent):
        return False
    if directed(graph, parent, child):
        return True
    dag = directed_part(graph)
    if nx.has_path(dag, child, parent):
        return False
    graph.remove_edge(child, parent)
    return True


def apply_meek_rules(graph: nx.DiGraph) -> int:
    """Apply the standard R1-R4 Meek rules until convergence."""
    orientations = 0
    changed = True
    while changed:
        changed = False
        nodes = list(graph.nodes())

        # R1: a -> b - c and a,c nonadjacent implies b -> c.
        for b in nodes:
            parents = [a for a in nodes if directed(graph, a, b)]
            neighbors = [c for c in nodes if undirected(graph, b, c)]
            for a in parents:
                for c in neighbors:
                    if not adjacent(graph, a, c) and orient(graph, b, c):
                        orientations += 1
                        changed = True

        # R2: a - b and a -> c -> b implies a -> b.
        for a in nodes:
            for b in nodes:
                if not undirected(graph, a, b):
                    continue
                if any(
                    directed(graph, a, c) and directed(graph, c, b) for c in nodes
                ) and orient(graph, a, b):
                    orientations += 1
                    changed = True

        # R3: a-b, a-c, a-d, c->b, d->b, c and d nonadjacent => a->b.
        for a in nodes:
            for b in nodes:
                if not undirected(graph, a, b):
                    continue
                candidates = [
                    c
                    for c in nodes
                    if c not in (a, b)
                    and undirected(graph, a, c)
                    and directed(graph, c, b)
                ]
                found = any(
                    not adjacent(graph, c, d)
                    for i, c in enumerate(candidates)
                    for d in candidates[i + 1 :]
                )
                if found and orient(graph, a, b):
                    orientations += 1
                    changed = True

        # R4: a-b, a-c, c->d->b, c and b nonadjacent => a->b.
        for a in nodes:
            for b in nodes:
                if not undirected(graph, a, b):
                    continue
                found = any(
                    c not in (a, b)
                    and undirected(graph, a, c)
                    and not adjacent(graph, c, b)
                    and any(
                        directed(graph, c, d) and directed(graph, d, b) for d in nodes
                    )
                    for c in nodes
                )
                if found and orient(graph, a, b):
                    orientations += 1
                    changed = True
    return orientations


def complete_to_dag(graph: nx.DiGraph, seed: int) -> nx.DiGraph:
    """Randomly complete the PDAG while preserving all compatible arrows."""
    rng = np.random.default_rng(seed)
    dag = directed_part(graph)
    if not nx.is_directed_acyclic_graph(dag):
        raise ValueError("The directed part of the PDAG contains a cycle")

    # A randomized Kahn order respects every already directed edge.
    indegree = dict(dag.in_degree())
    available = [node for node, degree in indegree.items() if degree == 0]
    order: list[str] = []
    while available:
        index = int(rng.integers(len(available)))
        node = available.pop(index)
        order.append(node)
        for child in dag.successors(node):
            indegree[child] -= 1
            if indegree[child] == 0:
                available.append(child)
    if len(order) != len(dag):
        raise ValueError("Could not construct a topological order")
    position = {node: index for index, node in enumerate(order)}
    for x, y in {
        tuple(sorted((left, right)))
        for left, right in graph.edges()
        if undirected(graph, left, right)
    }:
        dag.add_edge(x, y) if position[x] < position[y] else dag.add_edge(y, x)
    if not nx.is_directed_acyclic_graph(dag):
        raise AssertionError("PDAG completion unexpectedly produced a cycle")
    return dag
