"""Directed-graph metrics for causal structure discovery."""

from __future__ import annotations

import numpy as np


def directed_metrics(
    true_adjacency: np.ndarray,
    estimated_adjacency: np.ndarray,
) -> dict[str, float | int]:
    """Calculate directed Precision, Recall, F1, and SHD.

    Adjacency convention: ``adjacency[parent, child] = 1``. A reversed edge
    contributes one unit to SHD, which matches the common structural-Hamming
    distance definition used in causal discovery evaluations.
    """
    true = (np.asarray(true_adjacency) != 0).astype(np.int8)
    estimated = (np.asarray(estimated_adjacency) != 0).astype(np.int8)
    if true.ndim != 2 or true.shape[0] != true.shape[1]:
        raise ValueError("The true adjacency matrix must be square")
    if true.shape != estimated.shape:
        raise ValueError("True and estimated adjacency shapes differ")

    np.fill_diagonal(true, 0)
    np.fill_diagonal(estimated, 0)
    true_positive = int(np.sum((true == 1) & (estimated == 1)))
    false_positive = int(np.sum((true == 0) & (estimated == 1)))
    false_negative = int(np.sum((true == 1) & (estimated == 0)))

    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    shd = 0
    skeleton_tp = skeleton_fp = skeleton_fn = 0
    for left in range(true.shape[0]):
        for right in range(left + 1, true.shape[0]):
            true_state = 1 if true[left, right] else (-1 if true[right, left] else 0)
            estimated_state = (
                1
                if estimated[left, right]
                else (-1 if estimated[right, left] else 0)
            )
            shd += int(true_state != estimated_state)

            true_adjacent = true_state != 0
            estimated_adjacent = estimated_state != 0
            skeleton_tp += int(true_adjacent and estimated_adjacent)
            skeleton_fp += int(not true_adjacent and estimated_adjacent)
            skeleton_fn += int(true_adjacent and not estimated_adjacent)

    skeleton_precision = (
        skeleton_tp / (skeleton_tp + skeleton_fp)
        if skeleton_tp + skeleton_fp
        else 0.0
    )
    skeleton_recall = (
        skeleton_tp / (skeleton_tp + skeleton_fn)
        if skeleton_tp + skeleton_fn
        else 0.0
    )
    skeleton_f1 = (
        2.0
        * skeleton_precision
        * skeleton_recall
        / (skeleton_precision + skeleton_recall)
        if skeleton_precision + skeleton_recall
        else 0.0
    )

    return {
        "arc_tp": true_positive,
        "arc_fp": false_positive,
        "arc_fn": false_negative,
        "arc_precision": precision,
        "arc_recall": recall,
        "arc_f1": f1,
        "shd": shd,
        "skeleton_precision": skeleton_precision,
        "skeleton_recall": skeleton_recall,
        "skeleton_f1": skeleton_f1,
        "true_edges": int(true.sum()),
        "estimated_edges": int(estimated.sum()),
    }

