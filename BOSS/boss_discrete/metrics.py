from __future__ import annotations

import numpy as np


def directed_metrics(true: np.ndarray, estimated: np.ndarray) -> dict[str, float | int]:
    true = np.asarray(true, dtype=np.int8)
    estimated = np.asarray(estimated, dtype=np.int8)
    if true.shape != estimated.shape or true.ndim != 2 or true.shape[0] != true.shape[1]:
        raise ValueError("true and estimated must be equally sized square matrices")

    true = (true != 0).astype(np.int8)
    estimated = (estimated != 0).astype(np.int8)
    np.fill_diagonal(true, 0)
    np.fill_diagonal(estimated, 0)

    tp = int(np.sum((true == 1) & (estimated == 1)))
    fp = int(np.sum((true == 0) & (estimated == 1)))
    fn = int(np.sum((true == 1) & (estimated == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    shd = 0
    skeleton_tp = skeleton_fp = skeleton_fn = 0
    for left in range(true.shape[0]):
        for right in range(left + 1, true.shape[0]):
            true_state = 1 if true[left, right] else (-1 if true[right, left] else 0)
            est_state = 1 if estimated[left, right] else (-1 if estimated[right, left] else 0)
            shd += int(true_state != est_state)  # a reversal costs one operation

            true_adjacent = true_state != 0
            est_adjacent = est_state != 0
            skeleton_tp += int(true_adjacent and est_adjacent)
            skeleton_fp += int(not true_adjacent and est_adjacent)
            skeleton_fn += int(true_adjacent and not est_adjacent)

    sk_precision = skeleton_tp / (skeleton_tp + skeleton_fp) if skeleton_tp + skeleton_fp else 0.0
    sk_recall = skeleton_tp / (skeleton_tp + skeleton_fn) if skeleton_tp + skeleton_fn else 0.0
    sk_f1 = 2 * sk_precision * sk_recall / (sk_precision + sk_recall) if sk_precision + sk_recall else 0.0

    return {
        "arc_tp": tp,
        "arc_fp": fp,
        "arc_fn": fn,
        "arc_precision": precision,
        "arc_recall": recall,
        "arc_f1": f1,
        "shd": shd,
        "skeleton_precision": sk_precision,
        "skeleton_recall": sk_recall,
        "skeleton_f1": sk_f1,
        "true_edges": int(true.sum()),
        "estimated_edges": int(estimated.sum()),
    }

