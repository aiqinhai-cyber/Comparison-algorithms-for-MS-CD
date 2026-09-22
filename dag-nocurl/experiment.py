"""One DAG-NoCurl training task and its threshold evaluations."""

from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import config
from data_utils import load_and_sample
from metrics import compute_metrics, graph_at_threshold
from nocurl import NoCurl2
from result_utils import active_thresholds
from storage import save_csv, save_json, save_npz, utc_now


def _cache_is_reusable(
    result_path: Path,
    run_directory: Path,
    configuration_id: str,
    bif_hash: str,
) -> dict[str, Any] | None:
    if not config.RESUME_COMPLETED or not result_path.is_file():
        return None
    try:
        previous = json.loads(result_path.read_text(encoding="utf-8"))
        cached_thresholds = [
            row["threshold"] for row in previous.get("threshold_metrics", [])
        ]
        reusable = (
            previous.get("status") in config.EVALUABLE_STATUSES
            and previous.get("configuration_id") == configuration_id
            and previous.get("bif_sha256") == bif_hash
            and cached_thresholds == active_thresholds()
            and (
                not config.SAVE_DATA
                or (run_directory / "data.npz").is_file()
            )
            and (
                not config.SAVE_WEIGHTS
                or (run_directory / "weights.npz").is_file()
            )
        )
        if reusable:
            previous["loaded_from_cache"] = True
            return previous
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return None


def run_single_experiment(task: dict[str, Any]) -> dict[str, Any]:
    network = task["network"]
    sample_size = int(task["sample_size"])
    seed = int(task["seed"])
    run_root = Path(task["run_root"])
    run_directory = run_root / network / f"n_{sample_size}" / f"seed_{seed}"
    result_path = run_directory / "result.json"
    cached = _cache_is_reusable(
        result_path,
        run_directory,
        task["configuration_id"],
        task["bif_sha256"],
    )
    if cached is not None:
        return cached

    run_directory.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "network": network,
        "n_samples": sample_size,
        "seed": seed,
        "method": "DAG-NoCurl-NoCurl-2",
        "configuration_id": task["configuration_id"],
        "bif_sha256": task["bif_sha256"],
        "status": "failed",
        "loaded_from_cache": False,
        "started_at": utc_now(),
    }
    total_started = time.perf_counter()
    learner: NoCurl2 | None = None
    try:
        with threadpool_limits(limits=config.THREADS_PER_WORKER):
            data_started = time.perf_counter()
            data = load_and_sample(task["bif_path"], sample_size, seed)
            result["data_seconds"] = time.perf_counter() - data_started
            x = data["X"]
            node_names = data["node_names"]
            true_graph = data["true_graph"]
            result["n_variables"] = len(node_names)
            result["constant_columns"] = data["constant_columns"]
            result["sampling_warnings"] = data["sampling_warnings"]

            save_json(
                run_directory / "encoding.json",
                {
                    "node_names": [str(value) for value in node_names],
                    "states": data["states"],
                    "cardinalities": data["cardinalities"],
                    "means_before_centering": data["means_before_centering"],
                    "scale_by_declared_cardinality": (
                        config.SCALE_BY_DECLARED_CARDINALITY
                    ),
                    "center_data": config.CENTER_DATA,
                },
            )
            save_json(
                run_directory / "sampling_warnings.json",
                data["sampling_warnings"],
            )
            if config.SAVE_DATA:
                save_npz(
                    run_directory / "data.npz",
                    X=x,
                    codes=data["codes"],
                    node_names=np.asarray(node_names, dtype=str),
                    true_adjacency=nx.to_numpy_array(
                        true_graph,
                        nodelist=node_names,
                        dtype=np.int8,
                    ),
                )

            learner = NoCurl2()
            fit_started = time.perf_counter()
            raw_weights, loss = learner.fit_raw(x)
            result["fit_seconds"] = time.perf_counter() - fit_started
            result["optimizer_diagnostics"] = learner.diagnostics
            if config.SAVE_WEIGHTS:
                save_npz(
                    run_directory / "weights.npz",
                    raw_weights=raw_weights,
                    initial_weights=learner.initial_weights,
                    potential=learner.potential,
                    node_names=np.asarray(node_names, dtype=str),
                )
            if not np.isfinite(raw_weights).all() or not np.isfinite(loss):
                raise FloatingPointError("NoCurl produced non-finite output")
            result["loss"] = loss

            threshold_metrics: list[dict[str, Any]] = []
            for threshold in active_thresholds():
                predicted = graph_at_threshold(
                    raw_weights,
                    node_names,
                    threshold,
                )
                metrics = compute_metrics(true_graph, predicted)
                threshold_metrics.append({"threshold": threshold, **metrics})
                save_json(
                    run_directory / f"edges_threshold_{threshold:.1f}.json",
                    [[str(parent), str(child)] for parent, child in predicted.edges()],
                )
            result["threshold_metrics"] = threshold_metrics
            save_csv(
                run_directory / "threshold_metrics.csv",
                pd.DataFrame(threshold_metrics),
            )

            warning_stages = [
                stage
                for stage, record in learner.diagnostics.items()
                if not record["success"]
            ]
            result["optimizer_warning_stages"] = warning_stages
            result["all_optimizers_successful"] = not warning_stages
            result["status"] = (
                "completed_with_optimizer_warning" if warning_stages else "ok"
            )
    except Exception as error:  # noqa: BLE001 - persist every task failure
        result["status"] = "failed"
        result["error"] = f"{type(error).__name__}: {error}"
        result["traceback"] = traceback.format_exc()
        if learner is not None:
            result["optimizer_diagnostics"] = learner.diagnostics
    result["total_seconds"] = time.perf_counter() - total_started
    result["finished_at"] = utc_now()
    save_json(result_path, result)
    return result
