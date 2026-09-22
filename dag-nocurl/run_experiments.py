"""Command-line entry point for the DAG-NoCurl benchmark."""

# ruff: noqa: I001

from __future__ import annotations

import os
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

# Import config before numerical libraries so BLAS thread limits are applied.
import config

import numpy as np

from data_utils import resolve_bif_files
from experiment import run_single_experiment
from result_utils import (
    active_thresholds,
    print_final_table,
    write_summaries,
)
from storage import (
    configuration_id,
    environment_metadata,
    file_sha256,
    save_json,
    utc_now,
)


def validate_configuration() -> dict[str, Path]:
    for values, name in (
        (config.NETWORKS, "NETWORKS"),
        (config.SAMPLE_SIZES, "SAMPLE_SIZES"),
        (config.SEEDS, "SEEDS"),
        (active_thresholds(), "active thresholds"),
    ):
        if not values:
            raise ValueError(f"{name} cannot be empty")
        if len(set(values)) != len(values):
            raise ValueError(f"Duplicate values in {name}")
    if not 0 <= config.STD_DDOF < len(config.SEEDS):
        raise ValueError("STD_DDOF must satisfy 0 <= ddof < number of seeds")
    if config.N_JOBS < 1 or config.THREADS_PER_WORKER < 1:
        raise ValueError("Process and thread counts must be positive")
    if any(
        not np.isfinite(threshold) or not 0 <= threshold <= 1
        for threshold in active_thresholds()
    ):
        raise ValueError("Every graph threshold must be finite and in [0, 1]")
    if config.LAMBDA1 <= 0 or config.LAMBDA2 <= 0 or config.H_TOL <= 0:
        raise ValueError("LAMBDA1, LAMBDA2, and H_TOL must be positive")
    return resolve_bif_files()


def source_hashes() -> dict[str, str]:
    filenames = [
        "config.py",
        "storage.py",
        "data_utils.py",
        "nocurl.py",
        "metrics.py",
        "result_utils.py",
        "experiment.py",
        "run_experiments.py",
    ]
    return {
        filename: file_sha256(config.PROJECT_DIR / filename)
        for filename in filenames
    }


def experiment_configuration(bif_paths: dict[str, Path]) -> dict[str, Any]:
    return {
        "implementation_version": config.IMPLEMENTATION_VERSION,
        "paper_title": config.PAPER_TITLE,
        "paper_url": config.PAPER_URL,
        "upstream_repository": config.UPSTREAM_REPOSITORY,
        "method": "DAG-NoCurl-NoCurl-2",
        "lambda1": config.LAMBDA1,
        "lambda2": config.LAMBDA2,
        "h_tol": config.H_TOL,
        "internal_threshold": config.INTERNAL_THRESHOLD,
        "graph_thresholds": active_thresholds(),
        "fixed_threshold_mode": config.FIXED_THRESHOLD_MODE,
        "threshold_selection": (
            "fixed predeclared threshold"
            if config.FIXED_THRESHOLD_MODE
            else (
                "oracle: highest five-seed mean ground-truth F1; "
                "ties: lower mean SHD, then lower threshold"
            )
        ),
        "optimizer": "L-BFGS-B",
        "optimizer_options": {"ftol": config.H_TOL},
        "additional_optimizer_retries": False,
        "scale_by_declared_cardinality": (
            config.SCALE_BY_DECLARED_CARDINALITY
        ),
        "center_data": config.CENTER_DATA,
        "encoding": "BIF-declared state order",
        "std_ddof": config.STD_DDOF,
        "finite_optimizer_warning_outputs_included": True,
        "networks": config.NETWORKS,
        "sample_sizes": config.SAMPLE_SIZES,
        "seeds": config.SEEDS,
        "threads_per_worker": config.THREADS_PER_WORKER,
        "save_data": config.SAVE_DATA,
        "save_weights": config.SAVE_WEIGHTS,
        **environment_metadata(),
        "bif_sha256": {
            network: file_sha256(path)
            for network, path in bif_paths.items()
        },
        "source_sha256": source_hashes(),
    }


def print_run(result: dict[str, Any], completed: int, total: int) -> None:
    prefix = (
        f"[{completed}/{total}] {result['network']}, "
        f"n={result['n_samples']}, seed={result['seed']}"
    )
    if result.get("status") not in config.EVALUABLE_STATUSES:
        print(f"{prefix}: FAILED\n  {result.get('error', '')}", flush=True)
        return
    warning_marker = (
        " [optimizer warning]"
        if result["status"] == "completed_with_optimizer_warning"
        else ""
    )
    cache_marker = " [cached]" if result.get("loaded_from_cache") else ""
    print(
        f"{prefix}: evaluated {len(active_thresholds())} thresholds, "
        f"fit={result['fit_seconds']:.2f}s{warning_marker}{cache_marker}",
        flush=True,
    )
    for stage in result.get("optimizer_warning_stages", []):
        record = result["optimizer_diagnostics"][stage]
        print(
            f"  {stage}: {record['message']} (nit={record['nit']})",
            flush=True,
        )


def main() -> None:
    bif_paths = validate_configuration()
    metadata = experiment_configuration(bif_paths)
    config_id = configuration_id(metadata)
    run_root = config.OUTPUT_DIR / config_id
    run_root.mkdir(parents=True, exist_ok=True)
    save_json(
        run_root / "config.json",
        {
            **metadata,
            "configuration_id": config_id,
            "n_jobs": config.N_JOBS,
            "saved_at": utc_now(),
        },
    )

    tasks = [
        {
            "network": network,
            "sample_size": sample_size,
            "seed": seed,
            "bif_path": str(bif_paths[network]),
            "run_root": str(run_root),
            "configuration_id": config_id,
            "bif_sha256": metadata["bif_sha256"][network],
        }
        for network in config.NETWORKS
        for sample_size in config.SAMPLE_SIZES
        for seed in config.SEEDS
    ]
    workers = min(config.N_JOBS, os.cpu_count() or 1, len(tasks))
    print("DAG-NoCurl / NoCurl-2 discrete-BN benchmark")
    print(f"Training tasks: {len(tasks)}")
    print(f"Thresholds: {active_thresholds()}")
    print(f"Processes: {workers}")
    print(f"Threads per process: {config.THREADS_PER_WORKER}")
    print(f"Output: {run_root}")
    if config.FIXED_THRESHOLD_MODE:
        print("Threshold protocol: fixed and predeclared")
    else:
        print("WARNING: oracle ground-truth threshold selection is enabled")

    results: list[dict[str, Any]] = []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_task = {
            executor.submit(run_single_experiment, task): task for task in tasks
        }
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                result = future.result()
            except Exception as error:  # noqa: BLE001 - isolate worker failure
                result = {
                    "network": task["network"],
                    "n_samples": task["sample_size"],
                    "seed": task["seed"],
                    "status": "failed",
                    "configuration_id": config_id,
                    "bif_sha256": task["bif_sha256"],
                    "loaded_from_cache": False,
                    "error": f"Worker failure: {type(error).__name__}: {error}",
                }
                save_json(
                    run_root
                    / task["network"]
                    / f"n_{task['sample_size']}"
                    / f"seed_{task['seed']}"
                    / "result.json",
                    result,
                )
            results.append(result)
            if config.VERBOSE:
                print_run(result, len(results), len(tasks))
            write_summaries(results, run_root)

    final_frame = write_summaries(results, run_root)
    counts = Counter(result["status"] for result in results)
    print_final_table(final_frame)
    print("\nExecution summary")
    print(f"All optimizers successful: {counts['ok']}")
    print(
        "Evaluable with optimizer warnings: "
        f"{counts['completed_with_optimizer_warning']}"
    )
    print(f"Failed: {counts['failed']}")
    print(f"Wall time: {time.perf_counter() - started:.2f}s")
    print(f"Final results: {run_root / 'final_best_results.csv'}")
    print(f"All thresholds: {run_root / 'threshold_summary.csv'}")


if __name__ == "__main__":
    main()
