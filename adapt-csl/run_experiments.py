"""Ten-process experiment runner; this is the project's executable entry."""

from __future__ import annotations

import json
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import networkx as nx

import config
from adapt_csl import AdaptCSL
from data_utils import resolve_bif_file, sample_bif
from metrics import directed_metrics
from result_utils import (
    aggregate_results,
    build_experiment_metadata,
    configuration_id,
    save_edges,
    save_json,
)


def validate_configuration() -> dict[str, Path]:
    if len(config.SEEDS) != config.NUM_RUNS or len(set(config.SEEDS)) != len(
        config.SEEDS
    ):
        raise ValueError("NUM_RUNS must equal the number of distinct SEEDS")
    if not 0.0 < config.ALPHA < 1.0:
        raise ValueError("ALPHA must be in (0, 1)")
    if config.MAX_CONDITIONING_SET is not None and config.MAX_CONDITIONING_SET < 0:
        raise ValueError("MAX_CONDITIONING_SET must be None or non-negative")
    if config.BDEU_EQUIVALENT_SAMPLE_SIZE <= 0:
        raise ValueError("BDEU_EQUIVALENT_SAMPLE_SIZE must be positive")
    if not 1 <= config.NUM_WORKERS <= (os.cpu_count() or 1):
        raise ValueError("NUM_WORKERS exceeds the available logical CPU count")
    return {Path(name).stem: resolve_bif_file(name) for name in config.BIF_FILES}


def run_single_experiment(task: dict[str, Any]) -> dict[str, Any]:
    name = task["name"]
    size = int(task["sample_size"])
    seed = int(task["seed"])
    output_directory = Path(task["output_directory"])
    run_directory = output_directory / name / f"size_{size}" / f"seed_{seed}"
    result_path = run_directory / "result.json"
    if result_path.is_file() and not config.OVERWRITE_COMPLETED:
        try:
            cached = json.loads(result_path.read_text(encoding="utf-8"))
            if (
                cached.get("status") == "ok"
                and cached.get("configuration_id") == task["configuration_id"]
                and cached.get("bif_sha256") == task["bif_sha256"]
            ):
                cached["loaded_from_cache"] = True
                return cached
        except (OSError, ValueError):
            pass

    result: dict[str, Any] = {
        "status": "failed",
        "bif_name": name,
        "sample_size": size,
        "seed": seed,
        "configuration_id": task["configuration_id"],
        "bif_sha256": task["bif_sha256"],
        "loaded_from_cache": False,
    }
    started = time.perf_counter()
    try:
        sampled = sample_bif(Path(task["bif_path"]), size, seed)
        learner = AdaptCSL()
        learned = learner.fit(
            sampled["data"],
            sampled["variables"],
            sampled["cardinalities"],
            seed,
        )
        truth: nx.DiGraph = sampled["true_graph"]
        result.update(
            {
                "status": "ok",
                **directed_metrics(learned, truth),
                "n_variables": len(sampled["variables"]),
                "cardinalities": sampled["cardinalities"].tolist(),
                "sampling_warnings": sampled["sampling_warnings"],
                "diagnostics": learner.diagnostics,
            }
        )
        save_edges(run_directory / "predicted_edges.txt", set(learned.edges()))
        save_edges(run_directory / "true_edges.txt", set(truth.edges()))
    except Exception as error:  # noqa: BLE001 - persist every experiment failure
        result["error"] = f"{type(error).__name__}: {error}"
        result["traceback"] = traceback.format_exc()
    result["runtime_seconds"] = time.perf_counter() - started
    # Every single experiment is saved immediately in its worker process.
    save_json(result_path, result)
    return result


def main() -> None:
    bif_paths = validate_configuration()
    metadata = build_experiment_metadata(bif_paths)
    experiment_id = configuration_id(metadata)
    output_directory = config.OUTPUT_ROOT / experiment_id
    output_directory.mkdir(parents=True, exist_ok=True)
    save_json(
        output_directory / "config.json",
        {
            **metadata,
            "configuration_id": experiment_id,
            "last_started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
    )
    tasks = [
        {
            "name": name,
            "bif_path": str(path),
            "sample_size": size,
            "seed": seed,
            "output_directory": str(output_directory),
            "configuration_id": experiment_id,
            "bif_sha256": metadata["bif_sha256"][name],
        }
        for name, path in bif_paths.items()
        for size in config.SAMPLE_SIZES
        for seed in config.SEEDS
    ]
    print(f"Adapt-CSL | {len(tasks)} tasks | {config.NUM_WORKERS} CPU workers")
    print(f"Output: {output_directory}")

    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=config.NUM_WORKERS) as executor:
        future_map = {
            executor.submit(run_single_experiment, task): task for task in tasks
        }
        for index, future in enumerate(as_completed(future_map), start=1):
            task = future_map[future]
            try:
                result = future.result()
            except Exception as error:  # noqa: BLE001 - isolate worker failures
                result = {
                    "status": "failed",
                    "bif_name": task["name"],
                    "sample_size": task["sample_size"],
                    "seed": task["seed"],
                    "error": f"WorkerError: {error}",
                }
            records.append(result)
            # Update aggregate files after every completed run. Once all five
            # seeds exist, the corresponding mean ± standard deviation row is
            # written immediately.
            aggregate_results(records, output_directory)
            if config.SHOW_PROGRESS:
                if result["status"] == "ok":
                    print(
                        f"[{index}/{len(tasks)}] {result['bif_name']}, "
                        f"n={result['sample_size']}, seed={result['seed']} | "
                        f"P={result['precision']:.4f} R={result['recall']:.4f} "
                        f"F1={result['f1']:.4f} SHD={result['shd']}"
                        + (" [cached]" if result.get("loaded_from_cache") else ""),
                        flush=True,
                    )
                else:
                    print(
                        f"[{index}/{len(tasks)}] FAILED: {result.get('error')}",
                        flush=True,
                    )

    aggregate_results(records, output_directory)
    failures = sum(record.get("status") != "ok" for record in records)
    print(f"Finished: {len(records) - failures}/{len(records)} successful")
    print(f"Wall time: {time.perf_counter() - started:.2f} seconds")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
