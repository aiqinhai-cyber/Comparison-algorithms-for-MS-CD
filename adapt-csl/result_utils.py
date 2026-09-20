"""Atomic persistence, provenance metadata, and repeated-run aggregation."""

from __future__ import annotations

import hashlib
import json
import os
import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import config


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)}")


def save_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(
            payload, ensure_ascii=False, indent=2, default=json_default, allow_nan=False
        ),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def save_csv(path: str | Path, frame: pd.DataFrame) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    os.replace(temporary, path)


def save_edges(path: str | Path, edges: set[tuple[str, str]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for parent, child in sorted(edges):
            handle.write(f"{parent} -> {child}\n")
    os.replace(temporary, path)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def installed_versions() -> dict[str, str]:
    result = {}
    for package in ("numpy", "pandas", "scipy", "networkx", "pgmpy"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "not-installed"
    return result


def build_experiment_metadata(bif_paths: dict[str, Path]) -> dict[str, Any]:
    source_files = [
        "config.py",
        "data_utils.py",
        "scores.py",
        "graph_utils.py",
        "adapt_csl.py",
        "metrics.py",
        "result_utils.py",
        "run_experiments.py",
    ]
    return {
        "implementation_name": config.IMPLEMENTATION_NAME,
        "implementation_version": config.IMPLEMENTATION_VERSION,
        "paper_title": config.PAPER_TITLE,
        "paper_doi": config.PAPER_DOI,
        "networks": list(bif_paths),
        "sample_sizes": config.SAMPLE_SIZES,
        "seeds": config.SEEDS,
        "number_of_runs": config.NUM_RUNS,
        "num_workers": config.NUM_WORKERS,
        "alpha": config.ALPHA,
        "max_conditioning_set": config.MAX_CONDITIONING_SET,
        "bdeu_equivalent_sample_size": config.BDEU_EQUIVALENT_SAMPLE_SIZE,
        "score_tolerance": config.SCORE_TOLERANCE,
        "randomly_complete_undirected": (config.RANDOMLY_COMPLETE_UNDIRECTED),
        "overwrite_completed": config.OVERWRITE_COMPLETED,
        "std_ddof": config.STD_DDOF,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "logical_cpu_count": os.cpu_count(),
        "packages": installed_versions(),
        "bif_sha256": {name: file_sha256(path) for name, path in bif_paths.items()},
        "source_sha256": {
            name: file_sha256(config.PROJECT_DIR / name) for name in source_files
        },
    }


def configuration_id(metadata: dict[str, Any]) -> str:
    payload = json.dumps(metadata, sort_keys=True, default=json_default).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _formatted(mean: float, std: float, percentage: bool) -> str:
    scale = 100.0 if percentage else 1.0
    return f"{round(mean * scale):d} ± {std * scale:.{config.STD_DECIMALS}f}"


def aggregate_results(
    records: list[dict[str, Any]], output_directory: str | Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Persist all completed runs and every complete five-run summary."""
    output_directory = Path(output_directory)
    all_runs = pd.DataFrame(records)
    if not all_runs.empty:
        all_runs = all_runs.sort_values(
            ["bif_name", "sample_size", "seed"]
        ).reset_index(drop=True)
    save_csv(output_directory / "all_runs.csv", all_runs)

    metrics = ["precision", "recall", "f1", "shd", "missing", "extra", "reversed"]
    raw_rows: list[dict[str, Any]] = []
    formatted_rows: list[dict[str, Any]] = []
    successful = (
        all_runs.loc[all_runs["status"] == "ok"].copy()
        if not all_runs.empty and "status" in all_runs
        else pd.DataFrame()
    )
    if not successful.empty:
        for (name, size), group in successful.groupby(
            ["bif_name", "sample_size"], sort=True
        ):
            group = group.drop_duplicates(subset=["seed"], keep="last")
            means = group[metrics].mean()
            stds = group[metrics].std(ddof=config.STD_DDOF).fillna(0.0)
            seeds = {int(seed) for seed in group["seed"]}
            complete = len(group) == config.NUM_RUNS and seeds == set(config.SEEDS)
            row: dict[str, Any] = {
                "BIF": name,
                "Sample_Size": int(size),
                "run_count": len(group),
                "complete": complete,
                "seeds": ",".join(str(seed) for seed in sorted(seeds)),
            }
            for metric in metrics:
                row[f"{metric}_mean"] = float(means[metric])
                row[f"{metric}_std"] = float(stds[metric])
            raw_rows.append(row)

            setting_dir = output_directory / name / f"size_{int(size)}"
            save_csv(setting_dir / f"runs_{name}_{int(size)}.csv", group)
            if complete:
                formatted_rows.append(
                    {
                        "BIF": name,
                        "Sample_Size": int(size),
                        "Precision": _formatted(
                            means["precision"], stds["precision"], True
                        ),
                        "Recall": _formatted(means["recall"], stds["recall"], True),
                        "F1": _formatted(means["f1"], stds["f1"], True),
                        "SHD": _formatted(means["shd"], stds["shd"], False),
                    }
                )

    raw = pd.DataFrame(raw_rows)
    formatted = pd.DataFrame(formatted_rows)
    save_csv(output_directory / "global_summary_raw.csv", raw)
    save_csv(output_directory / "formatted_mean_std_summary.csv", formatted)
    if not formatted.empty:
        for name, group in formatted.groupby("BIF", sort=True):
            save_csv(output_directory / name / f"summary_{name}_all_sizes.csv", group)
    return raw, formatted
