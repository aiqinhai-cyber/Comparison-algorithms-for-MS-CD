"""Saving and five-run aggregation utilities."""

from __future__ import annotations

import json
import platform
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd


def save_adjacency(path: str | Path, matrix: np.ndarray, nodes: list[str]) -> None:
    """Save an adjacency matrix with node names as rows and columns."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(matrix, index=nodes, columns=nodes).to_csv(path)


def flatten_results(records: list[dict]) -> list[dict]:
    """Convert nested per-experiment JSON records into tabular rows."""
    rows: list[dict] = []
    for record in records:
        if record.get("status") != "ok":
            rows.append(record)
            continue
        for variant, values in record["variants"].items():
            row = {
                "status": "ok",
                "network": record["network"],
                "sample_size": record["sample_size"],
                "variables": record["variables"],
                "variant": variant,
                "graph_threshold": record["graph_threshold"],
                "device": record["device"],
                "dtype": record["dtype"],
                "seed": record["seed"],
                "preprocessing": record["preprocessing"]["method"],
                "model_data_warning": record["model_data_warning"],
            }
            row.update(values)
            rows.append(row)
    return rows


def aggregate_mean_and_std(
    all_runs: pd.DataFrame,
    number_of_runs: int,
    arc_metrics_as_percent: bool,
    mean_decimals: int,
    std_decimals: int,
) -> pd.DataFrame:
    """Aggregate directed Precision, Recall, F1, and SHD over repeated runs."""
    if all_runs.empty or "status" not in all_runs.columns:
        return pd.DataFrame()
    successful = all_runs.loc[all_runs["status"] == "ok"].copy()
    if successful.empty:
        return pd.DataFrame()

    metrics = [
        ("arc_precision", "precision", 100.0 if arc_metrics_as_percent else 1.0),
        ("arc_recall", "recall", 100.0 if arc_metrics_as_percent else 1.0),
        ("arc_f1", "f1", 100.0 if arc_metrics_as_percent else 1.0),
        ("shd", "shd", 1.0),
    ]
    output_rows: list[dict] = []
    group_columns = ["network", "sample_size", "variables", "variant"]
    for keys, group in successful.groupby(
        group_columns, sort=False, dropna=False
    ):
        row = dict(zip(group_columns, keys))
        row["run_count"] = int(len(group))
        row["expected_run_count"] = int(number_of_runs)
        row["seeds"] = ",".join(str(int(seed)) for seed in group["seed"])
        row["complete_five_runs"] = len(group) == number_of_runs
        row["arc_metric_unit"] = (
            "percent" if arc_metrics_as_percent else "proportion"
        )

        for source, output_name, scale in metrics:
            values = pd.to_numeric(group[source], errors="coerce").dropna() * scale
            if len(values):
                raw_mean = float(values.mean())
                rounded_mean = round(raw_mean, mean_decimals)
                mean_value: float | int = (
                    int(rounded_mean)
                    if mean_decimals == 0
                    else float(rounded_mean)
                )
            else:
                mean_value = np.nan

            raw_std = float(values.std(ddof=1)) if len(values) > 1 else np.nan
            std_value = (
                round(raw_std, std_decimals) if np.isfinite(raw_std) else np.nan
            )
            row[f"{output_name}_mean"] = mean_value
            row[f"{output_name}_std"] = std_value
            if np.isfinite(mean_value) and np.isfinite(std_value):
                row[f"{output_name}_mean_std"] = (
                    f"{mean_value:.{mean_decimals}f} ± "
                    f"{std_value:.{std_decimals}f}"
                )
            else:
                row[f"{output_name}_mean_std"] = ""
        output_rows.append(row)
    return pd.DataFrame(output_rows)


def save_per_setting_summaries(
    all_runs: pd.DataFrame,
    args: SimpleNamespace,
) -> None:
    """Save five raw runs and one mean ± std file per network/sample size."""
    for network in args.networks:
        for sample_size in args.sample_sizes:
            directory = (
                args.output_dir / args.variant / network / f"n{sample_size}"
            )
            directory.mkdir(parents=True, exist_ok=True)
            if all_runs.empty:
                setting_runs = all_runs.copy()
            else:
                setting_runs = all_runs.loc[
                    (all_runs["network"] == network)
                    & (all_runs["sample_size"] == sample_size)
                ].copy()
            setting_runs.to_csv(directory / "five_runs_results.csv", index=False)

            summary = aggregate_mean_and_std(
                setting_runs,
                number_of_runs=args.number_of_runs,
                arc_metrics_as_percent=args.report_arc_metrics_as_percent,
                mean_decimals=args.summary_mean_decimals,
                std_decimals=args.summary_std_decimals,
            )
            summary.to_csv(
                directory / "mean_std_metrics.csv",
                index=False,
                float_format=f"%.{args.summary_std_decimals}f",
            )


def save_experiment_metadata(args: SimpleNamespace, directory: Path) -> None:
    """Record resolved settings and software/hardware information."""
    import networkx as nx
    import torch

    configuration = {
        "bif_dir": str(args.bif_dir),
        "bif_fallback_dirs": [str(path) for path in args.bif_fallback_dirs],
        "sample_dir": str(args.sample_dir),
        "output_dir": str(args.output_dir),
        "networks": args.networks,
        "sample_sizes": args.sample_sizes,
        "number_of_runs": args.number_of_runs,
        "random_seeds": args.seeds,
        "data_mode": args.data_mode,
        "save_resampled_data": args.save_resampled_data,
        "variant": args.variant,
        "nv_initialization": args.nv_initialization,
        "ev_lambda_1": args.ev_lambda_1,
        "ev_lambda_2": args.ev_lambda_2,
        "ev_num_iter": args.ev_num_iter,
        "ev_learning_rate": args.ev_learning_rate,
        "nv_lambda_1": args.nv_lambda_1,
        "nv_lambda_2": args.nv_lambda_2,
        "nv_num_iter": args.nv_num_iter,
        "nv_learning_rate": args.nv_learning_rate,
        "graph_threshold": args.graph_threshold,
        "preprocessing": args.preprocessing,
        "device_requested": args.device,
        "device_resolved": args.resolved_device,
        "dtype": args.dtype,
        "deterministic": args.deterministic,
        "checkpoint_interval": args.checkpoint_interval,
        "overwrite_completed": args.overwrite,
        "report_arc_metrics_as_percent": args.report_arc_metrics_as_percent,
        "summary_mean_decimals": args.summary_mean_decimals,
        "summary_std_decimals": args.summary_std_decimals,
        "save_per_setting_summaries": args.save_per_setting_summaries,
    }
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "experiment_config.json").write_text(
        json.dumps(configuration, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "networkx": nx.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    (directory / "environment.json").write_text(
        json.dumps(environment, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
