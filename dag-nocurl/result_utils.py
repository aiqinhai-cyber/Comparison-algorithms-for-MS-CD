"""Aggregation, threshold selection, and benchmark summary files."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import config
from storage import save_csv, save_json


def active_thresholds() -> list[float]:
    if config.FIXED_THRESHOLD_MODE:
        return [float(config.FIXED_GRAPH_THRESHOLD)]
    return [float(value) for value in config.GRAPH_THRESHOLDS]


def mean_std_text(
    mean: float | None,
    standard_deviation: float | None,
    scale: float = 1.0,
    decimals: int = 2,
) -> str:
    if mean is None or standard_deviation is None:
        return ""
    return (
        f"{mean * scale:.{decimals}f}"
        f" ± {standard_deviation * scale:.{decimals}f}"
    )


def _setting_metadata(
    network: str,
    sample_size: int,
    group: Sequence[dict[str, Any]],
    evaluable: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    complete = (
        len(evaluable) == len(config.SEEDS)
        and {result["seed"] for result in evaluable} == set(config.SEEDS)
    )
    return {
        "network": network,
        "n_samples": sample_size,
        "expected_runs": len(config.SEEDS),
        "observed_runs": len(group),
        "evaluable_runs": len(evaluable),
        "optimizer_warning_runs": sum(
            result.get("status") == "completed_with_optimizer_warning"
            for result in group
        ),
        "failed_runs": sum(
            result.get("status") == "failed" for result in group
        ),
        "complete": complete,
        "std_ddof": config.STD_DDOF,
    }


def write_summaries(
    results: list[dict[str, Any]],
    run_root: str | Path,
) -> pd.DataFrame:
    """Write raw, all-threshold, and final threshold-selected summaries."""
    run_root = Path(run_root)
    ordered = sorted(
        results,
        key=lambda result: (
            config.NETWORKS.index(result["network"]),
            result["n_samples"],
            result["seed"],
        ),
    )

    run_rows: list[dict[str, Any]] = []
    all_threshold_rows: list[dict[str, Any]] = []
    for result in ordered:
        run_rows.append(
            {
                key: value
                for key, value in result.items()
                if not isinstance(value, (dict, list))
            }
        )
        if result.get("status") in config.EVALUABLE_STATUSES:
            for metrics in result["threshold_metrics"]:
                all_threshold_rows.append(
                    {
                        "network": result["network"],
                        "n_samples": result["n_samples"],
                        "seed": result["seed"],
                        "status": result["status"],
                        "fit_seconds": result["fit_seconds"],
                        **metrics,
                    }
                )
    save_csv(run_root / "all_runs.csv", pd.DataFrame(run_rows))
    save_csv(
        run_root / "all_threshold_runs.csv",
        pd.DataFrame(all_threshold_rows),
    )

    thresholds = active_thresholds()
    threshold_summary_rows: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    for network in config.NETWORKS:
        for sample_size in config.SAMPLE_SIZES:
            group = [
                result
                for result in ordered
                if result["network"] == network
                and result["n_samples"] == sample_size
            ]
            evaluable = [
                result
                for result in group
                if result.get("status") in config.EVALUABLE_STATUSES
            ]
            metadata = _setting_metadata(
                network,
                sample_size,
                group,
                evaluable,
            )
            current_rows: list[dict[str, Any]] = []
            for threshold_index, threshold in enumerate(thresholds):
                row: dict[str, Any] = {**metadata, "threshold": threshold}
                for metric in config.METRIC_COLUMNS:
                    mean: float | None = None
                    standard_deviation: float | None = None
                    if metadata["complete"]:
                        if metric == "fit_seconds":
                            values = [result["fit_seconds"] for result in evaluable]
                        else:
                            values = [
                                result["threshold_metrics"][threshold_index][metric]
                                for result in evaluable
                            ]
                        array = np.asarray(values, dtype=float)
                        mean = float(array.mean())
                        standard_deviation = float(
                            array.std(ddof=config.STD_DDOF)
                        )
                    row[f"{metric}_mean"] = mean
                    row[f"{metric}_std"] = standard_deviation
                current_rows.append(row)
                threshold_summary_rows.append(row)

            selection = (
                "fixed_predeclared_threshold"
                if config.FIXED_THRESHOLD_MODE
                else "oracle_ground_truth_mean_f1"
            )
            final: dict[str, Any] = {
                **metadata,
                "selection": selection,
                "best_threshold": None,
            }
            for metric in config.METRIC_COLUMNS:
                final[f"{metric}_mean"] = None
                final[f"{metric}_std"] = None

            if metadata["complete"]:
                if config.FIXED_THRESHOLD_MODE:
                    selected = current_rows[0]
                else:
                    selected = min(
                        current_rows,
                        key=lambda row: (
                            -row["f1_mean"],
                            row["shd_mean"],
                            row["threshold"],
                        ),
                    )
                final["best_threshold"] = selected["threshold"]
                for metric in config.METRIC_COLUMNS:
                    final[f"{metric}_mean"] = selected[f"{metric}_mean"]
                    final[f"{metric}_std"] = selected[f"{metric}_std"]

                selected_index = thresholds.index(selected["threshold"])
                selected_seed_rows = [
                    {
                        "network": network,
                        "n_samples": sample_size,
                        "seed": result["seed"],
                        "status": result["status"],
                        "fit_seconds": result["fit_seconds"],
                        **result["threshold_metrics"][selected_index],
                    }
                    for result in sorted(evaluable, key=lambda item: item["seed"])
                ]
                setting_directory = run_root / network / f"n_{sample_size}"
                save_csv(
                    setting_directory / "selected_threshold_runs.csv",
                    pd.DataFrame(selected_seed_rows),
                )

            for metric in ("precision", "recall", "f1"):
                mean = final[f"{metric}_mean"]
                standard_deviation = final[f"{metric}_std"]
                final[f"{metric}_percent_mean"] = (
                    mean * 100.0 if mean is not None else None
                )
                final[f"{metric}_percent_std"] = (
                    standard_deviation * 100.0
                    if standard_deviation is not None
                    else None
                )
                final[f"{metric}_percent_mean_std"] = mean_std_text(
                    mean,
                    standard_deviation,
                    scale=100.0,
                )
            final["shd_mean_std"] = mean_std_text(
                final["shd_mean"],
                final["shd_std"],
            )
            final["fit_seconds_mean_std"] = mean_std_text(
                final["fit_seconds_mean"],
                final["fit_seconds_std"],
            )
            final_rows.append(final)

    save_csv(
        run_root / "threshold_summary.csv",
        pd.DataFrame(threshold_summary_rows),
    )
    final_frame = pd.DataFrame(final_rows)
    save_csv(run_root / "final_best_results.csv", final_frame)
    save_json(run_root / "final_best_results.json", final_rows)
    if not final_frame.empty:
        for network, frame in final_frame.groupby("network", sort=False):
            save_csv(
                run_root / network / f"summary_{network}_all_sizes.csv",
                frame,
            )
    return final_frame


def print_final_table(final_frame: pd.DataFrame) -> None:
    columns = [
        "network",
        "n_samples",
        "best_threshold",
        "precision_percent_mean_std",
        "recall_percent_mean_std",
        "f1_percent_mean_std",
        "shd_mean_std",
        "optimizer_warning_runs",
        "complete",
    ]
    if final_frame.empty:
        print("No final results are available.")
        return
    display = final_frame[columns].copy()
    display.columns = [
        "Network",
        "Samples",
        "Threshold",
        "Precision (%)",
        "Recall (%)",
        "F1 (%)",
        "SHD",
        "Optimizer warnings",
        "Complete",
    ]
    print("\nFINAL RESULTS")
    print(
        "Threshold selection: fixed"
        if config.FIXED_THRESHOLD_MODE
        else "Threshold selection: oracle / ground-truth-based"
    )
    print(display.to_string(index=False))
