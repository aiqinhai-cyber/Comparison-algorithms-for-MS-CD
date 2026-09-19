"""Run GOLEM on ordinally encoded samples from categorical BIF networks.

GOLEM assumes a continuous linear structural equation model with Gaussian
noise. The data used here are categorical. Mapping BIF states to ordinal values
is therefore a deliberately model-misspecified baseline comparison; it does not
mean that GOLEM natively supports discrete variables.

Run from this directory with::

    python run_experiments.py
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

import config
from data_utils import (
    find_sample_file,
    find_seeded_sample_file,
    generate_samples_from_bif,
    load_bif_metadata,
    load_numeric_samples,
    preprocess_data,
    resolve_bif_file,
)
from golem import fit_golem, is_dag, postprocess, resolve_device
from metrics import directed_metrics
from result_utils import (
    aggregate_mean_and_std,
    flatten_results,
    save_adjacency,
    save_experiment_metadata,
    save_per_setting_summaries,
)


def build_configuration() -> SimpleNamespace:
    """Validate ``config.py`` and produce the resolved runtime configuration."""
    if config.NUMBER_OF_RUNS != 5:
        raise ValueError("This protocol requires NUMBER_OF_RUNS = 5")
    if (
        len(config.RANDOM_SEEDS) != config.NUMBER_OF_RUNS
        or len(set(config.RANDOM_SEEDS)) != config.NUMBER_OF_RUNS
    ):
        raise ValueError("RANDOM_SEEDS must contain five distinct seeds")
    if not config.NETWORKS or not config.SAMPLE_SIZES:
        raise ValueError("NETWORKS and SAMPLE_SIZES must not be empty")
    if any(size < 1 for size in config.SAMPLE_SIZES):
        raise ValueError("Every sample size must be positive")
    if config.VARIANT not in {"ev", "nv", "both"}:
        raise ValueError("VARIANT must be 'ev', 'nv', or 'both'")
    if config.NV_INITIALIZATION not in {"ev", "zero"}:
        raise ValueError("NV_INITIALIZATION must be 'ev' or 'zero'")
    if config.DATA_MODE not in {"resample", "seeded_files", "shared_file"}:
        raise ValueError(
            "DATA_MODE must be 'resample', 'seeded_files', or 'shared_file'"
        )
    if config.PREPROCESSING not in {"center", "zscore"}:
        raise ValueError("PREPROCESSING must be 'center' or 'zscore'")
    if config.DEVICE not in {"auto", "cpu", "cuda"}:
        raise ValueError("DEVICE must be 'auto', 'cpu', or 'cuda'")
    if config.DTYPE not in {"float32", "float64"}:
        raise ValueError("DTYPE must be 'float32' or 'float64'")
    if config.EV_NUM_ITER < 1 or config.NV_NUM_ITER < 1:
        raise ValueError("Iteration counts must be positive")
    if config.EV_LEARNING_RATE <= 0 or config.NV_LEARNING_RATE <= 0:
        raise ValueError("Learning rates must be positive")
    if config.GRAPH_THRESHOLD < 0:
        raise ValueError("GRAPH_THRESHOLD must be nonnegative")
    if config.CHECKPOINT_INTERVAL < 0:
        raise ValueError("CHECKPOINT_INTERVAL must be nonnegative")

    # Fail before optimization if any requested BIF is absent.
    for network in config.NETWORKS:
        resolve_bif_file(config.BIF_DIR, network, config.BIF_FALLBACK_DIRS)

    return SimpleNamespace(
        bif_dir=config.BIF_DIR,
        bif_fallback_dirs=list(config.BIF_FALLBACK_DIRS),
        sample_dir=config.SAMPLE_DIR,
        output_dir=config.OUTPUT_DIR,
        networks=list(config.NETWORKS),
        sample_sizes=list(config.SAMPLE_SIZES),
        number_of_runs=config.NUMBER_OF_RUNS,
        seeds=list(config.RANDOM_SEEDS),
        data_mode=config.DATA_MODE,
        save_resampled_data=config.SAVE_RESAMPLED_DATA,
        variant=config.VARIANT,
        nv_initialization=config.NV_INITIALIZATION,
        ev_lambda_1=config.EV_LAMBDA_1,
        ev_lambda_2=config.EV_LAMBDA_2,
        ev_num_iter=config.EV_NUM_ITER,
        ev_learning_rate=config.EV_LEARNING_RATE,
        nv_lambda_1=config.NV_LAMBDA_1,
        nv_lambda_2=config.NV_LAMBDA_2,
        nv_num_iter=config.NV_NUM_ITER,
        nv_learning_rate=config.NV_LEARNING_RATE,
        graph_threshold=config.GRAPH_THRESHOLD,
        preprocessing=config.PREPROCESSING,
        device=config.DEVICE,
        dtype=config.DTYPE,
        deterministic=config.DETERMINISTIC,
        checkpoint_interval=config.CHECKPOINT_INTERVAL,
        overwrite=config.OVERWRITE_COMPLETED,
        show_progress=config.SHOW_PROGRESS,
        report_arc_metrics_as_percent=config.REPORT_ARC_METRICS_AS_PERCENT,
        summary_mean_decimals=config.SUMMARY_MEAN_DECIMALS,
        summary_std_decimals=config.SUMMARY_STD_DECIMALS,
        save_per_setting_summaries=config.SAVE_PER_SETTING_SUMMARIES,
        resolved_device=resolve_device(config.DEVICE),
        resolved_dtype=(
            torch.float64 if config.DTYPE == "float64" else torch.float32
        ),
    )


def run_and_save_variant(
    variant: str,
    seed: int,
    data: np.ndarray,
    true_adjacency: np.ndarray,
    nodes: list[str],
    output_directory: Path,
    args: SimpleNamespace,
    initial_adjacency: np.ndarray | None,
) -> tuple[np.ndarray, dict]:
    """Train, post-process, evaluate, and save one GOLEM variant."""
    equal_variances = variant in {"ev", "ev_init"}
    lambda_1 = args.ev_lambda_1 if equal_variances else args.nv_lambda_1
    lambda_2 = args.ev_lambda_2 if equal_variances else args.nv_lambda_2
    num_iter = args.ev_num_iter if equal_variances else args.nv_num_iter
    learning_rate = (
        args.ev_learning_rate if equal_variances else args.nv_learning_rate
    )

    variant_directory = output_directory / variant
    variant_directory.mkdir(parents=True, exist_ok=True)
    fit = fit_golem(
        data=data,
        lambda_1=lambda_1,
        lambda_2=lambda_2,
        equal_variances=equal_variances,
        num_iter=num_iter,
        learning_rate=learning_rate,
        seed=seed,
        deterministic=args.deterministic,
        device=args.resolved_device,
        dtype=args.resolved_dtype,
        initial_adjacency=initial_adjacency,
        checkpoint_interval=args.checkpoint_interval,
        checkpoint_directory=variant_directory / "checkpoints",
        show_progress=args.show_progress,
    )

    processed, dag_enforcing_threshold = postprocess(
        fit.weighted_adjacency, args.graph_threshold
    )
    binary_adjacency = (processed != 0).astype(np.int8)
    if not is_dag(binary_adjacency):
        raise AssertionError("GOLEM post-processing did not produce a DAG")

    np.save(variant_directory / "weighted_adjacency.npy", fit.weighted_adjacency)
    np.save(
        variant_directory / "processed_weighted_adjacency.npy", processed
    )
    save_adjacency(
        variant_directory / "estimated_dag.csv", binary_adjacency, nodes
    )
    pd.DataFrame(fit.history).to_csv(
        variant_directory / "optimization_history.csv", index=False
    )

    record = {
        "variant": variant,
        "lambda_1": float(lambda_1),
        "lambda_2": float(lambda_2),
        "num_iter": int(num_iter),
        "learning_rate": float(learning_rate),
        "training_runtime_seconds": fit.runtime_seconds,
        "dag_enforcing_threshold": dag_enforcing_threshold,
        "final_objective": fit.history[-1]["objective"],
        "final_likelihood": fit.history[-1]["likelihood"],
        "final_l1": fit.history[-1]["l1"],
        "final_h": fit.history[-1]["h"],
        **directed_metrics(true_adjacency, binary_adjacency),
    }
    return fit.weighted_adjacency, record


def _resolve_sample_path(
    network: str,
    sample_size: int,
    seed: int,
    metadata,
    output_directory: Path,
    args: SimpleNamespace,
) -> Path:
    """Generate or locate the sample file selected by DATA_MODE."""
    if args.data_mode == "resample":
        sample_path = (
            args.output_dir
            / "generated_samples"
            / network
            / f"{network}_{sample_size}_seed{seed}.csv"
            if args.save_resampled_data
            else output_directory / "generated_samples.csv"
        )
        if args.overwrite or not sample_path.is_file():
            generate_samples_from_bif(metadata, sample_size, seed, sample_path)
        return sample_path

    if args.data_mode == "seeded_files":
        sample_path = find_seeded_sample_file(
            args.sample_dir, network, sample_size, seed
        )
        if sample_path is None:
            raise FileNotFoundError(
                f"No seed-specific sample for {network}, n={sample_size}, "
                f"seed={seed}"
            )
        return sample_path

    sample_path = find_sample_file(args.sample_dir, network, sample_size)
    if sample_path is None:
        raise FileNotFoundError(
            f"No shared sample file for {network}, n={sample_size}"
        )
    return sample_path


def run_one_experiment(
    network: str,
    sample_size: int,
    seed: int,
    args: SimpleNamespace,
) -> dict:
    """Run one network/sample-size/seed experiment."""
    output_directory = (
        args.output_dir
        / args.variant
        / network
        / f"n{sample_size}"
        / f"seed_{seed}"
    )
    result_path = output_directory / "result.json"
    if result_path.is_file() and not args.overwrite:
        return json.loads(result_path.read_text(encoding="utf-8"))

    bif_path = resolve_bif_file(
        args.bif_dir, network, args.bif_fallback_dirs
    )
    metadata = load_bif_metadata(bif_path)
    sample_path = _resolve_sample_path(
        network, sample_size, seed, metadata, output_directory, args
    )
    raw_data = load_numeric_samples(sample_path, metadata)
    if raw_data.shape[0] != sample_size:
        raise ValueError(
            f"{sample_path} contains {raw_data.shape[0]} rows; "
            f"expected {sample_size}"
        )
    data, preprocessing_information = preprocess_data(
        raw_data, args.preprocessing
    )

    output_directory.mkdir(parents=True, exist_ok=True)
    save_adjacency(
        output_directory / "true_adjacency.csv",
        metadata.true_adjacency,
        metadata.nodes,
    )

    variants: dict[str, dict] = {}
    ev_weighted: np.ndarray | None = None
    ev_initialization_runtime = 0.0
    if args.variant in {"ev", "both"}:
        ev_weighted, ev_record = run_and_save_variant(
            "ev",
            seed,
            data,
            metadata.true_adjacency,
            metadata.nodes,
            output_directory,
            args,
            initial_adjacency=None,
        )
        variants["ev"] = ev_record
        ev_initialization_runtime = float(ev_record["training_runtime_seconds"])

    if args.variant == "nv" and args.nv_initialization == "ev":
        ev_weighted, ev_initialization_record = run_and_save_variant(
            "ev_init",
            seed,
            data,
            metadata.true_adjacency,
            metadata.nodes,
            output_directory,
            args,
            initial_adjacency=None,
        )
        ev_initialization_runtime = float(
            ev_initialization_record["training_runtime_seconds"]
        )

    if args.variant in {"nv", "both"}:
        nv_initial = ev_weighted if args.nv_initialization == "ev" else None
        _, nv_record = run_and_save_variant(
            "nv",
            seed,
            data,
            metadata.true_adjacency,
            metadata.nodes,
            output_directory,
            args,
            initial_adjacency=nv_initial,
        )
        nv_record["initialization"] = args.nv_initialization
        nv_record["initialization_runtime_seconds"] = ev_initialization_runtime
        nv_record["total_pipeline_runtime_seconds"] = (
            ev_initialization_runtime
            + float(nv_record["training_runtime_seconds"])
            if args.nv_initialization == "ev"
            else float(nv_record["training_runtime_seconds"])
        )
        variants["nv"] = nv_record

    result = {
        "status": "ok",
        "network": network,
        "sample_size": int(sample_size),
        "seed": int(seed),
        "variables": len(metadata.nodes),
        "sample_path": str(sample_path),
        "preprocessing": preprocessing_information,
        "graph_threshold": float(args.graph_threshold),
        "device": args.resolved_device,
        "dtype": args.dtype,
        "data_mode": args.data_mode,
        "model_data_warning": (
            "Categorical BIF states were ordinally encoded. GOLEM's "
            "continuous linear-Gaussian model is misspecified."
        ),
        "variants": variants,
    }
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    args = build_configuration()
    print("=" * 80)
    print("GOLEM: five runs on ordinally encoded categorical BIF data")
    print("WARNING: GOLEM's continuous linear-Gaussian model is misspecified.")
    print(f"Device: {args.resolved_device}")
    if args.resolved_device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Networks: {args.networks}")
    print(f"Sample sizes: {args.sample_sizes}")
    print(f"Seeds: {args.seeds}")
    print(f"Data mode: {args.data_mode}")
    print(f"Variant: {args.variant}")
    print("=" * 80)

    records: list[dict] = []
    jobs = [
        (network, sample_size, seed)
        for network in args.networks
        for sample_size in args.sample_sizes
        for seed in args.seeds
    ]
    for index, (network, sample_size, seed) in enumerate(jobs, start=1):
        print(
            f"\n[{index}/{len(jobs)}] "
            f"{network}, n={sample_size}, seed={seed}"
        )
        try:
            record = run_one_experiment(network, sample_size, seed, args)
        except Exception as error:
            record = {
                "status": "failed",
                "network": network,
                "sample_size": int(sample_size),
                "seed": int(seed),
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
            print(record["error"])
        records.append(record)

    summary_directory = args.output_dir / args.variant
    summary_directory.mkdir(parents=True, exist_ok=True)
    (summary_directory / "all_results.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    all_runs = pd.DataFrame(flatten_results(records))
    all_runs.to_csv(summary_directory / "all_runs.csv", index=False)
    mean_std = aggregate_mean_and_std(
        all_runs,
        number_of_runs=args.number_of_runs,
        arc_metrics_as_percent=args.report_arc_metrics_as_percent,
        mean_decimals=args.summary_mean_decimals,
        std_decimals=args.summary_std_decimals,
    )
    mean_std.to_csv(
        summary_directory / "mean_std_results.csv",
        index=False,
        float_format=f"%.{args.summary_std_decimals}f",
    )
    if args.save_per_setting_summaries:
        save_per_setting_summaries(all_runs, args)
    save_experiment_metadata(args, summary_directory)

    failures = sum(record.get("status") != "ok" for record in records)
    print("\n" + "=" * 80)
    print(f"Completed: {len(records) - failures}/{len(records)}")
    print(f"Individual runs: {summary_directory / 'all_runs.csv'}")
    print(f"Mean ± std: {summary_directory / 'mean_std_results.csv'}")
    print("=" * 80)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

