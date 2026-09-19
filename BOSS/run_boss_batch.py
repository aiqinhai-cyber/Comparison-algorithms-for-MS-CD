from __future__ import annotations

import argparse
import json
import platform
import random
import sys
import traceback
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from boss_discrete.bif_data import (
    find_sample_file,
    generate_samples_from_bif,
    load_and_encode_samples,
    load_bif_metadata,
)
from boss_discrete.metrics import directed_metrics
from boss_discrete.search import boss_bdeu


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Batch discrete BOSS+BDeu experiments for BIF networks."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--network", default=None, help="Run only one network")
    parser.add_argument("--sample-size", type=int, default=None, help="Run only one n")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def save_adjacency(path: Path, matrix: np.ndarray, nodes: list[str]):
    pd.DataFrame(matrix, index=nodes, columns=nodes).to_csv(path)


def run_one(config: dict, network: str, sample_size: int, overwrite: bool) -> dict:
    bif_dir = Path(config["bif_dir"])
    sample_dir = Path(config["sample_dir"])
    output_root = Path(config["output_dir"])
    bif_path = bif_dir / f"{network}.bif"
    output_dir = output_root / network / f"n{sample_size}"
    result_path = output_dir / "result.json"

    if result_path.is_file() and not overwrite:
        with result_path.open("r", encoding="utf-8") as file:
            return json.load(file)

    metadata = load_bif_metadata(bif_path)
    sample_path = find_sample_file(sample_dir, network, sample_size)
    if sample_path is None:
        if not config.get("generate_if_missing", False):
            raise FileNotFoundError(
                f"No sample file found for {network}, n={sample_size}. "
                "See README.md for recognized filenames."
            )
        sample_path = sample_dir / f"{network}_{sample_size}.csv"
        generate_samples_from_bif(
            metadata, sample_size, int(config["seed"]), sample_path
        )

    data = load_and_encode_samples(sample_path, metadata)
    if data.shape[0] != sample_size:
        raise ValueError(
            f"{sample_path} has {data.shape[0]} rows, expected {sample_size}"
        )

    np.random.seed(int(config["seed"]))
    random.seed(int(config["seed"]))
    result = boss_bdeu(
        data=data,
        node_names=metadata.nodes,
        cardinalities=metadata.cardinalities,
        equivalent_sample_size=float(config["equivalent_sample_size"]),
        structure_prior=float(config["structure_prior"]),
        max_parents=config.get("max_parents"),
        max_parent_configurations=config.get("max_parent_configurations"),
        number_of_starts=int(config["number_of_starts"]),
        seed=int(config["seed"]),
        max_sweeps=int(config["max_sweeps"]),
    )

    if not nx.is_directed_acyclic_graph(nx.DiGraph(result.dag_adjacency)):
        raise AssertionError("The estimated graph is not a DAG")

    metrics = directed_metrics(metadata.true_adjacency, result.dag_adjacency)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_adjacency(output_dir / "true_adjacency.csv", metadata.true_adjacency, metadata.nodes)
    save_adjacency(output_dir / "estimated_dag.csv", result.dag_adjacency, metadata.nodes)
    # causal-learn endpoint convention: for i -> j, M[j, i] = 1 and M[i, j] = -1.
    pd.DataFrame(
        np.asarray(result.cpdag.graph), index=metadata.nodes, columns=metadata.nodes
    ).to_csv(output_dir / "estimated_cpdag_endpoints.csv")
    (output_dir / "estimated_cpdag.txt").write_text(
        str(result.cpdag), encoding="utf-8"
    )

    record = {
        "status": "ok",
        "network": network,
        "sample_size": sample_size,
        "bif_path": str(bif_path),
        "sample_path": str(sample_path),
        "variables": len(metadata.nodes),
        "cardinalities": metadata.cardinalities.tolist(),
        "equivalent_sample_size": float(config["equivalent_sample_size"]),
        "structure_prior": float(config["structure_prior"]),
        "max_parents": config.get("max_parents"),
        "max_parent_configurations": config.get("max_parent_configurations"),
        "number_of_starts": int(config["number_of_starts"]),
        "seed": int(config["seed"]),
        "boss_score": result.score,
        "runtime_seconds": result.runtime_seconds,
        "order_indices": result.order,
        "order_names": [metadata.nodes[index] for index in result.order],
        **metrics,
    }
    with result_path.open("w", encoding="utf-8") as file:
        json.dump(record, file, ensure_ascii=False, indent=2)
    return record


def main():
    arguments = parse_arguments()
    config_path = Path(arguments.config)
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    networks = [arguments.network] if arguments.network else config["networks"]
    sample_sizes = [arguments.sample_size] if arguments.sample_size else config["sample_sizes"]
    jobs = [(network, int(size)) for network in networks for size in sample_sizes]

    records = []
    for network, size in tqdm(jobs, desc="BOSS experiments"):
        try:
            records.append(run_one(config, network, size, arguments.overwrite))
        except Exception as error:
            records.append(
                {
                    "status": "failed",
                    "network": network,
                    "sample_size": size,
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                }
            )

    output_root = Path(config["output_dir"])
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "all_results.json").open("w", encoding="utf-8") as file:
        json.dump(records, file, ensure_ascii=False, indent=2)

    flat_records = [
        {key: value for key, value in record.items() if not isinstance(value, (list, dict))}
        for record in records
    ]
    pd.DataFrame(flat_records).to_csv(output_root / "all_results.csv", index=False)

    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    with (output_root / "environment.json").open("w", encoding="utf-8") as file:
        json.dump(environment, file, ensure_ascii=False, indent=2)

    failures = sum(record["status"] != "ok" for record in records)
    print(f"\nCompleted {len(records) - failures}/{len(records)} jobs.")
    print(f"Summary: {output_root / 'all_results.csv'}")
    if failures:
        print(f"Failures: {failures}; inspect all_results.json")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
