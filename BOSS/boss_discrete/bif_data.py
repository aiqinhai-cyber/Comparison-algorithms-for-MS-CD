from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Hashable

import numpy as np
import pandas as pd
from pgmpy.readwrite import BIFReader
from pgmpy.sampling import BayesianModelSampling


@dataclass(frozen=True)
class BifMetadata:
    model: object
    nodes: list[str]
    state_names: dict[str, list[Hashable]]
    cardinalities: np.ndarray
    true_adjacency: np.ndarray


def load_bif_metadata(path: str | Path) -> BifMetadata:
    """Read node order, complete state spaces, and the true DAG from a BIF."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"BIF file not found: {path}")

    model = BIFReader(path=str(path)).get_model()
    nodes = list(model.nodes())
    node_to_index = {node: index for index, node in enumerate(nodes)}

    state_names: dict[str, list[Hashable]] = {}
    for node in nodes:
        cpd = model.get_cpds(node)
        if cpd is None:
            raise ValueError(f"No CPD found for variable {node!r} in {path}")
        state_names[node] = list(cpd.state_names[node])

    cardinalities = np.asarray(
        [len(state_names[node]) for node in nodes], dtype=np.int64
    )
    true_adjacency = np.zeros((len(nodes), len(nodes)), dtype=np.int8)
    for parent, child in model.edges():
        true_adjacency[node_to_index[parent], node_to_index[child]] = 1

    return BifMetadata(
        model=model,
        nodes=nodes,
        state_names=state_names,
        cardinalities=cardinalities,
        true_adjacency=true_adjacency,
    )


def _drop_accidental_index_columns(frame: pd.DataFrame) -> pd.DataFrame:
    accidental = [
        column
        for column in frame.columns
        if str(column).strip().lower().startswith("unnamed:")
    ]
    return frame.drop(columns=accidental) if accidental else frame


def _encode_column(
    series: pd.Series,
    states: list[Hashable],
    variable: str,
) -> np.ndarray:
    """Encode using the BIF state order, not the states observed in a sample."""
    exact_map = {state: index for index, state in enumerate(states)}
    string_map = {str(state).strip(): index for index, state in enumerate(states)}

    encoded = np.empty(len(series), dtype=np.int64)
    for row, value in enumerate(series.tolist()):
        if pd.isna(value):
            raise ValueError(f"Missing value in variable {variable!r}, row {row}")

        try:
            if value in exact_map:
                encoded[row] = exact_map[value]
                continue
        except TypeError:
            pass

        normalized = str(value).strip()
        if normalized in string_map:
            encoded[row] = string_map[normalized]
            continue

        # Accept data already encoded as 0, ..., r-1.
        try:
            numeric = int(value)
        except (TypeError, ValueError):
            numeric = -1
        if str(numeric) == normalized or isinstance(value, (int, np.integer)):
            if 0 <= numeric < len(states):
                encoded[row] = numeric
                continue

        raise ValueError(
            f"Unknown state {value!r} for variable {variable!r}. "
            f"BIF states are {states!r}."
        )

    return encoded


def load_and_encode_samples(
    sample_path: str | Path,
    metadata: BifMetadata,
) -> np.ndarray:
    sample_path = Path(sample_path)
    if not sample_path.is_file():
        raise FileNotFoundError(f"Sample file not found: {sample_path}")

    separator = "\t" if sample_path.suffix.lower() in {".txt", ".tsv"} else ","
    frame = pd.read_csv(sample_path, sep=separator)
    frame = _drop_accidental_index_columns(frame)

    missing = [node for node in metadata.nodes if node not in frame.columns]
    extra = [column for column in frame.columns if column not in metadata.nodes]
    if missing or extra:
        raise ValueError(
            f"Columns do not match the BIF. Missing={missing}; extra={extra}. "
            "Column names are case-sensitive."
        )

    frame = frame.loc[:, metadata.nodes]
    encoded = np.column_stack(
        [
            _encode_column(frame[node], metadata.state_names[node], node)
            for node in metadata.nodes
        ]
    )
    return encoded.astype(np.int64, copy=False)


def generate_samples_from_bif(
    metadata: BifMetadata,
    size: int,
    seed: int,
    output_path: str | Path,
) -> Path:
    """Generate one i.i.d. observational sample file with pgmpy."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sampler = BayesianModelSampling(metadata.model)
    frame = sampler.forward_sample(size=size, seed=seed, show_progress=False)
    frame = frame.loc[:, metadata.nodes]
    frame.to_csv(output_path, index=False)
    return output_path


def find_sample_file(sample_dir: str | Path, network: str, size: int) -> Path | None:
    """Recognize several common layouts without silently choosing duplicates."""
    sample_dir = Path(sample_dir)
    candidates = [
        sample_dir / f"{network}_{size}.csv",
        sample_dir / network / f"{network}_{size}.csv",
        sample_dir / network / f"{size}.csv",
        sample_dir / network / f"n{size}.csv",
        sample_dir / f"{network}_{size}.txt",
        sample_dir / network / f"{network}_{size}.txt",
        sample_dir / network / f"{size}.txt",
    ]
    matches = [path for path in candidates if path.is_file()]
    if len(matches) > 1:
        raise RuntimeError(
            f"More than one sample file matches {network}, n={size}: {matches}"
        )
    return matches[0] if matches else None

