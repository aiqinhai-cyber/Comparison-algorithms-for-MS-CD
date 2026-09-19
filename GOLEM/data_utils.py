"""BIF parsing, categorical encoding, sampling, and preprocessing utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Hashable, Iterable

import numpy as np
import pandas as pd
from pgmpy.readwrite import BIFReader
from pgmpy.sampling import BayesianModelSampling


@dataclass(frozen=True)
class BifMetadata:
    """Ground-truth model and variable metadata read from a BIF file."""

    model: object
    nodes: list[str]
    state_names: dict[str, list[Hashable]]
    true_adjacency: np.ndarray


def resolve_bif_file(
    bif_dir: str | Path,
    network: str,
    fallback_dirs: Iterable[str | Path] = (),
) -> Path:
    """Find ``network.bif`` with a case-insensitive fallback.

    The primary location is ``bif_dir``. Each fallback directory is searched
    afterwards, allowing BIF files to remain beside the Python files.
    """
    search_dirs = [Path(bif_dir), *(Path(item) for item in fallback_dirs)]
    unique_dirs: list[Path] = []
    for directory in search_dirs:
        resolved = directory.resolve()
        if resolved not in unique_dirs:
            unique_dirs.append(resolved)

    matches: list[Path] = []
    for directory in unique_dirs:
        expected = directory / f"{network}.bif"
        if expected.is_file():
            matches.append(expected)
            continue
        if directory.is_dir():
            matches.extend(
                path
                for path in directory.iterdir()
                if path.is_file()
                and path.suffix.lower() == ".bif"
                and path.stem.lower() == network.lower()
            )

    # The same path can be reached through two equivalent directory entries.
    matches = list(dict.fromkeys(path.resolve() for path in matches))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple BIF files match network {network!r}: {matches}"
        )

    available = sorted(
        str(path)
        for directory in unique_dirs
        if directory.is_dir()
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() == ".bif"
    )
    raise FileNotFoundError(
        f"BIF file for network {network!r} was not found. "
        f"Searched directories: {[str(path) for path in unique_dirs]}. "
        f"Available BIF files: {available}"
    )


def load_bif_metadata(path: str | Path) -> BifMetadata:
    """Read the ground-truth DAG and complete categorical state space."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"BIF file not found: {path}")

    model = BIFReader(path=str(path)).get_model()
    nodes = list(model.nodes())
    node_index = {node: index for index, node in enumerate(nodes)}

    state_names: dict[str, list[Hashable]] = {}
    for node in nodes:
        cpd = model.get_cpds(node)
        if cpd is None:
            raise ValueError(f"No CPD found for variable {node!r} in {path}")
        state_names[node] = list(cpd.state_names[node])

    true_adjacency = np.zeros((len(nodes), len(nodes)), dtype=np.int8)
    for parent, child in model.edges():
        true_adjacency[node_index[parent], node_index[child]] = 1

    return BifMetadata(model, nodes, state_names, true_adjacency)


def find_sample_file(
    sample_dir: str | Path,
    network: str,
    size: int,
) -> Path | None:
    """Find one shared sample file and reject ambiguous matches."""
    root = Path(sample_dir)
    candidates = [
        root / f"{network}_{size}.csv",
        root / network / f"{network}_{size}.csv",
        root / network / f"{size}.csv",
        root / network / f"n{size}.csv",
        root / f"{network}_{size}.txt",
        root / network / f"{network}_{size}.txt",
        root / network / f"{size}.txt",
        root / network / f"n{size}.txt",
    ]
    matches = [path for path in candidates if path.is_file()]
    if len(matches) > 1:
        raise RuntimeError(
            f"More than one sample file matches {network}, n={size}: {matches}"
        )
    return matches[0] if matches else None


def find_seeded_sample_file(
    sample_dir: str | Path,
    network: str,
    size: int,
    seed: int,
) -> Path | None:
    """Find one seed-specific CSV and reject ambiguous matches."""
    root = Path(sample_dir)
    candidates = [
        root / f"{network}_{size}_seed{seed}.csv",
        root / f"{network}_{size}_seed_{seed}.csv",
        root / network / f"{network}_{size}_seed{seed}.csv",
        root / network / f"{network}_{size}_seed_{seed}.csv",
        root / network / f"n{size}_seed{seed}.csv",
        root / network / f"n{size}_seed_{seed}.csv",
        root / network / f"{size}_seed{seed}.csv",
        root / network / f"{size}_seed_{seed}.csv",
    ]
    matches = [path for path in candidates if path.is_file()]
    if len(matches) > 1:
        raise RuntimeError(
            f"More than one seeded sample file matches {network}, "
            f"n={size}, seed={seed}: {matches}"
        )
    return matches[0] if matches else None


def _drop_accidental_index_columns(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        column
        for column in frame.columns
        if str(column).strip().lower().startswith("unnamed:")
    ]
    return frame.drop(columns=columns) if columns else frame


def encode_state_column(
    series: pd.Series,
    states: list[Hashable],
    variable: str,
) -> np.ndarray:
    """Map states to ``0,...,r-1`` using the BIF declaration order."""
    state_map = {str(state).strip(): index for index, state in enumerate(states)}
    encoded = np.empty(len(series), dtype=np.float64)

    for row, value in enumerate(series.tolist()):
        if pd.isna(value):
            raise ValueError(f"Missing value in {variable!r}, row {row}")

        normalized = str(value).strip()
        if normalized in state_map:
            encoded[row] = state_map[normalized]
            continue

        try:
            numeric = int(value)
        except (TypeError, ValueError):
            numeric = -1
        if 0 <= numeric < len(states):
            encoded[row] = numeric
            continue

        raise ValueError(
            f"Unknown state {value!r} in variable {variable!r}, row {row}. "
            f"BIF states are {states!r}."
        )
    return encoded


def load_numeric_samples(path: str | Path, metadata: BifMetadata) -> np.ndarray:
    """Load samples and ordinally encode them in BIF node/state order."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Sample file not found: {path}")

    separator = "\t" if path.suffix.lower() in {".txt", ".tsv"} else ","
    frame = _drop_accidental_index_columns(pd.read_csv(path, sep=separator))
    missing = [node for node in metadata.nodes if node not in frame.columns]
    extra = [column for column in frame.columns if column not in metadata.nodes]
    if missing or extra:
        raise ValueError(
            f"Sample columns do not match the BIF. Missing={missing}; "
            f"extra={extra}. Names are case-sensitive."
        )

    frame = frame.loc[:, metadata.nodes]
    return np.column_stack(
        [
            encode_state_column(frame[node], metadata.state_names[node], node)
            for node in metadata.nodes
        ]
    )


def generate_samples_from_bif(
    metadata: BifMetadata,
    size: int,
    seed: int,
    output_path: str | Path,
) -> Path:
    """Draw one reproducible sample from the Bayesian network."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sampler = BayesianModelSampling(metadata.model)
    frame = sampler.forward_sample(size=size, seed=seed, show_progress=False)
    frame.loc[:, metadata.nodes].to_csv(output_path, index=False)
    return output_path


def preprocess_data(data: np.ndarray, method: str) -> tuple[np.ndarray, dict]:
    """Center data as in official GOLEM, or optionally z-score it."""
    data = np.asarray(data, dtype=np.float64)
    means = data.mean(axis=0)
    processed = data - means[None, :]
    information = {"method": method, "means": means.tolist()}

    if method == "center":
        return processed, information
    if method == "zscore":
        scales = processed.std(axis=0, ddof=0)
        constant = np.where(scales == 0)[0]
        if constant.size:
            raise ValueError(
                f"Cannot z-score constant variables at indices "
                f"{constant.tolist()}"
            )
        information["scales"] = scales.tolist()
        return processed / scales[None, :], information
    raise ValueError("PREPROCESSING must be either 'center' or 'zscore'")

