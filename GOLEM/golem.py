"""PyTorch implementation of GOLEM and official-style graph post-processing."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import torch
from tqdm import trange


@dataclass
class GolemFitResult:
    weighted_adjacency: np.ndarray
    history: list[dict[str, float | int]]
    runtime_seconds: float


class GolemModel(torch.nn.Module):
    """GOLEM objective from Ng, Ghassami, and Zhang (NeurIPS 2020)."""

    def __init__(
        self,
        number_of_variables: int,
        lambda_1: float,
        lambda_2: float,
        equal_variances: bool,
        initial_adjacency: np.ndarray | None,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.d = int(number_of_variables)
        self.lambda_1 = float(lambda_1)
        self.lambda_2 = float(lambda_2)
        self.equal_variances = bool(equal_variances)

        if initial_adjacency is None:
            initial = np.zeros((self.d, self.d), dtype=np.float32)
        else:
            initial = np.asarray(initial_adjacency)
            if initial.shape != (self.d, self.d):
                raise ValueError(
                    f"Initial adjacency shape is {initial.shape}; "
                    f"expected {(self.d, self.d)}"
                )

        self.raw_adjacency = torch.nn.Parameter(
            torch.as_tensor(initial, dtype=dtype).clone()
        )
        self.register_buffer(
            "off_diagonal_mask",
            torch.ones(self.d, self.d, dtype=dtype)
            - torch.eye(self.d, dtype=dtype),
        )

    def adjacency(self) -> torch.Tensor:
        """Return the weighted adjacency with an exactly zero diagonal."""
        return self.raw_adjacency * self.off_diagonal_mask

    def components(self, data: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute likelihood, sparsity, DAG penalty, and total objective."""
        adjacency = self.adjacency()
        residual = data - data @ adjacency
        tiny = torch.finfo(data.dtype).tiny

        if self.equal_variances:
            rss = torch.sum(residual.square()).clamp_min(tiny)
            likelihood = 0.5 * self.d * torch.log(rss)
        else:
            rss = torch.sum(residual.square(), dim=0).clamp_min(tiny)
            likelihood = 0.5 * torch.sum(torch.log(rss))

        identity = torch.eye(self.d, device=data.device, dtype=data.dtype)
        _, log_abs_determinant = torch.linalg.slogdet(identity - adjacency)
        likelihood = likelihood - log_abs_determinant

        l1_penalty = torch.sum(torch.abs(adjacency))
        dag_penalty = torch.trace(torch.matrix_exp(adjacency * adjacency)) - self.d
        objective = (
            likelihood
            + self.lambda_1 * l1_penalty
            + self.lambda_2 * dag_penalty
        )
        return {
            "objective": objective,
            "likelihood": likelihood,
            "l1": l1_penalty,
            "h": dag_penalty,
        }


def set_reproducibility(seed: int, deterministic: bool) -> None:
    """Set Python, NumPy, CPU, and CUDA random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True


def fit_golem(
    data: np.ndarray,
    lambda_1: float,
    lambda_2: float,
    equal_variances: bool,
    num_iter: int,
    learning_rate: float,
    seed: int,
    deterministic: bool,
    device: str,
    dtype: torch.dtype,
    initial_adjacency: np.ndarray | None,
    checkpoint_interval: int,
    checkpoint_directory: Path,
    show_progress: bool,
) -> GolemFitResult:
    """Fit GOLEM by full-batch Adam optimization."""
    set_reproducibility(seed, deterministic)
    tensor = torch.as_tensor(data, device=device, dtype=dtype)
    model = GolemModel(
        number_of_variables=data.shape[1],
        lambda_1=lambda_1,
        lambda_2=lambda_2,
        equal_variances=equal_variances,
        initial_adjacency=initial_adjacency,
        dtype=dtype,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    checkpoint_directory.mkdir(parents=True, exist_ok=True)

    history: list[dict[str, float | int]] = []
    started = time.perf_counter()
    iterator = trange(
        num_iter,
        disable=not show_progress,
        desc="GOLEM-EV" if equal_variances else "GOLEM-NV",
    )
    for iteration in iterator:
        optimizer.zero_grad(set_to_none=True)
        components = model.components(tensor)
        objective = components["objective"]
        if not torch.isfinite(objective):
            raise FloatingPointError(
                f"Non-finite GOLEM objective at iteration {iteration + 1}"
            )
        objective.backward()
        optimizer.step()

        record_now = (
            iteration == 0
            or iteration + 1 == num_iter
            or (
                checkpoint_interval > 0
                and (iteration + 1) % checkpoint_interval == 0
            )
        )
        if record_now:
            record: dict[str, float | int] = {"iteration": iteration + 1}
            record.update(
                {
                    name: float(value.detach().cpu())
                    for name, value in components.items()
                }
            )
            history.append(record)
            if show_progress:
                iterator.set_postfix(
                    objective=f"{record['objective']:.5f}",
                    h=f"{record['h']:.3e}",
                )
            np.save(
                checkpoint_directory / f"weighted_iter_{iteration + 1}.npy",
                model.adjacency().detach().cpu().numpy(),
            )

    return GolemFitResult(
        weighted_adjacency=model.adjacency().detach().cpu().numpy(),
        history=history,
        runtime_seconds=time.perf_counter() - started,
    )


def is_dag(weighted_adjacency: np.ndarray) -> bool:
    """Return whether the nonzero pattern represents a directed acyclic graph."""
    graph = nx.DiGraph(np.asarray(weighted_adjacency) != 0)
    return nx.is_directed_acyclic_graph(graph)


def threshold_till_dag(
    weighted_adjacency: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Delete weakest remaining edges until a DAG is obtained."""
    result = np.asarray(weighted_adjacency).copy()
    if is_dag(result):
        return result, 0.0

    rows, columns = np.where(result != 0)
    candidates = sorted(
        (
            (abs(float(result[row, column])), int(row), int(column))
            for row, column in zip(rows, columns)
        ),
        key=lambda item: item[0],
    )
    final_threshold = 0.0
    for absolute_weight, row, column in candidates:
        result[row, column] = 0.0
        final_threshold = absolute_weight
        if is_dag(result):
            return result, final_threshold
    raise RuntimeError("Failed to obtain a DAG during post-processing")


def postprocess(
    weighted_adjacency: np.ndarray,
    graph_threshold: float,
) -> tuple[np.ndarray, float]:
    """Apply absolute thresholding and then enforce acyclicity."""
    result = np.asarray(weighted_adjacency).copy()
    np.fill_diagonal(result, 0.0)
    result[np.abs(result) <= graph_threshold] = 0.0
    return threshold_till_dag(result)


def resolve_device(requested: str) -> str:
    """Resolve ``auto`` and validate an explicitly requested CUDA device."""
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    return requested

