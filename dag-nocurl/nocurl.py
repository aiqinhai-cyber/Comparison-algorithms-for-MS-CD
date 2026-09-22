"""NoCurl-2 core adapted from the authors' Apache-2.0 BPR.py implementation.

Upstream: https://github.com/fishmoon1234/DAG-NoCurl/blob/master/BPR.py
Authors: Yue Yu, Tian Gao, Naiyu Yin, and Qiang Ji.

The objective, gradients, matrix-exponential potential construction, and
L-BFGS-B stages follow the uploaded execution code and upstream computational
flow. Repository-specific experiment and persistence logic lives elsewhere.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import linalg
from scipy import optimize as sopt

import config


def optimizer_record(solution: Any) -> dict[str, Any]:
    return {
        "success": bool(solution.success),
        "status": int(solution.status),
        "message": str(solution.message),
        "nit": int(getattr(solution, "nit", 0)),
        "nfev": int(getattr(solution, "nfev", 0)),
        "objective": float(solution.fun) if np.isfinite(solution.fun) else None,
    }


class NoCurl2:
    """Linear NoCurl-2 estimator returning raw weighted adjacency values."""

    def __init__(
        self,
        lambda1: float = config.LAMBDA1,
        lambda2: float = config.LAMBDA2,
        h_tol: float = config.H_TOL,
        internal_threshold: float = config.INTERNAL_THRESHOLD,
    ) -> None:
        self.lambda1 = float(lambda1)
        self.lambda2 = float(lambda2)
        self.h_tol = float(h_tol)
        self.internal_threshold = float(internal_threshold)
        self.diagnostics: dict[str, dict[str, Any]] = {}
        self.initial_weights: np.ndarray | None = None
        self.potential: np.ndarray | None = None

    def fit_raw(self, x: np.ndarray) -> tuple[np.ndarray, float]:
        """Fit once and return weights before the final graph threshold."""
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError("X must be two-dimensional")
        if x.shape[0] < 2 or x.shape[1] < 2:
            raise ValueError("At least two samples and two variables are required")
        if not np.isfinite(x).all():
            raise ValueError("Input contains non-finite values")

        sample_count, variable_count = x.shape
        lower_count = variable_count * (variable_count - 1) // 2

        def relu(value: np.ndarray, derivative: bool = False) -> np.ndarray:
            if derivative:
                return (value > 0).astype(value.dtype)
            return value * (value > 0)

        def acyclicity(weight: np.ndarray) -> tuple[float, np.ndarray]:
            matrix = weight.reshape(variable_count, variable_count)
            transformed = np.eye(variable_count) + matrix * matrix / variable_count
            power = np.linalg.matrix_power(transformed, variable_count - 1)
            value = float((power.T * transformed).sum() - variable_count)
            gradient = power.T * matrix * 2.0
            return value, gradient

        def squared_loss(weight: np.ndarray) -> float:
            residual = x.dot(np.eye(variable_count) - weight)
            return float(0.5 / sample_count * np.linalg.norm(residual, "fro") ** 2)

        rho = 0.0
        alpha = self.lambda1

        def initial_objective(flat_weight: np.ndarray) -> float:
            matrix = flat_weight.reshape(variable_count, variable_count)
            loss = squared_loss(matrix)
            h_value, _ = acyclicity(matrix)
            return loss + 0.5 * rho * h_value**2 + alpha * h_value

        def initial_gradient(flat_weight: np.ndarray) -> np.ndarray:
            matrix = flat_weight.reshape(variable_count, variable_count)
            loss_gradient = (
                -1.0
                / sample_count
                * x.T.dot(x)
                .dot(np.eye(variable_count) - matrix)
            )
            h_value, h_gradient = acyclicity(matrix)
            return (loss_gradient + (rho * h_value + alpha) * h_gradient).ravel()

        def build_symmetric_weights(
            parameter: np.ndarray,
        ) -> np.ndarray:
            symmetric = np.zeros((variable_count, variable_count))
            lower_indices = np.tril_indices(variable_count, -1)
            symmetric[lower_indices] = parameter[:lower_count]
            return symmetric + symmetric.T

        def build_potential(parameter: np.ndarray) -> np.ndarray:
            return parameter[lower_count:].reshape(-1, 1)

        def projected_weight(
            lower_parameters: np.ndarray,
            fixed_potential: np.ndarray,
        ) -> np.ndarray:
            parameter = np.zeros(lower_count + variable_count)
            parameter[:lower_count] = lower_parameters
            parameter[lower_count : lower_count + variable_count - 1] = (
                fixed_potential
            )
            symmetric = build_symmetric_weights(parameter)
            potential = build_potential(parameter)
            ones = np.ones_like(potential)
            potential_difference = ones @ potential.T - potential @ ones.T
            return symmetric * relu(potential_difference)

        def projected_objective(
            lower_parameters: np.ndarray,
            fixed_potential: np.ndarray,
        ) -> float:
            return squared_loss(projected_weight(lower_parameters, fixed_potential))

        def projected_gradient(
            lower_parameters: np.ndarray,
            fixed_potential: np.ndarray,
        ) -> np.ndarray:
            parameter = np.zeros(lower_count + variable_count)
            parameter[:lower_count] = lower_parameters
            parameter[lower_count : lower_count + variable_count - 1] = (
                fixed_potential
            )
            symmetric = build_symmetric_weights(parameter)
            potential = build_potential(parameter)
            ones = np.ones_like(potential)
            potential_difference = ones @ potential.T - potential @ ones.T
            directed_weight = symmetric * relu(potential_difference)
            weight_gradient = (
                -1.0
                / sample_count
                * x.T.dot(x)
                .dot(np.eye(variable_count) - directed_weight)
            )
            symmetric_gradient = weight_gradient * relu(potential_difference)
            symmetric_gradient = symmetric_gradient + symmetric_gradient.T
            return symmetric_gradient[np.tril_indices(variable_count, -1)]

        initial = np.zeros(variable_count * variable_count)
        diagonal_bounds = [
            (0.0, 0.0) if row == column else (None, None)
            for row in range(variable_count)
            for column in range(variable_count)
        ]
        lower_bounds = [(None, None) for _ in range(lower_count)]

        first_solution = sopt.minimize(
            initial_objective,
            initial,
            method="L-BFGS-B",
            jac=initial_gradient,
            bounds=diagonal_bounds,
            options={"ftol": self.h_tol},
        )
        self.diagnostics["stage1"] = optimizer_record(first_solution)

        alpha = self.lambda2
        second_solution = sopt.minimize(
            initial_objective,
            np.asarray(first_solution.x).copy(),
            method="L-BFGS-B",
            jac=initial_gradient,
            bounds=diagonal_bounds,
            options={"ftol": self.h_tol},
        )
        self.diagnostics["stage2"] = optimizer_record(second_solution)
        initial_weight = np.asarray(second_solution.x).copy()
        self.initial_weights = initial_weight.reshape(variable_count, variable_count)

        thresholded = initial_weight.copy()
        thresholded[np.abs(thresholded) < self.internal_threshold] = 0.0
        reachability_input = np.sign(np.abs(thresholded.reshape(variable_count, variable_count)))
        reachability = np.sign(
            linalg.expm(reachability_input) - np.identity(variable_count)
        )
        antisymmetric = reachability - reachability.T

        reduced_laplacian = np.ones((variable_count - 1, variable_count - 1))
        for index in range(variable_count - 1):
            reduced_laplacian[index, index] = -(variable_count - 1)
        divergence = np.sum(antisymmetric, axis=1)
        estimated_potential = np.linalg.solve(
            reduced_laplacian,
            divergence[: variable_count - 1],
        )

        lower_solution = sopt.minimize(
            projected_objective,
            np.zeros(lower_count),
            args=(estimated_potential,),
            method="L-BFGS-B",
            jac=projected_gradient,
            bounds=lower_bounds,
            options={"ftol": self.h_tol},
        )
        self.diagnostics["weight_stage"] = optimizer_record(lower_solution)

        final_parameter = np.zeros(lower_count + variable_count)
        final_parameter[:lower_count] = lower_solution.x
        final_parameter[lower_count : lower_count + variable_count - 1] = (
            estimated_potential
        )
        symmetric = build_symmetric_weights(final_parameter)
        potential = build_potential(final_parameter)
        ones = np.ones_like(potential)
        raw_weight = symmetric * relu(ones @ potential.T - potential @ ones.T)
        self.potential = potential.copy()
        return raw_weight, squared_loss(raw_weight)
