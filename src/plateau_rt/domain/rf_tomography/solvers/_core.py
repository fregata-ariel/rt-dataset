"""Shared iterative machinery of the E2 solvers (private, NumPy only).

:mod:`power` and :mod:`coherent` both run a monotone FISTA (Beck & Teboulle
2009), estimate an operator norm by power iteration and count forward / adjoint
applications.  This module holds that common machinery once.  The
floating-point work stays in caller-supplied closures so every solver keeps its
own arithmetic exactly as before.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

__all__ = [
    "CallCounter",
    "IterativeResult",
    "mfista",
    "power_iteration",
    "relative_decrease_stop",
]


@dataclass(frozen=True)
class IterativeResult:
    """Result of one iterative E2 solve (shared by power and coherent solvers)."""

    x: np.ndarray
    objective: np.ndarray  # [n_iter + 1] float64, objective[0] is the start point
    n_iter: int  # iterations performed
    n_forward: int  # forward (A / K) applications, power-iteration calls included
    n_adjoint: int  # adjoint (A^H / K^T) applications
    converged: bool


class CallCounter:
    """Wrap a forward and an adjoint callable and count their calls."""

    def __init__(
        self,
        forward: Callable[[np.ndarray], np.ndarray],
        adjoint: Callable[[np.ndarray], np.ndarray],
    ) -> None:
        self._forward = forward
        self._adjoint = adjoint
        self.n_forward = 0
        self.n_adjoint = 0

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Return the wrapped forward result and count one application."""
        self.n_forward += 1
        return self._forward(x)

    def adjoint(self, y: np.ndarray) -> np.ndarray:
        """Return the wrapped adjoint result and count one application."""
        self.n_adjoint += 1
        return self._adjoint(y)


def relative_decrease_stop(objective: Sequence[float], tol: float) -> bool:
    """True when the last decrease lies in ``[0, tol * |previous|]``."""
    previous = objective[-2]
    current = objective[-1]
    decrease = previous - current
    return 0.0 <= decrease <= tol * abs(previous)


def power_iteration(
    normal: Callable[[np.ndarray], np.ndarray],
    v0: np.ndarray,
    n_iter: int,
    *,
    estimate: str,
) -> tuple[float, int]:
    """Power iteration on the normal operator; returns (value, number of normal() calls)."""
    if estimate not in ("norm", "rayleigh"):
        raise ValueError(f"estimate must be 'norm' or 'rayleigh', got {estimate!r}")
    v = v0
    value = 0.0
    calls = 0
    for _ in range(int(n_iter)):
        w = normal(v)
        calls += 1
        wnorm = float(np.linalg.norm(w.ravel()))
        if estimate == "norm":
            value = wnorm
        else:
            value = float(np.vdot(v, w).real)
        if wnorm == 0.0:
            break
        v = w / wnorm
    return value, calls


def mfista(
    x0: np.ndarray,
    ax0: np.ndarray,
    f0: float,
    *,
    forward: Callable[[np.ndarray], np.ndarray],
    prox_grad: Callable[[np.ndarray, np.ndarray], np.ndarray],
    objective: Callable[[np.ndarray, np.ndarray], float],
    n_iter: int,
    stop: str,
    tol: float,
) -> tuple[np.ndarray, list[float], int, bool]:
    """Monotone FISTA; returns (x, objective history, iterations done, converged)."""
    if stop not in ("step", "objective"):
        raise ValueError(f"stop must be 'step' or 'objective', got {stop!r}")
    x, ax, f = x0, ax0, f0
    history = [f0]
    w, aw = x, ax
    theta = 1.0
    converged = False
    n_done = 0
    residual = 0.0
    for k in range(int(n_iter)):
        z = prox_grad(w, aw)
        az = forward(z)
        fz = objective(z, az)
        theta_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * theta * theta))
        accepted = fz <= f
        x_new, ax_new = (z, az) if accepted else (x, ax)
        f = fz if accepted else f
        if stop == "step":
            denom = max(float(np.linalg.norm(z.ravel())), float(np.linalg.norm(w.ravel())))
            residual = 0.0 if denom == 0.0 else float(np.linalg.norm((z - w).ravel())) / denom
        w = x_new + (theta / theta_next) * (z - x_new) + ((theta - 1.0) / theta_next) * (x_new - x)
        aw = (
            ax_new
            + (theta / theta_next) * (az - ax_new)
            + ((theta - 1.0) / theta_next) * (ax_new - ax)
        )
        x, ax, theta = x_new, ax_new, theta_next
        history.append(f)
        n_done = k + 1
        if stop == "step" and residual <= tol:
            converged = True
            break
        if stop == "objective" and accepted and tol > 0.0 and relative_decrease_stop(history, tol):
            converged = True
            break
    return x, history, n_done, converged
