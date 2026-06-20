"""White-noise-gain-floor diagonal loading — the single robustness knob (Stage P2-1).

Regularization is mandatory: a handful of heterogeneous drivers gives superdirective
blow-up at low frequency. We expose ONE user knob, a white-noise-gain (WNG) floor in
dB. Under the distortionless constraint ``c^H w = 1``, ``WNG(w) = 1 / ||w||^2``, so a
WNG floor is a weight-norm cap. The loaded weights
``w(eps) = (R+eps I)^-1 c / (c^H (R+eps I)^-1 c)`` have ``WNG(eps)`` monotone increasing
in eps, so we hit a target floor by 1-D bisection on ``log eps`` per frequency. A small
``eps_min`` is always added before Cholesky; bins where the floor is unreachable are
flagged (the directivity rolls off gracefully — never silent garbage).

References
----------
Cox, Zucker, Owen, "Robust adaptive beamforming," IEEE TASSP 1987.
docs/Phase 2 - Filter Solver.md Section 5.3.
"""

from __future__ import annotations

import numpy as np


def white_noise_gain_db(w: np.ndarray, c: np.ndarray) -> float:
    """WNG in dB for distortionless weights: ``10 log10(|c^H w|^2 / ||w||^2)``.

    Parameters
    ----------
    w : np.ndarray
        ``[M]`` complex128 weights.
    c : np.ndarray
        ``[M]`` complex128 look vector (house convention).

    Returns
    -------
    float
        White-noise gain in dB.
    """
    num = float(np.abs(np.conj(c) @ w) ** 2)
    den = float(np.real(np.conj(w) @ w))
    return 10.0 * np.log10(num / den)


def loaded_mvdr_weights(R: np.ndarray, c: np.ndarray, eps: float) -> np.ndarray:
    """Diagonally-loaded MVDR weights ``(R+eps I)^-1 c / (c^H (R+eps I)^-1 c)``."""
    raise NotImplementedError("Stage P2-1: loaded MVDR / WNG bisection not yet implemented.")


def solve_loading_for_wng(R: np.ndarray, c: np.ndarray, wng_floor_db: float) -> float:
    """Bisection on log(eps) to reach a target WNG floor (Stage P2-1).

    Returns the loading ``eps`` whose loaded MVDR weights achieve ``wng_floor_db``
    (clamped to the feasible range; flags when unreachable).
    """
    raise NotImplementedError("Stage P2-1: WNG-floor bisection not yet implemented.")
