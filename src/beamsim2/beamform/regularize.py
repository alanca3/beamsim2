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

import math

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
    """Diagonally-loaded MVDR weights ``(R+eps I)^-1 c / (c^H (R+eps I)^-1 c)``.

    Parameters
    ----------
    R : np.ndarray
        ``[M, M]`` complex128 Hermitian PSD covariance.
    c : np.ndarray
        ``[M]`` complex128 look vector (house convention ``conj(H_look)``).
    eps : float
        Diagonal loading (>= 0). ``eps -> 0`` is max-directivity (fragile);
        ``eps -> inf`` is matched-field ``c/||c||^2`` (max WNG).

    Returns
    -------
    np.ndarray
        ``w[M]`` complex128, distortionless (``c^H w == 1``).
    """
    m = R.shape[0]
    rinv_c = np.linalg.solve(R + eps * np.eye(m), c)  # [M]
    return rinv_c / (np.conj(c) @ rinv_c)


def max_white_noise_gain_db(c: np.ndarray) -> float:
    """The WNG ceiling for the distortionless beamformer: ``10 log10(||c||^2)``.

    Reached as ``eps -> inf`` (matched-field). A requested floor above this is infeasible.
    """
    return 10.0 * np.log10(float(np.real(np.conj(c) @ c)))


def solve_loading_for_wng(
    R: np.ndarray,
    c: np.ndarray,
    wng_floor_db: float,
    *,
    tol_db: float = 0.02,
    max_iter: int = 80,
) -> tuple[float, bool]:
    """Bisection on ``log(eps)`` to reach a target WNG floor for loaded MVDR.

    ``WNG(eps)`` is monotone increasing in ``eps`` (matched-field is the ceiling), so a
    simple geometric bisection lands the loading that achieves ``wng_floor_db``.

    Parameters
    ----------
    R : np.ndarray
        ``[M, M]`` covariance.
    c : np.ndarray
        ``[M]`` look vector.
    wng_floor_db : float
        Target WNG floor in dB.
    tol_db : float
        Convergence tolerance on the achieved WNG.
    max_iter : int
        Bisection iteration cap.

    Returns
    -------
    eps : float
        The diagonal loading achieving the floor (or the max-robustness loading if the
        floor is unreachable).
    feasible : bool
        ``False`` if the requested floor exceeds the WNG ceiling (clamped to max robustness)
        — the caller flags the bin rather than emitting fragile garbage.
    """
    m = R.shape[0]
    scale = float(np.real(np.trace(R))) / m  # covariance scale for eps bracketing
    eps_lo, eps_hi = 1e-12 * scale, 1e6 * scale

    def wng_db(eps: float) -> float:
        w = loaded_mvdr_weights(R, c, eps)
        return 10.0 * np.log10(np.abs(np.conj(c) @ w) ** 2 / np.real(np.conj(w) @ w))

    if wng_floor_db >= max_white_noise_gain_db(c) - tol_db:
        return eps_hi, False  # unreachable: clamp to max robustness, flag infeasible
    if wng_db(eps_lo) >= wng_floor_db:
        return eps_lo, True  # already robust enough at minimal loading

    for _ in range(max_iter):
        eps_mid = math.sqrt(eps_lo * eps_hi)  # geometric mean = bisection on log eps
        if wng_db(eps_mid) < wng_floor_db:
            eps_lo = eps_mid
        else:
            eps_hi = eps_mid
        if abs(wng_db(eps_mid) - wng_floor_db) < tol_db:
            return eps_mid, True
    return math.sqrt(eps_lo * eps_hi), True


def lambda_for_ls(robustness: float, a_matrix: np.ndarray) -> float:
    """Map a robustness slider ``s in [0, 1]`` to an LS Tikhonov ``lambda``.

    Scale-invariant: ``lambda = frac(s) * trace(A)/M`` with ``frac`` log-spaced from a tiny
    value (sharp, fragile) to a large value (heavily smoothed, robust). For LS the WNG is
    reported as a diagnostic (it is not a clean monotone function of lambda), and bins below
    a WNG floor are flagged.

    Parameters
    ----------
    robustness : float
        ``s in [0, 1]`` (0 = sharpest, 1 = most robust).
    a_matrix : np.ndarray
        ``[M, M]`` the LS normal matrix ``conj(H) W H^T`` (for the trace scale).

    Returns
    -------
    float
        The Tikhonov ``lambda``.
    """
    s = float(np.clip(robustness, 0.0, 1.0))
    frac_lo, frac_hi = 1e-6, 1e1
    frac = frac_lo * (frac_hi / frac_lo) ** s
    scale = float(np.real(np.trace(a_matrix))) / a_matrix.shape[0]
    return frac * scale
