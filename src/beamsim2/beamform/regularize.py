"""White-noise-gain-floor diagonal loading — the single robustness knob (Stage P2-1).

Regularization is mandatory: a handful of heterogeneous drivers gives superdirective
blow-up at low frequency. We expose ONE user knob, a white-noise-gain (WNG) floor in
dB. WNG is the **dimensionless array gain against spatially-white sensor noise**,
``WNG(w) = |c^H w|^2 / ( ||w||^2 * (||c||^2 / M) )`` — i.e. normalized by the average
per-element power ``||c||^2/M`` so it does NOT depend on the absolute level of ``H``.
The matched-field (delay-and-sum) corner reaches the ceiling ``WNG = M`` (``10 log10 M``
dB); a requested floor above that is infeasible. This normalization is the Chunk-5a
fix: the previous un-normalized ``|c^H w|^2/||w||^2`` measured absolute Pa^2, so for
real BEM data (``|H| ~ 1e-3`` Pa) the ceiling sat tens of dB below any usable floor,
making every bin infeasible and collapsing every adaptive engine to the omni corner.
The loaded weights ``w(eps) = (R+eps I)^-1 c / (c^H (R+eps I)^-1 c)`` have ``WNG(eps)``
monotone increasing in eps (the normalization is a per-frequency constant offset, so
monotonicity is preserved), so we hit a target floor by 1-D bisection on ``log eps``
per frequency. A small ``eps_min`` is always added before Cholesky; bins where the
floor is unreachable are flagged (the directivity rolls off gracefully — never silent
garbage).

References
----------
Cox, Zucker, Owen, "Robust adaptive beamforming," IEEE TASSP 1987.
docs/Phase 2 - Filter Solver.md Section 5.3.
"""

from __future__ import annotations

import math

import numpy as np


def white_noise_gain_db(w: np.ndarray, c: np.ndarray) -> float:
    """WNG in dB: the dimensionless array gain ``10 log10(|c^H w|^2 / (||w||^2 * ||c||^2/M))``.

    Normalized by the average per-element power ``||c||^2/M`` (``M = len(c)``) so the metric
    is **scale-invariant in the absolute level of H** (and, as before, in ``w``: a global
    complex factor cancels). With this normalization the matched-field / delay-sum corner
    ``w = c/M`` reaches the ceiling ``WNG = M`` (``10 log10 M`` dB) regardless of how loud
    ``H`` is, so the user's WNG-floor knob (``-20 dB .. 10 log10 M``) has a consistent,
    physically meaningful range across datasets. The un-normalized form
    (``|c^H w|^2/||w||^2``) measured absolute Pa^2 and was unreachable for real BEM data.

    Parameters
    ----------
    w : np.ndarray
        ``[M]`` complex128 weights.
    c : np.ndarray
        ``[M]`` complex128 look vector (house convention).

    Returns
    -------
    float
        White-noise gain in dB; ``-inf`` for a silent look direction (``||c|| = 0``) or
        all-zero weights.
    """
    m = c.shape[0]
    cc = float(np.real(np.conj(c) @ c))  # ||c||^2 (avg element power = cc/M)
    ww = float(np.real(np.conj(w) @ w))  # ||w||^2
    if cc <= 0.0 or ww <= 0.0:
        return float("-inf")
    num = float(np.abs(np.conj(c) @ w) ** 2)
    return 10.0 * np.log10(num / (ww * cc / m))


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
    """The WNG ceiling for the distortionless beamformer: ``10 log10(M)`` (``M = len(c)``).

    With the per-element-power normalization in :func:`white_noise_gain_db`, the matched-field
    corner ``w = c/M`` reaches the array-gain ceiling ``M`` regardless of the absolute level of
    ``H``. Reached as ``eps -> inf`` (matched-field). A requested floor above this is infeasible.
    """
    return 10.0 * np.log10(float(c.shape[0]))


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
        # Use the single normalized definition (Chunk-5a) so the bracket test against
        # max_white_noise_gain_db compares like-for-like.
        return white_noise_gain_db(loaded_mvdr_weights(R, c, eps), c)

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


def floor_covariances(
    A: np.ndarray, R: np.ndarray, eps_min: float
) -> tuple[np.ndarray, np.ndarray]:
    """Add a small RELATIVE diagonal floor to both generalized-eigenproblem matrices.

    The constant-DI / max-directivity engines solve a generalized eigenproblem ``A w = tau R w``
    and a secular root of ``A - tau R``. At band edges (and for the rank-1 proper-DI ``A = c c^H``)
    these can be numerically barely-indefinite or near-singular. A scale-invariant floor
    ``eps_min * trace(R)/M`` on the diagonal of **both** A and R keeps every bin well-posed without
    disturbing well-conditioned bins (at a fixed ``tau`` the achieved directivity is invariant to
    ``eps_min`` to < 0.1 dB; ``docs/Chunk3b_Findings.md``).

    Parameters
    ----------
    A, R : np.ndarray
        ``[M, M]`` complex128 Hermitian PSD accept / reject covariances.
    eps_min : float
        Relative floor fraction (e.g. ``1e-7``).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(A_floored, R_floored)``.
    """
    m = R.shape[0]
    f = eps_min * float(np.real(np.trace(R))) / m  # scale-invariant absolute floor
    eye = f * np.eye(m)
    return A + eye, R + eye


def solve_maxdir_loading_for_wng(
    A: np.ndarray,
    R: np.ndarray,
    c: np.ndarray,
    wng_floor_db: float,
    *,
    eps_min: float = 1e-7,
    tol_db: float = 0.02,
    max_iter: int = 60,
) -> tuple[np.ndarray, float, float, bool]:
    """Diagonally load the *reject* covariance so max-directivity meets a WNG floor.

    Maximum-directivity (the top generalized eigenvector of ``(A, R)``) is the directivity
    *ceiling* and is freely superdirective — its white-noise gain can plunge tens of dB at low
    frequency. Loading the reject covariance, ``w = top-eig(A, R + eps I)``, walks the solution
    from the fragile ceiling (``eps -> 0``) toward the robust matched-field beam (``eps -> inf``);
    ``WNG(eps)`` is monotone increasing, so a geometric bisection on ``eps`` lands the floor. If
    the floor exceeds the matched-field ceiling ``10log10||c||^2`` it is unreachable: clamp to max
    robustness and flag (graceful roll-off, never silent superdirective garbage). Mirrors the
    MVDR floor (:func:`solve_loading_for_wng`) for the directivity engines.

    Parameters
    ----------
    A, R : np.ndarray
        ``[M, M]`` accept / reject covariances (whichever directivity objective is in force).
    c : np.ndarray
        ``[M]`` look vector (house convention).
    wng_floor_db : float
        Target WNG floor (dB).
    eps_min : float
        Relative ``eps_min`` floor applied to A and R first (:func:`floor_covariances`).
    tol_db, max_iter : float, int
        Bisection tolerance and iteration cap.

    Returns
    -------
    w : np.ndarray
        ``[M]`` complex128 distortionless max-directivity weights meeting the floor.
    eps : float
        The reject-covariance loading used.
    wng_db : float
        Achieved WNG (dB).
    feasible : bool
        ``False`` if the floor exceeds the matched-field WNG ceiling (clamped + flagged).
    """
    from beamsim2.beamform.weights import max_directivity

    A, R = floor_covariances(A, R, eps_min)
    m = R.shape[0]
    scale = float(np.real(np.trace(R))) / m
    wng_ceiling = max_white_noise_gain_db(c)

    def wng_w(eps: float) -> tuple[float, np.ndarray]:
        w, _ = max_directivity(A, R, eps=eps, c=c)  # top gen-eig of (A, R + eps I)
        return white_noise_gain_db(w, c), w

    eps_lo, eps_hi = 1e-12 * scale, 1e6 * scale
    wlo, wlo_w = wng_w(eps_lo)
    if wlo >= wng_floor_db:
        return wlo_w, eps_lo, wlo, True  # already robust at minimal load
    if wng_floor_db >= wng_ceiling - tol_db:
        whi, whi_w = wng_w(eps_hi)
        return whi_w, eps_hi, whi, False  # floor above ceiling -> clamp + flag

    w_mid = wlo_w
    for _ in range(max_iter):
        eps_mid = math.sqrt(eps_lo * eps_hi)  # geometric mean = bisection on log eps
        wm, w_mid = wng_w(eps_mid)
        if wm < wng_floor_db:
            eps_lo = eps_mid
        else:
            eps_hi = eps_mid
        if abs(wm - wng_floor_db) < tol_db:
            return w_mid, eps_mid, wm, True
    eps_f = math.sqrt(eps_lo * eps_hi)
    wf, w_mid = wng_w(eps_f)
    return w_mid, eps_f, wf, True


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


def ls_wng_lambda_grid(
    a_matrix: np.ndarray, *, n_grid: int = 48, frac_lo: float = 1e-6, frac_hi: float = 1e4
) -> np.ndarray:
    """Log-spaced per-bin Tikhonov ``lambda`` grid for the LS WNG-floor search.

    The LS WNG is *not* monotone in ``lambda`` (unlike loaded MVDR), so the floor is hit by
    a grid search rather than bisection. The grid is scale-invariant: spaced over
    ``[frac_lo, frac_hi] * trace(A)/M`` so it behaves the same across geometries and levels
    (mirrors :func:`lambda_for_ls` scaling).

    Parameters
    ----------
    a_matrix : np.ndarray
        ``[M, M]`` the LS normal matrix ``conj(H) W H^T`` (for the trace scale).
    n_grid : int
        Number of grid points.
    frac_lo, frac_hi : float
        Fractional bounds of the grid (relative to ``trace(A)/M``).

    Returns
    -------
    np.ndarray
        ``[n_grid]`` float64, ascending.
    """
    scale = float(np.real(np.trace(a_matrix))) / a_matrix.shape[0]
    return np.geomspace(frac_lo * scale, frac_hi * scale, n_grid)
