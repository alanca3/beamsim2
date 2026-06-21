"""Solver modes: per-frequency complex weights ``w_m(f)`` (Stages P2-1, P2-2).

All solvers return weights in the house convention (DR-P2-02) so that
``P(f, dir) = sum_m w_m(f) * H[m, f, dir]`` steers as commanded. The matched-field
(phase-conjugate / delay-sum) corner is implemented here as the robustness anchor and
the building block of the round-trip convention test; the regularized least-squares /
pressure-matching engine, MVDR/LCMV, and the Luo MECD/MSCD constant-directivity engine
follow in their stages.

References
----------
Luo, *Constant Directivity Loudspeaker Beamforming*, EUSIPCO 2024, arXiv:2407.01860.
Van Trees, *Optimum Array Processing*, Wiley, 2002 (MVDR/LCMV).
docs/Phase 2 - Filter Solver.md Section 5 (the verified equations).
"""

from __future__ import annotations

import numpy as np

from beamsim2.beamform.covariance import covariance, look_vector


def matched_field(H_f: np.ndarray, look_idx: int) -> np.ndarray:
    """Phase-conjugate (matched-field / delay-and-sum) weights toward ``look_idx``.

    ``w = conj(H_f[:, look]) / M``. This is the maximum-white-noise-gain corner
    (``eps -> inf`` of the loaded MVDR) and steers the main lobe to the look
    direction *by construction* in the house convention:
    ``P(look) = sum_m |H_m,look|^2 / M`` is real and maximal.

    Parameters
    ----------
    H_f : np.ndarray
        ``[M, N]`` complex128 — per-driver field at one frequency.
    look_idx : int
        Index of the look direction in the sphere grid.

    Returns
    -------
    np.ndarray
        ``w[M]`` complex128.
    """
    M = H_f.shape[0]
    return look_vector(H_f, look_idx) / M  # [M] complex128


def ls_pressure_match(
    H_f: np.ndarray,
    b_f: np.ndarray,
    weights: np.ndarray,
    lam: float,
) -> np.ndarray:
    """Regularized least-squares / pressure-matching weights (engine #1, Stage P2-1).

    ``w = (conj(H_f) W H_f^T + lam I)^-1 conj(H_f) W b_f`` with ``W = diag(weights)``.
    (Do NOT use the microphone ``(H W H^H + lam I)^-1 H W b`` form — it mirror-steers.)

    Parameters
    ----------
    H_f : np.ndarray
        ``[M, N]`` complex128 — per-driver field at one frequency.
    b_f : np.ndarray
        ``[N]`` complex128 — desired pressure pattern on the grid.
    weights : np.ndarray
        ``[N]`` float64 — Lebedev/icosphere quadrature weights.
    lam : float
        Tikhonov regularization (effort control); ``>= 0``.

    Returns
    -------
    np.ndarray
        ``w[M]`` complex128.
    """
    m = H_f.shape[0]
    cw = np.conj(H_f) * weights[None, :]  # conj(H_f) W  [M, N]
    a = cw @ H_f.T  # conj(H_f) W H_f^T   [M, M] Hermitian PSD
    rhs = cw @ b_f  # conj(H_f) W b_f     [M]
    return np.linalg.solve(a + lam * np.eye(m), rhs)  # [M]


def mvdr(H_f: np.ndarray, look_idx: int, weights: np.ndarray, eps: float) -> np.ndarray:
    """MVDR (minimum-variance distortionless response), loaded (Stage P2-1).

    ``w = (R+eps I)^-1 c / (c^H (R+eps I)^-1 c)``, ``c = conj(H_f[:, look])``.

    Parameters
    ----------
    H_f : np.ndarray
        ``[M, N]`` complex128 — per-driver field at one frequency.
    look_idx : int
        Look-direction index in the grid.
    weights : np.ndarray
        ``[N]`` float64 — quadrature weights (build the covariance).
    eps : float
        Diagonal loading (robustness; larger -> toward delay-and-sum).

    Returns
    -------
    np.ndarray
        ``w[M]`` complex128 (distortionless: ``c^H w == 1``).
    """
    r = covariance(H_f, weights)  # [M, M]
    c = look_vector(H_f, look_idx)  # [M]
    m = H_f.shape[0]
    rinv_c = np.linalg.solve(r + eps * np.eye(m), c)  # [M]
    return rinv_c / (np.conj(c) @ rinv_c)


def lcmv(
    H_f: np.ndarray,
    look_idx: int,
    null_idx: list[int],
    weights: np.ndarray,
    eps: float,
) -> np.ndarray:
    """LCMV with hard nulls (Stage P2-1). ``w = R^-1 C (C^H R^-1 C)^-1 g``.

    Constraints: unit response toward ``look_idx`` and exact zeros toward each
    ``null_idx``. At most ``M - 1`` independent nulls (M = number of drivers).

    Parameters
    ----------
    H_f : np.ndarray
        ``[M, N]`` complex128.
    look_idx : int
        Look-direction index (constrained to unit response).
    null_idx : list[int]
        Direction indices constrained to zero response.
    weights : np.ndarray
        ``[N]`` float64 quadrature weights.
    eps : float
        Diagonal loading.

    Returns
    -------
    np.ndarray
        ``w[M]`` complex128.
    """
    r = covariance(H_f, weights)  # [M, M]
    m = H_f.shape[0]
    cols = [look_vector(H_f, look_idx)] + [look_vector(H_f, j) for j in null_idx]
    c_mat = np.column_stack(cols)  # [M, K]
    g = np.zeros(c_mat.shape[1], dtype=np.complex128)  # [K]
    g[0] = 1.0  # unit toward look, zero toward nulls
    rinv_c = np.linalg.solve(r + eps * np.eye(m), c_mat)  # [M, K]
    return rinv_c @ np.linalg.solve(c_mat.conj().T @ rinv_c, g)  # [M]


def max_directivity(
    A: np.ndarray, R: np.ndarray, *, eps: float = 1e-9, c: np.ndarray | None = None
) -> tuple[np.ndarray, float]:
    """Maximum-directivity weights and the achievable directivity ceiling at one frequency.

    Maximizing the generalized Rayleigh quotient ``w^H A w / w^H R w`` is the generalized
    eigenproblem ``A w = tau R w``; the top eigenpair gives the max-directivity beamformer
    and ``tau_max`` (the per-frequency directivity ceiling used to pick a feasible constant
    target). This is Luo's "pass 1".

    Parameters
    ----------
    A : np.ndarray
        ``[M, M]`` accept-region covariance (Hermitian PSD).
    R : np.ndarray
        ``[M, M]`` reject-region covariance (Hermitian PSD).
    eps : float
        Diagonal loading on ``R`` for a well-posed generalized eigenproblem.
    c : np.ndarray | None
        If given, the weights are scaled to the distortionless normalization ``c^H w = 1``;
        otherwise unit-norm.

    Returns
    -------
    w : np.ndarray
        ``[M]`` complex128 max-directivity weights.
    tau_max : float
        The directivity ceiling (top generalized eigenvalue ``A`` vs ``R``).
    """
    from scipy.linalg import eigh

    m = A.shape[0]
    evals, evecs = eigh(A, R + eps * np.eye(m))  # ascending
    w = evecs[:, -1]  # top generalized eigenvector
    if c is not None:
        w = w / (np.conj(c) @ w)  # distortionless scaling
    return w, float(evals[-1])


def luo_mscd(A: np.ndarray, R: np.ndarray, c: np.ndarray, tau: float) -> np.ndarray:
    """Luo MSCD (max-sensitivity constant-directivity) QCQP at fixed ``tau`` (Stage P2-2).

    Solves ``min ||w||^2  s.t.  w^H D w = 0, c^H w = 1`` with ``D = A - tau R`` — the
    minimum-norm distortionless beamformer whose generalized directivity index equals the
    *constant* ``tau`` at this frequency. Stationarity gives ``w(lam) = mu (I - lam D)^-1 c``
    with ``mu = 1/(c^H (I - lam D)^-1 c)``; the scalar ``lam`` is the root of
    ``w(lam)^H D w(lam) = 0`` nearest 0, bracketed between the pole reciprocals
    ``1/lambda_min(D) < 0 < 1/lambda_max(D)`` (where ``I - lam D`` stays positive definite).

    Parameters
    ----------
    A, R : np.ndarray
        ``[M, M]`` accept / reject covariance.
    c : np.ndarray
        ``[M]`` look vector (house convention).
    tau : float
        The fixed constant directivity factor (must satisfy ``tau_min < tau < tau_max`` so
        ``D`` is indefinite and a real root exists).

    Returns
    -------
    np.ndarray
        ``w[M]`` complex128 (GDI ``== tau`` by construction, distortionless).

    Raises
    ------
    ValueError
        If ``D`` is not indefinite at ``tau`` (no valid constant-DI solution there).
    """
    from scipy.optimize import brentq

    m = A.shape[0]
    d = A - tau * R  # [M, M] Hermitian, indefinite for tau in (tau_min, tau_max)
    ev = np.linalg.eigvalsh(d)  # ascending real
    if ev[0] >= 0 or ev[-1] <= 0:
        raise ValueError(
            f"tau={tau} does not make A - tau R indefinite (eig range [{ev[0]:.3g}, "
            f"{ev[-1]:.3g}]); no constant-DI solution at this frequency."
        )

    def w_of(lam: float) -> np.ndarray:
        x = np.linalg.solve(np.eye(m) - lam * d, c)
        return x / (np.conj(c) @ x)

    def quad(lam: float) -> float:
        w = w_of(lam)
        return float(np.real(np.conj(w) @ d @ w))

    lo = (1.0 / ev[0]) * (1.0 - 1e-9)  # just inside the negative pole
    hi = (1.0 / ev[-1]) * (1.0 - 1e-9)  # just inside the positive pole
    lam = brentq(quad, lo, hi, xtol=1e-15, rtol=1e-13)
    return w_of(lam)


def luo_mecd(A: np.ndarray, R: np.ndarray, tau: float) -> np.ndarray:
    """Luo MECD (max-efficiency constant-directivity) QCQP at fixed tau (DEFERRED).

    MECD maximizes ``w^H A w`` under ``w^H D w = 0, ||w|| = 1`` via projected ascent over the
    quadric ``w^H D w = 0``. Deferred: the constant-DI capability is provided by MSCD
    (distortionless, closed-form secular root); MECD's quadric projection is a follow-up.
    """
    raise NotImplementedError("MECD is deferred; use luo_mscd for constant directivity.")
