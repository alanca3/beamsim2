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

from beamsim2.beamform.covariance import look_vector


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
    """
    raise NotImplementedError("Stage P2-1: LS/pressure-matching not yet implemented.")


def mvdr(H_f: np.ndarray, look_idx: int, weights: np.ndarray, eps: float) -> np.ndarray:
    """MVDR (minimum-variance distortionless response), loaded (Stage P2-1).

    ``w = (R+eps I)^-1 c / (c^H (R+eps I)^-1 c)``, ``c = conj(H_f[:, look])``.
    """
    raise NotImplementedError("Stage P2-1: MVDR not yet implemented.")


def lcmv(
    H_f: np.ndarray,
    look_idx: int,
    null_idx: list[int],
    weights: np.ndarray,
    eps: float,
) -> np.ndarray:
    """LCMV with hard nulls (Stage P2-1). ``w = R^-1 C (C^H R^-1 C)^-1 g``."""
    raise NotImplementedError("Stage P2-1: LCMV not yet implemented.")


def luo_mscd(A: np.ndarray, R: np.ndarray, c: np.ndarray, tau: float) -> np.ndarray:
    """Luo MSCD (max-sensitivity constant-directivity) QCQP at fixed tau (Stage P2-2)."""
    raise NotImplementedError("Stage P2-2: Luo MSCD not yet implemented.")


def luo_mecd(A: np.ndarray, R: np.ndarray, tau: float) -> np.ndarray:
    """Luo MECD (max-efficiency constant-directivity) QCQP at fixed tau (Stage P2-2)."""
    raise NotImplementedError("Stage P2-2: Luo MECD not yet implemented.")
