"""Look vector and weighted complex covariance — the house convention (DR-P2-02).

The coded forward model is ``P(f, dir) = sum_m w_m(f) * H[m, f, dir]``. To make every
solver's weights drop into that sum with no extra conjugation, we pin:

* look vector   ``c[m] = conj(H[m, f, look])``
* covariance    ``R[m, m'] = sum_n a_n * conj(H[m, f, n]) * H[m', f, n]``
                ``= conj(H_f) @ diag(a) @ H_f.T``   (Hermitian PSD, [M x M])

where ``a_n`` are the Lebedev quadrature weights (sum = 4*pi). The microphone-array
literature uses the conjugate convention (``R = sum a_n H_n H_n^H``, ``d = H_look``);
copied verbatim it mirror-steers the beam. The round-trip steering test
(``tests/test_beamform_convention.py``) is the empirical arbiter.
"""

from __future__ import annotations

import numpy as np


def look_vector(H_f: np.ndarray, look_idx: int) -> np.ndarray:
    """Return the look (steering) vector for the house convention.

    Parameters
    ----------
    H_f : np.ndarray
        ``[M, N]`` complex128 — the per-driver field at one frequency.
    look_idx : int
        Index of the look direction in the sphere grid.

    Returns
    -------
    np.ndarray
        ``c[M]`` complex128 ``= conj(H_f[:, look_idx])``.
    """
    return np.conj(H_f[:, look_idx])  # [M] complex128


def covariance(H_f: np.ndarray, weights: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Weighted complex covariance over the sphere (house convention).

    ``R = conj(H_f) @ diag(a * mask) @ H_f.T`` — Hermitian PSD ``[M, M]``.

    Parameters
    ----------
    H_f : np.ndarray
        ``[M, N]`` complex128 — per-driver field at one frequency.
    weights : np.ndarray
        ``[N]`` float64 — Lebedev quadrature weights (sum = 4*pi).
    mask : np.ndarray | None
        Optional ``[N]`` float64 region weighting (e.g. accept/reject); multiplies
        the quadrature weights. ``None`` -> whole sphere.

    Returns
    -------
    np.ndarray
        ``R[M, M]`` complex128, Hermitian PSD.
    """
    a = weights if mask is None else weights * mask  # [N]
    # conj(H_f) @ diag(a) @ H_f.T  ==  (conj(H_f) * a) @ H_f.T
    R = (np.conj(H_f) * a[None, :]) @ H_f.T  # [M, M] complex128
    # Hermitize to kill round-off asymmetry.
    return 0.5 * (R + R.conj().T)


def directivity_factor(w: np.ndarray, A: np.ndarray, R: np.ndarray) -> float:
    """Generalized Rayleigh quotient ``w^H A w / w^H R w`` (linear directivity factor).

    Parameters
    ----------
    w : np.ndarray
        ``[M]`` complex128 weights.
    A, R : np.ndarray
        ``[M, M]`` complex128 accept / reject covariance.

    Returns
    -------
    float
        ``(w^H A w) / (w^H R w)`` (real; the GDI in dB is ``10*log10`` of this).
    """
    num = float(np.real(np.conj(w) @ A @ w))
    den = float(np.real(np.conj(w) @ R @ w))
    return num / den
