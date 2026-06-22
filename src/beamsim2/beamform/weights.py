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


def ls_bricks(
    H_f: np.ndarray, b_f: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-frequency LS normal-equation bricks ``A_f`` and ``rhs_f`` (house convention).

    ``A_f = conj(H_f) W H_f^T`` (Hermitian PSD) and ``rhs_f = conj(H_f) W b_f``, with
    ``W = diag(weights)``. These are exactly the matrices :func:`ls_pressure_match` forms
    internally; factored out so the per-bin and frequency-coupled solvers share one
    definition.

    Parameters
    ----------
    H_f : np.ndarray
        ``[M, N]`` complex128 — per-driver field at one frequency.
    b_f : np.ndarray
        ``[N]`` complex128 — desired pressure pattern.
    weights : np.ndarray
        ``[N]`` float64 — quadrature weights.

    Returns
    -------
    A : np.ndarray
        ``[M, M]`` complex128 Hermitian PSD normal matrix.
    rhs : np.ndarray
        ``[M]`` complex128 right-hand side.
    """
    cw = np.conj(H_f) * weights[None, :]  # conj(H_f) W   [M, N]
    return cw @ H_f.T, cw @ b_f  # ([M, M], [M])


def _second_difference_gram(n_f: int) -> np.ndarray:
    """Gram matrix ``D^T D`` of the second-difference operator ``D`` (``[F, F]``, real).

    ``D`` is ``[(F-2) x F]`` with rows ``[.. 1 -2 1 ..]``; penalizing ``||D x||^2``
    penalizes curvature of ``x`` across frequency. Returns zeros for ``F < 3`` (no
    interior bin to curve), so the coupled solve degrades to independent per-bin solves.
    """
    if n_f < 3:
        return np.zeros((n_f, n_f))
    d = np.zeros((n_f - 2, n_f))  # [(F-2), F]
    for r in range(n_f - 2):
        d[r, r], d[r, r + 1], d[r, r + 2] = 1.0, -2.0, 1.0
    return d.T @ d  # [F, F] pentadiagonal, real symmetric PSD


def ls_pressure_match_coupled(
    H: np.ndarray,
    b_field: np.ndarray,
    weights: np.ndarray,
    lam: np.ndarray,
    mu: float,
    freqs: np.ndarray,
    tau: float,
) -> np.ndarray:
    """Frequency-COUPLED LS pressure-match: smooth (realizable) weights ``w_m(f)``.

    Solves all ``F`` bins jointly as one block-diagonal LS system plus a per-driver
    second-difference smoothness penalty across frequency, working in a "tilde" variable
    with ONE shared modeling delay ``tau`` factored out (a common latency applied
    identically to every driver — cardinal-rule safe; it never alters inter-driver phase).
    Stacking is ``x[f*M + m] = wtil[m, f]`` (f-major, m-minor)::

        wtil[m, f] = w[m, f] * exp(-j 2 pi f tau)              (de-ramped variable)
        (A_f + lam_f I) wtil_f  +  mu * sum_fj (D^T D)[f, fj] wtil[:, fj]  =  ramp_f * rhs_f
        w[m, f] = conj(ramp_f) * wtil[m, f]                    (re-apply the shared ramp)

    where ``A_f, rhs_f`` are the per-bin bricks (:func:`ls_bricks`), ``ramp_f = exp(-j 2 pi
    f tau)``, and ``D^T D`` is the second-difference Gram (zero for ``F < 3``, so this
    reduces exactly to ``F`` independent :func:`ls_pressure_match` solves).

    NOTE (verified, ``docs/Chunk3a_Findings.md``): with the complex virtual-source target
    the per-bin weights are already smooth, so for a well-posed compact array this coupling
    is near-inert (``mu`` small) — it is robustness insurance for the under-determined /
    superdirective regimes hardened in 3b. ``mu`` must therefore be safe to leave small.

    Parameters
    ----------
    H : np.ndarray
        ``[M, F, N]`` complex128 — per-driver tensor.
    b_field : np.ndarray
        ``[F, N]`` complex128 — desired pattern per frequency.
    weights : np.ndarray
        ``[N]`` float64 — quadrature weights (sum = 4 pi).
    lam : np.ndarray
        ``[F]`` float64 — per-frequency Tikhonov / WNG-floor load (``>= 0``).
    mu : float
        Scale-invariant curvature weight (``>= 0``).
    freqs : np.ndarray
        ``[F]`` float64 Hz — used only to build the shared-delay ramp.
    tau : float
        Shared modeling delay (s), applied identically to all drivers.

    Returns
    -------
    np.ndarray
        ``w[M, F]`` complex128 physical weights.
    """
    m, n_f, _ = H.shape
    ramp = np.exp(-1j * 2.0 * np.pi * freqs * tau)  # [F] ONE scalar per f, all drivers
    dtd = _second_difference_gram(n_f)  # [F, F] real

    big = np.zeros((n_f * m, n_f * m), dtype=np.complex128)  # [FM, FM]
    g = np.zeros(n_f * m, dtype=np.complex128)  # [FM]
    for fi in range(n_f):
        a_f, rhs_f = ls_bricks(H[:, fi, :], b_field[fi], weights)  # [M,M], [M]
        sl = slice(fi * m, (fi + 1) * m)
        big[sl, sl] = a_f + lam[fi] * np.eye(m)
        g[sl] = ramp[fi] * rhs_f  # tilde RHS so the solution is wtil = ramp * w
    # Per-driver mu * (D^T D) curvature coupling across frequency (real, symmetric PSD).
    if mu > 0.0:
        for mm in range(m):
            for fi in range(n_f):
                row = fi * m + mm
                for fj in range(n_f):
                    if dtd[fi, fj] != 0.0:
                        big[row, fj * m + mm] += mu * dtd[fi, fj]
    wtil = np.linalg.solve(big, g).reshape(n_f, m).T  # [M, F] de-ramped weights
    return wtil * np.conj(ramp)[None, :]  # [M, F] undo the shared ramp


def phase_roughness(w: np.ndarray, freqs: np.ndarray, tau: float) -> float:
    """Cross-frequency filter roughness: ``max_m max |2nd-diff of unwrapped phase|`` (rad).

    A realizability proxy for the per-driver filters. One shared modeling delay ``tau`` is
    removed first (``wtil[m, f] = w[m, f] exp(-j 2 pi f tau)``) so a common latency — which
    is allowed and does not change the beam — is not counted as roughness. A small value
    means the weight phase bends gently across frequency, i.e. a short, causal, realizable
    filter.

    Parameters
    ----------
    w : np.ndarray
        ``[M, F]`` complex128 per-driver weights.
    freqs : np.ndarray
        ``[F]`` float64 Hz.
    tau : float
        Shared modeling delay to remove (s).

    Returns
    -------
    float
        The worst-driver max second difference of unwrapped phase (rad); ``0.0`` if ``F<3``.
    """
    ramp = np.exp(-1j * 2.0 * np.pi * freqs * tau)  # [F]
    wt = w * ramp[None, :]  # [M, F] shared delay removed
    worst = 0.0
    for mm in range(wt.shape[0]):
        phi = np.unwrap(np.angle(wt[mm]))  # [F]
        d2 = np.abs(np.diff(phi, n=2))  # [F-2]
        if d2.size:
            worst = max(worst, float(np.max(d2)))
    return worst


def align_global_phase(w: np.ndarray) -> np.ndarray:
    """Rotate each frequency column by ONE global phase for cross-frequency continuity.

    Cardinal-rule safe: multiplying *all* drivers in a frequency bin by the same complex
    unit ``exp(j theta_f)`` is a per-frequency global scale — it leaves ``|P(f, dir)|``, the
    inter-driver phase, ``|w|``, the directivity, the beamwidth and the white-noise gain
    *exactly* invariant (only the arbitrary per-bin global phase from the QCQP secular root /
    eigenvector sign changes). Aligning these spurious global phases for continuity removes the
    bulk of the apparent cross-frequency roughness of the constant-DI weights without touching
    the beam (``docs/Chunk3b_Findings.md`` corrected premise #3). Each column ``f >= 1`` is
    rotated to maximize ``Re<w[:, f-1], w[:, f]>`` (the 1-D Procrustes alignment to its
    neighbour); column 0 is left as-is.

    Parameters
    ----------
    w : np.ndarray
        ``[M, F]`` complex128 per-driver weights.

    Returns
    -------
    np.ndarray
        ``[M, F]`` complex128 — the same weights with continuity-aligned per-bin global phase.
    """
    out = w.copy()  # [M, F]
    for f in range(1, out.shape[1]):
        ip = np.vdot(out[:, f - 1], out[:, f])  # <w_{f-1}, w_f> complex
        if abs(ip) > 0.0:
            out[:, f] *= np.exp(-1j * np.angle(ip))  # rotate to max Re<w_f, w_{f-1}>
    return out


def choose_shared_delay_complex(w: np.ndarray, freqs: np.ndarray) -> float:
    """Pick ONE shared modeling delay ``tau`` (s) minimizing COMPLEX cross-frequency roughness.

    Unlike :func:`phase_roughness` (unwrapped-phase second difference, ill-defined through
    ``|w| = 0``), this scores a per-driver-RMS-normalized *complex* second difference, which is
    well behaved for tapered heterogeneous arrays where some drivers fall to silence. The chosen
    ``tau`` is a common latency applied identically to all drivers (cardinal-rule safe). Returns
    ``0.0`` for ``F < 3`` (nothing to smooth).

    Parameters
    ----------
    w : np.ndarray
        ``[M, F]`` complex128 per-driver weights (typically already global-phase aligned).
    freqs : np.ndarray
        ``[F]`` float64 Hz.

    Returns
    -------
    float
        The shared modeling delay (s).
    """
    n_f = w.shape[1]
    if n_f < 3:
        return 0.0

    def rough(tau: float) -> float:
        ramp = np.exp(-1j * 2.0 * np.pi * freqs * tau)  # [F]
        wt = w * ramp[None, :]  # [M, F] de-ramped
        d2 = np.diff(wt, n=2, axis=1)  # [M, F-2] complex
        scale = np.sqrt((np.abs(wt) ** 2).mean(axis=1)) + 1e-12  # [M] per-driver RMS
        return float(np.max(np.abs(d2) / scale[:, None]))

    span = 1.0 / (freqs[-1] - freqs[0])  # period scale of the band (s)
    cand = np.linspace(-1.5 * span, 1.5 * span, 601)  # candidate shared delays (s)
    return float(cand[int(np.argmin([rough(t) for t in cand]))])


def magnitude_gated_phase_roughness(
    w: np.ndarray, freqs: np.ndarray, tau: float, *, gate_frac: float = 0.10
) -> float:
    """Realizability roughness counting ONLY drivers that are acoustically active.

    The raw :func:`phase_roughness` overcounts for tapered arrays: a driver whose weight decays
    to near zero contributes meaningless phase noise (an essentially-silent filter is trivially
    realizable). This variant ignores, per second-difference stencil, any driver whose ``|w|``
    is below ``gate_frac`` of the overall peak across the three stencil bins, then reports the
    worst remaining unwrapped-phase second difference (after removing one shared delay ``tau``).
    This is the honest gate metric for the constant-DI engine (``docs/Chunk3b_Findings.md``).

    Parameters
    ----------
    w : np.ndarray
        ``[M, F]`` complex128 per-driver weights.
    freqs : np.ndarray
        ``[F]`` float64 Hz.
    tau : float
        Shared modeling delay to remove (s).
    gate_frac : float
        A driver is "on" at a stencil if its smallest of the three ``|w|`` exceeds
        ``gate_frac * peak`` (default 0.10 = 10% of the peak weight magnitude).

    Returns
    -------
    float
        Worst on-driver max second difference of unwrapped phase (rad); ``0.0`` if ``F < 3``.
    """
    ramp = np.exp(-1j * 2.0 * np.pi * freqs * tau)  # [F]
    wt = w * ramp[None, :]  # [M, F]
    peak = float(np.abs(wt).max())
    if peak == 0.0 or wt.shape[1] < 3:
        return 0.0
    worst = 0.0
    for mm in range(wt.shape[0]):
        amp = np.abs(wt[mm])  # [F]
        phi = np.unwrap(np.angle(wt[mm]))  # [F]
        d2 = np.abs(np.diff(phi, n=2))  # [F-2]
        wmin = np.minimum.reduce([amp[:-2], amp[1:-1], amp[2:]])  # [F-2] min |w| over stencil
        on = wmin > gate_frac * peak  # [F-2] acoustically-active stencils
        if on.any():
            worst = max(worst, float(np.max(d2[on])))
    return worst


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
