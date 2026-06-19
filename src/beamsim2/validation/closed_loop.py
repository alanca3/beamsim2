"""Stage-4 close-the-loop validation: delay-and-sum beamforming from H tensor.

Demonstrates that the per-driver H[driver × frequency × direction] tensor can
drive a simple delay-and-sum beamformer to reproduce a known directivity null.
This is the §8 Stage-4 gate — the end-to-end proof that the single-phase-origin
contract (§3.4) correctly steers a beam.

Physics: two monopole-like sources at positions p_A and p_B. Applying the weight
    w_m(f) = exp(+j · k · (p_m · û))
aligns all sources coherently in direction û (the beam direction). The combined
response has a null in the opposite direction -û at the design frequency c/(4d),
where d is the inter-source spacing along û.

HEURISTIC: a single-frequency end-fire null is NOT "constant directivity across
frequency" — it is a two-element end-fire pattern at the design frequency.
The gate's intent (prove the contract steers a beam) is fully met at this level.
Broadband constant-directivity beamforming belongs to Phase 2.

This module is a *validation check*, not a Phase-2 beamformer. Keep it that way.

References
----------
Benesty, Cohen, Chen, *Microphone Array Signal Processing*, Springer, 2008, §2.3.
Van Trees, *Optimum Array Processing*, Wiley-Interscience, 2002.
DATA_CONTRACT.md §3.4 (single-phase-origin rule).
BEAMSIMII_Gameplan.md §8 Stage-4 gate.
"""

from __future__ import annotations

import numpy as np

from beamsim2.core.types import ObservationPoints


def monopole_field(
    positions: np.ndarray,
    obs: ObservationPoints,
    frequencies: np.ndarray,
    c: float = 343.2,
) -> np.ndarray:
    """Analytic point-monopole field in the engineering convention (exp(+jkr)).

    Computes unit-amplitude spherical wave pressure from each point source:

        H_analytic[m, f, n] = exp(+j · k_f · r_mn) / r_mn

    where r_mn = |R_n − p_m| (metres), k_f = 2π f / c (rad/m), and R_n is
    the Cartesian position of observation point n.

    Convention: exp(−jωt) time factor, outgoing waves ~ exp(+jkr). This is
    the NumCalc engineering convention (CLAUDE.md; VERIFIED for single-source
    case by V-2 sphere benchmark).

    Amplitude constants (ρ, c, source strength) are omitted — only spatial phase
    shapes matter for the directivity / null test.

    Parameters
    ----------
    positions : np.ndarray, shape [M, 3], float64
        Source positions in metres.
    obs : ObservationPoints
        Sphere grid; carries ``unit_vectors`` [N, 3] float64 and ``radius`` (m).
    frequencies : np.ndarray, shape [F], float64
        Frequencies in Hz.
    c : float
        Speed of sound in m/s.

    Returns
    -------
    np.ndarray, shape [M, F, N], complex128
        H_analytic[m, f, n] — unit point-monopole pressure at each
        (source, frequency, observation-direction) triple.
    """
    R_obs = obs.unit_vectors * obs.radius  # [N, 3] observation point coordinates (m)
    k = 2.0 * np.pi * frequencies / c  # [F] wavenumber (rad/m)

    M = positions.shape[0]
    F = len(frequencies)
    N = obs.unit_vectors.shape[0]

    H = np.empty((M, F, N), dtype=np.complex128)
    for m, p_m in enumerate(positions):
        r_mn = np.linalg.norm(R_obs - p_m[None, :], axis=1)  # [N] distances (m)
        # [F, N] via outer product of k[F] and r_mn[N]
        H[m] = np.exp(1j * k[:, None] * r_mn[None, :]) / r_mn[None, :]  # [F, N]

    return H  # [M, F, N] complex128


def delay_sum_weights(
    positions: np.ndarray,
    steer_dir: np.ndarray,
    frequencies: np.ndarray,
    c: float = 343.2,
) -> np.ndarray:
    """Delay-and-sum weights to steer coherently toward ``steer_dir``.

    For each source m at position p_m, the weight is:

        w_m(f) = exp(+j · k_f · (p_m · û))

    where û = steer_dir and k_f = 2π f / c.

    This *advances* source m by (p_m · û)/c so that all sources arrive in phase
    at the far-field in direction û. The null forms in direction −û at frequency
    f = c / (4 d), where d is the inter-source spacing projected along û.

    In acoustics terms: each source contributes a propagation delay proportional
    to its projection onto the beam axis; the weight pre-compensates for that
    delay so all contributions add constructively in û (and destructively in −û).

    VERIFIED consistent with engineering convention exp(+jkr) — a source at p_m
    contributes exp(+jk(R − p_m · r̂))/R in the far field; the weight removes
    the exp(−jk(p_m · û)) term to align phases in direction û.

    Parameters
    ----------
    positions : np.ndarray, shape [M, 3], float64
        Source positions in metres.
    steer_dir : np.ndarray, shape [3], float64
        Unit beam direction vector.
    frequencies : np.ndarray, shape [F], float64
        Frequencies in Hz.
    c : float
        Speed of sound in m/s.

    Returns
    -------
    np.ndarray, shape [M, F], complex128
        Delay-and-sum weights. Row m, column f = w_m(f).
    """
    k = 2.0 * np.pi * frequencies / c  # [F] wavenumber (rad/m)
    proj = positions @ steer_dir  # [M] projection of each source onto û
    # w[m, f] = exp(+j * k[f] * proj[m])
    return np.exp(1j * k[None, :] * proj[:, None])  # [M, F] complex128


def steer_response(
    H_bem: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Apply delay-and-sum weights to H tensor; return steered field [F, N].

    Implements the delay-and-sum sum:

        P_steered[f, n] = Σ_m  w_m[f] · H_bem[m, f, n]

    Parameters
    ----------
    H_bem : np.ndarray, shape [M, F, N], complex128
        Per-driver pressure tensor. Must have the same M and F as ``weights``.
    weights : np.ndarray, shape [M, F], complex128
        Delay-and-sum weights from :func:`delay_sum_weights`.

    Returns
    -------
    np.ndarray, shape [F, N], complex128
        Steered complex pressure field.
    """
    # weights [M, F, 1] broadcast over N → sum over M → [F, N]
    return np.sum(weights[:, :, None] * H_bem, axis=0)  # [F, N] complex128


def null_depth_db(
    P_steered: np.ndarray,
    lebedev_weights: np.ndarray,
) -> np.ndarray:
    """Null depth: minimum-to-maximum intensity ratio in dB, per frequency.

    A deep cardioid null gives a large negative value (e.g. −30 dB).
    An omnidirectional (unsteered) pattern → ~ 0 dB.

    Parameters
    ----------
    P_steered : np.ndarray, shape [F, N], complex128
        Steered pressure field from :func:`steer_response`.
    lebedev_weights : np.ndarray, shape [N], float64
        Lebedev quadrature weights (sum = 4π); used only for correct averaging
        in future extensions. The null depth itself uses min/max over points.

    Returns
    -------
    np.ndarray, shape [F], float64
        10 · log10(I_min / I_max) per frequency (dB, ≤ 0).
    """
    intensity = np.abs(P_steered) ** 2  # [F, N]
    i_min = np.min(intensity, axis=1)  # [F]
    i_max = np.max(intensity, axis=1)  # [F]
    return 10.0 * np.log10(i_min / i_max)  # [F] dB (≤ 0)


def field_agreement_db(
    P_bem: np.ndarray,
    P_analytic: np.ndarray,
    lebedev_weights: np.ndarray,
) -> np.ndarray:
    """RMS magnitude difference between two steered fields, dB per frequency.

    Computes the quadrature-weighted RMS of (|P_bem| − |P_analytic|) in dB
    across all observation directions. Both fields are first normalised to their
    respective on-sphere mean intensities so that the comparison is of *shape*
    (directivity pattern) not absolute level.

    Parameters
    ----------
    P_bem : np.ndarray, shape [F, N], complex128
        Steered BEM field.
    P_analytic : np.ndarray, shape [F, N], complex128
        Steered analytic reference field.
    lebedev_weights : np.ndarray, shape [N], float64
        Lebedev quadrature weights (sum = 4π).

    Returns
    -------
    np.ndarray, shape [F], float64
        Quadrature-weighted RMS magnitude difference in dB per frequency.
    """
    # Normalise each field to its sphere-average power = 1 (shape-only comparison)
    w = lebedev_weights  # [N]

    def _norm(P: np.ndarray) -> np.ndarray:
        """[F, N] → normalised so mean |p|² over sphere = 1."""
        intensity = np.abs(P) ** 2  # [F, N]
        mean_pwr = np.sum(w * intensity, axis=1, keepdims=True) / (4.0 * np.pi)  # [F, 1]
        return P / np.sqrt(mean_pwr)

    P_b = _norm(P_bem)
    P_a = _norm(P_analytic)

    mag_b_db = 20.0 * np.log10(np.abs(P_b) + 1e-30)  # [F, N]
    mag_a_db = 20.0 * np.log10(np.abs(P_a) + 1e-30)  # [F, N]
    diff_db = mag_b_db - mag_a_db  # [F, N]

    # Quadrature-weighted RMS over directions
    rms_db = np.sqrt(np.sum(w * diff_db**2, axis=1) / (4.0 * np.pi))  # [F]
    return rms_db  # [F] dB
