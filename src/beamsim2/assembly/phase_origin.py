"""Single-phase-origin discipline: residual metric and assertion guardrail.

§3.4 cardinal rule: every per-driver H_bem must be referenced to the single global
coordinate origin (0,0,0) — never re-zeroed, never minimum-phase-ified per-driver.
The inter-driver phase differences ARE the beamforming steering information.

This module provides:
  - ``superposition_residual``: computes agreement metrics between a superposed field
    (sum of per-driver solves) and a direct multi-driver BEM solve.
  - ``assert_superposition_matches``: raises loudly if the residual exceeds tolerance.

Because all three V-5 solves use the *identical* mesh and frequency grid, the BEM
system matrix is the same — only the RHS (velocity BC) differs.  Linear superposition
therefore holds EXACTLY up to iterative solver residual (~1e-5 – 1e-6).  A tolerance
of rtol=1e-3 is a sharp guardrail with no false-failure risk.  Any re-zeroing bug
produces order-1 mismatch and is caught immediately.
VERIFIED: BEM linearity (Kreuzer et al. 2024); residual analysis INFERRED from
standard constant-collocation BEM iterative convergence rates.

Risk mitigations:
  - R-02 (phase-reference discipline silently violated → mis-steered beams).
"""

from __future__ import annotations

import numpy as np


def superposition_residual(
    summed: np.ndarray,
    direct: np.ndarray,
) -> dict[str, float]:
    """Compute agreement metrics between superposed and direct multi-driver fields.

    Parameters
    ----------
    summed : np.ndarray
        Sum of per-driver BEM fields from ``superpose_fields()``.
        Shape ``[F × N]`` complex128.
    direct : np.ndarray
        Pressure from a direct multi-driver BEM solve (all vibrating simultaneously).
        Shape ``[F × N]`` complex128, same shape as ``summed``.

    Returns
    -------
    dict with keys:
        ``relative_l2`` : float
            Relative L2 residual ``||summed − direct|| / ||direct||``.
            Zero for identical fields; ~1 for totally uncorrelated fields.
        ``max_abs_db`` : float
            Maximum pointwise magnitude difference in dB
            ``max |20 log10(|summed| / |direct|)|``.
        ``max_phase_deg`` : float
            Maximum pointwise phase difference in degrees.

    Raises
    ------
    ValueError
        If shapes do not match.
    """
    if summed.shape != direct.shape:
        raise ValueError(
            f"superposition_residual: shape mismatch — summed {summed.shape} "
            f"vs direct {direct.shape}"
        )

    diff = summed.astype(np.complex128) - direct.astype(np.complex128)

    norm_direct = np.linalg.norm(direct)
    if norm_direct == 0.0:
        relative_l2 = 0.0 if np.linalg.norm(diff) == 0.0 else float("inf")
    else:
        relative_l2 = float(np.linalg.norm(diff) / norm_direct)

    # magnitude error in dB — guard against zeros
    abs_summed = np.abs(summed)
    abs_direct = np.abs(direct)
    mask = abs_direct > 0.0
    if not np.any(mask):
        max_abs_db = 0.0
    else:
        ratio = np.where(mask, abs_summed / (abs_direct + 1e-300), 1.0)
        max_abs_db = float(np.max(np.abs(20.0 * np.log10(ratio[mask] + 1e-300))))

    # phase error in degrees
    phase_diff = np.angle(summed) - np.angle(direct)
    # wrap to [-π, π]
    phase_diff = (phase_diff + np.pi) % (2 * np.pi) - np.pi
    max_phase_deg = float(np.degrees(np.max(np.abs(phase_diff))))

    return {
        "relative_l2": relative_l2,
        "max_abs_db": max_abs_db,
        "max_phase_deg": max_phase_deg,
    }


def assert_superposition_matches(
    summed: np.ndarray,
    direct: np.ndarray,
    rtol: float = 1e-3,
) -> None:
    """Assert that superposed and direct multi-driver fields agree within tolerance.

    V-5 guardrail (§7, R-02): any code path that re-zeros a driver independently
    will produce order-1 relative_l2 and fail loudly here.

    Parameters
    ----------
    summed : np.ndarray
        Sum of per-driver fields from ``superpose_fields()``.
        Shape ``[F × N]`` complex128.
    direct : np.ndarray
        Direct multi-driver BEM solve pressure.
        Shape ``[F × N]`` complex128.
    rtol : float
        Relative L2 tolerance (default 1e-3).  All three V-5 solves share an
        identical BEM system matrix, so agreement is expected at ~1e-5.
        Anything looser than ~1e-4 in practice signals a real problem — most
        likely the multi-group BC writer (``_group_element_runs`` in
        ``ncinp_writer``).  Stop and report.

    Raises
    ------
    AssertionError
        If ``relative_l2 > rtol``, with a plain-English diagnostic message
        including the three metrics.
    """
    metrics = superposition_residual(summed, direct)
    if metrics["relative_l2"] > rtol:
        raise AssertionError(
            f"Phase-origin / superposition check FAILED — single-phase-origin "
            f"discipline (§3.4) is violated.\n"
            f"  relative_l2  = {metrics['relative_l2']:.3e}  (limit {rtol:.1e})\n"
            f"  max |dB|     = {metrics['max_abs_db']:.2f} dB\n"
            f"  max phase Δ  = {metrics['max_phase_deg']:.1f}°\n"
            f"If this is looser than ~1e-4, check the multi-group BC writer "
            f"(ncinp_writer._group_element_runs) before suspecting assembly code."
        )
