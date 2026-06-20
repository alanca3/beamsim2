"""Forward model and achieved-directivity metrics (Stage P2-1).

The forward model (AES GLL complex summation) already exists as
:func:`beamsim2.validation.closed_loop.steer_response`; this module re-exports it
under a Phase-2 name and adds achieved-pattern metrics (directivity index, -6 dB
beamwidth, target error) used by the designer and the V-tests.
"""

from __future__ import annotations

import numpy as np

from beamsim2.validation.closed_loop import steer_response

__all__ = ["steered_field", "steer_response", "directivity_metrics"]


def steered_field(H: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Achieved field ``P[F, N] = sum_m weights[m, f] * H[m, f, :]``.

    Thin alias for :func:`beamsim2.validation.closed_loop.steer_response` so Phase-2
    code reads in beamformer terms.

    Parameters
    ----------
    H : np.ndarray
        ``[M, F, N]`` complex128 per-driver tensor.
    weights : np.ndarray
        ``[M, F]`` complex128 weights.

    Returns
    -------
    np.ndarray
        ``[F, N]`` complex128 steered field.
    """
    return steer_response(H, weights)  # [F, N] complex128


def directivity_metrics(P: np.ndarray, directions, b_target: np.ndarray | None = None) -> dict:
    """Achieved-pattern metrics per frequency (Stage P2-1).

    Computes directivity index (Lebedev quadrature), -6 dB beamwidth, and (if a target
    is supplied) the quadrature-weighted magnitude error vs the target.
    """
    raise NotImplementedError("Stage P2-1: directivity metrics not yet implemented.")
