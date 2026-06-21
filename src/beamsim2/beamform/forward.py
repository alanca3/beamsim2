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


def beamwidth_deg(
    P: np.ndarray, directions, *, level_db: float = -6.0, order: int | None = None
) -> np.ndarray:
    """-``level_db`` beamwidth (degrees) through each frequency's main lobe.

    For each frequency the main-lobe axis is the peak-magnitude direction; the field is
    SH-resampled onto a fine great-circle arc through that axis, and the angular span where
    the level stays above ``level_db`` (relative to the peak) about angle 0 is returned.

    Parameters
    ----------
    P : np.ndarray
        ``[F, N]`` complex128 steered field.
    directions : ObservationPoints
        The grid ``P`` lives on.
    level_db : float
        Threshold relative to the peak (default -6 dB).
    order : int | None
        SH order for the resample (default: a safe order for the grid).

    Returns
    -------
    np.ndarray
        ``[F]`` beamwidth in degrees (``nan`` if the lobe is not closed within +/-180 deg).
    """
    from beamsim2.core.sh_transform import great_circle_arc, resample, safe_order_for_grid

    n = directions.unit_vectors.shape[0]
    # Cap the resample order: order ~16 resolves typical loudspeaker lobes and keeps the
    # per-frequency least-squares fit fast. Callers needing finer resolution pass `order`.
    sh_order = order if order is not None else min(safe_order_for_grid(n), 16)
    out = np.full(P.shape[0], np.nan)
    for f in range(P.shape[0]):
        peak_idx = int(np.argmax(np.abs(P[f])))
        axis = directions.unit_vectors[peak_idx]
        angle, arc_uv = great_circle_arc(axis, n_points=721)  # angle in [-pi, pi]
        p_arc = resample(P[f], directions, arc_uv, sh_order)  # [721]
        level = 20.0 * np.log10(np.abs(p_arc) / np.max(np.abs(p_arc)) + 1e-300)
        mid = int(np.argmin(np.abs(angle)))  # angle ~ 0 (the main-lobe axis)
        # Walk outward from the axis until the level drops below threshold each side.
        lo = mid
        while lo > 0 and level[lo] >= level_db:
            lo -= 1
        hi = mid
        while hi < len(level) - 1 and level[hi] >= level_db:
            hi += 1
        if lo == 0 or hi == len(level) - 1:
            continue  # lobe not closed within the arc -> leave nan
        out[f] = np.rad2deg(angle[hi] - angle[lo])
    return out


def directivity_metrics(
    P: np.ndarray, directions, b_target: np.ndarray | None = None, *, with_beamwidth: bool = True
) -> dict:
    """Achieved-pattern metrics per frequency (Stage P2-1).

    Parameters
    ----------
    P : np.ndarray
        ``[F, N]`` complex128 steered field.
    directions : ObservationPoints
        The grid (provides quadrature weights).
    b_target : np.ndarray | None
        ``[F, N]`` complex target; if given, a quadrature-weighted magnitude error
        (``target_error_db``) is reported.
    with_beamwidth : bool
        Whether to compute the (more expensive) -6 dB beamwidth.

    Returns
    -------
    dict
        ``di_db[F]`` and optionally ``beamwidth_deg[F]`` and ``target_error_db[F]``.
    """
    from beamsim2.validation.closed_loop import field_agreement_db
    from beamsim2.validation.power_di import directivity_index

    metrics: dict = {"di_db": directivity_index(P, directions.weights)}
    if with_beamwidth:
        metrics["beamwidth_deg"] = beamwidth_deg(P, directions)
    if b_target is not None:
        metrics["target_error_db"] = field_agreement_db(P, b_target, directions.weights)
    return metrics
