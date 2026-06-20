"""Top-level designer orchestrator: (RadiationDataset, TargetSpec) -> DesignResult.

Stages P2-1 (LS / pressure-matching, MVDR/LCMV) and P2-2 (Luo constant-DI). This is the
single entry point the GUI worker and the V-tests call. It assembles the look vector /
covariance per frequency (house convention), dispatches to the requested engine in
:mod:`beamsim2.beamform.weights`, applies the WNG-floor robustness loading
(:mod:`beamsim2.beamform.regularize`), evaluates the achieved field and metrics
(:mod:`beamsim2.beamform.forward`), and returns a :class:`DesignResult`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from beamsim2.beamform.targets import TargetSpec


@dataclass
class DesignResult:
    """The output of a beamforming design.

    Attributes
    ----------
    weights : np.ndarray
        ``[M, F]`` complex128 — per-driver complex weights ``w_m(f)``.
    steered_field : np.ndarray
        ``[F, N]`` complex128 — achieved ``P = sum_m w_m H_m``.
    metrics : dict
        Per-frequency arrays: ``di_db[F]``, ``beamwidth_deg[F]``, ``wng_db[F]``,
        ``target_error_db[F]``, ``feasible_mask[F]`` (bool).
    spec : TargetSpec
        The request, echoed back.
    attrs : dict
        Provenance (engine, convention, speed of sound used, etc.).
    """

    weights: np.ndarray
    steered_field: np.ndarray
    metrics: dict
    spec: TargetSpec
    attrs: dict = field(default_factory=dict)


def design(ds, spec: TargetSpec) -> DesignResult:
    """Design per-driver weights for ``spec`` against the dataset ``ds`` (Stages P2-1/2).

    Parameters
    ----------
    ds : RadiationDataset
        Phase-1 output (carries ``H``, the sphere grid + weights, driver positions,
        and the speed of sound used by the BEM).
    spec : TargetSpec
        The user request.

    Returns
    -------
    DesignResult
    """
    raise NotImplementedError("Stages P2-1/P2-2: designer orchestration not yet implemented.")
