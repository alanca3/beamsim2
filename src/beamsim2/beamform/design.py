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

from beamsim2.assembly.tensor import stacked_h_full
from beamsim2.beamform.covariance import covariance, look_vector
from beamsim2.beamform.forward import directivity_metrics, steered_field
from beamsim2.beamform.regularize import (
    lambda_for_ls,
    loaded_mvdr_weights,
    solve_loading_for_wng,
    white_noise_gain_db,
)
from beamsim2.beamform.targets import TargetSpec, build_target
from beamsim2.beamform.weights import (
    lcmv,
    ls_pressure_match,
    luo_mscd,
    matched_field,
    max_directivity,
)


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
    h = stacked_h_full(ds)  # [M, F, N] complex128
    m, n_f, _ = h.shape
    obs = ds.directions
    w_quad = obs.weights  # [N]
    c_sound = float(ds.attrs.get("speed_of_sound", 343.2))

    target = build_target(spec, obs, ds.frequencies)
    look = target.look_idx

    wng_ceiling_db = 10.0 * np.log10(m)  # the M-driver delay-and-sum WNG ceiling
    # Single robustness knob (wng_floor_db) -> an LS Tikhonov fraction s in [0, 1].
    s_ls = float(np.clip((spec.wng_floor_db + 20.0) / (wng_ceiling_db + 20.0), 0.0, 1.0))

    weights = np.zeros((m, n_f), dtype=np.complex128)  # [M, F]
    wng_db = np.zeros(n_f)  # [F]
    feasible = np.ones(n_f, dtype=bool)  # [F]

    # Constant-DI (engine #2) is a two-pass: pass 1 finds each frequency's directivity
    # ceiling tau_max; the constant target tau* is the min over frequency (so it is feasible
    # everywhere) capped by any requested target_gdi_db. Pass 2 runs MSCD at that fixed tau*.
    tau_star: float | None = None
    if spec.engine == "constant_di":
        tau_maxes = []
        for f in range(n_f):
            a_mat = covariance(h[:, f, :], w_quad, mask=target.accept_mask)
            r_mat = covariance(h[:, f, :], w_quad, mask=target.reject_mask)
            _, tau_max = max_directivity(a_mat, r_mat)
            tau_maxes.append(tau_max)
        ceiling = float(np.min(tau_maxes)) * 0.98  # just below the min ceiling -> always feasible
        if spec.target_gdi_db is not None:
            ceiling = min(ceiling, 10.0 ** (spec.target_gdi_db / 10.0))
        tau_star = ceiling

    for f in range(n_f):
        h_f = h[:, f, :]  # [M, N]
        c = look_vector(h_f, look)  # [M]
        if spec.engine == "delay_sum":
            w_f = matched_field(h_f, look)
        elif spec.engine == "ls":
            a_mat = (np.conj(h_f) * w_quad[None, :]) @ h_f.T  # [M, M]
            lam = lambda_for_ls(s_ls, a_mat)
            w_f = ls_pressure_match(h_f, target.b_field[f], w_quad, lam)
        elif spec.engine == "mvdr":
            r = covariance(h_f, w_quad)
            eps, feas = solve_loading_for_wng(r, c, spec.wng_floor_db)
            w_f = loaded_mvdr_weights(r, c, eps)
            feasible[f] = feas
        elif spec.engine == "lcmv":
            r = covariance(h_f, w_quad)
            eps, feas = solve_loading_for_wng(r, c, spec.wng_floor_db)
            w_f = lcmv(h_f, look, target.null_idx, w_quad, eps)
            feasible[f] = feas
        elif spec.engine == "max_directivity":
            a_mat = covariance(h_f, w_quad, mask=target.accept_mask)
            r_mat = covariance(h_f, w_quad, mask=target.reject_mask)
            w_f, _ = max_directivity(a_mat, r_mat, c=c)
        elif spec.engine == "constant_di":
            a_mat = covariance(h_f, w_quad, mask=target.accept_mask)
            r_mat = covariance(h_f, w_quad, mask=target.reject_mask)
            try:
                w_f = luo_mscd(a_mat, r_mat, c, tau_star)
            except ValueError:
                # tau* not feasible at this bin (edge of band) -> fall back to max directivity.
                w_f, _ = max_directivity(a_mat, r_mat, c=c)
                feasible[f] = False
        else:
            raise ValueError(f"Unknown engine {spec.engine!r}.")
        weights[:, f] = w_f
        wng_db[f] = white_noise_gain_db(w_f, c)

    p = steered_field(h, weights)  # [F, N]
    metrics = directivity_metrics(p, obs, target.b_field, with_beamwidth=True)
    metrics["wng_db"] = wng_db
    # A bin is feasible if the solver met the WNG floor (within 1 dB) and the engine flag held.
    metrics["feasible_mask"] = feasible & (wng_db >= spec.wng_floor_db - 1.0)

    attrs = {
        "engine": spec.engine,
        "convention": "house: P = sum_m w_m * H_m  (c = conj(H_look))",
        "speed_of_sound": c_sound,
        "wng_floor_db": spec.wng_floor_db,
        "look_idx": look,
    }
    if tau_star is not None:
        attrs["constant_gdi_db"] = 10.0 * np.log10(tau_star)
    return DesignResult(weights=weights, steered_field=p, metrics=metrics, spec=spec, attrs=attrs)
