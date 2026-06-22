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
    loaded_mvdr_weights,
    ls_wng_lambda_grid,
    max_white_noise_gain_db,
    solve_loading_for_wng,
    white_noise_gain_db,
)
from beamsim2.beamform.targets import TargetSpec, build_target
from beamsim2.beamform.weights import (
    lcmv,
    ls_bricks,
    ls_pressure_match,
    ls_pressure_match_coupled,
    luo_mscd,
    matched_field,
    max_directivity,
    phase_roughness,
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


def _choose_shared_delay(
    h: np.ndarray,
    b_field: np.ndarray,
    w_quad: np.ndarray,
    freqs: np.ndarray,
    a_list: np.ndarray,
) -> float:
    """Pick ONE shared modeling delay ``tau`` (s) that minimizes cross-frequency roughness.

    Solves the per-bin LS at a tiny load to get raw weights, then chooses the single shared
    delay (a common latency applied identically to all drivers -> cardinal-rule safe) that
    minimizes the worst-driver second-difference of the unwrapped weight phase. With the
    complex virtual-source target this comes out ~0, confirming the realizability win lives
    in the target, not the delay (``docs/Chunk3a_Findings.md``).
    """
    m, n_f, _ = h.shape
    w_raw = np.empty((m, n_f), dtype=np.complex128)  # [M, F] lightly-loaded per-bin weights
    for fi in range(n_f):
        lam0 = 1e-6 * float(np.real(np.trace(a_list[fi]))) / m
        w_raw[:, fi] = ls_pressure_match(h[:, fi, :], b_field[fi], w_quad, lam0)
    span = 1.0 / (freqs[-1] - freqs[0])  # period scale of the band
    cand = np.linspace(-1.5 * span, 1.5 * span, 301)  # candidate shared delays (s)
    rough = [phase_roughness(w_raw, freqs, float(t)) for t in cand]
    return float(cand[int(np.argmin(rough))])


def _design_ls_coupled(
    h: np.ndarray,
    b_field: np.ndarray,
    w_quad: np.ndarray,
    freqs: np.ndarray,
    look: int,
    wng_floor_db: float,
    *,
    frac_mu: float = 1e-2,
    n_grid: int = 48,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Frequency-coupled LS pressure-match with an honest per-frequency WNG floor.

    Fixes Chunk-3a defects #2 (uncoupled, ringy solves) and #3 (no honest LS WNG floor):

    1. Per bin, grid-search the Tikhonov load ``lam_f`` for the SMALLEST loading whose
       distortionless WNG meets ``wng_floor_db`` (LS WNG is non-monotone in ``lam``, so a
       grid search, not bisection). Bins whose WNG ceiling is below the floor are flagged
       infeasible and given the most-robust loading (graceful roll-off, never silent garbage).
    2. One final frequency-COUPLED solve (:func:`ls_pressure_match_coupled`) with those loads
       and a small second-difference smoothness penalty (a no-op for ``F<3``), yielding smooth
       realizable filters. The coupling is near-inert for a well-posed compact array (the
       target already gives smooth filters) and earns its keep in 3b's harder regimes.

    Returns ``(weights[M, F], feasible[F], lam[F], tau)``. The caller recomputes the achieved
    WNG from the final coupled weights, so any bin pushed below the floor by coupling is still
    flagged; ``lam`` and ``tau`` are recorded as provenance.
    """
    m, n_f, _ = h.shape
    a_list = np.empty((n_f, m, m), dtype=np.complex128)  # [F, M, M] per-bin normal matrices
    rhs_list = np.empty((n_f, m), dtype=np.complex128)  # [F, M]
    c_list = np.empty((n_f, m), dtype=np.complex128)  # [F, M] look vectors
    for fi in range(n_f):
        a_list[fi], rhs_list[fi] = ls_bricks(h[:, fi, :], b_field[fi], w_quad)
        c_list[fi] = look_vector(h[:, fi, :], look)

    tau = _choose_shared_delay(h, b_field, w_quad, freqs, a_list) if n_f >= 3 else 0.0
    mu = frac_mu * float(np.mean([np.real(np.trace(a_list[fi])) for fi in range(n_f)])) / 6.0

    # (1) per-bin WNG-floor search (cheap, uncoupled): smallest lam_f meeting the floor.
    lam = np.empty(n_f)  # [F]
    feasible = np.zeros(n_f, dtype=bool)  # [F]
    for fi in range(n_f):
        grid = ls_wng_lambda_grid(a_list[fi], n_grid=n_grid)  # [n_grid] ascending
        if wng_floor_db >= max_white_noise_gain_db(c_list[fi]) - 0.02:
            lam[fi] = grid[-1]  # floor above ceiling -> max robustness, infeasible
            feasible[fi] = False
            continue
        best_lam, best_wng = grid[-1], -np.inf
        for lg in grid:
            w_f = np.linalg.solve(a_list[fi] + lg * np.eye(m), rhs_list[fi])  # [M] per-bin
            wng = white_noise_gain_db(w_f, c_list[fi])
            if wng >= wng_floor_db:
                lam[fi], feasible[fi] = lg, True  # smallest loading meeting the floor
                break
            if wng > best_wng:
                best_lam, best_wng = lg, wng
        else:
            lam[fi], feasible[fi] = best_lam, False  # floor unreachable on grid -> best effort

    # (2) one coupled solve with the chosen loads -> smooth, realizable weights.
    w = ls_pressure_match_coupled(h, b_field, w_quad, lam, mu, freqs, tau)  # [M, F]
    return w, feasible, lam, tau


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

    target = build_target(spec, obs, ds.frequencies, c_sound=c_sound)
    look = target.look_idx

    weights = np.zeros((m, n_f), dtype=np.complex128)  # [M, F]
    wng_db = np.zeros(n_f)  # [F]
    feasible = np.ones(n_f, dtype=bool)  # [F]

    # The LS engine is frequency-COUPLED (DR-P2-03): all bins are solved jointly once, before
    # the per-frequency loop, for smooth realizable filters + an honest WNG floor. It degrades
    # to independent per-bin solves for F<3 (so single-bin tests are unchanged). Every other
    # engine stays per-frequency in the loop below.
    ls_weights: np.ndarray | None = None  # [M, F]
    ls_feasible: np.ndarray | None = None  # [F]
    ls_lam: np.ndarray | None = None  # [F]
    ls_tau: float = 0.0
    if spec.engine == "ls":
        ls_weights, ls_feasible, ls_lam, ls_tau = _design_ls_coupled(
            h, target.b_field, w_quad, ds.frequencies, look, spec.wng_floor_db
        )

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
            w_f = ls_weights[:, f]  # from the frequency-coupled pre-solve (DR-P2-03)
            feasible[f] = ls_feasible[f]
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
    if spec.engine == "ls":
        attrs["ls_tau_s"] = ls_tau  # shared modeling delay (s) used by the coupled solve
        attrs["ls_lambda"] = ls_lam.tolist()  # per-bin Tikhonov / WNG-floor loads
    return DesignResult(weights=weights, steered_field=p, metrics=metrics, spec=spec, attrs=attrs)
