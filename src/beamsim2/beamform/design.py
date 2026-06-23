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
    floor_covariances,
    loaded_mvdr_weights,
    ls_wng_lambda_grid,
    max_white_noise_gain_db,
    solve_loading_for_wng,
    solve_maxdir_loading_for_wng,
    white_noise_gain_db,
)
from beamsim2.beamform.targets import TargetSpec, build_target
from beamsim2.beamform.weights import (
    align_global_phase,
    choose_shared_delay_complex,
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


def _constant_di_ar(h_f, w_quad, look, mode, target, eps_min):
    """Build the (A, R, c) generalized-eigenproblem triple for one bin, eps_min-floored.

    ``mode="index"`` (Luo's directivity INDEX, recommended): ``A = c c^H`` (rank-1), ``R`` =
    whole-sphere covariance, so ``4*pi*(w^H A w)/(w^H R w)`` is the classical directivity factor.
    ``mode="region"`` (the pre-3b objective): ``A`` = accept-cap covariance, ``R`` = reject
    covariance — a front-to-region power ratio, NOT the directivity index (kept as an option).
    """
    c = look_vector(h_f, look)  # [M]
    if mode == "index":
        a = np.outer(c, np.conj(c))  # [M, M] proper DI: A = c c^H (rank-1, Hermitian PSD)
        r = covariance(h_f, w_quad)  # [M, M] whole-sphere reject
    else:  # "region"
        a = covariance(h_f, w_quad, mask=target.accept_mask)  # [M, M]
        r = covariance(h_f, w_quad, mask=target.reject_mask)  # [M, M]
    a, r = floor_covariances(a, r, eps_min)
    return a, r, c


def _design_constant_di(
    h, w_quad, look, freqs, spec, target, *, eps_min=1e-7, n_scan=40, n_bisect=44
):
    """Constant-directivity design: ONE shared tau*, honest WNG floor, cardinal-safe realization.

    Two-pass (Luo MSCD): pass 1 finds each bin's directivity ceiling ``tau_max`` from the
    generalized eigenproblem; the shared constant level ``tau* = 0.98 * min_f tau_max`` (capped by
    ``target_gdi_db``). Pass 2 enforces an honest white-noise-gain floor by lowering the SINGLE
    shared ``tau*`` — never a per-bin tau, which would break constant directivity. ``WNG(tau)`` for
    the proper-DI objective is *unimodal* (``docs/Chunk3b_Findings.md``), so the floor search
    bisects on the descending branch ``[min_f tau_peak, ceiling]`` where the min-over-bins WNG
    rises as tau drops. If even the most-robust constant tau cannot meet the floor, the band is
    flagged infeasible (directivity stays flat — graceful, never silent superdirective garbage).
    The per-bin MSCD weights are then global-phase-aligned and de-ramped by one shared modeling
    delay for realizable filters (both are per-frequency global factors -> cardinal-rule safe).

    Returns ``(weights[M,F], feasible[F], wng_db[F], tau_star, shared_tau, level_db, band_ok)``.
    """
    m, n_f, _ = h.shape
    mode = getattr(spec, "directivity_mode", "region")
    a_list, r_list, c_list = [], [], []
    for f in range(n_f):
        a, r, c = _constant_di_ar(h[:, f, :], w_quad, look, mode, target, eps_min)
        a_list.append(a)
        r_list.append(r)
        c_list.append(c)

    # Pass 1 — per-bin directivity ceiling; feasible shared constant level (min over band).
    tau_max = np.array(
        [max_directivity(a_list[f], r_list[f], eps=0.0)[1] for f in range(n_f)]
    )  # [F]
    ceiling = float(np.min(tau_max)) * 0.98
    if spec.target_gdi_db is not None:
        cap = 10.0 ** (spec.target_gdi_db / 10.0)
        if mode == "index":
            cap /= 4.0 * np.pi  # target_gdi_db is the desired directivity INDEX in index mode
        ceiling = min(ceiling, cap)

    def mscd_wng(f, tau):
        """(WNG_dB, w) for bin f at tau; (nan, None) if tau is infeasible there."""
        try:
            w = luo_mscd(a_list[f], r_list[f], c_list[f], tau)
        except ValueError:
            return np.nan, None
        return white_noise_gain_db(w, c_list[f]), w

    def min_wng(tau):
        worst = np.inf
        for f in range(n_f):
            wng, w = mscd_wng(f, tau)
            if w is None or not np.isfinite(wng):
                return -np.inf
            worst = min(worst, wng)
        return worst

    # Pass 2 — honest WNG floor on the shared tau* (unimodal-aware; one tau* => constant DI kept).
    floor = spec.wng_floor_db
    band_feasible = True
    if min_wng(ceiling) >= floor:
        tau_star = ceiling  # the ceiling already meets the floor
    else:
        tau_peak = np.empty(n_f)  # [F] per-bin WNG-hump location (lower bracket end)
        scan = np.geomspace(1e-4 * ceiling, ceiling, n_scan)
        for f in range(n_f):
            vals = np.array([mscd_wng(f, t)[0] for t in scan])  # [n_scan]
            tau_peak[f] = scan[int(np.nanargmax(vals))] if np.any(np.isfinite(vals)) else ceiling
        tau_lo = float(np.clip(np.min(tau_peak), 1e-12, ceiling))
        if min_wng(tau_lo) < floor:
            tau_star, band_feasible = tau_lo, False  # floor unreachable at constant DI -> flag
        elif min_wng(tau_lo) >= min_wng(ceiling):
            # Expected case: min-over-bins WNG is monotone-decreasing on [tau_lo, ceiling]
            # (tau_lo sits at the per-bin WNG humps). Bisect for the largest feasible tau*.
            lo, hi = tau_lo, ceiling  # lo = robust/high-WNG, hi = sharp/low-WNG
            for _ in range(n_bisect):
                mid = 0.5 * (lo + hi)
                if min_wng(mid) >= floor:
                    lo = mid  # passes -> push tau up for more directivity
                else:
                    hi = mid
            tau_star = lo
        else:
            # Safety net (unimodality is empirical, not proven): a non-monotone bracket would
            # mislead the bisection, so grid-search the largest tau meeting the floor instead.
            grid = np.linspace(tau_lo, ceiling, 4 * n_bisect)  # [4*n_bisect]
            ok = [float(t) for t in grid if min_wng(t) >= floor]
            tau_star = max(ok) if ok else tau_lo

    # Final per-bin solve at the shared tau*; flag any bin where MSCD is infeasible there.
    weights = np.zeros((m, n_f), dtype=np.complex128)  # [M, F]
    feasible = np.ones(n_f, dtype=bool)  # [F]
    for f in range(n_f):
        _, w = mscd_wng(f, tau_star)
        if w is None:  # numeric edge exactly at tau_star -> nudge just inside the indefinite region
            _, w = mscd_wng(f, tau_star * 0.999)
        if w is None:
            feasible[f] = False
            continue
        weights[:, f] = w

    # Cardinal-safe realization: per-bin global-phase continuity + one shared modeling delay.
    weights = align_global_phase(weights)  # [M, F]
    shared_tau = choose_shared_delay_complex(weights, freqs)

    wng_db = np.array(
        [
            white_noise_gain_db(weights[:, f], c_list[f]) if np.any(weights[:, f]) else -np.inf
            for f in range(n_f)
        ]
    )  # [F]
    feasible = feasible & band_feasible & (wng_db >= floor - 1.0)
    level_db = 10.0 * np.log10((4.0 * np.pi if mode == "index" else 1.0) * tau_star)
    return weights, feasible, wng_db, tau_star, shared_tau, level_db, band_feasible


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
    if spec.engine == "auto":
        # Auto-Design (Chunk 3c): a principled escalation ladder over the well-posed engines that
        # picks the one best meeting the target and reports its choice honestly. Lazy import keeps
        # the orchestrator <-> design() back-reference from being a circular import at module load.
        from beamsim2.beamform.orchestrator import design_auto

        return design_auto(ds, spec)

    h = stacked_h_full(ds)  # [M, F, N] complex128
    m, n_f, _ = h.shape
    obs = ds.directions
    w_quad = obs.weights  # [N]
    c_sound = float(ds.attrs.get("speed_of_sound", 343.2))

    target = build_target(spec, obs, ds.frequencies, c_sound=c_sound)
    look = target.look_idx
    # Directivity objective for the constant_di / max_directivity engines (Chunk 3b): "index" is
    # Luo's proper directivity index (A = c c^H), "region" is the pre-3b front-cap power ratio.
    mode = getattr(spec, "directivity_mode", "region")
    eps_min = 1e-7  # relative diagonal floor for the generalized eigenproblem / secular root

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

    # Constant-DI (engine #2) is fully precomputed across all bins (Chunk 3b): one shared
    # directivity level tau*, an honest WNG floor, eps_min well-posedness, and a cardinal-safe
    # global-phase + shared-delay realization (see :func:`_design_constant_di`).
    cd_weights: np.ndarray | None = None  # [M, F]
    cd_feasible: np.ndarray | None = None  # [F]
    cd_tau: float = 0.0
    cd_shared_tau: float = 0.0
    cd_level_db: float | None = None
    cd_band_feasible: bool = True
    if spec.engine == "constant_di":
        (
            cd_weights,
            cd_feasible,
            _cd_wng,
            cd_tau,
            cd_shared_tau,
            cd_level_db,
            cd_band_feasible,
        ) = _design_constant_di(h, w_quad, look, ds.frequencies, spec, target, eps_min=eps_min)

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
            if mode == "index":
                a_mat = np.outer(c, np.conj(c))  # proper-DI: A = c c^H
                r_mat = covariance(h_f, w_quad)  # whole-sphere reject
            else:
                a_mat = covariance(h_f, w_quad, mask=target.accept_mask)
                r_mat = covariance(h_f, w_quad, mask=target.reject_mask)
            # Honest WNG floor: the unloaded max-directivity beam is freely superdirective.
            w_f, _eps, _wng, feas = solve_maxdir_loading_for_wng(
                a_mat, r_mat, c, spec.wng_floor_db, eps_min=eps_min
            )
            feasible[f] = feas
        elif spec.engine == "constant_di":
            w_f = cd_weights[:, f]  # from the all-bin constant-DI pre-solve (one shared tau*)
            feasible[f] = cd_feasible[f]
        else:
            raise ValueError(f"Unknown engine {spec.engine!r}.")
        weights[:, f] = w_f
        # A degenerate / infeasible bin can return all-zero weights (e.g. a rank-deficient array
        # collapsed to a single point); report WNG as -inf rather than dividing by ||w||^2 = 0.
        wng_db[f] = white_noise_gain_db(w_f, c) if np.any(w_f) else -np.inf

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
    if spec.engine == "constant_di":
        attrs["directivity_mode"] = mode
        attrs["constant_di_tau"] = cd_tau  # the shared generalized-eigenvalue level held constant
        attrs["constant_di_shared_tau_s"] = cd_shared_tau  # shared modeling delay (s)
        attrs["band_feasible"] = cd_band_feasible
        if mode == "index":
            attrs["constant_di_db"] = cd_level_db  # the held-constant directivity INDEX (dB)
        else:
            attrs["constant_gdi_db"] = cd_level_db  # the held-constant cap-ratio GDI (dB)
    if spec.engine == "ls":
        attrs["ls_tau_s"] = ls_tau  # shared modeling delay (s) used by the coupled solve
        attrs["ls_lambda"] = ls_lam.tolist()  # per-bin Tikhonov / WNG-floor loads
    return DesignResult(weights=weights, steered_field=p, metrics=metrics, spec=spec, attrs=attrs)
