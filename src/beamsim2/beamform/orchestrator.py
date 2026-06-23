"""Auto-Design orchestrator (Stage P2-1, Chunk 3c): pick the engine that best meets the target.

``engine="auto"`` dispatches here. The orchestrator does **not** re-tune any engine (3a/3b did
that) and **never** re-zeros or minimum-phase-ifies a driver — it only *calls* :func:`design`
with each candidate engine, scores the result against the target's OWN objective metric (reusing
the metrics ``design()`` already reports), and returns the best feasible candidate together with
an honest report of what it chose and where the target cannot be met.

The flagged design decision (confirmed at the 3c kickoff; ``docs/Bug_Fix_Proposal.md`` Open
Question 1): a **principled escalation ladder** over the *well-posed* engines, NOT a literal
"try every algorithm and stack them." Blindly stacking solvers that optimize incommensurable
objectives (pressure-match + superdirective + hard-null) has no well-posed combined objective and
risks non-convergence / unrealizable filters. The user-facing outcome is the same ("Auto-Design
finds a good filter without me picking the algorithm"), but this actually converges and reports
honestly.

Target classes and their winners are empirically grounded on the real CI-safe fixtures
(``docs/Chunk3c_Findings.md``); each wins on its OWN metric by a decisive margin, so the gate is
non-circular (it asserts engine *behavior*, not orchestrator wiring):

==========================  ==============  ==================================================
class (from objective)      winner          its margin on the class metric (real fixtures)
==========================  ==============  ==================================================
``shape``                   ``ls``          target_error 5.8 dB  vs  delay_sum 11.1 / mvdr 12.8
``constant_directivity``    ``constant_di`` DI ptp 0.000 dB      vs  ls 1.96 / max_dir 3.33
``nulls``                   ``lcmv``        null -310 dB         vs  ls -3 / mvdr -2
``max_directivity``         ``max_directivity``  DI 16.7 dB     vs  delay_sum 12.0 / ls 8.7
==========================  ==============  ==================================================

Cardinal rule: the orchestrator composes existing engines; steering stays entirely in H's
inter-driver phase. The collapse-to-origin control under ``engine="auto"`` still yields DI -> 0.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from beamsim2.assembly.tensor import stacked_h_full
from beamsim2.beamform.covariance import look_vector
from beamsim2.beamform.regularize import max_white_noise_gain_db
from beamsim2.beamform.targets import TargetSpec, build_target
from beamsim2.metrics.cea2034 import compute_cea2034

# ---------------------------------------------------------------------------------------------
# Per-class candidate ladders (robust -> aggressive). Every candidate is run through the real
# design(); the selector picks the optimizer of the class's own objective metric (see _select).
# ---------------------------------------------------------------------------------------------
_LADDERS: dict[str, list[str]] = {
    "shape": ["delay_sum", "ls", "mvdr", "max_directivity"],
    "constant_directivity": ["ls", "max_directivity", "constant_di"],
    "nulls": ["ls", "mvdr", "lcmv"],
    "max_directivity": ["delay_sum", "ls", "mvdr", "max_directivity"],
}

# Tie-break preference (used only when two candidates score within `_TIE_EPS_DB`). For most
# classes the margins are huge and ties never occur, so robust-first (ladder order) is the
# default. For max_directivity, loaded-MVDR and the WNG-floored generalized-Rayleigh max-
# directivity converge to the SAME directivity (16.67 dB on the cap) -> prefer the engine that
# is literally the max-directivity formulation (docs/Chunk3c_Findings.md).
_TIE_PREFERENCE: dict[str, list[str]] = {
    "max_directivity": ["max_directivity", "mvdr", "ls", "delay_sum"],
}

# Acceptance thresholds == the "converged" honesty flag, NOT the selector (the selector is the
# metric-optimizer, which is decisive in the data). Calibrated on the real fixtures.
_FEAS_FRAC = 0.5  # a candidate is "feasible" if >= this fraction of in-band bins meet the WNG floor
_SHAPE_TE_DB = 9.0  # shape: target_error_db median <= this -> converged (ls ~5.8; others >11)
_CONST_DI_PTP_DB = 0.25  # constant_directivity: DI ptp <= this -> converged (constant_di ~0)
_CONST_DI_MIN_DB = 3.0  # omni-trap guard: a constant-DI candidate must be at least this directive
_NULL_DEPTH_DB = (
    -30.0
)  # nulls: worst-bin null depth <= this -> converged (lcmv ~ -300; others ~ -3)
_TIE_EPS_DB = 0.25  # scores within this many dB are treated as ties and broken by preference order
_OMNI_PENALTY = 1.0e6  # added to a constant-DI score whose level is below the omni-trap guard


def _classify(spec: TargetSpec) -> str:
    """Map a TargetSpec to its Auto-Design target class.

    ``objective == "multi"`` (Chunk 3d) is checked first: in the multi class a non-empty
    ``nulls`` is a *feasibility gate*, NOT a class override (a deliberate divergence — multi
    keeps its scalarized objective and discards candidates that fail to null). For the
    single-objective classes a non-empty ``nulls`` dominates (a hard-null request is a null
    target regardless of ``objective``); otherwise the explicit ``objective`` decides,
    defaulting to ``"shape"``.
    """
    if spec.objective == "multi":
        return "multi"
    if spec.nulls:
        return "nulls"
    if spec.objective in ("constant_directivity", "max_directivity"):
        return spec.objective
    return "shape"


def _null_depth_worst(r, null_idx: list[int], look_idx: int) -> float:
    """Worst (shallowest) on-axis-relative null depth in dB across the band (``inf`` if none).

    For each frequency, the deepest requested null is only as good as its shallowest bin; a
    beamformer that fails to null at any one frequency is not a null solution. ``inf`` for a
    target with no nulls (the metric is unused for non-null classes).
    """
    if not null_idx:
        return float("inf")
    worst = -np.inf
    for f in range(r.steered_field.shape[0]):
        p = r.steered_field[f]  # [N]
        look_mag = np.abs(p[look_idx]) + 1e-300
        here = max(20.0 * np.log10(np.abs(p[j]) / look_mag + 1e-300) for j in null_idx)
        worst = max(worst, here)  # worst = shallowest across the band
    return float(worst)


def _evaluate(r, target_class: str, null_idx: list[int], look_idx: int) -> dict:
    """Score one candidate ``DesignResult`` on its class metric; return a JSON-friendly trace dict.

    Reuses the metrics ``design()`` already computed (``di_db``, ``wng_db``, ``target_error_db``,
    ``feasible_mask``) plus the achieved-field null depth — never recomputes patterns. ``score``
    is lower-is-better so the selector is a single ``argmin``; ``feasible`` gates on the honest
    per-bin WNG floor; ``converged`` is the acceptance-threshold honesty flag.
    """
    di = r.metrics["di_db"]  # [F]
    wng = r.metrics["wng_db"]  # [F]
    te = r.metrics["target_error_db"]  # [F]
    feas_frac = float(np.mean(r.metrics["feasible_mask"]))
    di_med = float(np.median(di))
    di_ptp = float(np.ptp(di))
    di_min = float(np.min(di))
    wng_min = float(np.min(wng))
    te_med = float(np.median(te))
    null_worst = _null_depth_worst(r, null_idx, look_idx)
    feasible = feas_frac >= _FEAS_FRAC

    if target_class == "shape":
        score = te_med  # minimize shape error
        converged = feasible and te_med <= _SHAPE_TE_DB
    elif target_class == "constant_directivity":
        score = di_ptp + (
            0.0 if di_med >= _CONST_DI_MIN_DB else _OMNI_PENALTY
        )  # flattest, not omni
        converged = feasible and di_ptp <= _CONST_DI_PTP_DB and di_med >= _CONST_DI_MIN_DB
    elif target_class == "nulls":
        score = null_worst  # deepest (most negative) worst-bin null
        converged = feasible and null_worst <= _NULL_DEPTH_DB
    else:  # max_directivity
        score = -di_med  # maximize directivity subject to the WNG floor (feasibility)
        converged = feasible

    return {
        "engine": r.attrs.get("engine", "?"),
        "score": float(score),
        "di_med": di_med,
        "di_ptp": di_ptp,
        "di_min": di_min,
        "wng_min": wng_min,
        "te_med": te_med,
        "null_worst": (None if not np.isfinite(null_worst) else null_worst),
        "feas_frac": feas_frac,
        "feasible": bool(feasible),
        "converged": bool(converged),
    }


def _select(cands: list[tuple[dict, object]], target_class: str) -> tuple[dict, object]:
    """Pick the best candidate: the class-metric optimizer among feasible ones.

    Among feasible candidates (best-effort over ALL if none is feasible), take the minimum score;
    candidates within ``_TIE_EPS_DB`` of it are genuine ties (only happens for max_directivity,
    where mvdr == max_directivity numerically) and are broken by a fixed preference order, so the
    choice is deterministic (never iteration-order dependent).
    """
    feasible = [(ev, r) for ev, r in cands if ev["feasible"]]
    pool = feasible if feasible else cands  # honest best-effort when nothing meets the floor
    tie_pref = _TIE_PREFERENCE.get(target_class, _LADDERS[target_class])
    best_score = min(ev["score"] for ev, _ in pool)
    near = [(ev, r) for ev, r in pool if ev["score"] <= best_score + _TIE_EPS_DB]
    near.sort(key=lambda er: tie_pref.index(er[0]["engine"]) if er[0]["engine"] in tie_pref else 99)
    return near[0]


def _reason(ev: dict, target_class: str, band_feasible: bool) -> str:
    """One-line, honest explanation of the choice for the GUI status line / audit trail."""
    eng = ev["engine"]
    if not band_feasible:
        return (
            f"best-effort: '{eng}' came closest, but no candidate met the WNG floor across the "
            f"band (the array's directivity ceiling is below the requested robustness — physics, "
            f"not a solver defect; see auto_prescreen / feasible_mask)."
        )
    if target_class == "shape":
        detail = f"lowest shape error ({ev['te_med']:.1f} dB) among the candidates"
    elif target_class == "constant_directivity":
        detail = (
            f"flattest directivity across the band (ptp {ev['di_ptp']:.3f} dB at DI "
            f"{ev['di_med']:.1f} dB)"
        )
    elif target_class == "nulls":
        detail = f"deepest null ({ev['null_worst']:.0f} dB) at the requested direction(s)"
    else:  # max_directivity
        detail = f"highest directivity ({ev['di_med']:.1f} dB) within the WNG floor"
    return f"chose '{eng}': {detail}."


def _wng_prescreen(ds, spec: TargetSpec, look_idx: int) -> dict:
    """Per-band feasibility pre-screen: matched-field WNG ceiling vs the requested floor.

    The distortionless WNG ceiling is ``10 log10(||c||^2)`` per bin (reached at infinite loading);
    a floor above it cannot be met by ANY engine at that bin — an array/physics limit, not a
    solver defect. Reported so the honest "where it's infeasible" message is grounded.
    """
    h = stacked_h_full(ds)  # [M, F, N]
    n_f = h.shape[1]
    ceil = np.array(
        [max_white_noise_gain_db(look_vector(h[:, f, :], look_idx)) for f in range(n_f)]
    )  # [F]
    floor = float(spec.wng_floor_db)
    return {
        "wng_floor_db": floor,
        "wng_ceiling_db": [float(x) for x in ceil],
        "floor_exceeds_ceiling_bins": int(np.sum(floor > ceil + 1e-9)),
        "n_bins": int(n_f),
    }


def design_auto(ds, spec: TargetSpec):
    """Auto-Design: run the target-conditioned engine ladder, score, pick best, report honestly.

    Parameters
    ----------
    ds : RadiationDataset
        Phase-1 output (the same object the concrete engines consume).
    spec : TargetSpec
        The user request with ``engine == "auto"``; ``objective`` (and any ``nulls``) selects the
        target class.

    Returns
    -------
    DesignResult
        The chosen candidate's result, unchanged numerically (it IS a real ``design()`` output for
        the winning engine), with an honest Auto-Design report grafted onto ``attrs``:
        ``engine`` (the concrete engine actually used), ``auto_selected``, ``auto_class``,
        ``auto_trace`` (every candidate's metrics), ``auto_reason``, ``auto_prescreen``,
        ``band_feasible``; per-bin honesty stays in ``metrics["feasible_mask"]``. ``spec`` is the
        original ``engine="auto"`` request, echoed back.
    """
    from beamsim2.beamform.design import design  # lazy: design.py dispatches into this module

    target_class = _classify(spec)
    if target_class == "multi":
        return design_multi(ds, spec)  # Chunk 3d: scalarized multi-objective (engine, knob) search

    obs = ds.directions
    c_sound = float(ds.attrs.get("speed_of_sound", 343.2))
    target = build_target(spec, obs, ds.frequencies, c_sound=c_sound)
    null_idx, look_idx = target.null_idx, target.look_idx
    prescreen = _wng_prescreen(ds, spec, look_idx)

    cands: list[tuple[dict, object]] = []  # [(trace_dict, DesignResult)] one per ladder engine
    for eng in _LADDERS[target_class]:
        cand_spec = replace(spec, engine=eng)
        # Force Luo's proper directivity INDEX for the DI-objective engines, independent of what
        # the caller passed: the "region" default optimizes a different (cap-ratio) objective and
        # would stop constant_di from holding DI flat (advisor-flagged silent-failure trap).
        if eng in ("constant_di", "max_directivity"):
            cand_spec = replace(cand_spec, directivity_mode="index")
        r = design(ds, cand_spec)
        cands.append((_evaluate(r, target_class, null_idx, look_idx), r))

    chosen_ev, chosen_r = _select(cands, target_class)
    # band_feasible: the chosen engine met the floor on a band majority AND (if it sets its own
    # band_feasible flag, e.g. constant_di) that flag held too.
    band_feasible = bool(chosen_ev["feasible"]) and bool(chosen_r.attrs.get("band_feasible", True))

    chosen_r.attrs["engine"] = chosen_ev["engine"]  # honest: the concrete engine actually used
    chosen_r.attrs["auto_selected"] = True
    chosen_r.attrs["auto_class"] = target_class
    chosen_r.attrs["auto_trace"] = [ev for ev, _ in cands]
    chosen_r.attrs["auto_reason"] = _reason(chosen_ev, target_class, band_feasible)
    chosen_r.attrs["auto_prescreen"] = prescreen
    chosen_r.attrs["band_feasible"] = band_feasible
    chosen_r.spec = spec  # echo the user's original auto request, not the overridden candidate
    return chosen_r


# =================================================================================================
# Chunk 3d — Multi-target objectives (scalarized weighted-sum over a curated (engine, knob) search)
# =================================================================================================
#
# A designer that targets {directivity index, -6 dB beamwidth, in-room (CEA-2034-A EIR) slope}
# *jointly*. The combination rule (confirmed at the 3d kickoff) is a **scalarized weighted-sum of
# NORMALIZED per-objective deviations**, with the hard constraints (the WNG floor, and any nulls)
# kept as lexicographically-prior **feasibility gates**. The optimization variable is the
# **(engine, knob)** combo — the engines + their tunable knobs (built in 3a/3b/3c) already span the
# {DI, beamwidth, robustness} space, so multi-target is a principled *search + scoring* layer, NOT a
# new joint solver: it only *calls* :func:`design` and scores the result. Steering stays entirely in
# H's inter-driver phase (cardinal rule); the collapse-to-origin control still yields DI -> 0.
#
# Why these objectives / this conflict (``docs/Chunk3d_Findings.md``, measured on the real cap):
#   - DI and the in-room *downtilt level* are tightly coupled (r=-0.95) -> the in-room axis used is
#     the EIR *slope* (constant_di ~ -0.7 dB/oct vs max_directivity ~ -2.6 vs delay_sum ~ -3.9).
#   - DI and -6 dB beamwidth conflict (r=-0.89) but only partially: at fixed DI the achievable
#     beamwidth still spans ~15 deg (lever = engine/shape), so beamwidth is a genuine second axis.
#   - The genuine 3-way conflict {high DI, wide beam, flat in-room} has no single winner, so the
#     scalarized optimum genuinely TRADES (its worst normalized deviation is lower than every
#     single-objective optimum's) -> a non-circular Pareto gate.

# Fixed physical normalization scales (dB / deg / dB-per-octave). Chosen ONCE from the diagnostic's
# measured metric SPANS so each objective's achievable deviation is ~3 normalized units (DI span
# ~9 dB, beamwidth span ~35 deg, clean EIR-slope span ~3 dB/oct), i.e. no objective dominates the
# unweighted sum. Hardcoded + reproducible (NOT per-run candidate spread, which would make the score
# depend on the candidate set). The minimax trade holds *structurally* because the objectives
# conflict (r(DI,BW)=-0.89), not because of the scale pick. See ``docs/Chunk3d_Findings.md``.
_NORM: dict[str, float] = {"di": 3.0, "beamwidth": 12.0, "inroom": 1.0}

# An unclosed main lobe (beamwidth_deg -> nan) is a hard miss, not a free pass: charge a large fixed
# normalized deviation (~60 deg / 12) so a no-lobe candidate cannot win the beamwidth objective.
_BW_NAN_DEV = 5.0

# Curated multi-target candidate ladder (robust -> aggressive), spanning DI x beamwidth x in-room
# space. ``mvdr`` is intentionally omitted: index-mode ``max_directivity`` is numerically identical
# to loaded-MVDR (3c finding) and listing both would let a tie name "mvdr" over the canonical
# "max_directivity". Index mode is forced on the DI engines (the 3c silent-failure trap: "region"
# default optimizes a different cap-ratio objective and would not hold the directivity index).
_MULTI_LADDER: list[tuple[str, dict]] = [
    ("ls", {"mode": "preset", "preset": "cardioid"}),
    ("ls", {"mode": "preset", "preset": "supercardioid"}),
    ("ls", {"mode": "preset", "preset": "hypercardioid"}),
    ("delay_sum", {"mode": "steering_only"}),
    ("constant_di", {"mode": "steering_only", "directivity_mode": "index", "target_gdi_db": 10.0}),
    ("constant_di", {"mode": "steering_only", "directivity_mode": "index", "target_gdi_db": 12.0}),
    ("constant_di", {"mode": "steering_only", "directivity_mode": "index", "target_gdi_db": 14.0}),
    ("constant_di", {"mode": "steering_only", "directivity_mode": "index", "target_gdi_db": None}),
    ("max_directivity", {"mode": "steering_only", "directivity_mode": "index"}),
]

# JSON-friendly label per objective for the honest report.
_OBJ_LABEL = {"di": "DI (dB)", "beamwidth": "beamwidth (deg)", "inroom": "in-room slope (dB/oct)"}


def _multi_targets(spec: TargetSpec) -> dict[str, float]:
    """The active multi-objectives = those whose target is set (non-``None``)."""
    t: dict[str, float] = {}
    if spec.target_di_db is not None:
        t["di"] = float(spec.target_di_db)
    if spec.target_beamwidth_deg is not None:
        t["beamwidth"] = float(spec.target_beamwidth_deg)
    if spec.target_inroom_slope_db_per_oct is not None:
        t["inroom"] = float(spec.target_inroom_slope_db_per_oct)
    return t


def _multi_weights(spec: TargetSpec, targets: dict[str, float]) -> dict[str, float]:
    """Resolve weights over the active objectives (default: equal; an all-zero set -> equal)."""
    if spec.objective_weights:
        w = {k: float(spec.objective_weights.get(k, 0.0)) for k in targets}
        if any(v > 0.0 for v in w.values()):
            return w
    return {k: 1.0 for k in targets}  # default balanced (or rescue an all-zero weight set)


def _eir_slope(steered_field, ds, steer_unit) -> float:
    """In-room (CEA-2034-A Estimated-In-Room) spectral slope in dB/octave for a steered field.

    Reuses Chunk-2's :func:`compute_cea2034` (the confirmed in-room approximation); the spinorama is
    referenced to the BEAM axis (``steer_unit``), since that is on-axis for the listener. A single
    bin cannot define a slope -> returns 0.0 (in-room then contributes nothing, honestly).
    """
    freqs = np.asarray(ds.frequencies, dtype=np.float64)
    if freqs.size < 2:
        return 0.0
    cea = compute_cea2034(steered_field, freqs, ds.directions, np.asarray(steer_unit, float))
    return float(np.polyfit(np.log2(freqs), cea["estimated_in_room"], 1)[0])


def _multi_achieved(r, ds, steer_unit, null_idx: list[int], look_idx: int) -> dict:
    """Achieved {DI, beamwidth, in-room slope, feasibility, null} for one candidate DesignResult.

    Reuses the metrics ``design()`` already reported (``di_db``, ``beamwidth_deg``, feasible_mask)
    plus the achieved-field EIR slope and null depth. ``bw_med`` is ``nan`` when no bin's main lobe
    closes (handled by the scorer's nan-guard).
    """
    di = r.metrics["di_db"]  # [F]
    bw = r.metrics["beamwidth_deg"]  # [F]
    bw_med = float(np.nanmedian(bw)) if np.any(np.isfinite(bw)) else float("nan")
    return {
        "di_med": float(np.median(di)),
        "di_ptp": float(np.ptp(di)),
        "bw_med": bw_med,
        "bw_nan": int(np.sum(~np.isfinite(bw))),
        "eir_slope": _eir_slope(r.steered_field, ds, steer_unit),
        "feas_frac": float(np.mean(r.metrics["feasible_mask"])),
        "null_worst": _null_depth_worst(r, null_idx, look_idx),
    }


def _multi_devs(ach: dict, targets: dict[str, float]) -> dict[str, float]:
    """Normalized per-objective deviations |achieved - target| / scale (active objectives only)."""
    d: dict[str, float] = {}
    if "di" in targets:
        d["di"] = abs(ach["di_med"] - targets["di"]) / _NORM["di"]
    if "beamwidth" in targets:
        bw = ach["bw_med"]
        d["beamwidth"] = (
            _BW_NAN_DEV
            if not np.isfinite(bw)
            else abs(bw - targets["beamwidth"]) / _NORM["beamwidth"]
        )
    if "inroom" in targets:
        d["inroom"] = abs(ach["eir_slope"] - targets["inroom"]) / _NORM["inroom"]
    return d


def _combined(devs: dict[str, float], weights: dict[str, float]) -> float:
    """Scalarized score = weighted mean of the normalized deviations (lower is better)."""
    keys = [k for k in devs if weights.get(k, 0.0) > 0.0]
    wsum = sum(weights[k] for k in keys)
    if not wsum:
        return float("inf")
    return sum(weights[k] * devs[k] for k in keys) / wsum


def _null_feasible(ach: dict, spec: TargetSpec) -> bool:
    """Multi-target null gate: if nulls were requested, the worst-bin null must be deep enough."""
    if not spec.nulls:
        return True
    nw = ach["null_worst"]
    return bool(np.isfinite(nw) and nw <= _NULL_DEPTH_DB)


def _knob_summary(eng: str, knobs: dict) -> dict:
    """JSON-friendly (engine, knobs) label for the honest trace."""
    out = {"engine": eng}
    if "preset" in knobs:
        out["preset"] = knobs["preset"]
    if knobs.get("target_gdi_db") is not None:
        out["target_gdi_db"] = knobs["target_gdi_db"]
    return out


def _multi_achieved_report(ach: dict, targets: dict[str, float]) -> dict:
    """Per-objective achieved-vs-target honesty report for ``attrs`` / the GUI."""
    rep: dict[str, dict] = {}
    if "di" in targets:
        rep["di"] = {
            "target": targets["di"],
            "achieved": ach["di_med"],
            "dev_norm": ach["devs"]["di"],
        }
    if "beamwidth" in targets:
        rep["beamwidth"] = {
            "target": targets["beamwidth"],
            "achieved": ach["bw_med"],
            "dev_norm": ach["devs"]["beamwidth"],
        }
    if "inroom" in targets:
        rep["inroom"] = {
            "target": targets["inroom"],
            "achieved": ach["eir_slope"],
            "dev_norm": ach["devs"]["inroom"],
        }
    return rep


def _multi_reason(ach: dict, report: dict, weights: dict, band_feasible: bool) -> str:
    """One-line, honest explanation of the multi-target choice for the GUI / audit trail."""
    eng = ach["engine"]
    parts = []
    for k, lab in _OBJ_LABEL.items():
        if k in report:
            a = report[k]["achieved"]
            t = report[k]["target"]
            a_s = "n/a" if (k == "beamwidth" and not np.isfinite(a)) else f"{a:.1f}"
            parts.append(f"{lab} {a_s}/{t:.1f}")
    wtxt = ", ".join(f"{k}:{weights[k]:g}" for k in weights)
    pre = (
        ""
        if band_feasible
        else "best-effort (no candidate met the WNG floor / nulls across the band): "
    )
    return f"{pre}chose '{eng}' — best scalarized fit [{'; '.join(parts)}] at weights ({wtxt})."


def design_multi(ds, spec: TargetSpec):
    """Multi-target Auto-Design: scalarized (engine, knob) search over {DI, beamwidth, in-room}.

    Runs the curated :data:`_MULTI_LADDER` through the real :func:`design`, scores each candidate's
    achieved field on the scalarized weighted-sum of normalized per-objective deviations (gated by
    the honest WNG floor / ``feasible_mask`` and any ``nulls``), and returns the best feasible
    candidate with an honest report of what each objective achieved vs its target and which
    (engine, knobs) was chosen. Reuses the 3c run->score->select skeleton; the only thing 3d
    computes beyond the metrics ``design()`` already reports is the CEA-2034-A in-room slope (one
    resample per candidate).

    ``nulls`` under ``"multi"`` are a *best-effort* feasibility gate only: the multi ladder has no
    hard-null engine (``lcmv`` is excluded), so a requested null usually cannot satisfy the gate and
    the result degrades to best-effort (``band_feasible=False``, the reason naming nulls). For a
    HARD null, use the dedicated null objective: a non-empty ``nulls`` with a non-``"multi"``
    objective routes to ``lcmv`` via :func:`design_auto`.

    Parameters
    ----------
    ds : RadiationDataset
        Phase-1 output.
    spec : TargetSpec
        The user request with ``engine == "auto"`` and ``objective == "multi"``; at least one of
        ``target_di_db`` / ``target_beamwidth_deg`` / ``target_inroom_slope_db_per_oct`` set.

    Returns
    -------
    DesignResult
        The chosen candidate's real ``design()`` output (numerically unchanged), with the honest
        multi-target report grafted onto ``attrs``: ``engine`` (concrete engine used),
        ``auto_selected``, ``auto_class="multi"``, ``multi_targets``, ``multi_weights``,
        ``multi_norm``, ``multi_trace`` (every candidate's achieved metrics + deviations + combined
        score), ``multi_achieved`` (per-objective achieved-vs-target), ``auto_reason``,
        ``auto_prescreen``, ``band_feasible``. ``spec`` is the original ``engine="auto"`` request.
    """
    from beamsim2.beamform.design import design  # lazy: design.py dispatches into this module

    targets = _multi_targets(spec)
    if not targets:
        raise ValueError(
            "objective='multi' needs at least one of target_di_db / target_beamwidth_deg / "
            "target_inroom_slope_db_per_oct."
        )
    weights = _multi_weights(spec, targets)

    obs = ds.directions
    c_sound = float(ds.attrs.get("speed_of_sound", 343.2))
    steer = np.asarray(spec.steer_dir, dtype=np.float64)
    steer = steer / np.linalg.norm(steer)
    target = build_target(spec, obs, ds.frequencies, c_sound=c_sound)  # for look/null indices
    null_idx, look_idx = target.null_idx, target.look_idx
    prescreen = _wng_prescreen(ds, spec, look_idx)

    cands: list[tuple[dict, object]] = []  # [(achieved+score dict, DesignResult)]
    for eng, knobs in _MULTI_LADDER:
        # objective="shape" on the candidate so the concrete engine never re-enters the auto path;
        # concrete engines ignore `objective` anyway, but this keeps the dispatch unambiguous.
        cand_spec = replace(spec, engine=eng, objective="shape", **knobs)
        r = design(ds, cand_spec)
        ach = _multi_achieved(r, ds, steer, null_idx, look_idx)
        ach["devs"] = _multi_devs(ach, targets)
        ach["combined"] = _combined(ach["devs"], weights)
        ach["worst_dev"] = max(ach["devs"].values()) if ach["devs"] else float("inf")
        ach["engine"] = r.attrs.get("engine", eng)
        ach["knobs"] = _knob_summary(eng, knobs)
        ach["feasible"] = bool(ach["feas_frac"] >= _FEAS_FRAC and _null_feasible(ach, spec))
        cands.append((ach, r))

    feasible = [(a, r) for a, r in cands if a["feasible"]]
    pool = feasible if feasible else cands  # honest best-effort when nothing meets the gates
    chosen_ach, chosen_r = min(pool, key=lambda ar: ar[0]["combined"])  # ladder order breaks ties
    band_feasible = bool(chosen_ach["feasible"]) and bool(chosen_r.attrs.get("band_feasible", True))

    report = _multi_achieved_report(chosen_ach, targets)
    chosen_r.attrs["engine"] = chosen_ach["engine"]
    chosen_r.attrs["auto_selected"] = True
    chosen_r.attrs["auto_class"] = "multi"
    chosen_r.attrs["multi_targets"] = targets
    chosen_r.attrs["multi_weights"] = weights
    chosen_r.attrs["multi_norm"] = dict(_NORM)
    chosen_r.attrs["multi_trace"] = [
        {
            "knobs": a["knobs"],
            "engine": a["engine"],
            "di_med": a["di_med"],
            "di_ptp": a["di_ptp"],
            "bw_med": a["bw_med"],
            "eir_slope": a["eir_slope"],
            "feas_frac": a["feas_frac"],
            "devs": a["devs"],
            "combined": a["combined"],
            "worst_dev": a["worst_dev"],
            "feasible": a["feasible"],
        }
        for a, _ in cands
    ]
    chosen_r.attrs["multi_achieved"] = report
    chosen_r.attrs["auto_reason"] = _multi_reason(chosen_ach, report, weights, band_feasible)
    chosen_r.attrs["auto_prescreen"] = prescreen
    chosen_r.attrs["band_feasible"] = band_feasible
    chosen_r.spec = spec  # echo the user's original auto request, not the overridden candidate
    return chosen_r
