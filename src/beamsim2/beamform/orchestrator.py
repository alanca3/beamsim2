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

    A non-empty ``nulls`` always dominates (a hard-null request is a null target regardless of
    ``objective``); otherwise the explicit ``objective`` decides, defaulting to ``"shape"``.
    """
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
