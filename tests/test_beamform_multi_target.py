"""V-MULTI (Chunk 3d gate): multi-target Auto-Design trades {DI, beamwidth, in-room} sensibly.

``engine="auto"`` + ``objective="multi"`` dispatches the scalarized weighted-sum search in
:func:`beamsim2.beamform.orchestrator.design_multi`. It runs a curated (engine, knob) ladder
through the real ``design()``, scores each candidate's achieved field on a weighted sum of
*normalized* per-objective deviations (DI vs ``target_di_db``, -6 dB beamwidth vs
``target_beamwidth_deg``, CEA-2034-A in-room EIR slope vs ``target_inroom_slope_db_per_oct``),
gates on the honest WNG floor / nulls, and picks the best feasible candidate.

The gate is deliberately **non-circular**. The load-bearing claim is a **minimax fact about the
achieved fields**, independent of the selector: the balanced design's *worst* normalized
per-objective deviation is strictly lower than the worst deviation of every *single-objective*
optimum (the DI-best, beamwidth-best, and in-room-best candidates). That can only hold because the
objectives genuinely conflict on the fixture (r(DI,BW) = -0.89; ``docs/Chunk3d_Findings.md``) — it
is NOT the tautology "argmin(combined) wins the combined score" (which is also reported, as the
trivial corollary of the kickoff's literal acceptance wording). Plus an explicit trade check
(3c-style behavioral facts), an end-to-end Pareto weight-trace, the honest report, and the
cardinal-rule controls on the new path.

Engineering convention exp(-jwt), outgoing exp(+jkr) (the repo ``monopole_field``).
"""

from __future__ import annotations

import numpy as np
import pytest

from beamsim2.assembly.tensor import build_dataset, stacked_h_full
from beamsim2.beamform.design import design
from beamsim2.beamform.targets import TargetSpec
from beamsim2.core.sphere import icosphere
from beamsim2.core.types import ComplexField
from beamsim2.validation.closed_loop import monopole_field

_C = 343.2
_D = 0.086  # compact end-fire spacing (m) — the cardinal-control pair


# ---------------------------------------------------------------------------
# Fixtures (synthetic multi-monopole, CI-safe — no hardware/NumCalc); verbatim from V-CBT/V-AUTO.
# ---------------------------------------------------------------------------
def _dataset(positions, freqs):
    obs = icosphere(4)
    H = monopole_field(np.asarray(positions, float), obs, np.asarray(freqs, float), c=_C)
    inputs = [
        (
            f"d{i}",
            ComplexField(
                frequencies=np.asarray(freqs, float),
                pressure=H[i],
                convergence_flags=np.ones(len(freqs), bool),
            ),
            {"name": f"d{i}", "position": list(positions[i])},
        )
        for i in range(len(positions))
    ]
    ds = build_dataset(inputs, obs, root_attrs={"phase_origin": [0, 0, 0], "speed_of_sound": _C})
    return ds, obs


def _pair():
    return [np.array([0.0, 0.0, -_D / 2]), np.array([0.0, 0.0, _D / 2])]


def _cbt_cap(Rc=0.12, theta0_deg=45.0, n_rings=6, dx=0.035):
    """~50-element spherical cap (the V-CBT constant-directivity fixture)."""
    th0 = np.deg2rad(theta0_deg)
    pts = []
    for ir in range(n_rings):
        x = ir / (n_rings - 1)
        psi = th0 * x
        n_phi = 1 if ir == 0 else max(4, int(round(2 * np.pi * Rc * np.sin(psi) / dx)))
        for k in range(n_phi):
            ph = 2 * np.pi * k / n_phi
            pts.append(
                Rc * np.array([np.sin(psi) * np.cos(ph), np.sin(psi) * np.sin(ph), np.cos(psi)])
            )
    return np.array(pts)


# The gate array + band: the ~50-driver cap in the flat-CBT band where beamwidth is callable.
_BAND = np.geomspace(2300.0, 4200.0, 6)
_AXIS = np.array([0.0, 0.0, 1.0])

# A genuinely 3-way-conflicting target: a DIRECTIVE beam that is ALSO WIDE with a FLAT in-room tilt.
# No single candidate satisfies all three (high DI forces a narrow beam; a wide beam forces low DI),
# so the scalarized optimum must trade — which is exactly what makes the minimax gate non-vacuous.
_TARGETS = dict(target_di_db=16.0, target_beamwidth_deg=55.0, target_inroom_slope_db_per_oct=-1.0)


def _multi_spec(weights=None, **over):
    kw = dict(engine="auto", objective="multi", steer_dir=_AXIS, wng_floor_db=-6.0, **_TARGETS)
    kw.update(over)
    if weights is not None:
        kw["objective_weights"] = weights
    return TargetSpec(**kw)


@pytest.fixture(scope="module")
def balanced():
    """One balanced multi-target design on the cap, shared across the cheap inspection tests."""
    ds, obs = _dataset(_cbt_cap(), _BAND)
    return ds, obs, design(ds, _multi_spec())


def _extremes(trace):
    """The single-objective optima + the balanced (min-combined feasible) point, from one trace.

    All three are points IN the trace; the comparison below is on their *achieved-field* deviations,
    not on the selection metric, so it is a fact about the engines, not the orchestrator wiring.
    """
    feas = [t for t in trace if t["feasible"]]
    chosen = min(feas, key=lambda t: t["combined"])
    di_ext = min(trace, key=lambda t: t["devs"]["di"])
    bw_ext = min(trace, key=lambda t: t["devs"]["beamwidth"])
    ir_ext = min(trace, key=lambda t: t["devs"]["inroom"])
    return chosen, di_ext, bw_ext, ir_ext


# ---------------------------------------------------------------------------
# Headline gate: the NON-CIRCULAR minimax trade (load-bearing) + explicit trade
# ---------------------------------------------------------------------------
def test_multi_target_minimax_trade_is_non_circular(balanced):
    """Balanced design's WORST normalized deviation < every single-objective optimum's worst.

    This is a fact about the achieved fields (worst_dev is max over objectives), distinct from the
    combined (mean) score the selector minimizes — so it is NOT the tautology argmin(combined) wins
    combined. It can only hold because the objectives conflict on this fixture.
    """
    _ds, _obs, r = balanced
    trace = r.attrs["multi_trace"]
    chosen, di_ext, bw_ext, ir_ext = _extremes(trace)

    # (1) the non-circular minimax win, with margin.
    assert chosen["worst_dev"] < di_ext["worst_dev"], (chosen["worst_dev"], di_ext["worst_dev"])
    assert chosen["worst_dev"] < bw_ext["worst_dev"], (chosen["worst_dev"], bw_ext["worst_dev"])
    assert chosen["worst_dev"] < ir_ext["worst_dev"], (chosen["worst_dev"], ir_ext["worst_dev"])

    # (2) the explicit trade (3c-style behavioral facts): the balanced point sits genuinely BETWEEN
    # the extremes — more directive than the wide-beam optimum, closer in beamwidth AND flatter
    # in-room than the directivity optimum.
    assert chosen["di_med"] > bw_ext["di_med"]
    assert chosen["devs"]["beamwidth"] < di_ext["devs"]["beamwidth"]
    assert chosen["devs"]["inroom"] < di_ext["devs"]["inroom"]

    # (3) the chosen engine is one of the directive families (property, NOT the exact knob: the top
    # two candidates score within ~0.02, so asserting the precise target_gdi_db would be brittle).
    assert r.attrs["engine"] in ("constant_di", "max_directivity")

    # (4) in-room is an ACTIVE, separately-pulling axis here (not a silent restatement of DI): the
    # in-room optimum is a different candidate from the DI optimum, with a clearly different tilt.
    assert ir_ext["knobs"] != di_ext["knobs"]
    assert abs(ir_ext["eir_slope"] - di_ext["eir_slope"]) > 0.5


def test_multi_target_literal_combined_score_corollary(balanced):
    """The kickoff's literal acceptance ("beats the extremes on the COMBINED score").

    This is the *trivial* corollary of argmin(combined) and is reported only so the written
    acceptance line is visibly satisfied; the load-bearing, non-circular claim is the minimax test.
    """
    _ds, _obs, r = balanced
    trace = r.attrs["multi_trace"]
    chosen, di_ext, bw_ext, ir_ext = _extremes(trace)
    for ext in (di_ext, bw_ext, ir_ext):
        assert chosen["combined"] <= ext["combined"] + 1e-9


# ---------------------------------------------------------------------------
# Honest report: per-objective achieved-vs-target + the chosen (engine, knobs)
# ---------------------------------------------------------------------------
def test_multi_target_honest_report(balanced):
    """Every multi-target result carries a complete, honest per-objective report."""
    _ds, _obs, r = balanced
    assert r.attrs["auto_selected"] is True
    assert r.attrs["auto_class"] == "multi"
    assert r.spec.engine == "auto"  # the user's original request is echoed back
    assert isinstance(r.attrs["auto_reason"], str) and r.attrs["auto_reason"]

    # active targets recorded; weights default to balanced.
    assert r.attrs["multi_targets"] == {"di": 16.0, "beamwidth": 55.0, "inroom": -1.0}
    assert r.attrs["multi_weights"] == {"di": 1.0, "beamwidth": 1.0, "inroom": 1.0}

    # per-objective achieved-vs-target for all three objectives.
    rep = r.attrs["multi_achieved"]
    for k in ("di", "beamwidth", "inroom"):
        assert set(rep[k]) == {"target", "achieved", "dev_norm"}
    # the report's achieved values match the realized field metrics (no bookkeeping drift).
    assert rep["di"]["achieved"] == pytest.approx(float(np.median(r.metrics["di_db"])), abs=1e-6)

    # full ladder recorded (one trace entry per candidate the search tried) + prescreen grounding.
    assert len(r.attrs["multi_trace"]) >= 5
    assert "wng_ceiling_db" in r.attrs["auto_prescreen"]
    assert r.attrs["band_feasible"] is True


# ---------------------------------------------------------------------------
# Pareto: sliding the weights traces a sensible trade-off (end-to-end, the real selector moves)
# ---------------------------------------------------------------------------
def test_multi_target_pareto_weight_trace():
    """DI-heavy weights -> a more directive (narrower) design; beamwidth-heavy -> a wider one."""
    ds, _obs = _dataset(_cbt_cap(), _BAND)
    r_di = design(ds, _multi_spec(weights={"di": 4.0, "beamwidth": 1.0, "inroom": 1.0}))
    r_bw = design(ds, _multi_spec(weights={"di": 1.0, "beamwidth": 4.0, "inroom": 1.0}))
    di_di = r_di.attrs["multi_achieved"]["di"]["achieved"]
    di_bw = r_bw.attrs["multi_achieved"]["di"]["achieved"]
    bw_di = r_di.attrs["multi_achieved"]["beamwidth"]["achieved"]
    bw_bw = r_bw.attrs["multi_achieved"]["beamwidth"]["achieved"]
    # A genuine Pareto trade-off: more directivity weight buys DI at the cost of beamwidth.
    assert di_di > di_bw + 1.0, (di_di, di_bw)
    assert bw_di < bw_bw - 1.0, (bw_di, bw_bw)


# ---------------------------------------------------------------------------
# CARDINAL-RULE PROOFS on the multi-target path (steering is entirely in H's inter-driver phase)
# ---------------------------------------------------------------------------
def test_multi_target_cardinal_rule_collapse():
    """Collapse the cap to a sub-mm cluster (no inter-driver phase) -> the chosen design's DI -> ~0.

    Any code path that re-zeroed a driver would also zero its inter-driver phase and (correctly)
    fail here, so this is the strongest single-phase-origin guard on the new path.
    """
    rng = np.random.default_rng(0)
    tiny = 5e-4 * rng.standard_normal((len(_cbt_cap()), 3))  # 0.5 mm cluster
    ds, _obs = _dataset(tiny, _BAND)
    r = design(ds, _multi_spec())
    assert (
        float(np.nanmax(r.metrics["di_db"])) < 0.5
    ), "DI should collapse with no inter-driver phase"


def test_multi_target_cardinal_rule_shared_ramp_invariant(balanced):
    """A shared modeling delay (common latency, all drivers) cannot change |P| (cardinal-rule)."""
    ds, _obs, r = balanced
    h = stacked_h_full(ds)  # [M, F, N]
    ramp = np.exp(-1j * 2.0 * np.pi * _BAND * 4.1e-4)  # arbitrary shared delay
    P0 = np.sum(r.weights[:, :, None] * h, axis=0)  # [F, N]
    P1 = np.sum((r.weights * ramp[None, :])[:, :, None] * h, axis=0)  # [F, N]
    assert np.max(np.abs(np.abs(P0) - np.abs(P1))) < 1e-9


# ---------------------------------------------------------------------------
# Contract: multi needs a target; nulls is a feasibility GATE (not a class override)
# ---------------------------------------------------------------------------
def test_multi_target_requires_a_target():
    """objective='multi' with no target set is a usage error, not a silent no-op."""
    ds, _obs = _dataset(_cbt_cap(), _BAND)
    with pytest.raises(ValueError, match="needs at least one"):
        design(ds, TargetSpec(engine="auto", objective="multi", steer_dir=_AXIS))


def test_multi_target_nulls_are_a_feasibility_gate_not_a_class_override():
    """Under objective='multi', a requested null does NOT switch to the null class (deliberate)."""
    ds, _obs = _dataset(_pair(), np.geomspace(150.0, 600.0, 6))
    spec = _multi_spec(
        target_di_db=4.0,  # achievable on the 2-driver pair; cap-DI targets are not
        target_beamwidth_deg=None,
        target_inroom_slope_db_per_oct=None,
        nulls=[np.array([0.0, 0.0, -1.0])],  # rear null
    )
    r = design(ds, spec)
    assert r.attrs["auto_class"] == "multi"  # nulls did NOT override the multi class
