"""V-AUTO (Chunk 3c gate): the Auto-Design orchestrator selects the expert-appropriate engine.

``engine="auto"`` dispatches to :mod:`beamsim2.beamform.orchestrator`, a *principled escalation
ladder* (the confirmed alternative to literal "try every algorithm and stack them";
``docs/Bug_Fix_Proposal.md`` Open Question 1). For each target class it runs a target-conditioned
candidate ladder through the real ``design()``, scores each on the target's OWN objective metric,
and picks the best feasible candidate — reporting its choice and where it is infeasible honestly.

The gate is deliberately **non-circular**: each scenario asserts not just that the expected engine
is chosen, but that the chosen engine *beats the runner-up on that target's own metric* in the
recorded ``auto_trace`` — a fact about engine behavior, not orchestrator wiring (see
``docs/Chunk3c_Findings.md`` for the measured margins). It also proves the honest-reporting and
cardinal-rule contracts hold on the new code path.

Engineering convention exp(-jwt), outgoing exp(+jkr) (the repo ``monopole_field``).
"""

from __future__ import annotations

import numpy as np

from beamsim2.assembly.tensor import build_dataset
from beamsim2.beamform.design import design
from beamsim2.beamform.targets import TargetSpec
from beamsim2.core.sphere import icosphere
from beamsim2.core.types import ComplexField
from beamsim2.validation.closed_loop import monopole_field

_C = 343.2
_D = 0.086  # compact end-fire spacing (m) — the cardioid/null fixture


# ---------------------------------------------------------------------------
# Fixtures (synthetic multi-monopole, CI-safe — no hardware/NumCalc)
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


def _trace(result, engine):
    """The auto_trace entry for a given candidate engine."""
    return next(t for t in result.attrs["auto_trace"] if t["engine"] == engine)


def _assert_honest_report(r, expected_engine, expected_class):
    """Every Auto-Design result must carry a complete, honest provenance report."""
    assert r.attrs["auto_selected"] is True
    assert r.attrs["engine"] == expected_engine  # the concrete engine actually used
    assert r.attrs["auto_class"] == expected_class
    assert isinstance(r.attrs["auto_reason"], str) and r.attrs["auto_reason"]
    # The full ladder is recorded (one entry per candidate the orchestrator tried).
    engines = {t["engine"] for t in r.attrs["auto_trace"]}
    assert expected_engine in engines and len(engines) >= 2
    # Per-band feasibility pre-screen is reported (the honest "where it's infeasible" grounding).
    assert "auto_prescreen" in r.attrs and "wng_ceiling_db" in r.attrs["auto_prescreen"]
    # The user's original auto request is echoed back, not the overridden candidate spec.
    assert r.spec.engine == "auto"


# ---------------------------------------------------------------------------
# SHAPE class -> least-squares (it best matches the requested pattern)
# ---------------------------------------------------------------------------
def test_auto_shape_selects_ls():
    """A cardioid SHAPE target on a compact pair: Auto-Design picks ls and it holds the cardioid."""
    band = np.geomspace(150.0, 600.0, 8)
    ds, obs = _dataset(_pair(), band)
    r = design(
        ds,
        TargetSpec(
            mode="preset",
            preset="cardioid",
            objective="shape",
            engine="auto",
            steer_dir=np.array([0.0, 0.0, 1.0]),
            wng_floor_db=-12.0,
        ),
    )
    _assert_honest_report(r, "ls", "shape")
    # Non-circular: ls wins on the shape metric (lower target_error) vs the runner-up engines.
    assert _trace(r, "ls")["te_med"] < _trace(r, "delay_sum")["te_med"]
    assert _trace(r, "ls")["te_med"] < _trace(r, "mvdr")["te_med"]
    # The chosen design actually realizes the cardioid across the band, peaking on-axis.
    look = int(np.argmax(obs.unit_vectors[:, 2]))
    assert np.allclose(r.metrics["di_db"], 4.77, atol=0.6)
    for fi in range(len(band)):
        assert int(np.argmax(np.abs(r.steered_field[fi]))) == look
    assert r.attrs["band_feasible"] is True


# ---------------------------------------------------------------------------
# CONSTANT_DIRECTIVITY class -> Luo constant-DI (it holds DI flat across the band)
# ---------------------------------------------------------------------------
def test_auto_constant_directivity_selects_constant_di():
    """A constant-directivity target on the cap: Auto-Design picks constant_di and DI is flat."""
    band = np.geomspace(2300.0, 4200.0, 6)
    ds, obs = _dataset(_cbt_cap(), band)
    r = design(
        ds,
        TargetSpec(
            objective="constant_directivity",
            engine="auto",
            steer_dir=np.array([0.0, 0.0, 1.0]),
            accept_halfangle_deg=45.0,
            wng_floor_db=0.0,
        ),
    )
    _assert_honest_report(r, "constant_di", "constant_directivity")
    # Non-circular: constant_di wins on flatness (smaller DI ptp) vs the shape/superdir rivals.
    assert _trace(r, "constant_di")["di_ptp"] < _trace(r, "ls")["di_ptp"]
    assert _trace(r, "constant_di")["di_ptp"] < _trace(r, "max_directivity")["di_ptp"]
    # The chosen design holds the directivity index constant by construction, at a directive level.
    assert np.ptp(r.metrics["di_db"]) < 0.1
    assert np.median(r.metrics["di_db"]) > 3.0  # not the omni trap
    assert r.attrs["band_feasible"] is True


# ---------------------------------------------------------------------------
# NULLS class -> LCMV (the only engine that places a hard null)
# ---------------------------------------------------------------------------
def test_auto_nulls_selects_lcmv():
    """A target with a hard null: Auto-Design picks lcmv and the rear is driven to a deep null."""
    band = np.geomspace(300.0, 1500.0, 6)
    ds, obs = _dataset(_pair(), band)
    r = design(
        ds,
        TargetSpec(
            mode="steering_only",
            nulls=[np.array([0.0, 0.0, -1.0])],
            engine="auto",
            steer_dir=np.array([0.0, 0.0, 1.0]),
            wng_floor_db=-6.0,
        ),
    )
    _assert_honest_report(r, "lcmv", "nulls")
    # Non-circular: lcmv wins on null depth (it nulls; ls/mvdr that ignore the null cannot).
    assert _trace(r, "lcmv")["null_worst"] < _trace(r, "ls")["null_worst"] - 50.0
    assert _trace(r, "lcmv")["null_worst"] < _trace(r, "mvdr")["null_worst"] - 50.0
    # The chosen design realizes a deep null at -z while keeping the main lobe at +z.
    look = int(np.argmax(obs.unit_vectors[:, 2]))
    null = int(np.argmin(obs.unit_vectors[:, 2]))
    for fi in range(len(band)):
        depth = 20.0 * np.log10(
            np.abs(r.steered_field[fi, null]) / np.abs(r.steered_field[fi, look])
        )
        assert depth < -40.0


# ---------------------------------------------------------------------------
# MAX_DIRECTIVITY class -> max_directivity (highest DI within the WNG floor)
# ---------------------------------------------------------------------------
def test_auto_max_directivity_selects_max_directivity():
    """A 'be as directive as possible' target on the cap: Auto-Design picks max_directivity."""
    band = np.geomspace(2300.0, 4200.0, 6)
    ds, obs = _dataset(_cbt_cap(), band)
    r = design(
        ds,
        TargetSpec(
            mode="steering_only",
            objective="max_directivity",
            engine="auto",
            steer_dir=np.array([0.0, 0.0, 1.0]),
            accept_halfangle_deg=45.0,
            wng_floor_db=-6.0,
        ),
    )
    _assert_honest_report(r, "max_directivity", "max_directivity")
    # Non-circular: the chosen engine is the most directive (it beats the robust anchors clearly).
    assert _trace(r, "max_directivity")["di_med"] > _trace(r, "delay_sum")["di_med"] + 1.0
    assert _trace(r, "max_directivity")["di_med"] > _trace(r, "ls")["di_med"] + 1.0
    # mvdr ties it numerically (both maximize DI subject to the floor); the fixed-order tie-break
    # deterministically prefers the canonical max-directivity formulation.
    assert abs(_trace(r, "max_directivity")["di_med"] - _trace(r, "mvdr")["di_med"]) < 0.25
    assert r.attrs["band_feasible"] is True


# ---------------------------------------------------------------------------
# Honest infeasibility — a superdirective ask the array cannot meet is FLAGGED, not faked
# ---------------------------------------------------------------------------
def test_auto_flags_infeasible_superdirective_request():
    """A +6 dB WNG floor on a 2-driver array (ceiling 10log10(2)=3.01 dB) is impossible for any
    engine -> Auto-Design returns a best-effort design and honestly flags band_feasible=False."""
    band = np.geomspace(150.0, 600.0, 6)
    ds, obs = _dataset(_pair(), band)
    r = design(
        ds,
        TargetSpec(
            mode="steering_only",
            objective="max_directivity",
            engine="auto",
            steer_dir=np.array([0.0, 0.0, 1.0]),
            wng_floor_db=6.0,
        ),
    )
    assert r.attrs["band_feasible"] is False
    assert not np.any(r.metrics["feasible_mask"])  # no bin meets the impossible floor
    assert "best-effort" in r.attrs["auto_reason"]
    # The pre-screen identifies every bin as above the array's WNG ceiling (physics, not solver).
    ps = r.attrs["auto_prescreen"]
    assert ps["floor_exceeds_ceiling_bins"] == ps["n_bins"]


# ---------------------------------------------------------------------------
# CARDINAL RULE — the new auto path still steers only via H's inter-driver phase
# ---------------------------------------------------------------------------
def test_auto_cardinal_rule_collapse_control():
    """Collapse both drivers to the origin (zero inter-driver phase): under engine='auto' the
    directivity must still collapse to ~0 — the orchestrator never re-zeros/min-phases a driver."""
    band = np.geomspace(150.0, 600.0, 8)
    ds, obs = _dataset([np.zeros(3), np.zeros(3)], band)
    r = design(
        ds,
        TargetSpec(
            mode="preset",
            preset="cardioid",
            objective="shape",
            engine="auto",
            steer_dir=np.array([0.0, 0.0, 1.0]),
            wng_floor_db=-12.0,
        ),
    )
    assert float(np.max(r.metrics["di_db"])) < 0.3, "DI must collapse with no inter-driver phase"


def test_auto_cardinal_rule_shared_ramp_invariant():
    """A shared modeling delay (common latency, all drivers) cannot change |P| on the auto path."""
    from beamsim2.assembly.tensor import stacked_h_full

    band = np.geomspace(150.0, 600.0, 8)
    ds, obs = _dataset(_pair(), band)
    r = design(
        ds,
        TargetSpec(
            mode="preset",
            preset="cardioid",
            objective="shape",
            engine="auto",
            steer_dir=np.array([0.0, 0.0, 1.0]),
            wng_floor_db=-12.0,
        ),
    )
    h = stacked_h_full(ds)  # [M, F, N]
    ramp = np.exp(-1j * 2.0 * np.pi * band * 2.9e-4)  # arbitrary shared delay
    p0 = np.sum(r.weights[:, :, None] * h, axis=0)  # [F, N]
    p1 = np.sum((r.weights * ramp[None, :])[:, :, None] * h, axis=0)  # [F, N]
    assert np.max(np.abs(np.abs(p0) - np.abs(p1))) < 1e-9
