"""V-cardioid (Chunk 3a gate): a 2-driver dual-opposed cardioid across a BAND.

This is the Chunk-3a finish line and the new cardinal-rule proof. A compact end-fire pair
(two monopoles at z = +/-0.043 m, d = 0.086 m) is driven, end-to-end through the LS
pressure-match designer, to a first-order cardioid. We assert it holds across a band — not
one bin — with realizable (smooth) filters and an honest white-noise-gain (WNG) floor, and
that the steering comes ENTIRELY from H's native inter-driver phase.

Why these specific tests (see ``docs/Chunk3a_Findings.md``):
- The bare DI/null band test is a regression lock, NOT a discriminator: the pre-3a code at
  ``wng_floor=-60`` already made a per-frequency cardioid (LS absorbs the global phase).
- The DISCRIMINATORS — what 3a actually fixed — are filter *realizability* (the pre-3a
  filters measured 0.47 rad of phase curvature; the fixed ones ~0.02 rad) and an *honest*
  WNG floor (pre-3a default robustness collapsed DI to ~2.5 dB; the fix rolls off gracefully).
- The cardinal-rule proof is the collapse-to-origin control: with zero inter-driver phase the
  cardioid dies to DI 0.

Engineering convention exp(-jwt), outgoing exp(+jkr) (the repo ``monopole_field``).
"""

from __future__ import annotations

import numpy as np
import pytest

from beamsim2.assembly.tensor import build_dataset, stacked_h_full
from beamsim2.beamform.design import design
from beamsim2.beamform.targets import TargetSpec, build_target
from beamsim2.beamform.weights import (
    ls_pressure_match,
    ls_pressure_match_coupled,
    phase_roughness,
)
from beamsim2.core.sphere import icosphere
from beamsim2.core.types import ComplexField
from beamsim2.validation.closed_loop import monopole_field

_C = 343.2
_D = 0.086  # inter-driver spacing (m); cardioid band is roughly k*d < pi -> ~150-670 Hz


def _dual_opposed_dataset(freqs, positions=None):
    """Build a RadiationDataset for a dual-opposed monopole pair on an icosphere-2562 grid."""
    obs = icosphere(4)
    if positions is None:
        positions = [np.array([0.0, 0.0, -_D / 2]), np.array([0.0, 0.0, _D / 2])]
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


def _rear_null_db(P_f, rear_idx):
    """On-axis-relative level (dB) at the rear grid point (negative = deep null)."""
    return 20.0 * np.log10(np.abs(P_f[rear_idx]) / np.max(np.abs(P_f)))


_BAND = np.geomspace(150.0, 600.0, 8)  # comfortably inside the achievable cardioid band


# ---------------------------------------------------------------------------
# Headline gate: cardioid held across the band (regression lock)
# ---------------------------------------------------------------------------
def test_cardioid_held_across_band():
    """DI ~ 4.77 dB and a deep rear null at every bin across the band, not just one bin."""
    ds, obs = _dual_opposed_dataset(_BAND)
    rear = int(np.argmin(obs.unit_vectors[:, 2]))
    spec = TargetSpec(
        mode="preset",
        preset="cardioid",
        steer_dir=np.array([0.0, 0.0, 1.0]),
        engine="ls",
        wng_floor_db=-12.0,
    )
    r = design(ds, spec)
    for fi in range(len(_BAND)):
        di = r.metrics["di_db"][fi]
        rn = _rear_null_db(r.steered_field[fi], rear)
        assert di == pytest.approx(4.77, abs=0.5), f"{_BAND[fi]:.0f} Hz: DI={di:.2f}"
        assert rn <= -22.0, f"{_BAND[fi]:.0f} Hz: rear null only {rn:.1f} dB"
    # The main lobe points at +z at every bin.
    look = int(np.argmax(obs.unit_vectors[:, 2]))
    for fi in range(len(_BAND)):
        assert int(np.argmax(np.abs(r.steered_field[fi]))) == look


# ---------------------------------------------------------------------------
# DISCRIMINATOR 1: realizable (smooth) filters — the cross-frequency fix
# ---------------------------------------------------------------------------
def test_filters_are_realizable():
    """Filter phase bends gently across frequency (short, causal filters).

    The pre-3a real-target LS produced ~0.47 rad of 2nd-difference phase curvature on this
    exact fixture (``docs/Chunk3a_Findings.md``); the complex virtual-source target brings it
    to ~0.02 rad. A threshold of 0.1 rad fails the old code and passes the new.
    """
    ds, obs = _dual_opposed_dataset(_BAND)
    spec = TargetSpec(
        mode="preset",
        preset="cardioid",
        steer_dir=np.array([0.0, 0.0, 1.0]),
        engine="ls",
        wng_floor_db=-12.0,
    )
    r = design(ds, spec)
    # Allow one shared modeling delay (a common latency is permitted; cardinal-rule safe).
    tau = float(r.attrs.get("ls_tau_s", 0.0))
    rough = phase_roughness(r.weights, _BAND, tau)
    assert rough < 0.1, f"filters too rough: {rough:.3f} rad (pre-3a was ~0.47)"


# ---------------------------------------------------------------------------
# DISCRIMINATOR 2: honest WNG floor — graceful roll-off, not collapse
# ---------------------------------------------------------------------------
def test_robust_wng_floor_binds_gracefully():
    """A binding WNG floor is respected and trades directivity gracefully (no collapse).

    At a robust floor the low bins (where the unconstrained beam is superdirective) must give
    up directivity to honor the floor — but degrade gracefully, never collapse to noise. The
    pre-3a heuristic collapsed DI to ~2.5 dB at the default robustness; the honest floor keeps
    a usable beam while actually meeting the floor.
    """
    ds, obs = _dual_opposed_dataset(_BAND)
    spec = TargetSpec(
        mode="preset",
        preset="cardioid",
        steer_dir=np.array([0.0, 0.0, 1.0]),
        engine="ls",
        wng_floor_db=-3.0,
    )
    r = design(ds, spec)
    wng = r.metrics["wng_db"]
    di = r.metrics["di_db"]
    # Floor respected at every bin (small tolerance for the grid step).
    assert np.all(wng >= -3.0 - 0.6), f"WNG floor violated: min {wng.min():.2f} dB"
    # Directivity rolls off but never collapses to noise.
    assert np.all(di > 1.5), f"DI collapsed: min {di.min():.2f} dB"

    # Compare to a fragile floor: it buys more low-f directivity at the cost of a worse WNG.
    fragile = design(
        ds,
        TargetSpec(
            mode="preset",
            preset="cardioid",
            steer_dir=np.array([0.0, 0.0, 1.0]),
            engine="ls",
            wng_floor_db=-60.0,
        ),
    )
    assert fragile.metrics["di_db"][0] > di[0]  # fragile is sharper at the lowest bin...
    assert fragile.metrics["wng_db"][0] < wng[0]  # ...but less robust there (lower WNG)


# ---------------------------------------------------------------------------
# CARDINAL-RULE PROOFS
# ---------------------------------------------------------------------------
def test_cardinal_rule_collapse_control():
    """Decisive proof: with NO inter-driver phase, the cardioid dies (steering is all in H).

    Collapse both drivers onto the origin (identical H rows -> zero inter-driver phase) and run
    the SAME cardioid design. With no spatial diversity the array can only radiate ~omni, so
    DI -> 0. This is the strongest single-phase-origin guard: any code path that re-zeroed a
    driver would *also* zero its inter-driver phase and fail here.
    """
    ds, obs = _dual_opposed_dataset(_BAND, positions=[np.zeros(3), np.zeros(3)])
    spec = TargetSpec(
        mode="preset",
        preset="cardioid",
        steer_dir=np.array([0.0, 0.0, 1.0]),
        engine="ls",
        wng_floor_db=-12.0,
    )
    r = design(ds, spec)
    assert float(np.max(r.metrics["di_db"])) < 0.3, "DI should collapse with no inter-driver phase"


def test_cardinal_rule_shared_ramp_invariant():
    """A shared modeling delay (common latency, all drivers) cannot change |P| (cardinal-rule)."""
    ds, obs = _dual_opposed_dataset(_BAND)
    spec = TargetSpec(
        mode="preset",
        preset="cardioid",
        steer_dir=np.array([0.0, 0.0, 1.0]),
        engine="ls",
        wng_floor_db=-12.0,
    )
    r = design(ds, spec)
    h = stacked_h_full(ds)  # [M,F,N]
    ramp = np.exp(-1j * 2.0 * np.pi * _BAND * 3.7e-4)  # arbitrary shared delay
    P0 = np.sum(r.weights[:, :, None] * h, axis=0)  # [F,N]
    P1 = np.sum((r.weights * ramp[None, :])[:, :, None] * h, axis=0)  # [F,N]
    assert np.max(np.abs(np.abs(P0) - np.abs(P1))) < 1e-9


# ---------------------------------------------------------------------------
# Complex, frequency-dependent target (defect #1 fix)
# ---------------------------------------------------------------------------
def test_cardioid_target_is_complex_and_frequency_dependent():
    """build_target for a cardioid is a complex, frequency-dependent virtual-source field."""
    freqs = np.array([200.0, 400.0, 800.0])
    ds, obs = _dual_opposed_dataset(freqs)
    spec = TargetSpec(mode="preset", preset="cardioid", steer_dir=np.array([0.0, 0.0, 1.0]))
    target = build_target(spec, obs, freqs, c_sound=_C)
    b = target.b_field  # [F, N]
    # It must be genuinely complex (not the old real broadcast) and vary across frequency.
    assert np.max(np.abs(b.imag)) > 1e-3, "target field is purely real (defect #1 not fixed)"
    assert np.max(np.abs(b[0] - b[1])) > 1e-6, "target field is frequency-independent"
    # And it is the correct first-order shape: b / g_mono == 0.5 + 0.5 cos, phase-referenced.
    g_mono = monopole_field(np.zeros((1, 3)), obs, freqs, _C)[0]  # [F, N]
    cos_ang = obs.unit_vectors @ np.array([0.0, 0.0, 1.0])  # [N]
    ideal = 0.5 + 0.5 * cos_ang  # [N] signed cardioid shape
    for fi in range(len(freqs)):
        assert np.max(np.abs(b[fi] / g_mono[fi] - ideal)) < 1e-6
        look = int(np.argmax(cos_ang))
        assert abs(float(np.angle(b[fi, look] / g_mono[fi, look]))) < 1e-9


def test_custom_complex_target_preserved():
    """A complex custom target is no longer silently cast through np.real (defect #1)."""
    freqs = np.array([500.0])
    ds, obs = _dual_opposed_dataset(freqs)
    n = obs.unit_vectors.shape[0]
    custom = (np.linspace(0.5, 1.0, n) + 0.4j * np.ones(n)).astype(np.complex128)  # [N] complex
    spec = TargetSpec(mode="custom", custom_target=custom, steer_dir=np.array([0.0, 0.0, 1.0]))
    target = build_target(spec, obs, freqs, c_sound=_C)
    # The custom magnitude shape survives (a real-cast would zero the imaginary content's effect).
    assert np.max(np.abs(np.abs(target.b_field[0]) - np.abs(custom))) < 1e-9


# ---------------------------------------------------------------------------
# Frequency-coupling unit tests (the gate does NOT exercise these — test directly)
# ---------------------------------------------------------------------------
def test_coupling_reduces_to_per_bin_at_mu_zero():
    """mu=0, tau=0 coupled solve equals F independent per-bin ls_pressure_match solves."""
    freqs = np.geomspace(200.0, 800.0, 6)
    ds, obs = _dual_opposed_dataset(freqs)
    h = stacked_h_full(ds)  # [M,F,N]
    spec = TargetSpec(mode="preset", preset="cardioid", steer_dir=np.array([0.0, 0.0, 1.0]))
    b = build_target(spec, obs, freqs, c_sound=_C).b_field  # [F,N]
    lam = np.full(len(freqs), 1e-3)
    w_coupled = ls_pressure_match_coupled(h, b, obs.weights, lam, mu=0.0, freqs=freqs, tau=0.0)
    for fi in range(len(freqs)):
        w_bin = ls_pressure_match(h[:, fi, :], b[fi], obs.weights, lam[fi])
        assert np.allclose(w_coupled[:, fi], w_bin, atol=1e-10)


def _complex_curvature(w):
    """Normalized RMS of the complex 2nd-difference of ``w[M,F]`` across frequency.

    This is exactly what the frequency coupling penalizes; a smaller value means a smoother
    complex weight trajectory ``w_m(f)``, i.e. a shorter / more realizable filter.
    """
    d2 = w[:, 2:] - 2.0 * w[:, 1:-1] + w[:, :-2]  # [M, F-2]
    return float(np.sqrt(np.sum(np.abs(d2) ** 2) / np.sum(np.abs(w) ** 2)))


def test_coupling_reduces_curvature_preserving_null():
    """Increasing mu reduces the complex weight curvature (shorter filters) while keeping the
    cardioid null — the role the coupling plays as 3b insurance.

    NOTE (``docs/Chunk3a_Findings.md``): for this well-posed compact array the complex
    virtual-source target already makes the per-bin weights smooth, so the coupling's effect is
    small and the realizability of the 3a gate comes from the target, not the coupling. This
    test confirms the coupling mechanism is wired correctly and is beam-preserving — it does NOT
    claim the coupling is what passes the gate.
    """
    freqs = np.geomspace(200.0, 800.0, 10)
    ds, obs = _dual_opposed_dataset(freqs)
    h = stacked_h_full(ds)  # [M,F,N]
    rear = int(np.argmin(obs.unit_vectors[:, 2]))
    spec = TargetSpec(mode="preset", preset="cardioid", steer_dir=np.array([0.0, 0.0, 1.0]))
    b = build_target(spec, obs, freqs, c_sound=_C).b_field  # [F,N] complex virtual-source
    lam = np.zeros(len(freqs))
    trace = float(
        np.mean(
            [
                np.real(np.trace((np.conj(h[:, fi, :]) * obs.weights) @ h[:, fi, :].T))
                for fi in range(len(freqs))
            ]
        )
    )
    w0 = ls_pressure_match_coupled(h, b, obs.weights, lam, mu=0.0, freqs=freqs, tau=0.0)
    w1 = ls_pressure_match_coupled(
        h, b, obs.weights, lam, mu=1.0 * trace / 6.0, freqs=freqs, tau=0.0
    )
    assert _complex_curvature(w1) < _complex_curvature(w0), "coupling did not reduce curvature"
    # The rear null still holds after coupling (beam-preserving).
    P1 = np.sum(w1[:, :, None] * h, axis=0)  # [F,N]
    for fi in range(len(freqs)):
        assert _rear_null_db(P1[fi], rear) <= -20.0


# ---------------------------------------------------------------------------
# frac_mu re-validation (Chunk-3b carry-forward of the 3a open action)
# ---------------------------------------------------------------------------
def test_ls_frac_mu_default_is_active_and_beam_safe_on_under_determined_stressor():
    """The shipped ``frac_mu=1e-2`` meaningfully smooths an under-determined LS solve, beam-safely.

    3a's findings flagged that ``frac_mu=1e-2`` is near-inert on the well-posed 2-driver cardioid
    (the integration path never exercised the coupling). 3b re-validates the default on a fixture
    where it matters: a 3-driver near-collinear array driven BELOW its comfortable band
    (80-500 Hz, supercardioid, a strict WNG floor) makes the per-bin WNG-floor loading swing across
    frequency, giving the cross-frequency coupling real authority. Here ``frac_mu=1e-2`` cuts the
    cross-frequency phase roughness meaningfully while preserving the beam (DI drift small), so the
    default is kept (``docs/Chunk3b_Findings.md``).
    """
    from beamsim2.beamform.design import _design_ls_coupled
    from beamsim2.beamform.forward import steered_field
    from beamsim2.validation.power_di import directivity_index

    freqs = np.geomspace(80.0, 500.0, 12)
    pos = [np.array([0.0, 0.0, -0.05]), np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.05])]
    ds, obs = _dual_opposed_dataset(freqs, positions=pos)
    h = stacked_h_full(ds)  # [M,F,N]
    spec = TargetSpec(
        mode="preset",
        preset="supercardioid",
        steer_dir=np.array([0.0, 0.0, 1.0]),
        engine="ls",
        wng_floor_db=-3.0,
    )
    target = build_target(spec, obs, freqs, c_sound=_C)
    look = target.look_idx

    b = target.b_field
    w0, feas0, _, tau0 = _design_ls_coupled(h, b, obs.weights, freqs, look, -3.0, frac_mu=0.0)
    w1, feas1, _, tau1 = _design_ls_coupled(h, b, obs.weights, freqs, look, -3.0, frac_mu=1e-2)

    # (1) The coupling is ACTIVE here: it cuts the worst-driver cross-frequency phase roughness.
    r0 = phase_roughness(w0, freqs, tau0)
    r1 = phase_roughness(w1, freqs, tau1)
    assert r1 < 0.85 * r0, f"frac_mu=1e-2 inert on the stressor: roughness {r0:.3f} -> {r1:.3f}"

    # (2) It is BEAM-SAFE: the directivity is essentially unchanged at every bin.
    di0 = directivity_index(steered_field(h, w0), obs.weights)
    di1 = directivity_index(steered_field(h, w1), obs.weights)
    assert np.max(np.abs(di1 - di0)) < 0.5, "coupling shifted the beam too much"
    assert np.all(feas1), "the stressor band should stay feasible at the -3 dB floor"
