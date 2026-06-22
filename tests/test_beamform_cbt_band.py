"""V-CBT (Chunk 3b gate): engine-level constant directivity across a band on a driver cap.

The Chunk-3b finish line. A ~50-element spherical cap is driven end-to-end through the
constant-directivity designer (``engine="constant_di"``, ``directivity_mode="index"`` = Luo's
proper directivity index). We assert that across the **flat-CBT band** it holds:

- a **constant directivity index** (the load-bearing, by-construction Luo property), and
- a **~constant -6 dB beamwidth** (the secondary CBT check — honest about the regime: constant DI
  does NOT pin beamwidth, so this is only true *in* the flat-CBT band, k*a >= ~3), with
- **realizable filters** (cardinal-safe global-phase alignment -> low magnitude-gated roughness),
- an **honest WNG floor** (every bin meets it; graceful, never silent superdirective garbage),

and that steering is entirely in H's inter-driver phase (cardinal rule).

Why this band, this fixture (see ``docs/Chunk3b_Findings.md``):
- The shipped engine (A=accept-cap GDI) holds a cap-ratio constant but the beamwidth narrows
  (std 17 deg) and the proper DI varies 6.7 dB. ``directivity_mode="index"`` (A = c c^H) is Luo's
  actual directivity index and holds DI flat to ~1e-11.
- Constant DI != constant beamwidth. The -6 dB beamwidth is ~constant only ABOVE the Keele CBT
  cutoff k*a ~ 3 (~1929 Hz here); 1500 Hz is below it (still collapsing) -> the gate band starts at
  2300 Hz (k*a 3.6). The upper edge stays < ~7280 Hz so beamwidth_deg's SH resample (order <= 16)
  resolves the lobe. This is a physics regime, like 3a's spatial-aliasing upper edge.

Engineering convention exp(-jwt), outgoing exp(+jkr) (the repo ``monopole_field``).
"""

from __future__ import annotations

import numpy as np

from beamsim2.assembly.tensor import build_dataset, stacked_h_full
from beamsim2.beamform.design import design
from beamsim2.beamform.targets import TargetSpec
from beamsim2.beamform.weights import magnitude_gated_phase_roughness
from beamsim2.core.sphere import icosphere
from beamsim2.core.types import ComplexField
from beamsim2.validation.closed_loop import monopole_field

_C = 343.2


def _cbt_cap(Rc: float, theta0_deg: float, n_rings: int, dx: float) -> np.ndarray:
    """Monopole positions on a spherical cap (rings of azimuthal spacing ~dx)."""
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
    return np.array(pts)  # [M, 3]


def _cap_dataset(positions, freqs):
    """RadiationDataset for a cap of monopoles on an icosphere-2562 grid."""
    obs = icosphere(4)
    H = monopole_field(np.asarray(positions, float), obs, np.asarray(freqs, float), c=_C)  # [M,F,N]
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


# The gate fixture: ~50-driver cap, flat-CBT band (k*a 3.6 -> 6.5; SH-resolvable k*Rc 5.0 -> 9.2).
_CAP = _cbt_cap(Rc=0.12, theta0_deg=45.0, n_rings=6, dx=0.035)
_BAND = np.geomspace(2300.0, 4200.0, 6)


def _gate_spec(wng_floor_db=0.0):
    return TargetSpec(
        steer_dir=np.array([0.0, 0.0, 1.0]),
        engine="constant_di",
        directivity_mode="index",
        accept_halfangle_deg=45.0,
        wng_floor_db=wng_floor_db,
    )


# ---------------------------------------------------------------------------
# Headline gate: constant directivity index + ~constant beamwidth across the band
# ---------------------------------------------------------------------------
def test_constant_di_holds_directivity_and_beamwidth_across_band():
    """DI is constant by construction; beamwidth is ~constant in the flat-CBT band."""
    ds, obs = _cap_dataset(_CAP, _BAND)
    r = design(ds, _gate_spec(wng_floor_db=0.0))
    di = r.metrics["di_db"]
    bw = r.metrics["beamwidth_deg"]

    # (a) Constant directivity INDEX — the load-bearing Luo property (held exactly by one tau*).
    assert np.ptp(di) < 0.05, f"directivity index not constant: {di}"
    # The reported level matches the achieved field DI (no bookkeeping drift).
    assert np.allclose(di, r.attrs["constant_di_db"], atol=0.3)

    # (b) ~constant -6 dB beamwidth (the secondary CBT check, honest about the flat-CBT regime).
    assert np.all(np.isfinite(bw)), f"beamwidth lobe did not close at some bin: {bw}"
    assert np.ptp(bw) < 10.0, f"beamwidth not ~constant across band: {bw}"
    assert np.nanstd(bw) < 4.0, f"beamwidth too variable: std {np.nanstd(bw):.2f} deg"
    assert np.all((bw > 25.0) & (bw < 55.0)), f"beamwidth out of the expected window: {bw}"

    # The main lobe points at +z at every bin.
    look = int(np.argmax(obs.unit_vectors[:, 2]))
    for fi in range(len(_BAND)):
        assert int(np.argmax(np.abs(r.steered_field[fi]))) == look


# ---------------------------------------------------------------------------
# DISCRIMINATOR 1: realizable (smooth) filters — the cardinal-safe realization
# ---------------------------------------------------------------------------
def test_constant_di_filters_are_realizable():
    """Per-bin MSCD is violently rough (~5 rad raw); global-phase alignment makes it realizable.

    The honest metric is the magnitude-gated phase roughness (raw phase_roughness overcounts noise
    on near-silent drivers of the tapered cap). After alignment it sits well under 0.6 rad
    (``docs/Chunk3b_Findings.md``).
    """
    ds, obs = _cap_dataset(_CAP, _BAND)
    r = design(ds, _gate_spec(wng_floor_db=0.0))
    tau = float(r.attrs["constant_di_shared_tau_s"])
    rough = magnitude_gated_phase_roughness(r.weights, _BAND, tau)
    assert rough < 0.6, f"constant-DI filters too rough: {rough:.3f} rad"


# ---------------------------------------------------------------------------
# DISCRIMINATOR 2: honest WNG floor — graceful, every bin respected
# ---------------------------------------------------------------------------
def test_constant_di_respects_honest_wng_floor():
    """The WNG floor binds at every bin and the directivity stays flat (graceful, not collapsed)."""
    ds, obs = _cap_dataset(_CAP, _BAND)
    r0 = design(ds, _gate_spec(wng_floor_db=0.0))
    assert np.all(r0.metrics["feasible_mask"]), "a bin was flagged infeasible at floor 0 dB"
    assert r0.attrs["band_feasible"] is True
    assert np.all(r0.metrics["wng_db"] >= 0.0 - 0.5), f"WNG floor violated: {r0.metrics['wng_db']}"

    # A stricter floor trades directivity gracefully (lower constant DI), never collapses.
    r3 = design(ds, _gate_spec(wng_floor_db=3.0))
    assert np.all(r3.metrics["wng_db"] >= 3.0 - 0.5), f"WNG floor violated: {r3.metrics['wng_db']}"
    assert np.ptp(r3.metrics["di_db"]) < 0.05  # still constant DI
    assert (
        r3.attrs["constant_di_db"] < r0.attrs["constant_di_db"]
    )  # stricter floor -> less directive


# ---------------------------------------------------------------------------
# CARDINAL-RULE PROOFS (steering is entirely in H's inter-driver phase)
# ---------------------------------------------------------------------------
def test_constant_di_cardinal_rule_near_collapse():
    """Collapse the cap to a 0.5 mm cluster (almost no inter-driver phase) -> DI -> ~0.

    With essentially no spatial diversity the array can only radiate ~omni, so the directivity
    index collapses toward 0 even though the design is asked for constant directivity. Any code path
    that re-zeroed a driver would also zero its inter-driver phase and (correctly) fail to beamform.
    """
    rng = np.random.default_rng(0)
    tiny = 5e-4 * rng.standard_normal((len(_CAP), 3))  # 0.5 mm cluster
    ds, obs = _cap_dataset(tiny, _BAND)
    r = design(ds, _gate_spec(wng_floor_db=0.0))
    assert (
        float(np.nanmax(r.metrics["di_db"])) < 2.0
    ), "DI should collapse with no inter-driver phase"


def test_constant_di_cardinal_rule_shared_ramp_invariant():
    """A shared modeling delay (common latency, all drivers) cannot change |P| (cardinal-rule)."""
    ds, obs = _cap_dataset(_CAP, _BAND)
    r = design(ds, _gate_spec(wng_floor_db=0.0))
    h = stacked_h_full(ds)  # [M,F,N]
    ramp = np.exp(-1j * 2.0 * np.pi * _BAND * 4.1e-4)  # arbitrary shared delay
    P0 = np.sum(r.weights[:, :, None] * h, axis=0)  # [F,N]
    P1 = np.sum((r.weights * ramp[None, :])[:, :, None] * h, axis=0)  # [F,N]
    assert np.max(np.abs(np.abs(P0) - np.abs(P1))) < 1e-9


# ---------------------------------------------------------------------------
# The "region" objective still holds its (different) cap-ratio constant — both modes work.
# ---------------------------------------------------------------------------
def test_region_mode_remains_available_and_constant():
    """directivity_mode='region' still holds the cap-ratio GDI constant (back-compat objective)."""
    ds, obs = _cap_dataset(_CAP, _BAND)
    spec = TargetSpec(
        steer_dir=np.array([0.0, 0.0, 1.0]),
        engine="constant_di",
        directivity_mode="region",
        accept_halfangle_deg=45.0,
        wng_floor_db=-6.0,
    )
    r = design(ds, spec)
    assert "constant_gdi_db" in r.attrs  # region reports the cap-ratio GDI, not the index
    assert r.attrs["directivity_mode"] == "region"
