"""@local_only end-to-end test: full pipeline through pipeline/run.py.

Requires the NumCalc binary (BEAMSIM2_NUMCALC_BIN env var set).
Uses the same geometry as V-5 (test_phase_origin.py): a small box with two
flush-disk drivers — the proven template.

Asserts:
  - run_simulation produces a RadiationDataset with M=2 drivers, F=3, N=26
  - HDF5 file is written and round-trips losslessly
  - V-5 guardrail: H_summed ≈ H_both (per-driver superposition == direct BEM)
  - work_dirs populated (resumable if re-run on same tmp_path)

Build-order item 10 de-facto acceptance gate (no §7 entry exists for the GUI/
orchestrator; this is the 'one small real solve' the user requested).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from beamsim2.assembly.phase_origin import assert_superposition_matches, superposition_residual
from beamsim2.assembly.superpose import superpose_fields
from beamsim2.assembly.tensor import stacked_h_full
from beamsim2.core.sphere import lebedev
from beamsim2.core.types import BoundaryConditions, FrequencyGrid, SolverConfig
from beamsim2.geometry.assemble import DriverSpec, assemble_box_driver
from beamsim2.io.hdf5_store import read_dataset
from beamsim2.pipeline.run import (
    BoxGeometry,
    DriverPlacement,
    SimulationRequest,
    run_simulation,
)


def _skip_if_no_binary() -> None:
    binary = os.environ.get("BEAMSIM2_NUMCALC_BIN", "")
    if not binary or not Path(binary).is_file():
        pytest.skip("BEAMSIM2_NUMCALC_BIN not set or binary not found")


# ---------------------------------------------------------------------------
# E2E: box + 2-driver case (same geometry as V-5)
# ---------------------------------------------------------------------------

# V-5 geometry constants (must match test_phase_origin.py exactly for the
# cross-check to be meaningful)
_W, _H_BOX, _D = 0.12, 0.10, 0.08
_DRIVER_RADIUS = 0.020
_Z_FACE = _D  # +z face at z = D
_FREQS = np.array([250.0, 500.0, 1000.0])
_H_ELEM = 0.020  # coarse mesh for speed (m), same as V-5


@pytest.mark.local_only
def test_end_to_end_box_two_drivers(tmp_path: Path) -> None:
    """Full pipeline through run_simulation; V-5 superposition check on output.

    Three solves happen inside run_simulation (one per driver).  We then do a
    fourth direct two-driver solve (bc_both) outside the orchestrator — same
    mesh, same freqs — and verify that the superposed H matches the direct
    result within rtol=1e-3, exactly as V-5 asserts.
    """
    _skip_if_no_binary()

    from beamsim2.backends.numcalc.adapter import NumCalcBackend

    # ── Build request ────────────────────────────────────────────────────────
    driver_a_spec = DriverSpec(
        center=(_W / 2 - 0.025, _H_BOX / 2, _Z_FACE),
        normal=(0.0, 0.0, 1.0),
        radius=_DRIVER_RADIUS,
    )
    driver_b_spec = DriverSpec(
        center=(_W / 2 + 0.025, _H_BOX / 2, _Z_FACE),
        normal=(0.0, 0.0, 1.0),
        radius=_DRIVER_RADIUS,
    )

    h5_out = tmp_path / "e2e_out.h5"
    req = SimulationRequest(
        geometry=BoxGeometry(_W, _H_BOX, _D),
        drivers=[
            DriverPlacement(driver_a_spec, terminal=None, driver_id="drv_a"),
            DriverPlacement(driver_b_spec, terminal=None, driver_id="drv_b"),
        ],
        frequencies=FrequencyGrid(_FREQS, spacing="log"),
        sphere_n_points=26,
        sphere_radius=1.0,
        config=SolverConfig(),
        output_h5=h5_out,
    )

    # ── Run orchestrator ─────────────────────────────────────────────────────
    result = run_simulation(req)

    # ── Shape checks ─────────────────────────────────────────────────────────
    ds = result.dataset
    assert len(ds.drivers) == 2
    assert len(ds.frequencies) == 3
    H_stack = stacked_h_full(ds)  # [M, F, N] = [2, 3, 26]
    assert H_stack.shape == (2, 3, 26), f"Unexpected shape: {H_stack.shape}"
    assert H_stack.dtype == np.complex128

    # ── HDF5 round-trip ─────────────────────────────────────────────────────
    assert h5_out.exists(), "HDF5 output file was not written"
    ds_reload = read_dataset(h5_out)
    np.testing.assert_array_equal(
        ds_reload.drivers[0].H_bem,
        ds.drivers[0].H_bem,
        err_msg="HDF5 round-trip: drv_a H_bem mismatch",
    )
    np.testing.assert_array_equal(
        ds_reload.drivers[1].H_bem,
        ds.drivers[1].H_bem,
        err_msg="HDF5 round-trip: drv_b H_bem mismatch",
    )

    # ── work_dirs populated (resumable) ─────────────────────────────────────
    assert set(result.work_dirs.keys()) == {"drv_a", "drv_b"}

    # ── V-5 superposition check ──────────────────────────────────────────────
    # The orchestrator ran driver A alone and driver B alone.  Now run a
    # FOURTH solve with both drivers vibrating simultaneously.  Compare
    # H_summed = H_A + H_B (from orchestrator) vs H_both (direct solve here).
    #
    # All four solves share the SAME mesh topology and frequencies, so the BEM
    # system matrix is identical → agreement expected at ~1e-5; the rtol=1e-3
    # gate gives ample margin for numerical noise.
    backend = NumCalcBackend()
    obs = lebedev(n_points=26, radius=1.0)
    config = SolverConfig()
    freqs = FrequencyGrid(_FREQS, spacing="log")

    # Build the shared mesh with both groups vibrating
    mesh, bc_both = assemble_box_driver(
        width=_W,
        height=_H_BOX,
        depth=_D,
        drivers=[driver_a_spec, driver_b_spec],
        h_elem=_H_ELEM,
    )

    spec_both = backend.prepare(mesh, bc_both, freqs, obs, config)
    raw_both = backend.solve(spec_both)
    field_both = backend.extract(raw_both, obs)
    H_both = field_both.pressure  # [F, N] complex128

    H_A = ds.drivers[0].H_bem  # [F, N] complex128
    H_B = ds.drivers[1].H_bem  # [F, N] complex128
    H_summed = superpose_fields([H_A, H_B])  # [F, N] complex128

    metrics = superposition_residual(H_summed, H_both)
    print(
        f"\n[E2E V-5] relative_l2={metrics['relative_l2']:.3e}  "
        f"max|dB|={metrics['max_abs_db']:.2f} dB  "
        f"max phase Δ={metrics['max_phase_deg']:.2f}°"
    )

    if metrics["relative_l2"] > 1e-4:
        pytest.fail(
            f"V-5 relative_l2={metrics['relative_l2']:.3e} > 1e-4 — "
            "suggests phase-origin discipline violated or BC writer bug; "
            "stop and ask for guidance."
        )

    assert_superposition_matches(H_summed, H_both, rtol=1e-3)
    print("[E2E] V-5 superposition check: PASS")
