"""Round-trip smoke test for the NumCalc BEMBackend adapter (build-order item 3).

Feeds normalized core-type inputs through the full interface (prepare → solve → extract)
using the real NumCalc binary on a pulsating-sphere mesh at two frequencies, and asserts
the returned ComplexField has the correct shape, dtype, finite non-zero pressures, and
convergence flags.

Test geometry (chosen so item 4 can validate against the exact analytic solution):
  - Pulsating sphere, radius a = 0.10 m, centered at the global coordinate origin.
  - Boundary: icosphere subdivision 1 (~80 triangles), all elements vibrating at unit
    normal velocity (uniform complex scalar 1+0j), a single group tag (1).
  - Frequencies: [250, 500] Hz (ka ≈ 0.46, 0.92).
  - Observation: Lebedev N=14 at r = 1.0 m.
  - Solver: conventional collocation BEM (method 0), no Burton–Miller (simple geometry).

The mesh builder lives in beamsim2.validation.sphere_benchmark so that item 4 (V-2)
can reuse it without duplicating geometry code.

Marker: local_only — this test requires the NumCalc binary. It is skipped automatically
when BEAMSIM2_NUMCALC_BIN is unset, keeping 'uv run pytest' (without the binary) green.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from beamsim2.core.sphere import lebedev
from beamsim2.core.types import FrequencyGrid, SolverConfig
from beamsim2.validation.sphere_benchmark import make_pulsating_sphere_mesh

# ── Resolve the binary once at collection time and skip if absent ─────────────

try:
    from beamsim2.backends.numcalc.config import resolve_numcalc_binary

    _BINARY = resolve_numcalc_binary()
except FileNotFoundError:
    _BINARY = None

pytestmark = pytest.mark.local_only


def _skip_if_no_binary() -> None:
    if _BINARY is None:
        pytest.skip("NumCalc binary not found. Set BEAMSIM2_NUMCALC_BIN to run this test.")


# ---------------------------------------------------------------------------
# Round-trip smoke test
# ---------------------------------------------------------------------------


def test_numcalc_roundtrip() -> None:
    """Prove that normalized inputs round-trip through the NumCalc adapter.

    Checks shape, dtype, finiteness, non-zero magnitude, convergence flags,
    and frequency echo. Does NOT check phase or SPL accuracy — that is item 4.
    """
    _skip_if_no_binary()

    from beamsim2.backends.numcalc.adapter import NumCalcBackend

    # ── Inputs ───────────────────────────────────────────────────────────────
    # ~80 triangles, all in group 1, unit normal velocity
    sphere_mesh, bc = make_pulsating_sphere_mesh(radius=0.10, subdivisions=1)
    freqs = FrequencyGrid(
        frequencies=np.array([250.0, 500.0], dtype=np.float64),
        spacing="linear",
        fractional_octave=None,
    )
    obs = lebedev(n_points=14, radius=1.0)  # 14-point Lebedev at 1 m
    config = SolverConfig(
        n_epw=6,
        tolerance=1e-6,
        max_iterations=1000,
        burton_miller=False,  # simple geometry; BM not needed for smoke test
        speed_of_sound=343.2,
        air_density=1.2041,
    )

    backend = NumCalcBackend()

    # ── prepare ───────────────────────────────────────────────────────────────
    spec = backend.prepare(sphere_mesh, bc, freqs, obs, config)

    assert os.path.isfile(spec.nc_inp_paths[0]), "NC.inp was not written"
    assert os.path.isdir(os.path.join(spec.work_dir, "ObjectMesh")), "ObjectMesh/ missing"
    assert os.path.isdir(os.path.join(spec.work_dir, "EvalGrid")), "EvalGrid/ missing"
    assert spec.frequency_grid is freqs

    # ── solve ─────────────────────────────────────────────────────────────────
    raw = backend.solve(spec)

    assert raw.work_dir == spec.work_dir
    assert len(raw.completed_steps) == 2, f"Expected 2 completed steps, got {raw.completed_steps}"
    assert raw.convergence_flags.shape == (2,), "convergence_flags shape wrong"

    # ── extract ───────────────────────────────────────────────────────────────
    field = backend.extract(raw, obs)

    # ── Shape and dtype ───────────────────────────────────────────────────────
    assert field.pressure.shape == (
        2,
        14,
    ), f"Expected pressure shape (2, 14), got {field.pressure.shape}"
    assert field.pressure.dtype == np.complex128, f"Expected complex128, got {field.pressure.dtype}"
    assert field.convergence_flags.shape == (2,)
    assert field.frequencies.shape == (2,)

    # ── Physical sanity ───────────────────────────────────────────────────────
    assert np.all(np.isfinite(field.pressure)), "pressure contains non-finite values"
    assert np.all(np.abs(field.pressure) > 0), "pressure magnitude is zero"

    # ── Convergence ───────────────────────────────────────────────────────────
    assert field.convergence_flags.all(), (
        "CGS solver did not converge at all steps. " f"Flags: {field.convergence_flags}"
    )

    # ── Frequency echo ────────────────────────────────────────────────────────
    np.testing.assert_array_equal(field.frequencies, [250.0, 500.0])

    print(
        f"\nRound-trip passed:\n"
        f"  pressure shape : {field.pressure.shape}  dtype={field.pressure.dtype}\n"
        f"  |p| range      : {np.abs(field.pressure).min():.4e} … "
        f"{np.abs(field.pressure).max():.4e}  Pa\n"
        f"  converged      : {field.convergence_flags.tolist()}\n"
        f"  work_dir       : {spec.work_dir}"
    )
