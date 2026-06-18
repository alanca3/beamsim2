"""V-2: BEM pressure for pulsating and oscillating sphere vs. exact closed-form solutions; magnitude error ≤0.5 dB in converged regime."""

from __future__ import annotations

import numpy as np
import pytest

from beamsim2.core.sphere import lebedev
from beamsim2.core.types import FrequencyGrid, SolverConfig
from beamsim2.validation.sphere_benchmark import make_pulsating_sphere_mesh, sphere_benchmark_errors

# ── Binary resolution (skip gracefully when NumCalc is absent) ────────────────

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
# V-2: pulsating sphere — mean magnitude error ≤ 0.5 dB
# ---------------------------------------------------------------------------


def test_sphere_benchmark() -> None:
    """V-2 acceptance gate: pulsating-sphere BEM magnitude and phase vs. analytic.

    Runs NumCalc on a 320-element icosphere (a = 0.10 m, subdivision 2)
    at three frequencies (250, 500, 1000 Hz; ka ≈ 0.46, 0.92, 1.83).
    The Lebedev-26 observation sphere is at r = 1.0 m.

    Subdivision 1 (80 triangles) achieves only 92.8 % of the sphere's surface
    area, causing a geometric amplitude error of ≈ 0.57 dB at 250 Hz that
    exceeds the 0.5 dB gate.  Subdivision 2 (320 triangles, ≈ 98 % area ratio)
    reduces the error to < 0.15 dB at all test frequencies.

    Pass criterion: mean |mag error| ≤ 0.5 dB and mean |phase error| ≤ 5°
    at every frequency.
    """
    _skip_if_no_binary()

    from beamsim2.backends.numcalc.adapter import NumCalcBackend

    a = 0.10  # m — sphere radius
    c = 343.2  # m/s
    rho = 1.2041  # kg/m³

    mesh, bc = make_pulsating_sphere_mesh(radius=a, subdivisions=2)
    freqs = FrequencyGrid(
        frequencies=np.array([250.0, 500.0, 1000.0], dtype=np.float64),
        spacing="linear",
        fractional_octave=None,
    )
    obs = lebedev(26, radius=1.0)
    config = SolverConfig(
        n_epw=6,
        tolerance=1e-6,
        max_iterations=1000,
        burton_miller=False,
        speed_of_sound=c,
        air_density=rho,
    )

    backend = NumCalcBackend()
    spec = backend.prepare(mesh, bc, freqs, obs, config)
    raw = backend.solve(spec)
    field = backend.extract(raw, obs)

    assert (
        field.convergence_flags.all()
    ), f"V-2: NumCalc did not converge at all steps. Flags: {field.convergence_flags}"

    result = sphere_benchmark_errors(
        field.pressure,
        field.frequencies,
        obs,
        a=a,
        c=c,
        rho=rho,
    )

    print(
        "\nV-2 sphere benchmark results:"
        f"\n  frequencies      : {field.frequencies} Hz"
        f"\n  mean |mag error| : {result['mean_mag_db']} dB"
        f"\n  max  |mag error| : {result['max_mag_db']} dB"
        f"\n  mean |phase err| : {result['mean_phase_deg']} deg"
        f"\n  max  |phase err| : {result['max_phase_deg']} deg"
        f"\n  passed           : {result['passed']}"
    )

    assert result["passed"], (
        f"V-2 failed: mag error = {result['mean_mag_db']} dB "
        f"(threshold 0.5 dB), phase error = {result['mean_phase_deg']} deg "
        f"(threshold 5 deg)"
    )
