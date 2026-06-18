"""V-1: BEM directivity for a piston in a large finite baffle vs. closed-form D(θ)=2·J₁(ka sinθ)/(ka sinθ); mean error ≤1 dB acceptance gate."""

from __future__ import annotations

import numpy as np
import pytest

from beamsim2.core.sphere import lebedev
from beamsim2.core.types import FrequencyGrid, SolverConfig
from beamsim2.validation.analytic_piston import make_piston_mesh, piston_benchmark_errors

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
# V-1: flat piston in large baffle — mean directivity error ≤ 1 dB
# ---------------------------------------------------------------------------


def test_piston_benchmark() -> None:
    """V-1 acceptance gate: piston directivity BEM vs. analytic formula.

    Runs NumCalc on a flat piston mesh (a = 0.05 m, baffle = 0.40 m half-width)
    at three frequencies chosen to give ka ≈ 1, 2, 3:
        f ≈ 1091 Hz  (ka = 2π·1091·0.05/343.2 ≈ 1.0)
        f ≈ 2182 Hz  (ka ≈ 2.0)
        f ≈ 3274 Hz  (ka ≈ 3.0)

    The Lebedev-26 observation sphere is at r = 1.0 m.
    BEM is normalised by the on-axis value and compared to D(θ, ka).

    Pass criterion V-1: mean |error| ≤ 1 dB at every frequency.
    """
    _skip_if_no_binary()

    from beamsim2.backends.numcalc.adapter import NumCalcBackend

    a = 0.05  # m — piston radius
    c = 343.2  # m/s
    rho = 1.2041  # kg/m³

    # ka = 2π·f·a/c → f = ka·c/(2π·a)
    mesh, bc = make_piston_mesh(a_piston=a, baffle_half_width=0.40, h_elem=0.013)
    freqs = FrequencyGrid(
        frequencies=np.array([1091.0, 2182.0, 3274.0], dtype=np.float64),
        spacing="linear",
        fractional_octave=None,
    )
    obs = lebedev(26, radius=1.0)
    config = SolverConfig(
        n_epw=8,
        tolerance=1e-6,
        max_iterations=2000,
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
    ), f"V-1: NumCalc did not converge at all steps. Flags: {field.convergence_flags}"

    result = piston_benchmark_errors(
        field.pressure,
        field.frequencies,
        obs,
        a_piston=a,
        c=c,
    )

    print(
        "\nV-1 piston benchmark results:"
        f"\n  frequencies    : {field.frequencies} Hz"
        f"\n  mean |error|   : {result['mean_mag_db']} dB"
        f"\n  max  |error|   : {result['max_mag_db']} dB"
        f"\n  passed         : {result['passed']}"
    )

    assert result["passed"], (
        f"V-1 failed: mean |directivity error| = {result['mean_mag_db']} dB "
        f"(threshold 1.0 dB per frequency)"
    )
