"""V-1: BEM directivity of a vibrating spherical cap on a rigid sphere vs. the
exact spherical-cap closed form; mean error ≤ 1 dB acceptance gate.

The original V-1 used a flat piston in a finite baffle compared to
``2·J₁(ka·sinθ)/(ka·sinθ)``. That coplanar geometry crashes NumCalc (ε = 0 in
``NC_GenerateSubelements`` → MSBE overrun), so V-1 is validated instead on a
curved geometry — a polar cap on a rigid sphere — against its own exact
solution (``spherical_cap_directivity``). This exercises mixed velocity/rigid
boundary conditions on a closed surface, which the all-vibrating V-2 sphere
does not.
"""

from __future__ import annotations

import numpy as np
import pytest

from beamsim2.core.sphere import lebedev
from beamsim2.core.types import FrequencyGrid, SolverConfig
from beamsim2.validation.analytic_piston import (
    cap_benchmark_errors,
    make_spherical_cap_piston_mesh,
)

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
# V-1: vibrating spherical cap on a rigid sphere — mean directivity error ≤ 1 dB
# ---------------------------------------------------------------------------


def test_cap_benchmark() -> None:
    """V-1 acceptance gate: spherical-cap directivity BEM vs. exact closed form.

    Runs NumCalc on a 1280-triangle icosphere (a = 0.10 m, subdivision 2→3) with
    a 45° polar cap vibrating at unit radial velocity and the rest rigid, at
    three frequencies chosen to give ka_sphere = 1, 2, 3:
        f ≈  546 Hz  (ka_sphere = 1, nearly omnidirectional)
        f ≈ 1093 Hz  (ka_sphere = 2)
        f ≈ 1639 Hz  (ka_sphere = 3, clearly directional, DI ≈ 7 dB)

    The Lebedev-26 observation sphere is at r = 1.0 m. The BEM field is
    normalised by its on-axis (+z) value and compared, over the forward
    hemisphere, to the exact ``spherical_cap_directivity`` for the same
    geometry — so the only expected disagreement is BEM discretization plus a
    little icosahedral azimuthal asymmetry (the cap is not a perfect figure of
    revolution; the analytic is azimuthally symmetric). Observed mean error is
    ≈ 0.6–0.8 dB, comfortably inside the gate.

    Pass criterion V-1: mean |directivity error| ≤ 1 dB at every frequency.
    """
    _skip_if_no_binary()

    from beamsim2.backends.numcalc.adapter import NumCalcBackend

    R = 0.10  # m — sphere radius
    cap_deg = 45.0  # cap half-angle (degrees)
    c = 343.2  # m/s
    rho = 1.2041  # kg/m³

    # ka_sphere = 2π·f·R/c → f = ka·c/(2π·R)
    mesh, bc = make_spherical_cap_piston_mesh(
        sphere_radius=R, cap_half_angle_deg=cap_deg, subdivisions=3
    )
    freqs = FrequencyGrid(
        frequencies=np.array([546.3, 1092.6, 1638.8], dtype=np.float64),
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
    ), f"V-1: NumCalc did not converge at all steps. Flags: {field.convergence_flags}"

    result = cap_benchmark_errors(
        field.pressure,
        field.frequencies,
        obs,
        sphere_radius=R,
        cap_half_angle_deg=cap_deg,
        c=c,
    )

    print(
        "\nV-1 spherical-cap benchmark results:"
        f"\n  frequencies    : {field.frequencies} Hz"
        f"\n  mean |error|   : {result['mean_mag_db']} dB"
        f"\n  max  |error|   : {result['max_mag_db']} dB"
        f"\n  passed         : {result['passed']}"
    )

    assert result["passed"], (
        f"V-1 failed: mean |directivity error| = {result['mean_mag_db']} dB "
        f"(threshold 1.0 dB per frequency)"
    )
