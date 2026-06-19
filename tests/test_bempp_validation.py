"""Item 11 finish-line test: bempp-cl V-2 sphere benchmark.

Runs the pulsating-sphere analytic cross-check (V-2) through the BemppBackend,
using exactly the same validation function (sphere_benchmark_errors) as the
NumCalc V-2 test — proving the BEMBackend interface is backend-agnostic and that
an independent solver reproduces the same analytic truth.

Pass criteria (same as NumCalc V-2):
  mean |magnitude error| ≤ 0.5 dB   at every frequency
  mean |phase error|     ≤ 5.0°      at every frequency

The test is marked ``bempp`` and skips cleanly when the optional bempp-cl
dependency group is not installed (``uv sync --group bempp`` to enable).
"""

import numpy as np
import pytest

pytestmark = pytest.mark.bempp  # skip when bempp-cl not installed


def _skip_if_no_bempp() -> None:
    """Skip gracefully when the bempp optional group is absent."""
    pytest.importorskip("bempp_cl", reason="bempp-cl not installed — run: uv sync --group bempp")


# ---------------------------------------------------------------------------
# Main V-2 test
# ---------------------------------------------------------------------------


def test_bempp_sphere_benchmark() -> None:
    """V-2 gate via bempp-cl: pulsating-sphere BEM pressure vs. exact closed form.

    Geometry: icosphere of radius a=0.10 m, subdivisions=2 (320 triangles).
    Frequencies: 250, 500, 1000 Hz (ka ≈ 0.46, 0.92, 1.83 — same as NumCalc V-2).
    Observation grid: Lebedev-26 at r=1.0 m (N=26 directions).
    Config: n_epw=6, tolerance=1e-6, max_iterations=1000, burton_miller=False.

    The physics is the same as test_sphere_benchmark.py::test_sphere_benchmark;
    only the backend (BemppBackend vs NumCalcBackend) differs. Using the same
    sphere_benchmark_errors() function on both backends proves that the BEMBackend
    abstraction (DR-02) is truly backend-agnostic.
    """
    _skip_if_no_bempp()

    from beamsim2.backends.bempp.adapter import BemppBackend
    from beamsim2.core.sphere import lebedev
    from beamsim2.core.types import FrequencyGrid, SolverConfig
    from beamsim2.validation.sphere_benchmark import (
        make_pulsating_sphere_mesh,
        sphere_benchmark_errors,
    )

    a = 0.10  # m — sphere radius
    c = 343.2  # m/s — speed of sound in air at ~20°C
    rho = 1.2041  # kg/m³ — air density at ~20°C

    # Geometry: same as NumCalc V-2 (subdivisions=2 → 320 triangles)
    mesh, bc = make_pulsating_sphere_mesh(radius=a, subdivisions=2)

    # Frequencies: ka ≈ 0.46, 0.92, 1.83 — below ka=π (first irregular freq of sphere)
    freqs = FrequencyGrid(
        frequencies=np.array([250.0, 500.0, 1000.0], dtype=np.float64),
        spacing="linear",
        fractional_octave=None,
    )

    # Lebedev-26 quadrature (exact for polynomials up to degree 9)
    obs = lebedev(26, radius=1.0)

    config = SolverConfig(
        n_epw=6,
        tolerance=1e-6,
        max_iterations=1000,
        burton_miller=False,  # below first irregular frequency (ka=π); plain BIE stable
        speed_of_sound=c,
        air_density=rho,
    )

    # --- BEMBackend sequence: prepare → solve → extract ---
    backend = BemppBackend()
    spec = backend.prepare(mesh, bc, freqs, obs, config)
    raw = backend.solve(spec)
    field = backend.extract(raw, obs)

    # Dense LU always converges
    assert field.convergence_flags.all(), "Expected all convergence flags True (dense LU)"

    # Shape sanity
    n_freq = len(freqs.frequencies)
    n_obs = len(obs.unit_vectors)
    assert field.pressure.shape == (
        n_freq,
        n_obs,
    ), f"pressure shape mismatch: got {field.pressure.shape}, expected ({n_freq}, {n_obs})"
    assert field.frequencies.shape == (n_freq,)
    np.testing.assert_allclose(field.frequencies, freqs.frequencies)

    # --- V-2 analytic comparison ---
    # sphere_benchmark_errors() computes mag and phase error vs the exact
    # pulsating-sphere closed form (engineering exp(−iωt) convention).
    result = sphere_benchmark_errors(
        H_bem=field.pressure,  # [F, N] complex128 — raw bempp pressure
        frequencies=field.frequencies,
        obs_points=obs,
        a=a,
        c=c,
        rho=rho,
    )

    # Report per-frequency errors before asserting (helps diagnose failures)
    for i, f in enumerate(field.frequencies):
        ka = 2 * np.pi * f / c * a
        print(
            f"  f={f:.0f} Hz  ka={ka:.3f}  "
            f"mean_mag={result['mean_mag_db'][i]:.3f} dB  "
            f"max_mag={result['max_mag_db'][i]:.3f} dB  "
            f"mean_phase={result['mean_phase_deg'][i]:.2f}°  "
            f"max_phase={result['max_phase_deg'][i]:.2f}°"
        )

    # --- gate: ≤0.5 dB magnitude AND ≤5° phase at every frequency ---
    assert result["passed"], (
        "V-2 bempp benchmark FAILED.\n"
        f"  mean_mag_db  = {result['mean_mag_db']}  (gate: all ≤ 0.5)\n"
        f"  mean_phase_deg = {result['mean_phase_deg']}  (gate: all ≤ 5.0)\n"
        "Check the Neumann-datum sign (g_N = iωρ v_n) and the representation "
        "formula sign (p_ext = V[g_N] - K[p_s]) in adapter.py."
    )


# ---------------------------------------------------------------------------
# Interface conformance (CI-safe, no bempp needed)
# ---------------------------------------------------------------------------


def test_bempp_backend_is_registered() -> None:
    """BemppBackend imports cleanly and is a BEMBackend subclass."""
    from beamsim2.backends.base import BEMBackend
    from beamsim2.backends.bempp.adapter import BemppBackend

    assert issubclass(BemppBackend, BEMBackend), "BemppBackend must subclass BEMBackend (DR-02)"
    b = BemppBackend()
    assert hasattr(b, "estimate")
    assert hasattr(b, "prepare")
    assert hasattr(b, "solve")
    assert hasattr(b, "extract")


def test_bempp_backend_estimate_no_bempp() -> None:
    """estimate() works without bempp installed (pure numpy, no lazy import)."""
    from beamsim2.backends.bempp.adapter import BemppBackend
    from beamsim2.core.types import (
        BoundaryConditions,
        FrequencyGrid,
        Mesh,
        SolverConfig,
    )

    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    tris = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int32)
    tags = np.ones(4, dtype=np.int32)
    mesh = Mesh(vertices=verts, triangles=tris, group_tags=tags)
    bc = BoundaryConditions(vibrating_groups={1: complex(1.0, 0.0)})
    freqs = FrequencyGrid(
        frequencies=np.array([250.0, 500.0], dtype=np.float64),
        spacing="linear",
        fractional_octave=None,
    )
    config = SolverConfig()

    plan = BemppBackend().estimate(mesh, bc, freqs, config)
    assert plan.ram_bytes_per_step.shape == (2,)
    assert plan.time_seconds_per_step.shape == (2,)
    assert np.all(plan.ram_bytes_per_step > 0)
    assert np.all(np.isnan(plan.time_seconds_per_step))
