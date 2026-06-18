"""V-5 (critical): two-driver per-driver superposition matches direct two-driver BEM
solve within solver tolerance — guardrail for the single-phase-origin rule.

Also contains pure-Python (CI-safe) unit tests for all three assembly modules:
  - superpose: linearity, shape-mismatch guard, dtype guard
  - phase_origin: residual ≈0 for identical fields; positive proof that a simulated
    per-driver phase-ramp bug is detected by the guardrail (no NumCalc needed)
  - tensor: build_dataset shapes, H_full contract, mismatch guards

The module-level pytestmark is NOT used here — instead each V-5 test is decorated
individually with @pytest.mark.local_only so the pure-Python tests still run in CI.
"""

from __future__ import annotations

import numpy as np
import pytest

from beamsim2.assembly.phase_origin import (
    assert_superposition_matches,
    superposition_residual,
)
from beamsim2.assembly.superpose import driver_h_bem, superpose_fields
from beamsim2.assembly.tensor import build_dataset, stacked_h_full
from beamsim2.core.sphere import lebedev
from beamsim2.core.types import ComplexField, FrequencyGrid

# ── binary resolution (V-5 only) ────────────────────────────────────────────

try:
    from beamsim2.backends.numcalc.config import resolve_numcalc_binary

    _BINARY = resolve_numcalc_binary()
except FileNotFoundError:
    _BINARY = None


def _skip_if_no_binary() -> None:
    if _BINARY is None:
        pytest.skip("NumCalc binary not found. Set BEAMSIM2_NUMCALC_BIN to run V-5.")


# ── helpers ──────────────────────────────────────────────────────────────────


def _rand_field(
    rng: np.random.Generator,
    n_freq: int,
    n_dir: int,
    freqs: np.ndarray,
) -> np.ndarray:
    """Random [F × N] complex128 pressure array."""
    return (
        rng.standard_normal((n_freq, n_dir)) + 1j * rng.standard_normal((n_freq, n_dir))
    ).astype(np.complex128)


def _make_complex_field(pressure: np.ndarray, freqs: np.ndarray) -> ComplexField:
    """Wrap a pressure array in a ComplexField."""
    F = len(freqs)
    return ComplexField(
        pressure=pressure,
        convergence_flags=np.ones(F, dtype=bool),
        frequencies=freqs,
    )


# ── superpose.py unit tests (pure-Python, CI-safe) ──────────────────────────


class TestSuperpose:
    def test_single_field_passthrough(self) -> None:
        """superpose_fields with one field returns it unchanged."""
        rng = np.random.default_rng(0)
        arr = _rand_field(rng, 3, 6, np.array([100.0, 200.0, 400.0]))
        result = superpose_fields([arr])
        assert np.array_equal(result, arr)

    def test_two_fields_sum(self) -> None:
        """superpose_fields sums two fields correctly."""
        rng = np.random.default_rng(1)
        a = _rand_field(rng, 4, 14, np.linspace(100, 1000, 4))
        b = _rand_field(rng, 4, 14, np.linspace(100, 1000, 4))
        result = superpose_fields([a, b])
        assert np.allclose(result, a + b, rtol=0, atol=1e-15)

    def test_linearity(self) -> None:
        """Superposition is linear: 3×field == superpose_fields([f, f, f])."""
        rng = np.random.default_rng(2)
        a = _rand_field(rng, 2, 6, np.array([250.0, 500.0]))
        result = superpose_fields([a, a, a])
        assert np.allclose(result, 3.0 * a, rtol=0, atol=1e-14)

    def test_shape_mismatch_raises(self) -> None:
        """Mismatched field shapes raise ValueError."""
        rng = np.random.default_rng(3)
        a = _rand_field(rng, 3, 6, np.array([100.0, 200.0, 400.0]))
        b = _rand_field(rng, 3, 14, np.array([100.0, 200.0, 400.0]))
        with pytest.raises(ValueError, match="shape mismatch"):
            superpose_fields([a, b])

    def test_empty_raises(self) -> None:
        """Empty list raises ValueError."""
        with pytest.raises(ValueError, match="at least one"):
            superpose_fields([])

    def test_output_dtype_is_complex128(self) -> None:
        """Output dtype is always complex128 regardless of input dtype."""
        arr = np.ones((2, 3), dtype=np.complex64)
        result = superpose_fields([arr])
        assert result.dtype == np.complex128

    def test_driver_h_bem_returns_pressure(self) -> None:
        """driver_h_bem returns field.pressure unchanged."""
        rng = np.random.default_rng(5)
        freqs = np.array([250.0, 500.0])
        arr = _rand_field(rng, 2, 6, freqs)
        cf = _make_complex_field(arr, freqs)
        result = driver_h_bem(cf)
        assert result is cf.pressure


# ── phase_origin.py unit tests (pure-Python, CI-safe) ───────────────────────


class TestPhaseOrigin:
    def test_identical_fields_residual_zero(self) -> None:
        """Identical summed and direct fields → relative_l2 ≈ 0."""
        rng = np.random.default_rng(10)
        arr = _rand_field(rng, 3, 14, np.array([100.0, 200.0, 400.0]))
        metrics = superposition_residual(arr, arr)
        assert metrics["relative_l2"] == pytest.approx(0.0, abs=1e-15)
        assert metrics["max_abs_db"] == pytest.approx(0.0, abs=1e-10)
        assert metrics["max_phase_deg"] == pytest.approx(0.0, abs=1e-8)

    def test_identical_assertion_passes(self) -> None:
        """assert_superposition_matches does not raise for identical arrays."""
        rng = np.random.default_rng(11)
        arr = _rand_field(rng, 2, 6, np.array([250.0, 500.0]))
        assert_superposition_matches(arr, arr)  # must not raise

    def test_phase_ramp_bug_detected(self) -> None:
        """Positive proof: a simulated per-driver phase-zeroing bug is detected.

        We simulate the worst-case re-zeroing: driver A is given a 90° phase
        ramp across frequencies (as if its time-of-flight phase was stripped),
        then the 'summed' field is constructed from the re-zeroed version.
        The residual relative_l2 must be large (>> rtol=1e-3).
        """
        rng = np.random.default_rng(12)
        n_freq, n_dir = 6, 14
        freqs = np.linspace(100.0, 1000.0, n_freq)
        H_A_correct = _rand_field(rng, n_freq, n_dir, freqs)
        H_B = _rand_field(rng, n_freq, n_dir, freqs)

        # "direct" solve: both drivers, natural phases
        H_direct = H_A_correct + H_B

        # Bug: H_A is re-zeroed (phase ramp removed) — strip per-frequency phase
        phase_ramp = np.exp(-1j * np.angle(H_A_correct[:, 0]))  # [F] per-freq
        H_A_rezeroed = H_A_correct * phase_ramp[:, None]  # phase-stripped
        H_summed_buggy = H_A_rezeroed + H_B

        metrics = superposition_residual(H_summed_buggy, H_direct)
        # re-zeroing bug produces order-1 mismatch
        assert (
            metrics["relative_l2"] > 0.1
        ), f"Phase-ramp bug was not detected: relative_l2 = {metrics['relative_l2']:.3e}"

    def test_phase_ramp_bug_triggers_assertion(self) -> None:
        """assert_superposition_matches raises AssertionError for a re-zeroing bug."""
        rng = np.random.default_rng(13)
        n_freq, n_dir = 4, 6
        freqs = np.linspace(100.0, 400.0, n_freq)
        H_A = _rand_field(rng, n_freq, n_dir, freqs)
        H_B = _rand_field(rng, n_freq, n_dir, freqs)
        H_direct = H_A + H_B
        # simulate re-zeroing: rotate H_A by 180° at all frequencies
        H_A_bad = -H_A
        H_summed_bad = H_A_bad + H_B
        with pytest.raises(AssertionError, match="Phase-origin"):
            assert_superposition_matches(H_summed_bad, H_direct, rtol=1e-3)

    def test_shape_mismatch_raises(self) -> None:
        """superposition_residual raises on shape mismatch."""
        rng = np.random.default_rng(14)
        a = _rand_field(rng, 3, 6, np.array([100.0, 200.0, 400.0]))
        b = _rand_field(rng, 3, 14, np.array([100.0, 200.0, 400.0]))
        with pytest.raises(ValueError, match="shape mismatch"):
            superposition_residual(a, b)


# ── tensor.py unit tests (pure-Python, CI-safe) ─────────────────────────────


class TestTensor:
    def _make_ds(
        self,
        n_drivers: int = 2,
        n_freq: int = 3,
        n_dir: int = 26,
        seed: int = 0,
    ):
        rng = np.random.default_rng(seed)
        freqs = np.linspace(100.0, 1000.0, n_freq)
        obs = lebedev(n_points=n_dir, radius=1.0)
        driver_inputs = [
            (
                f"drv_{i}",
                _make_complex_field(_rand_field(rng, n_freq, n_dir, freqs), freqs),
                {"name": f"driver_{i}"},
            )
            for i in range(n_drivers)
        ]
        return build_dataset(driver_inputs=driver_inputs, directions=obs), freqs, obs

    def test_h_full_equals_h_bem_times_terminal_default(self) -> None:
        """With default terminal_response (ones), H_full == H_bem."""
        ds, freqs, _ = self._make_ds()
        for d in ds.drivers:
            assert np.allclose(d.H_full, d.H_bem, rtol=0, atol=1e-15)

    def test_h_full_with_nontrivial_terminal(self) -> None:
        """H_full = H_bem * terminal_response[:, None] with a non-trivial TR."""
        rng = np.random.default_rng(20)
        n_freq, n_dir = 4, 14
        freqs = np.linspace(100.0, 800.0, n_freq)
        obs = lebedev(n_points=n_dir, radius=1.0)
        pressure = _rand_field(rng, n_freq, n_dir, freqs)
        field = _make_complex_field(pressure, freqs)
        tr = (rng.standard_normal(n_freq) + 1j * rng.standard_normal(n_freq)).astype(np.complex128)

        ds = build_dataset(
            driver_inputs=[("d0", field, {})],
            directions=obs,
            terminal_responses=[tr],
        )
        d = ds.drivers[0]
        expected = pressure.astype(np.complex128) * tr[:, None]
        assert np.allclose(d.H_full, expected, rtol=1e-14, atol=0)

    def test_shapes(self) -> None:
        """DriverData and RadiationDataset arrays have correct shapes."""
        n_freq, n_dir = 5, 14
        ds, freqs, obs = self._make_ds(n_drivers=3, n_freq=n_freq, n_dir=n_dir)
        assert ds.frequencies.shape == (n_freq,)
        assert ds.interpolated_mask.shape == (n_freq,)
        for d in ds.drivers:
            assert d.H_bem.shape == (n_freq, n_dir)
            assert d.H_full.shape == (n_freq, n_dir)
            assert d.terminal_response.shape == (n_freq,)
            assert d.convergence_flags.shape == (n_freq,)

    def test_stacked_h_full_shape(self) -> None:
        """stacked_h_full returns [M × F × N]."""
        M, F, N = 3, 5, 14
        ds, _, _ = self._make_ds(n_drivers=M, n_freq=F, n_dir=N)
        H = stacked_h_full(ds)
        assert H.shape == (M, F, N)
        assert H.dtype == np.complex128

    def test_freq_mismatch_raises(self) -> None:
        """build_dataset raises ValueError if driver frequency grids differ."""
        rng = np.random.default_rng(30)
        n_freq, n_dir = 3, 6
        obs = lebedev(n_points=n_dir, radius=1.0)
        freqs_a = np.array([100.0, 200.0, 400.0])
        freqs_b = np.array([100.0, 200.0, 800.0])  # different last freq

        field_a = _make_complex_field(_rand_field(rng, n_freq, n_dir, freqs_a), freqs_a)
        field_b = _make_complex_field(_rand_field(rng, n_freq, n_dir, freqs_b), freqs_b)
        with pytest.raises(ValueError, match="frequency grid"):
            build_dataset(
                driver_inputs=[("a", field_a, {}), ("b", field_b, {})],
                directions=obs,
            )

    def test_direction_mismatch_raises(self) -> None:
        """build_dataset raises ValueError if direction count doesn't match obs."""
        rng = np.random.default_rng(31)
        n_freq = 3
        freqs = np.array([100.0, 200.0, 400.0])
        obs = lebedev(n_points=6, radius=1.0)  # 6 directions
        # pressure has 14 directions — mismatch
        pressure = _rand_field(rng, n_freq, 14, freqs)
        field = _make_complex_field(pressure, freqs)
        with pytest.raises(ValueError, match="directions"):
            build_dataset(driver_inputs=[("d", field, {})], directions=obs)

    def test_empty_driver_list_raises(self) -> None:
        """build_dataset raises ValueError for empty driver list."""
        obs = lebedev(n_points=6, radius=1.0)
        with pytest.raises(ValueError, match="at least one"):
            build_dataset(driver_inputs=[], directions=obs)

    def test_terminal_response_wrong_length_raises(self) -> None:
        """build_dataset raises ValueError if terminal_responses[i] length != F."""
        rng = np.random.default_rng(40)
        n_freq, n_dir = 4, 6
        freqs = np.linspace(100.0, 800.0, n_freq)
        obs = lebedev(n_points=n_dir, radius=1.0)
        field = _make_complex_field(_rand_field(rng, n_freq, n_dir, freqs), freqs)
        wrong_tr = np.ones(n_freq + 1, dtype=np.complex128)  # length F+1 — wrong
        with pytest.raises(ValueError, match="terminal_responses"):
            build_dataset(
                driver_inputs=[("d0", field, {})],
                directions=obs,
                terminal_responses=[wrong_tr],
            )


# ── V-5: two-driver superposition vs direct BEM solve (@local_only) ─────────


@pytest.mark.local_only
def test_v5_two_driver_superposition() -> None:
    """V-5: sum of per-driver fields matches direct two-driver BEM solve.

    Geometry: a small box (0.12 m × 0.10 m × 0.08 m) with two flush-mounted
    drivers on the front face (+z face), each radius 0.020 m, centres at
    ±0.025 m from the box centre.  Coarse h_elem = 0.020 m for speed.

    Three NumCalc solves on the identical mesh:
      1. Driver A only  (group 1 vibrates, group 2 + shell sound-hard)
      2. Driver B only  (group 2 vibrates, group 1 + shell sound-hard)
      3. Both drivers   (groups 1 + 2 vibrate — exercises _group_element_runs)

    H_A + H_B must equal H_both within relative_l2 ≤ 1e-3.
    All three solves share the same system matrix (same mesh + frequencies),
    so agreement is expected at ~1e-5; anything looser than ~1e-4 signals a
    real problem (most likely the multi-group BC writer — stop and ask).

    Frequencies: 3 log-spaced steps (250, 500, 1000 Hz).
    Observation: Lebedev-26 sphere at r = 1.0 m.
    """
    _skip_if_no_binary()

    from beamsim2.backends.numcalc.adapter import NumCalcBackend
    from beamsim2.core.types import BoundaryConditions, SolverConfig
    from beamsim2.geometry.assemble import DriverSpec, assemble_box_driver

    # ── geometry ──────────────────────────────────────────────────────────
    W, H_box, D = 0.12, 0.10, 0.08  # box dimensions (m)
    driver_radius = 0.020  # m
    z_face = D  # +z face at z = D (outward normal +z)

    # Two drivers on the front (+z) face, side by side
    driver_a = DriverSpec(
        center=(W / 2 - 0.025, H_box / 2, z_face),
        normal=(0.0, 0.0, 1.0),
        radius=driver_radius,
    )
    driver_b = DriverSpec(
        center=(W / 2 + 0.025, H_box / 2, z_face),
        normal=(0.0, 0.0, 1.0),
        radius=driver_radius,
    )

    h_elem = 0.020  # coarse mesh for speed (m)
    mesh, bc_both = assemble_box_driver(
        width=W,
        height=H_box,
        depth=D,
        drivers=[driver_a, driver_b],
        h_elem=h_elem,
    )

    # group 1 = driver A, group 2 = driver B, group 3 = shell (sound-hard)
    bc_a = BoundaryConditions(vibrating_groups={1: complex(1.0, 0.0)})
    bc_b = BoundaryConditions(vibrating_groups={2: complex(1.0, 0.0)})
    # bc_both already returned by assemble_box_driver: {1: 1+0j, 2: 1+0j}

    # ── solve config ──────────────────────────────────────────────────────
    freqs = FrequencyGrid(
        frequencies=np.array([250.0, 500.0, 1000.0]),
        spacing="log",
    )
    obs = lebedev(n_points=26, radius=1.0)
    config = SolverConfig()

    backend = NumCalcBackend()

    def run_solve(bc: BoundaryConditions) -> np.ndarray:
        """Run one prepare/solve/extract cycle, return pressure [F × N] complex128."""
        spec = backend.prepare(mesh, bc, freqs, obs, config)
        raw = backend.solve(spec)
        field = backend.extract(raw, obs)
        return field.pressure  # [F × N] complex128

    # ── three solves ──────────────────────────────────────────────────────
    H_A = run_solve(bc_a)  # [F × N] complex128 — driver A alone
    H_B = run_solve(bc_b)  # [F × N] complex128 — driver B alone
    H_both = run_solve(bc_both)  # [F × N] complex128 — both together

    # ── V-5 check ─────────────────────────────────────────────────────────
    H_summed = superpose_fields([H_A, H_B])  # [F × N] complex128
    metrics = superposition_residual(H_summed, H_both)

    # Diagnostic output (always print so it appears in pytest -s output)
    print(
        f"\n[V-5] relative_l2 = {metrics['relative_l2']:.3e}  "
        f"max|dB| = {metrics['max_abs_db']:.2f} dB  "
        f"max phase Δ = {metrics['max_phase_deg']:.2f}°"
    )

    # Stop and report if looser than expected (possible BC-writer bug)
    if metrics["relative_l2"] > 1e-4:
        pytest.fail(
            f"[V-5] relative_l2 = {metrics['relative_l2']:.3e} > 1e-4 — "
            f"this is looser than expected for identical BEM system matrix.\n"
            f"Most likely cause: multi-group BC writer (_group_element_runs in "
            f"ncinp_writer.py).  STOP and investigate before proceeding."
        )

    # Official pass criterion (hard gate rtol = 1e-3)
    assert_superposition_matches(H_summed, H_both, rtol=1e-3)
