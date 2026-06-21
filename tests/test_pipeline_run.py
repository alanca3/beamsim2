"""CI tests for pipeline/run.py — headless orchestrator.

Uses a fake BEMBackend that returns synthetic ComplexField data so no NumCalc
binary is required.  Covers:
  - Per-driver BoundaryConditions (only the right group vibrates each iteration)
  - H_full contract (H_bem × terminal_response[:, None])
  - Non-convergence flag propagation (flagged[driver_id][step] == True)
  - HDF5 export round-trip via run_simulation(output_h5=...)
  - Orchestrator with terminal responses applied

The @local_only end-to-end test (real NumCalc solve) is in test_pipeline_e2e.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from beamsim2.assembly.tensor import stacked_h_full
from beamsim2.backends.base import BEMBackend
from beamsim2.core.types import (
    BoundaryConditions,
    ComplexField,
    FrequencyGrid,
    Mesh,
    ObservationPoints,
    RawSolveResult,
    ResourcePlan,
    SolverConfig,
    SolveSpec,
)
from beamsim2.geometry.assemble import DriverSpec
from beamsim2.pipeline.run import (
    BoxGeometry,
    DriverPlacement,
    ResourceEstimate,
    SimulationRequest,
    estimate_resources,
    run_simulation,
)

# ---------------------------------------------------------------------------
# Test fixtures / helpers
# ---------------------------------------------------------------------------

F = 3  # frequency steps
N = 14  # Lebedev-14 directions


def _freq_grid() -> FrequencyGrid:
    return FrequencyGrid(np.array([250.0, 500.0, 1000.0]), spacing="log")


def _driver_spec(offset_x: float = 0.0) -> DriverSpec:
    return DriverSpec(
        center=(0.06 + offset_x, 0.05, 0.08),
        normal=(0.0, 0.0, 1.0),
        radius=0.020,
    )


def _fake_pressure(rng: np.random.Generator, converged: bool = True) -> ComplexField:
    """Random synthetic ComplexField [F, N] complex128."""
    pressure = (rng.standard_normal((F, N)) + 1j * rng.standard_normal((F, N))).astype(
        np.complex128
    )
    conv = np.ones(F, dtype=bool)
    if not converged:
        conv[1] = False  # mark step 1 non-converged
    return ComplexField(
        pressure=pressure,
        convergence_flags=conv,
        frequencies=np.array([250.0, 500.0, 1000.0]),
    )


class FakeBackend(BEMBackend):
    """Injects synthetic pressure data; records calls for assertion."""

    def __init__(
        self,
        rng: np.random.Generator,
        n_drivers: int = 2,
        make_flagged: bool = False,
    ) -> None:
        self._rng = rng
        self._fields: list[ComplexField] = [
            _fake_pressure(rng, converged=(not make_flagged or i > 0)) for i in range(n_drivers)
        ]
        # Record BoundaryConditions passed to prepare() so tests can assert
        self.prepare_calls: list[BoundaryConditions] = []
        self._call_count = 0

    def estimate(
        self,
        mesh: Mesh,
        bc: BoundaryConditions,
        frequencies: FrequencyGrid,
        config: SolverConfig,
    ) -> ResourcePlan:
        F_local = len(frequencies.frequencies)
        return ResourcePlan(
            ram_bytes_per_step=np.full(F_local, 1e9),
            time_seconds_per_step=np.full(F_local, np.nan),
        )

    def prepare(
        self,
        mesh: Mesh,
        bc: BoundaryConditions,
        frequencies: FrequencyGrid,
        observation_points: ObservationPoints,
        config: SolverConfig,
    ) -> SolveSpec:
        self.prepare_calls.append(bc)
        return SolveSpec(
            work_dir=f"/tmp/fake_work_{self._call_count}",
            nc_inp_paths=[],
            frequency_grid=frequencies,
        )

    def solve(
        self,
        spec: SolveSpec,
        scheduler: Optional[object] = None,
    ) -> RawSolveResult:
        idx = self._call_count
        self._call_count += 1
        field = self._fields[idx % len(self._fields)]
        return RawSolveResult(
            work_dir=spec.work_dir,
            completed_steps=set(range(F)),
            convergence_flags=field.convergence_flags.copy(),
        )

    def extract(
        self,
        raw: RawSolveResult,
        observation_points: ObservationPoints,
    ) -> ComplexField:
        idx = (self._call_count - 1) % len(self._fields)
        return self._fields[idx]


def _two_driver_request(
    output_h5: Optional[Path] = None,
) -> SimulationRequest:
    return SimulationRequest(
        geometry=BoxGeometry(0.12, 0.10, 0.08),
        drivers=[
            DriverPlacement(_driver_spec(-0.025), terminal=None, driver_id="drv_a"),
            DriverPlacement(_driver_spec(+0.025), terminal=None, driver_id="drv_b"),
        ],
        frequencies=_freq_grid(),
        sphere_n_points=14,
        sphere_radius=1.0,
        output_h5=output_h5,
    )


# ---------------------------------------------------------------------------
# Per-driver BoundaryConditions tests
# ---------------------------------------------------------------------------


def test_per_driver_bc_only_one_group_vibrating() -> None:
    """Each driver's solve must use only its own group; others remain sound-hard."""
    rng = np.random.default_rng(0)
    backend = FakeBackend(rng, n_drivers=2)
    run_simulation(_two_driver_request(), backend=backend)

    assert len(backend.prepare_calls) == 2
    bc_a, bc_b = backend.prepare_calls
    # Driver A (m=0) → group 1 vibrates
    assert 1 in bc_a.vibrating_groups
    assert 2 not in bc_a.vibrating_groups
    # Driver B (m=1) → group 2 vibrates
    assert 2 in bc_b.vibrating_groups
    assert 1 not in bc_b.vibrating_groups


def test_per_driver_bc_vibrating_velocity_is_unit() -> None:
    """Vibrating group must be at unit cone velocity (1+0j)."""
    rng = np.random.default_rng(1)
    backend = FakeBackend(rng, n_drivers=2)
    run_simulation(_two_driver_request(), backend=backend)
    for bc in backend.prepare_calls:
        for v in bc.vibrating_groups.values():
            assert v == complex(1.0, 0.0), f"Expected unit velocity, got {v}"


def test_three_driver_bc_groups() -> None:
    """With M=3, each driver gets exactly its own group (1, 2, 3)."""
    rng = np.random.default_rng(2)
    # Wide box (0.40 m) with 0.10 m spacing so gmsh can mesh all 3 disks cleanly.
    # Use coarse h_elem (0.05 m) via a low f_max to avoid tiny element counts.
    req = SimulationRequest(
        geometry=BoxGeometry(0.40, 0.20, 0.12),
        drivers=[
            DriverPlacement(
                DriverSpec((0.10, 0.10, 0.12), (0.0, 0.0, 1.0), 0.020),
                terminal=None,
                driver_id="drv_0",
            ),
            DriverPlacement(
                DriverSpec((0.20, 0.10, 0.12), (0.0, 0.0, 1.0), 0.020),
                terminal=None,
                driver_id="drv_1",
            ),
            DriverPlacement(
                DriverSpec((0.30, 0.10, 0.12), (0.0, 0.0, 1.0), 0.020),
                terminal=None,
                driver_id="drv_2",
            ),
        ],
        frequencies=_freq_grid(),
        sphere_n_points=14,
    )
    backend = FakeBackend(rng, n_drivers=3)
    run_simulation(req, backend=backend)

    assert len(backend.prepare_calls) == 3
    for m, bc in enumerate(backend.prepare_calls):
        expected_group = m + 1
        assert expected_group in bc.vibrating_groups
        for g in bc.vibrating_groups:
            assert (
                g == expected_group
            ), f"Driver {m} solve must only vibrate group {expected_group}, found {g}"


# ---------------------------------------------------------------------------
# Dataset shape and contract tests
# ---------------------------------------------------------------------------


def test_result_dataset_shape() -> None:
    """stacked_h_full must be [M, F, N] = [2, 3, 14]."""
    rng = np.random.default_rng(3)
    backend = FakeBackend(rng, n_drivers=2)
    result = run_simulation(_two_driver_request(), backend=backend)
    ds = result.dataset

    assert len(ds.drivers) == 2
    assert len(ds.frequencies) == F
    H_stack = stacked_h_full(ds)  # [M, F, N] complex128
    assert H_stack.shape == (2, F, N)
    assert H_stack.dtype == np.complex128


def test_h_full_identity_without_terminal() -> None:
    """With terminal=None, H_full must equal H_bem (terminal_response = ones)."""
    rng = np.random.default_rng(4)
    backend = FakeBackend(rng, n_drivers=1)
    req = SimulationRequest(
        geometry=BoxGeometry(0.12, 0.10, 0.08),
        drivers=[DriverPlacement(_driver_spec(), terminal=None, driver_id="drv_0")],
        frequencies=_freq_grid(),
        sphere_n_points=14,
    )
    result = run_simulation(req, backend=backend)
    d = result.dataset.drivers[0]
    np.testing.assert_array_equal(
        d.H_full, d.H_bem, err_msg="H_full should == H_bem when terminal=None"
    )


# ---------------------------------------------------------------------------
# Non-convergence flag propagation
# ---------------------------------------------------------------------------


def test_flagged_frequency_propagates() -> None:
    """Non-converged step (index 1) must appear in flagged_frequencies."""
    rng = np.random.default_rng(5)
    backend = FakeBackend(rng, n_drivers=2, make_flagged=True)
    result = run_simulation(_two_driver_request(), backend=backend)

    # Driver A's field is marked with conv[1]=False
    assert "drv_a" in result.flagged_frequencies
    flags_a = result.flagged_frequencies["drv_a"]
    assert flags_a.shape == (F,)
    assert bool(flags_a[1]) is True, "Step 1 must be flagged for drv_a"
    assert bool(flags_a[0]) is False


def test_flagged_propagates_to_hdf5(tmp_path: Path) -> None:
    """Flagged convergence_flags must persist in the written HDF5 file."""
    from beamsim2.io.hdf5_store import read_dataset

    rng = np.random.default_rng(6)
    backend = FakeBackend(rng, n_drivers=2, make_flagged=True)
    h5 = tmp_path / "flagged.h5"
    run_simulation(_two_driver_request(output_h5=h5), backend=backend)

    ds = read_dataset(h5)
    conv_a = ds.drivers[0].convergence_flags  # [F] bool; False = non-converged
    assert bool(conv_a[1]) is False, "HDF5 convergence_flags[1] must be False"


# ---------------------------------------------------------------------------
# HDF5 export round-trip
# ---------------------------------------------------------------------------


def test_hdf5_roundtrip_via_orchestrator(tmp_path: Path) -> None:
    """write/read through run_simulation produces exact bit-for-bit match."""
    from beamsim2.io.hdf5_store import read_dataset

    rng = np.random.default_rng(7)
    backend = FakeBackend(rng, n_drivers=2)
    h5 = tmp_path / "out.h5"
    result = run_simulation(_two_driver_request(output_h5=h5), backend=backend)

    assert h5.exists()
    ds = read_dataset(h5)
    assert len(ds.drivers) == 2
    assert stacked_h_full(ds).shape == (2, F, N)
    # Bit-exact H_bem roundtrip
    np.testing.assert_array_equal(ds.drivers[0].H_bem, result.dataset.drivers[0].H_bem)
    np.testing.assert_array_equal(ds.drivers[1].H_bem, result.dataset.drivers[1].H_bem)


# ---------------------------------------------------------------------------
# estimate_resources
# ---------------------------------------------------------------------------


def test_estimate_resources_shape() -> None:
    """estimate_resources must return valid ResourceEstimate without solving."""
    rng = np.random.default_rng(8)
    backend = FakeBackend(rng, n_drivers=2)
    req = _two_driver_request()
    est = estimate_resources(req, backend=backend)

    assert isinstance(est, ResourceEstimate)
    assert est.n_steps_total == F * 2  # M=2 drivers
    assert est.ram_bytes_per_step.shape == (F,)
    # peak_ram should be ≥ 0 (the fake returns 1e9 per step)
    assert est.peak_ram_bytes > 0
    # total_wall_seconds: fake returns all-NaN time, so falls back to heuristic
    assert est.total_wall_seconds >= 0


# ---------------------------------------------------------------------------
# work_dirs populated
# ---------------------------------------------------------------------------


def test_work_dirs_populated() -> None:
    """result.work_dirs must have one entry per driver."""
    rng = np.random.default_rng(9)
    backend = FakeBackend(rng, n_drivers=2)
    result = run_simulation(_two_driver_request(), backend=backend)
    assert set(result.work_dirs.keys()) == {"drv_a", "drv_b"}


# ---------------------------------------------------------------------------
# Reference-axis metadata production (Chunk 1) — the PRODUCER half
# ---------------------------------------------------------------------------


def test_reference_axis_default_written_to_dataset() -> None:
    """Default request stores reference_axis = +z in the dataset root attrs."""
    rng = np.random.default_rng(10)
    backend = FakeBackend(rng, n_drivers=2)
    result = run_simulation(_two_driver_request(), backend=backend)
    assert result.dataset.attrs["reference_axis"] == [0.0, 0.0, 1.0]


def test_reference_axis_custom_threads_through_pipeline() -> None:
    """A non-default reference_axis on the request must reach the dataset attrs.

    Guards the producer side: the results views read this attr, so if the
    pipeline failed to write it they would silently revert to hardcoded +z.
    """
    rng = np.random.default_rng(11)
    backend = FakeBackend(rng, n_drivers=2)
    req = _two_driver_request()
    req.reference_axis = (1.0, 0.0, 0.0)
    result = run_simulation(req, backend=backend)
    assert result.dataset.attrs["reference_axis"] == [1.0, 0.0, 0.0]


# ---------------------------------------------------------------------------
# Duplicate-id validation fires BEFORE the (expensive) solve (Chunk 1)
# ---------------------------------------------------------------------------


def test_duplicate_driver_id_rejected_before_solve() -> None:
    """run_simulation must reject a duplicate driver_id before touching the backend.

    The whole point of the up-front guard is to fail in milliseconds, not after a
    multi-hour solve — so assert prepare()/solve() were never reached.
    """
    import pytest

    rng = np.random.default_rng(12)
    backend = FakeBackend(rng, n_drivers=2)
    req = _two_driver_request()
    req.drivers[1].driver_id = req.drivers[0].driver_id  # force a collision
    with pytest.raises(ValueError, match="unique"):
        run_simulation(req, backend=backend)
    assert backend.prepare_calls == [], "validation must precede backend.prepare()"
    assert backend._call_count == 0, "validation must precede backend.solve()"


# ---------------------------------------------------------------------------
# Self-test smoke
# ---------------------------------------------------------------------------


def test_pipeline_run_self_test() -> None:
    """Verify that run.py's _self_test() passes without any backend."""
    from beamsim2.pipeline.run import _self_test

    _self_test()  # should not raise
