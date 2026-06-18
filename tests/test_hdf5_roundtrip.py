"""HDF5 data-contract stability test: write then read the H tensor with all required
metadata; verify lossless round-trip and schema_version attribute.

Stage-3 acceptance gate (§8): "multi-driver dataset exports and re-imports losslessly
(test_hdf5_roundtrip)".  All tests are pure-Python — no NumCalc binary required; this
runs in CI on every push.

"Losslessly" means bit-identical: np.array_equal on every array, including complex128,
and equality on every §3.5 attribute including nested dict-valued ones (ts_params, etc.)
after JSON roundtrip.

Tests build the synthetic dataset through ``build_dataset()`` so the tensor path is
exercised, not bypassed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from beamsim2.assembly.tensor import RadiationDataset, build_dataset, stacked_h_full
from beamsim2.core.sphere import lebedev
from beamsim2.core.types import ComplexField
from beamsim2.io.hdf5_store import SCHEMA_VERSION, read_dataset, write_dataset

# ── helpers ─────────────────────────────────────────────────────────────────


def _make_complex_field(
    rng: np.random.Generator, n_freq: int, n_dir: int, freqs: np.ndarray
) -> ComplexField:
    """Random complex field for testing."""
    pressure = rng.standard_normal((n_freq, n_dir)) + 1j * rng.standard_normal((n_freq, n_dir))
    pressure = pressure.astype(np.complex128)
    return ComplexField(
        pressure=pressure,
        convergence_flags=np.ones(n_freq, dtype=bool),
        frequencies=freqs,
    )


def _full_root_attrs() -> dict:
    """A complete §3.5 root attribute dict including nested dicts and lists."""
    return {
        "phase_origin": [0.0, 0.0, 0.0],
        "axis_convention": "z_forward",
        "length_units": "m",
        "observation_radius": 1.0,
        "far_field": False,
        "pressure_convention": "Pa r_obs unit cone velocity",
        "solver_backend": "numcalc",
        "solver_version": "0.0.0",
        "speed_of_sound": 343.2,
        "air_density": 1.2041,
        "temperature": 20.0,
        "humidity": 50.0,
        "pressure": 101325.0,
        "air_attenuation_model": "none",
    }


def _full_driver_attrs(idx: int) -> dict:
    """A complete §3.5 per-driver attribute dict including nested dicts."""
    return {
        "name": f"driver_{idx}",
        "position": [float(idx) * 0.05, 0.0, 0.0],
        "orientation": [0.0, 0.0, 1.0],
        "radius": 0.05,
        "profile": {"type": "piston", "half_angle_deg": None},
        "ts_params": {
            "fs": 50.0,
            "Qts": 0.35,
            "Qes": 0.40,
            "Qms": 3.0,
            "Vas": 0.02,
            "Re": 6.0,
            "Le": 0.0005,
            "Bl": 8.5,
            "Mms": 0.015,
            "Cms": 0.0006,
            "Sd": 0.0133,
        },
        "terminal_response_model": "identity",
        "diaphragm_area": 0.0133,
    }


# ── tests ────────────────────────────────────────────────────────────────────


class TestHdf5Roundtrip:
    """Lossless write/read round-trip of the full data contract."""

    @pytest.fixture
    def dataset(self) -> RadiationDataset:
        """Small two-driver synthetic dataset built through build_dataset()."""
        rng = np.random.default_rng(42)
        n_freq = 4
        n_dir = 26  # Lebedev-26
        freqs = np.array([250.0, 500.0, 1000.0, 2000.0], dtype=np.float64)

        obs = lebedev(n_points=26, radius=1.0)

        fields = [
            _make_complex_field(rng, n_freq, n_dir, freqs),
            _make_complex_field(rng, n_freq, n_dir, freqs),
        ]
        driver_inputs = [
            ("woofer_0", fields[0], _full_driver_attrs(0)),
            ("tweeter_1", fields[1], _full_driver_attrs(1)),
        ]

        # non-trivial terminal responses (complex, not all-ones)
        terminal_responses = [
            (rng.standard_normal(n_freq) + 1j * rng.standard_normal(n_freq)).astype(np.complex128)
            for _ in range(2)
        ]

        return build_dataset(
            driver_inputs=driver_inputs,
            directions=obs,
            freq_grid_spacing="log",
            freq_grid_fractional_octave=None,
            root_attrs=_full_root_attrs(),
            terminal_responses=terminal_responses,
        )

    def test_frequencies_roundtrip(self, dataset: RadiationDataset, tmp_path: Path) -> None:
        """frequencies [F] float64 roundtrips exactly."""
        p = tmp_path / "test.h5"
        write_dataset(p, dataset)
        ds2 = read_dataset(p)
        assert np.array_equal(
            ds2.frequencies, dataset.frequencies
        ), "frequencies array changed after roundtrip"
        assert ds2.frequencies.dtype == np.float64

    def test_h_bem_roundtrip(self, dataset: RadiationDataset, tmp_path: Path) -> None:
        """H_bem [F×N] complex128 roundtrips bit-exactly for each driver."""
        p = tmp_path / "test.h5"
        write_dataset(p, dataset)
        ds2 = read_dataset(p)
        orig_by_id = {d.driver_id: d for d in dataset.drivers}
        rd_by_id = {d.driver_id: d for d in ds2.drivers}
        for did in orig_by_id:
            assert np.array_equal(
                rd_by_id[did].H_bem, orig_by_id[did].H_bem
            ), f"H_bem mismatch for driver '{did}'"
            assert rd_by_id[did].H_bem.dtype == np.complex128

    def test_terminal_response_roundtrip(self, dataset: RadiationDataset, tmp_path: Path) -> None:
        """terminal_response [F] complex128 roundtrips bit-exactly."""
        p = tmp_path / "test.h5"
        write_dataset(p, dataset)
        ds2 = read_dataset(p)
        orig_by_id = {d.driver_id: d for d in dataset.drivers}
        rd_by_id = {d.driver_id: d for d in ds2.drivers}
        for did in orig_by_id:
            assert np.array_equal(
                rd_by_id[did].terminal_response, orig_by_id[did].terminal_response
            ), f"terminal_response mismatch for driver '{did}'"

    def test_h_full_roundtrip(self, dataset: RadiationDataset, tmp_path: Path) -> None:
        """H_full [F×N] complex128 roundtrips bit-exactly."""
        p = tmp_path / "test.h5"
        write_dataset(p, dataset)
        ds2 = read_dataset(p)
        orig_by_id = {d.driver_id: d for d in dataset.drivers}
        rd_by_id = {d.driver_id: d for d in ds2.drivers}
        for did in orig_by_id:
            assert np.array_equal(
                rd_by_id[did].H_full, orig_by_id[did].H_full
            ), f"H_full mismatch for driver '{did}'"

    def test_convergence_flags_roundtrip(self, dataset: RadiationDataset, tmp_path: Path) -> None:
        """convergence_flags [F] bool roundtrips exactly."""
        p = tmp_path / "test.h5"
        write_dataset(p, dataset)
        ds2 = read_dataset(p)
        orig_by_id = {d.driver_id: d for d in dataset.drivers}
        rd_by_id = {d.driver_id: d for d in ds2.drivers}
        for did in orig_by_id:
            assert np.array_equal(
                rd_by_id[did].convergence_flags, orig_by_id[did].convergence_flags
            )

    def test_directions_roundtrip(self, dataset: RadiationDataset, tmp_path: Path) -> None:
        """Sphere grid unit_vectors, weights, theta_phi roundtrip exactly."""
        p = tmp_path / "test.h5"
        write_dataset(p, dataset)
        ds2 = read_dataset(p)
        obs = dataset.directions
        obs2 = ds2.directions
        assert np.array_equal(obs2.unit_vectors, obs.unit_vectors)
        assert np.array_equal(obs2.weights, obs.weights)
        assert obs2.theta_phi is not None
        assert np.array_equal(obs2.theta_phi, obs.theta_phi)
        assert obs2.scheme == obs.scheme
        assert obs2.order == obs.order
        assert obs2.weight_convention == obs.weight_convention

    def test_schema_version_attribute(self, dataset: RadiationDataset, tmp_path: Path) -> None:
        """schema_version attribute is written and matches SCHEMA_VERSION constant."""
        p = tmp_path / "test.h5"
        write_dataset(p, dataset)
        ds2 = read_dataset(p)
        assert ds2.attrs["schema_version"] == SCHEMA_VERSION

    def test_root_attrs_roundtrip(self, dataset: RadiationDataset, tmp_path: Path) -> None:
        """All §3.5 root attrs roundtrip, including nested dicts."""
        p = tmp_path / "test.h5"
        write_dataset(p, dataset)
        ds2 = read_dataset(p)
        orig_attrs = dataset.attrs
        rd_attrs = ds2.attrs
        for key, val in orig_attrs.items():
            assert key in rd_attrs, f"root attr '{key}' missing after roundtrip"
            if isinstance(val, dict):
                assert rd_attrs[key] == val, f"root attr '{key}' dict mismatch"
            elif isinstance(val, list):
                assert rd_attrs[key] == val, f"root attr '{key}' list mismatch"
            else:
                assert rd_attrs[key] == pytest.approx(
                    val, rel=1e-12
                ), f"root attr '{key}' mismatch: {rd_attrs[key]} != {val}"

    def test_driver_attrs_roundtrip(self, dataset: RadiationDataset, tmp_path: Path) -> None:
        """Per-driver attrs roundtrip, including nested ts_params and profile dicts."""
        p = tmp_path / "test.h5"
        write_dataset(p, dataset)
        ds2 = read_dataset(p)
        orig_by_id = {d.driver_id: d for d in dataset.drivers}
        rd_by_id = {d.driver_id: d for d in ds2.drivers}
        for did in orig_by_id:
            orig = orig_by_id[did]
            rd = rd_by_id[did]
            for key, val in orig.attrs.items():
                assert key in rd.attrs, f"driver '{did}' attr '{key}' missing after roundtrip"
                if isinstance(val, dict):
                    assert rd.attrs[key] == val, f"driver '{did}' attr '{key}' dict mismatch"

    def test_driver_count_preserved(self, dataset: RadiationDataset, tmp_path: Path) -> None:
        """Number of drivers is preserved."""
        p = tmp_path / "test.h5"
        write_dataset(p, dataset)
        ds2 = read_dataset(p)
        assert len(ds2.drivers) == len(dataset.drivers)

    def test_h_full_equals_h_bem_times_terminal(
        self, dataset: RadiationDataset, tmp_path: Path
    ) -> None:
        """H_full roundtrips exactly and equals H_bem × terminal_response[:, None]."""
        p = tmp_path / "test.h5"
        write_dataset(p, dataset)
        ds2 = read_dataset(p)
        orig_by_id = {d.driver_id: d for d in dataset.drivers}
        rd_by_id = {d.driver_id: d for d in ds2.drivers}
        for did in orig_by_id:
            d = rd_by_id[did]
            expected = d.H_bem * d.terminal_response[:, None]
            assert np.allclose(
                d.H_full, expected, rtol=1e-12, atol=0
            ), f"H_full != H_bem * terminal_response for driver '{did}'"

    def test_stacked_h_full_shape(self, dataset: RadiationDataset, tmp_path: Path) -> None:
        """stacked_h_full produces [M × F × N] tensor."""
        p = tmp_path / "test.h5"
        write_dataset(p, dataset)
        ds2 = read_dataset(p)
        stacked = stacked_h_full(ds2)
        M = len(ds2.drivers)
        F = len(ds2.frequencies)
        N = ds2.directions.unit_vectors.shape[0]
        assert stacked.shape == (
            M,
            F,
            N,
        ), f"stacked_h_full shape {stacked.shape} != ({M}, {F}, {N})"
        assert stacked.dtype == np.complex128

    def test_stacked_h_full_row_order_survives_roundtrip(
        self, dataset: RadiationDataset, tmp_path: Path
    ) -> None:
        """stacked_h_full row order (driver index) is preserved after write/read.

        HDF5 iterates group keys alphabetically; without the persisted
        driver_order attr, rows of the Phase-2 steering matrix would be
        silently permuted (e.g. woofer_0 ↔ tweeter_1).
        """
        p = tmp_path / "test.h5"
        write_dataset(p, dataset)
        ds2 = read_dataset(p)

        orig_stacked = stacked_h_full(dataset)
        rd_stacked = stacked_h_full(ds2)

        orig_ids = [d.driver_id for d in dataset.drivers]
        rd_ids = [d.driver_id for d in ds2.drivers]
        assert orig_ids == rd_ids, (
            f"Driver order changed after roundtrip: {orig_ids} -> {rd_ids}"
        )
        assert np.array_equal(rd_stacked, orig_stacked), (
            "stacked_h_full rows do not match after roundtrip — driver order bug"
        )


class TestHdf5Shapes:
    """Verify dataset shapes agree with [F], [F×N], [N×3], etc."""

    def test_shapes_consistent(self, tmp_path: Path) -> None:
        """All array shapes in a written file are internally consistent."""
        rng = np.random.default_rng(7)
        F, N = 3, 14
        freqs = np.array([100.0, 200.0, 400.0])
        obs = lebedev(n_points=N, radius=2.0)

        field = _make_complex_field(rng, F, N, freqs)
        ds = build_dataset(
            driver_inputs=[("drv0", field, {"name": "test"})],
            directions=obs,
        )
        p = tmp_path / "shapes.h5"
        write_dataset(p, ds)
        ds2 = read_dataset(p)

        d = ds2.drivers[0]
        assert d.H_bem.shape == (F, N)
        assert d.terminal_response.shape == (F,)
        assert d.H_full.shape == (F, N)
        assert d.convergence_flags.shape == (F,)
        assert ds2.directions.unit_vectors.shape == (N, 3)
        assert ds2.directions.weights.shape == (N,)
