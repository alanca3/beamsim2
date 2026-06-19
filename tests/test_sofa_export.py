"""Tests for io/sofa_export.py — SOFA AES69 GeneralTF exporter.

All tests are pure Python (no NumCalc binary required); runs in CI.
sofar is a production dependency (not dev-only).

Coverage
--------
- File is written and readable with sofar.read_sofa
- Exact complex roundtrip: Data_Real + j*Data_Imag reconstructs H_full exactly
- Dimension layout: Data shape [M=drivers, R=directions, N=freqs]
- SourcePosition matches driver positions from attrs
- ReceiverPosition matches unit_vectors × observation_radius
- Frequency vector (N) matches dataset.frequencies
- H_full vs H_bem produce different data when terminal_response ≠ 1
- Driver subset (driver_ids) selection
- Global comment contains phase-origin note (§3.4 guardrail)
- Error on unknown field name
- Error on unknown driver_id
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import sofar

from beamsim2.assembly.tensor import RadiationDataset, build_dataset
from beamsim2.core.sphere import lebedev
from beamsim2.core.types import ComplexField
from beamsim2.io.sofa_export import write_sofa

# ── fixture helpers ────────────────────────────────────────────────────────────


def _make_complex_field(
    rng: np.random.Generator, n_freq: int, n_dir: int, freqs: np.ndarray
) -> ComplexField:
    pressure = (
        rng.standard_normal((n_freq, n_dir)) + 1j * rng.standard_normal((n_freq, n_dir))
    ).astype(
        np.complex128
    )  # [F × N] complex128
    return ComplexField(
        pressure=pressure,
        convergence_flags=np.ones(n_freq, dtype=bool),
        frequencies=freqs,
    )


def _driver_attrs(idx: int) -> dict:
    return {
        "name": f"driver_{idx}",
        "position": [0.0, float(idx) * 0.05, 0.0],
        "orientation": [0.0, 0.0, 1.0],
        "radius": 0.065,
        "profile": "piston",
        "ts_params": {"Re": 6.0},
        "terminal_response_model": "identity",
    }


def _root_attrs() -> dict:
    return {
        "phase_origin": [0.0, 0.0, 0.0],
        "axis_convention": "x-right y-up z-front",
        "length_units": "metres",
        "observation_radius": 1.0,
        "far_field": False,
        "pressure_convention": "Pa at r_obs for unit cone velocity",
        "speed_of_sound": 343.2,
        "air_density": 1.204,
    }


@pytest.fixture
def dataset() -> RadiationDataset:
    """Two-driver, Lebedev-26, 4-frequency synthetic dataset."""
    rng = np.random.default_rng(13)
    n_freq, n_dir = 4, 26
    freqs = np.array([250.0, 500.0, 1000.0, 2000.0], dtype=np.float64)
    obs = lebedev(n_points=26, radius=1.5)  # r_obs = 1.5 m to exercise ReceiverPosition

    fields = [_make_complex_field(rng, n_freq, n_dir, freqs) for _ in range(2)]
    driver_inputs = [
        ("bass_0", fields[0], _driver_attrs(0)),
        ("mid_1", fields[1], _driver_attrs(1)),
    ]
    terminal_responses = [
        (rng.standard_normal(n_freq) + 1j * rng.standard_normal(n_freq)).astype(np.complex128)
        for _ in range(2)
    ]
    return build_dataset(
        driver_inputs=driver_inputs,
        directions=obs,
        freq_grid_spacing="log",
        root_attrs=_root_attrs(),
        terminal_responses=terminal_responses,
    )


# ── tests ─────────────────────────────────────────────────────────────────────


class TestSofaRoundtrip:
    def test_file_written_and_readable(self, dataset: RadiationDataset, tmp_path: Path) -> None:
        path = tmp_path / "out.sofa"
        write_sofa(path, dataset)
        assert path.with_suffix(".sofa").exists() or path.exists()
        # sofar appends .sofa if missing; tolerate either form
        p = path if path.exists() else path.with_suffix(".sofa")
        sofa_obj = sofar.read_sofa(str(p))
        assert sofa_obj is not None

    def _sofa_path(self, tmp_path: Path, name: str = "out.sofa") -> Path:
        """Return the actual path sofar wrote (handles optional .sofa append)."""
        candidate = tmp_path / name
        if candidate.exists():
            return candidate
        with_ext = candidate.with_suffix(".sofa")
        if with_ext.exists():
            return with_ext
        # sofar may have written <name>.sofa where name already has .sofa → out.sofa.sofa
        doubled = tmp_path / (name + ".sofa")
        if doubled.exists():
            return doubled
        raise FileNotFoundError(f"Cannot find SOFA file at {candidate} or {with_ext}")

    def test_complex_roundtrip_h_full(self, dataset: RadiationDataset, tmp_path: Path) -> None:
        """Data_Real + j·Data_Imag exactly reconstructs H_full for all drivers."""
        path = tmp_path / "rt.sofa"
        write_sofa(path, dataset, field="H_full")
        sofa_obj = sofar.read_sofa(str(self._sofa_path(tmp_path, "rt.sofa")))

        H_rt = sofa_obj.Data_Real + 1j * sofa_obj.Data_Imag  # [M, R, N] complex128

        for m, drv in enumerate(dataset.drivers):
            # dataset H_full is [F, N_dir]; SOFA is [M, R, N] = [drivers, dirs, freqs]
            # so H_rt[m] is [R, N] = [dirs, freqs] = H_full.T
            H_expected = drv.H_full.T  # [N_dir, F] complex128 = [R, N]
            assert np.allclose(H_rt[m], H_expected, rtol=1e-12, atol=0), (
                f"H_full roundtrip mismatch for driver '{drv.driver_id}'"
                f"; max err = {np.max(np.abs(H_rt[m] - H_expected)):.2e}"
            )

    def test_complex_roundtrip_h_bem(self, dataset: RadiationDataset, tmp_path: Path) -> None:
        """Data roundtrip works for H_bem field too."""
        path = tmp_path / "bem.sofa"
        write_sofa(path, dataset, field="H_bem")
        sofa_obj = sofar.read_sofa(str(self._sofa_path(tmp_path, "bem.sofa")))
        H_rt = sofa_obj.Data_Real + 1j * sofa_obj.Data_Imag  # [M, R, N]
        drv = dataset.drivers[0]
        assert np.allclose(H_rt[0], drv.H_bem.T, rtol=1e-12, atol=0)


class TestSofaDimensions:
    def _write_read(self, dataset: RadiationDataset, tmp_path: Path, name: str = "dim.sofa"):
        path = tmp_path / name
        write_sofa(path, dataset)
        # resolve actual path
        for p in [path, path.with_suffix(".sofa"), tmp_path / (name + ".sofa")]:
            if p.exists():
                return sofar.read_sofa(str(p))
        raise FileNotFoundError(f"SOFA file not found near {path}")

    def test_data_shape(self, dataset: RadiationDataset, tmp_path: Path) -> None:
        """Data_Real shape is [M=2, R=26, N=4]."""
        sofa_obj = self._write_read(dataset, tmp_path)
        M = len(dataset.drivers)
        R = len(dataset.directions.unit_vectors)
        N = len(dataset.frequencies)
        assert sofa_obj.Data_Real.shape == (
            M,
            R,
            N,
        ), f"Expected ({M}, {R}, {N}), got {sofa_obj.Data_Real.shape}"

    def test_frequency_vector(self, dataset: RadiationDataset, tmp_path: Path) -> None:
        """N (frequency) array round-trips exactly."""
        sofa_obj = self._write_read(dataset, tmp_path, "freq.sofa")
        assert np.allclose(sofa_obj.N, dataset.frequencies, rtol=1e-10, atol=0)

    def test_receiver_positions(self, dataset: RadiationDataset, tmp_path: Path) -> None:
        """ReceiverPosition = unit_vectors × r_obs, shape [R, 3]."""
        sofa_obj = self._write_read(dataset, tmp_path, "recv.sofa")
        r_obs = dataset.directions.radius
        expected = dataset.directions.unit_vectors * r_obs  # [R, 3] float64
        assert np.allclose(sofa_obj.ReceiverPosition, expected, rtol=1e-12, atol=0), (
            f"ReceiverPosition mismatch; max err = "
            f"{np.max(np.abs(sofa_obj.ReceiverPosition - expected)):.2e}"
        )

    def test_source_positions(self, dataset: RadiationDataset, tmp_path: Path) -> None:
        """SourcePosition entries match driver positions from attrs."""
        sofa_obj = self._write_read(dataset, tmp_path, "src.sofa")
        for m, drv in enumerate(dataset.drivers):
            expected = np.array(drv.attrs.get("position", [0.0, 0.0, 0.0]), dtype=np.float64)
            actual = np.asarray(sofa_obj.SourcePosition)[m]
            assert np.allclose(
                actual, expected, atol=1e-12
            ), f"SourcePosition[{m}] mismatch: got {actual}, expected {expected}"


class TestSofaPhaseOrigin:
    """§3.4 guardrail: comment must state phase is not re-zeroed."""

    def test_global_comment_mentions_phase_origin(
        self, dataset: RadiationDataset, tmp_path: Path
    ) -> None:
        path = tmp_path / "phase.sofa"
        write_sofa(path, dataset)
        for p in [path, path.with_suffix(".sofa"), tmp_path / "phase.sofa.sofa"]:
            if p.exists():
                sofa_obj = sofar.read_sofa(str(p))
                comment = sofa_obj.GLOBAL_Comment
                assert (
                    "NOT re-zeroed" in comment or "not re-zeroed" in comment.lower()
                ), f"GLOBAL_Comment does not state phase-origin rule: {comment[:200]}"
                return
        raise FileNotFoundError("SOFA file not found")


class TestSofaFieldDifference:
    def test_h_full_differs_from_h_bem(self, dataset: RadiationDataset, tmp_path: Path) -> None:
        """Writing H_full vs H_bem produces different complex data."""
        p_full = tmp_path / "full.sofa"
        p_bem = tmp_path / "bem.sofa"
        write_sofa(p_full, dataset, field="H_full")
        write_sofa(p_bem, dataset, field="H_bem")

        def _resolve(p: Path) -> Path:
            for c in [p, p.with_suffix(".sofa"), Path(str(p) + ".sofa")]:
                if c.exists():
                    return c
            raise FileNotFoundError(str(p))

        s_full = sofar.read_sofa(str(_resolve(p_full)))
        s_bem = sofar.read_sofa(str(_resolve(p_bem)))
        H_full = s_full.Data_Real + 1j * s_full.Data_Imag
        H_bem = s_bem.Data_Real + 1j * s_bem.Data_Imag
        assert not np.allclose(
            H_full, H_bem
        ), "H_full and H_bem data are identical but terminal_response ≠ 1"


class TestSofaSubset:
    def test_driver_subset(self, dataset: RadiationDataset, tmp_path: Path) -> None:
        """Requesting one driver produces M=1 in Data_Real."""
        path = tmp_path / "one.sofa"
        write_sofa(path, dataset, driver_ids=["bass_0"])
        for p in [path, path.with_suffix(".sofa"), tmp_path / "one.sofa.sofa"]:
            if p.exists():
                sofa_obj = sofar.read_sofa(str(p))
                assert (
                    sofa_obj.Data_Real.shape[0] == 1
                ), f"Expected M=1, got {sofa_obj.Data_Real.shape[0]}"
                return
        raise FileNotFoundError("SOFA subset file not found")


class TestSofaErrors:
    def test_unknown_field_raises(self, dataset: RadiationDataset, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="unknown field"):
            write_sofa(tmp_path / "err.sofa", dataset, field="H_mystery")

    def test_unknown_driver_id_raises(self, dataset: RadiationDataset, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="driver_id"):
            write_sofa(tmp_path / "err2.sofa", dataset, driver_ids=["no_such"])
