"""Tests for io/frd_export.py — VituixCAD .frd per-driver/per-angle exporter.

All tests are pure Python (no NumCalc binary required); runs in CI.

Coverage
--------
- Correct file count (M × fields × N) + manifest.csv
- Manifest rows match files on disk
- Each .frd has correct row count (F frequency rows)
- Frequency column round-trips exactly
- Magnitude column matches 20·log10(|H|/p_ref)
- Phase column matches np.rad2deg(np.angle(H))  — NOT re-zeroed
- Phase-origin guardrail (§3.4): a deliberate path-delay phase ramp survives
- H_full and H_bem produce different magnitude/phase when terminal_response ≠ 1
- Subset selection (driver_ids, fields)
- Error on unknown field name
- Error on unknown driver_id
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from beamsim2.assembly.tensor import RadiationDataset, build_dataset
from beamsim2.core.sphere import lebedev
from beamsim2.core.types import ComplexField
from beamsim2.io.frd_export import write_frd

# ── test fixture helpers ───────────────────────────────────────────────────────


def _make_complex_field(
    rng: np.random.Generator, n_freq: int, n_dir: int, freqs: np.ndarray
) -> ComplexField:
    """Synthetic complex field for testing."""
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
    """Minimal driver attrs for testing."""
    return {
        "name": f"driver_{idx}",
        "position": [0.0, 0.05 * idx, 0.0],
        "orientation": [0.0, 0.0, 1.0],
        "radius": 0.065,
        "profile": "piston",
        "ts_params": {"Re": 6.0, "Bl": 8.5, "Mms": 0.02, "Cms": 0.0006, "Rms": 1.2, "Sd": 0.0133},
        "terminal_response_model": "identity",
        "diaphragm_area": 0.0133,
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
    """Small two-driver synthetic dataset (4 freq, Lebedev-26 dirs)."""
    rng = np.random.default_rng(7)
    n_freq, n_dir = 4, 26
    freqs = np.array([250.0, 500.0, 1000.0, 2000.0], dtype=np.float64)
    obs = lebedev(n_points=26, radius=1.0)

    fields = [_make_complex_field(rng, n_freq, n_dir, freqs) for _ in range(2)]
    driver_inputs = [
        ("woofer_0", fields[0], _driver_attrs(0)),
        ("tweeter_1", fields[1], _driver_attrs(1)),
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


class TestFrdFileCount:
    def test_default_fields_all_drivers(self, dataset: RadiationDataset, tmp_path: Path) -> None:
        """Default exports 2 drivers × 2 fields × 26 dirs = 104 .frd files."""
        write_frd(tmp_path, dataset)
        frd_files = list(tmp_path.rglob("*.frd"))
        assert len(frd_files) == 2 * 2 * 26, f"Expected 104, got {len(frd_files)}"

    def test_single_field(self, dataset: RadiationDataset, tmp_path: Path) -> None:
        """Only H_full: 2 drivers × 1 field × 26 dirs = 52 files."""
        write_frd(tmp_path, dataset, fields=("H_full",))
        frd_files = list(tmp_path.rglob("*.frd"))
        assert len(frd_files) == 2 * 1 * 26

    def test_single_driver(self, dataset: RadiationDataset, tmp_path: Path) -> None:
        """Subset to one driver: 1 × 2 × 26 = 52 files."""
        write_frd(tmp_path, dataset, driver_ids=["woofer_0"])
        frd_files = list(tmp_path.rglob("*.frd"))
        assert len(frd_files) == 1 * 2 * 26

    def test_manifest_exists(self, dataset: RadiationDataset, tmp_path: Path) -> None:
        write_frd(tmp_path, dataset)
        assert (tmp_path / "manifest.csv").exists()

    def test_manifest_row_count(self, dataset: RadiationDataset, tmp_path: Path) -> None:
        """manifest.csv has one row per file (plus header)."""
        write_frd(tmp_path, dataset)
        with (tmp_path / "manifest.csv").open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 2 * 2 * 26


class TestFrdFormat:
    def _read_frd(self, path: Path) -> tuple[list[str], list[list[str]]]:
        """Return (comment_lines, data_rows_as_str_lists)."""
        lines = path.read_text(encoding="utf-8").splitlines()
        comments = [ln for ln in lines if ln.startswith("*")]
        data = [ln.split() for ln in lines if ln and not ln.startswith("*")]
        return comments, data

    def test_row_count_equals_freq_count(self, dataset: RadiationDataset, tmp_path: Path) -> None:
        """Each .frd has exactly F data rows."""
        write_frd(tmp_path, dataset, fields=("H_full",))
        sample = next((tmp_path / "woofer_0" / "H_full").glob("*.frd"))
        _, data = self._read_frd(sample)
        assert len(data) == len(dataset.frequencies), f"Expected {len(dataset.frequencies)} rows"

    def test_frequency_column_exact(self, dataset: RadiationDataset, tmp_path: Path) -> None:
        """Frequency column round-trips exactly (within float formatting)."""
        write_frd(tmp_path, dataset, fields=("H_full",))
        for path in (tmp_path / "woofer_0" / "H_full").glob("*.frd"):
            _, data = self._read_frd(path)
            read_freqs = np.array([float(row[0]) for row in data])
            assert np.allclose(
                read_freqs, dataset.frequencies, rtol=1e-6, atol=0
            ), f"Frequency mismatch in {path.name}"
            break  # one file sufficient

    def test_three_columns(self, dataset: RadiationDataset, tmp_path: Path) -> None:
        """Every data row has exactly 3 columns."""
        write_frd(tmp_path, dataset, fields=("H_full",))
        for path in (tmp_path / "woofer_0" / "H_full").glob("*.frd"):
            _, data = self._read_frd(path)
            for i, row in enumerate(data):
                assert len(row) == 3, f"Row {i} has {len(row)} columns in {path.name}"
            break

    def test_magnitude_formula(self, dataset: RadiationDataset, tmp_path: Path) -> None:
        """Magnitude column = 20·log10(|H|/20µPa), verified for direction 0."""
        p_ref = 20e-6
        write_frd(tmp_path, dataset, fields=("H_full",))
        path = tmp_path / "woofer_0" / "H_full" / "woofer_0_H_full_dir0000.frd"
        _, data = self._read_frd(path)
        read_db = np.array([float(row[1]) for row in data])

        drv = next(d for d in dataset.drivers if d.driver_id == "woofer_0")
        H_col = drv.H_full[:, 0]  # [F] complex128 — direction 0
        expected_db = 20.0 * np.log10(np.maximum(np.abs(H_col), 1e-300) / p_ref)
        assert np.allclose(
            read_db, expected_db, rtol=1e-5, atol=1e-5
        ), f"Magnitude mismatch: max diff = {np.max(np.abs(read_db - expected_db)):.2e}"

    def test_phase_formula(self, dataset: RadiationDataset, tmp_path: Path) -> None:
        """Phase column = np.rad2deg(np.angle(H)), direction 0."""
        write_frd(tmp_path, dataset, fields=("H_full",))
        path = tmp_path / "woofer_0" / "H_full" / "woofer_0_H_full_dir0000.frd"
        _, data = self._read_frd(path)
        read_phase = np.array([float(row[2]) for row in data])

        drv = next(d for d in dataset.drivers if d.driver_id == "woofer_0")
        H_col = drv.H_full[:, 0]  # [F] complex128
        expected_phase = np.rad2deg(np.angle(H_col))  # (-180, 180]
        assert np.allclose(
            read_phase, expected_phase, rtol=1e-5, atol=1e-4
        ), f"Phase mismatch: max diff = {np.max(np.abs(read_phase - expected_phase)):.4f}°"


class TestFrdPhaseOriginGuardrail:
    """§3.4 cardinal rule: deliberate path-delay phase must survive write/read."""

    def test_phase_ramp_preserved(self, tmp_path: Path) -> None:
        """A linear path-delay phase ramp must be written faithfully (never zeroed).

        We inject a known complex phase ramp (e^{j·k·d} at each frequency, mimicking
        a driver sitting at distance d from the origin) and confirm the .frd phase
        column reproduces it exactly.  Any re-zeroing or minimum-phasing would
        destroy this ramp.
        """
        rng = np.random.default_rng(99)
        n_freq = 5
        n_dir = 14
        c = 343.2  # m/s
        d = 0.1  # 10 cm offset — substantial delay

        freqs = np.array([200.0, 400.0, 800.0, 1600.0, 3200.0], dtype=np.float64)
        obs = lebedev(n_points=n_dir, radius=1.0)

        # Phase ramp: exp(j · 2π · f · d / c)
        omega = 2.0 * np.pi * freqs  # [F] rad/s
        k = omega / c  # [F] wavenumber
        phase_ramp = np.exp(1j * k * d)  # [F] unit-magnitude complex phasor

        # Combine with random non-unit magnitudes so the phase matters
        magnitudes = rng.uniform(0.01, 1.0, (n_freq, n_dir))
        pressure = magnitudes * phase_ramp[:, None]  # [F, N] complex128

        field = ComplexField(
            pressure=pressure.astype(np.complex128),
            convergence_flags=np.ones(n_freq, dtype=bool),
            frequencies=freqs,
        )
        ds = build_dataset(
            driver_inputs=[("drv_offset", field, {"name": "offset driver"})],
            directions=obs,
            root_attrs={"pressure_convention": "test"},
        )

        write_frd(tmp_path, ds, fields=("H_full",))

        # Read back direction 0
        path = tmp_path / "drv_offset" / "H_full" / "drv_offset_H_full_dir0000.frd"
        lines = path.read_text().splitlines()
        data = [ln.split() for ln in lines if ln and not ln.startswith("*")]
        read_phase = np.array([float(row[2]) for row in data])  # [F] degrees

        expected_phase = np.rad2deg(np.angle(pressure[:, 0]))  # [F] degrees
        # The ramp is monotonic (increases with frequency); check exact reproduction
        assert np.allclose(read_phase, expected_phase, atol=1e-4), (
            f"Phase ramp corrupted — max error {np.max(np.abs(read_phase - expected_phase)):.4f}°"
            " (§3.4: global-origin phase must NOT be re-zeroed)"
        )


class TestFrdFieldDifference:
    def test_h_full_differs_from_h_bem(self, dataset: RadiationDataset, tmp_path: Path) -> None:
        """H_full and H_bem files differ when terminal_response ≠ 1."""
        write_frd(tmp_path, dataset, fields=("H_full", "H_bem"))

        def read_col(path: Path, col: int) -> np.ndarray:
            lines = path.read_text().splitlines()
            return np.array(
                [float(ln.split()[col]) for ln in lines if ln and not ln.startswith("*")]
            )

        mag_full = read_col(tmp_path / "woofer_0" / "H_full" / "woofer_0_H_full_dir0000.frd", 1)
        mag_bem = read_col(tmp_path / "woofer_0" / "H_bem" / "woofer_0_H_bem_dir0000.frd", 1)
        # terminal_response ≠ ones, so magnitudes must differ
        assert not np.allclose(
            mag_full, mag_bem
        ), "H_full and H_bem magnitudes are identical but terminal_response ≠ 1"


class TestFrdErrors:
    def test_unknown_field_raises(self, dataset: RadiationDataset, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="unknown field"):
            write_frd(tmp_path, dataset, fields=("H_mystery",))

    def test_unknown_driver_id_raises(self, dataset: RadiationDataset, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="driver_id"):
            write_frd(tmp_path, dataset, driver_ids=["no_such_driver"])


class TestManifestContent:
    def test_manifest_fields_present(self, dataset: RadiationDataset, tmp_path: Path) -> None:
        """manifest.csv has all required column headers."""
        write_frd(tmp_path, dataset, fields=("H_full",))
        with (tmp_path / "manifest.csv").open(encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            headers = reader.fieldnames or []
        required = {
            "file",
            "driver_id",
            "field",
            "direction_index",
            "ux",
            "uy",
            "uz",
            "theta_deg",
            "phi_deg",
        }
        assert required.issubset(set(headers)), f"Missing columns: {required - set(headers)}"

    def test_manifest_files_exist(self, dataset: RadiationDataset, tmp_path: Path) -> None:
        """Every path in manifest.csv exists on disk."""
        write_frd(tmp_path, dataset, fields=("H_full",))
        with (tmp_path / "manifest.csv").open(encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                p = tmp_path / row["file"]
                assert p.exists(), f"manifest entry missing: {row['file']}"

    def test_manifest_theta_phi_range(self, dataset: RadiationDataset, tmp_path: Path) -> None:
        """theta ∈ [0, 180], phi ∈ [0, 360) for all manifest rows."""
        write_frd(tmp_path, dataset, fields=("H_full",))
        with (tmp_path / "manifest.csv").open(encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                th = float(row["theta_deg"])
                ph = float(row["phi_deg"])
                assert 0.0 <= th <= 180.0, f"theta out of range: {th}"
                assert 0.0 <= ph < 360.0, f"phi out of range: {ph}"
