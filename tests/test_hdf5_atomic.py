"""HDF5 atomic write + attr hardening tests.

CI-safe (no NumCalc binary, no large RAM).  Verifies three guarantees:

1. **Exact-bug reproduction (asymmetric None):** The real-world failure —
   ``driver_0`` carries ``box_volume_m3=<float>`` (writes fine), ``driver_1``
   carries ``box_volume_m3=None`` (used to crash h5py with object-dtype error)
   — now succeeds.  The float survives; the None attr is absent on read-back.

2. **Atomicity:** A write that raises mid-stream (via an injected object-dtype
   ndarray attr that cannot be serialised) leaves any *pre-existing* file
   byte-identical.  The temp file is cleaned up; the error message names the
   offending driver id and attr key.

3. **Lossless 2-driver round-trip:** An all-valid-attrs 2-driver dataset
   survives write → read with arrays identical to the originals and driver
   insertion order preserved.

References
----------
docs/Chunk5c_Kickoff_Prompt.md — Chunk 5c acceptance gate.
DATA_CONTRACT.md §3.5 — the per-driver attr contract.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from beamsim2.assembly.tensor import build_dataset
from beamsim2.core.sphere import lebedev
from beamsim2.core.types import ComplexField
from beamsim2.io.hdf5_store import read_dataset, write_dataset

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_F = 3  # frequency steps (minimal for speed)
_N = 14  # Lebedev-14 directions (smallest valid grid)


def _freqs() -> np.ndarray:
    return np.array([250.0, 500.0, 1000.0])  # [F] Hz


def _obs() -> object:
    """Smallest valid Lebedev grid."""
    return lebedev(n_points=_N, radius=1.0)


def _field(seed: int) -> ComplexField:
    """Minimal random complex128 ComplexField."""
    rng = np.random.default_rng(seed)
    pressure = (rng.standard_normal((_F, _N)) + 1j * rng.standard_normal((_F, _N))).astype(
        np.complex128
    )
    return ComplexField(
        pressure=pressure,
        convergence_flags=np.ones(_F, dtype=bool),
        frequencies=_freqs(),
    )


def _build_2driver(
    extra_d0: dict | None = None,
    extra_d1: dict | None = None,
) -> object:
    """Assemble a minimal 2-driver RadiationDataset.

    Parameters
    ----------
    extra_d0 : dict, optional
        Extra attrs merged into driver_0's attr dict.
    extra_d1 : dict, optional
        Extra attrs merged into driver_1's attr dict.
    """
    obs = _obs()
    base_attrs = {
        "name": "placeholder",
        "position": [0.0, 0.0, 0.0],
        "orientation": [0.0, 0.0, 1.0],
        "radius": 0.02,
        "profile": "flush_disk",
    }

    def _attrs(driver_name: str, extra: dict | None) -> dict:
        d = {**base_attrs, "name": driver_name}
        if extra:
            d.update(extra)
        return d

    return build_dataset(
        driver_inputs=[
            ("driver_0", _field(0), _attrs("driver_0", extra_d0)),
            ("driver_1", _field(1), _attrs("driver_1", extra_d1)),
        ],
        directions=obs,
        freq_grid_spacing="log",
        root_attrs={"phase_origin": [0.0, 0.0, 0.0]},
    )


# ---------------------------------------------------------------------------
# 1.  Exact-bug reproduction — asymmetric None (box_volume_m3)
# ---------------------------------------------------------------------------


class TestAsymmetricNone:
    """Regression test for the real failure: driver_1 had box_volume_m3=None."""

    def test_write_succeeds_with_none_attr(self, tmp_path: Path) -> None:
        """Write must not raise when one driver has box_volume_m3=None."""
        ds = _build_2driver(
            extra_d0={"box_volume_m3": 0.02},  # float — used to write fine
            extra_d1={"box_volume_m3": None},  # None — used to crash h5py
        )
        p = tmp_path / "asymmetric.h5"
        write_dataset(p, ds)
        assert p.exists(), "File must exist after successful write"

    def test_float_attr_survives_roundtrip(self, tmp_path: Path) -> None:
        """The float box_volume_m3 on driver_0 must survive read-back."""
        ds = _build_2driver(
            extra_d0={"box_volume_m3": 0.02},
            extra_d1={"box_volume_m3": None},
        )
        p = tmp_path / "roundtrip_float.h5"
        write_dataset(p, ds)
        ds2 = read_dataset(p)

        d0 = next(d for d in ds2.drivers if d.driver_id == "driver_0")
        assert "box_volume_m3" in d0.attrs, "driver_0 box_volume_m3 must be written"
        assert float(d0.attrs["box_volume_m3"]) == pytest.approx(0.02)

    def test_none_attr_is_absent_on_readback(self, tmp_path: Path) -> None:
        """The None box_volume_m3 on driver_1 must be absent after read-back
        (skip-None policy: absent attr = unset is the HDF5 idiom)."""
        ds = _build_2driver(
            extra_d0={"box_volume_m3": 0.02},
            extra_d1={"box_volume_m3": None},
        )
        p = tmp_path / "roundtrip_none.h5"
        write_dataset(p, ds)
        ds2 = read_dataset(p)

        d1 = next(d for d in ds2.drivers if d.driver_id == "driver_1")
        # None was skipped on write → key should be absent or None on read-back
        assert d1.attrs.get("box_volume_m3") is None, (
            "driver_1 box_volume_m3 (None) should not materialise as a non-None "
            f"value after read-back; got {d1.attrs.get('box_volume_m3')!r}"
        )

    def test_both_drivers_present_after_write(self, tmp_path: Path) -> None:
        """Both drivers must be present — no corrupt partial file."""
        ds = _build_2driver(
            extra_d0={"box_volume_m3": 0.02},
            extra_d1={"box_volume_m3": None},
        )
        p = tmp_path / "both_drivers.h5"
        write_dataset(p, ds)
        ds2 = read_dataset(p)

        ids = [d.driver_id for d in ds2.drivers]
        assert "driver_0" in ids
        assert "driver_1" in ids


# ---------------------------------------------------------------------------
# 2.  Atomicity — failed write must not corrupt a pre-existing file
# ---------------------------------------------------------------------------


class TestAtomicWrite:
    """A write that raises mid-stream must leave the original file intact."""

    def _bad_dataset(self) -> object:
        """Build a dataset with an object-dtype ndarray attr on driver_1.

        An object-dtype ndarray survives the None-skip and dict/list-encode
        branches but still makes h5py raise TypeError — the ideal probe for
        the fallback error path.
        """
        # Construct an object-dtype numpy array (h5py cannot serialise these)
        bad_val = np.array([{"a": 1}], dtype=object)
        return _build_2driver(
            extra_d0={"box_volume_m3": 0.02},  # driver_0 writes fine
            extra_d1={"_bad_attr": bad_val},  # driver_1 raises mid-write
        )

    def test_raises_on_bad_attr(self, tmp_path: Path) -> None:
        """Write with an unserializable attr must raise TypeError."""
        ds = self._bad_dataset()
        p = tmp_path / "bad.h5"
        with pytest.raises(TypeError):
            write_dataset(p, ds)

    def test_error_message_names_driver_and_key(self, tmp_path: Path) -> None:
        """The error message must name the offending driver id and attr key."""
        ds = self._bad_dataset()
        p = tmp_path / "bad_msg.h5"
        with pytest.raises(TypeError, match="driver_1") as exc_info:
            write_dataset(p, ds)
        # Must also name the offending key
        assert "_bad_attr" in str(
            exc_info.value
        ), f"Error should name the offending attr key; got: {exc_info.value}"

    def test_original_file_unchanged_after_failure(self, tmp_path: Path) -> None:
        """A pre-existing good file must be byte-identical after a failed write."""
        # Write a clean file first and capture its bytes
        good_ds = _build_2driver()
        p = tmp_path / "original.h5"
        write_dataset(p, good_ds)
        original_bytes = p.read_bytes()

        # Attempt an overwrite with a bad dataset — must raise
        bad_ds = self._bad_dataset()
        with pytest.raises(TypeError):
            write_dataset(p, bad_ds)

        # Original must be untouched
        assert p.read_bytes() == original_bytes, (
            "Pre-existing file must be byte-identical after a failed write "
            "(atomic-write guarantee)"
        )

    def test_no_leftover_temp_file(self, tmp_path: Path) -> None:
        """After a failed write no ``.tmp`` file must remain in the directory."""
        ds = self._bad_dataset()
        p = tmp_path / "no_tmp.h5"
        with pytest.raises(TypeError):
            write_dataset(p, ds)

        tmp_files = list(tmp_path.glob("*.tmp"))
        assert not tmp_files, f"Temp file(s) left on disk after failed write: {tmp_files}"


# ---------------------------------------------------------------------------
# 3.  Lossless 2-driver round-trip (all-valid attrs)
# ---------------------------------------------------------------------------


class TestRoundTrip:
    """All-valid-attrs 2-driver dataset round-trips exactly."""

    def _full_ds(self) -> object:
        """2-driver dataset with a rich attr set (no None values)."""
        return _build_2driver(
            extra_d0={
                "box_volume_m3": 0.020,
                "reference_voltage_V": 2.83,
                "terminal_response_model": "thiele_small+ladder_rl",
                "ts_params": {"fs": 55.0, "Qts": 0.38, "Re": 6.0},
            },
            extra_d1={
                "box_volume_m3": 0.012,
                "reference_voltage_V": 2.83,
                "terminal_response_model": "thiele_small+ladder_rl",
                "ts_params": {"fs": 65.0, "Qts": 0.42, "Re": 6.0},
            },
        )

    def test_driver_count_and_order(self, tmp_path: Path) -> None:
        """Driver count and insertion order must be preserved."""
        ds = self._full_ds()
        p = tmp_path / "full.h5"
        write_dataset(p, ds)
        ds2 = read_dataset(p)

        assert len(ds2.drivers) == 2
        assert [d.driver_id for d in ds2.drivers] == ["driver_0", "driver_1"]

    def test_H_full_roundtrip(self, tmp_path: Path) -> None:
        """H_full arrays must survive write → read exactly."""
        ds = self._full_ds()
        p = tmp_path / "hfull.h5"
        write_dataset(p, ds)
        ds2 = read_dataset(p)

        orig = {d.driver_id: d.H_full for d in ds.drivers}
        read = {d.driver_id: d.H_full for d in ds2.drivers}
        for did in orig:
            assert np.array_equal(read[did], orig[did]), f"H_full mismatch for {did}"

    def test_frequencies_roundtrip(self, tmp_path: Path) -> None:
        """Frequency grid must survive exactly."""
        ds = self._full_ds()
        p = tmp_path / "freq.h5"
        write_dataset(p, ds)
        ds2 = read_dataset(p)
        assert np.array_equal(ds2.frequencies, ds.frequencies)
        assert ds2.frequencies.dtype == np.float64

    def test_driver_order_attr_in_file(self, tmp_path: Path) -> None:
        """The raw ``driver_order`` attr in the HDF5 file must be a JSON list."""
        import h5py

        ds = self._full_ds()
        p = tmp_path / "order_attr.h5"
        write_dataset(p, ds)
        with h5py.File(p, "r") as f:
            raw = f.attrs["driver_order"]
        decoded = json.loads(raw)
        assert decoded == ["driver_0", "driver_1"]

    def test_scalar_driver_attr_roundtrip(self, tmp_path: Path) -> None:
        """Scalar driver attrs (float, str) must round-trip faithfully."""
        ds = self._full_ds()
        p = tmp_path / "scalar_attrs.h5"
        write_dataset(p, ds)
        ds2 = read_dataset(p)

        d0 = next(d for d in ds2.drivers if d.driver_id == "driver_0")
        assert float(d0.attrs["box_volume_m3"]) == pytest.approx(0.020)
        assert float(d0.attrs["reference_voltage_V"]) == pytest.approx(2.83)
        assert d0.attrs["terminal_response_model"] == "thiele_small+ladder_rl"

    def test_dict_driver_attr_roundtrip(self, tmp_path: Path) -> None:
        """Dict-valued driver attrs (ts_params) must round-trip as dicts."""
        ds = self._full_ds()
        p = tmp_path / "dict_attrs.h5"
        write_dataset(p, ds)
        ds2 = read_dataset(p)

        d0 = next(d for d in ds2.drivers if d.driver_id == "driver_0")
        ts = d0.attrs["ts_params"]
        assert isinstance(ts, dict)
        assert ts["fs"] == pytest.approx(55.0)
        assert ts["Qts"] == pytest.approx(0.38)
