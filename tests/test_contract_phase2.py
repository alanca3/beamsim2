"""Phase-2 contract-hardening tests (Stage P2-0a).

Two small guards the Phase-2 filter designer relies on:

* ``diaphragm_area`` is written to each driver's metadata (sensitivity normalization).
* ``read_dataset`` guards ``schema_version``: warns on a missing / minor-mismatched
  version and refuses an incompatible major version (contract risk R-09).
"""

from __future__ import annotations

import warnings

import h5py
import numpy as np
import pytest

from beamsim2.assembly.tensor import build_dataset
from beamsim2.core.sphere import lebedev
from beamsim2.core.types import ComplexField
from beamsim2.geometry.assemble import DriverSpec
from beamsim2.io.hdf5_store import read_dataset, write_dataset
from beamsim2.pipeline.run import DriverPlacement, _driver_attrs


def test_diaphragm_area_written_to_driver_attrs():
    """_driver_attrs includes diaphragm_area = pi * radius^2."""
    radius = 0.075
    dp = DriverPlacement(
        driver_id="woofer_0",
        spec=DriverSpec((0.0, 0.0, 0.1), (0.0, 0.0, 1.0), radius),
        terminal=None,
    )
    attrs = _driver_attrs(dp)
    assert "diaphragm_area" in attrs
    assert attrs["diaphragm_area"] == pytest.approx(np.pi * radius**2, rel=1e-12)


def _tiny_dataset():
    """A minimal 1-driver dataset built through build_dataset()."""
    obs = lebedev(n_points=14, radius=1.0)
    freqs = np.array([500.0, 1000.0])  # [F]
    n = obs.unit_vectors.shape[0]
    pressure = (np.ones((2, n)) + 0.5j * np.ones((2, n))).astype(np.complex128)  # [F,N]
    field = ComplexField(
        frequencies=freqs,
        pressure=pressure,
        convergence_flags=np.ones(2, dtype=bool),  # [F]
    )
    return build_dataset(
        driver_inputs=[("drv0", field, {"name": "drv0", "position": [0.0, 0.0, 0.0]})],
        directions=obs,
        root_attrs={"phase_origin": [0.0, 0.0, 0.0], "speed_of_sound": 343.2},
    )


def test_schema_guard_accepts_matching_version(tmp_path):
    """A freshly written file (matching schema_version) reads with no warning/error."""
    path = tmp_path / "ok.h5"
    write_dataset(path, _tiny_dataset())
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning -> failure
        ds = read_dataset(path)
    assert len(ds.drivers) == 1


def test_schema_guard_warns_on_missing_version(tmp_path):
    """A file with no schema_version reads but warns."""
    path = tmp_path / "noversion.h5"
    write_dataset(path, _tiny_dataset())
    with h5py.File(path, "a") as f:
        del f.attrs["schema_version"]
    with pytest.warns(UserWarning, match="no schema_version"):
        read_dataset(path)


def test_schema_guard_refuses_incompatible_major(tmp_path):
    """A file whose schema major version differs is refused."""
    path = tmp_path / "future.h5"
    write_dataset(path, _tiny_dataset())
    with h5py.File(path, "a") as f:
        f.attrs["schema_version"] = "2.0"
    with pytest.raises(ValueError, match="incompatible schema_version"):
        read_dataset(path)
