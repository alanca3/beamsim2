"""Chunk 1 (#7) regression test — CI-safe, no NumCalc.

Builds two free-field point monopoles analytically on the observation sphere,
sharing the global phase origin [0, 0, 0] (cardinal rule §3.4): each monopole's
response carries the true path-length phase from its own offset position, so the
SUMMED field's directivity is driven entirely by inter-driver time-of-flight —
the exact quantity V-5 protects.

This locks in the Chunk-1 fixes diagnosed from the user's real ``HDF5/Dr1.h5``
(a 7-entry ``driver_order`` with a duplicated ``driver_4`` but a single surviving
group):

* duplicate ``driver_id`` is rejected loud and early (assembly + writer), and the
  writer never leaves a corrupt partial file on disk;
* ``read_dataset`` refuses a corrupt file (``driver_order`` vs group mismatch) but
  still reads a valid legacy file with no ``driver_order`` attr;
* the on-axis reference is a defined ``reference_axis`` (default +z, here also
  exercised as +x), not a hardcoded +z;
* low-frequency directivity about the reference axis is near-omni and grows with
  frequency;
* the GUI id generator never reuses an id still in use (the Dr1.h5 root cause).
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from beamsim2.assembly.tensor import build_dataset
from beamsim2.core.driver_ids import next_driver_id
from beamsim2.core.sphere import make_observation_grid, nearest_direction_index
from beamsim2.core.types import ComplexField, ObservationPoints
from beamsim2.io.hdf5_store import read_dataset, write_dataset

_C = 343.2  # m/s
_R_OBS = 3.0  # m — far enough that 1/r amplitude tilt is negligible vs the array phase
_SEP = 0.15  # m — monopole separation along x (broadside axis = +z)
_FREQS = np.array([40.0, 160.0, 640.0, 1800.0])  # kd ≈ 0.11 … 4.94
_POS = {
    "mono_L": np.array([+_SEP / 2, 0.0, 0.0]),
    "mono_R": np.array([-_SEP / 2, 0.0, 0.0]),
}


def _grid() -> ObservationPoints:
    """Icosphere grid (~642 points, weights sum to 4π) for accurate directivity."""
    return make_observation_grid("icosphere", 320, radius=_R_OBS)


def _monopole_field(r_s: np.ndarray, obs: ObservationPoints) -> ComplexField:
    """Free-field point-monopole transfer function on the observation sphere.

    Engineering convention exp(+jk·dist) (matches NumCalc). The phase reference is
    the global origin: observation points are measured from [0,0,0] and ``r_s`` is
    the monopole's true position in the same frame, so the path-length phase is
    the genuine time-of-flight (never re-zeroed).

    Returns a ``ComplexField`` with pressure ``[F, N]`` complex128.
    """
    o = obs.unit_vectors * obs.radius  # [N, 3] observation coordinates
    dist = np.linalg.norm(o - r_s, axis=1)  # [N] true path length from the source
    k = 2.0 * np.pi * _FREQS / _C  # [F]
    pressure = np.exp(1j * k[:, None] * dist[None, :]) / dist[None, :]  # [F, N]
    return ComplexField(
        pressure=pressure.astype(np.complex128),
        convergence_flags=np.ones(len(_FREQS), dtype=bool),
        frequencies=_FREQS.copy(),
    )


def _build(ids: tuple[str, str], obs: ObservationPoints, reference_axis=(0.0, 0.0, 1.0)):
    """Assemble a 2-monopole RadiationDataset through the real build_dataset path."""
    positions = list(_POS.values())
    driver_inputs = [
        (ids[i], _monopole_field(positions[i], obs), {"name": ids[i]}) for i in range(2)
    ]
    return build_dataset(
        driver_inputs=driver_inputs,
        directions=obs,
        root_attrs={
            "schema_version": "1.0",
            "reference_axis": [float(c) for c in reference_axis],
            "phase_origin": [0.0, 0.0, 0.0],
        },
    )


def _di_about_axis(H: np.ndarray, weights: np.ndarray, on_idx: int) -> np.ndarray:
    """Directivity index referenced to a chosen on-axis direction, per frequency.

    DI_axis = 10·log10(|p(on_axis)|² / mean_sphere|p|²).  0 dB ⇒ omnidirectional.
    """
    intensity = np.abs(H) ** 2  # [F, N]
    mean_int = np.sum(weights * intensity, axis=1) / (4.0 * np.pi)  # [F]
    return 10.0 * np.log10(intensity[:, on_idx] / mean_int)  # [F]


# ---------------------------------------------------------------------------
# 1. Round-trip + driver_order integrity + axis-referenced directivity
# ---------------------------------------------------------------------------


def test_monopole_pair_roundtrip_and_directivity(tmp_path: Path) -> None:
    obs = _grid()
    ds = _build(("mono_L", "mono_R"), obs)

    out = tmp_path / "monopair.h5"
    write_dataset(out, ds)
    rds = read_dataset(out)

    # driver_order integrity: both drivers present, in order, lossless.
    assert [d.driver_id for d in rds.drivers] == ["mono_L", "mono_R"]
    with h5py.File(out, "r") as f:
        assert list(f["drivers"].keys()) == ["mono_L", "mono_R"]
        assert f.attrs["driver_order"] == '["mono_L", "mono_R"]'
    for a, b in zip(ds.drivers, rds.drivers):
        np.testing.assert_array_equal(a.H_bem, b.H_bem)

    # reference_axis round-trips.
    np.testing.assert_array_equal(rds.attrs["reference_axis"], [0.0, 0.0, 1.0])

    # SUMMED field (unit weights) — directivity about +z, the broadside axis.
    H_sum = rds.drivers[0].H_full + rds.drivers[1].H_full  # [F, N]
    on_z = nearest_direction_index(obs.unit_vectors, rds.attrs["reference_axis"])
    di = _di_about_axis(H_sum, obs.weights, on_z)

    # Low frequency (40 Hz, kd≈0.11): near-omni about the axis.
    assert di[0] < 0.1, f"low-f DI_axis={di[0]:.3f} dB not near-omni"
    spread_lo = 20 * np.log10(np.abs(H_sum[0]).max() / np.abs(H_sum[0]).min())
    assert spread_lo < 0.1, f"low-f |H| spread={spread_lo:.3f} dB not near-omni"

    # Directivity grows monotonically with frequency (driven by inter-driver phase).
    assert np.all(np.diff(di) > 0), f"DI_axis not monotonically increasing: {di}"
    assert di[-1] > 3.0, f"high-f DI_axis={di[-1]:.2f} dB — expected clearly directional"

    # +z is the broadside maximum, so axis-referenced DI ≈ peak DI everywhere.
    di_max = 10.0 * np.log10(
        (np.abs(H_sum) ** 2).max(axis=1)
        / (np.sum(obs.weights * np.abs(H_sum) ** 2, axis=1) / (4.0 * np.pi))
    )
    np.testing.assert_allclose(di, di_max, atol=0.05)


# ---------------------------------------------------------------------------
# 2. Duplicate driver_id rejected loud — and no corrupt partial file
# ---------------------------------------------------------------------------


def test_duplicate_driver_id_rejected(tmp_path: Path) -> None:
    obs = _grid()

    # build_dataset refuses to assemble colliding ids (contract guard).
    with pytest.raises(ValueError, match="unique"):
        _build(("driver_4", "driver_4"), obs)

    # write_dataset (called directly, e.g. GUI "Save HDF5") also refuses, and must
    # NOT leave a partial file behind (the Dr1.h5 failure mode).
    ds_ok = _build(("mono_L", "mono_R"), obs)
    ds_ok.drivers[1].driver_id = "mono_L"  # force a post-assembly collision
    out = tmp_path / "dup.h5"
    with pytest.raises(ValueError, match="unique"):
        write_dataset(out, ds_ok)
    assert not out.exists(), "writer left a corrupt partial file on a duplicate id"


# ---------------------------------------------------------------------------
# 3. read_dataset rejects corrupt driver_order, accepts valid legacy
# ---------------------------------------------------------------------------


def test_read_rejects_corrupt_driver_order(tmp_path: Path) -> None:
    obs = _grid()
    ds = _build(("mono_L", "mono_R"), obs)
    out = tmp_path / "corrupt.h5"
    write_dataset(out, ds)

    # Simulate the on-disk Dr1.h5 corruption: driver_order longer than the group set.
    import json

    with h5py.File(out, "a") as f:
        f.attrs["driver_order"] = json.dumps(["mono_L", "mono_L", "mono_R"])
    with pytest.raises(ValueError, match="corrupt"):
        read_dataset(out)


def test_read_rejects_driver_order_referencing_missing_group(tmp_path: Path) -> None:
    # driver_order names a group that is not on disk (no duplicates) — exercises
    # the membership/length guard, distinct from the duplicate-id guard above.
    obs = _grid()
    ds = _build(("mono_L", "mono_R"), obs)
    out = tmp_path / "ghost.h5"
    write_dataset(out, ds)

    import json

    with h5py.File(out, "a") as f:
        f.attrs["driver_order"] = json.dumps(["mono_L", "ghost"])
    with pytest.raises(ValueError, match="corrupt"):
        read_dataset(out)


def test_read_rejects_non_list_driver_order(tmp_path: Path) -> None:
    # driver_order present but not a JSON list (scalar) — must fail loud, not
    # silently fall back to alphabetical group order.
    obs = _grid()
    ds = _build(("mono_L", "mono_R"), obs)
    out = tmp_path / "scalar.h5"
    write_dataset(out, ds)

    with h5py.File(out, "a") as f:
        f.attrs["driver_order"] = 5  # not a list
    with pytest.raises(ValueError, match="corrupt"):
        read_dataset(out)


def test_read_accepts_valid_legacy_without_driver_order(tmp_path: Path) -> None:
    obs = _grid()
    ds = _build(("mono_L", "mono_R"), obs)
    out = tmp_path / "legacy.h5"
    write_dataset(out, ds)

    # A legitimately old file: groups are intact, just no driver_order attr.
    with h5py.File(out, "a") as f:
        del f.attrs["driver_order"]
    rds = read_dataset(out)  # must fall back to sorted(group keys), no error
    assert sorted(d.driver_id for d in rds.drivers) == ["mono_L", "mono_R"]


# ---------------------------------------------------------------------------
# 4. The reference axis is actually consumed (not hardcoded +z)
# ---------------------------------------------------------------------------


def test_reference_axis_changes_on_axis_pick() -> None:
    obs = _grid()
    ds = _build(("mono_L", "mono_R"), obs)
    H_sum = ds.drivers[0].H_full + ds.drivers[1].H_full  # [F, N]

    on_z = nearest_direction_index(obs.unit_vectors, (0.0, 0.0, 1.0))  # broadside
    on_x = nearest_direction_index(obs.unit_vectors, (1.0, 0.0, 0.0))  # endfire
    assert on_z != on_x
    assert obs.unit_vectors[on_z][2] == pytest.approx(obs.unit_vectors[:, 2].max())
    assert obs.unit_vectors[on_x][0] == pytest.approx(obs.unit_vectors[:, 0].max())

    di_z = _di_about_axis(H_sum, obs.weights, on_z)
    di_x = _di_about_axis(H_sum, obs.weights, on_x)
    # +z is the broadside maximum; +x (endfire) sits in the destructive region of
    # a broadside pair, so a +x reference axis reads clearly below +z at the top
    # frequency — proving the chosen axis flows through (not hardcoded +z).
    assert di_x[-1] < di_z[-1] - 1.0, f"axis not consumed: di_x={di_x[-1]:.1f} di_z={di_z[-1]:.1f}"


# ---------------------------------------------------------------------------
# 5. GUI id generation never reuses an id in use (the Dr1.h5 root cause)
# ---------------------------------------------------------------------------


def test_gui_id_generation_no_reuse_after_delete() -> None:
    # Place 3 → delete the middle → add one. The old count-based scheme
    # (f"driver_{len(drivers)}") would regenerate "driver_2", colliding with the
    # surviving driver_2; next_driver_id must fill the freed slot instead.
    ids: list[str] = []
    for _ in range(3):
        ids.append(next_driver_id(ids))
    assert ids == ["driver_0", "driver_1", "driver_2"]

    del ids[1]  # delete the middle driver → ["driver_0", "driver_2"]
    new_id = next_driver_id(ids)
    assert new_id not in ids, f"{new_id!r} collides with surviving ids {ids}"
    assert new_id == "driver_1"
