"""Native HDF5 read/write for the H-tensor data contract.

Writes (and reads back) the Phase-1 output in the exact layout defined in
DATA_CONTRACT.md §3.6:

    /frequencies            [F] float64
    /directions/
        unit_vectors        [N×3] float64
        weights             [N]   float64
        theta_phi           [N×2] float64
        attrs: scheme, order, weight_convention
    /drivers/<id>/
        H_bem               [F×N] complex128
        terminal_response   [F]   complex128
        H_full              [F×N] complex128
        convergence_flags   [F]   bool
        attrs: name, position, orientation, radius, profile, ts_params, …
    root attrs: schema_version, beamsim_version, created_utc, …

Complex128 is stored as two-field compound (real/imag float64 each) so it
roundtrips exactly — ``np.array_equal`` on every element.  Dict-valued attrs
(ts_params, profile params) are JSON-encoded as strings so h5py never has to
serialise arbitrary Python objects.

schema_version = "1.0" — bumped only when the on-disk layout changes (§11.2,
CLAUDE.md).  Separate from the application version.

References
----------
DATA_CONTRACT.md §3.5, §3.6.  BEAMSIMII_Gameplan.md §3 (phase-origin rule §3.4).
"""

from __future__ import annotations

import importlib.metadata
import json
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

from beamsim2.assembly.tensor import DriverData, RadiationDataset
from beamsim2.core.types import ObservationPoints

SCHEMA_VERSION = "1.0"


def _write_attrs(h5obj: h5py.HLObject, attrs: dict) -> None:
    """Write a metadata dict onto an HDF5 object's attribute space.

    Dict-valued or list-valued entries are JSON-encoded so h5py does not have
    to serialise arbitrary Python objects.  Scalars and strings pass through.

    Parameters
    ----------
    h5obj : h5py group or dataset
    attrs : dict
    """
    for key, val in attrs.items():
        if isinstance(val, (dict, list)):
            h5obj.attrs[key] = json.dumps(val)
        elif isinstance(val, np.ndarray):
            h5obj.attrs[key] = val
        else:
            h5obj.attrs[key] = val


def _read_attrs(h5obj: h5py.HLObject) -> dict:
    """Read HDF5 attributes back to a Python dict, un-JSON-encoding where needed.

    Parameters
    ----------
    h5obj : h5py group or dataset

    Returns
    -------
    dict  — same structure as what was passed to ``_write_attrs``
    """
    result: dict = {}
    for key, val in h5obj.attrs.items():
        if isinstance(val, str):
            try:
                decoded = json.loads(val)
                if isinstance(decoded, (dict, list)):
                    val = decoded
            except (json.JSONDecodeError, ValueError):
                pass
        result[key] = val
    return result


def write_dataset(path: str | Path, ds: RadiationDataset) -> None:
    """Write a RadiationDataset to the on-disk HDF5 contract format.

    Parameters
    ----------
    path : str or Path
        Output file path.  Extension should be ``.h5`` or ``.bsim``; both are
        HDF5 internally.
    ds : RadiationDataset
        Assembled dataset from ``build_dataset()``.

    Notes
    -----
    - complex128 stored as compound (real, imag) float64 for exact roundtrip.
    - All §3.5 root attrs are written; dict/list attrs are JSON-encoded strings.
    - ``beamsim_version`` read from the installed package metadata; falls back
      to ``"unknown"`` in editable installs before first build.
    """
    try:
        bsim_version = importlib.metadata.version("beamsim2")
    except importlib.metadata.PackageNotFoundError:
        bsim_version = "unknown"

    path = Path(path)
    with h5py.File(path, "w") as f:
        # ── root datasets ──────────────────────────────────────────────────
        f.create_dataset("frequencies", data=ds.frequencies)  # [F] float64

        # ── root attrs (§3.5) ─────────────────────────────────────────────
        root_meta = {
            "schema_version": SCHEMA_VERSION,
            "beamsim_version": bsim_version,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "spacing": ds.spacing,
        }
        if ds.fractional_octave is not None:
            root_meta["fractional_octave"] = ds.fractional_octave
        root_meta.update(ds.attrs)
        _write_attrs(f, root_meta)

        # ── /directions group ─────────────────────────────────────────────
        dg = f.create_group("directions")
        obs = ds.directions
        dg.create_dataset("unit_vectors", data=obs.unit_vectors)  # [N×3] float64
        dg.create_dataset("weights", data=obs.weights)  # [N] float64
        if obs.theta_phi is not None:
            dg.create_dataset("theta_phi", data=obs.theta_phi)  # [N×2] float64
        _write_attrs(
            dg,
            {
                "scheme": obs.scheme,
                "order": obs.order,
                "weight_convention": obs.weight_convention,
                "radius": obs.radius,
            },
        )

        # ── /drivers group ────────────────────────────────────────────────
        drv_grp = f.create_group("drivers")
        for d in ds.drivers:
            dg2 = drv_grp.create_group(d.driver_id)
            # h5py stores/reads complex128 natively — exact lossless roundtrip
            dg2.create_dataset("H_bem", data=d.H_bem.astype(np.complex128))  # [F×N]
            dg2.create_dataset(
                "terminal_response", data=d.terminal_response.astype(np.complex128)
            )  # [F]
            dg2.create_dataset("H_full", data=d.H_full.astype(np.complex128))  # [F×N]
            dg2.create_dataset("convergence_flags", data=d.convergence_flags)  # [F] bool
            _write_attrs(dg2, d.attrs)


def read_dataset(path: str | Path) -> RadiationDataset:
    """Read a RadiationDataset from the on-disk HDF5 contract format.

    Parameters
    ----------
    path : str or Path
        Path to the ``.h5`` / ``.bsim`` file written by ``write_dataset()``.

    Returns
    -------
    RadiationDataset
        Fully reconstructed dataset with complex128 arrays and decoded attrs.

    Raises
    ------
    KeyError
        If a mandatory dataset or group is missing (file does not conform to
        schema_version "1.0").
    """
    path = Path(path)
    with h5py.File(path, "r") as f:
        frequencies = f["frequencies"][:]  # [F] float64
        root_attrs = _read_attrs(f)

        # ── /directions ───────────────────────────────────────────────────
        dg = f["directions"]
        unit_vectors = dg["unit_vectors"][:]  # [N×3] float64
        weights = dg["weights"][:]  # [N] float64
        theta_phi = dg["theta_phi"][:] if "theta_phi" in dg else None  # [N×2] float64
        dir_attrs = _read_attrs(dg)
        directions = ObservationPoints(
            unit_vectors=unit_vectors,
            radius=float(dir_attrs.get("radius", 1.0)),
            weights=weights,
            scheme=str(dir_attrs.get("scheme", "lebedev")),
            order=int(dir_attrs.get("order", 0)),
            weight_convention=str(dir_attrs.get("weight_convention", "sum_4pi")),
            theta_phi=theta_phi,
        )

        # ── /drivers ──────────────────────────────────────────────────────
        # Sort by driver_id for deterministic ordering (HDF5 iterates keys
        # in alphabetical order; sort makes read order match a sort of the
        # original insertion order).
        drivers: list[DriverData] = []
        drv_grp = f["drivers"]
        for driver_id in sorted(drv_grp.keys()):
            dg2 = drv_grp[driver_id]
            H_bem = dg2["H_bem"][:].astype(np.complex128)  # [F×N] complex128
            terminal_response = dg2["terminal_response"][:].astype(np.complex128)  # [F] complex128
            H_full = dg2["H_full"][:].astype(np.complex128)  # [F×N] complex128
            convergence_flags = dg2["convergence_flags"][:].astype(bool)  # [F] bool
            driver_attrs = _read_attrs(dg2)
            drivers.append(
                DriverData(
                    driver_id=str(driver_id),
                    H_bem=H_bem,
                    terminal_response=terminal_response,
                    H_full=H_full,
                    convergence_flags=convergence_flags,
                    attrs=driver_attrs,
                )
            )

    # spacing attrs may be stored in root_attrs
    spacing = str(root_attrs.get("spacing", "log"))
    fo = root_attrs.get("fractional_octave", None)
    if fo is not None:
        fo = float(fo)

    # interpolated_mask not stored on disk yet (all False); reconstruct
    interpolated_mask = np.zeros(len(frequencies), dtype=bool)

    return RadiationDataset(
        frequencies=frequencies,
        spacing=spacing,
        fractional_octave=fo,
        interpolated_mask=interpolated_mask,
        directions=directions,
        drivers=drivers,
        attrs=root_attrs,
    )
