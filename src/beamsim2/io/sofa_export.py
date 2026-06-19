"""SOFA (AES69 / HDF5-based) exporter for BeamSimII directivity data.

Writes the assembled multi-driver radiation dataset to a SOFA file using the
``GeneralTF`` convention (sofar v1.2.3, AES69-2022).

Convention choice
-----------------
``FreeFieldDirectivityTF`` was evaluated but is designed for *rotating-speaker*
measurement setups (M = directions, R = 1 fixed microphone).  Its ``Data_Real``
dimension is ``MRN`` with ``ReceiverPosition: IC`` (one fixed receiver), making it
awkward to store multiple drivers in one file.  INFERRED (empirically verified with
sofar v1.2.3).

``GeneralTF`` is a better fit for our multi-driver simulation output:

- ``M`` = number of drivers (one measurement sequence per driver)
- ``R`` = number of directions (Lebedev observation points)
- ``N`` = number of frequencies
- ``Data_Real[m, r, n]`` = Re(H[driver=m, direction=r, freq=n])
- ``Data_Imag[m, r, n]`` = Im(H[driver=m, direction=r, freq=n])
- ``SourcePosition[m]``   = physical position of driver *m* (metres, cartesian)
- ``ReceiverPosition[r]`` = observation point = ``unit_vector[r] × r_obs`` (metres)

Phase rule (§3.4 cardinal rule)
---------------------------------
The complex data written to ``Data_Real`` / ``Data_Imag`` is taken *directly* from
the assembled BEM solve.  Each driver's phase is referenced to the **single global
spatial origin (0, 0, 0)** and carries its true time-of-flight path length.  It is
**never** minimum-phased or re-zeroed.  This is stated explicitly in
``GLOBAL_Comment`` so downstream readers know not to strip the phase.

References
----------
AES69-2022 (SOFA standard).  VERIFIED: sofar library, pyfar.org.
DATA_CONTRACT.md §3.4, §3.6.  BEAMSIMII_Gameplan.md §2 Stage G, §3.4, DR-06.
"""

from __future__ import annotations

import importlib.metadata
from pathlib import Path

import numpy as np
import sofar

from beamsim2.assembly.tensor import RadiationDataset

# ── internal helpers ──────────────────────────────────────────────────────────


def _driver_positions(ds: RadiationDataset, driver_ids: list[str]) -> np.ndarray:
    """Extract cartesian driver positions from attrs.

    Parameters
    ----------
    ds : RadiationDataset
    driver_ids : list of str
        IDs of the drivers to include (in order).

    Returns
    -------
    np.ndarray
        Shape ``[M, 3]`` float64.  Falls back to ``[0, 0, 0]`` if ``position``
        key absent.
    """
    drv_map = {d.driver_id: d for d in ds.drivers}
    positions = []
    for did in driver_ids:
        pos = drv_map[did].attrs.get("position", [0.0, 0.0, 0.0])
        positions.append(np.asarray(pos, dtype=np.float64))
    return np.array(positions, dtype=np.float64)  # [M, 3] float64


def _receiver_positions(ds: RadiationDataset) -> np.ndarray:
    """Observation-point positions: unit_vectors × observation_radius.

    Parameters
    ----------
    ds : RadiationDataset

    Returns
    -------
    np.ndarray
        Shape ``[R, 3]`` float64, metres.
    """
    r_obs = float(ds.directions.radius)
    return ds.directions.unit_vectors * r_obs  # [N, 3] float64 broadcast


# ── public API ────────────────────────────────────────────────────────────────


def write_sofa(
    path: str | Path,
    ds: RadiationDataset,
    *,
    field: str = "H_full",
    driver_ids: list[str] | None = None,
) -> None:
    """Write a RadiationDataset to a SOFA file (GeneralTF convention, AES69-2022).

    Dimension layout (SOFA ``GeneralTF``)::

        M = num_drivers   (one measurement sequence per driver)
        R = num_directions  (Lebedev observation-sphere points)
        N = num_frequencies

        Data_Real[m, r, n] = Re(H[driver=m, direction=r, freq=n])
        Data_Imag[m, r, n] = Im(H[driver=m, direction=r, freq=n])
        SourcePosition[m]  = cartesian position of driver m (metres)
        ReceiverPosition[r]= unit_vector[r] × r_obs (metres)

    The complex data carries the **global-origin phase** for each driver — never
    minimum-phased or re-zeroed (§3.4 cardinal rule, recorded in GLOBAL_Comment).

    Parameters
    ----------
    path : str or Path
        Output ``.sofa`` file path.  ``.sofa`` extension appended by sofar if
        absent.
    ds : RadiationDataset
        Assembled dataset from ``build_dataset()``.
    field : str
        Which per-driver array to export: ``"H_full"`` (default) or ``"H_bem"``.
        ``H_full`` is the measurement-equivalent response; ``H_bem`` is the raw
        geometric BEM response at unit cone velocity.
    driver_ids : list of str or None
        Subset of driver IDs to include; default includes all drivers in
        dataset order.

    Raises
    ------
    ValueError
        Unknown field name or requested driver_id not in dataset.
    """
    if field not in ("H_full", "H_bem"):
        raise ValueError(f"write_sofa: unknown field '{field}'; expected 'H_full' or 'H_bem'")

    drv_map = {d.driver_id: d for d in ds.drivers}
    if driver_ids is not None:
        missing = [did for did in driver_ids if did not in drv_map]
        if missing:
            raise ValueError(f"write_sofa: driver_id(s) not in dataset: {missing}")
        selected_ids = driver_ids
    else:
        selected_ids = [d.driver_id for d in ds.drivers]

    selected = [drv_map[did] for did in selected_ids]

    # Build [M, R, F] data arrays by transposing each driver's [F × R] H matrix.
    # H per driver has shape [F × N_dir]; SOFA wants [M, R, N] = [drivers, dirs, freqs].
    H_stack = np.stack(
        [getattr(d, field).T for d in selected],  # each .T is [R, F]
        axis=0,
    )  # [M, R, F] complex128

    try:
        bsim_version = importlib.metadata.version("beamsim2")
    except importlib.metadata.PackageNotFoundError:
        bsim_version = "unknown"

    phase_origin = list(ds.attrs.get("phase_origin", [0.0, 0.0, 0.0]))
    pressure_convention = str(ds.attrs.get("pressure_convention", "unknown"))
    field_desc = (
        "H_full (H_bem x terminal_response, measurement-equivalent)"
        if field == "H_full"
        else "H_bem (raw BEM at unit cone velocity, geometry only)"
    )

    comment = (
        f"BeamSimII v{bsim_version} — multi-driver radiation dataset. "
        f"field={field_desc}. "
        f"pressure_convention={pressure_convention}. "
        f"Phase origin={phase_origin} (global spatial origin). "
        "Complex data NOT re-zeroed per driver — global-origin phase preserved "
        "(§3.4 cardinal rule, DATA_CONTRACT.md). "
        "SOFA GeneralTF: M=drivers, R=Lebedev_directions, N=frequencies. "
        "INFERRED: GeneralTF chosen over FreeFieldDirectivityTF because the latter "
        "is designed for rotating-speaker setups (M=directions, R=1 mic) and cannot "
        "naturally accommodate multiple drivers in one file."
    )

    sofa_obj = sofar.Sofa("GeneralTF")

    # ── frequencies [N] ───────────────────────────────────────────────────────
    sofa_obj.N = ds.frequencies.astype(np.float64)  # [F] float64, Hz

    # ── source positions = driver positions [M, 3] ────────────────────────────
    sofa_obj.SourcePosition = _driver_positions(ds, selected_ids)  # [M, 3] float64
    sofa_obj.SourcePosition_Type = "cartesian"
    sofa_obj.SourcePosition_Units = "metre"

    # ── receiver positions = observation sphere points [R, 3] ─────────────────
    sofa_obj.ReceiverPosition = _receiver_positions(ds)  # [R, 3] float64
    sofa_obj.ReceiverPosition_Type = "cartesian"
    sofa_obj.ReceiverPosition_Units = "metre"

    # ── complex TF data [M, R, N] ─────────────────────────────────────────────
    sofa_obj.Data_Real = H_stack.real.astype(np.float64)  # [M, R, F] float64
    sofa_obj.Data_Imag = H_stack.imag.astype(np.float64)  # [M, R, F] float64

    # ── global metadata ───────────────────────────────────────────────────────
    sofa_obj.GLOBAL_Title = f"BeamSimII directivity — {field}"
    sofa_obj.GLOBAL_Comment = comment

    sofar.write_sofa(str(path), sofa_obj)
