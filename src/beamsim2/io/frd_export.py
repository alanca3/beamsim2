"""VituixCAD .frd text-format exporter: per-driver, per-angle frequency/magnitude/phase files.

Writes one ``.frd`` file per (driver, field, direction) so VituixCAD or any FRD-reading
tool can import individual directional frequency responses.  Two field types are supported:

- ``H_full``  — measurement-equivalent response (H_bem × terminal_response); use this
  for crossover/directivity work in VituixCAD.
- ``H_bem``   — raw BEM response at unit cone velocity (geometry + diffraction only,
  no electrical terminal chain); use this to audit the acoustic contribution separately.

Phase is written as computed from the complex data — referenced to the global spatial
origin (0, 0, 0).  It is **never** minimum-phased or re-zeroed per driver.  Inter-driver
phase differences encode true time-of-flight and are the steering information for
Phase-2 beamforming.  VERIFIED: §3.4 cardinal rule.

The Lebedev sphere grid does not map to VituixCAD's named horizontal/vertical angles,
so a ``manifest.csv`` is written alongside the .frd files.  Each manifest row maps
file path → driver_id, field, direction index, unit-vector (x, y, z), and θ/φ in degrees.

References
----------
DATA_CONTRACT.md §3.4 (phase origin), §3.6 (export formats).
BEAMSIMII_Gameplan.md §2 Stage G, §3.4.
VituixCAD .frd format: text file, optional ``*`` comment lines, then
``frequency_Hz  magnitude_dB  phase_deg`` rows.  VERIFIED (VituixCAD documentation).
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

from beamsim2.assembly.tensor import RadiationDataset

# ── constants ─────────────────────────────────────────────────────────────────
_P_REF_PA: float = 20e-6  # dB SPL reference — 20 µPa (VERIFIED: IEC 61672-1)
_MAG_FLOOR: float = 1e-300  # prevents log10(0); well below any physical pressure
_TWO_PI: float = 2.0 * math.pi


# ── helpers ───────────────────────────────────────────────────────────────────


def _ensure_theta_phi(ds: RadiationDataset) -> np.ndarray:
    """Return theta_phi [N × 2] float64 in radians; compute from unit_vectors if absent.

    Parameters
    ----------
    ds : RadiationDataset
        The assembled dataset.

    Returns
    -------
    np.ndarray
        Shape ``[N, 2]`` float64.  Column 0 = θ (colatitude from +z, [0, π]).
        Column 1 = φ (azimuth from +x toward +y, [0, 2π)).
    """
    if ds.directions.theta_phi is not None:
        return ds.directions.theta_phi  # [N, 2] float64
    uv = ds.directions.unit_vectors  # [N, 3] float64
    theta = np.arccos(np.clip(uv[:, 2], -1.0, 1.0))  # [N] colatitude in [0, π]
    phi = np.arctan2(uv[:, 1], uv[:, 0]) % _TWO_PI  # [N] azimuth in [0, 2π)
    return np.column_stack([theta, phi])  # [N, 2] float64


def _header_lines(
    driver_id: str,
    field: str,
    n: int,
    uv: np.ndarray,
    theta_deg: float,
    phi_deg: float,
    pressure_convention: str,
    p_ref: float,
) -> list[str]:
    """Build the ``*``-prefixed comment block for one .frd file.

    Parameters
    ----------
    driver_id : str
        Driver identifier string.
    field : str
        ``"H_full"`` or ``"H_bem"``.
    n : int
        Direction index (0-based, Lebedev ordering).
    uv : np.ndarray
        Shape ``[3]`` float64 — unit-vector Cartesian direction cosines.
    theta_deg, phi_deg : float
        Spherical coordinates in degrees (colatitude θ, azimuth φ).
    pressure_convention : str
        Pulled from ``ds.attrs["pressure_convention"]``; recorded verbatim.
    p_ref : float
        Magnitude reference in Pa (default 20 µPa = dB SPL reference).

    Returns
    -------
    list[str]
        Comment lines, each starting with ``*``.
    """
    field_desc = (
        "H_full: measurement-equivalent response (H_bem × terminal_response)"
        if field == "H_full"
        else "H_bem: raw BEM response at unit cone velocity (geometry only)"
    )
    return [
        "* BeamSimII FRD export",
        f"* driver_id: {driver_id}",
        f"* field: {field_desc}",
        f"* direction_index: {n:04d}",
        f"* unit_vector: x={uv[0]:.6f} y={uv[1]:.6f} z={uv[2]:.6f}",
        f"* theta_deg: {theta_deg:.3f}  phi_deg: {phi_deg:.3f}",
        "*   theta = colatitude from +z in [0, 180]; phi = azimuth from +x in [0, 360)",
        f"* pressure_convention: {pressure_convention}",
        f"* magnitude_reference: dB re {p_ref:.2e} Pa"
        + ("  [dB SPL re 20 µPa]" if abs(p_ref - 20e-6) < 1e-12 else ""),
        "* HEURISTIC: dB SPL re 20 µPa is the standard acoustics amplitude reference.",
        "* Phase referenced to global spatial origin (0,0,0); NOT re-zeroed per driver.",
        "* Inter-driver phase differences encode true time-of-flight (§3.4 cardinal rule).",
        "* frequency_Hz  magnitude_dB  phase_deg",
    ]


# ── public API ────────────────────────────────────────────────────────────────


def write_frd(
    out_dir: str | Path,
    ds: RadiationDataset,
    *,
    fields: tuple[str, ...] = ("H_full", "H_bem"),
    p_ref: float = _P_REF_PA,
    driver_ids: list[str] | None = None,
) -> Path:
    """Write per-driver, per-direction ``.frd`` files and a manifest CSV.

    One file is written for every combination of (driver, field, direction).
    Directory layout::

        <out_dir>/<driver_id>/<field>/<driver_id>_<field>_dir{n:04d}.frd

    A ``manifest.csv`` is written at ``<out_dir>/manifest.csv`` mapping every
    file to its direction metadata (index, unit-vector, θ, φ in degrees).

    Parameters
    ----------
    out_dir : str or Path
        Output root directory (created if absent).
    ds : RadiationDataset
        Assembled dataset from ``build_dataset()``.
    fields : tuple of str
        Which per-driver arrays to export.  Valid values: ``"H_full"``,
        ``"H_bem"``.  Default exports both.
    p_ref : float
        Magnitude reference in Pa used for dB conversion.  Default is
        20 µPa (dB SPL).  HEURISTIC: standard acoustics amplitude reference.
    driver_ids : list of str or None
        Subset of driver IDs to export; default exports all drivers.

    Returns
    -------
    Path
        The ``out_dir`` as a resolved ``Path``.

    Notes
    -----
    - Phase is ``np.angle(p)`` in degrees — exactly the complex-valued phase
      from the BEM solve, referenced to the global origin.  Not re-zeroed.
    - ``H_bem`` magnitudes are "Pa at r_obs for unit cone velocity" (see
      ``ds.attrs["pressure_convention"]``).  Absolute SPL depends on the
      driver sensitivity and the applied terminal-response drive level.
    - ``H_full = H_bem × terminal_response``.  For a 2.83 V drive level,
      ``terminal_response`` encodes the sensitivity scaling.

    Raises
    ------
    ValueError
        Unknown field name, or requested ``driver_id`` not found in dataset.
    """
    for f in fields:
        if f not in ("H_full", "H_bem"):
            raise ValueError(f"write_frd: unknown field '{f}'; expected 'H_full' or 'H_bem'")

    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # resolve driver subset
    all_ids = {d.driver_id: d for d in ds.drivers}
    if driver_ids is not None:
        missing = [did for did in driver_ids if did not in all_ids]
        if missing:
            raise ValueError(f"write_frd: driver_id(s) not in dataset: {missing}")
        selected = [all_ids[did] for did in driver_ids]
    else:
        selected = list(ds.drivers)

    # precompute direction data
    uv = ds.directions.unit_vectors  # [N, 3] float64
    theta_phi = _ensure_theta_phi(ds)  # [N, 2] float64, radians
    theta_deg = np.rad2deg(theta_phi[:, 0])  # [N] colatitude in [0, 180]
    phi_deg = np.rad2deg(theta_phi[:, 1])  # [N] azimuth in [0, 360)
    freq = ds.frequencies  # [F] float64, Hz
    F = len(freq)
    N = len(uv)

    pressure_convention = str(ds.attrs.get("pressure_convention", "unknown"))

    # manifest rows: one per file
    manifest_rows: list[dict] = []

    for drv in selected:
        for field in fields:
            # retrieve H array: [F × N] complex128
            H: np.ndarray = getattr(drv, field)  # [F × N] complex128

            field_dir = out_dir / drv.driver_id / field
            field_dir.mkdir(parents=True, exist_ok=True)

            for n in range(N):
                p_slice = H[:, n]  # [F] complex128 — pressure at direction n

                # magnitude in dB re p_ref; floor guards log10(0)
                mag_db = 20.0 * np.log10(np.maximum(np.abs(p_slice), _MAG_FLOOR) / p_ref)
                # [F] float64

                # phase in degrees, (-180, 180] — global-origin-referenced, never re-zeroed
                phase_d = np.rad2deg(np.angle(p_slice))  # [F] float64

                fname = f"{drv.driver_id}_{field}_dir{n:04d}.frd"
                fpath = field_dir / fname

                header = _header_lines(
                    driver_id=drv.driver_id,
                    field=field,
                    n=n,
                    uv=uv[n],
                    theta_deg=float(theta_deg[n]),
                    phi_deg=float(phi_deg[n]),
                    pressure_convention=pressure_convention,
                    p_ref=p_ref,
                )

                lines = header.copy()
                for f_idx in range(F):
                    lines.append(f"{freq[f_idx]:.4f}  {mag_db[f_idx]:.6f}  {phase_d[f_idx]:.4f}")

                fpath.write_text("\n".join(lines) + "\n", encoding="utf-8")

                manifest_rows.append(
                    {
                        "file": str(fpath.relative_to(out_dir)),
                        "driver_id": drv.driver_id,
                        "field": field,
                        "direction_index": n,
                        "ux": f"{uv[n, 0]:.8f}",
                        "uy": f"{uv[n, 1]:.8f}",
                        "uz": f"{uv[n, 2]:.8f}",
                        "theta_deg": f"{float(theta_deg[n]):.4f}",
                        "phi_deg": f"{float(phi_deg[n]):.4f}",
                    }
                )

    # write manifest.csv
    manifest_path = out_dir / "manifest.csv"
    fieldnames = [
        "file",
        "driver_id",
        "field",
        "direction_index",
        "ux",
        "uy",
        "uz",
        "theta_deg",
        "phi_deg",
    ]
    with manifest_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    n_files = len(manifest_rows)
    n_drivers = len(selected)
    print(
        f"write_frd: {n_files} files written"
        f" ({n_drivers} driver(s) × {len(fields)} field(s) × {N} direction(s))"
        f"  →  {out_dir}"
    )

    return out_dir
