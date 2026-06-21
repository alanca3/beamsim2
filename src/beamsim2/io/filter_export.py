"""Phase-2 audit export: filtered per-driver / combined .frd + raw weights (Stage P2-3).

The Phase-2 designer's v1 export is **audit-first** (DR-P2-03): rather than deployable
DSP coefficients, it writes artifacts the user audits in their own tools (VituixCAD / REW):

* **filtered per-driver ``.frd``** — each driver's directional response with its design weight
  ``w_m(f)`` baked in (``w_m(f) * H_full[m, f, :]``), resampled (spherical-harmonic, see
  :mod:`beamsim2.core.sh_transform`) onto matched horizontal/vertical polar arcs through the
  steering axis — the arcs VituixCAD/REW expect (the scattered Lebedev/icosphere grid does not
  lie on named H/V angles);
* **combined steered ``.frd``** — the achieved system response ``P = sum_m w_m H_m`` on the
  same arcs;
* **raw weights** — ``w[M, F]`` complex (``.npz``), re-loadable to reconstruct the beam.

Phase is written as computed, referenced to the global origin (the cardinal rule); the weights
are baked in, so these files are an **audit artifact** (the user cannot re-tune the filters in
VituixCAD from them). Deployable FIR/biquad coefficient export is Stage P2-5.

References
----------
DATA_CONTRACT.md §3.4 (phase origin); docs/Phase 2 - Filter Solver.md §3.4 (export schema).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from beamsim2.assembly.tensor import RadiationDataset, stacked_h_full
from beamsim2.core.sh_transform import forward_sh, inverse_sh, safe_order_for_grid

_P_REF_PA: float = 20e-6  # dB SPL reference (20 µPa)
_MAG_FLOOR: float = 1e-300


def polar_arcs(
    steer_dir: np.ndarray, step_deg: float = 10.0
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Two orthogonal great-circle arcs (H, V) through the steering axis.

    Both arcs pass through ``steer_dir`` at angle 0 (the on-axis point). The horizontal arc
    sweeps the plane spanned by the steer axis and an in-plane reference; the vertical arc the
    orthogonal plane.

    Parameters
    ----------
    steer_dir : np.ndarray
        ``[3]`` steering/look direction.
    step_deg : float
        Angular step in degrees over ``[-180, 180]``.

    Returns
    -------
    dict
        ``{"H": (angles_deg[A], unit_vectors[A, 3]), "V": (...)}``.
    """
    u = np.asarray(steer_dir, dtype=np.float64)
    u = u / np.linalg.norm(u)
    ref = np.array([0.0, 0.0, 1.0])
    if abs(u @ ref) > 0.999:
        ref = np.array([1.0, 0.0, 0.0])
    v1 = ref - (ref @ u) * u
    v1 /= np.linalg.norm(v1)  # in-plane reference (horizontal)
    v2 = np.cross(u, v1)  # orthogonal (vertical)
    angles = np.arange(-180.0, 180.0 + 0.5 * step_deg, step_deg)
    rad = np.deg2rad(angles)
    arcs = {}
    for name, v in (("H", v1), ("V", v2)):
        uv = np.cos(rad)[:, None] * u[None, :] + np.sin(rad)[:, None] * v[None, :]  # [A, 3]
        arcs[name] = (angles, uv)
    return arcs


def _write_frd(
    path: Path, freqs: np.ndarray, resp: np.ndarray, header: list[str], p_ref: float
) -> None:
    """Write one ``.frd`` file: comment header then ``freq_Hz  mag_dB  phase_deg`` rows."""
    mag_db = 20.0 * np.log10(np.maximum(np.abs(resp), _MAG_FLOOR) / p_ref)  # [F]
    phase_deg = np.rad2deg(np.angle(resp))  # [F]
    with open(path, "w") as fh:
        for line in header:
            fh.write(line + "\n")
        for fr, m, ph in zip(freqs, mag_db, phase_deg):
            fh.write(f"{fr:.4f}\t{m:.4f}\t{ph:.4f}\n")


def export_filter_design(
    out_dir: str | Path,
    ds: RadiationDataset,
    result,
    *,
    step_deg: float = 10.0,
    sh_order: int | None = None,
    p_ref: float = _P_REF_PA,
) -> Path:
    """Write the audit-first export for a :class:`~beamsim2.beamform.design.DesignResult`.

    Layout::

        <out_dir>/drivers/<driver_id>/<driver_id>_<H|V>_{angle:+04.0f}.frd   (filtered)
        <out_dir>/combined/combined_<H|V>_{angle:+04.0f}.frd                  (steered P)
        <out_dir>/weights.npz                                                 (raw w[M,F])
        <out_dir>/manifest.csv                                                (file -> angle map)
        <out_dir>/design.json                                                 (spec/metrics summary)

    Parameters
    ----------
    out_dir : str or Path
        Output root (created if absent).
    ds : RadiationDataset
        The dataset the design was computed against.
    result : DesignResult
        Output of :func:`beamsim2.beamform.design.design`.
    step_deg : float
        Polar-arc angular step in degrees.
    sh_order : int or None
        SH order for the grid->arc resample (default: a safe order for the grid, capped at 30).
    p_ref : float
        dB reference (20 µPa default).

    Returns
    -------
    Path
        The resolved ``out_dir``.
    """
    out_dir = Path(out_dir).resolve()
    (out_dir / "drivers").mkdir(parents=True, exist_ok=True)
    (out_dir / "combined").mkdir(parents=True, exist_ok=True)

    h = stacked_h_full(ds)  # [M, F, N]
    obs = ds.directions
    freqs = np.asarray(ds.frequencies)
    weights = result.weights  # [M, F]
    driver_ids = [d.driver_id for d in ds.drivers]
    steer = np.asarray(result.spec.steer_dir, dtype=np.float64)

    order = (
        sh_order
        if sh_order is not None
        else min(safe_order_for_grid(obs.unit_vectors.shape[0]), 20)
    )
    arcs = polar_arcs(steer, step_deg)
    # Precompute each arc's (theta, phi) once; the SH fit is shared across both arcs per field.
    arc_tp = {}
    for plane, (angles, uv) in arcs.items():
        theta = np.arccos(np.clip(uv[:, 2], -1.0, 1.0))
        phi = np.arctan2(uv[:, 1], uv[:, 0]) % (2.0 * np.pi)
        arc_tp[plane] = (angles, theta, phi)

    def resample_to_arcs(field: np.ndarray) -> dict[str, np.ndarray]:
        """Fit SH once, then evaluate on every arc: {plane: [F, A]}."""
        coeffs = forward_sh(field, obs, order)  # the one expensive (lstsq) step
        return {plane: inverse_sh(coeffs, th, ph) for plane, (_, th, ph) in arc_tp.items()}

    manifest: list[dict] = []

    # Filtered per-driver responses (weight baked in), resampled to the H/V arcs.
    for m, did in enumerate(driver_ids):
        filtered = weights[m, :, None] * h[m]  # [F, N] = w_m(f) * H_full[m]
        ddir = out_dir / "drivers" / did
        ddir.mkdir(parents=True, exist_ok=True)
        per_plane = resample_to_arcs(filtered)
        for plane, (angles, _, _) in arc_tp.items():
            resampled = per_plane[plane]  # [F, A]
            for ai, ang in enumerate(angles):
                fname = f"{did}_{plane}_{ang:+04.0f}.frd"
                header = [
                    "* BeamSimII Phase-2 filtered driver response (audit)",
                    f"* driver_id: {did}   plane: {plane}   angle_deg: {ang:.1f}",
                    "* response = design_weight w_m(f) * H_full[m]  (weight BAKED IN; audit only)",
                    f"* engine: {result.attrs.get('engine')}   on-axis = steering direction",
                    "* phase referenced to global origin (0,0,0); NOT re-zeroed per driver.",
                    "* frequency_Hz  magnitude_dB  phase_deg",
                ]
                _write_frd(ddir / fname, freqs, resampled[:, ai], header, p_ref)
                manifest.append(
                    {
                        "file": f"drivers/{did}/{fname}",
                        "kind": "driver",
                        "driver_id": did,
                        "plane": plane,
                        "angle_deg": f"{ang:.1f}",
                    }
                )

    # Combined steered response on the same arcs.
    combined_per_plane = resample_to_arcs(result.steered_field)
    for plane, (angles, _, _) in arc_tp.items():
        resampled = combined_per_plane[plane]  # [F, A]
        for ai, ang in enumerate(angles):
            fname = f"combined_{plane}_{ang:+04.0f}.frd"
            header = [
                "* BeamSimII Phase-2 combined steered response (audit)",
                f"* plane: {plane}   angle_deg: {ang:.1f}   engine: {result.attrs.get('engine')}",
                "* response = sum_m w_m(f) * H_full[m]  (the achieved system directivity)",
                "* phase referenced to global origin (0,0,0).",
                "* frequency_Hz  magnitude_dB  phase_deg",
            ]
            _write_frd(out_dir / "combined" / fname, freqs, resampled[:, ai], header, p_ref)
            manifest.append(
                {
                    "file": f"combined/{fname}",
                    "kind": "combined",
                    "driver_id": "",
                    "plane": plane,
                    "angle_deg": f"{ang:.1f}",
                }
            )

    # Raw weights (re-loadable) and a small JSON summary.
    np.savez(
        out_dir / "weights.npz",
        weights=weights,
        frequencies=freqs,
        driver_ids=np.array(driver_ids),
        steer_dir=steer,
        engine=str(result.attrs.get("engine")),
    )
    with open(out_dir / "manifest.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["file", "kind", "driver_id", "plane", "angle_deg"])
        writer.writeheader()
        writer.writerows(manifest)
    summary = {
        "engine": result.attrs.get("engine"),
        "convention": result.attrs.get("convention"),
        "wng_floor_db": result.attrs.get("wng_floor_db"),
        "n_drivers": len(driver_ids),
        "n_frequencies": int(len(freqs)),
        "di_db": np.asarray(result.metrics["di_db"]).round(3).tolist(),
        "feasible_mask": np.asarray(result.metrics["feasible_mask"]).astype(bool).tolist(),
    }
    with open(out_dir / "design.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    return out_dir


def load_design_weights(path: str | Path) -> dict:
    """Load a ``weights.npz`` written by :func:`export_filter_design`.

    Returns
    -------
    dict
        ``{"weights": [M, F] complex, "frequencies": [F], "driver_ids": [M], "steer_dir": [3],
        "engine": str}`` — feed ``weights`` back through
        :func:`beamsim2.validation.closed_loop.steer_response` to reconstruct the beam.
    """
    data = np.load(path, allow_pickle=False)
    return {
        "weights": data["weights"],
        "frequencies": data["frequencies"],
        "driver_ids": list(data["driver_ids"]),
        "steer_dir": data["steer_dir"],
        "engine": str(data["engine"]),
    }
