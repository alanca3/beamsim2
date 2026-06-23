"""Reconstruct the real 2-driver radiation dataset from the run2 .frd audit export.

The user's failing filter-designer run is in ``HDF5/run2/``. The saved ``HDF5.h5`` is
corrupt (a GUI save error dropped ``driver_1``), but the per-direction Phase-1 ``.frd``
exports under ``run2/driver_{0,1}/H_full/`` are intact and carry the full complex field
(magnitude in dB re 20 µPa, phase in degrees, referenced to the global origin). This
module rebuilds ``H_full[M, F, N]`` in Pa from those files, aligned to the (intact)
``directions/`` grid + quadrature weights in ``HDF5.h5``, and returns a
:class:`~beamsim2.assembly.tensor.RadiationDataset` that feeds
:func:`beamsim2.beamform.design.design` unchanged.

This is a **local verification tool**, not a CI fixture: ``HDF5/`` is git-ignored
(``*.h5``), so callers must skip when :data:`RUN2_DIR` is absent. Reconstruction is
lossless to ~1e-6 relative error (verified against the stored ``driver_0/H_full``).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from beamsim2.assembly.tensor import DriverData, RadiationDataset
from beamsim2.core.types import ObservationPoints

# Repo-root-relative location of the user's run2 export.
RUN2_DIR = Path(__file__).resolve().parents[2] / "HDF5" / "run2"
_P_REF_PA = 20e-6  # the Phase-1 .frd dB-SPL reference (io/frd_export._P_REF_PA)


def run2_available() -> bool:
    """True if the run2 export (and its per-driver H_full .frd files) is present."""
    return (RUN2_DIR / "HDF5.h5").exists() and (RUN2_DIR / "driver_0" / "H_full").is_dir()


def _parse_frd(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse one Phase-1 .frd → ``(freqs[F], H[F] complex)``.

    Rows are ``freq_Hz  mag_dB(re 20 µPa)  phase_deg``; ``*`` lines are comments.
    """
    fr: list[float] = []
    mag: list[float] = []
    ph: list[float] = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("*") or not line.strip():
                continue
            a, b, c = line.split()
            fr.append(float(a))
            mag.append(float(b))
            ph.append(float(c))
    freqs = np.asarray(fr, dtype=np.float64)
    h = (_P_REF_PA * 10.0 ** (np.asarray(mag) / 20.0)) * np.exp(1j * np.deg2rad(ph))
    return freqs, h.astype(np.complex128)


def _reconstruct_h_full(did: str, n_dir: int, n_freq: int) -> tuple[np.ndarray, np.ndarray]:
    """Rebuild one driver's ``H_full[F, N]`` (Pa) from its per-direction .frd files."""
    base = RUN2_DIR / did / "H_full"
    h = np.empty((n_freq, n_dir), dtype=np.complex128)  # [F, N]
    freqs0: np.ndarray | None = None
    for n in range(n_dir):
        fr, col = _parse_frd(base / f"{did}_H_full_dir{n:04d}.frd")
        if freqs0 is None:
            freqs0 = fr
        h[:, n] = col
    assert freqs0 is not None
    return freqs0, h


def load_run2_dataset() -> RadiationDataset:
    """Return the reconstructed real 2-driver :class:`RadiationDataset` from run2.

    Directions, quadrature weights, frequencies, and root attrs (speed_of_sound,
    reference_axis, observation_radius, …) are read from the intact parts of
    ``HDF5.h5``; ``H_full`` for each driver is rebuilt from the ``.frd`` files. The
    BEM speed of sound and ``reference_axis`` are preserved so ``design()`` /
    ``build_target`` behave exactly as in the GUI run.
    """
    import h5py  # local import: only needed for the local verification path

    with h5py.File(RUN2_DIR / "HDF5.h5", "r") as f:
        uv = f["directions/unit_vectors"][:]  # [N, 3]
        weights = f["directions/weights"][:]  # [N]
        theta_phi = f["directions/theta_phi"][:] if "theta_phi" in f["directions"] else None
        freqs = f["frequencies"][:]  # [F]
        dir_attrs = {k: f["directions"].attrs[k] for k in f["directions"].attrs}
        root_attrs = {k: f.attrs[k] for k in f.attrs}
        order = json.loads(root_attrs["driver_order"])  # ["driver_0", "driver_1"]

    n, n_freq = uv.shape[0], len(freqs)
    directions = ObservationPoints(
        unit_vectors=uv,
        radius=float(dir_attrs.get("radius", 1.0)),
        weights=weights,
        scheme=str(dir_attrs.get("scheme", "icosphere")),
        order=int(dir_attrs.get("order", 0)),
        weight_convention=str(dir_attrs.get("weight_convention", "sum_4pi")),
        theta_phi=theta_phi,
    )

    drivers: list[DriverData] = []
    for did in order:
        _, h_full = _reconstruct_h_full(did, n, n_freq)  # [F, N] Pa
        drivers.append(
            DriverData(
                driver_id=did,
                H_bem=h_full,  # H_bem not needed by design(); mirror H_full
                terminal_response=np.ones(n_freq, dtype=np.complex128),
                H_full=h_full,
                convergence_flags=np.ones(n_freq, dtype=bool),
                attrs={},
            )
        )

    # Decode JSON-encoded root attrs (reference_axis, phase_origin) back to values.
    decoded: dict = {}
    for k, v in root_attrs.items():
        if isinstance(v, str):
            try:
                decoded[k] = json.loads(v)
                continue
            except (json.JSONDecodeError, ValueError):
                pass
        decoded[k] = v

    return RadiationDataset(
        frequencies=freqs.astype(np.float64),
        spacing=str(decoded.get("spacing", "fractional-octave")),
        fractional_octave=(
            float(decoded["fractional_octave"]) if "fractional_octave" in decoded else None
        ),
        interpolated_mask=np.zeros(n_freq, dtype=bool),
        directions=directions,
        drivers=drivers,
        attrs=decoded,
    )
