"""H[driver × frequency × direction] tensor assembly from per-driver ComplexField results.

Defines the Phase-1 output data model (``RadiationDataset`` / ``DriverData``)
and the ``build_dataset`` factory that wires BEM results, directions, and
metadata into the contract format (DATA_CONTRACT.md §3).

``terminal_response`` is the per-driver complex on-axis multiplier from the
driver electrical chain (T/S + semi-inductance HF, DR-05, build-order item 8).
Until item 8 is implemented it defaults to ``np.ones(F)``, so H_full == H_bem.
INFERRED: forced by build order; the field is built and exercised now.

References
----------
DATA_CONTRACT.md §3.1–§3.5.  BEAMSIMII_Gameplan.md §2 Stage F, §3.4.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from beamsim2.core.types import ComplexField, ObservationPoints


@dataclass
class DriverData:
    """All data and metadata for one driver's contribution.

    Attributes
    ----------
    driver_id : str
        Unique string identifier (e.g. ``"woofer_0"``).
    H_bem : np.ndarray
        Raw geometric response at unit cone velocity (BEM only).
        Shape ``[F × N]`` complex128.  Natural phase preserved — never re-zeroed.
    terminal_response : np.ndarray
        Per-driver complex on-axis multiplier.
        Shape ``[F]`` complex128.  ``ones(F)`` until item 8 (T/S + semi-inductance).
    H_full : np.ndarray
        ``H_bem × terminal_response[:, None]`` broadcast over directions.
        Shape ``[F × N]`` complex128.  Phase-2 default input.
    convergence_flags : np.ndarray
        Per-frequency BEM convergence flag from the solve.
        Shape ``[F]`` bool.
    attrs : dict
        Per-driver metadata (§3.5): ``name``, ``position`` [3], ``orientation`` [3],
        ``radius``, ``profile`` (dict), ``ts_params`` (dict),
        ``terminal_response_model`` (str), ``diaphragm_area`` (float).
        Any serialisable values; dict-valued items are JSON-encoded on disk.
    """

    driver_id: str
    H_bem: np.ndarray  # [F × N] complex128
    terminal_response: np.ndarray  # [F] complex128
    H_full: np.ndarray  # [F × N] complex128
    convergence_flags: np.ndarray  # [F] bool
    attrs: dict = field(default_factory=dict)


@dataclass
class RadiationDataset:
    """Full Phase-1 output dataset — the contract handed to Phase-2.

    Attributes
    ----------
    frequencies : np.ndarray
        Explicit frequency grid.  Shape ``[F]`` float64, Hz.
    spacing : str
        Grid spacing type: ``"fractional-octave"``, ``"log"``, or ``"linear"``.
    fractional_octave : float or None
        Octave fraction (e.g. 1/12) when spacing == "fractional-octave";
        ``None`` otherwise.
    interpolated_mask : np.ndarray
        ``True`` where a frequency bin was filled by SH/min-phase interpolation
        rather than a direct BEM solve.  Shape ``[F]`` bool.
    directions : ObservationPoints
        Sphere sampling grid with quadrature weights.  Carries ``unit_vectors``
        ``[N × 3]``, ``weights [N]``, ``theta_phi [N × 2]``, scheme, order,
        weight_convention, and observation radius.
    drivers : list[DriverData]
        One ``DriverData`` per driver, in list order.
    attrs : dict
        Root-level metadata (§3.5): schema/provenance, ``phase_origin`` [3],
        ``axis_convention``, ``length_units``, ``observation_radius``,
        ``far_field`` (bool), ``pressure_convention``, medium properties.
        Any serialisable values; dict-valued items are JSON-encoded on disk.
    """

    frequencies: np.ndarray  # [F] float64, Hz
    spacing: str
    fractional_octave: float | None
    interpolated_mask: np.ndarray  # [F] bool
    directions: ObservationPoints
    drivers: list[DriverData]
    attrs: dict = field(default_factory=dict)


def build_dataset(
    driver_inputs: list[tuple[str, ComplexField, dict]],
    directions: ObservationPoints,
    freq_grid_spacing: str = "log",
    freq_grid_fractional_octave: float | None = None,
    root_attrs: dict | None = None,
    terminal_responses: list[np.ndarray] | None = None,
) -> RadiationDataset:
    """Assemble a ``RadiationDataset`` from per-driver BEM solve results.

    Parameters
    ----------
    driver_inputs : list of (driver_id, ComplexField, attrs_dict)
        One entry per driver.  ``ComplexField.pressure`` ``[F × N]`` complex128
        is the raw BEM field (H_bem).  ``attrs_dict`` holds the §3.5 per-driver
        metadata (name, position, orientation, radius, profile, ts_params, …).
    directions : ObservationPoints
        Sphere sampling grid (same one used in the BEM solve).  ``N`` must match
        ``ComplexField.pressure.shape[1]``.
    freq_grid_spacing : str
        Grid spacing label stored in the dataset.  Default ``"log"``.
    freq_grid_fractional_octave : float or None
        Octave fraction; ``None`` unless spacing == "fractional-octave".
    root_attrs : dict or None
        Root-level §3.5 metadata.  ``None`` → empty dict.
    terminal_responses : list of np.ndarray or None
        One ``[F]`` complex128 array per driver.  ``None`` → ``ones(F)`` for
        each driver (H_full == H_bem until item 8 implements DR-05).

    Returns
    -------
    RadiationDataset

    Raises
    ------
    ValueError
        If any driver's ComplexField has a frequency array that does not match the
        first driver's, or if the direction count N does not match ``directions``.
        Assembly refuses to emit a silently-incomplete tensor (§2 Stage F).
    """
    if len(driver_inputs) == 0:
        raise ValueError("build_dataset: need at least one driver")

    # reference shape from first driver
    first_field = driver_inputs[0][1]
    ref_freqs = first_field.frequencies  # [F] float64
    ref_F = len(ref_freqs)
    ref_N = directions.unit_vectors.shape[0]

    # validate N matches the prepared observation grid
    if first_field.pressure.shape[1] != ref_N:
        raise ValueError(
            f"build_dataset: driver[0] pressure has N={first_field.pressure.shape[1]} "
            f"directions but ObservationPoints has N={ref_N}"
        )

    drivers: list[DriverData] = []
    for i, (driver_id, field_obj, driver_attrs) in enumerate(driver_inputs):
        # validate frequency consistency
        if len(field_obj.frequencies) != ref_F or not np.allclose(field_obj.frequencies, ref_freqs):
            raise ValueError(
                f"build_dataset: driver[{i}] ('{driver_id}') frequency grid "
                f"does not match driver[0]"
            )
        # validate direction count
        if field_obj.pressure.shape[1] != ref_N:
            raise ValueError(
                f"build_dataset: driver[{i}] ('{driver_id}') pressure has "
                f"N={field_obj.pressure.shape[1]} directions, expected {ref_N}"
            )
        if field_obj.pressure.shape[0] != ref_F:
            raise ValueError(
                f"build_dataset: driver[{i}] ('{driver_id}') pressure has "
                f"F={field_obj.pressure.shape[0]} frequencies, expected {ref_F}"
            )

        H_bem = field_obj.pressure.astype(np.complex128)  # [F × N] complex128

        if terminal_responses is not None:
            tr = terminal_responses[i].astype(np.complex128)  # [F] complex128
            if tr.ndim != 1 or len(tr) != ref_F:
                raise ValueError(
                    f"build_dataset: terminal_responses[{i}] ('{driver_id}') has "
                    f"shape {tr.shape}, expected ({ref_F},)"
                )
        else:
            tr = np.ones(ref_F, dtype=np.complex128)  # [F] complex128 — identity

        # H_full = H_bem × terminal_response[:, None]  broadcasts over N directions
        H_full = H_bem * tr[:, None]  # [F × N] complex128

        drivers.append(
            DriverData(
                driver_id=driver_id,
                H_bem=H_bem,
                terminal_response=tr,
                H_full=H_full,
                convergence_flags=field_obj.convergence_flags.copy(),
                attrs=dict(driver_attrs),
            )
        )

    interpolated_mask = np.zeros(ref_F, dtype=bool)  # [F] bool — no interpolation
    # incorporate any interpolated_mask from first driver's frequency grid
    # (future: propagate from FrequencyGrid.interpolated_mask)

    return RadiationDataset(
        frequencies=ref_freqs.astype(np.float64),  # [F] float64
        spacing=freq_grid_spacing,
        fractional_octave=freq_grid_fractional_octave,
        interpolated_mask=interpolated_mask,
        directions=directions,
        drivers=drivers,
        attrs=dict(root_attrs) if root_attrs is not None else {},
    )


def stacked_h_full(ds: RadiationDataset) -> np.ndarray:
    """Stack all drivers' H_full into one tensor.

    Parameters
    ----------
    ds : RadiationDataset

    Returns
    -------
    np.ndarray
        Shape ``[M × F × N]`` complex128, where M = number of drivers.
        This is the steering matrix Phase-2 assembles from.
    """
    return np.stack([d.H_full for d in ds.drivers], axis=0)  # [M × F × N] complex128
