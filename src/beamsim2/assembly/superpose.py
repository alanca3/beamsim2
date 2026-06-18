"""Per-driver field superposition: stacks independently solved driver BEM results
into the combined multi-driver pressure field.

Each driver is solved once on its own (unit cone velocity; all other driver surfaces
left sound-hard). Because BEM is linear, the sum of independent fields equals the
field produced by all drivers vibrating simultaneously — §3.4, superposition principle.
VERIFIED: standard BEM linearity (Kinsler et al. §7.1; Kreuzer et al. 2024).

This module has deliberately no phase processing. The raw ComplexField.pressure IS
H_bem — the contract field with natural time-of-flight phase preserved (§3.4 cardinal
rule). Any re-zeroing here would silently mis-steer the Phase-2 beam.
"""

from __future__ import annotations

import numpy as np

from beamsim2.core.types import ComplexField


def driver_h_bem(field: ComplexField) -> np.ndarray:
    """Return the raw BEM pressure array for one driver.

    Parameters
    ----------
    field : ComplexField
        Result of ``BEMBackend.extract()`` for a single driver solved at unit
        normal cone velocity.

    Returns
    -------
    np.ndarray
        ``field.pressure`` unchanged.  Shape ``[F × N]`` complex128.
        No phase modification — the natural time-of-flight phase from the
        driver's position must be preserved for Phase-2 beamforming (§3.4).
    """
    return field.pressure  # [F × N] complex128


def superpose_fields(fields: list[np.ndarray]) -> np.ndarray:
    """Sum per-driver complex pressure fields (linear superposition).

    Parameters
    ----------
    fields : list of np.ndarray
        Per-driver ``H_bem`` arrays, each ``[F × N]`` complex128.  All must
        share the same shape — same frequency grid and same observation sphere.

    Returns
    -------
    np.ndarray
        Complex sum ``Σ fields[m]``, shape ``[F × N]`` complex128.  This is
        what a direct multi-driver BEM solve would return, up to solver
        tolerance (the invariant V-5 verifies).

    Raises
    ------
    ValueError
        If ``fields`` is empty, or if any field has a different shape from the
        first, or if any field is not complex128.
    """
    if len(fields) == 0:
        raise ValueError("superpose_fields: need at least one field")

    ref = fields[0]
    if ref.ndim != 2:
        raise ValueError(f"superpose_fields: expected 2-D arrays [F × N], got shape {ref.shape}")
    for i, f in enumerate(fields[1:], start=1):
        if f.shape != ref.shape:
            raise ValueError(
                f"superpose_fields: shape mismatch — field[0] {ref.shape} vs "
                f"field[{i}] {f.shape}"
            )
        if not np.issubdtype(f.dtype, np.complexfloating):
            raise ValueError(f"superpose_fields: field[{i}] dtype {f.dtype} is not complex")

    # sum over driver axis — [F × N] complex128
    result = np.zeros_like(ref, dtype=np.complex128)
    for f in fields:
        result += f.astype(np.complex128)
    return result  # [F × N] complex128
