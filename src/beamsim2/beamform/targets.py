"""Target beam specification and target-field construction (Stage P2-1).

A :class:`TargetSpec` is the user's request: a beam *shape* (preset, a continuous
cardioid order, or an arbitrary custom pattern), a *steering direction*, optional
*nulls*, a design *band*, the chosen solver *engine*, and the single *robustness*
knob. :func:`build_target` turns it into a complex target field ``b_f[F, N]`` on the
dataset's sphere grid and/or accept/reject angular masks the solvers integrate over.

The first-order pattern family is ``T(theta) = a + (1 - a) * cos(theta)`` with
``a in [0, 1]`` (a=1 omni, a=0.5 cardioid, a=0.25 supercardioid, a=0 figure-8),
measured from the steering direction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Preset name -> first-order coefficient a in T(theta) = a + (1-a) cos(theta).
PRESET_ORDER_A: dict[str, float] = {
    "omni": 1.0,
    "subcardioid": 0.7,
    "cardioid": 0.5,
    "supercardioid": 0.366,  # max front-to-back ratio (a = (sqrt(3)-1)/2)
    "hypercardioid": 0.25,  # max directivity index (first order)
    "figure8": 0.0,
}


@dataclass
class TargetSpec:
    """A user request for a beam shape + steering, plus solver/robustness choices.

    Parameters
    ----------
    mode : str
        ``"preset" | "cardioid_order" | "steering_only" | "custom"``.
    preset : str | None
        One of :data:`PRESET_ORDER_A` keys when ``mode == "preset"``.
    order_a : float | None
        First-order coefficient ``a in [0, 1]`` when ``mode == "cardioid_order"``.
    steer_dir : np.ndarray
        Unit beam direction ``[3]`` float64 (the main-lobe axis).
    nulls : list[np.ndarray]
        Unit directions ``[3]`` to place nulls (soft target zeros or LCMV hard nulls).
    custom_target : np.ndarray | None
        ``[N]`` (or ``[F, N]``) complex/real desired pattern when ``mode == "custom"``.
    band_hz : tuple[float, float]
        Design band ``(f_lo, f_hi)``; out-of-band handling per engine.
    wng_floor_db : float
        The single robustness knob: white-noise-gain floor in dB (see ``regularize``).
    accept_halfangle_deg : float
        Half-angle of the "accept" cap about the steering direction (constant-DI engine #2).
    target_gdi_db : float | None
        Desired constant generalized directivity index (dB) for ``engine="constant_di"``.
        ``None`` -> use the maximum feasible value (the min over frequency of the ceiling).
    engine : str
        ``"delay_sum" | "ls" | "mvdr" | "lcmv" | "max_directivity" | "constant_di"``.
    """

    mode: str = "preset"
    preset: str | None = "cardioid"
    order_a: float | None = None
    steer_dir: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    nulls: list[np.ndarray] = field(default_factory=list)
    custom_target: np.ndarray | None = None
    band_hz: tuple[float, float] = (20.0, 20000.0)
    wng_floor_db: float = -6.0
    accept_halfangle_deg: float = 60.0
    target_gdi_db: float | None = None
    engine: str = "ls"


@dataclass
class Target:
    """The resolved target a solver consumes (output of :func:`build_target`).

    Attributes
    ----------
    b_field : np.ndarray
        ``[F, N]`` complex128 — desired pressure pattern on the grid (LS engine).
    look_idx : int
        Grid index nearest the steering direction (MVDR/LCMV/matched-field).
    null_idx : list[int]
        Grid indices nearest each requested null (LCMV hard nulls / soft zeros).
    accept_mask : np.ndarray
        ``[N]`` float64 in {0,1} — the "accept"/target angular region (Luo engine #2).
    reject_mask : np.ndarray
        ``[N]`` float64 in {0,1} — the "reject" region (whole sphere by default).
    pattern : np.ndarray
        ``[N]`` float64 — the frequency-independent directivity shape (for reference/plots).
    """

    b_field: np.ndarray
    look_idx: int
    null_idx: list[int]
    accept_mask: np.ndarray
    reject_mask: np.ndarray
    pattern: np.ndarray


def _first_order_pattern(cos_ang: np.ndarray, a: float) -> np.ndarray:
    """First-order directivity ``T = a + (1 - a) cos(angle)`` (signed; rear lobe allowed)."""
    return a + (1.0 - a) * cos_ang


def _shape_pattern(spec: TargetSpec, cos_ang: np.ndarray) -> np.ndarray:
    """Resolve the (real, signed) directivity pattern over the grid from ``spec``."""
    if spec.mode == "custom":
        if spec.custom_target is None:
            raise ValueError("TargetSpec.mode == 'custom' requires custom_target.")
        return np.real(np.asarray(spec.custom_target, dtype=np.float64))
    if spec.mode == "cardioid_order":
        if spec.order_a is None:
            raise ValueError("TargetSpec.mode == 'cardioid_order' requires order_a.")
        return _first_order_pattern(cos_ang, float(spec.order_a))
    if spec.mode == "steering_only":
        # A moderately narrow lobe (hypercardioid) — pure pointing without a named shape.
        return _first_order_pattern(cos_ang, 0.25)
    # mode == "preset"
    preset = spec.preset or "cardioid"
    if preset in PRESET_ORDER_A:
        return _first_order_pattern(cos_ang, PRESET_ORDER_A[preset])
    if preset == "wide":
        return _first_order_pattern(cos_ang, 0.7)  # subcardioid (broad)
    if preset == "narrow":
        # Narrower than first order allows: a non-negative raised-cosine squared lobe.
        return (0.5 * (1.0 + cos_ang)) ** 2
    raise ValueError(f"Unknown preset {preset!r}.")


def build_target(spec: TargetSpec, directions, frequencies: np.ndarray) -> Target:
    """Build the :class:`Target` (field + look/null indices + accept/reject masks).

    Parameters
    ----------
    spec : TargetSpec
        The user request.
    directions : ObservationPoints
        Sphere grid (``unit_vectors[N, 3]``, ``weights[N]``).
    frequencies : np.ndarray
        ``[F]`` float64 Hz. The pattern is frequency-independent here; the field is
        broadcast across F (per-frequency shaping is the Luo engine's job).

    Returns
    -------
    Target
    """
    uv = directions.unit_vectors  # [N, 3]
    steer = np.asarray(spec.steer_dir, dtype=np.float64)
    steer = steer / np.linalg.norm(steer)
    cos_ang = uv @ steer  # [N] cos(angle from steering axis)

    pattern = _shape_pattern(spec, cos_ang)  # [N] real signed
    n_f = len(frequencies)
    b_field = np.broadcast_to(pattern.astype(np.complex128), (n_f, pattern.shape[0])).copy()

    look_idx = int(np.argmax(cos_ang))
    null_idx = [int(np.argmax(uv @ (np.asarray(d, float) / np.linalg.norm(d)))) for d in spec.nulls]

    # Accept = forward cap about the steering axis; reject = whole sphere.
    accept_mask = (cos_ang >= np.cos(np.deg2rad(spec.accept_halfangle_deg))).astype(np.float64)
    reject_mask = np.ones(uv.shape[0], dtype=np.float64)  # [N]

    return Target(
        b_field=b_field,
        look_idx=look_idx,
        null_idx=null_idx,
        accept_mask=accept_mask,
        reject_mask=reject_mask,
        pattern=pattern,
    )
