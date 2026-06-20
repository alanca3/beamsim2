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
    engine : str
        ``"delay_sum" | "ls" | "mvdr" | "lcmv" | "luo_mscd" | "luo_mecd"``.
    """

    mode: str = "preset"
    preset: str | None = "cardioid"
    order_a: float | None = None
    steer_dir: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    nulls: list[np.ndarray] = field(default_factory=list)
    custom_target: np.ndarray | None = None
    band_hz: tuple[float, float] = (20.0, 20000.0)
    wng_floor_db: float = -6.0
    engine: str = "ls"


def build_target(spec: TargetSpec, directions, frequencies: np.ndarray) -> np.ndarray:
    """Build the complex target field ``b_f[F, N]`` from a :class:`TargetSpec`.

    Parameters
    ----------
    spec : TargetSpec
        The user request.
    directions : ObservationPoints
        Sphere grid (``unit_vectors[N, 3]``, ``weights[N]``).
    frequencies : np.ndarray
        ``[F]`` float64 Hz.

    Returns
    -------
    np.ndarray
        ``b_f[F, N]`` complex128 desired pressure pattern on the grid.
    """
    raise NotImplementedError("Stage P2-1: target-field construction not yet implemented.")
