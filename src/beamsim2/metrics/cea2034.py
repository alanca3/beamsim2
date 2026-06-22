"""CEA-2034-A / spinorama directivity curves from a Phase-1 sphere dataset.

Computes the loudspeaker "spinorama": **On-Axis**, **Listening Window** (LW), **Early
Reflections** (ER), **Sound Power** (SP), the two directivity-index curves (**SPDI**,
**ERDI**), and the **Estimated In-Room Response** (PIR/EIR), per ANSI/CTA-2034-A.

The dataset stores complex pressure on a scattered Lebedev/icosphere grid; this module
**SH-resamples** it (``core.sh_transform``) onto the exact CTA-2034-A 10°-spaced
horizontal and vertical measurement orbits, built from the dataset's **reference axis**
(loudspeaker front) and a world-up vector (``core.sphere.reference_frame``), then forms
the curves.

Authoritative reference implementation (verified against, this session):
``pierreaubert/spinorama`` master ``src/spinorama/compute_cea2034.py`` (GPLv3,
© Pierre Aubert).  All spatial averages are **power-domain** (pressure-squared / energy):
``avg = pressure→spl( sqrt( mean(|p|²) ) )``; SP is the same with the CTA area weights.

Key definitions (VERIFIED, spinorama master + CTA-2034-A):
- **LW** = unweighted power-RMS of H{0,±10,±20,±30} ∪ V{±10}  (9 directions).
- **ER** = two-level power average: each of the five bounce groups is an unweighted
  power-RMS of its angles, then ER is the unweighted power-RMS of the **five group
  curves** (each *bounce* equally weighted, not each angle).
- **SP** = area-weighted power-RMS over all 70 unique orbit points (V-orbit 0°/180°
  dropped so the poles are counted once); weight = solid angle of the 10° latitude band
  at the point's polar angle from front.
- **SPDI** = LW_dB − SP_dB ;  **ERDI** = LW_dB − ER_dB  (plain dB subtraction).
- **PIR** = pressure→spl( sqrt(0.12·p_LW² + 0.44·p_ER² + 0.44·p_SP²) ).

NOTE / departure from the bug-fix proposal's loose wording (flagged per CLAUDE.md): the
proposal said "reuse ``power_di.directivity_index``".  The CEA DI curves are a *different*
quantity (LW−SP and LW−ER, not max/mean intensity); we compute the CTA definitions here
and keep the sphere-quadrature SP only as a sanity cross-check in the tests.

Phase is irrelevant to these magnitude/power averages, so the curves are unaffected by the
single-phase-origin convention — but the near-field 1/r ripple is, which is why the GUI
offers a far-field display referencing (``core.field_referencing``) the caller can apply
to ``H`` before passing it here.
"""

from __future__ import annotations

import numpy as np

from beamsim2.core.sh_transform import resample, safe_order_for_grid
from beamsim2.core.sphere import reference_frame
from beamsim2.core.types import ObservationPoints

_P_REF = 20e-6  # Pa — dB SPL reference (20 µPa), consistent with the Results views
_MAG_FLOOR = 1e-300

# ---------------------------------------------------------------------------
# Angle sets (degrees).  H = horizontal orbit, V = vertical orbit; +V = up.
# ---------------------------------------------------------------------------
_LW_H = [0, 10, -10, 20, -20, 30, -30]
_LW_V = [10, -10]

_ER_GROUPS: dict[str, list[tuple[str, int]]] = {
    "floor_bounce": [("V", -20), ("V", -30), ("V", -40)],
    "ceiling_bounce": [("V", 40), ("V", 50), ("V", 60)],
    "front_wall": [("H", a) for a in (0, 10, -10, 20, -20, 30, -30)],
    "side_wall": [("H", a) for a in (40, -40, 50, -50, 60, -60, 70, -70, 80, -80)],
    "rear_wall": [("H", a) for a in (90, -90, 100, -100, 110, -110, 120, -120, 130, -130,
                                     140, -140, 150, -150, 160, -160, 170, -170, 180)],
}  # fmt: skip

# Sound-power CTA area weights, keyed by polar angle from front (0..180°).  Symmetric about
# 90°.  VERIFIED against spinorama ``compute_weigths`` (10° latitude-band solid angles, with
# the 90° equatorial band doubled); only relative magnitudes matter (normalised by Σw).
_SP_WEIGHTS: dict[int, float] = {
    0: 0.0303847862, 10: 0.2377652066, 20: 0.4501287512, 30: 0.6226563048,
    40: 0.7534600535, 50: 0.8478858765, 60: 0.9131208534, 70: 0.9553831439,
    80: 0.9790603712, 90: 0.9866799194, 100: 0.9790603712, 110: 0.9553831439,
    120: 0.9131208534, 130: 0.8478858765, 140: 0.7534600535, 150: 0.6226563048,
    160: 0.4501287512, 170: 0.2377652066, 180: 0.0303847862,
}  # fmt: skip

# Estimated-In-Room (PIR/EIR) mix weights on squared pressure (sum to 1.0).
_PIR_WEIGHTS = {"listening_window": 0.12, "early_reflections": 0.44, "sound_power": 0.44}

#: Curves shown on the left (SPL) axis of the spinorama panel.
SPL_CURVES = [
    "on_axis",
    "listening_window",
    "early_reflections",
    "sound_power",
    "estimated_in_room",
]
#: Curves shown on the right (DI) axis.
DI_CURVES = ["sound_power_di", "early_reflections_di"]


def _orbit_direction(plane: str, angle_deg: float, frame: tuple) -> np.ndarray:
    """Unit vector at ``angle_deg`` on the H or V orbit of a reference ``frame``."""
    front, right, up = frame
    a = np.deg2rad(angle_deg)
    inplane = right if plane == "H" else up
    return np.cos(a) * front + np.sin(a) * inplane  # [3]


def _sp_points() -> list[tuple[str, int]]:
    """The 70 unique sound-power orbit points (V-orbit 0° and 180° dropped)."""
    pts = [("H", a) for a in range(0, 360, 10)]  # 36
    pts += [("V", a) for a in range(0, 360, 10) if a not in (0, 180)]  # 34
    return pts


def _fold_polar(angle_deg: int) -> int:
    """Polar angle from front (0..180°) for an orbit angle in 0..350°."""
    a = angle_deg % 360
    return a if a <= 180 else 360 - a


def cea2034_directions(reference_axis) -> dict[str, np.ndarray]:
    """All CEA-2034-A measurement directions, grouped by curve, for a reference axis.

    Returns a dict whose values are ``[n, 3]`` unit-vector arrays for ``on_axis``,
    ``listening_window``, each early-reflections group, and ``sound_power``.  Used by
    :func:`compute_cea2034`; exposed so tests can inspect/recolour the geometry.
    """
    frame = reference_frame(reference_axis)
    out: dict[str, np.ndarray] = {
        "on_axis": np.array([_orbit_direction("H", 0, frame)]),
        "listening_window": np.array(
            [_orbit_direction("H", a, frame) for a in _LW_H]
            + [_orbit_direction("V", a, frame) for a in _LW_V]
        ),
    }
    for name, pts in _ER_GROUPS.items():
        out[name] = np.array([_orbit_direction(pl, a, frame) for pl, a in pts])
    out["sound_power"] = np.array([_orbit_direction(pl, a, frame) for pl, a in _sp_points()])
    return out


def _to_db(p_lin: np.ndarray) -> np.ndarray:
    """Linear pressure magnitude → dB SPL re 20 µPa."""
    return 20.0 * np.log10(p_lin / _P_REF + _MAG_FLOOR)


def _power_rms(mags: np.ndarray) -> np.ndarray:
    """Unweighted power-domain (pressure²) RMS over the last axis. ``mags`` = |p| [..., K]."""
    return np.sqrt(np.mean(mags**2, axis=-1))


def _weighted_power_rms(mags: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Area-weighted power-domain RMS over the last axis."""
    return np.sqrt(np.sum(w * mags**2, axis=-1) / np.sum(w))


def compute_cea2034(
    H: np.ndarray,
    frequencies: np.ndarray,
    obs: ObservationPoints,
    reference_axis,
    *,
    sh_order: int | None = None,
) -> dict[str, np.ndarray]:
    """Compute the full CEA-2034-A spinorama from a per-driver (or steered) field.

    Parameters
    ----------
    H : np.ndarray
        ``[F, N]`` complex128 pressure on ``obs`` (``H_full``, ``H_bem``, or a steered
        field; the caller may pre-apply a far-field display referencing).
    frequencies : np.ndarray
        ``[F]`` float64, Hz.
    obs : ObservationPoints
        The dataset's sphere grid.
    reference_axis : array-like
        The loudspeaker front axis (defines the H/V orbits; default +z upstream).
    sh_order : int | None
        SH order for the grid→orbit resample.  Default ``min(safe_order_for_grid(N), 16)``.

    Returns
    -------
    dict of np.ndarray
        ``frequencies`` plus dB-SPL curves ``on_axis``, ``listening_window``,
        ``early_reflections``, ``sound_power``, ``estimated_in_room``, the DI curves
        ``sound_power_di``/``early_reflections_di``, and the five bounce-group curves.
        Every curve is ``[F]`` float64.
    """
    H = np.asarray(H, dtype=np.complex128)  # [F, N]
    if H.ndim != 2:
        raise ValueError(f"compute_cea2034: H must be [F, N]; got shape {H.shape}.")
    freqs = np.asarray(frequencies, dtype=np.float64)
    n = obs.unit_vectors.shape[0]
    order = sh_order if sh_order is not None else min(safe_order_for_grid(n), 16)

    dirs = cea2034_directions(reference_axis)
    # Resample the field onto every needed direction in one SH fit per group.
    mags = {key: np.abs(resample(H, obs, uv, order)) for key, uv in dirs.items()}  # {key: [F, n]}

    on_axis = mags["on_axis"][:, 0]  # [F] linear
    lw = _power_rms(mags["listening_window"])  # [F]

    group_lin = np.stack([_power_rms(mags[g]) for g in _ER_GROUPS], axis=-1)  # [F, 5]
    er = _power_rms(group_lin)  # [F] — power-RMS of the five bounce curves (equal weight)

    sp_w = np.array([_SP_WEIGHTS[_fold_polar(a)] for _, a in _sp_points()])  # [70]
    sp = _weighted_power_rms(mags["sound_power"], sp_w)  # [F]

    pir = np.sqrt(
        _PIR_WEIGHTS["listening_window"] * lw**2
        + _PIR_WEIGHTS["early_reflections"] * er**2
        + _PIR_WEIGHTS["sound_power"] * sp**2
    )  # [F] linear

    on_axis_db = _to_db(on_axis)
    lw_db = _to_db(lw)
    er_db = _to_db(er)
    sp_db = _to_db(sp)
    out = {
        "frequencies": freqs,
        "on_axis": on_axis_db,
        "listening_window": lw_db,
        "early_reflections": er_db,
        "sound_power": sp_db,
        "estimated_in_room": _to_db(pir),
        "sound_power_di": lw_db - sp_db,  # SPDI
        "early_reflections_di": lw_db - er_db,  # ERDI
    }
    for i, g in enumerate(_ER_GROUPS):
        out[g] = _to_db(group_lin[:, i])
    return out


def _self_test() -> None:
    """Validate monopole/dipole/cos² curves against hand-computed spinorama values."""
    from beamsim2.core.sphere import icosphere

    obs = icosphere(3)  # 642 points
    uv = obs.unit_vectors
    theta = np.arccos(np.clip(uv[:, 2], -1.0, 1.0))  # colatitude from +z (= front)
    freqs = np.array([200.0, 1000.0])
    axis = (0.0, 0.0, 1.0)

    def curves(field_pattern):
        H = np.tile(field_pattern.astype(np.complex128), (len(freqs), 1))  # [F, N] signed field
        c = compute_cea2034(H, freqs, obs, axis, sh_order=6)
        # SPL curves re on-axis; DI curves are already on-axis-independent differences.
        return {
            k: (v if k in DI_CURVES else v - c["on_axis"])
            for k, v in c.items()
            if k != "frequencies"
        }

    # (A) Offset monopole far-field: |H| = 1 everywhere, but a strong phase ramp.
    k = 2.0 * np.pi * 1000.0 / 343.2
    ramp = np.exp(1j * k * (uv @ np.array([0.10, 0.0, 0.0])))  # |·| = 1, phase varies
    mono = compute_cea2034(np.tile(ramp, (len(freqs), 1)), freqs, obs, axis, sh_order=8)
    for key in ("listening_window", "early_reflections", "sound_power"):
        rel = mono[key] - mono["on_axis"]
        assert np.max(np.abs(rel)) < 0.05, f"monopole {key} not flat: {rel}"
    assert np.max(np.abs(mono["sound_power_di"])) < 0.05, "monopole SPDI != 0"

    # (B) True dipole |H| = |cos θ|  — feed the SIGNED order-1 field cos θ (exactly band-
    # limited, so SH recovers it; |·| is taken inside).  Expected dB re on-axis (hand-comp).
    dip = curves(np.cos(theta))
    expect_b = {
        "listening_window": -0.433,
        "early_reflections": -2.524,
        "sound_power": -4.416,
        "sound_power_di": 3.983,
        "early_reflections_di": 2.091,
        "estimated_in_room": -2.892,
    }
    for key, exp in expect_b.items():
        got = float(dip[key][0])
        assert abs(got - exp) < 0.15, f"dipole {key}: got {got:.3f}, expected {exp:.3f}"

    # (C) cos²θ magnitude — expected dB re on-axis.
    cos2 = curves(np.cos(theta) ** 2)
    expect_c = {"listening_window": -0.823, "sound_power": -6.481, "sound_power_di": 5.658}
    for key, exp in expect_c.items():
        got = float(cos2[key][0])
        assert abs(got - exp) < 0.2, f"cos2 {key}: got {got:.3f}, expected {exp:.3f}"

    print("metrics/cea2034.py self-test: PASS (monopole flat; dipole & cos² match hand-comp)")


if __name__ == "__main__":
    _self_test()
