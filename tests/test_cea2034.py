"""CEA-2034-A / spinorama metrics (Chunk 2).

Verifies the angle sets, and the computed curves against hand-computed values for known
axisymmetric patterns (offset monopole → flat; true dipole and cos² → spinorama-derived
numbers).  The expected dB values were verified this session against the authoritative
``pierreaubert/spinorama`` master implementation (see docs/Chunk2_Results_Viz_Design.md).
"""

from __future__ import annotations

import numpy as np

from beamsim2.core.sphere import icosphere
from beamsim2.metrics.cea2034 import (
    _ER_GROUPS,
    _LW_H,
    _LW_V,
    DI_CURVES,
    _sp_points,
    cea2034_directions,
    compute_cea2034,
)
from beamsim2.validation.power_di import directivity_index


def _theta_from(uvecs: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Polar angle (rad) of each direction from ``axis``."""
    a = axis / np.linalg.norm(axis)
    return np.arccos(np.clip(uvecs @ a, -1.0, 1.0))


def _curves_for_pattern(field: np.ndarray, obs, axis, freqs, sh_order=6):
    """Compute CEA curves for a real signed field; SPL curves returned re on-axis."""
    H = np.tile(field.astype(np.complex128), (len(freqs), 1))  # [F, N]
    c = compute_cea2034(H, freqs, obs, axis, sh_order=sh_order)
    return {
        k: (v if k in DI_CURVES else v - c["on_axis"]) for k, v in c.items() if k != "frequencies"
    }


def test_angle_set_sizes():
    """The CEA-2034-A angle sets have the standard cardinalities."""
    assert len(_LW_H) + len(_LW_V) == 9, "Listening Window must be 9 directions"
    sizes = {g: len(pts) for g, pts in _ER_GROUPS.items()}
    assert sizes == {
        "floor_bounce": 3,
        "ceiling_bounce": 3,
        "front_wall": 7,
        "side_wall": 10,
        "rear_wall": 19,
    }
    assert len(_sp_points()) == 70, "Sound Power must use 70 unique orbit points"
    # Directions arrays have the matching row counts and are unit vectors.
    dirs = cea2034_directions((0.0, 0.0, 1.0))
    assert dirs["listening_window"].shape == (9, 3)
    assert dirs["sound_power"].shape == (70, 3)
    for uv in dirs.values():
        assert np.allclose(np.linalg.norm(uv, axis=1), 1.0)


def test_monopole_is_flat_and_zero_di():
    """A flat-magnitude (omni) field with a phase ramp → all curves equal, DI = 0."""
    obs = icosphere(3)
    freqs = np.array([200.0, 1000.0])
    axis = np.array([0.0, 0.0, 1.0])
    k = 2.0 * np.pi * 1000.0 / 343.2
    ramp = np.exp(1j * k * (obs.unit_vectors @ np.array([0.1, 0.0, 0.0])))  # |·| = 1
    c = compute_cea2034(np.tile(ramp, (2, 1)), freqs, obs, axis, sh_order=8)
    for key in ("listening_window", "early_reflections", "sound_power", "estimated_in_room"):
        np.testing.assert_allclose(c[key], c["on_axis"], atol=0.05, err_msg=f"{key} not flat")
    np.testing.assert_allclose(c["sound_power_di"], 0.0, atol=0.05)
    np.testing.assert_allclose(c["early_reflections_di"], 0.0, atol=0.05)


def test_true_dipole_matches_hand_computed():
    """|H| = |cos θ| dipole → spinorama hand-computed dB (re on-axis)."""
    obs = icosphere(3)
    freqs = np.array([500.0])
    axis = np.array([0.0, 0.0, 1.0])
    field = np.cos(_theta_from(obs.unit_vectors, axis))  # signed order-1 field
    got = _curves_for_pattern(field, obs, axis, freqs)
    expect = {
        "listening_window": -0.433,
        "early_reflections": -2.524,
        "sound_power": -4.416,
        "sound_power_di": 3.983,
        "early_reflections_di": 2.091,
        "estimated_in_room": -2.892,
    }
    for key, exp in expect.items():
        assert abs(float(got[key][0]) - exp) < 0.15, f"{key}: {float(got[key][0]):.3f} vs {exp}"


def test_cos2_pattern_matches_hand_computed():
    """|H| = cos²θ → spinorama hand-computed dB (re on-axis)."""
    obs = icosphere(3)
    freqs = np.array([500.0])
    axis = np.array([0.0, 0.0, 1.0])
    field = np.cos(_theta_from(obs.unit_vectors, axis)) ** 2  # order-2, non-negative
    got = _curves_for_pattern(field, obs, axis, freqs)
    expect = {
        "listening_window": -0.823,
        "early_reflections": -3.929,
        "sound_power": -6.481,
        "sound_power_di": 5.658,
        "early_reflections_di": 3.106,
    }
    for key, exp in expect.items():
        assert abs(float(got[key][0]) - exp) < 0.2, f"{key}: {float(got[key][0]):.3f} vs {exp}"


def test_reference_axis_rotation_invariance():
    """A dipole about +x with reference_axis=+x gives the same curves as about +z."""
    obs = icosphere(3)
    freqs = np.array([500.0])
    z_axis = np.array([0.0, 0.0, 1.0])
    x_axis = np.array([1.0, 0.0, 0.0])
    cz = _curves_for_pattern(np.cos(_theta_from(obs.unit_vectors, z_axis)), obs, z_axis, freqs)
    cx = _curves_for_pattern(np.cos(_theta_from(obs.unit_vectors, x_axis)), obs, x_axis, freqs)
    for key in cz:
        np.testing.assert_allclose(cz[key], cx[key], atol=0.05, err_msg=f"{key} axis-dependent")


def test_sound_power_cross_check_against_sphere_quadrature():
    """CTA-weighted Sound Power agrees with the exact sphere-quadrature power within ~0.6 dB."""
    obs = icosphere(4)  # 2562 points — accurate quadrature
    freqs = np.array([500.0])
    axis = np.array([0.0, 0.0, 1.0])
    field = np.cos(_theta_from(obs.unit_vectors, axis))  # dipole
    c = compute_cea2034(np.tile(field.astype(np.complex128), (1, 1)), freqs, obs, axis, sh_order=6)
    sp_re_onaxis = float(c["sound_power"][0] - c["on_axis"][0])
    # Sphere-quadrature "sound power" relative to on-axis (max) intensity = -DI.
    quad_sp_re_onaxis = -float(directivity_index(field[None, :], obs.weights)[0])
    assert (
        abs(sp_re_onaxis - quad_sp_re_onaxis) < 0.6
    ), f"CTA SP {sp_re_onaxis:.3f} dB vs quadrature {quad_sp_re_onaxis:.3f} dB"
