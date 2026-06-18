"""Construction and sanity tests for core types and medium properties.

These tests verify that the data contract types (§3) can be constructed with
correct shapes, that shape validation catches malformed inputs, and that the
medium-property formulas give physically correct values.
"""

import math

import numpy as np
import pytest

from beamsim2.core.sphere import lebedev
from beamsim2.core.types import (
    BoundaryConditions,
    ComplexField,
    FrequencyGrid,
    Mesh,
    ObservationPoints,
    ResourcePlan,
    SolverConfig,
)
from beamsim2.core.units import air_attenuation, air_density, speed_of_sound

# ---------------------------------------------------------------------------
# Mesh
# ---------------------------------------------------------------------------


def test_mesh_construction():
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    tris = np.array([[0, 1, 2], [0, 1, 3]], dtype=np.int32)
    tags = np.array([1, 2], dtype=np.int32)
    mesh = Mesh(vertices=verts, triangles=tris, group_tags=tags)
    assert mesh.vertices.shape == (4, 3)
    assert mesh.triangles.shape == (2, 3)
    assert mesh.group_tags.shape == (2,)


def test_mesh_bad_vertices_shape():
    with pytest.raises(ValueError, match="vertices must be"):
        Mesh(
            vertices=np.zeros((4,)),  # wrong: must be [V,3]
            triangles=np.zeros((2, 3), dtype=np.int32),
            group_tags=np.zeros(2, dtype=np.int32),
        )


def test_mesh_mismatched_tag_length():
    with pytest.raises(ValueError, match="group_tags must be"):
        Mesh(
            vertices=np.zeros((4, 3)),
            triangles=np.zeros((2, 3), dtype=np.int32),
            group_tags=np.zeros(5, dtype=np.int32),  # wrong length
        )


# ---------------------------------------------------------------------------
# BoundaryConditions
# ---------------------------------------------------------------------------


def test_boundary_conditions_scalar_velocity():
    bc = BoundaryConditions(vibrating_groups={1: complex(0.1, 0.0)})
    assert 1 in bc.vibrating_groups
    assert len(bc.sound_hard_groups) == 0


def test_boundary_conditions_per_element_velocity():
    vel = np.ones(100, dtype=np.complex128) * 0.1
    bc = BoundaryConditions(vibrating_groups={1: vel}, sound_hard_groups={2, 3})
    assert bc.vibrating_groups[1].shape == (100,)
    assert 2 in bc.sound_hard_groups


# ---------------------------------------------------------------------------
# FrequencyGrid
# ---------------------------------------------------------------------------


def test_frequency_grid_log():
    freqs = np.geomspace(100.0, 20000.0, 48)
    fg = FrequencyGrid(frequencies=freqs, spacing="fractional-octave", fractional_octave=1 / 12)
    assert len(fg.frequencies) == 48
    assert fg.spacing == "fractional-octave"
    assert fg.fractional_octave == pytest.approx(1 / 12)


def test_frequency_grid_bad_shape():
    with pytest.raises(ValueError, match="1-D"):
        FrequencyGrid(frequencies=np.zeros((4, 2)))


def test_frequency_grid_empty():
    with pytest.raises(ValueError, match="must not be empty"):
        FrequencyGrid(frequencies=np.array([]))


# ---------------------------------------------------------------------------
# ObservationPoints
# ---------------------------------------------------------------------------


def test_observation_points_from_lebedev():
    obs = lebedev(26, radius=1.0)
    assert obs.unit_vectors.shape == (26, 3)
    assert obs.weights.shape == (26,)
    assert obs.radius == 1.0
    assert obs.weight_convention == "sum_4pi"


def test_observation_points_bad_unit_vectors():
    with pytest.raises(ValueError, match="unit_vectors must be"):
        ObservationPoints(
            unit_vectors=np.zeros((26,)),  # wrong: must be [N,3]
            radius=1.0,
            weights=np.ones(26),
        )


def test_observation_points_weight_length_mismatch():
    with pytest.raises(ValueError, match="weights must be"):
        ObservationPoints(
            unit_vectors=np.zeros((26, 3)),
            radius=1.0,
            weights=np.ones(25),  # wrong length
        )


# ---------------------------------------------------------------------------
# SolverConfig
# ---------------------------------------------------------------------------


def test_solver_config_defaults():
    cfg = SolverConfig()
    assert cfg.n_epw == 6
    assert cfg.burton_miller is True
    assert cfg.air_attenuation_model == "none"


def test_solver_config_custom():
    cfg = SolverConfig(n_epw=8, tolerance=1e-8, speed_of_sound=346.0)
    assert cfg.n_epw == 8
    assert cfg.tolerance == 1e-8
    assert cfg.speed_of_sound == 346.0


# ---------------------------------------------------------------------------
# ComplexField
# ---------------------------------------------------------------------------


def test_complex_field_construction():
    F, N = 48, 26
    field = ComplexField(
        pressure=np.zeros((F, N), dtype=np.complex128),
        convergence_flags=np.ones(F, dtype=bool),
        frequencies=np.geomspace(100.0, 20000.0, F),
    )
    assert field.pressure.shape == (F, N)
    assert field.convergence_flags.shape == (F,)
    assert field.frequencies.shape == (F,)


def test_complex_field_wrong_pressure_shape():
    with pytest.raises(ValueError, match="pressure must be"):
        ComplexField(
            pressure=np.zeros((10, 26), dtype=np.complex128),
            convergence_flags=np.ones(48, dtype=bool),  # F mismatch
            frequencies=np.geomspace(100.0, 20000.0, 48),
        )


# ---------------------------------------------------------------------------
# ResourcePlan
# ---------------------------------------------------------------------------


def test_resource_plan_construction():
    F = 48
    plan = ResourcePlan(
        ram_bytes_per_step=np.linspace(1e9, 20e9, F),
        time_seconds_per_step=np.linspace(1.0, 300.0, F),
    )
    assert plan.ram_bytes_per_step.shape == (F,)


# ---------------------------------------------------------------------------
# units: speed of sound
# ---------------------------------------------------------------------------


def test_speed_of_sound_at_20C():
    """c ≈ 343.2 m/s at 20 °C — the standard audio reference value."""
    c = speed_of_sound(20.0)
    assert abs(c - 343.2) < 0.2, f"c(20°C) = {c:.4f} m/s, expected ~343.2"


def test_speed_of_sound_at_0C():
    """c ≈ 331.3 m/s at 0 °C — standard dry air reference."""
    c = speed_of_sound(0.0)
    assert abs(c - 331.3) < 0.2, f"c(0°C) = {c:.4f} m/s, expected ~331.3"


def test_speed_of_sound_increases_with_temperature():
    """Higher temperature → faster sound (more thermal energy)."""
    assert speed_of_sound(30.0) > speed_of_sound(20.0) > speed_of_sound(10.0)


def test_speed_of_sound_humidity_ignored():
    """RH and P parameters are accepted but don't change the result (documented behaviour)."""
    c0 = speed_of_sound(20.0, RH_pct=0.0)
    c1 = speed_of_sound(20.0, RH_pct=100.0)
    assert c0 == c1  # same formula, RH not used


# ---------------------------------------------------------------------------
# units: air density
# ---------------------------------------------------------------------------


def test_air_density_standard():
    """ρ ≈ 1.204 kg/m³ at 20 °C, 101325 Pa (standard conditions for audio)."""
    rho = air_density(20.0, 0.0, 101325.0)
    assert abs(rho - 1.2041) < 0.001, f"ρ = {rho:.5f} kg/m³, expected ~1.2041"


def test_air_density_decreases_with_temperature():
    """Warm air is less dense (ideal gas: ρ ∝ 1/T)."""
    assert air_density(30.0) < air_density(20.0) < air_density(10.0)


def test_air_density_increases_with_pressure():
    """Higher pressure → denser air."""
    assert air_density(20.0, P_Pa=110000.0) > air_density(20.0, P_Pa=101325.0)


# ---------------------------------------------------------------------------
# units: air attenuation
# ---------------------------------------------------------------------------


def test_attenuation_none_scalar():
    alpha = air_attenuation(1000.0, model="none")
    assert alpha == 0.0


def test_attenuation_none_list():
    alphas = air_attenuation([500.0, 1000.0, 4000.0], model="none")
    assert alphas == [0.0, 0.0, 0.0]


def test_attenuation_unknown_model_raises():
    with pytest.raises(NotImplementedError, match="iso9613-1"):
        air_attenuation(1000.0, model="iso9613-1")
