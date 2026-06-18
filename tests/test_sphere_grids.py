"""Tests for Lebedev sphere grid construction, quadrature accuracy, and SH orthogonality.

Acoustics context
-----------------
A Lebedev grid is a set of microphone positions on an imaginary sphere, with weights
that let you compute any surface integral exactly up to the grid's algebraic degree.
The tests here verify three things:

1. The weights sum to 4π (the surface area of the unit sphere).
2. Integrating the constant function 1 over the sphere recovers 4π.
3. Spherical harmonics Y_l^m are orthonormal: integrating Y_l^m * Y_l'^m'* gives
   1 if (l,m)==(l',m') and 0 otherwise. This is the acoustic analogue of checking
   that each directivity pattern in your basis is independent of every other.

For the 26-point grid (algebraic degree 7), all polynomial integrals up to degree 7
are exact. Products Y_l^m * Y_l^m are degree 2l, so orthogonality tests pass for
l ≤ 3 (degree ≤ 6 < 7). Tests on l > 3 with n=26 would fail — correctly — and are
not included.
"""

import math

import numpy as np
import pytest
from scipy.special import sph_harm_y

from beamsim2.core.sphere import (
    LEBEDEV_AVAILABLE,
    fliege_maier,
    icosphere,
    lebedev,
    t_design,
)
from beamsim2.core.types import ObservationPoints

_FOUR_PI = 4.0 * math.pi


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sh(l: int, m: int, obs: ObservationPoints) -> np.ndarray:
    """Evaluate complex spherical harmonic Y_l^m at all directions in obs.

    scipy 1.15+ API: sph_harm_y(n, m, theta, phi)
    where n=degree (l), m=order, theta=colatitude, phi=azimuth.

    Returns a [N] complex array.
    """
    assert obs.theta_phi is not None
    theta = obs.theta_phi[:, 0]  # [N] colatitude in [0, π]
    phi = obs.theta_phi[:, 1]  # [N] azimuth in [0, 2π)
    return sph_harm_y(l, m, theta, phi)  # [N] complex


# ---------------------------------------------------------------------------
# Basic construction
# ---------------------------------------------------------------------------


def test_lebedev_returns_observation_points():
    obs = lebedev(26)
    assert isinstance(obs, ObservationPoints)
    assert obs.scheme == "lebedev"
    assert obs.order == 26
    assert obs.weight_convention == "sum_4pi"
    assert obs.radius == 1.0


def test_lebedev_custom_radius():
    obs = lebedev(26, radius=2.0)
    assert obs.radius == 2.0


def test_lebedev_shapes():
    for n in LEBEDEV_AVAILABLE:
        obs = lebedev(n)
        assert obs.unit_vectors.shape == (n, 3), f"n={n}: unit_vectors shape"
        assert obs.weights.shape == (n,), f"n={n}: weights shape"
        assert obs.theta_phi is not None
        assert obs.theta_phi.shape == (n, 2), f"n={n}: theta_phi shape"


# ---------------------------------------------------------------------------
# Unit vectors lie on the unit sphere
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", LEBEDEV_AVAILABLE)
def test_unit_vectors_on_sphere(n: int):
    """Every direction vector must have L2-norm exactly 1.0."""
    obs = lebedev(n)
    norms = np.linalg.norm(obs.unit_vectors, axis=1)  # [N]
    np.testing.assert_allclose(norms, 1.0, atol=1e-14, err_msg=f"n={n}")


# ---------------------------------------------------------------------------
# Quadrature weight tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", LEBEDEV_AVAILABLE)
def test_weights_sum_to_4pi(n: int):
    """The quadrature weights must sum to 4π (surface area of the unit sphere).

    In acoustics terms: if you 'measure' the constant function 1 at every direction
    and accumulate with the weights, you should recover the total solid angle.
    """
    obs = lebedev(n)
    total = obs.weights.sum()
    assert (
        abs(total - _FOUR_PI) < 1e-12
    ), f"n={n}: sum(weights) = {total:.16g}, expected 4π = {_FOUR_PI:.16g}"


@pytest.mark.parametrize("n", LEBEDEV_AVAILABLE)
def test_integrate_constant_is_4pi(n: int):
    """∫_S² 1 dΩ = 4π, recovered as Σ wᵢ · 1."""
    obs = lebedev(n)
    integral = np.dot(obs.weights, np.ones(n))
    assert abs(integral - _FOUR_PI) < 1e-12, f"n={n}: ∫1 dΩ = {integral}"


@pytest.mark.parametrize("n", LEBEDEV_AVAILABLE)
def test_all_weights_positive(n: int):
    """Lebedev weights are always positive (a known property of Gauss-type rules)."""
    obs = lebedev(n)
    assert np.all(obs.weights > 0), f"n={n}: found non-positive weights"


# ---------------------------------------------------------------------------
# Spherical harmonic orthogonality (n=26, degree 7, tests up to l=3)
# ---------------------------------------------------------------------------


def test_Y00_integral():
    """∫ Y_0^0 dΩ = √(4π) ≈ 3.545.

    Y_0^0 = 1/√(4π) everywhere; integrating over the sphere gives
    (1/√(4π)) × 4π = √(4π).
    """
    obs = lebedev(26)
    Y00 = _sh(0, 0, obs)  # [N] — real and constant
    integral = np.dot(obs.weights, Y00.real)
    expected = math.sqrt(_FOUR_PI)
    assert (
        abs(integral - expected) < 1e-12
    ), f"∫Y_0^0 dΩ = {integral:.16g}, expected √(4π) = {expected:.16g}"


@pytest.mark.parametrize(
    "l, m",
    [
        (1, -1),
        (1, 0),
        (1, 1),
        (2, -2),
        (2, 0),
        (2, 2),
        (3, -3),
        (3, 0),
        (3, 3),
    ],
)
def test_sh_self_inner_product(l: int, m: int):
    """∫ |Y_l^m|² dΩ = 1 (orthonormality diagonal).

    Each spherical harmonic is a distinct radiation pattern; this test confirms
    that each pattern has unit 'energy' when integrated over the sphere.
    The 26-point grid is exact for algebraic degree 7, covering l ≤ 3 (degree 2l = 6 ≤ 7).
    """
    obs = lebedev(26)
    Y = _sh(l, m, obs)  # [N] complex
    integrand = (Y.conj() * Y).real  # |Y|² — always real and non-negative
    integral = np.dot(obs.weights, integrand)
    assert abs(integral - 1.0) < 1e-11, f"Y_{l}^{m}: ∫|Y|² dΩ = {integral:.16g}, expected 1.0"


@pytest.mark.parametrize(
    "l1, m1, l2, m2",
    [
        # Different l, same m
        (1, 0, 2, 0),
        (1, 0, 3, 0),
        (2, 0, 3, 0),
        # Same l, different m
        (2, -2, 2, 0),
        (2, 0, 2, 2),
        (3, -1, 3, 1),
        # Different l and m
        (1, 1, 2, -1),
        (1, -1, 3, 2),
    ],
)
def test_sh_cross_inner_product(l1: int, m1: int, l2: int, m2: int):
    """∫ Y_l1^m1 * conj(Y_l2^m2) dΩ = 0 for (l1,m1) ≠ (l2,m2).

    Different radiation patterns are orthogonal — they don't cross-talk when
    their directivity is measured over the full sphere.
    """
    obs = lebedev(26)
    Y1 = _sh(l1, m1, obs)  # [N] complex
    Y2 = _sh(l2, m2, obs)  # [N] complex
    integrand = Y1 * Y2.conj()  # complex; imaginary part should vanish
    integral = np.dot(obs.weights, integrand)
    assert (
        abs(integral) < 1e-11
    ), f"Y_{l1}^{m1} × Y_{l2}^{m2}: ∫ cross dΩ = {integral:.4g}, expected 0"


# ---------------------------------------------------------------------------
# theta_phi consistency
# ---------------------------------------------------------------------------


def test_theta_phi_roundtrip():
    """Recover unit_vectors from theta_phi and confirm consistency."""
    obs = lebedev(26)
    assert obs.theta_phi is not None
    theta = obs.theta_phi[:, 0]  # [N]
    phi = obs.theta_phi[:, 1]  # [N]
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    recovered = np.column_stack([x, y, z])  # [N, 3]
    np.testing.assert_allclose(recovered, obs.unit_vectors, atol=1e-14)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_invalid_n_raises_value_error():
    with pytest.raises(ValueError, match="not a valid Lebedev order"):
        lebedev(7)  # 7 is not in LEBEDEV_AVAILABLE


def test_too_large_n_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="vendored"):
        lebedev(590)  # valid Lebedev order, but not yet implemented here


def test_fliege_maier_raises():
    with pytest.raises(NotImplementedError):
        fliege_maier(5)


def test_t_design_raises():
    with pytest.raises(NotImplementedError):
        t_design(5)


def test_icosphere_raises():
    with pytest.raises(NotImplementedError):
        icosphere(2)
