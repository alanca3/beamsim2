"""V-SH — spherical-harmonic transform + resampling round-trip (Stage P2-0c).

A band-limited field, sampled on the scattered (icosphere/Lebedev) grid, must return to
itself after forward+inverse SH, and must resample faithfully onto a regular grid. This is
the capability that lets the Phase-2 audit export put the steered field onto VituixCAD/REW
polar arcs and measure CBT beamwidth on a fine arc.
"""

from __future__ import annotations

import numpy as np
import pytest

from beamsim2.core.sh_transform import (
    forward_sh,
    great_circle_arc,
    inverse_sh,
    n_coeffs,
    regular_lat_lon_grid,
    resample,
    safe_order_for_grid,
    sh_design_matrix,
)
from beamsim2.core.sphere import icosphere, lebedev


def _bandlimited(obs, order, seed=0):
    """A field that is exactly band-limited to ``order`` (random SH coeffs)."""
    rng = np.random.default_rng(seed)
    k = n_coeffs(order)
    coeffs = rng.standard_normal(k) + 1j * rng.standard_normal(k)  # [K]
    th, ph = obs.theta_phi[:, 0], obs.theta_phi[:, 1]
    field = coeffs @ sh_design_matrix(order, th, ph)  # [N]
    return coeffs, field


def test_n_coeffs():
    assert [n_coeffs(deg) for deg in range(4)] == [1, 4, 9, 16]


def test_forward_inverse_roundtrip_icosphere():
    """Fit a band-limited field on the icosphere and reconstruct it exactly."""
    obs = icosphere(4)  # 2562 points
    _, field = _bandlimited(obs, order=4)
    coeffs = forward_sh(field, obs, order=6)  # fit above the bandlimit
    th, ph = obs.theta_phi[:, 0], obs.theta_phi[:, 1]
    recon = inverse_sh(coeffs, th, ph)
    assert np.max(np.abs(recon - field)) < 1e-8


def test_resample_icosphere_to_regular_grid_matches_analytic():
    """Resampling a band-limited field to a regular grid matches the analytic field."""
    obs = icosphere(4)
    c0, field = _bandlimited(obs, order=4)
    uv, th_r, ph_r = regular_lat_lon_grid(37, 72)
    got = resample(field, obs, uv, order=6)
    analytic = inverse_sh(c0, th_r, ph_r)  # the true field at the regular directions
    assert np.max(np.abs(got - analytic)) < 1e-7


def test_forward_handles_multifrequency():
    """[F, N] fields fit per-frequency in one call."""
    obs = icosphere(3)  # 642 points
    _, fa = _bandlimited(obs, 3, seed=0)
    _, fb = _bandlimited(obs, 3, seed=1)
    field = np.stack([fa, fb])  # [2, N]
    coeffs = forward_sh(field, obs, order=4)
    assert coeffs.shape == (2, n_coeffs(4))
    recon = inverse_sh(coeffs, obs.theta_phi[:, 0], obs.theta_phi[:, 1])  # [2, N]
    assert np.max(np.abs(recon - field)) < 1e-8


def test_quadrature_forward_exact_on_lebedev():
    """On an exact-quadrature Lebedev grid, the quadrature projection recovers the coeffs."""
    obs = lebedev(26)  # exact to algebraic degree 7
    c0, field = _bandlimited(obs, order=2)  # products stay <= degree 4 <= 7
    coeffs = forward_sh(field, obs, order=2, method="quadrature")
    assert np.max(np.abs(coeffs - c0)) < 1e-9


def test_forward_raises_when_order_exceeds_grid():
    obs = lebedev(14)  # only 14 points
    _, field = _bandlimited(obs, 2)
    with pytest.raises(ValueError, match="needs"):
        forward_sh(field, obs, order=4)  # (4+1)^2 = 25 > 14


def test_safe_order_for_grid():
    order = safe_order_for_grid(2562)
    assert n_coeffs(order) <= 0.5 * 2562
    assert n_coeffs(order + 1) > 0.5 * 2562


def test_regular_lat_lon_grid_shapes_and_norms():
    uv, th, ph = regular_lat_lon_grid(19, 36)
    assert uv.shape == (19 * 36, 3)
    assert np.max(np.abs(np.linalg.norm(uv, axis=1) - 1.0)) < 1e-12
    assert th.min() >= 0.0 and th.max() <= np.pi + 1e-12
    assert ph.min() >= 0.0 and ph.max() < 2.0 * np.pi


def test_great_circle_arc_passes_through_axis():
    axis = np.array([0.0, 0.0, 1.0])
    angle, uv = great_circle_arc(axis, n_points=361)
    assert np.max(np.abs(np.linalg.norm(uv, axis=1) - 1.0)) < 1e-12
    # The middle sample (angle 0) sits on the axis.
    mid = np.argmin(np.abs(angle))
    assert np.allclose(uv[mid], axis, atol=1e-12)
    assert angle[0] == pytest.approx(-np.pi)
    assert angle[-1] == pytest.approx(np.pi)
