"""Dense near-uniform sphere grid (icosphere) — Stage P2-0b.

The Phase-2 filter designer needs hundreds-to-thousands of directions (the stored
Lebedev grid topped out at 26). The icosphere supplies them, generated entirely in
code (no vendored tables), near-uniform, with spherical-area quadrature weights that
sum to 4*pi. These tests pin the point-count law, the weight normalization, and the
quadrature accuracy a beamformer relies on (an analytic cardioid's DI and SH
orthonormality).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import sph_harm_y

from beamsim2.core.sphere import icosphere


@pytest.mark.parametrize(
    "subdiv,expected_n",
    [(0, 12), (1, 42), (2, 162), (3, 642), (4, 2562), (5, 10242)],
)
def test_point_count_law(subdiv, expected_n):
    """N = 10 * 4**subdivisions + 2."""
    obs = icosphere(subdiv)
    assert obs.unit_vectors.shape == (expected_n, 3)
    assert obs.weights.shape == (expected_n,)


@pytest.mark.parametrize("subdiv", [0, 2, 4, 5])
def test_weights_sum_to_4pi_and_positive(subdiv):
    obs = icosphere(subdiv)
    assert obs.weights.sum() == pytest.approx(4.0 * np.pi, rel=1e-12)
    assert np.all(obs.weights > 0.0)


@pytest.mark.parametrize("subdiv", [3, 4, 5])
def test_unit_vectors_normalized(subdiv):
    obs = icosphere(subdiv)
    assert np.max(np.abs(np.linalg.norm(obs.unit_vectors, axis=1) - 1.0)) < 1e-12


def test_reaches_thousands_of_points():
    """The headline requirement: the simulator can produce thousands of directions."""
    assert icosphere(4).unit_vectors.shape[0] == 2562
    assert icosphere(5).unit_vectors.shape[0] == 10242


def test_near_uniform_weight_spread():
    """Weights are near-uniform (no lat/long pole clustering); spread stays modest."""
    obs = icosphere(4)
    ratio = obs.weights.max() / obs.weights.min()
    assert ratio < 1.6, f"icosphere weight spread too large: {ratio:.2f}"


@pytest.mark.parametrize("subdiv", [2, 3, 4, 5])
def test_cardioid_directivity_index(subdiv):
    """An analytic cardioid integrates to DI = 10 log10(3) = 4.7712 dB."""
    obs = icosphere(subdiv)
    ct = obs.unit_vectors[:, 2]  # cos(theta) from +z
    intensity = (0.5 * (1.0 + ct)) ** 2  # cardioid magnitude squared
    mean_i = np.sum(obs.weights * intensity) / (4.0 * np.pi)
    di_db = 10.0 * np.log10(np.max(intensity) / mean_i)
    assert di_db == pytest.approx(10.0 * np.log10(3.0), abs=1e-3)


def test_omnidirectional_directivity_index_zero():
    """A constant (monopole) field integrates to DI = 0 dB."""
    obs = icosphere(4)
    intensity = np.ones(obs.unit_vectors.shape[0])
    mean_i = np.sum(obs.weights * intensity) / (4.0 * np.pi)
    di_db = 10.0 * np.log10(np.max(intensity) / mean_i)
    assert di_db == pytest.approx(0.0, abs=1e-9)


def test_sh_orthonormality_quadrature():
    """Quadrature integrates low-degree spherical harmonics: Gram ~ I for L <= 4."""
    obs = icosphere(4)
    th, ph, w = obs.theta_phi[:, 0], obs.theta_phi[:, 1], obs.weights
    rows = [sph_harm_y(deg, m, th, ph) for deg in range(5) for m in range(-deg, deg + 1)]
    Y = np.array(rows)  # [K, N]
    gram = (Y * w[None, :]) @ Y.conj().T  # ~ identity
    assert np.max(np.abs(gram - np.eye(Y.shape[0]))) < 1e-3


def test_negative_subdivisions_raises():
    with pytest.raises(ValueError, match="subdivisions"):
        icosphere(-1)


def test_make_observation_grid_dispatch():
    """The scheme dispatcher the pipeline/GUI use: exact Lebedev vs target-count icosphere."""
    from beamsim2.core.sphere import make_observation_grid

    assert make_observation_grid("lebedev", 26).unit_vectors.shape[0] == 26
    # icosphere: smallest subdivision with >= the target count.
    assert make_observation_grid("icosphere", 2562).unit_vectors.shape[0] == 2562
    assert make_observation_grid("icosphere", 600).unit_vectors.shape[0] == 642
    assert make_observation_grid("icosphere", 1).unit_vectors.shape[0] == 12
    with pytest.raises(ValueError, match="Unknown sphere scheme"):
        make_observation_grid("nope", 26)
