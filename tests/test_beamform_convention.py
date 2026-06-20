"""V-RT — sign-convention round-trip steering test (Stage P2-0a, Phase-2 gate).

Pins the house convention (``docs/Phase 2 - Filter Solver.md`` DR-P2-02): the coded
forward model is ``P(f, dir) = sum_m w_m(f) * H[m, f, dir]`` with look vector
``c = conj(H[:, f, look])``. The matched-field (phase-conjugate) beamformer
``w = conj(H_look)/M`` must steer the main lobe to the *commanded* direction.

The discriminating control is **bug injection**: the non-conjugated weight
``w = H_look/M`` (the microphone-array convention copied verbatim) must steer the lobe to
the *mirror* direction, not the commanded one. If both conventions passed, the test would
be worthless — so we assert the conjugated form points at the look and the non-conjugated
form points away from it.

This guards DR-P2-02 and, because it depends entirely on the inter-driver phase carried in
``H``, the single-phase-origin cardinal rule (``DATA_CONTRACT.md`` §3.4). It runs at
Lebedev-26 (no dense grid needed) because a first-order pattern is exactly resolved there.
"""

from __future__ import annotations

import numpy as np
import pytest

from beamsim2.beamform.covariance import covariance, directivity_factor, look_vector
from beamsim2.beamform.weights import matched_field
from beamsim2.core.sphere import lebedev
from beamsim2.validation.closed_loop import monopole_field

_C = 343.2


def _nearest_idx(unit_vectors: np.ndarray, direction: np.ndarray) -> int:
    """Index of the grid point whose unit vector is closest to ``direction``."""
    d = direction / np.linalg.norm(direction)
    return int(np.argmax(unit_vectors @ d))


def _endfire_setup():
    """Two monopoles on the z-axis (an end-fire pair) on a Lebedev-26 sphere."""
    obs = lebedev(26, radius=1.0)
    positions = np.array([[0.0, 0.0, -0.12], [0.0, 0.0, 0.12]], float)  # [M,3] along z
    freqs = np.array([1000.0, 2000.0, 4000.0])  # [F]
    H = monopole_field(positions, obs, freqs, c=_C)  # [M,F,N]
    return obs, positions, freqs, H


def test_covariance_hermitian_and_psd():
    """R = conj(H) diag(a) H^T is Hermitian and positive semidefinite."""
    obs, _, _, H = _endfire_setup()
    H_f = H[:, 0, :]  # [M,N]
    R = covariance(H_f, obs.weights)
    assert np.max(np.abs(R - R.conj().T)) < 1e-9, "covariance not Hermitian"
    assert np.min(np.linalg.eigvalsh(R)) > -1e-9, "covariance not PSD"


def test_matched_field_maximizes_white_noise_gain():
    """Matched-field is the max-WNG distortionless beamformer (the robustness corner).

    Closed form: WNG(matched) = ||c||^2 = sum_m |H_m,look|^2 (which equals M only for a
    unit-magnitude steering vector). Max property: w = c/M is parallel to c, so any
    distortionless perturbation u with c^H u = 0 only adds orthogonal norm and *lowers* WNG.
    """
    obs, _, _, H = _endfire_setup()
    H_f = H[:, 0, :]
    look = _nearest_idx(obs.unit_vectors, np.array([0.0, 0.0, 1.0]))
    w = matched_field(H_f, look)
    c = look_vector(H_f, look)

    def wng(ww: np.ndarray) -> float:
        return float(np.abs(np.conj(c) @ ww) ** 2 / np.real(np.conj(ww) @ ww))

    # Closed form: WNG = ||c||^2.
    assert wng(w) == pytest.approx(float(np.real(np.conj(c) @ c)), rel=1e-9)

    # Max-WNG property: any null-space (distortionless) perturbation reduces WNG.
    rng = np.random.default_rng(0)
    u = rng.standard_normal(c.shape) + 1j * rng.standard_normal(c.shape)
    u -= (np.conj(c) @ u) / (np.conj(c) @ c) * c  # project out the c-component => c^H u = 0
    assert wng(w + 0.5 * u) < wng(w)


@pytest.mark.parametrize("steer_z", [+1.0, -1.0])
def test_matched_field_steers_to_look(steer_z):
    """Conjugated matched-field steers the end-fire main lobe to +z or -z (the look)."""
    obs, _, freqs, H = _endfire_setup()
    look = _nearest_idx(obs.unit_vectors, np.array([0.0, 0.0, steer_z]))
    mirror = _nearest_idx(obs.unit_vectors, np.array([0.0, 0.0, -steer_z]))
    for fi in range(len(freqs)):
        H_f = H[:, fi, :]
        w = matched_field(H_f, look)  # conj(H_look)/M
        P = np.sum(w[:, None] * H_f, axis=0)  # [N]
        peak = int(np.argmax(np.abs(P)))
        assert peak == look, f"f={freqs[fi]:.0f}Hz: main lobe at {peak}, expected look {look}"
        # And the look is strictly louder than the mirror.
        assert np.abs(P[look]) > np.abs(P[mirror]) + 1e-9


@pytest.mark.parametrize("steer_z", [+1.0, -1.0])
def test_wrong_convention_mirror_steers(steer_z):
    """BUG-INJECTION CONTROL: the non-conjugated weight steers to the MIRROR, not the look.

    This proves the convention test discriminates: if H were stored in the conjugate
    convention (or we forgot the conj), the beam would silently point the wrong way.
    """
    obs, _, freqs, H = _endfire_setup()
    look = _nearest_idx(obs.unit_vectors, np.array([0.0, 0.0, steer_z]))
    mirror = _nearest_idx(obs.unit_vectors, np.array([0.0, 0.0, -steer_z]))
    M = H.shape[0]
    for fi in range(len(freqs)):
        H_f = H[:, fi, :]
        w_bug = H_f[:, look] / M  # NO conjugation — the wrong (mic-array) convention
        P = np.sum(w_bug[:, None] * H_f, axis=0)
        peak = int(np.argmax(np.abs(P)))
        assert peak == mirror, (
            f"f={freqs[fi]:.0f}Hz: non-conjugated weight peaked at {peak}; "
            f"a mirror-steer should land at {mirror} (look was {look})"
        )
        assert peak != look


def test_matched_field_peaks_at_look_general_3d_array():
    """Matched-field peaks at the commanded direction for a compact non-collinear array."""
    obs = lebedev(26, radius=1.0)
    # Four monopoles in a compact tetrahedral-ish cloud (non-degenerate in 3-D).
    positions = 0.08 * np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float)  # [M,3]
    freqs = np.array([1500.0, 3000.0])
    H = monopole_field(positions, obs, freqs, c=_C)  # [M,F,N]
    # Steer toward several distinct grid directions; matched-field must peak there.
    look_dirs = [
        np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([1.0, 1.0, 1.0]),
    ]
    for d in look_dirs:
        look = _nearest_idx(obs.unit_vectors, d)
        for fi in range(len(freqs)):
            H_f = H[:, fi, :]
            w = matched_field(H_f, look)
            P = np.sum(w[:, None] * H_f, axis=0)
            peak = int(np.argmax(np.abs(P)))
            assert (
                peak == look
            ), f"dir={d.tolist()} f={freqs[fi]:.0f}Hz: peak at {peak}, expected {look}"


def test_directivity_factor_matches_manual():
    """directivity_factor(w,A,R) equals the manual Rayleigh quotient."""
    obs, _, _, H = _endfire_setup()
    H_f = H[:, 0, :]
    look = _nearest_idx(obs.unit_vectors, np.array([0.0, 0.0, 1.0]))
    # Accept = a tight cap around the look; reject = whole sphere.
    cos_look = obs.unit_vectors @ obs.unit_vectors[look]
    accept = (cos_look > 0.9).astype(float)
    A = covariance(H_f, obs.weights, mask=accept)
    R = covariance(H_f, obs.weights)
    w = matched_field(H_f, look)
    df = directivity_factor(w, A, R)
    manual = float(np.real(np.conj(w) @ A @ w) / np.real(np.conj(w) @ R @ w))
    assert df == pytest.approx(manual, rel=1e-12)
