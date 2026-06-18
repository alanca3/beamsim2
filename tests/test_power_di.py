"""V-4: radiated-power and directivity-index conservation checks using quadrature-weighted sphere integration over the BEM field."""

from __future__ import annotations

import math

import numpy as np

from beamsim2.core.sphere import lebedev
from beamsim2.validation.power_di import directivity_index

# Lebedev-26 integrates polynomials exactly up to degree 7.
# cos²θ is degree 2, so the dipole test below is exact to floating-point precision.
_OBS = lebedev(26)
_WEIGHTS = _OBS.weights  # [26] float64, sum = 4π
_N = len(_WEIGHTS)


def _const_field(n_freq: int = 2) -> np.ndarray:
    """Return a flat [F, N] complex field (omnidirectional, DI = 0 dB)."""
    return np.ones((n_freq, _N), dtype=np.complex128)


def _dipole_field(n_freq: int = 2) -> np.ndarray:
    """Return a [F, N] field with |p|² = cos²θ = z².

    Lebedev-26 integrates ∫cos²θ dΩ = 4π/3 exactly.
    max(cos²θ) = 1 at the (0,0,1) axis point.
    → DI = 10·log10(1 / (1/3)) = 10·log10(3) ≈ 4.771 dB.
    """
    z = _OBS.unit_vectors[:, 2]  # [N] — z-component = cos θ
    p_row = z.astype(np.complex128)  # imaginary part zero → |p|² = z²
    return np.tile(p_row, (n_freq, 1))


# ---------------------------------------------------------------------------
# V-4 tests — no @local_only; run without the NumCalc binary
# ---------------------------------------------------------------------------


def test_monopole_di_zero() -> None:
    """Omnidirectional (monopole) source must give DI = 0 dB."""
    H = _const_field()
    di = directivity_index(H, _WEIGHTS)
    np.testing.assert_allclose(di, 0.0, atol=0.1, err_msg="Monopole DI must be 0 ± 0.1 dB")


def test_dipole_di_exact() -> None:
    """cos²θ dipole must give DI = 10·log10(3) ≈ 4.771 dB on Lebedev-26.

    Lebedev-26 integrates degree-7 polynomials exactly, so cos²θ (degree 2)
    is exact to floating point. This replaces a naive half-space step-function
    test that the discrete grid cannot integrate accurately (~1.7 dB error due
    to equatorial quadrature points carrying finite weight on both sides).
    """
    H = _dipole_field()
    di = directivity_index(H, _WEIGHTS)
    expected = 10.0 * math.log10(3.0)  # ≈ 4.771 dB
    np.testing.assert_allclose(
        di, expected, atol=0.05, err_msg=f"Dipole DI must be {expected:.3f} ± 0.05 dB"
    )


def test_di_frequency_independent() -> None:
    """Scaling all pressures uniformly must not change DI (intensity cancels)."""
    H_base = _dipole_field(n_freq=4)
    scales = np.array([0.5, 1.0, 2.0, 10.0])
    H_scaled = H_base * scales[:, np.newaxis]  # [4, N] — different amplitude per freq
    di_base = directivity_index(H_base, _WEIGHTS)
    di_scaled = directivity_index(H_scaled, _WEIGHTS)
    np.testing.assert_allclose(
        di_scaled,
        di_base,
        atol=1e-10,
        err_msg="DI must be independent of pressure amplitude",
    )


def test_radiated_power_positive_finite() -> None:
    """Weighted power integral Σ wᵢ|pᵢ|² must be positive and finite for any non-zero field."""
    for H in (_const_field(), _dipole_field()):
        intensity = np.abs(H) ** 2  # [F, N]
        power = np.sum(_WEIGHTS * intensity, axis=1)  # [F]
        assert np.all(power > 0), "Radiated power must be positive"
        assert np.all(np.isfinite(power)), "Radiated power must be finite"
