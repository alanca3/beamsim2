"""V-4: radiated-power and directivity-index conservation checks using quadrature-weighted sphere integration."""

from __future__ import annotations

import numpy as np


def directivity_index(H_bem: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Compute directivity index (DI) from BEM pressure using Lebedev quadrature.

    DI is defined as the on-axis intensity relative to the isotropic mean:
    ``DI_dB = 10 * log10(max_intensity / mean_intensity)``.

    Anchors: 0 dB for an omnidirectional (monopole) source; ~4.77 dB for a
    cosine dipole (cos²θ radiation pattern); ~10 dB for a narrowly focused beam.

    VERIFIED: standard acoustic directivity definition (Benesty et al.,
    Microphone Array Signal Processing, §2.3, 2008).

    Parameters
    ----------
    H_bem : np.ndarray, shape [F, N], complex128
        Complex pressure field at N observation points for F frequencies.
    weights : np.ndarray, shape [N], float64
        Lebedev quadrature weights, sum_4pi convention (sum = 4π).

    Returns
    -------
    np.ndarray, shape [F], float64
        Directivity index in dB for each frequency.
    """
    intensity = np.abs(H_bem) ** 2  # [F, N] — intensity ∝ |p|²
    mean_intensity = np.sum(weights * intensity, axis=1) / (4.0 * np.pi)  # [F]
    max_intensity = np.max(intensity, axis=1)  # [F]
    return 10.0 * np.log10(max_intensity / mean_intensity)  # [F] dB
