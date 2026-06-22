"""Display-only field referencing for the Results views (Chunk 2).

A loudspeaker's directivity is conventionally a **far-field** quantity, but the BEM
solve stores the near-field complex pressure on a finite observation sphere (radius
``r_obs``) with phase referenced to the **global origin** (DATA_CONTRACT.md §3.4).  At
low frequency a single offset driver therefore reads *not-quite-omni*: the geometric
``1/r`` spreading and path-length phase vary across directions because the source sits
off the origin.  This module offers two **display transforms** that re-reference the
field so a low-frequency single driver reads near-omni, exactly as a far-field /
acoustic-center measurement would.

**Cardinal rule (DATA_CONTRACT.md §3.4): these transforms NEVER mutate or re-zero the
stored ``H``-tensor.**  They take a field array and return a *new* array for plotting
only.  The on-disk / in-memory near-field tensor that Phase-2 beamforming consumes is
left byte-for-byte untouched — the inter-driver phase that steers a beam is sacred.

Two modes (the user selected "both, user-selectable"):

* **Acoustic-center referencing** — remove each driver's geometric ``1/r`` spreading and
  path-length phase about its own position (acoustic center).  Simple, convention-safe.
* **SH far-field extrapolation** — fit spherical harmonics at ``r_obs`` and extrapolate
  to ``r → ∞`` via outgoing spherical-Hankel ratios.  The rigorous radiating far-field
  pattern (captures the radiating part of box diffraction).

NumCalc time convention (CLAUDE.md): engineering ``exp(−jωt)``, outgoing waves
``~ exp(+jkr)``, so the outgoing radial eigenfunction is ``h_l^(1)`` and
``h_l^(1)(x) ~ (−j)^{l+1} e^{+jx}/x`` as ``x → ∞``.

References
----------
DATA_CONTRACT.md §3.4 (single-phase-origin rule).  Williams, *Fourier Acoustics*,
Academic Press, 1999, §6 (spherical-wave expansion, outgoing Hankel functions).
Trott 1977 / VituixCAD dual-channel note (acoustic-center referencing).
"""

from __future__ import annotations

import numpy as np
from scipy.special import spherical_jn, spherical_yn

from beamsim2.core.logging_setup import get_logger
from beamsim2.core.sh_transform import (
    forward_sh,
    inverse_sh,
    n_coeffs,
    safe_order_for_grid,
)
from beamsim2.core.types import ObservationPoints

logger = get_logger(__name__)

NEAR_FIELD = "Near-field (as solved)"
FAR_ACOUSTIC_CENTER = "Far-field: acoustic-center"
FAR_SH_EXTRAPOLATION = "Far-field: SH extrapolation"

#: Selectable referencing modes, in display order (first = default).
REFERENCING_MODES: list[str] = [NEAR_FIELD, FAR_ACOUSTIC_CENTER, FAR_SH_EXTRAPOLATION]

_DEFAULT_C = 343.2  # m/s — fallback speed of sound when the dataset omits it


def _as_2d(field: np.ndarray) -> tuple[np.ndarray, bool]:
    """Return ``field`` as ``[F, N]`` plus a flag marking an original ``[N]`` input."""
    arr = np.asarray(field, dtype=np.complex128)
    if arr.ndim == 1:
        return arr[None, :], True
    return arr, False


def acoustic_center_field(
    field: np.ndarray,
    frequencies: np.ndarray,
    obs: ObservationPoints,
    position: np.ndarray,
    *,
    c: float = _DEFAULT_C,
) -> np.ndarray:
    """Re-reference a driver's field to its acoustic center (display transform).

    For a source at ``position`` ``p``, the observation point in direction ``û_n`` sits
    at ``R_n = r_obs · û_n`` (global frame), a distance ``r_n = |R_n − p|`` from the
    source.  The transform removes the geometric spreading and path-length phase about
    ``p`` and re-references to the nominal radius ``r_obs``::

        H_ac[f, n] = H[f, n] · (r_n / r_obs) · exp(−j · k_f · (r_n − r_obs))

    where ``k_f = 2π f / c``.  For an ideal offset monopole
    ``H = A·exp(+j k r_n)/r_n`` this collapses to the direction-independent
    ``A·exp(+j k r_obs)/r_obs`` — i.e. **exactly omni** — which is the acceptance test.

    Parameters
    ----------
    field : np.ndarray
        ``[F, N]`` or ``[N]`` complex128 near-field pressure on ``obs``.
    frequencies : np.ndarray
        ``[F]`` float64, Hz.
    obs : ObservationPoints
        The observation grid (provides ``unit_vectors`` and ``radius`` = ``r_obs``).
    position : np.ndarray
        ``[3]`` driver position / acoustic center in metres (global frame).
    c : float
        Speed of sound (m/s).

    Returns
    -------
    np.ndarray
        Same shape as ``field`` — the re-referenced pressure (a NEW array).
    """
    H, was_1d = _as_2d(field)  # [F, N]
    r_obs = float(obs.radius)
    p = np.asarray(position, dtype=np.float64).reshape(3)
    R = obs.unit_vectors * r_obs  # [N, 3] observation coordinates
    r_n = np.linalg.norm(R - p[None, :], axis=1)  # [N] source→obs distance
    k = 2.0 * np.pi * np.asarray(frequencies, dtype=np.float64) / c  # [F]

    spread = (r_n / r_obs)[None, :]  # [1, N] — undo 1/r amplitude variation
    phase = np.exp(-1j * k[:, None] * (r_n - r_obs)[None, :])  # [F, N] — undo path-length phase
    out = H * spread * phase  # [F, N]
    return out[0] if was_1d else out


def _l_index_for_order(order: int) -> np.ndarray:
    """Degree ``l`` for each flattened SH coefficient (``l=0..order, m=−l..l``)."""
    return np.concatenate([np.full(2 * li + 1, li) for li in range(order + 1)])  # [K]


def _spherical_hankel1(li: int, x: np.ndarray) -> np.ndarray:
    """Outgoing spherical Hankel function ``h_l^(1)(x) = j_l(x) + i·y_l(x)``.

    In the engineering ``exp(−jωt)`` convention used here (``j`` ≡ ``i``), ``h_l^(1)`` is
    the *outgoing* radial eigenfunction (``~ exp(+jx)`` at large ``x``).
    """
    return spherical_jn(li, x) + 1j * spherical_yn(li, x)


def farfield_extrapolated_field(
    field: np.ndarray,
    frequencies: np.ndarray,
    obs: ObservationPoints,
    *,
    c: float = _DEFAULT_C,
    order: int | None = None,
) -> np.ndarray:
    """Extrapolate a field at ``r_obs`` to the radiating far field via SH (display only).

    The exterior pressure expands in outgoing spherical waves
    ``p(r,Ω) = Σ_lm c_lm · h_l^(1)(kr) · Y_lm(Ω)``.  Fitting SH at ``r_obs`` gives
    ``a_lm = c_lm · h_l^(1)(k r_obs)``; the far-field directivity coefficients are::

        b_lm = a_lm · (−j)^(l+1) / (k · h_l^(1)(k r_obs))

    (from ``h_l^(1)(kr) → (−j)^(l+1) e^{+jkr}/(kr)`` as ``r → ∞``).  ``b_lm`` is then
    evaluated back on the original grid so downstream resampling is unchanged.  For an
    offset monopole the far-field pattern is a pure phase ramp ``e^{−j k (p·û)}`` —
    **omni magnitude** — the acceptance test.

    Numerical note: ``|h_l^(1)(k r_obs)|`` grows with ``l`` (the ``y_l`` term), so the
    division **suppresses** high-``l`` modes — inherently stable.  A tiny floor guards
    the (here unreachable) zero-magnitude case.

    Parameters
    ----------
    field : np.ndarray
        ``[F, N]`` or ``[N]`` complex128 near-field pressure on ``obs``.
    frequencies : np.ndarray
        ``[F]`` float64, Hz.
    obs : ObservationPoints
        The observation grid (provides ``theta_phi``, ``unit_vectors``, ``radius``).
    c : float
        Speed of sound (m/s).
    order : int | None
        SH order for the fit.  Default ``min(safe_order_for_grid(N), 16)``.

    Returns
    -------
    np.ndarray
        Same shape as ``field`` — the far-field-referenced pressure (a NEW array).
    """
    H, was_1d = _as_2d(field)  # [F, N]
    n = obs.unit_vectors.shape[0]
    L = order if order is not None else min(safe_order_for_grid(n), 16)
    if obs.theta_phi is None:
        raise ValueError("farfield_extrapolated_field: ObservationPoints needs theta_phi.")
    theta, phi = obs.theta_phi[:, 0], obs.theta_phi[:, 1]

    a = forward_sh(H, obs, L)  # [F, K] complex — modal coeffs at r_obs
    if a.ndim == 1:
        a = a[None, :]
    k = 2.0 * np.pi * np.asarray(frequencies, dtype=np.float64) / c  # [F]
    r_obs = float(obs.radius)
    l_of_k = _l_index_for_order(L)  # [K]

    # Per-(frequency, coefficient) far-field radial filter G[f, k] = (−j)^(l+1) / (k h_l1(k r_obs)).
    G = np.empty((len(k), n_coeffs(L)), dtype=np.complex128)  # [F, K]
    for li in range(L + 1):
        h = _spherical_hankel1(li, k * r_obs)  # [F]
        # Guard the (unreachable for k r_obs in range) near-zero magnitude case.
        h = np.where(np.abs(h) < 1e-300, 1e-300, h)
        g_l = ((-1j) ** (li + 1)) / (k * h)  # [F]
        G[:, l_of_k == li] = g_l[:, None]
    # Reference the far-field DIRECTIVITY back to pressure-at-r_obs so all three display
    # modes (near-field / acoustic-center / SH) share ONE absolute level — otherwise the
    # raw directivity coefficient reads 20·log10(r_obs) hotter (a spurious ~6 dB jump at
    # r_obs=2 m) on the absolute-SPL views when the combo is toggled.  This per-frequency,
    # direction-independent scale preserves the directivity shape; for an origin monopole
    # it makes the SH output identical to acoustic-center (|·|=1/r_obs, phase exp(+jk r_obs)).
    ref_scale = (np.exp(1j * k * r_obs) / r_obs)[:, None]  # [F, 1]
    b = a * G * ref_scale  # [F, K] far-field directivity coeffs, referenced to r_obs
    out = inverse_sh(b, theta, phi)  # [F, N] — evaluate back on the grid
    out = np.atleast_2d(out)
    return out[0] if was_1d else out


def referencing_mode_for(label: str) -> str:
    """Normalise a UI label to a known mode (falls back to near-field)."""
    return label if label in REFERENCING_MODES else NEAR_FIELD


def apply_referencing(
    field: np.ndarray,
    mode: str,
    *,
    frequencies: np.ndarray,
    obs: ObservationPoints,
    position: np.ndarray | None = None,
    c: float = _DEFAULT_C,
    order: int | None = None,
) -> np.ndarray:
    """Dispatch a field through the selected display referencing transform.

    Parameters
    ----------
    field : np.ndarray
        ``[F, N]`` or ``[N]`` complex128 near-field pressure.
    mode : str
        One of :data:`REFERENCING_MODES`.  Unknown values → near-field (identity).
    frequencies : np.ndarray
        ``[F]`` float64, Hz.
    obs : ObservationPoints
        The observation grid.
    position : np.ndarray | None
        Driver position (required for acoustic-center; ``None`` → origin = identity).
    c : float
        Speed of sound (m/s).
    order : int | None
        SH order override for the far-field extrapolation.

    Returns
    -------
    np.ndarray
        The transformed field (a NEW array; the input is never modified).
    """
    mode = referencing_mode_for(mode)
    if mode == NEAR_FIELD:
        return np.array(field, dtype=np.complex128)
    if mode == FAR_ACOUSTIC_CENTER:
        if position is None:
            return np.array(field, dtype=np.complex128)
        return acoustic_center_field(field, frequencies, obs, position, c=c)
    return farfield_extrapolated_field(field, frequencies, obs, c=c, order=order)


def _self_test() -> None:
    """Offset monopole → near-omni under both far-field modes; stored field untouched."""
    from beamsim2.core.sphere import icosphere
    from beamsim2.validation.closed_loop import monopole_field

    obs = icosphere(3, radius=2.0)  # 642 points
    freqs = np.array([100.0, 300.0])
    p = np.array([0.10, 0.0, 0.0])  # 10 cm offset from origin
    H = monopole_field(p[None, :], obs, freqs, c=_DEFAULT_C)[0]  # [F, N]
    H_orig = H.copy()

    def omni_ripple_db(field: np.ndarray) -> float:
        mag = np.abs(field)
        return float(20.0 * np.log10(mag.max(axis=1) / mag.min(axis=1)).max())

    near = omni_ripple_db(H)
    ac = omni_ripple_db(acoustic_center_field(H, freqs, obs, p))
    ff = omni_ripple_db(farfield_extrapolated_field(H, freqs, obs))
    assert ac < 1e-6, f"acoustic-center monopole not omni: {ac:.3f} dB ripple"
    assert ff < 0.2, f"far-field monopole not near-omni: {ff:.3f} dB ripple"
    assert np.array_equal(H, H_orig), "referencing mutated the input field (cardinal-rule break)"

    # All three modes must share one absolute level: an origin monopole's mean magnitude
    # is identical across near / acoustic-center / SH (no spurious ~6 dB jump on dB-SPL views).
    H0 = monopole_field(np.zeros((1, 3)), obs, freqs, c=_DEFAULT_C)[0]
    levels = {
        "near": np.abs(H0).mean(),
        "ac": np.abs(acoustic_center_field(H0, freqs, obs, np.zeros(3))).mean(),
        "ff": np.abs(farfield_extrapolated_field(H0, freqs, obs)).mean(),
    }
    assert max(levels.values()) - min(levels.values()) < 1e-6, f"modes disagree on level: {levels}"
    print(
        f"core/field_referencing.py self-test: PASS (near={near:.2f} dB → ac={ac:.1e}, "
        f"ff={ff:.1e}; level={levels['near']:.4f})"
    )


if __name__ == "__main__":
    _self_test()
