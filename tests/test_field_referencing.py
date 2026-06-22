"""Display-only field referencing (Chunk 2): far-field modes + cardinal-rule guard.

Both far-field transforms must turn an offset monopole's near-field pattern (which reads
not-quite-omni from 1/r spreading) into a near-omni far-field pattern, and NEITHER may
ever mutate the stored field (DATA_CONTRACT.md §3.4 single-phase-origin rule).
"""

from __future__ import annotations

import numpy as np

from beamsim2.core.field_referencing import (
    FAR_ACOUSTIC_CENTER,
    FAR_SH_EXTRAPOLATION,
    NEAR_FIELD,
    REFERENCING_MODES,
    acoustic_center_field,
    apply_referencing,
    farfield_extrapolated_field,
)
from beamsim2.core.sphere import icosphere
from beamsim2.validation.closed_loop import monopole_field

_C = 343.2


def _offset_monopole():
    """Offset-monopole near field on a dense grid: [F, N], plus (obs, freqs, position)."""
    obs = icosphere(3, radius=2.0)  # 642 points
    freqs = np.array([80.0, 250.0, 800.0])
    pos = np.array([0.12, 0.0, 0.0])  # 12 cm off the origin
    H = monopole_field(pos[None, :], obs, freqs, c=_C)[0]  # [F, N]
    return H, obs, freqs, pos


def _omni_ripple_db(field: np.ndarray) -> float:
    """Max over frequency of the per-frequency max/min magnitude ratio, in dB (0 = omni)."""
    mag = np.abs(field)
    return float(np.max(20.0 * np.log10(mag.max(axis=1) / mag.min(axis=1))))


def test_modes_listed_in_order():
    assert REFERENCING_MODES[0] == NEAR_FIELD
    assert FAR_ACOUSTIC_CENTER in REFERENCING_MODES
    assert FAR_SH_EXTRAPOLATION in REFERENCING_MODES


def test_near_field_is_identity():
    H, obs, freqs, pos = _offset_monopole()
    out = apply_referencing(H, NEAR_FIELD, frequencies=freqs, obs=obs, position=pos)
    np.testing.assert_array_equal(out, H)


def test_acoustic_center_makes_monopole_omni():
    """Acoustic-center referencing collapses an offset monopole to exactly omni."""
    H, obs, freqs, pos = _offset_monopole()
    assert _omni_ripple_db(H) > 0.3, "near-field offset monopole should NOT be omni"
    out = acoustic_center_field(H, freqs, obs, pos, c=_C)
    assert _omni_ripple_db(out) < 1e-6, "acoustic-center monopole must be omni"


def test_sh_extrapolation_makes_monopole_near_omni():
    """SH far-field extrapolation makes an offset monopole near-omni."""
    H, obs, freqs, pos = _offset_monopole()
    out = farfield_extrapolated_field(H, freqs, obs, c=_C)
    assert _omni_ripple_db(out) < 0.25, "SH-extrapolated monopole must be near-omni"


def test_referencing_never_mutates_input():
    """The cardinal rule: the stored field is byte-for-byte unchanged after referencing."""
    H, obs, freqs, pos = _offset_monopole()
    for mode in REFERENCING_MODES:
        H_ref = H.copy()
        apply_referencing(H, mode, frequencies=freqs, obs=obs, position=pos, c=_C)
        np.testing.assert_array_equal(H, H_ref, err_msg=f"{mode} mutated the input field")


def test_acoustic_center_without_position_is_identity():
    H, obs, freqs, _ = _offset_monopole()
    out = apply_referencing(H, FAR_ACOUSTIC_CENTER, frequencies=freqs, obs=obs, position=None)
    np.testing.assert_array_equal(out, H)


def test_acoustic_center_origin_monopole_is_identity():
    """A monopole AT the origin has r_n == r_obs ∀n, so acoustic-center is a no-op."""
    obs = icosphere(3, radius=2.0)
    freqs = np.array([80.0, 250.0, 800.0])
    H = monopole_field(np.zeros((1, 3)), obs, freqs, c=_C)[0]  # [F, N], already omni
    assert _omni_ripple_db(H) < 1e-9
    out = acoustic_center_field(H, freqs, obs, np.zeros(3), c=_C)
    np.testing.assert_allclose(out, H, rtol=1e-12, atol=1e-12)


def test_far_field_modes_share_absolute_level():
    """All three modes agree on absolute level for an origin monopole (no spurious dB jump).

    Regression for the review finding that the SH directivity coefficient read
    20·log10(r_obs) (≈6 dB at r_obs=2 m) hotter than near-field/acoustic-center on the
    absolute-SPL views — a level shift the ripple-only tests cannot see.
    """
    obs = icosphere(3, radius=2.0)
    freqs = np.array([100.0, 400.0])
    H0 = monopole_field(np.zeros((1, 3)), obs, freqs, c=_C)[0]  # [F, N], at the origin
    levels = {
        m: float(
            np.abs(
                apply_referencing(H0, m, frequencies=freqs, obs=obs, position=np.zeros(3), c=_C)
            ).mean()
        )
        for m in REFERENCING_MODES
    }
    spread = max(levels.values()) - min(levels.values())
    assert spread < 1e-6, f"modes disagree on absolute level: {levels}"


def test_sh_extrapolation_matches_analytic_far_field_phase():
    """SH extrapolation matches the analytic monopole far field in MAGNITUDE AND PHASE.

    The analytic far field of an offset monopole is ``exp(−jk p·û)`` (omni magnitude with a
    phase ramp).  Comparing up to one global complex constant per frequency pins the
    load-bearing ``(−j)^(l+1)/(k·h_l^(1))`` convention — which the omni-magnitude check alone
    cannot (a sign flip stays omni but corrupts the phase).
    """
    H, obs, freqs, pos = _offset_monopole()
    out = farfield_extrapolated_field(H, freqs, obs, c=_C)  # [F, N]
    k = 2.0 * np.pi * freqs / _C
    ff = np.exp(-1j * k[:, None] * (obs.unit_vectors @ pos)[None, :])  # [F, N] analytic, |·|=1
    ratio = out / ff  # should be a per-frequency complex constant if the math is right
    ratio = ratio / ratio[:, :1]  # normalise by the first direction
    assert np.max(np.abs(np.angle(ratio))) < 1e-3, "SH far-field phase does not match analytic"
    assert float(np.ptp(np.abs(ratio), axis=1).max()) < 1e-3, "SH far-field magnitude not flat"


def test_single_frequency_1d_input_shapes():
    """A [N] (single-frequency) field round-trips to [N]."""
    H, obs, freqs, pos = _offset_monopole()
    out_ac = acoustic_center_field(H[0], freqs[:1], obs, pos, c=_C)
    out_ff = farfield_extrapolated_field(H[0], freqs[:1], obs, c=_C)
    assert out_ac.shape == (obs.unit_vectors.shape[0],)
    assert out_ff.shape == (obs.unit_vectors.shape[0],)
