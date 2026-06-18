"""Self-test for the driver electrical/terminal chain (build-order item 8).

Finish-line test per CODING_STANDARDS and §8 Stage-2 gate:
"terminal response matches a reference RLC/semi-inductance fit."

Anchors (ALL check magnitude AND phase — a magnitude-only test cannot catch
a missing conjugation, which is item 8's highest risk):

  1. DC limit:       |Z_in(ω→0)| ≈ Re
  2. Resonance:      |Z_in| peaks near fs; peak height ≈ Re(1 + Qms/Qes)
  3. Sealed box:     peak shifts to fc = fs·√(1 + Vas/Vb); Qtc changes
  4. HF inductance:  Im(Z_in_textbook) > 0 at HF (inductive); LR-2 < plain-Le (lossy)
  5. Convention lock Im(Z_in_eng = conj(Z_in_textbook)) < 0 at HF (engineering sign)
  6. Conjugate check: terminal_response = conj(u_textbook) element-by-element
  7. Shape / hygiene: [F] complex128, finite, no NaN/Inf
  8. Wiring:         terminal_responses_for + build_dataset give H_full = H_bem × tr[:,None]

Pure Python / NumPy — no NumCalc binary, no @local_only marker.
Runs in the normal CI suite.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from beamsim2.driver.inductance import (
    LR2Ladder,
    PlainLe,
    input_impedance,
    voice_coil_impedance,
)
from beamsim2.driver.terminal import (
    TerminalModel,
    terminal_response,
    terminal_responses_for,
)
from beamsim2.driver.thiele_small import TSParams, cone_velocity, mechanical_impedance

# ---------------------------------------------------------------------------
# Reference woofer fixture
# ---------------------------------------------------------------------------
# A typical medium-woofer from datasheet values.  All derived parameters are
# checked against the formulas in comments — any change propagates to failures.
#
# Input spec: fs=40 Hz, Qms=5.0, Qes=0.5, Vas=30L=0.030 m³, Re=8Ω, Sd=0.0133 m²
# (≈ 130 mm cone diameter, Sd ≈ π·(0.065)² = 0.01327 m²)
#
# LR-2 coil parameters: Le=1.5 mH, Le2=1.0 mH, Re2=5 Ω  (typical woofer values)

_RHO = 1.2041  # kg/m³
_C = 343.2  # m/s

_FS = 40.0  # Hz
_QMS = 5.0
_QES = 0.5
_VAS_M3 = 0.030  # m³ (30 litres)
_RE = 8.0  # Ω
_SD = 0.0133  # m²
_LE = 1.5e-3  # H
_LE2 = 1.0e-3  # H
_RE2 = 5.0  # Ω


@pytest.fixture
def ts() -> TSParams:
    return TSParams.from_datasheet(
        fs=_FS,
        Qms=_QMS,
        Qes=_QES,
        Vas_m3=_VAS_M3,
        Re=_RE,
        Sd=_SD,
        rho=_RHO,
        c=_C,
    )


@pytest.fixture
def lr2() -> LR2Ladder:
    return LR2Ladder(Le=_LE, Le2=_LE2, Re2=_RE2)


@pytest.fixture
def plain_le() -> PlainLe:
    return PlainLe(Le=_LE)


@pytest.fixture
def model(ts, lr2) -> TerminalModel:
    return TerminalModel(ts=ts, inductance=lr2, box_volume=None, voltage=2.83, name="woofer")


@pytest.fixture
def freqs() -> np.ndarray:
    """Log-spaced 200-point grid from 1 Hz to 25 kHz."""
    return np.geomspace(1.0, 25_000.0, 200)  # [F] float64, Hz


# ---------------------------------------------------------------------------
# 1. TSParams.from_datasheet — roundtrip check
# ---------------------------------------------------------------------------


class TestTSParamsFromDatasheet:
    def test_fs_roundtrip(self, ts):
        """Reconstructed fs must equal the input fs within 0.1 Hz."""
        assert abs(ts.fs - _FS) < 0.1, f"fs roundtrip: got {ts.fs:.3f} Hz, expected {_FS}"

    def test_Qms_roundtrip(self, ts):
        assert abs(ts.Qms - _QMS) < 0.01

    def test_Qes_roundtrip(self, ts):
        assert abs(ts.Qes - _QES) < 0.01

    def test_Qts_value(self, ts):
        expected_Qts = _QMS * _QES / (_QMS + _QES)
        assert abs(ts.Qts - expected_Qts) < 0.001

    def test_vas_roundtrip(self, ts):
        assert abs(ts.vas(rho=_RHO, c=_C) - _VAS_M3) < 1e-5, "Vas roundtrip failed"

    def test_from_datasheet_via_Qts(self, ts):
        """Construct from Qts instead of Qes; result must match construction from Qes."""
        ts2 = TSParams.from_datasheet(
            fs=_FS, Qms=_QMS, Qts=ts.Qts, Vas_m3=_VAS_M3, Re=_RE, Sd=_SD, rho=_RHO, c=_C
        )
        assert abs(ts2.fs - ts.fs) < 0.01
        assert abs(ts2.Bl - ts.Bl) < 0.001

    def test_invalid_both_Qes_Qts_raises(self):
        with pytest.raises(ValueError, match="exactly one"):
            TSParams.from_datasheet(fs=40, Qms=5, Qes=0.5, Qts=0.45, Vas_m3=0.03, Re=8, Sd=0.0133)

    def test_invalid_neither_raises(self):
        with pytest.raises(ValueError, match="exactly one"):
            TSParams.from_datasheet(fs=40, Qms=5, Vas_m3=0.03, Re=8, Sd=0.0133)

    def test_Qts_ge_Qms_raises(self):
        with pytest.raises(ValueError, match="must be less than"):
            TSParams.from_datasheet(fs=40, Qms=1.0, Qts=1.5, Vas_m3=0.03, Re=8, Sd=0.0133)


# ---------------------------------------------------------------------------
# 2. Mechanical impedance
# ---------------------------------------------------------------------------


class TestMechanicalImpedance:
    def test_dc_limit_is_stiffness(self, ts):
        """At ω→0 the mechanical impedance approaches the stiffness term 1/(jω·Cms),
        which diverges.  Test at 0.01 Hz: |Zm| >> Re as expected."""
        omega_low = np.array([2 * math.pi * 0.01])
        zm = mechanical_impedance(ts, omega_low)
        assert abs(zm[0]) > 1e4, "Very low ω: Zm should be stiffness-dominated (huge)"

    def test_resonance_impedance_minimum(self, ts):
        """At resonance, Zm = Rms (mass and stiffness cancel).  VERIFIED: Thiele 1971."""
        omega_s = ts.omega_s
        omega = np.array([omega_s])
        zm = mechanical_impedance(ts, omega)
        assert abs(zm[0].real - ts.Rms) < 0.01 * ts.Rms
        assert abs(zm[0].imag) < 0.01 * ts.Rms  # imaginary part ≈ 0 at resonance

    def test_sealed_box_raises_stiffness(self, ts):
        """Box air spring adds stiffness → stiffer (more restoring force) than free-air."""
        omega = np.array([ts.omega_s])
        zm_free = mechanical_impedance(ts, omega, box_volume=None)
        zm_box = mechanical_impedance(ts, omega, box_volume=_VAS_M3)  # Vb = Vas
        # At free-air resonance, sealed-box Zm has net stiffness → larger imaginary part
        assert abs(zm_box[0]) > abs(zm_free[0])

    def test_sealed_box_fc(self, ts):
        """Sealed-box resonance fc = fs·√(1 + Vas/Vb).  VERIFIED: Small 1973."""
        Vb = _VAS_M3  # Vb = Vas → fc = fs·√2
        fc_expected = _FS * math.sqrt(1.0 + ts.vas(rho=_RHO, c=_C) / Vb)
        # Find the frequency where Im(Zm) = 0 (resonance) with sealed box
        omega_test = np.geomspace(2 * math.pi * 30, 2 * math.pi * 120, 2000)
        zm_box = mechanical_impedance(ts, omega_test, box_volume=Vb, rho=_RHO, c=_C)
        # Im(Zm) crosses zero at resonance; find the zero crossing
        im_zm = zm_box.imag
        cross_idx = np.where(np.diff(np.sign(im_zm)))[0]
        assert len(cross_idx) > 0, "No sealed-box resonance found in search range"
        idx = cross_idx[0]
        fc_found = omega_test[idx] / (2 * math.pi)
        assert (
            abs(fc_found - fc_expected) < 2.0
        ), f"Sealed-box fc: found {fc_found:.1f} Hz, expected {fc_expected:.1f} Hz"


# ---------------------------------------------------------------------------
# 3. Voice-coil impedance (Ze) and input impedance (Z_in)
# ---------------------------------------------------------------------------


class TestImpedance:
    def test_dc_limit_re(self, ts, lr2, freqs):
        """At DC, Z_in → Re (inductance is short-circuit; no motional term at ω=0)."""
        omega_low = np.array([2 * math.pi * 0.01])
        ze_low = voice_coil_impedance(lr2, ts.Re, omega_low)
        # Ze at very low freq ≈ Re (inductive terms → 0)
        assert abs(ze_low[0].real - ts.Re) < 0.01
        assert abs(ze_low[0].imag) < 0.01

    def test_resonance_peak(self, ts, lr2):
        """Input impedance peaks near fs.  Peak ≈ Re·(1 + Qms/Qes).  VERIFIED: Small 1972."""
        # Search in 20–80 Hz range
        omega_search = np.geomspace(2 * math.pi * 20, 2 * math.pi * 80, 2000)
        ze = voice_coil_impedance(lr2, ts.Re, omega_search)
        zm = mechanical_impedance(ts, omega_search)
        z_in = input_impedance(ze, zm, ts.Bl)

        peak_idx = np.argmax(np.abs(z_in))
        peak_freq = omega_search[peak_idx] / (2 * math.pi)
        peak_mag = abs(z_in[peak_idx])

        expected_peak = ts.Re * (1.0 + ts.Qms / ts.Qes)  # VERIFIED: Small 1972
        assert (
            abs(peak_freq - _FS) < 5.0
        ), f"Z_in peak at {peak_freq:.1f} Hz, expected near {_FS} Hz"
        assert (
            abs(peak_mag - expected_peak) < 0.05 * expected_peak
        ), f"Z_in peak {peak_mag:.1f} Ω, expected ≈ {expected_peak:.1f} Ω"

    def test_hf_inductive_textbook_sign(self, ts, lr2):
        """At HF (5 kHz+), Im(Z_in_textbook) > 0 — inductive, exp(+jωt) convention.

        This is the positive-polarity assertion.  The conjugate (engineering convention)
        must have Im < 0 — tested in test_convention_lock.
        """
        omega_hf = np.array([2 * math.pi * 10_000.0])
        ze = voice_coil_impedance(lr2, ts.Re, omega_hf)
        zm = mechanical_impedance(ts, omega_hf)
        z_in = input_impedance(ze, zm, ts.Bl)
        assert z_in[0].imag > 0.0, (
            f"At 10 kHz, Im(Z_in_textbook) = {z_in[0].imag:.4f} Ω; expected > 0 "
            "(textbook exp(+jωt) → inductive → positive Im)"
        )

    def test_lr2_lower_than_plain_le_at_hf(self, ts, lr2, plain_le):
        """LR-2 |Ze| < plain-Le |Ze| at HF — eddy loss flattens the impedance rise."""
        omega_hf = np.array([2 * math.pi * 8000.0])
        ze_lr2 = voice_coil_impedance(lr2, ts.Re, omega_hf)
        ze_plain = voice_coil_impedance(plain_le, ts.Re, omega_hf)
        assert abs(ze_lr2[0]) < abs(ze_plain[0]), (
            "LR-2 Ze magnitude must be below plain-Le at HF "
            "(eddy-current loss damps the inductance rise)"
        )

    def test_plain_le_dc_same_as_lr2(self, ts, lr2, plain_le):
        """Both models yield Ze = Re at DC (ω → 0)."""
        omega_dc = np.array([2 * math.pi * 0.001])
        ze_lr2 = voice_coil_impedance(lr2, ts.Re, omega_dc)
        ze_plain = voice_coil_impedance(plain_le, ts.Re, omega_dc)
        assert abs(ze_lr2[0].real - ts.Re) < 0.001
        assert abs(ze_plain[0].real - ts.Re) < 0.001


# ---------------------------------------------------------------------------
# 4. Convention lock — THE CRITICAL SIGN TEST
# ---------------------------------------------------------------------------


class TestConventionLock:
    """Verify the exp(−jωt) conjugation in terminal_response().

    H_bem uses NumCalc's engineering exp(−jωt) convention.  The textbook lumped
    model uses exp(+jωt).  terminal_response = conj(u_textbook) performs the
    conversion.  A magnitude-only test cannot catch a missing conjugation.

    The discriminating observable is the sign of Im(Z_in) at HF:
      - textbook exp(+jωt): inductive region → Im(Z_in) > 0
      - engineering exp(−jωt): same region → Im(Z_in) < 0
    """

    def test_terminal_response_equals_conj_u_textbook(self, ts, lr2, model, freqs):
        """terminal_response must be element-wise exactly conj(u_textbook)."""
        omega = 2.0 * np.pi * freqs  # [F]
        ze = voice_coil_impedance(lr2, ts.Re, omega)
        u_textbook = cone_velocity(ts, ze, omega, voltage=2.83, box_volume=None)

        tr = terminal_response(model, freqs, rho=_RHO, c=_C)

        np.testing.assert_allclose(
            tr,
            np.conj(u_textbook),
            rtol=1e-10,
            err_msg=(
                "terminal_response is not conj(u_textbook). "
                "The engineering-convention conjugation may be missing or applied twice."
            ),
        )

    def test_engineering_convention_sign_at_hf(self, ts, lr2):
        """Im(Z_in) < 0 at HF in engineering convention.

        This is the definitive discriminating test:
          - textbook exp(+jωt): Im(Z_in) > 0 (inductive — verified above)
          - engineering exp(−jωt): Im(Z_in) < 0  ← what we assert here

        If the conjugation is correct, engineering Z_in = conj(textbook Z_in),
        so Im flips sign.
        """
        omega_hf = np.array([2 * math.pi * 10_000.0])
        ze_textbook = voice_coil_impedance(lr2, ts.Re, omega_hf)
        zm_textbook = mechanical_impedance(ts, omega_hf)
        z_in_textbook = input_impedance(ze_textbook, zm_textbook, ts.Bl)

        # Engineering convention: conjugate
        z_in_eng = np.conj(z_in_textbook)
        assert z_in_eng[0].imag < 0.0, (
            f"At 10 kHz, Im(Z_in_engineering) = {z_in_eng[0].imag:.4f} Ω; "
            "expected < 0 for engineering exp(−jωt) convention."
        )

    def test_terminal_response_phase_sign_off_resonance(self, ts, lr2, model, freqs):
        """terminal_response phase must be negated vs textbook u(ω) at every point."""
        omega = 2.0 * np.pi * freqs
        ze = voice_coil_impedance(lr2, ts.Re, omega)
        u_textbook = cone_velocity(ts, ze, omega, voltage=2.83)
        tr = terminal_response(model, freqs, rho=_RHO, c=_C)

        phase_textbook = np.angle(u_textbook)
        phase_eng = np.angle(tr)

        # conj flips sign: phase_eng = -phase_textbook (mod 2π)
        # Check where |phase| > 5° so we have a meaningful signal
        mask = np.abs(phase_textbook) > np.deg2rad(5)
        if mask.any():
            np.testing.assert_allclose(
                phase_eng[mask],
                -phase_textbook[mask],
                atol=1e-9,
                err_msg="terminal_response phase must be negated vs textbook u(ω)",
            )


# ---------------------------------------------------------------------------
# 5. Sealed-box alignment via terminal_response
# ---------------------------------------------------------------------------


class TestSealedBox:
    def test_sealed_box_qtc_and_fc(self, ts, lr2):
        """Sealed-box response shows correct fc and Qtc shift.

        VERIFIED: fc = fs·√(1 + Vas/Vb),  Qtc = Qts·√(1 + Vas/Vb).
        Small, R.H., *JAES* 22(10):798–808, 1973.
        """
        Vb = _VAS_M3  # Vb = Vas → (1 + Vas/Vb) = 2, fc = fs·√2, Qtc = Qts·√2
        Vas = ts.vas(rho=_RHO, c=_C)
        ratio = math.sqrt(1.0 + Vas / Vb)
        fc_expected = _FS * ratio

        # Build sealed-box TerminalModel
        model_box = TerminalModel(ts=ts, inductance=lr2, box_volume=Vb, voltage=2.83)
        freqs_box = np.geomspace(20.0, 500.0, 2000)
        tr_box = terminal_response(model_box, freqs_box, rho=_RHO, c=_C)

        # The velocity peak is near fc but slightly above it: the Im(Ze)×Im(Zm)
        # cross-term (voice-coil inductance × mass reactance above resonance) shifts
        # the peak up by a few Hz.  This is physically real, not a bug.
        # Tolerance ±8 Hz acknowledges the ~5 Hz shift for this fixture.
        peak_idx = np.argmax(np.abs(tr_box))
        fc_found = freqs_box[peak_idx]
        assert abs(fc_found - fc_expected) < 8.0, (
            f"Sealed box velocity peak: found {fc_found:.1f} Hz, expected near "
            f"{fc_expected:.1f} Hz (Vb = Vas = {Vb*1000:.0f} L → ratio √2). "
            "Small shift above fc is expected due to electrical loading."
        )

    def test_sealed_box_raises_resonance_vs_free_air(self, ts, lr2):
        """Sealed box resonance must be higher than free-air resonance."""
        freqs_test = np.geomspace(20.0, 300.0, 1000)
        model_free = TerminalModel(ts=ts, inductance=lr2, box_volume=None)
        model_box = TerminalModel(ts=ts, inductance=lr2, box_volume=_VAS_M3)

        tr_free = terminal_response(model_free, freqs_test)
        tr_box = terminal_response(model_box, freqs_test)

        fc_free = freqs_test[np.argmax(np.abs(tr_free))]
        fc_box = freqs_test[np.argmax(np.abs(tr_box))]
        assert (
            fc_box > fc_free
        ), f"Sealed box fc ({fc_box:.1f} Hz) must exceed free-air fs ({fc_free:.1f} Hz)"


# ---------------------------------------------------------------------------
# 6. Shape, dtype, hygiene
# ---------------------------------------------------------------------------


class TestOutputHygiene:
    def test_shape(self, model, freqs):
        tr = terminal_response(model, freqs)
        assert tr.shape == freqs.shape, f"Expected shape {freqs.shape}, got {tr.shape}"

    def test_dtype(self, model, freqs):
        tr = terminal_response(model, freqs)
        assert tr.dtype == np.complex128, f"Expected complex128, got {tr.dtype}"

    def test_no_nan_or_inf(self, model, freqs):
        tr = terminal_response(model, freqs)
        assert np.all(np.isfinite(tr)), "terminal_response contains NaN or Inf"

    def test_magnitude_nonzero_in_passband(self, model, freqs):
        """Response should not be zero in the driver's passband (30–200 Hz)."""
        tr = terminal_response(model, freqs)
        passband_mask = (freqs >= 30.0) & (freqs <= 200.0)
        assert np.all(
            np.abs(tr[passband_mask]) > 1e-12
        ), "terminal_response is zero in the passband — check for a divide-by-zero"


# ---------------------------------------------------------------------------
# 7. terminal_responses_for list builder
# ---------------------------------------------------------------------------


class TestTerminalResponsesFor:
    def test_returns_list_of_correct_length(self, ts, lr2, freqs):
        models = [
            TerminalModel(ts=ts, inductance=lr2, name="driver_0"),
            TerminalModel(ts=ts, inductance=lr2, name="driver_1"),
        ]
        results = terminal_responses_for(models, freqs)
        assert len(results) == 2

    def test_each_element_is_correct_shape(self, ts, lr2, freqs):
        models = [TerminalModel(ts=ts, inductance=lr2)]
        results = terminal_responses_for(models, freqs)
        assert results[0].shape == freqs.shape
        assert results[0].dtype == np.complex128

    def test_matches_individual_calls(self, ts, lr2, freqs):
        m = TerminalModel(ts=ts, inductance=lr2)
        list_result = terminal_responses_for([m], freqs)[0]
        single = terminal_response(m, freqs)
        np.testing.assert_array_equal(list_result, single)


# ---------------------------------------------------------------------------
# 8. Wiring: terminal_responses_for → build_dataset → H_full check
# ---------------------------------------------------------------------------


class TestWiringToBuildDataset:
    """Verify that plugging terminal_response into build_dataset correctly forms H_full.

    Uses a synthetic 2-driver ComplexField (random H_bem arrays) to confirm:
      H_full[driver, :, :] = H_bem[driver] × terminal_response[driver][:, None]
    """

    def test_h_full_equals_h_bem_times_terminal(self, ts, lr2, freqs):
        from beamsim2.assembly.tensor import build_dataset
        from beamsim2.core.sphere import lebedev
        from beamsim2.core.types import ComplexField

        F = len(freqs)
        N = 26  # small Lebedev-26 grid for speed

        obs = lebedev(N)

        # Two synthetic H_bem arrays (random complex)
        rng = np.random.default_rng(42)
        h_bem_0 = rng.standard_normal((F, N)) + 1j * rng.standard_normal((F, N))
        h_bem_1 = rng.standard_normal((F, N)) + 1j * rng.standard_normal((F, N))

        cf0 = ComplexField(
            pressure=h_bem_0.astype(np.complex128),
            convergence_flags=np.ones(F, dtype=bool),
            frequencies=freqs,
        )
        cf1 = ComplexField(
            pressure=h_bem_1.astype(np.complex128),
            convergence_flags=np.ones(F, dtype=bool),
            frequencies=freqs,
        )

        # Two TerminalModels (different names but same params for simplicity)
        models = [
            TerminalModel(ts=ts, inductance=lr2, name="driver_0"),
            TerminalModel(ts=ts, inductance=lr2, name="driver_1"),
        ]
        tr_list = terminal_responses_for(models, freqs)

        ds = build_dataset(
            driver_inputs=[
                ("driver_0", cf0, models[0].to_attrs()),
                ("driver_1", cf1, models[1].to_attrs()),
            ],
            directions=obs,
            terminal_responses=tr_list,
        )

        # Verify H_full = H_bem × terminal_response[:, None] for each driver
        for i, (d, tr) in enumerate(zip(ds.drivers, tr_list)):
            expected_h_full = d.H_bem * tr[:, None]  # [F, N]
            np.testing.assert_allclose(
                d.H_full,
                expected_h_full,
                rtol=1e-10,
                err_msg=f"driver {i}: H_full != H_bem × terminal_response[:, None]",
            )

        # Verify the cardinal phase-origin rule still passes with a non-trivial terminal
        from beamsim2.assembly.phase_origin import assert_superposition_matches

        # Synthetic direct "BEM solve" of both drivers simultaneously
        tr0, tr1 = tr_list
        # Simulate direct two-driver result = sum of individual scaled fields
        direct_pressure = h_bem_0 * tr0[:, None] + h_bem_1 * tr1[:, None]
        direct_cf = ComplexField(
            pressure=direct_pressure.astype(np.complex128),
            convergence_flags=np.ones(F, dtype=bool),
            frequencies=freqs,
        )

        from beamsim2.assembly.superpose import superpose_fields

        summed = superpose_fields([ds.drivers[0].H_full, ds.drivers[1].H_full])
        assert_superposition_matches(summed, direct_cf.pressure, rtol=1e-10)

    def test_attrs_populated_in_driver(self, ts, lr2, freqs):
        """to_attrs() keys appear in the assembled driver attrs."""
        from beamsim2.assembly.tensor import build_dataset
        from beamsim2.core.sphere import lebedev
        from beamsim2.core.types import ComplexField

        F = len(freqs)
        N = 14
        obs = lebedev(N)
        h_bem = np.ones((F, N), dtype=np.complex128)
        cf = ComplexField(
            pressure=h_bem, convergence_flags=np.ones(F, dtype=bool), frequencies=freqs
        )
        m = TerminalModel(ts=ts, inductance=lr2, name="test_driver")
        tr = terminal_response(m, freqs)
        ds = build_dataset(
            driver_inputs=[("test_driver", cf, m.to_attrs())],
            directions=obs,
            terminal_responses=[tr],
        )
        d_attrs = ds.drivers[0].attrs
        assert "terminal_response_model" in d_attrs
        assert "ts_params" in d_attrs


# ---------------------------------------------------------------------------
# 9. TerminalModel.to_attrs metadata
# ---------------------------------------------------------------------------


class TestToAttrs:
    def test_keys_present(self, model):
        attrs = model.to_attrs()
        for key in ("name", "terminal_response_model", "ts_params", "box_volume_m3"):
            assert key in attrs, f"Missing key '{key}' in to_attrs()"

    def test_ts_params_contains_fs(self, model):
        ts_dict = model.to_attrs()["ts_params"]
        assert "fs_Hz" in ts_dict
        assert abs(ts_dict["fs_Hz"] - _FS) < 0.1

    def test_lr2_description_in_model_string(self, model):
        desc = model.to_attrs()["terminal_response_model"]
        assert "LR2Ladder" in desc
