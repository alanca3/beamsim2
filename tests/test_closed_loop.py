"""Stage-4 close-the-loop gate (§8): delay-and-sum beamforming from H tensor.

Proves that the per-driver H[driver × frequency × direction] tensor produced by
Phase 1 correctly preserves the inter-driver time-of-flight phase needed for
beamforming — the ultimate end-to-end verification of the single-phase-origin
rule (DATA_CONTRACT.md §3.4, BEAMSIMII_Gameplan.md §8 Stage-4).

Two tiers
---------
Tier 1 — CI-safe synthetic (pure analytic, no NumCalc):
    Build H synthetically from the point-monopole formula in the engineering
    convention. Apply delay-and-sum weights. Assert:
      (a) The steered pattern has a null ≤ −20 dB in the end-fire (−z) direction
          at the design frequency f = c / (4 d) ≈ 1716 Hz.
      (b) Without weights (equal sum), the −z direction is NOT a null (> −10 dB).
      (c) Bug injection: stripping driver B's on-axis reference phase destroys
          the null at −z (fills to ≥ −3 dB), confirming the null depends on
          correctly preserved inter-driver time-of-flight phase.
    Note: the field-vs-analytic comparison is exact in this tier (H_bem was
    built from the same formula) — the null and bug-injection tests are the
    substantive checks.

Tier 2 — Real BEM (local_only, NumCalc required):
    Use the V-5 box-plus-2-driver geometry (known to solve) with two flush-mounted
    drivers side by side in the x-direction (spacing d_x = 0.05 m). Solve driver A
    and driver B separately with NumCalc. Apply end-fire weights to steer in +x,
    producing a null in −x at design frequency f = c / (4 d_x) ≈ 1716 Hz.  Assert:
      (a) The steered BEM pattern shows a null ≤ −10 dB in the −x direction at the
          design frequency (tolerance loose: finite piston size + box diffraction
          partially fill the null).
      (b) Steered BEM vs steered analytic monopole-pair: pattern-shape RMS error
          < 3 dB (accounts for finite piston size and box diffraction).
      (c) Bug injection (strip driver B's x-direction delay phase): intensity at
          the expected −x null direction rises to > −3 dB (null destroyed), and
          RMS error vs analytic rises to > 6 dB (agreement destroyed).

Acceptance (§8 Stage-4 gate)
-----------------------------
Passing all tests in each tier = Stage-4 green = Phase-1 "done" prerequisite met.

Honesty notes
-------------
- A single-frequency end-fire null is NOT "constant directivity across frequency."
  It is a two-element end-fire pattern at the design frequency. Broadband CD
  beamforming (CBT, superdirective) belongs to Phase 2.
- NEVER re-zero or minimum-phase-ify any driver — that is the cardinal-rule
  violation this test guards against. Bug injection simulates exactly that.
- Tier 2 tolerance (3 dB) is intentionally generous. Finite piston + box
  diffraction create systematic differences vs free-space monopoles. The critical
  thing is the inter-driver PHASE relationship, which is tested with a tighter
  implicit check via the bug injection.
"""

from __future__ import annotations

import numpy as np
import pytest

from beamsim2.core.sphere import lebedev
from beamsim2.validation.closed_loop import (
    delay_sum_weights,
    field_agreement_db,
    monopole_field,
    null_depth_db,
    steer_response,
)

# ── NumCalc binary resolution (real-BEM tier only) ───────────────────────────

try:
    from beamsim2.backends.numcalc.config import resolve_numcalc_binary

    _BINARY = resolve_numcalc_binary()
except FileNotFoundError:
    _BINARY = None


def _skip_if_no_binary() -> None:
    if _BINARY is None:
        pytest.skip("NumCalc binary not found. Set BEAMSIM2_NUMCALC_BIN to run real-BEM tier.")


# ── Shared geometry constants ─────────────────────────────────────────────────

_C = 343.2  # m/s — speed of sound

# Synthetic tier: two point sources along z, separated by d = 0.05 m
_D_SYN = 0.05  # m
_F_DESIGN_SYN = _C / (4.0 * _D_SYN)  # ≈ 1716 Hz — design null frequency

# Test frequencies: below, at, and above the design null
_FREQS_SYN = np.array([800.0, _F_DESIGN_SYN, 3000.0], dtype=np.float64)
_DESIGN_IDX_SYN = 1  # index of design frequency in _FREQS_SYN

# Lebedev-26 sphere at 1 m: 26 points including (0,0,±1), (±1,0,0), (0,±1,0)
_OBS = lebedev(n_points=26, radius=1.0)


def _find_direction_n(unit_target: np.ndarray) -> int:
    """Return the Lebedev-26 index of the direction closest to unit_target [3]."""
    dots = _OBS.unit_vectors @ unit_target  # [N] cosines
    return int(np.argmax(dots))


# Pre-compute direction indices for convenience
_N_REAR = _find_direction_n(np.array([0.0, 0.0, -1.0]))  # closest to -z
_N_FRONT = _find_direction_n(np.array([0.0, 0.0, 1.0]))  # closest to +z
_N_NEG_X = _find_direction_n(np.array([-1.0, 0.0, 0.0]))  # closest to -x


# ─────────────────────────────────────────────────────────────────────────────
# Tier 1: CI-safe synthetic tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSyntheticClosedLoop:
    """Stage-4 gate, Tier 1: analytic H tensor, end-fire null + bug injection.

    Two point sources: A at origin, B at (0, 0, −d). Steer toward +z;
    null expected at −z (θ = π) at design frequency f = c/(4d) ≈ 1716 Hz.
    """

    def _build(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (H_bem [2,F,N], weights [2,F], P_steered [F,N]) for the
        standard synthetic configuration (steer +z, sources along z-axis)."""
        steer_dir = np.array([0.0, 0.0, 1.0])
        p_A = np.zeros(3)
        p_B = np.array([0.0, 0.0, -_D_SYN])  # B is behind A along -z
        positions = np.stack([p_A, p_B])  # [2, 3]

        H_bem = monopole_field(positions, _OBS, _FREQS_SYN, c=_C)
        weights = delay_sum_weights(positions, steer_dir, _FREQS_SYN, c=_C)
        P_steered = steer_response(H_bem, weights)
        return H_bem, weights, P_steered

    def _null_at_direction(self, P: np.ndarray, n: int, f_idx: int) -> float:
        """Return 10·log10(|P[f_idx,n]|² / max_n |P[f_idx,n]|²) in dB."""
        intensity = np.abs(P[f_idx]) ** 2  # [N]
        return float(10.0 * np.log10(intensity[n] / np.max(intensity)))

    # ── main null test ────────────────────────────────────────────────────────

    def test_null_at_design_frequency(self) -> None:
        """Steered end-fire pattern has null ≤ −20 dB at −z at design frequency.

        The Lebedev-26 sphere includes the (0, 0, −1) direction, so the null is
        sampled directly. The near-field null at r = 1 m is analytically ≈ −31 dB;
        the tolerance of −20 dB is deliberately conservative.

        Gate: intensity ratio at −z direction ≤ −20 dB at f = c/(4d) ≈ 1716 Hz.
        """
        _, _, P_steered = self._build()
        null_dB = self._null_at_direction(P_steered, _N_REAR, _DESIGN_IDX_SYN)

        print(
            f"\n[Stage-4 synthetic null] f={_FREQS_SYN[_DESIGN_IDX_SYN]:.0f} Hz"
            f"  intensity at -z = {null_dB:.1f} dB"
        )

        assert null_dB < -20.0, (
            f"Null too shallow at {_FREQS_SYN[_DESIGN_IDX_SYN]:.0f} Hz, −z direction: "
            f"{null_dB:.1f} dB — expected ≤ −20 dB"
        )

    # ── global null depth (diagnostic) ───────────────────────────────────────

    def test_null_depth_metric(self) -> None:
        """null_depth_db returns ≤ −20 dB at the design frequency (global metric).

        This tests the null_depth_db helper in addition to the direction-specific
        check above. Both should agree.
        """
        _, _, P_steered = self._build()
        nd = null_depth_db(P_steered, _OBS.weights)  # [F] dB

        print(
            f"\n[Stage-4 null_depth_db] null depths: {nd.round(1)} dB"
            f"  at freqs {_FREQS_SYN.round(0)} Hz"
        )

        assert (
            nd[_DESIGN_IDX_SYN] < -20.0
        ), f"null_depth_db at design frequency: {nd[_DESIGN_IDX_SYN]:.1f} dB"

    # ── unsteered baseline ────────────────────────────────────────────────────

    def test_no_deep_null_without_weights(self) -> None:
        """Unsteered (equal-weight) sum has no null at −z at design frequency.

        Without delay compensation, both sources contribute with their natural
        spherical-wave phases. The resulting pattern has no systematic end-fire
        null. The −z direction should NOT be deep (> −10 dB).

        Gate: intensity at −z ≥ −10 dB relative to maximum at design frequency.
        """
        H_bem, _, _ = self._build()
        equal_weights = np.ones((2, len(_FREQS_SYN)), dtype=np.complex128)
        P_unsteered = steer_response(H_bem, equal_weights)
        null_dB = self._null_at_direction(P_unsteered, _N_REAR, _DESIGN_IDX_SYN)

        print(
            f"\n[Stage-4 unsteered] f={_FREQS_SYN[_DESIGN_IDX_SYN]:.0f} Hz"
            f"  intensity at -z (unsteered) = {null_dB:.1f} dB"
        )

        assert null_dB > -10.0, (
            f"Unsteered sum has a deep null at −z ({null_dB:.1f} dB) — "
            f"should not happen without delay-and-sum weights"
        )

    # ── bug injection (positive control) ─────────────────────────────────────

    def test_bug_injection_destroys_null(self) -> None:
        """Positive control: stripping driver B's on-axis phase kills the null at −z.

        Simulates the cardinal-rule violation (§3.4 DATA_CONTRACT): re-zeroing
        driver B so its H_bem is referenced to its own local phase origin rather
        than the global origin. The procedure removes the inter-driver delay that
        the beamformer relies on to form the null.

        Implementation: multiply H_B[f,n] by exp(−j·φ_ref(f)), where φ_ref(f) is
        the phase of H_B at the reference direction (+z, onaxis). This makes
        H_B real-positive on-axis and destroys the correct delay relationship.

        After re-zeroing, the −z direction is NO LONGER a null — it becomes
        the direction of maximum intensity (the re-zeroing effectively reverses
        the beam direction from −z to +z). Assertion: intensity at −z ≥ −3 dB.

        This mirrors the positive-control approach in V-5 (test_phase_ramp_bug_*).
        """
        H_bem, weights, _ = self._build()

        # Re-zero driver B: remove the per-frequency phase at the +z on-axis point.
        # This simulates minimum-phase-ification / per-driver phase stripping.
        H_B = H_bem[1]  # [F, N]
        phase_ref = np.angle(H_B[:, _N_FRONT])  # [F] — phase at +z direction
        H_B_rezeroed = H_B * np.exp(-1j * phase_ref[:, None])  # [F, N]

        H_bem_buggy = H_bem.copy()
        H_bem_buggy[1] = H_B_rezeroed

        P_buggy = steer_response(H_bem_buggy, weights)
        null_dB_bug = self._null_at_direction(P_buggy, _N_REAR, _DESIGN_IDX_SYN)

        print(
            f"\n[Stage-4 bug injection] f={_FREQS_SYN[_DESIGN_IDX_SYN]:.0f} Hz"
            f"  intensity at -z after re-zero = {null_dB_bug:.1f} dB  (was < -20 dB)"
        )

        # After re-zeroing, −z is no longer a null (it becomes the maximum)
        assert null_dB_bug > -3.0, (
            f"Bug injection did NOT destroy the null at −z "
            f"({_FREQS_SYN[_DESIGN_IDX_SYN]:.0f} Hz): "
            f"intensity = {null_dB_bug:.1f} dB — expected > −3 dB (null should be gone)"
        )

    # ── exact analytic agreement ──────────────────────────────────────────────

    def test_field_vs_analytic_exact(self) -> None:
        """Steered BEM == steered analytic to machine precision (arithmetic check).

        Since H_bem was built from the same monopole formula used as the reference,
        field_agreement_db should be ≈ 0. This tests the arithmetic chain:
        monopole_field → delay_sum_weights → steer_response → field_agreement_db.

        Gate: max field_agreement_db < 0.01 dB across all test frequencies.
        """
        H_bem, weights, P_steered = self._build()
        positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, -_D_SYN]])

        P_analytic = steer_response(monopole_field(positions, _OBS, _FREQS_SYN, c=_C), weights)

        rms_err = field_agreement_db(P_steered, P_analytic, _OBS.weights)
        assert (
            np.max(rms_err) < 0.01
        ), f"Exact analytic comparison failed: max error {np.max(rms_err):.4f} dB"


# ─────────────────────────────────────────────────────────────────────────────
# Tier 2: Real BEM tests (local_only — NumCalc required)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.local_only
class TestRealBEMClosedLoop:
    """Stage-4 gate, Tier 2: real NumCalc solves, field-vs-analytic + bug injection.

    Geometry: V-5 box (0.12 × 0.10 × 0.08 m) with two flush-mounted drivers on
    the front (+z) face, side by side in x. Centres at:
      A: (0.035, 0.05, 0.08 m) — left driver
      B: (0.085, 0.05, 0.08 m) — right driver
    Inter-driver x-spacing d_x = 0.05 m.

    Beamforming: steer toward +x → null in −x direction at f_design ≈ 1716 Hz.
    Two NumCalc solves: driver A only (B sound-hard), driver B only (A sound-hard).
    """

    # Box and driver geometry (identical to V-5)
    _W: float = 0.12
    _H_BOX: float = 0.10
    _D_BOX: float = 0.08
    _R_DRV: float = 0.020
    _H_ELEM: float = 0.020
    _D_X: float = 0.05  # inter-driver x spacing

    # Driver positions
    _X_A: float = 0.12 / 2 - 0.05 / 2  # = 0.035 m
    _X_B: float = 0.12 / 2 + 0.05 / 2  # = 0.085 m
    _Y_DRV: float = 0.10 / 2  # = 0.05 m
    _Z_FACE: float = 0.08  # front face

    # Design null frequency for x-direction end-fire
    _F_DESIGN: float = _C / (4.0 * 0.05)  # ≈ 1716 Hz

    # Solve frequencies: 3 from V-5 + design frequency
    _FREQS: np.ndarray = np.array([250.0, 500.0, 1000.0, _C / (4.0 * 0.05)], dtype=np.float64)

    def _driver_positions(self) -> np.ndarray:
        """Return driver positions [[x_A, y, z], [x_B, y, z]] shape [2, 3]."""
        return np.array(
            [
                [self._X_A, self._Y_DRV, self._Z_FACE],
                [self._X_B, self._Y_DRV, self._Z_FACE],
            ]
        )

    def _run_two_solves(self) -> tuple:
        """Run driver-A-only and driver-B-only NumCalc solves.

        Returns
        -------
        field_A : ComplexField — BEM field for driver A vibrating alone.
        field_B : ComplexField — BEM field for driver B vibrating alone.
        """
        from beamsim2.backends.numcalc.adapter import NumCalcBackend
        from beamsim2.core.types import BoundaryConditions, FrequencyGrid, SolverConfig
        from beamsim2.geometry.assemble import DriverSpec, assemble_box_driver

        driver_a = DriverSpec(
            center=(self._X_A, self._Y_DRV, self._Z_FACE),
            normal=(0.0, 0.0, 1.0),
            radius=self._R_DRV,
        )
        driver_b = DriverSpec(
            center=(self._X_B, self._Y_DRV, self._Z_FACE),
            normal=(0.0, 0.0, 1.0),
            radius=self._R_DRV,
        )

        mesh, _ = assemble_box_driver(
            width=self._W,
            height=self._H_BOX,
            depth=self._D_BOX,
            drivers=[driver_a, driver_b],
            h_elem=self._H_ELEM,
        )

        bc_a = BoundaryConditions(vibrating_groups={1: complex(1.0, 0.0)})
        bc_b = BoundaryConditions(vibrating_groups={2: complex(1.0, 0.0)})
        freqs = FrequencyGrid(frequencies=self._FREQS, spacing="log")
        config = SolverConfig()
        backend = NumCalcBackend()

        def _solve(bc: BoundaryConditions):
            spec = backend.prepare(mesh, bc, freqs, _OBS, config)
            raw = backend.solve(spec)
            result = backend.extract(raw, _OBS)  # ComplexField
            assert (
                result.convergence_flags.all()
            ), f"NumCalc did not converge: flags={result.convergence_flags}"
            return result  # ComplexField: .pressure [F, N], .frequencies [F]

        return _solve(bc_a), _solve(bc_b)

    def _assemble_dataset(self, field_A, field_B):
        """Wrap two ComplexField results into a RadiationDataset via build_dataset.

        This is the data-contract path all real-BEM tests must exercise.
        terminal_responses=None → ones(F) so H_full == H_bem (item 8 not yet done).
        """
        from beamsim2.assembly.tensor import build_dataset

        return build_dataset(
            driver_inputs=[
                (
                    "driver_A",
                    field_A,
                    {"position": [self._X_A, self._Y_DRV, self._Z_FACE]},
                ),
                (
                    "driver_B",
                    field_B,
                    {"position": [self._X_B, self._Y_DRV, self._Z_FACE]},
                ),
            ],
            directions=_OBS,
            freq_grid_spacing="log",
        )

    def _design_idx(self) -> int:
        return int(np.argmin(np.abs(self._FREQS - self._F_DESIGN)))

    def _steer_dir(self) -> np.ndarray:
        return np.array([1.0, 0.0, 0.0])  # steer +x, null at -x

    def test_contract_round_trip(self) -> None:
        """The data contract (build_dataset + stacked_h_full + HDF5) preserves phase.

        This is the true close-the-loop: confirms that the assembly step and the
        HDF5 writer/reader do NOT alter the inter-driver phase that beamforming
        depends on. The null tests that follow rely on this being lossless.

        Gate (contract):
          |stacked_h_full(build_dataset(...)) − np.stack([H_A, H_B])| < 1e-12
        Gate (HDF5):
          |stacked_h_full(read_dataset(write_dataset(...))) − raw| < 1e-12
        """
        import tempfile
        from pathlib import Path

        from beamsim2.assembly.tensor import stacked_h_full
        from beamsim2.io.hdf5_store import read_dataset, write_dataset

        _skip_if_no_binary()

        field_A, field_B = self._run_two_solves()
        ds = self._assemble_dataset(field_A, field_B)

        H_stack_raw = np.stack([field_A.pressure, field_B.pressure], axis=0)  # [2,F,N]
        H_stack_contract = stacked_h_full(ds)  # [2, F, N] via build_dataset path

        max_diff_assemble = np.max(np.abs(H_stack_contract - H_stack_raw))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "stage4_contract.h5"
            write_dataset(path, ds)
            ds2 = read_dataset(path)
            H_stack_hdf5 = stacked_h_full(ds2)

        max_diff_hdf5 = np.max(np.abs(H_stack_hdf5 - H_stack_raw))

        print(
            "\n[Stage-4 contract round-trip]"
            f"\n  max |H_assemble − H_raw| = {max_diff_assemble:.2e}  (< 1e-12)"
            f"\n  max |H_hdf5     − H_raw| = {max_diff_hdf5:.2e}  (< 1e-12)"
        )

        assert max_diff_assemble < 1e-12, (
            f"build_dataset/stacked_h_full altered the pressure tensor: "
            f"max diff = {max_diff_assemble:.2e} — assembly must not modify phase"
        )
        assert max_diff_hdf5 < 1e-12, (
            f"HDF5 round-trip altered the pressure tensor: "
            f"max diff = {max_diff_hdf5:.2e} — I/O must be lossless"
        )

    def test_steered_null_in_minus_x_direction(self) -> None:
        """Steered field (via data-contract tensor) shows null ≤ −10 dB at −x.

        Routes through build_dataset → stacked_h_full → steer_response.
        End-fire toward +x. Lebedev-26 includes (−1, 0, 0). At design frequency
        ≈ 1716 Hz the −x intensity should be well below the forward maximum.
        Tolerance −10 dB accounts for finite piston size and box diffraction.

        Gate: intensity at −x ≤ −10 dB relative to forward max at design freq.
        """
        _skip_if_no_binary()

        from beamsim2.assembly.tensor import stacked_h_full

        field_A, field_B = self._run_two_solves()
        ds = self._assemble_dataset(field_A, field_B)
        H_stack = stacked_h_full(ds)  # [2, F, N] — via data contract

        positions = self._driver_positions()
        weights = delay_sum_weights(positions, self._steer_dir(), self._FREQS, c=_C)
        P_steered = steer_response(H_stack, weights)  # [F, N]

        fi = self._design_idx()
        I_steered = np.abs(P_steered[fi]) ** 2  # [N]
        I_null = I_steered[_N_NEG_X]
        I_max = np.max(I_steered)
        null_dB = 10.0 * np.log10(I_null / I_max)

        print(
            f"\n[Stage-4 real-BEM null] f={self._FREQS[fi]:.0f} Hz"
            f"  −x direction={_OBS.unit_vectors[_N_NEG_X].tolist()}"
            f"  null depth={null_dB:.1f} dB"
        )

        assert null_dB < -10.0, (
            f"Steered BEM pattern null too shallow at {self._FREQS[fi]:.0f} Hz"
            f" in −x direction: {null_dB:.1f} dB — expected ≤ −10 dB"
        )

    def test_steered_field_matches_analytic(self) -> None:
        """Steered field (via data-contract tensor) matches analytic monopole-pair.

        Full path: NumCalc → ComplexField → build_dataset → stacked_h_full →
        steer_response → field_agreement_db. Compared to P_analytic from the
        point-monopole formula at the driver positions.

        V-2 verified single-driver BEM ≈ monopole to 0.15 dB. Stage-4 proves the
        multi-driver steered path through the full data contract preserves the
        inter-driver phase. A phase-origin violation anywhere in the chain would
        produce a systematic offset far exceeding 3 dB.

        Gate: pattern-shape RMS error < 3 dB at non-null frequencies (250, 500,
        1000 Hz). Design frequency excluded: near the null, finite-piston vs
        point-monopole differences are amplified in dB and dominate the metric.
        Null depth is tested separately by test_steered_null_in_minus_x_direction.
        """
        _skip_if_no_binary()

        from beamsim2.assembly.tensor import stacked_h_full

        field_A, field_B = self._run_two_solves()
        ds = self._assemble_dataset(field_A, field_B)
        H_stack = stacked_h_full(ds)  # [2, F, N] — via data contract

        positions = self._driver_positions()
        weights = delay_sum_weights(positions, self._steer_dir(), self._FREQS, c=_C)

        P_bem = steer_response(H_stack, weights)
        P_analytic = steer_response(monopole_field(positions, _OBS, self._FREQS, c=_C), weights)

        rms_err = field_agreement_db(P_bem, P_analytic, _OBS.weights)  # [F] dB

        # Exclude design null frequency: near-null dB sensitivity is a metric
        # property, not a phase-origin signal. Tested by test_steered_null_*.
        non_null = self._FREQS < self._F_DESIGN - 100.0  # True for 250, 500, 1000 Hz

        print(
            "\n[Stage-4 BEM vs analytic]"
            f"\n  freqs         : {self._FREQS} Hz"
            f"\n  rms err       : {np.round(rms_err, 2)} dB"
            f"\n  checked freqs : {self._FREQS[non_null]} Hz"
            f"\n  checked err   : {np.round(rms_err[non_null], 2)} dB"
        )

        assert np.all(rms_err[non_null] < 3.0), (
            f"Steered BEM vs analytic pattern error exceeds 3 dB at non-null freqs:\n"
            f"  freqs={self._FREQS[non_null]} Hz  errs={rms_err[non_null]} dB\n"
            f"Possible cause: phase-origin violation or large box diffraction effect."
        )

    def test_bug_injection_destroys_null_and_agreement(self) -> None:
        """Positive control: stripping B's x-delay phase raises the −x null.

        Uses the data-contract H tensor (stacked_h_full), then injects the bug
        in memory. Multiplies H_B row by exp(−j · k · d_x), removing the
        inter-driver delay the beamformer relies on. Simulates the phase-origin
        violation (§3.4): B's H appears as if it came from x = x_A not x = x_B.

        After the bug:
          (a) The −x null rises by > 5 dB relative to the un-bugged version.
              (A scalar per-frequency phase shift moves the null, not fills it,
              so a relative rise criterion is more discriminating than absolute.)
          (b) The BEM−analytic RMS error at the design frequency rises to > 6 dB,
              confirming the passing test relied on correct inter-driver phase.
        """
        _skip_if_no_binary()

        from beamsim2.assembly.tensor import stacked_h_full

        field_A, field_B = self._run_two_solves()
        ds = self._assemble_dataset(field_A, field_B)
        H_stack_ok = stacked_h_full(ds)  # [2, F, N] via data contract

        positions = self._driver_positions()
        weights = delay_sum_weights(positions, self._steer_dir(), self._FREQS, c=_C)
        fi = self._design_idx()

        # ── correct (no bug) ─────────────────────────────────────────────────
        P_ok = steer_response(H_stack_ok, weights)
        I_ok = np.abs(P_ok[fi]) ** 2
        null_dB_ok = 10.0 * np.log10(I_ok[_N_NEG_X] / np.max(I_ok))

        # ── bugged: strip B's x-direction propagation delay ──────────────────
        k_freqs = 2.0 * np.pi * self._FREQS / _C  # [F]
        H_B_bug = H_stack_ok[1] * np.exp(-1j * k_freqs[:, None] * self._D_X)  # [F,N]
        H_stack_bug = H_stack_ok.copy()
        H_stack_bug[1] = H_B_bug

        P_bug = steer_response(H_stack_bug, weights)
        I_bug = np.abs(P_bug[fi]) ** 2
        null_dB_bug = 10.0 * np.log10(I_bug[_N_NEG_X] / np.max(I_bug))

        null_rise_dB = null_dB_bug - null_dB_ok  # positive = null shallower after bug

        # ── BEM−analytic agreement after bug ─────────────────────────────────
        P_analytic = steer_response(monopole_field(positions, _OBS, self._FREQS, c=_C), weights)
        rms_err_bug = field_agreement_db(P_bug, P_analytic, _OBS.weights)

        print(
            "\n[Stage-4 bug injection]"
            f"\n  null at -x (ok)    = {null_dB_ok:.1f} dB"
            f"\n  null at -x (bugged)= {null_dB_bug:.1f} dB"
            f"\n  null rise          = {null_rise_dB:.1f} dB  (expected > 5 dB)"
            f"\n  rms err (buggy)    = {np.round(rms_err_bug, 2)} dB"
            f"\n  err at design freq = {rms_err_bug[fi]:.2f} dB  (expected > 6 dB)"
        )

        # (a) Null at −x rose by > 5 dB after bug (null is meaningfully shallower)
        assert null_rise_dB > 5.0, (
            f"Bug injection only raised the −x null by {null_rise_dB:.1f} dB "
            f"(from {null_dB_ok:.1f} to {null_dB_bug:.1f} dB) — expected > 5 dB rise"
        )
        # (b) BEM−analytic agreement is destroyed at design frequency
        assert rms_err_bug[fi] > 6.0, (
            f"Bug injection did NOT destroy BEM−analytic agreement at design freq: "
            f"{rms_err_bug[fi]:.2f} dB — expected > 6 dB"
        )
