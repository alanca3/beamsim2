"""Per-driver complex terminal response: T/S cone velocity → engineering-convention multiplier.

This module is the single entry point for the driver electrical chain.  Its output —
``terminal_response[F]`` complex128 in NumCalc's engineering exp(−jωt) convention —
plugs directly into ``assembly.tensor.build_dataset(terminal_responses=...)``,
replacing the identity-ones placeholder that ships with item 7.

CONVENTION — the most important correctness invariant in this module
--------------------------------------------------------------------
NumCalc uses the **engineering convention** exp(−jωt): outgoing waves propagate
as exp(+jkr), inductances appear as Z_L = −jωL, capacitors as Z_C = +1/(jωC).
This is the complex-conjugate of the standard textbook / EE convention.

The Thiele/Small formula u = Bl·V / (Ze·Zm + Bl²) is derived in the textbook
convention (exp(+jωt)), where Z_L = +jωL.  To convert, we conjugate once:

    terminal_response = conj( u_textbook(ω) )

Magnitude is unchanged; phase flips sign.  This one conjugation ensures that
H_full = H_bem × terminal_response has consistent phase across the BEM and
electrical contributions.  A magnitude-only test cannot detect a missing
conjugation; the self-test (test_driver_terminal.py) explicitly verifies the
phase sign with an Im(Z_in) < 0 check at HF.

VERIFIED: NumCalc convention — CLAUDE.md "NumCalc time convention" section;
Kreuzer et al., *Eng. Analysis Boundary Elements* 161:157-178, 2024.

References
----------
Thiele, A.N., *JAES* 19(5):382–391, 1971.  VERIFIED.
Small, R.H., *JAES* 20(5):383–395, 1972.  VERIFIED.
Wright, J.R., *JAES* 38(10):749–754, 1990.  VERIFIED (LR-2 model).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from beamsim2.driver.inductance import LR2Ladder, PlainLe, voice_coil_impedance
from beamsim2.driver.thiele_small import TSParams, cone_velocity

_RHO_DEFAULT = 1.2041  # kg/m³ — dry air 20°C (core.units.air_density default)
_C_DEFAULT = 343.2  # m/s — dry air 20°C (core.units.speed_of_sound default)


# ---------------------------------------------------------------------------
# Terminal model dataclass
# ---------------------------------------------------------------------------


@dataclass
class TerminalModel:
    """Bundle of everything needed to compute one driver's terminal response.

    Parameters
    ----------
    ts : TSParams
        Small-signal Thiele/Small parameters.
    inductance : PlainLe or LR2Ladder
        Voice-coil HF inductance model.  Use ``LR2Ladder`` for production;
        ``PlainLe`` for initial checks when only Le is known.
    box_volume : float or None
        Internal box (enclosure) volume, m³.  None → free-air / infinite-baffle
        alignment (cone sees no back-side air spring).
    voltage : float
        Reference drive voltage, V_rms.  Default 2.83 V = 1 W into 8 Ω.
        HEURISTIC: 2.83 V is the standard sensitivity reference for 8 Ω drivers.
    name : str
        Human-readable driver identifier, copied into HDF5 driver attrs.
    """

    ts: TSParams
    inductance: PlainLe | LR2Ladder
    box_volume: float | None = None  # m³ — None = free-air / infinite-baffle
    voltage: float = 2.83  # V_rms — reference drive level
    name: str = "driver"

    def to_attrs(self) -> dict:
        """Generate the §3.5 per-driver metadata dict for HDF5 storage.

        Returns
        -------
        dict
            Keys: ``terminal_response_model``, ``ts_params``, ``box_volume_m3``,
            ``reference_voltage_V``.  Values are JSON-serialisable.
        """
        ts = self.ts
        inductance_desc: str
        if isinstance(self.inductance, LR2Ladder):
            inductance_desc = (
                f"LR2Ladder(Le={self.inductance.Le:.6g}H, "
                f"Le2={self.inductance.Le2:.6g}H, "
                f"Re2={self.inductance.Re2:.6g}Ω)"
            )
        elif isinstance(self.inductance, PlainLe):
            inductance_desc = f"PlainLe(Le={self.inductance.Le:.6g}H)"
        else:
            inductance_desc = str(type(self.inductance).__name__)

        return {
            "name": self.name,
            "terminal_response_model": f"thiele_small+{inductance_desc}",
            "ts_params": {
                "Re_ohm": ts.Re,
                "Bl_Tm": ts.Bl,
                "Mms_kg": ts.Mms,
                "Cms_m_per_N": ts.Cms,
                "Rms_Ns_per_m": ts.Rms,
                "Sd_m2": ts.Sd,
                # derived — for human readability in the HDF5 file
                "fs_Hz": ts.fs,
                "Qms": ts.Qms,
                "Qes": ts.Qes,
                "Qts": ts.Qts,
            },
            "box_volume_m3": self.box_volume,
            "reference_voltage_V": self.voltage,
        }


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def terminal_response(
    model: TerminalModel,
    frequencies: np.ndarray,
    rho: float = _RHO_DEFAULT,
    c: float = _C_DEFAULT,
) -> np.ndarray:
    """Compute the per-driver terminal response in the engineering exp(−jωt) convention.

    Flow:
    1. Ze(ω) = Re + Z_L(ω)           — blocked electrical impedance, textbook sign
    2. Zm(ω) = jωMms + Rms + 1/jωCmt — mechanical impedance, textbook sign
    3. u(ω)  = Bl·V / (Ze·Zm + Bl²)  — cone velocity m/s, textbook sign
    4. terminal_response = conj(u)    — convert to engineering exp(−jωt) convention

    Units check: H_bem is Pa per (m/s at unit cone velocity); terminal_response
    is m/s; product H_full = H_bem × terminal_response[:, None] is Pa.

    Parameters
    ----------
    model : TerminalModel
        Driver parameters and inductance model.
    frequencies : np.ndarray
        Frequency points, Hz.  Shape [F] float64.
    rho : float
        Air density, kg/m³.
    c : float
        Speed of sound, m/s.

    Returns
    -------
    np.ndarray
        Shape [F] complex128, m/s, **engineering exp(−jωt) convention**.
        Multiply H_bem[F, N] by this[:, None] to form H_full[F, N].
    """
    omega = 2.0 * np.pi * frequencies  # [F] float64, rad/s

    # Step 1: blocked electrical impedance — textbook convention
    ze = voice_coil_impedance(model.inductance, model.ts.Re, omega)  # [F] complex128, Ω

    # Steps 2–3: mechanical impedance + coupled cone velocity — textbook convention
    u_textbook = cone_velocity(
        model.ts,
        ze,
        omega,
        voltage=model.voltage,
        box_volume=model.box_volume,
        rho=rho,
        c=c,
    )  # [F] complex128, m/s, exp(+jωt)

    # Step 4: convert to NumCalc's engineering convention exp(−jωt) by conjugation.
    # Magnitude is unchanged; phase flips sign.  This is the one place the
    # convention conversion happens; do not conjugate again downstream.
    # VERIFIED: NumCalc engineering convention — CLAUDE.md; Kreuzer et al. 2024.
    return np.conj(u_textbook)  # [F] complex128, m/s, exp(−jωt)


# ---------------------------------------------------------------------------
# Convenience: list builder for build_dataset()
# ---------------------------------------------------------------------------


def default_terminal_model(name: str = "driver") -> TerminalModel:
    """Build a TerminalModel with standard woofer defaults.

    Defaults mirror TSDialog's spin-box initial values so a click-placed driver
    shows the same numbers when the user opens Edit T/S to tune it.

    Parameters
    ----------
    name : str
        Human-readable driver identifier (typically the driver_id string).

    Returns
    -------
    TerminalModel
        Re=6 Ω, Bl=7 T·m, Mms=12 g, Cms=0.8 mm/N, Rms=1 N·s/m,
        Sd=133 cm², LR-2 inductance Le=0.5 mH / Le2=0.2 mH / Re2=3 Ω.
    """
    ts = TSParams(Re=6.0, Bl=7.0, Mms=0.012, Cms=8e-4, Rms=1.0, Sd=0.0133)
    return TerminalModel(
        ts=ts,
        inductance=LR2Ladder(Le=0.5e-3, Le2=0.2e-3, Re2=3.0),
        name=name,
    )


def terminal_responses_for(
    models: list[TerminalModel],
    frequencies: np.ndarray,
    rho: float = _RHO_DEFAULT,
    c: float = _C_DEFAULT,
) -> list[np.ndarray]:
    """Compute terminal_response for every driver in a list.

    Returns the list in the same order as ``models``, ready to pass directly
    to ``assembly.tensor.build_dataset(terminal_responses=...)``.

    Parameters
    ----------
    models : list of TerminalModel
        One model per driver, in the same order as the BEM ComplexField results.
    frequencies : np.ndarray
        Frequency points, Hz.  Shape [F] float64.
    rho : float
        Air density, kg/m³.
    c : float
        Speed of sound, m/s.

    Returns
    -------
    list of np.ndarray
        Each element shape [F] complex128, engineering exp(−jωt) convention.
    """
    return [terminal_response(m, frequencies, rho=rho, c=c) for m in models]
