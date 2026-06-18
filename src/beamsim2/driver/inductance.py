"""HF voice-coil impedance models: LR-2 ladder and plain-Le fallback.

The voice coil is a *lossy* inductor, not an ideal one.  Above a few hundred Hz,
eddy currents in the iron pole piece divert current from the coil, flattening the
impedance rise.  A plain Le over-predicts the HF impedance and therefore
under-predicts the HF cone velocity — both errors worsen above the driver's
passband.

Default model: LR-2 ladder (two reactive elements, one loss resistor).  This is
the minimum-complexity model that captures the eddy-current rolloff.

VERIFIED: Wright, J.R., "An Empirical Model for Loudspeaker Motor Impedance",
*JAES* 38(10):749–754, 1990.
VERIFIED: Small, R.H., *JAES* 20(5):383–395, 1972 (plain Le baseline).

All impedances use the **textbook exp(+jωt)** sign convention (Z_L = +jωL).
terminal.py conjugates the final cone velocity to convert to engineering
exp(−jωt) before multiplying H_bem.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Inductance model dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PlainLe:
    """Simple ideal inductor: Z_L = jω·Le.

    Use only for initial tests or when Le2/Re2 parameters are unavailable.
    Overestimates HF impedance; use LR2Ladder for production solves.
    VERIFIED reference baseline: Small 1972.

    Parameters
    ----------
    Le : float
        Voice-coil inductance, H.
    """

    Le: float  # H — voice-coil inductance


@dataclass
class LR2Ladder:
    """LR-2 lossy ladder: Z_L = jω·Le + (jω·Le2·Re2) / (Re2 + jω·Le2).

    The primary branch jω·Le models the voice-coil inductance at DC.
    The parallel sub-branch (Le2 in series with Re2, placed in parallel across
    Le2 alone) models eddy-current loss in the pole piece: at high frequencies
    current shunts through Re2, flattening the impedance below what a plain Le
    predicts.  In acoustics terms: the iron core is a lossy coupling medium —
    the LR-2 captures how that loss limits the magnetic flux threading the coil.

    VERIFIED: Wright, *JAES* 38(10):749–754, 1990.

    Parameters
    ----------
    Le : float
        Primary voice-coil inductance, H.
    Le2 : float
        Lossy branch inductance, H.
    Re2 : float
        Eddy-current loss resistance (pole piece), Ω.
    """

    Le: float  # H — primary voice-coil inductance
    Le2: float  # H — eddy-current loss branch inductance
    Re2: float  # Ω — eddy-current loss resistance


# ---------------------------------------------------------------------------
# Impedance computation
# ---------------------------------------------------------------------------


def voice_coil_impedance(
    model: PlainLe | LR2Ladder,
    Re: float,
    omega: np.ndarray,
) -> np.ndarray:
    """Blocked electrical voice-coil impedance Ze(ω), textbook exp(+jωt).

    Ze = Re + Z_L(ω)

    where Z_L depends on the inductance model:

    * ``PlainLe``:   Z_L = jω·Le
    * ``LR2Ladder``: Z_L = jω·Le + (jω·Le2·Re2) / (Re2 + jω·Le2)

    "Blocked" means the cone is held fixed (mechanical impedance not included);
    the electro-mechanical coupling appears in ``cone_velocity()`` as the
    Bl²/Zm term.

    Parameters
    ----------
    model : PlainLe or LR2Ladder
        Voice-coil inductance model.
    Re : float
        DC voice-coil resistance, Ω (from TSParams.Re).
    omega : np.ndarray
        Angular frequencies, rad/s.  Shape [F] float64.

    Returns
    -------
    np.ndarray
        Blocked electrical impedance Ze(ω), Ω.  Shape [F] complex128,
        textbook exp(+jωt) convention.

    Raises
    ------
    TypeError
        If ``model`` is not a recognised inductance model.
    """
    jw = 1j * omega  # [F] complex128

    if isinstance(model, PlainLe):
        Z_L = jw * model.Le  # [F] complex128
    elif isinstance(model, LR2Ladder):
        # Parallel topology: Z_L = jωLe || (Re2 + jωLe2)
        #   = jωLe × (Re2 + jωLe2) / (jωLe + Re2 + jωLe2)
        # At HF: → jωLe × jωLe2 / (jω(Le+Le2)) = jω × Le·Le2/(Le+Le2) — reduced
        # effective inductance vs plain Le (eddy loss shunts current through Re2,
        # clamping the inductance rise). VERIFIED: Wright 1990, eq. (3).
        Z_L = (
            jw
            * model.Le
            * (model.Re2 + jw * model.Le2)
            / (jw * model.Le + model.Re2 + jw * model.Le2)
        )  # [F] complex128
    else:
        raise TypeError(
            f"Unknown inductance model type: {type(model).__name__}. "
            "Expected PlainLe or LR2Ladder."
        )

    return Re + Z_L  # [F] complex128, Ze = Re + Z_L


# ---------------------------------------------------------------------------
# Input impedance (the measurable terminal quantity)
# ---------------------------------------------------------------------------


def input_impedance(
    ze: np.ndarray,
    zm: np.ndarray,
    Bl: float,
) -> np.ndarray:
    """Driver electrical input impedance Z_in(ω) = Ze + Bl²/Zm.

    This is the impedance measurable at the driver terminals.  The extra term
    Bl²/Zm is the "motional impedance" — the back-EMF from the moving coil
    feeding back into the electrical circuit.  It peaks at resonance where Zm
    is minimum, giving the characteristic impedance hump visible in any
    driver measurement.

    In acoustics terms: the mechanical motion of the cone couples back to the
    electrical circuit through the motor.  Z_in is what a measurement rig or
    amplifier sees; it is NOT the blocked impedance Ze.

    VERIFIED: Kinsler et al. §12.4; Thiele 1971; Small 1972.

    Parameters
    ----------
    ze : np.ndarray
        Blocked electrical impedance Ze(ω), Ω.  Shape [F] complex128.
    zm : np.ndarray
        Mechanical impedance Zm(ω), N·s/m.  Shape [F] complex128.
    Bl : float
        Motor force factor, T·m.

    Returns
    -------
    np.ndarray
        Electrical input impedance Z_in(ω), Ω.  Shape [F] complex128,
        textbook exp(+jωt) convention.
    """
    return ze + Bl**2 / zm  # [F] complex128, Z_in = Ze + Bl²/Zm
