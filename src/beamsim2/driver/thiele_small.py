"""Thiele/Small lumped electro-mechanical model: small-signal parameters → cone velocity.

Covers the LF (Thiele/Small) half of the driver model (DR-05).  The result —
complex cone velocity u(ω) for a reference drive — is the ``terminal_response``
scalar multiplied across all BEM directions in ``H_full``.

All impedances here are in the **textbook exp(+jωt)** time convention (standard EE/
acoustics sign: Z_L = +jωL, capacitor = 1/(+jωC)).  The caller (terminal.py)
conjugates once to land in NumCalc's engineering exp(−jωt) convention before
multiplying H_bem.

References
----------
Thiele, A.N., *JAES* 19(5):382–391, 1971.  VERIFIED.
Small, R.H., *JAES* 20(5):383–395, 1972.  VERIFIED.
Kinsler, Frey, Coppens, Sanders, *Fundamentals of Acoustics*, 4th ed., §12.
VERIFIED (electro-acoustic analogies).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

_RHO_DEFAULT = 1.2041  # kg/m³ — dry air 20°C 101325 Pa (core.units.air_density default)
_C_DEFAULT = 343.2  # m/s — dry air 20°C (core.units.speed_of_sound default)


# ---------------------------------------------------------------------------
# Small-signal parameter set
# ---------------------------------------------------------------------------


@dataclass
class TSParams:
    """Fundamental Thiele/Small small-signal parameters.

    All six quantities are the irreducible mechanical/electrical primitives.
    Derived specifications (fs, Qms, Qes, Qts, Vas, sensitivity) are
    computable as property accessors.

    Parameters
    ----------
    Re : float
        DC voice-coil resistance, Ω.
    Bl : float
        Motor force factor (B-field × coil length), T·m.
    Mms : float
        Moving mass (cone + coil + air load), kg.
        HEURISTIC: radiation mass is included in the standard Mms as measured
        by the added-mass method; do not subtract it here — it appears separately
        as radiation resistance in H_bem.  Kinsler et al. §12.3; Small 1972.
    Cms : float
        Mechanical suspension compliance, m/N.
    Rms : float
        Mechanical resistance (viscous losses in surround + spider), N·s/m.
    Sd : float
        Effective piston area of the diaphragm, m².
        NOTE: H_bem was solved with this exact area as the vibrating BEM surface,
        so ``terminal_response`` is cone *velocity* (m/s), not volume velocity —
        no Sd factor is applied here.
    """

    Re: float  # Ω
    Bl: float  # T·m
    Mms: float  # kg
    Cms: float  # m/N
    Rms: float  # N·s/m
    Sd: float  # m²

    # ------------------------------------------------------------------
    # Derived quantities  (VERIFIED: Thiele 1971; Small 1972)
    # ------------------------------------------------------------------

    @property
    def omega_s(self) -> float:
        """Free-air angular resonance, rad/s."""
        return 1.0 / math.sqrt(self.Mms * self.Cms)

    @property
    def fs(self) -> float:
        """Free-air resonance frequency, Hz.
        VERIFIED: fs = 1 / (2π √(Mms·Cms)).
        """
        return self.omega_s / (2.0 * math.pi)

    @property
    def Qms(self) -> float:
        """Mechanical Q.  VERIFIED: Qms = ωs·Mms / Rms."""
        return self.omega_s * self.Mms / self.Rms

    @property
    def Qes(self) -> float:
        """Electrical Q.  VERIFIED: Qes = ωs·Mms·Re / Bl²."""
        return self.omega_s * self.Mms * self.Re / (self.Bl**2)

    @property
    def Qts(self) -> float:
        """Total Q.  VERIFIED: 1/Qts = 1/Qms + 1/Qes."""
        return self.Qms * self.Qes / (self.Qms + self.Qes)

    def vas(self, rho: float = _RHO_DEFAULT, c: float = _C_DEFAULT) -> float:
        """Equivalent compliance volume, m³.  VERIFIED: Vas = ρ·c²·Sd²·Cms.

        Parameters
        ----------
        rho : float
            Air density, kg/m³.
        c : float
            Speed of sound, m/s.

        Returns
        -------
        float
            Vas in m³.
        """
        return rho * c**2 * self.Sd**2 * self.Cms

    # ------------------------------------------------------------------
    # Named constructor — datasheet form
    # ------------------------------------------------------------------

    @classmethod
    def from_datasheet(
        cls,
        *,
        fs: float,
        Qms: float,
        Qes: float | None = None,
        Qts: float | None = None,
        Vas_m3: float,
        Re: float,
        Sd: float,
        rho: float = _RHO_DEFAULT,
        c: float = _C_DEFAULT,
    ) -> "TSParams":
        """Construct from standard loudspeaker datasheet parameters.

        Derive the six fundamental parameters (Re, Bl, Mms, Cms, Rms, Sd) from
        the published specification.  Exactly one of ``Qes`` or ``Qts`` is
        required; if ``Qts`` is given, ``Qes`` is derived from
        ``1/Qts = 1/Qms + 1/Qes``.

        Parameters
        ----------
        fs : float
            Free-air resonance frequency, Hz.
        Qms : float
            Mechanical Q at resonance.
        Qes : float or None
            Electrical Q.  Provide either this or ``Qts``.
        Qts : float or None
            Total Q.  Provide either this or ``Qes``.
        Vas_m3 : float
            Equivalent compliance volume, **m³** (divide litres by 1 000).
        Re : float
            DC voice-coil resistance, Ω.
        Sd : float
            Effective piston area, m².
        rho : float
            Air density, kg/m³.
        c : float
            Speed of sound, m/s.

        Returns
        -------
        TSParams

        Raises
        ------
        ValueError
            If neither or both of ``Qes`` / ``Qts`` are supplied, or if
            ``Qts ≥ Qms`` (unphysical — would require Qes ≤ 0).
        """
        if (Qes is None) == (Qts is None):
            raise ValueError("Supply exactly one of Qes or Qts, not both or neither.")
        if Qts is not None:
            if Qts >= Qms:
                raise ValueError(
                    f"Qts ({Qts}) must be less than Qms ({Qms}); "
                    "1/Qts = 1/Qms + 1/Qes requires Qes > 0."
                )
            Qes = Qms * Qts / (Qms - Qts)

        omega_s = 2.0 * math.pi * fs  # rad/s
        # VERIFIED: Vas = ρ c² Sd² Cms  →  Cms = Vas / (ρ c² Sd²)
        Cms = Vas_m3 / (rho * c**2 * Sd**2)  # m/N
        # VERIFIED: ωs = 1/√(Mms Cms)  →  Mms = 1 / (ωs² Cms)
        Mms = 1.0 / (omega_s**2 * Cms)  # kg
        # VERIFIED: Qms = ωs Mms / Rms  →  Rms = ωs Mms / Qms
        Rms = omega_s * Mms / Qms  # N·s/m
        # VERIFIED: Qes = ωs Mms Re / Bl²  →  Bl = √(ωs Mms Re / Qes)
        Bl = math.sqrt(omega_s * Mms * Re / Qes)  # T·m

        return cls(Re=Re, Bl=Bl, Mms=Mms, Cms=Cms, Rms=Rms, Sd=Sd)


# ---------------------------------------------------------------------------
# Mechanical impedance
# ---------------------------------------------------------------------------


def mechanical_impedance(
    ts: TSParams,
    omega: np.ndarray,
    box_volume: float | None = None,
    rho: float = _RHO_DEFAULT,
    c: float = _C_DEFAULT,
) -> np.ndarray:
    """Mechanical impedance Zm(ω), textbook exp(+jωt) convention.

    ``Zm = jω·Mms + Rms + 1/(jω·Cmt)``

    where the total compliance Cmt is:

    * Free air / infinite baffle (``box_volume=None``): ``Cmt = Cms``
    * Sealed box: ``1/Cmt = 1/Cms + 1/Cab``
      with ``Cab = Vb / (ρ·c²·Sd²)`` the acoustic air-spring on the cone side.
      VERIFIED: Small, R.H., *JAES* 22(10):798–808, 1973 (sealed-box derivation).

    In acoustics terms: the three terms are mass (like water inertia), damping
    (friction), and stiffness (spring restoring force).  The box volume adds an
    extra stiffness term — air trapped in a sealed box acts as a spring on the
    back of the cone, raising the resonance and tightening Q.

    HEURISTIC: radiation mass and radiation resistance from the front-face air
    load are not added here.  They are already captured in the measured Mms and
    in H_bem respectively.  Standard T/S practice (Small 1972).

    Parameters
    ----------
    ts : TSParams
        Small-signal parameters.
    omega : np.ndarray
        Angular frequencies, rad/s.  Shape [F] float64.
    box_volume : float or None
        Internal box volume, m³.  None → free-air / infinite-baffle.
    rho : float
        Air density, kg/m³.
    c : float
        Speed of sound, m/s.

    Returns
    -------
    np.ndarray
        Shape [F] complex128, N·s/m, textbook exp(+jωt) convention.
    """
    if box_volume is not None:
        # Acoustic compliance of the sealed box air spring, referred to the cone
        # VERIFIED: Cab = Vb / (ρ c² Sd²).  Small 1973; Beranek §8.2.
        Cab = box_volume / (rho * c**2 * ts.Sd**2)  # m/N
        Cmt = ts.Cms * Cab / (ts.Cms + Cab)  # series compliance, m/N
    else:
        Cmt = ts.Cms

    jw = 1j * omega  # [F] complex128
    return jw * ts.Mms + ts.Rms + 1.0 / (jw * Cmt)  # [F] complex128


# ---------------------------------------------------------------------------
# Cone velocity (textbook convention)
# ---------------------------------------------------------------------------


def cone_velocity(
    ts: TSParams,
    ze: np.ndarray,
    omega: np.ndarray,
    voltage: float = 2.83,
    box_volume: float | None = None,
    rho: float = _RHO_DEFAULT,
    c: float = _C_DEFAULT,
) -> np.ndarray:
    """Complex cone velocity u(ω) for a terminal voltage, textbook exp(+jωt).

    The coupled electro-mechanical loops (Kirchhoff voltage + Newton's law,
    coupled by motor force Bl·i and back-EMF Bl·u) give:

        u(ω) = Bl · V / ( Ze(ω) · Zm(ω) + Bl² )

    In acoustics terms: this is the piston speed produced by voltage V at the
    voice-coil terminals.  H_bem already encodes how much pressure one m/s
    produces at the observation sphere, so ``H_full = H_bem × u`` gives the
    acoustic transfer function at the reference drive level.

    IMPORTANT: the result is in **textbook exp(+jωt)** convention.
    Call ``terminal_response()`` in ``terminal.py`` for the conjugated
    engineering-convention version that multiplies H_bem.

    VERIFIED: Thiele 1971 eq. (2); Small 1972 eq. (9).

    Parameters
    ----------
    ts : TSParams
        Small-signal parameters.
    ze : np.ndarray
        Blocked electrical impedance Ze(ω), Ω.  Shape [F] complex128.
        Obtain from ``inductance.voice_coil_impedance()``.
    omega : np.ndarray
        Angular frequencies, rad/s.  Shape [F] float64.
    voltage : float
        Drive voltage at the terminals, V_rms.  Default 2.83 V = 1 W into 8 Ω,
        the standard loudspeaker sensitivity reference.
        HEURISTIC: 2.83 V is conventional for 8 Ω drivers; use 2.0 V for 4 Ω.
    box_volume : float or None
        Internal box volume, m³.  None → free-air / infinite-baffle.
    rho : float
        Air density, kg/m³.
    c : float
        Speed of sound, m/s.

    Returns
    -------
    np.ndarray
        Shape [F] complex128, m/s, textbook exp(+jωt) convention.
        **Conjugate before multiplying H_bem** (see terminal.py).
    """
    zm = mechanical_impedance(ts, omega, box_volume=box_volume, rho=rho, c=c)
    # VERIFIED: u = Bl·V / (Ze·Zm + Bl²).  Thiele 1971; Small 1972.
    return ts.Bl * voltage / (ze * zm + ts.Bl**2)  # [F] complex128
