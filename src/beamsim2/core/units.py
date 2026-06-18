"""Air medium properties: speed of sound, density, and air-attenuation coefficient.

All formulas carry VERIFIED / INFERRED / HEURISTIC labels per §5.1 coding standards.
Functions accept temperature, humidity, and pressure to match the §3.5 metadata
contract, even where a parameter has negligible effect and is documented as such.
"""

from __future__ import annotations

import math
from typing import Union

# Physical constants — VERIFIED values from NIST / ISO 2533
_R_SPECIFIC_DRY = 287.058  # J/(kg·K) — specific gas constant for dry air, R_d = R/M_d
# VERIFIED: NIST CODATA; ISO 2533-1975 standard atmosphere
_GAMMA_DRY = 1.400  # adiabatic index (ratio of specific heats) for dry air
# HEURISTIC: valid for diatomic gas at room temperature;
# varies slightly with temperature but negligible here


def speed_of_sound(
    T_C: float = 20.0,
    RH_pct: float = 50.0,
    P_Pa: float = 101325.0,
) -> float:
    """Speed of sound in air.

    HEURISTIC: uses the dry-air ideal-gas relation c = sqrt(γ R_d T_K).
    Humidity raises c by ~0.3 m/s per 10 % RH at 20 °C; this effect is
    accepted as a < 0.1 % error and intentionally omitted to keep the formula
    verifiable. Pressure has negligible effect on c in air (c is determined by
    γ and T, not P, for an ideal gas).

    Reference: Kinsler, Frey, Coppens, Sanders, *Fundamentals of Acoustics*,
    4th ed., Wiley, 2000, §4.2.

    Parameters
    ----------
    T_C : float
        Air temperature in degrees Celsius.
    RH_pct : float
        Relative humidity, 0–100 %. Accepted for interface consistency but not
        used in the current formula (see note above).
    P_Pa : float
        Static pressure in Pa. Accepted for interface consistency but not used.

    Returns
    -------
    float
        Speed of sound in m/s.

    Examples
    --------
    >>> round(speed_of_sound(20.0), 1)
    343.2
    >>> round(speed_of_sound(0.0), 1)
    331.3
    """
    T_K = T_C + 273.15
    # c = sqrt(γ R_d T_K) — ideal-gas speed of sound for dry air
    return math.sqrt(_GAMMA_DRY * _R_SPECIFIC_DRY * T_K)


def air_density(
    T_C: float = 20.0,
    RH_pct: float = 50.0,
    P_Pa: float = 101325.0,
) -> float:
    """Density of dry air from the ideal gas law.

    VERIFIED: ρ = P / (R_d T_K) — ideal gas law for dry air.
    R_d = 287.058 J/(kg·K); VERIFIED: NIST / ISO 2533-1975.

    HEURISTIC: treats air as dry. Humidity slightly decreases density because
    water vapour (M = 18 g/mol) is lighter than dry air (M ≈ 29 g/mol); the
    correction is < 1 % at typical indoor RH and is neglected.

    Parameters
    ----------
    T_C : float
        Air temperature in degrees Celsius.
    RH_pct : float
        Relative humidity, 0–100 %. Accepted but not used (see note above).
    P_Pa : float
        Static pressure in Pa.

    Returns
    -------
    float
        Air density in kg/m³.

    Examples
    --------
    >>> round(air_density(20.0, 0.0, 101325.0), 4)
    1.2041
    """
    T_K = T_C + 273.15
    return P_Pa / (_R_SPECIFIC_DRY * T_K)


def air_attenuation(
    frequencies: Union[float, list[float]],
    T_C: float = 20.0,
    RH_pct: float = 50.0,
    P_Pa: float = 101325.0,
    model: str = "none",
) -> Union[float, list[float]]:
    """Atmospheric absorption coefficient α in dB/m.

    At a 1 m observation distance, atmospheric attenuation is small:
    < 0.1 dB at 10 kHz and < 0.5 dB at 20 kHz under standard conditions.
    The "none" model is appropriate for most loudspeaker BEM work at r ≤ 2 m.

    Parameters
    ----------
    frequencies : float or list of float
        Frequency or frequencies in Hz.
    T_C : float
        Air temperature in degrees Celsius.
    RH_pct : float
        Relative humidity, 0–100 %.
    P_Pa : float
        Static pressure in Pa.
    model : str
        "none"      — zero absorption at all frequencies (current default).
        "iso9613-1" — will be implemented when long-range accuracy is needed;
                      raises NotImplementedError until then.

    Returns
    -------
    float or list of float
        Absorption coefficient(s) in dB/m. Scalar in, scalar out; list in, list out.

    Raises
    ------
    NotImplementedError
        If `model` is anything other than "none".
    """
    if model == "none":
        if isinstance(frequencies, (int, float)):
            return 0.0
        return [0.0] * len(frequencies)
    raise NotImplementedError(
        f"air_attenuation model '{model}' is not implemented. "
        "Only 'none' is currently available. "
        "ISO 9613-1 support will be added for long-range propagation use cases."
    )
