"""Target beam specification and target-field construction (Stage P2-1).

A :class:`TargetSpec` is the user's request: a beam *shape* (preset, a continuous
cardioid order, or an arbitrary custom pattern), a *steering direction*, optional
*nulls*, a design *band*, the chosen solver *engine*, and the single *robustness*
knob. :func:`build_target` turns it into a complex target field ``b_f[F, N]`` on the
dataset's sphere grid and/or accept/reject angular masks the solvers integrate over.

The first-order pattern family is ``T(theta) = a + (1 - a) * cos(theta)`` with
``a in [0, 1]`` (a=1 omni, a=0.5 cardioid, a=0.25 supercardioid, a=0 figure-8),
measured from the steering direction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from beamsim2.validation.closed_loop import monopole_field

# Preset name -> first-order coefficient a in T(theta) = a + (1-a) cos(theta).
PRESET_ORDER_A: dict[str, float] = {
    "omni": 1.0,
    "subcardioid": 0.7,
    "cardioid": 0.5,
    "supercardioid": 0.366,  # max front-to-back ratio (a = (sqrt(3)-1)/2)
    "hypercardioid": 0.25,  # max directivity index (first order)
    "figure8": 0.0,
}


@dataclass
class TargetSpec:
    """A user request for a beam shape + steering, plus solver/robustness choices.

    Parameters
    ----------
    mode : str
        ``"preset" | "cardioid_order" | "steering_only" | "custom"``.
    preset : str | None
        One of :data:`PRESET_ORDER_A` keys when ``mode == "preset"``.
    order_a : float | None
        First-order coefficient ``a in [0, 1]`` when ``mode == "cardioid_order"``.
    steer_dir : np.ndarray
        Unit beam direction ``[3]`` float64 (the main-lobe axis).
    nulls : list[np.ndarray]
        Unit directions ``[3]`` to place nulls (soft target zeros or LCMV hard nulls).
    custom_target : np.ndarray | None
        ``[N]`` (or ``[F, N]``) complex/real desired pattern when ``mode == "custom"``.
    band_hz : tuple[float, float]
        Design band ``(f_lo, f_hi)``; out-of-band handling per engine.
    wng_floor_db : float
        The single robustness knob: white-noise-gain floor in dB (see ``regularize``).
    accept_halfangle_deg : float
        Half-angle of the "accept" cap about the steering direction (constant-DI engine #2).
    target_gdi_db : float | None
        Desired constant generalized directivity index (dB) for ``engine="constant_di"``.
        ``None`` -> use the maximum feasible value (the min over frequency of the ceiling).
    engine : str
        ``"delay_sum" | "ls" | "mvdr" | "lcmv" | "max_directivity" | "constant_di"``.
    """

    mode: str = "preset"
    preset: str | None = "cardioid"
    order_a: float | None = None
    steer_dir: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    nulls: list[np.ndarray] = field(default_factory=list)
    custom_target: np.ndarray | None = None
    band_hz: tuple[float, float] = (20.0, 20000.0)
    wng_floor_db: float = -6.0
    accept_halfangle_deg: float = 60.0
    target_gdi_db: float | None = None
    engine: str = "ls"


@dataclass
class Target:
    """The resolved target a solver consumes (output of :func:`build_target`).

    Attributes
    ----------
    b_field : np.ndarray
        ``[F, N]`` complex128 — desired pressure pattern on the grid (LS engine).
    look_idx : int
        Grid index nearest the steering direction (MVDR/LCMV/matched-field).
    null_idx : list[int]
        Grid indices nearest each requested null (LCMV hard nulls / soft zeros).
    accept_mask : np.ndarray
        ``[N]`` float64 in {0,1} — the "accept"/target angular region (Luo engine #2).
    reject_mask : np.ndarray
        ``[N]`` float64 in {0,1} — the "reject" region (whole sphere by default).
    pattern : np.ndarray
        ``[N]`` float64 — the frequency-independent directivity shape (for reference/plots).
    """

    b_field: np.ndarray
    look_idx: int
    null_idx: list[int]
    accept_mask: np.ndarray
    reject_mask: np.ndarray
    pattern: np.ndarray


def _first_order_pattern(cos_ang: np.ndarray, a: float) -> np.ndarray:
    """First-order directivity ``T = a + (1 - a) cos(angle)`` (signed; rear lobe allowed)."""
    return a + (1.0 - a) * cos_ang


def _shape_pattern(spec: TargetSpec, cos_ang: np.ndarray) -> np.ndarray:
    """Resolve the (real, signed) directivity pattern over the grid from ``spec``."""
    if spec.mode == "custom":
        if spec.custom_target is None:
            raise ValueError("TargetSpec.mode == 'custom' requires custom_target.")
        # Real part for the plot `pattern` only (the solver uses the complex b_field);
        # use `.real` rather than a float cast so a complex custom target raises no warning.
        return np.asarray(spec.custom_target).real.astype(np.float64)
    if spec.mode == "cardioid_order":
        if spec.order_a is None:
            raise ValueError("TargetSpec.mode == 'cardioid_order' requires order_a.")
        return _first_order_pattern(cos_ang, float(spec.order_a))
    if spec.mode == "steering_only":
        # A moderately narrow lobe (hypercardioid) — pure pointing without a named shape.
        return _first_order_pattern(cos_ang, 0.25)
    # mode == "preset"
    preset = spec.preset or "cardioid"
    if preset in PRESET_ORDER_A:
        return _first_order_pattern(cos_ang, PRESET_ORDER_A[preset])
    if preset == "wide":
        return _first_order_pattern(cos_ang, 0.7)  # subcardioid (broad)
    if preset == "narrow":
        # Narrower than first order allows: a non-negative raised-cosine squared lobe.
        return (0.5 * (1.0 + cos_ang)) ** 2
    raise ValueError(f"Unknown preset {preset!r}.")


def _first_order_a(spec: TargetSpec) -> float | None:
    """The first-order coefficient ``a`` implied by ``spec``, or ``None`` if the shape
    is not a first-order ``a + (1-a) cos`` pattern (custom / ``'narrow'`` preset).

    Used to route first-order requests through the exact complex virtual-source target
    (:func:`build_virtual_target`) while non-first-order shapes fall back to the
    phase-referenced shape path in :func:`build_target`.
    """
    if spec.mode == "cardioid_order":
        return float(spec.order_a)  # validated in _shape_pattern
    if spec.mode == "steering_only":
        return 0.25  # hypercardioid pointing lobe
    if spec.mode == "preset":
        preset = spec.preset or "cardioid"
        if preset in PRESET_ORDER_A:
            return PRESET_ORDER_A[preset]
        if preset == "wide":
            return 0.7
    return None  # custom, 'narrow', or unknown -> not a first-order pattern


def _shape_pattern_complex(spec: TargetSpec, cos_ang: np.ndarray) -> np.ndarray:
    """As :func:`_shape_pattern` but preserves a *complex* ``custom_target`` verbatim.

    The original ``_shape_pattern`` casts custom targets through ``np.real`` (the Chunk-3a
    defect #1): that silently discards the imaginary part a complex desired pattern needs.
    This variant keeps it complex. Non-custom shapes are returned as the real pattern cast
    to complex128. Used only by the non-first-order / custom branch of :func:`build_target`,
    which attaches a shared time-of-flight phase reference.
    """
    if spec.mode == "custom":
        if spec.custom_target is None:
            raise ValueError("TargetSpec.mode == 'custom' requires custom_target.")
        return np.asarray(spec.custom_target, dtype=np.complex128)  # [N] keep complex
    return _shape_pattern(spec, cos_ang).astype(np.complex128)  # [N]


def build_virtual_target(
    directions,
    frequencies: np.ndarray,
    steer_unit: np.ndarray,
    look_idx: int,
    a: float,
    c_sound: float,
    *,
    eps: float = 1e-4,
) -> np.ndarray:
    """Complex, frequency-dependent first-order virtual-source target ``b_f[F, N]``.

    The near/far field of an ideal first-order source (a monopole plus a normalized
    point dipole) co-located at the global origin, synthesized with the **same**
    :func:`~beamsim2.validation.closed_loop.monopole_field` forward operator the LS
    pressure-match inverts. Because the target is built from the same point-source model,
    it lives in (or very near) the array's reachable subspace and carries the correct
    complex angular structure **and** the finite-radius radial phase ``exp(+jk r_obs)/r_obs``
    automatically (engineering convention exp(-jωt), outgoing ~ exp(+jkr)).

    Construction::

        b_f[n] = a * g_mono[f, n] + (1 - a) * g_dip[f, n] / s_f
        s_f    = g_dip[f, look] / g_mono[f, look]     (normalize dipole to unit on-axis)

    where ``g_mono`` is a unit monopole at the origin and ``g_dip`` is a steer-aligned
    point dipole formed as the centered difference of two opposed monopoles at
    ``± eps/2 * steer_unit`` divided by ``eps``.

    Why this and not the old real broadcast: a real ``a + (1-a) cos`` target and this
    complex target give the *same* per-frequency directivity for a small array (the LS
    absorbs any global complex scale/phase). The complex virtual-source target's real
    payoff is **cross-frequency realizability** — the recovered filters ``w_m(f)`` come out
    smooth (short, causal), 30× smoother than the old real-target filters
    (``docs/Chunk3a_Findings.md``). VERIFIED: ``b_f / g_mono == a + (1-a) cos`` to ~6e-8 and
    ``arg(b_look / g_mono_look) == 0`` to ~7e-17 rad against the real forward model.

    Parameters
    ----------
    directions : ObservationPoints
        Sphere grid (``unit_vectors[N, 3]`` float64, ``radius`` m).
    frequencies : np.ndarray
        ``[F]`` float64 Hz.
    steer_unit : np.ndarray
        ``[3]`` float64 unit steering direction (already normalized by the caller).
    look_idx : int
        Grid index of the look direction (used to normalize the dipole to unit on-axis).
    a : float
        First-order coefficient (``a=1`` omni, ``0.5`` cardioid, ``0`` figure-8).
    c_sound : float
        Speed of sound (m/s). MUST match the dataset's BEM speed of sound.
    eps : float
        Finite-difference half-spacing for the point dipole (m); result is insensitive
        for ``eps`` in 1e-3..1e-6.

    Returns
    -------
    np.ndarray
        ``b_field[F, N]`` complex128.
    """
    origin = np.zeros((1, 3))  # [1, 3] virtual source at the global origin
    g_mono = monopole_field(origin, directions, frequencies, c_sound)[0]  # [F, N]
    p_plus = (+eps / 2.0) * steer_unit[None, :]  # [1, 3]
    p_minus = (-eps / 2.0) * steer_unit[None, :]  # [1, 3]
    g_dip = (
        monopole_field(p_plus, directions, frequencies, c_sound)[0]
        - monopole_field(p_minus, directions, frequencies, c_sound)[0]
    ) / eps  # [F, N] steer-aligned point dipole (centered difference)
    s = g_dip[:, look_idx] / g_mono[:, look_idx]  # [F] dipole-on-axis normalization
    b = a * g_mono + (1.0 - a) * g_dip / s[:, None]  # [F, N] complex128
    return b


def build_target(
    spec: TargetSpec, directions, frequencies: np.ndarray, *, c_sound: float = 343.2
) -> Target:
    """Build the :class:`Target` (field + look/null indices + accept/reject masks).

    Parameters
    ----------
    spec : TargetSpec
        The user request.
    directions : ObservationPoints
        Sphere grid (``unit_vectors[N, 3]``, ``weights[N]``).
    frequencies : np.ndarray
        ``[F]`` float64 Hz. The target field is **complex and frequency-dependent**
        (Chunk-3a fix): first-order shapes use the exact virtual-source target
        (:func:`build_virtual_target`); other shapes carry a shared time-of-flight phase
        reference. ``pattern`` (for plots) stays the real signed ``a + (1-a) cos``.
    c_sound : float
        Speed of sound (m/s) for the virtual-source / phase-reference construction; MUST
        match the dataset's BEM speed of sound. ``design()`` passes the dataset value;
        the keyword default keeps older positional callers working.

    Returns
    -------
    Target
    """
    uv = directions.unit_vectors  # [N, 3]
    steer = np.asarray(spec.steer_dir, dtype=np.float64)
    steer = steer / np.linalg.norm(steer)
    cos_ang = uv @ steer  # [N] cos(angle from steering axis)

    pattern = _shape_pattern(spec, cos_ang)  # [N] real signed (plots / back-compat)
    look_idx = int(np.argmax(cos_ang))  # needed by the virtual-source target below

    a_coef = _first_order_a(spec)
    if a_coef is not None:
        # First-order family -> exact complex, frequency-dependent virtual-source target.
        b_field = build_virtual_target(
            directions, frequencies, steer, look_idx, a_coef, c_sound
        )  # [F, N] complex128
    else:
        # Non-first-order shape (custom complex pattern, or 'narrow' preset): carry the
        # load-bearing time-of-flight phase via the shared origin-monopole reference (a
        # per-frequency global complex factor -> cardinal-rule safe), NEVER a real broadcast.
        g_mono = monopole_field(np.zeros((1, 3)), directions, frequencies, c_sound)[0]  # [F, N]
        phase_ref = g_mono / np.abs(g_mono)  # [F, N] unit-modulus ToF phase
        shape_c = _shape_pattern_complex(spec, cos_ang)  # [N] complex (custom kept complex)
        b_field = shape_c[None, :] * phase_ref  # [F, N] complex128

    null_idx = [int(np.argmax(uv @ (np.asarray(d, float) / np.linalg.norm(d)))) for d in spec.nulls]

    # Accept = forward cap about the steering axis; reject = whole sphere.
    accept_mask = (cos_ang >= np.cos(np.deg2rad(spec.accept_halfangle_deg))).astype(np.float64)
    reject_mask = np.ones(uv.shape[0], dtype=np.float64)  # [N]

    return Target(
        b_field=b_field,
        look_idx=look_idx,
        null_idx=null_idx,
        accept_mask=accept_mask,
        reject_mask=reject_mask,
        pattern=pattern,
    )
