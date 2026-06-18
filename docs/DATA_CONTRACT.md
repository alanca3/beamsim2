# BeamSimII — Data Contract (Phase 1 → Phase 2)

> Extracted from `BEAMSIMII_Gameplan.md §3`. This is the standalone reference for
> the output data schema. **Do not edit here without updating §3.**

---

## Purpose

This contract defines every array, attribute, and on-disk layout in the BeamSimII
output file. It is designed backward from what the Phase-2 beamformer consumes:
complex per-driver directional responses that become columns of a steering matrix,
integrated over the sphere to build "accept" and "reject" covariance matrices.

**VERIFIED** need: Luo, *Constant Directivity Loudspeaker Beamforming*, EUSIPCO 2024,
arXiv:2407.01860; Feistel et al., AES Paper 7254, 2007.

---

## 3.1 The core tensor

```
H : [ driver  ×  frequency  ×  direction ]   complex128
      M             F            N
```

- **M** = number of drivers. **F** = frequency bins. **N** = sphere directions.
- Each `H[m, :, :]` is driver *m*'s full directional response — the block of columns
  Phase 2 assembles into its steering matrix.
- Stored **per-driver** on disk so drivers can be solved and reused independently.

Two companion arrays per driver (nothing entangled, everything auditable):

| Field | Shape | Dtype | Description |
|---|---|---|---|
| `H_bem` | `[F × N]` | complex128 | Raw geometric response at unit cone velocity (BEM only; baffle / box / diffraction / aperture). Reusable without re-solving. |
| `terminal_response` | `[F]` | complex128 | Per-driver complex on-axis multiplier (T/S LF + semi-inductance HF, DR-05). Broadcast over directions. |
| `H_full` | `[F × N]` | complex128 | `H_bem × terminal_response[:, None]`. Phase-2 default. |

---

## 3.2 Sphere sampling and integration weights

**Default scheme:** Lebedev quadrature. Fliege–Maier, spherical t-designs, and
icosphere are selectable alternatives.

**Why Lebedev:** comes with exact quadrature weights, so integrating any function
over the sphere reduces to `Σ wᵢ · f(directionᵢ)` with no ad-hoc cos(θ) correction.
Radiated power, directivity index, and Phase-2 covariance matrices all become
simple weighted dot products.

**VERIFIED** rationale (First Research report; standard spherical-quadrature practice).
Never a naive latitude/longitude grid (oversamples poles, integrates poorly).

Stored under a `directions` group:

| Array | Shape | Dtype | Description |
|---|---|---|---|
| `unit_vectors` | `[N × 3]` | float64 | Cartesian direction cosines. Each row is a unit vector. |
| `weights` | `[N]` | float64 | Quadrature weights. Sum to 4π (or 1; convention in attribute). |
| `theta_phi` | `[N × 2]` | float64 | Convenience spherical coordinates (convention recorded). |

Attributes: `scheme` (e.g. "lebedev"), `order`, `weight_convention`.

**Resolution guidance.** Use one fixed grid sized for `f_max` (or `bem_cap_hz`
if splicing analytic tail). Required SH order ≈ `⌈k · r_source⌉ + margin`, where
`k = 2πf/c`. **HEURISTIC** (spherical-array/SH literature; First Research report).

---

## 3.3 Frequency grid

| Field | Shape | Dtype | Description |
|---|---|---|---|
| `frequencies` | `[F]` | float64 | Explicit array in Hz. |

Attributes: `spacing` ("log" / "linear" / "fractional-octave"), `fractional_octave`.

**Default:** logarithmic, 1/12-octave. Directivity varies smoothly enough with
frequency that 1/12-octave is sufficient for loudspeaker work.

**Sparse-simulate + interpolate** supported: simulate a sparse subset, interpolate
to the full stored grid in SH + minimum-phase domain, mark interpolated bins in
`interpolated_mask [F] bool`. **VERIFIED** saves ~86% compute but requires care
to avoid artifacts (First Research report).

---

## 3.4 The single-phase-origin rule (highest-risk rule)

Every value in `H_bem[m]` is the complex pressure at the observation point (radius
`r_obs`, direction) produced by driver *m* vibrating at unit normal velocity, **with
phase referenced to:**

1. A **single common time origin** (excitation at `t = 0`)
2. A **single common spatial origin** — conventionally `(0, 0, 0)` in global
   coordinates

Because driver *m* sits at position `p_m ≠ origin`, its responses **naturally
carry path-length phase** corresponding to its real location. **We do not remove
that phase. We do not minimum-phase-ify or re-zero any driver independently.**
The inter-driver phase differences *are* the beamforming steering information.

**Why this matters.** A two-driver array's beam direction is encoded in the phase
difference between `H_bem[0]` and `H_bem[1]`. If either is re-zeroed, the
beamformer receives wrong steering information and the beam points somewhere else —
silently. This is §3's highest-risk rule because violations are not obvious from
polar plots of a single driver.

**Why simulation makes this easy.** In anechoic measurement the acoustic centre
rarely coincides with the turntable pivot, producing frequency-dependent phase
errors that must be corrected. In simulation, we define one origin and keep every
driver referenced to it — the geometry gives the correct time-of-flight for free.
**VERIFIED** (Trott 1977 acoustic-center definition; VituixCAD dual-channel note;
First Research report).

**Optional decomposition** (allowed, origin is sacred). A driver response may be
factored into minimum-phase × excess-phase (pure time-of-flight delay), as long
as the common origin and true delay are preserved and recombine exactly.

**Enforcement in code.** A dedicated assertion in `assembly/phase_origin.py`:
assemble a synthetic two-source field, compare summed `H` against a direct
two-source BEM solve, require agreement within tolerance (Stage-3 acceptance test,
§8). Any refactor that re-zeros a driver fails this test loudly.

---

## 3.5 Required metadata (HDF5 attributes)

Everything needed to interpret the file without external context.

**Root level:**
```
schema_version         str
beamsim_version        str
created_utc            str  (ISO 8601)
solver_backend         str  (e.g. "numcalc")
solver_version         str
phase_origin           [3] float64  (always [0,0,0])
axis_convention        str  (e.g. "x=right, y=front, z=up")
length_units           str  ("metres")
```

**Per driver** (`/drivers/<id>/attrs`):
```
name                   str
position               [3] float64  (metres)
orientation            [3] float64  (outward normal unit vector)
radius                 float        (metres)
profile                str          ("piston" or "cap" + parameters)
ts_params              dict         (full small-signal parameter set)
terminal_response_model str         (e.g. "thorborg-futtrup")
diaphragm_area         float        (m²)
```

**Observation:**
```
observation_radius     float  (metres; e.g. 1.0)
far_field              bool
pressure_convention    str    ("Pa at r_obs for unit cone velocity" for H_bem)
```

**Medium / reference conditions:**
```
speed_of_sound         float  (m/s)
air_density            float  (kg/m³)
temperature            float  (°C)
humidity               float  (% RH)
pressure               float  (Pa)
air_attenuation_model  str    ("none" or "iso9613-1")
```

---

## 3.6 On-disk layout (HDF5)

Native working format: `.h5` (or branded `.bsim` extension; HDF5 inside).

```
/                                  (root; all root attributes above)
  /frequencies                     [F] float64
  /directions/
      unit_vectors                 [N×3] float64
      weights                      [N]   float64
      theta_phi                    [N×2] float64
      attrs: scheme, order, weight_convention
  /drivers/
      /<driver_id>/
          H_bem                    [F×N] complex128
          terminal_response        [F]   complex128
          H_full                   [F×N] complex128
          attrs: name, position, orientation, radius, profile, ts_params, ...
```

### Export formats

| Format | Notes |
|---|---|
| HDF5 `.h5` / `.bsim` | Always written. The native working format. |
| `.frd` per driver/angle | VituixCAD import (freq / mag dB / phase deg, text). **VERIFIED.** |
| CLF balloon | Open room-acoustics interop format. **VERIFIED.** |
| SOFA AES69 (optional) | HDF5-based; natural later addition. |

---

*Contract is concrete enough that Phase 2 can be written against it today.*
