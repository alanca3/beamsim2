# Chunk 2 — Results Visualization & Diagnostics — Design Note

Status: **in progress** (branch `feature/chunk2-results-viz`). Implements `docs/Bug_Fix_Proposal.md`
Chunk 2 (#9, #10, #11) + a far-field directivity display option. Built on Chunk 1's proven
`reference_axis` metadata + `core.sphere.nearest_direction_index` + logging foundation.

## Diagnose-first verdict (confirmed in code + real data)
- **#10 `_PolarView`** masked ~3–6 scattered Lebedev/icosphere points by a crude `|cos θ|<0.25`
  band and line-plotted them sorted-by-angle → jagged; also hardcoded +z planes.
- **#9 `_DirectivityMapView`** sorted all N points by θ only (mixing all azimuths), `imshow` on a
  **linear** frequency axis, single map; hardcoded +z.
- Real `HDF5/Dr1.h5` is the corrupt multi-driver file Chunk 1 flagged (`driver_order` duplicates
  `driver_4`, one group) — Chunk 1's read guard now correctly **rejects** it. It predates
  `reference_axis` (falls back to +z); grid = icosphere-4 (2562 pts), `r_obs = 2.0`.
  → Chunk 2 verifies against analytic synthetic datasets (CI-safe).

## Reference frame (shared by polar, sonograms, CEA2034)
`core.sphere.reference_frame(reference_axis) -> (front, right, up)`:
- `front` = normalized `reference_axis` (loudspeaker front; default +z).
- `up` = world +z, unless `front` ∥ +z, then world +y (degenerate-axis fallback).
- `up_perp` = component of `up` ⟂ `front`, normalized; `right = cross(up_perp, front)`.
- Horizontal orbit: `cos α·front + sin α·right`. Vertical orbit: `cos β·front + sin β·up_perp`.
- **Load-bearing sign:** +vertical = UP (toward `up`) so CEA ceiling lands at +β, floor at −β.

## #10 Polar (SH-resampled arcs)
Replace scattered-point masking with **SH resample** (`core.sh_transform.forward_sh`/`inverse_sh`,
order = `min(safe_order_for_grid(N), 19)`) onto a 361-pt great-circle arc in the H or V plane built
from the reference frame. One SH fit per frequency, evaluated on the arc (mirrors
`io.filter_export.resample_to_arcs`). Smooth closed outline.

## #9 Directivity sonograms (H + V, log-f)
Replace the single θ-sorted linear-f map with **two sonograms** (horizontal + vertical plane). For
each: SH-resample the field onto a fine arc (angle −180..180°) per frequency → `[F, A]`; `pcolormesh`
with **log** frequency axis (y), angle (x), color = normalized dB. Legible, plane-separated.

## #11 H_bem vs H_full (in-UI)
Field selector already exists on On-axis; add it to polar/sonogram/balloon with a tooltip quoting the
data contract: `H_bem` = raw BEM at unit cone velocity (geometry only); `H_full = H_bem ×
terminal_response` (T/S + voice-coil), the Phase-2 default.

## CEA2034 / spinorama panel (`metrics/cea2034.py`)
**Authoritative source:** `pierreaubert/spinorama` master `compute_cea2034.py` (verified this session,
hand-computed test values reproduced). All spatial averages are **power-domain** (pressure²):
`spatial_average(p) = pressure2spl(rms(spl2pressure(dB)))`.

- **Measurement grid:** H orbit 0..350° + V orbit 0..350° at 10°, sharing on-axis(0°) & rear(180°) →
  70 unique points. Built from the reference frame; SH-resampled from the dataset grid.
- **On-Axis:** H 0°.
- **Listening Window (LW):** unweighted power-RMS of H{0,±10,±20,±30} + V{±10} (9 curves).
- **Early Reflections (ER):** two-level power average. Each bounce group = unweighted power-RMS of
  its angles; ER = unweighted power-RMS of the **5 group curves** (each bounce weighted equally,
  not each angle). Groups: floor V{−20,−30,−40}; ceiling V{+40,+50,+60}; front H{0,±10,±20,±30};
  side H{±40..±80}; rear H{±90..±170,180}.
- **Sound Power (SP):** area-weighted power-RMS over all 70 points; weight = solid angle of the 10°
  latitude band at |orbit angle| (table below); V 0° and V 180° dropped so the poles count once.
- **DI:** `SPDI = LW_dB − SP_dB`; `ERDI = LW_dB − ER_dB` (plain dB subtraction).
- **Estimated In-Room (PIR/EIR):** `pressure2spl(sqrt(0.12·pLW² + 0.44·pER² + 0.44·pSP²))`.

SP band weights (per |orbit angle|°): 0:0.03038, 10:0.23777, 20:0.45013, 30:0.62266, 40:0.75346,
50:0.84789, 60:0.91312, 70:0.95538, 80:0.97906, 90:0.98668 (symmetric for 100..180 via 180−|θ|).

**Departure from proposal wording (flagged per CLAUDE.md):** the proposal said "reuse
`power_di.directivity_index`." CEA DI is a *different quantity* (LW−SP / LW−ER, not max/mean
intensity). We reuse the *quadrature machinery* conceptually but compute the CEA definitions; the
sphere-quadrature SP is kept only as a ±0.5 dB sanity cross-check.

### Hand-computed acceptance values (dB re on-axis), exact-weights, axisymmetric about ref axis
- **Offset monopole** (|H|=1 ∀θ): On-Axis=LW=ER=SP=PIR=0; SPDI=ERDI=0. (Resample/cardinal-rule stress
  test: flat magnitude despite a strong phase ramp.)
- **True dipole** |H|=|cos θ|: LW −0.433, ER −2.524, SP −4.416, SPDI +3.983, ERDI +2.091, PIR −2.892.
- **cos²θ** |H|=cos²θ: LW −0.823, ER −3.929, SP −6.481, SPDI +5.658, ERDI +3.106, PIR −4.245.

## Far-field display option (`core/field_referencing.py`) — DISPLAY TRANSFORM ONLY
User chose **both, user-selectable**. Never mutates/writes the stored near-field H-tensor (cardinal
rule); computed on the fly per driver. Modes:
1. **Near-field (as solved)** — identity.
2. **Far-field: acoustic-center** — per driver at position `p`, with `r_n = |r_obs·û_n − p|`:
   `H_ac[f,n] = H[f,n] · (r_n / r_obs) · exp(−j k_f (r_n − r_obs))`. Removes geometric 1/r spreading +
   path-length phase about the driver center. Offset monopole → exactly omni. Convention-safe.
3. **Far-field: SH extrapolation** — fit SH `a_lm`, extrapolate r→∞ via outgoing spherical Hankel
   `h_l^(1)` (engineering exp(−jωt), outgoing ~ exp(+jkr) ⇒ `h_l^(1)(kr) ~ (−j)^{l+1} e^{jkr}/(kr)`):
   `b_lm = a_lm · (−j)^{l+1} / (k · h_l^(1)(k r_obs))`, evaluated back on the grid (the `|h_l^(1)|`
   growth with l makes the division a stable low-pass). The result is then **referenced back to
   pressure-at-`r_obs`** by a per-frequency, direction-independent scale `exp(+jk r_obs)/r_obs` so all
   three display modes share one absolute level (else the raw directivity coefficient reads
   `20·log10(r_obs)` ≈ 6 dB hotter on absolute-SPL views). True radiating far-field; offset monopole
   → omni, with the physical far-field phase ramp preserved (verified vs the analytic monopole in
   magnitude **and** phase).

Both verified against the analytic offset monopole (→ near-omni at low f) in
`tests/test_field_referencing.py`.

## Tests
- `tests/test_cea2034.py` — angle-set sizes; monopole→all-flat/DI=0; dipole & cos² vs hand-comp;
  SP CTA-weighted vs sphere-quadrature sanity bound; reference-axis rotation invariance.
- `tests/test_field_referencing.py` — identity for near-field; offset monopole → omni for both
  far-field modes; **stored H untouched** (cardinal-rule guard).
- `tests/test_gui_smoke.py` — extended: polar/sonogram/CEA sub-tabs load; referencing combo switches.
