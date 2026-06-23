# Chunk 3d — Empirical findings & implementation decisions (multi-target objectives)

> Authoritative record of what the 3d diagnose → empirical-prototype → synthesize campaign actually
> showed. Methodology mirrors 3a/3b/3c: a self-contained map against the REAL repo forward model
> (`design()` + `compute_cea2034` + `directivity_metrics`, engineering convention exp(−jωt)/exp(+jkr),
> c = 343.2) on the same CI-safe fixtures the V-gates use, then the orchestrator validated end-to-end
> through `design(engine="auto", objective="multi")`. Written 2026-06-22.

## The 3d feature, in one line
A new `objective="multi"` (dispatched through `engine="auto"`) targets **{directivity index, −6 dB
beamwidth, in-room CEA-2034-A EIR slope} jointly** via a **scalarized weighted-sum of normalized
per-objective deviations** over a curated **(engine, knob) search** — picking the best feasible
candidate (the WNG floor / `nulls` are lexicographically-prior **feasibility gates**), and reporting
each objective's achieved-vs-target + the chosen (engine, knobs) + the weight trace honestly. It is a
principled *search + scoring* layer on top of the 3a/3b/3c engines — it only **calls** `design()` and
`compute_cea2034`; it never re-tunes a solver and never re-zeros/min-phases a driver.

## Confirmed kickoff decisions (AskUserQuestion)
1. **Combination rule = scalarized weighted-sum** of normalized deviations (hard constraints as
   feasibility gates), NOT a strict lexicographic cascade. ✅ Slides to a Pareto point; reuses the 3c
   scoring infra; degrades to single-objective when one weight is 1.
2. **Optimization variable = (engine + existing knobs)**, NOT a new joint QCQP solver. ✅ The engines
   + knobs already span {DI, beamwidth, robustness}; 3d is a search/scoring layer, never a re-tune.
3. (Restated, already confirmed) **in-room = CEA-2034-A Estimated-In-Room** (`metrics/cea2034.py`);
   target slope research-led — see below.

## Diagnose-first: the conflict map (the linchpin — the gate is vacuous without a real conflict)
Pearson r across the cap candidate cloud (~50-driver cap, flat-CBT band 2300–4200 Hz):

| relation | r | reading |
|---|---|---|
| DI ↔ −6 dB beamwidth | **−0.893** | strong but NOT total — see beamwidth-independence below |
| DI ↔ in-room downtilt LEVEL (EIR−on-axis) | **−0.945** | the *level* is a near-restatement of DI |
| DI ↔ SPDI (CEA sound-power DI) | **+0.980** | SPDI ≈ DI (redundant) |
| DI ↔ WNG floor (min) | +0.101 | weak cross-candidate (the floor is constraint-enforced) |

- **Beamwidth-independence test (the key result):** within a narrow DI band the achievable −6 dB
  beamwidth still spans **~12–17°** (e.g. at DI≈9: ls/cardioid 53°, ls/supercardioid 64°,
  constant_di/gdi10 47°). So **beamwidth is a genuine semi-independent axis** — the lever is the
  engine/shape (constant_di gives a *narrower* beam than an ls shape at the same DI). → keep
  beamwidth first-class (answers the kickoff's open question).
- **DI vs WNG is a within-engine tension** (clean in the `max_directivity` floor sweep: DI 17.24 @
  −12 dB → 15.57 @ +6 dB), kept as a lexicographic **feasibility gate**, not a scored axis.
- **2-driver ceiling honesty:** a +6 dB WNG floor on the pair (ceiling 10log₁₀2 = 3.01 dB) →
  `feasible_mask` all False (the multi search reports best-effort), as in 3c.

## Settled premises (measured / advisor-vetted, several overturning the naïve framing)
1. **The kickoff's literal acceptance wording is CIRCULAR; the load-bearing gate is a minimax fact.**
   "Balanced beats the single-objective extremes on the *combined* score" is a tautology (balanced =
   argmin(combined)). The **non-circular** claim — proven, with margin — is a **minimax** fact about
   the *achieved fields*: the balanced design's **worst** normalized per-objective deviation is
   strictly lower than the worst deviation of **every** single-objective optimum. This mirrors how 3c
   dodged circularity (asserting per-engine behavior, not selector wiring). The combined-score line is
   still reported, labeled as the trivial corollary.
2. **In-room: the downtilt LEVEL tracks DI; the active axis is the EIR SLOPE — and on a <1-octave band
   the slope is largely a *constant-directivity proxy*.** EIR−on-axis (downtilt level) is r=−0.95 with
   DI → it adds nothing as an independent axis. The EIR **slope** *does* move across the clean engines
   (constant_di −0.7 dB/oct, max_directivity −2.6, delay_sum −3.9) and pulls to a **different**
   candidate than DI (the in-room optimum is ls/hypercardioid, the DI optimum is max_directivity), so
   it is NOT a dead axis. But on the narrow gate band the slope is largely a restatement of `di_ptp`
   (constant DI ⇒ flat slope). The **genuine multi-octave downtilt axis** is demonstrated on a wider
   band (800–8000 Hz): EIR slope ls −2.89 vs max_directivity −2.36 vs constant_di −0.97 dB/oct.
   Honest about the regime (the 3b pattern): the narrow-band gate proves in-room *pulls separately*;
   the wide band proves the *downtilt* axis is real.
3. **`mvdr` ≡ index-mode `max_directivity` → dropped from the multi ladder.** Both reach DI 16.67 /
   BW 31.5° / slope −2.60 on the cap (the 3c "loaded-MVDR ≡ WNG-floored max-directivity" result).
   Listing both would let `min()` name "mvdr" on a tie over the canonical "max_directivity" — the very
   ambiguity 3c's tie-preference existed to prevent. Dedup removes it and a redundant `design()` call.
4. **Normalization scales are FIXED physical constants derived from the measured spans, not per-run
   spread.** `_NORM = {di: 3 dB, beamwidth: 12°, inroom: 1 dB/oct}`: each objective's achievable span
   (DI ~9 dB, beamwidth ~35°, clean EIR-slope ~3 dB/oct) ÷ ~3 → comparable ~3-unit deviation ranges,
   so no objective dominates the unweighted sum. Hardcoded before scoring (per-run candidate spread
   would make the score depend on the candidate set and the gate non-reproducible). The minimax trade
   holds **structurally** because the objectives conflict (r(DI,BW)=−0.89), not because of the scales.
5. **The in-room target slope default is research-backed −1.0 dB/oct.** Harman/Olive "preferred"
   in-room (PIR/EIR) slope ≈ −1 dB/oct (B&K ≈ −0.9), with Olive's own caveat that very directive
   speakers warrant a *flatter* target (the in-room is more direct-sound-dominated). Sources:
   spinorama / Dirac / HouseCurve target-curve guidance. The GUI defaults the slope spin-box to −1.0;
   the core leaves it `None` unless set (no silent default).

## Implementation spec
- **`TargetSpec`** (`beamform/targets.py`): `objective` gains `"multi"`; four optional fields
  (`target_di_db`, `target_beamwidth_deg`, `target_inroom_slope_db_per_oct`, `objective_weights`) all
  default `None` ⇒ fully back-compatible. Under `"multi"`, `nulls` is a feasibility **gate** (a
  deliberate divergence from `_classify`, where nulls dominate).
- **`beamform/orchestrator.py`**: `_classify` checks `"multi"` first; `design_auto` early-returns
  `design_multi`. `design_multi` enumerates `_MULTI_LADDER` (`ls`×{cardioid, supercardioid,
  hypercardioid}, `delay_sum`, `constant_di`×{gdi 10/12/14/None, index}, `max_directivity` index) via
  the real `design()`, computes each candidate's `{di_med, bw_med (nan-guarded), eir_slope}`, scores
  the scalarized weighted-sum of normalized deviations, gates on `feasible_mask` (`_FEAS_FRAC`) + the
  null gate, argmin(combined) with ladder-order tie-break, and grafts the honest report onto `attrs`
  (`auto_class="multi"`, `multi_targets`, `multi_weights`, `multi_norm`, `multi_trace`,
  `multi_achieved`, `auto_reason`, `auto_prescreen`, `band_feasible`). The in-room slope is the only
  thing computed beyond `design()`'s metrics (one `compute_cea2034` resample per candidate, referenced
  to the beam axis `steer_dir`). A `"multi"` spec with no active target **raises** (not a silent
  no-op).
- **`gui/filter_designer_view.py`**: a "Multi-target (DI/beamwidth/in-room)" pattern that locks the
  engine to Auto-Design and enables a per-objective {use, target, weight} control group; the status
  line appends the per-objective achieved-vs-target summary.

## Gate (3d): `tests/test_beamform_multi_target.py` (V-MULTI)
On the ~50-driver cap, the 3-way-conflicting target **{DI 16 dB, beamwidth 55°, in-room −1 dB/oct}**
with balanced weights:
- **Non-circular minimax (load-bearing):** balanced worst-dev **1.250** < DI-optimum 1.958,
  beamwidth-optimum 2.293, in-room-optimum 2.442 (margin +0.708). Plus the explicit 3c-style trade
  (balanced more directive than the wide-beam optimum; closer beamwidth AND flatter in-room than the
  DI optimum) and `engine ∈ {constant_di, max_directivity}` — a **property, not the exact knob** (the
  top two candidates score within ~0.02: constant_di/gdi14 0.693 vs constant_di/None 0.710 — minimax
  + trade pass for both).
- **In-room is an active, separately-pulling axis:** the in-room optimum is a different candidate from
  the DI optimum, slopes differing by >0.5 dB/oct.
- **Literal acceptance corollary** reported (trivially true), labeled as such.
- **End-to-end Pareto:** DI-heavy weights → DI 15.07 / BW 36.5°; beamwidth-heavy → DI 12.0 / BW 45.5°
  (more directivity weight buys DI at the cost of beamwidth — a real front).
- **Honest report** shape + per-objective achieved-vs-target asserted; `band_feasible` True.
- **Cardinal-rule controls on the new path:** collapse the cap to a sub-mm cluster → chosen design
  DI ≈ 0.01 (steering is entirely in H's inter-driver phase); a shared modeling delay leaves |P|
  invariant to <1e-9.
- **Contract:** multi-with-no-target raises; under multi a requested null does NOT switch to the null
  class (`auto_class=="multi"`). Note: `nulls` under multi are a **best-effort gate only** — the
  multi ladder has no hard-null engine (`lcmv` excluded), so a requested null usually degrades to
  best-effort (`band_feasible=False`); for a HARD null, use the dedicated null objective (a non-empty
  `nulls` with a non-`"multi"` objective routes to `lcmv`).

All prior gates stay green (462 passed, 14 deselected): V-AUTO, V-cardioid, V-CBT, V-RT, V-5
phase-origin, V-1/V-2, the constant-DI tests, `test_beamform_engine.py`, and the GUI smoke.
