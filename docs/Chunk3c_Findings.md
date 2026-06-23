# Chunk 3c — Empirical findings & implementation decisions (Auto-Design orchestrator)

> Authoritative record of what the 3c diagnose → empirical-prototype → synthesize campaign
> actually showed. Methodology mirrors 3a/3b: a self-contained map against the REAL repo
> forward model (`design()` + `monopole_field`, engineering convention exp(−jωt)/exp(+jkr),
> c = 343.2) on the same CI-safe fixtures the V-gates use, then the orchestrator validated
> end-to-end against `design(engine="auto")`. Written 2026-06-22.

## The 3c feature, in one line
A new `engine="auto"` dispatches to a **principled escalation ladder** (`beamform/orchestrator.py`)
that, for the target's class, runs a target-conditioned set of the already-built well-posed engines
through the real `design()`, **scores each on the target's OWN objective metric** (reusing the
metrics `design()` already reports — never recomputing patterns), and returns the best feasible
candidate with an **honest report** of the engine it chose and where the target can't be met. It is
NOT the literal "try every algorithm and stack them" the user first described (confirmed at kickoff
— Open Question 1): blindly stacking incommensurable objectives has no well-posed combined objective
and risks non-convergence / unrealizable filters. Same user-facing outcome ("Auto-Design finds a
good filter without me picking the algorithm"), but it actually converges.

## The flagged decisions (confirmed at kickoff)
1. **Principled escalation ladder**, not literal try-every-and-stack. ✅ confirmed.
2. **One best engine per design + honest per-bin `feasible_mask` / `band_feasible`**, deferring
   per-band engine *blending* (filter-discontinuity risk). ✅ confirmed.

## The linchpin (the kickoff's diagnose-first): which engine wins for which target class
Empirically mapped on the real fixtures (2-driver cardioid pair d=0.086; ~50-driver CBT cap;
a null-requiring pair). Each class's expert engine wins on **its own metric by a decisive margin**,
so the gate is **non-circular** — it asserts engine *behavior* in the recorded trace, not the
orchestrator's class→engine wiring:

| target class | fixture | winner | metric | winner vs runners-up |
|---|---|---|---|---|
| `shape` (cardioid) | 2-driver pair | **`ls`** | `target_error_db` (min) | **5.80** dB vs delay_sum 11.13 / mvdr 12.84 |
| `constant_directivity` | 50-driver cap | **`constant_di`** | `ptp(di_db)` (min) | **0.000** dB vs ls 1.96 / max_dir 3.33 |
| `nulls` | 2-driver pair | **`lcmv`** | worst-bin null depth (min) | **−310** dB vs ls −3.0 / mvdr −1.6 |
| `max_directivity` | 50-driver cap | **`max_directivity`** | `di_db` (max) s.t. WNG | **16.67** dB vs delay_sum 12.05 / ls 8.68 |

## Three things the evidence settled (one of them overturned a working premise)
1. **The target-class signal is a SPEC-DESIGN decision, not an empirical finding.** The advisor
   caught the trap: the proposal's `target_gdi_db`-based routing for constant-DI *breaks*, because
   `target_gdi_db=None` is a legal value ("max feasible level") **and the V-CBT gate spec sets no
   `target_gdi_db` at all**. "Constant directivity" is a *cross-frequency-constancy intent* that no
   per-frequency shape field encodes; pre-3c it was expressed ONLY by picking the `constant_di`
   engine — a signal that disappears under `engine="auto"`. **Fix: an explicit `TargetSpec.objective`
   enum** (`"shape"` default · `"max_directivity"` · `"constant_directivity"`; a non-empty `nulls`
   overrides to the null class). Default `"shape"`, ignored by every concrete engine → zero
   back-compat risk.
2. **The selector is the class-metric optimizer; the acceptance threshold is only the `converged`
   honesty flag.** An early-exit-on-threshold ladder is fragile (a loose threshold lets a robust-but-
   wrong engine "converge" first). Best-score-on-the-class-metric is cleaner and self-evidently
   non-circular: `ls`'s `target_error` < `delay_sum`'s, `lcmv`'s null < `mvdr`'s — facts about engine
   behavior. The threshold drives `converged` / `band_feasible` reporting, and the
   superdirective-infeasible scenario exercises the "nothing feasible → honest best-effort" path so
   the flag is not dead code.
3. **The DI-objective candidates MUST be forced to `directivity_mode="index"` inside the
   orchestrator** (advisor-flagged silent-failure trap). `TargetSpec.directivity_mode` defaults to
   `"region"`; if a `constant_di` candidate inherited that, it would optimize the cap-ratio objective
   (3b measured proper-DI varying ~6.7 dB there) and **lose its own class**. The orchestrator sets
   `index` when constructing the `constant_di` / `max_directivity` candidate specs, independent of
   the caller. Confirmed: `constant_di` then holds `di_ptp = 0.000`.

## A real numerical finding: loaded-MVDR ≡ WNG-floored max-directivity
On the cap, `mvdr` and `max_directivity` reach the **same** directivity (16.674 vs 16.673 dB) — both
maximize directivity subject to the WNG floor via diagonal loading, and `mvdr` is even fractionally
higher. A raw `argmax` would therefore pick `mvdr`. The orchestrator breaks ties (within
`_TIE_EPS_DB = 0.25 dB`) with a **fixed preference order** (`max_directivity` first for that class) so
the choice is deterministic and lands on the canonically-named formulation — never iteration-order
dependent. This is why the tie-break is a ranked list, not a float compare.

## Implementation spec (`beamform/orchestrator.py`)
- **Classify** → `nulls` (if `spec.nulls`) else `spec.objective` else `"shape"`.
- **Candidate ladders** (robust→aggressive): shape `[delay_sum, ls, mvdr, max_directivity]`;
  constant_directivity `[ls, max_directivity, constant_di]`; nulls `[ls, mvdr, lcmv]`;
  max_directivity `[delay_sum, ls, mvdr, max_directivity]`.
- **Run** each candidate via the REAL `design()` (engine overridden; `index` forced for the DI
  engines). The orchestrator only *calls* `design()` — it never re-tunes an engine (3a/3b did that)
  and never re-zeros/min-phases a driver.
- **Score** (lower-is-better, single argmin): shape = median `target_error_db`; constant_directivity
  = `ptp(di_db)` (+`1e6` omni-trap penalty if median DI < 3 dB); nulls = worst-bin null depth (from
  `steered_field` at `target.null_idx`); max_directivity = `−median(di_db)`.
- **Select**: best score among **feasible** candidates (`mean(feasible_mask) ≥ 0.5`); tie-break by
  fixed preference order. If none feasible → best-effort over all + `band_feasible=False`.
- **Feasibility pre-screen**: per-bin matched-field WNG ceiling `10log10(‖c‖²)` vs the floor; bins
  where floor > ceiling are physically infeasible for ANY engine (array limit, not solver) — reported
  in `auto_prescreen`.
- **Honest report** (grafted onto the chosen result's `attrs`): `engine` = the concrete engine
  actually used; `auto_selected`, `auto_class`, `auto_trace` (every candidate's metrics + `converged`
  flag), `auto_reason` (one line), `auto_prescreen`, `band_feasible`; per-bin honesty stays in
  `metrics["feasible_mask"]`. The returned `.spec` echoes the user's original `engine="auto"` request.

## Cardinal rule
Auto-Design composes existing engines; steering stays entirely in H's inter-driver phase. The
collapse-to-origin control under `engine="auto"` still gives DI ≈ 0 (inherited from whichever engine
it dispatches), and a shared modeling delay leaves |P| invariant — both asserted on the auto path
(`test_beamform_auto_design.py`).

## Gate (3c): `tests/test_beamform_auto_design.py`
For each class: the expert engine is chosen, AND (non-circular) it beats the runner-up on that
target's own metric in `auto_trace`, AND the chosen design actually realizes the target (cardioid
DI≈4.77 / flat DI / −40 dB null / high DI). Plus: a +6 dB WNG floor on a 2-driver array (ceiling
10log10(2)=3.01 dB) is honestly flagged `band_feasible=False` + best-effort (every bin above the
array ceiling in the pre-screen); cardinal-rule collapse + shared-ramp controls on the auto path.
All prior gates stay green (V-cardioid, V-CBT, V-RT, V-5 phase-origin, V-1/V-2, constant-DI tests,
`test_beamform_engine.py`, GUI smoke).

## GUI
`gui/filter_designer_view.py`: **"Auto-Design (pick best engine)" leads the engine list** (the user
picks a *target*, not an algorithm) but is **opt-in** — Least-squares stays the active default so the
default "Design" is one fast solve, not the auto ladder's up-to-4 (confirmed with the user; the
kickoff scoped "add an entry"). Two new pattern entries — "Constant directivity" and "Maximum
directivity" — carry the `objective` (mapped in `_build_spec` via `_PATTERN_OBJECTIVE`). The status
line names the engine Auto-Design **chose** (`Auto → ls`), shows a `⚠ best-effort` note when the
target is not fully feasible, and exposes `auto_reason` as the tooltip.

## Scope note (deferred to 3d/3e)
Per-band engine *blending* (different engines across frequency) is deferred — realizability /
discontinuity risk. Multi-target scalarization (beamwidth + DI + in-room/CEA2034 together) is 3d.
Filter-designer visualization of the ladder trace is 3e.
