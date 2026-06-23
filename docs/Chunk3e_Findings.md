# Chunk 3e Findings — Filter-designer visualization (the final Chunk-3 sub-chunk)

**Status:** complete. Closes Chunk 3 (#8, the beamforming/filter-designer rebuild) → milestone
`v1.3.0`. Builds on 3a–3d (`docs/Chunk3a..3d_Findings.md`); read those for the *engine* behavior
this sub-chunk merely *draws*.

3e is **pure visualization**. It adds the remaining plot views to the Filter Designer GUI tab
(`gui/filter_designer_view.py`) and **visualizes the existing `DesignResult` outputs** — it does
**not** recompute, re-tune, or re-solve anything, and it never re-zeros / minimum-phase-ifies a
driver. The five proposal deliverables (`docs/Bug_Fix_Proposal.md` ~line 164) are now all on screen:
per-driver responses, filter magnitude/phase, achieved-vs-target directivity, CEA-2034-A in-room, and
beamwidth/DI/WNG vs frequency — reusing the Chunk-2 plotting infrastructure.

## What 3e added (one file of GUI code + render-gate tests)

The right-hand plot panel of `FilterDesignerTab` was restructured from two stacked canvases
(`_polar` + `_di`) into a **`QTabWidget` of five `_MplCanvas` sub-tabs**, mirroring `results_view`'s
`ResultsTab`. Each view is a `_replot_*` method reading only the frozen `DesignResult`:

| Sub-tab | Deliverable | Source (all read-only) |
|---|---|---|
| **Polar** | achieved-vs-target directivity (one freq) | `steered_field[fi]` + `build_target(...).b_field[fi]`, SH-resampled to the H arc through the steer axis |
| **Directivity** | beamwidth / DI / WNG vs frequency (+ target-error) | `metrics["di_db"/"beamwidth_deg"/"wng_db"/"target_error_db"/"feasible_mask"]` |
| **Filters** | filter magnitude / phase | `result.weights[M,F]` — |w| in dB re the loudest weight, phase `np.unwrap`'d per driver |
| **Per-driver** | per-driver responses | `weights[m]·stacked_h_full(ds)[m]` resampled to the steer axis, + the combined `steered_field` |
| **CEA2034 / in-room** | CEA-2034-A in-room | `compute_cea2034(steered_field, freqs, obs, steer_dir)` spinorama, Estimated-In-Room bold |

The `_on_design_done` path calls `_replot()` (refreshes all five). The "Polar frequency" combo calls
`_replot_polar` **alone**. Reused verbatim from Chunk 2: `_MplCanvas`, `_quiet_redraw_warnings`, `_db`,
`_CEA_COLORS`/`_CEA_LABELS` (from `results_view`), `compute_cea2034`/`SPL_CURVES`/`DI_CURVES` (from
`metrics.cea2034`), plus `stacked_h_full`, `resample`, `safe_order_for_grid`, `great_circle_arc` — no
new plotting helper was added (the one local constant, `_MAG_FLOOR`, guards only the weights-magnitude
path, which references `|w|` to the loudest weight rather than to 20 µPa).

## Settled premises (load-bearing — read before any 3e follow-up)

1. **"Don't recompute" bans re-solving, not display derivation.** Computing the CEA spinorama from
   the *frozen* `steered_field` and reading `metrics[...]` is exactly what the proposal endorses
   ("in-room via `compute_cea2034(steered_field, …)`") and what `results_view` already does for the
   per-driver tensor. What is banned is calling `design()`/`orchestrator`/`weights` to re-tune.
   3e calls none of those.

2. **The CEA panel references the spinorama to `steer_dir` (the beam axis), NOT the dataset front
   axis.** This is the one place a naïve "reuse `_Cea2034View`" instinct goes wrong:
   `_Cea2034View` references to `_reference_axis(ds)` (the loudspeaker front), but
   `orchestrator._eir_slope` references the EIR to the **beam axis** (on-axis for the listener), and
   the multi-target metrics text line already shows that number. Referencing the plotted spinorama to
   the dataset front axis would make the **plotted in-room slope silently disagree** with the reported
   `multi_achieved["inroom"]` value in the same tab. `_replot_cea` passes the normalized `steer_dir`,
   so picture and number agree.

3. **Cardinal rule in a read-only view = mutate nothing + plot phase as-stored.** `stacked_h_full`
   returns a fresh `np.stack` (never a view onto stored data); `weights[m,:,None]*h[m]` allocates a
   new array. Per-driver **radiated** phase is plotted directly (referenced to the global origin),
   never re-zeroed per driver. The only `np.unwrap` is on the **filter weight** phase, purely for
   legibility of a 1-D curve — it does not re-baseline any driver's radiated response. A snapshot
   guard (`H_bem`/`H_full` copied before, `np.array_equal` after) is asserted in the gate test.

4. **The frequency combo drives the polar view only.** The vs-frequency dashboards (Directivity,
   Filters, Per-driver) and the spinorama span the whole band, and CEA/per-driver do an SH resample
   per redraw — so re-running them on every frequency-combo change would be wasted work. Only
   `_replot_polar` is wired to the combo; `_replot()` (all views) runs once per finished design.

5. **Display must survive the honest non-ideal values the engines emit.** `beamwidth_deg` is `nan`
   where the main lobe does not close (physics, e.g. below the array's directive band); `wng_db` is
   `-inf` for collapsed/degenerate bins. The metrics view masks `-inf → nan` (gaps, not a broken log
   axis) and lets `nan` beamwidth read as gaps. Target reference lines (`target DI`/`target BW`,
   `WNG floor`) are drawn only when the corresponding target is set (i.e. for the multi-target
   pattern; `wng_floor_db` is always set). Infeasible bins (`~feasible_mask`) are marked in red on the
   WNG panel — the honest "where the array's directivity ceiling beat the requested robustness" flag.

## Gate

Render-focused, extending `tests/test_gui_smoke.py` (no NumCalc, offscreen Qt):

- `test_filter_designer_3e_views_render_correct_series` — five sub-tabs present; after a default
  (cardioid/LS) design, asserts the **correct series** on each view (achieved+target on the polar; the
  DI / `-6 dB beamwidth` / `achieved WNG` / `WNG floor` labels on the metrics view; **M** lines on the
  filter panels; **M+1** lines incl. `combined` on the per-driver panels; `len(SPL_CURVES)` /
  `len(DI_CURVES)` lines + the Estimated-In-Room curve on the CEA view) — discriminating checks, not
  "did not raise". Includes the **cardinal-rule snapshot guard**.
- `test_filter_designer_3e_multi_target_reference_lines` — the multi-target objectives show as dashed
  `target DI` / `target BW` reference lines.
- `test_filter_designer_3e_cea_references_steer_axis` — **guards premise 2**: steers the beam to +x on
  a +z-front dataset and asserts the plotted On-Axis spinorama curve matches the *steer*-referenced
  `compute_cea2034` and differs from the *front*-referenced one (a regression to `_reference_axis(ds)`
  fails here; every other test uses +z-on-+z where the axes coincide).
- `test_filter_designer_3e_metrics_handles_inf_wng_and_infeasible` — injects `wng_db = -inf` and an
  infeasible bin into a frozen result and asserts the metrics view does not raise and draws the
  `infeasible bin` marker (exercises premise 5's edge handling).
- `test_filter_designer_3e_freq_combo_redraws_polar` — the combo redraws the polar alone.

All prior gates (V-1/V-2/V-5, 3a–3d, the existing GUI smoke tests) stay green. Verified visually by
rendering all five views on a real two-monopole array (cardioid/LS and multi-target) — DI rises
through the cardioid band, the inter-driver filter phase forms the beam, the in-room curve is bold,
WNG tracks its floor.

## Methodology

diagnose (read `DesignResult`/`metrics`/`attrs` + the Chunk-2 plotting helpers) → architect (sub-tab
panel, reuse map, the `steer_dir` consistency decision) → advisor-vet before package code → implement
→ render + assert correct series → **adversarial multi-dimension review workflow** (cardinal-rule
safety, plotted-series correctness, reuse vs duplication, test adequacy/completeness; each finding
adversarially verified) → advisor-vet before declaring done.

**Review outcome.** The cardinal-rule/read-only dimension surfaced **no** confirmed findings — the
plots never mutate stored H and never re-zero a driver. The verified findings were all low/nit and
were applied: the duplicated SPL helper now reuses `results_view._db`; the stale top docstring was
updated; the `steer_dir`-vs-front consistency (premise 2) and the WNG `-inf`/infeasible edge cases
(premise 5), previously *implemented but unguarded*, now each have a dedicated test; the target-error
twin axis is labelled and asserted; black/ruff are clean.
