# Chunk 5b — Kickoff Prompt: steer-to-front-axis + engine guidance (GUI)

> Paste this to start the next session. Read first: `docs/Chunk5_Gameplan.md` (diagnosis + RC2/RC3),
> `docs/Chunk5a_Findings.md` (what 5a fixed), `docs/Chunk5_Status.md`, and the memory index.

## Goal
Fix RC2 + RC3 in the Filter Designer GUI so the user can produce a cardioid in **one or two clicks**,
aimed at the loudspeaker front, with a capable engine — without hand-entering steering angles or
knowing which algorithm to pick. 5a already made the engines *work*; 5b makes them *reachable*.

## Context (confirmed in Session 1)
- RC2: `gui/filter_designer_view.py::_steer_dir()` builds the steer from θ-from-+z / φ, both default
  0 → **+z**, with **no link to the dataset's front**. The user's box front is `reference_axis=+x`
  and the opposed drivers are separated along x, so the default steered **broadside** to the pair →
  no cardioid possible along the front. (`weights.npz steer_dir=[0,0,1]` in run2.)
- RC3: the GUI default engine is `ls` but it lets a "Cardioid" pattern pair with `delay_sum`
  (the omni corner) and gives no steer-axis or engine guidance; run2 ended on `delay_sum`.
- The dataset carries `ds.attrs["reference_axis"]` (e.g. `[1,0,0]`). `core.sphere.reference_frame`
  builds a (front, right, up) frame already used by the Results views — reuse it.

## Scope (do)
1. **Default the steer direction to the front axis.** On `FilterDesignerTab.load(ds)`, initialize the
   steer controls from `ds.attrs["reference_axis"]` (fallback +z if absent). Decide and implement how
   the θ/φ controls read: either (a) interpret θ/φ in the front frame (0° = on the front axis) via
   `reference_frame`, or (b) keep world θ-from-+z but preset the spin-boxes to the front direction and
   show the current front axis in the panel. Prefer (a) — "0° = the loudspeaker front" is what an
   acoustician expects — but keep it cardinal-rule safe (display/intent only; never moves geometry or
   the phase origin).
2. **Engine guidance (RC3).** Make shape patterns (cardioid/super/hyper/figure-8/wide/narrow) default
   to a capable engine (Auto-Design or LS), and warn when a first-order shape is paired with
   `delay_sum` ("delay-and-sum steers but cannot shape a cardioid — use Least-squares or Auto").
3. **Surface the honest band.** The result line / Directivity plot already has `feasible_mask` and
   `band_feasible`; make the achievable-band / infeasible info legible (the cardioid is band-limited
   by physics — show where it holds vs rolls off, e.g. via DI / rear-null / target-error).

## Verify
- Extend `tests/test_gui_smoke.py` (offscreen): steer default follows `reference_axis`; the
  engine-default + delay-sum warning logic; a "Cardioid" design on the reconstructed run2 dataset
  (`tests/_fixtures/reconstruct_run2.py`, skip if absent) renders an in-band rear null.
- Manual GUI pass (PyVista/VTK interaction is invisible to headless tests): load a 2-opposed-driver
  dataset, confirm Cardioid + Auto/LS one-click gives a forward cardioid balloon + null, plots
  populate. (Lesson from Chunk 4: require a human interactive pass before declaring a viewer/GUI
  milestone done.)

## Constraints
- One-way `core ← gui` dependency; reuse `core.sphere.reference_frame` and `results_view` helpers.
- Cardinal rule: steering stays in H's inter-driver phase; nothing re-zeroed. V-5 stays green.

## Close-out
Write `docs/Chunk5b_Findings.md`, update `Chunk5_Log.md` + `Chunk5_Status.md`, and emit
`docs/Chunk5c_Kickoff_Prompt.md` (HDF5 atomic write + attr hardening; **ask the user for the exact
GUI save-error text first**).
