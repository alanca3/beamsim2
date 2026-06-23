# Chunk 4 — Findings: Model-viewer UX & driver interaction (#1, #2, #3)

Shipped **v1.4.0** (2026-06-22) on `feature/chunk4-viewer-ux`. Source bugs: `First_Run_Bugs.txt`
#1/#2/#3; scope from `docs/Bug_Fix_Proposal.md` (Chunk 4). All three are GUI/geometry fixes — no
solver, H-tensor, or phase-origin change (cardinal rule preserved; V-5 green throughout).

## What each bug actually was (verified against current code, not the bug text)

### #3 — driver orientation reset (the only true logic bug; two layers)
1. **The reported symptom.** `parameters_panel.TSDialog._prefill` set id/center/radius/T-S but **never
   set `_normal_combo`**, so the editor always re-defaulted to index 0 (+z). On OK,
   `_on_ok` writes `spec.normal = _normal_from_combo()` = +z → editing a driver silently re-zeroed its
   true orientation. *Fix:* `_prefill` maps `dp.spec.normal → combo index` via `faces.face_id_from_normal`.
2. **The deeper trap — face-normal authority (advisor-flagged).** The two edit paths behaved
   *oppositely*: `GeometryTab._on_canvas_driver_edited` re-derived `spec` from `face_placement`
   (discarding the combo normal), while `DriversTab._edit_driver` kept the combo normal but left
   `face_placement` disagreeing (which then reverted on the next dims-change/drag). These were one rule
   applied to two divergent code paths. *Fix:* a single shared helper
   `geometry.faces.reconcile_placement(chosen_normal, old_fp, radius, w, h, d) → (spec, face_placement)`,
   called by **both** paths. `FacePlacement` is the single source of truth; `spec` is always derived
   from it, so they can never silently disagree. Same face → keep `(u,v)`; new orientation → move the
   driver to that face (recentred, radius clamped to fit the possibly-smaller face).

### #1 — no reference-axis / virtual-mic indicator (the main new work)
Confirmed absent from the editor. But `SimulationRequest.reference_axis` already existed and was already
written to the dataset's `reference_axis` root attr (Chunk 1, `pipeline/run.py`); it just wasn't
surfaced in `AppState` or the editor, and `build_request` never passed it. *Fix:*
- Pure `geometry.faces.reference_axis_indicator(axis, w, h, d) → AxisIndicator` (box-centre origin,
  unit direction via `core.sphere.reference_frame`, scaled stand-off). `_DriverEditorCanvas._draw_reference_axis`
  renders an arrow + microphone glyph + `0° / on-axis mic` label.
- New `AppState.reference_axis` (default +z) + a 6-way Geometry-tab combo, threaded through
  `build_request → SimulationRequest.reference_axis`. This **closes the Chunk-1 cross-cutting thread**:
  the editor indicator, the stored attr, and the Results On-axis/Balloon/Polar/CEA views now share one
  axis source.

### #2 — drag/right-click (mostly already built; verify + polish)
The interactive path (`_DriverEditorCanvas` click-to-place, constrained drag, right-click menu) was
already implemented and the prior session's editor fixes were intact. The real residual gap was the
user's literal complaint: a new driver always landed at the **face centre**. *Fix:* `driverAdded` now
carries the clicked, face-clamped `(u, v)`, so drivers place where you click. Added a discovery hint.

## Decisions taken (the §4 flagged items)

1. **Reference-axis source (AppState).** Added `AppState.reference_axis` and made it **settable** and
   **threaded through the solve** (recommended option), not just a static +z indicator — it closes the
   cross-cutting thread and wires the previously-unused `SimulationRequest.reference_axis`. Display/
   metadata only; never moves geometry or the phase origin.
2. **PyVista hard vs soft — MOOT.** PyVista/pyvistaqt are already in `[project].dependencies` (a soft
   requirement is already satisfied). Kept the Matplotlib fallback. No `pyproject` change.
3. **Mic representation.** A microphone glyph (sphere) at a **view-scaled stand-off** (`1.6 × max dim`)
   along the axis, labelled **direction-only** (`0° / on-axis mic`). The label deliberately claims **no
   distance** — `AppState.sphere_radius` is not synced from the Simulation tab, so a "1.0 m" label
   would be false (advisor point 4). Position is scaled for visibility; direction is the honest signal.

## Settled premises (read before any Chunk-4 follow-up)

- **The combo is index-aligned with `face_id`.** `TSDialog` combo item *i* == `faces.FACE_NORMALS[i]`
  == outward normal of face *i* == inverse `face_id_from_normal`. The whole #3 fix rests on this; it is
  asserted by `test_face_normal_combo_invariant`. Don't reorder the combo or the normal table.
- **One reconcile rule, two callers.** Never hand-write "combo-change → move to new face" twice — that
  is exactly how #3 regresses through the path you didn't test. Both editors call `reconcile_placement`.
- **Dims source of truth.** `AppState.box_dims` is the shared dims the Drivers-list edit reconcile
  reads (the Geometry tab's spin-boxes remain the interactive truth and mirror into `box_dims`). The
  canvas edit path sources dims from its own spin-boxes; both are tested independently.
- **Reference axis = front, display/metadata only.** Reuse `core.sphere.reference_frame` (gives the +z
  zero-axis fallback for free); never invent a new convention. (Note: the *filter-designer* CEA in-room
  is referenced to the **beam** axis, not the front — that is Chunk 3e's concern, unchanged here.)

## Testing approach (PyVista cannot run under offscreen CI)

VTK needs a real OpenGL context, so the GL path is only manually testable. The load-bearing logic was
therefore factored into **pure, headless functions** in `geometry/faces.py` and covered by 9 new
`tests/test_gui_smoke.py` cases (combo↔face_id invariant; prefill restore for all 6 normals; reconcile
same-/new-face incl. radius clamp; the **edit→reopen persistence** through both real edit paths with a
stubbed dialog; the indicator geometry rotating with the axis incl. zero-axis fallback; place-at-click
`(u,v)`; `build_request` threading `reference_axis`). Plus an **off-screen PyVista API check** of the
exact `pv.Arrow`/`pv.Sphere`/`add_point_labels` calls `_draw_reference_axis` makes (headless pytest can
never reach them), incl. asserting the indicator actors are `pickable=False`. Full CI-safe suite:
**476 passed, 14 deselected** (`-m 'not local_only and not bempp'`); V-5 (cardinal rule) green.

**Still requires a human interactive pass** (no display in the build environment): launch
`uv run python -m beamsim2.gui.app`, then place a driver by clicking a face, drag it, right-click →
Edit T/S / Delete, reopen the editor and confirm a changed orientation persists, and confirm the
reference-axis arrow + mic are clearly visible and rotate when the reference-axis combo changes. This
is the kickoff's **primary** acceptance gate (§8.7, "load-bearing since CI can't cover the drag/
right-click path") and the only part automated tests structurally cannot reach — the 4-arg
`driverAdded` emission (GL-bound `_on_left_press`), real click-through past the now-`pickable=False`
indicator, and the live `TSDialog.exec()` → OK → re-orient round-trip are each tested *in pieces* but
never end-to-end. **Released only after the user confirmed this interactive pass** (or explicitly
authorised shipping on the automated proxy).

**Known limitation (non-blocking):** the Matplotlib fallback (`_MeshCanvas`, used only when PyVista is
absent — i.e. headless, since PyVista is a default dep) does **not** draw the #1 reference-axis
indicator; a no-PyVista user sees the box preview without the arrow/mic. Acceptable because the
interactive editor (the subject of all three bugs) requires PyVista anyway.

## Review outcome

Verified by an **adversarial multi-dimension review workflow** (ultracode): dimensions = (a) orientation
round-trip / face-normal authority, (b) reference-axis/mic indicator vs Chunk-1/2 conventions,
(c) PyVista-active vs fallback safety + no regression of prior editor fixes, (d) test adequacy given
PyVista can't run in CI. Each finding adversarially verified (a skeptic tried to refute it) before
acting. **15 findings raised, 7 confirmed** (all nit/low). Refuted findings included "build_request
never tested" and "_on_canvas_driver_edited reconcile unguarded" — both refuted *because the diff
already added those exact tests*, plus several design choices the verifier judged intentional.

The one **high-severity** confirmed finding was a genuine regression I introduced and fixed:

- **(c, HIGH) The reference-axis arrow/mic actors were pickable** → they would intercept add-driver /
  select / drag picks where the arrow pierces the box front (the cell-picker hits the nearest actor).
  *Fixed:* `pickable=False` on both `add_mesh` calls; verified off-screen that `actor.GetPickable()==0`.

Other confirmed findings applied:

- **(d, MED) `box_dims` sync after a dims change was untested** → added an assertion in the control test
  (change a spin-box, assert `state.box_dims` re-mirrors), guarding the DriversTab reconcile basis.
- **(a, nit) Canvas edit path didn't enforce `driver_id` uniqueness** like the Drivers-list path →
  added `make_unique_id` to `_on_canvas_driver_edited` so both unified edit paths behave identically.
- **(b/c, nit) Dead `set_reference_axis` method** (never called; redrew without `clear_actors`) → removed;
  the single live redraw path is `render_scene(reference_axis=…)`.
- **(d, low) Same-face re-clamp branch of `reconcile_placement` wasn't exercised** (the test's clamp was
  a no-op) → added a same-face case where an enlarged radius forces an inward re-clamp.

Deliberately **not** changed:

- **(b, nit) The reference-axis combo offers only the 6 cardinal faces** although the dataset/Results
  stack accepts arbitrary direction vectors. This is an intentional, spec-consistent UI simplification
  (the kickoff makes the control optional; cardinal-only matches a box's faces and the loudspeaker-front
  convention); editor, stored attr, and Results stay consistent. Revisit only if diagonal axes are needed.
