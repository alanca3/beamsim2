# Changelog

All notable changes to BeamSimII are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning: [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] — App-Shell Chunk (project system + menu bar + undo/redo + view manager + GUI logging)

Introduces the full application shell for BeamSimII. **This is the "App-Shell Chunk" from
`docs/Bug_Fix_Proposal.md` §5 (items #4, #5, #6) — not to be confused with the prior
filter-designer "Chunk 5a/b/c" entries below.** Adds `.bsim` project files, a five-menu bar
(File / Edit / View / Settings / Help), snapshot-based undo/redo, a Preferences dialog for logging
and NumCalc path, and unsaved-changes guards. No on-disk solve schema changes (HDF5
`schema_version` unchanged); `.bsim` is a new, separate format at `project_version = 1`.

### Added
- **`.bsim` project file format (`src/beamsim2/io/project_io.py`)**: JSON document storing all
  editable inputs (box, reference axis, drivers, simulation params, solver config, optional results
  path). Handles `PlainLe`/`LR2Ladder` inductance discriminator, `box_volume=None`, tuple↔list
  coercion. `project_version = 1`, separate from `schema_version`.
- **Section-preserving settings layer (`src/beamsim2/backends/numcalc/config.py`)**: replaced the
  clobbering `_write_numcalc_config` with `update_settings(section, mapping)` (read-merge-write),
  plus `read_settings()`, `write_logging_prefs()`, `read_logging_prefs()`,
  `push_recent_project()`, `read_recent_projects()`. Hand-rolled minimal TOML writer (str/bool/int/
  float/list[str]); no new dependency; existing `resolve_numcalc_binary` public API unchanged.
- **Preferences dialog (`src/beamsim2/gui/preferences_dialog.py`)**: logging group (enable toggle,
  level combo, log-file picker) + NumCalc group (binary path + Browse), wired into the settings layer
  and `configure_logging`.
- **Full five-menu bar (`src/beamsim2/gui/app.py`)**: File (New ⌘N / Open ⌘O / Save ⌘S / Save As
  ⇧⌘S / Recent Projects submenu / Open Dataset / Quit), Edit (Undo ⌘Z / Redo ⇧⌘Z with step counts),
  View (Reset R / Perspective/Orthographic toggle / Front F / Back / Left / Right / Top / Bottom /
  Isometric I), Settings (Preferences ⌘,), Help (About).
- **Snapshot-based undo/redo** (cap 50): every dims, fillet, ref-axis, sim-param, or driver change
  pushes a `.bsim`-format snapshot onto the undo stack. `_apply_state` restores from a snapshot
  without triggering re-capture (`_applying` guard). Edit menu shows step counts.
- **Unsaved-changes guard**: Save/Discard/Cancel dialog on New, Open Project, and Quit when the
  project is dirty. Window title shows `ProjectName *` when unsaved changes exist.
- **`get_project_params()/apply_project_params()` on GeometryTab and SimulationTab**: these gather
  and distribute widget-authoritative values (spin-boxes, combos) that never round-trip through
  `AppState`.
- **View-manager camera methods on GeometryTab** (`reset_view`, `set_parallel_projection`,
  `set_view`): delegate to pyvistaqt; no-op safely when `_PV_OK = False` (headless/CI).
- **18-test pure-Python test suite (`tests/test_project_io.py`)**: round-trips a
  `DriverPlacement` through `driver_to_dict`/`driver_from_dict` with both inductance kinds,
  `box_volume=None`, multi-driver, schema-marker rejection.
- **15-test pure-Python test suite (`tests/test_settings_merge.py`)**: section-preserving
  `update_settings`, TOML writer correctness, recent-projects dedup/cap.
- **8-test pure-Python test suite (`tests/test_logging_prefs.py`)**: enable/disable/re-enable
  cycle via `configure_logging`.
- **4-test Qt-offscreen test suite (`tests/test_gui_project_roundtrip.py`)**: gather/apply
  round-trip, save→New→load round-trip, undo/redo single step, cardinal-rule guard (no H-tensor
  mutation on load).

### Fixed
- **Duplicate undo entries per dims change** (`geometry_view.py`): `_on_dims_changed` previously
  emitted both `geometryChanged` and `stateChanged`, both connected to `_on_state_changed`, creating
  two undo stack entries per user edit. Removed redundant `stateChanged.emit()` from
  `_on_dims_changed` (geometryChanged alone is sufficient since it is also connected to
  `_on_state_changed`).
- **Open-dataset HDF5 dialog no longer offers `*.bsim`** (`parameters_panel.py`): `.bsim` belongs
  to the separate "Open Project" action.

### Known issues
- **⌘Z / ⇧⌘Z keyboard shortcuts for Undo/Redo do not fire** when a spin box or other
  input widget has keyboard focus (macOS Qt shortcut routing). The `ApplicationShortcut`
  context fix was applied but did not resolve it on this hardware; root cause is still
  under investigation. Undo/Redo remain fully functional via Edit → Undo / Edit → Redo.

## [1.4.3] — 2026-06-23 — Bug-Fix Chunk 5c: HDF5 atomic write + attr hardening (data-integrity)

Third sub-chunk of Chunk 5 (`docs/Chunk5_Gameplan.md`); closes Chunk 5. Saves the dataset to HDF5
from the GUI without corrupting or partially writing the file. Schema (`schema_version`) unchanged.

### Fixed
- **HDF5 save no longer crashes or produces a corrupt partial file (`io/hdf5_store.py`).**
  Root cause: `box_volume_m3=None` (emitted by `TerminalModel.to_attrs()` for free-air/infinite-baffle
  drivers) reached `h5py.attrs[key] = None`, which does `np.asarray(None)` → object dtype →
  `TypeError: Object dtype dtype('O') has no native HDF5 equivalent`. Because `write_dataset` opened
  `h5py.File(path, "w")` (truncates immediately), a raise on the *second* driver left the file with
  only the first driver on disk — the exact `HDF5/run2/HDF5.h5` corruption observed (lists 2 drivers,
  stores 1).
- **`_write_attrs` skips `None` values** (absent attr = unset, the natural HDF5 idiom). The free-air
  `box_volume_m3=None` is not in the DATA_CONTRACT §3.5 list and has no disk representation.
- **`_write_attrs` now gives a clear, actionable error** on any other un-serialisable value: names the
  attr key, the owning driver id, the Python type, and the original h5py message — instead of a raw
  `TypeError` with no context.
- **`write_dataset` is now atomic**: writes to a temp file in the same directory, then
  `os.replace()` (same-filesystem rename) to the target only on full success. A failed write
  never truncates or corrupts a pre-existing file. Temp file is cleaned up on failure.
- `dict` / `list` / `tuple` values are all JSON-encoded (previous code handled `dict`/`list`; `tuple`
  is now also safe, coerced to list before encoding).

### Added
- **`tests/test_hdf5_atomic.py` (14 tests, CI-safe):** (1) exact-bug reproduction — asymmetric
  `box_volume_m3` (float on driver_0, None on driver_1) now writes and round-trips correctly; (2)
  atomicity — a write that raises mid-stream leaves the pre-existing file byte-identical and no `.tmp`
  file on disk; (3) lossless 2-driver round-trip with a rich attr set.

## [1.4.2] — 2026-06-23 — Bug-Fix Chunk 5b: filter-designer steer-to-front + engine guidance (RC2, RC3)

Second sub-chunk of Chunk 5 (`docs/Chunk5_Gameplan.md`); GUI-only, no core/solver change. After 5a
made the engines work, the cardioid still aimed the wrong way: the Filter Designer steered from world
`+z` with no link to the loudspeaker front, so on the user's `+x`-facing opposed-driver box the beam
pointed broadside to the driver pair (run2 `steer_dir=[0,0,1]`). And a "Cardioid" pattern could be
paired with `delay_sum` (the omni corner) with no warning. Schema unchanged.

### Fixed
- **Steering measured from the loudspeaker front axis (RC2, `gui/filter_designer_view.py`).**
  `_steer_dir()` now builds the steer in the dataset's reference frame
  (`core.sphere.reference_frame` → front/right/up): `θ=0` aims straight out `reference_axis`, so the
  default cardioid points where the speaker faces. `load(ds)` initializes `_front_axis` from
  `ds.attrs["reference_axis"]`, resets the steer to on-axis, and shows the front axis in the panel
  (labels: "Steer θ (off front axis)" / "Steer φ (around front)"). Display/intent only — geometry and
  the phase origin are untouched (cardinal rule). Back-compatible: on `+z`-front datasets the reframed
  steer reduces to the previous formula (existing fixtures unchanged).
- **Delay-and-sum guidance (RC3).** A live note under the engine combo warns when `delay_sum` is
  paired with any non-omni target: *"Delay-and-sum only steers — it cannot shape a cardioid or hold
  directivity. Use Least-squares or Auto-Design."* (LS remains the default engine.)

### Added
- **GUI tests (`tests/test_gui_smoke.py`, 3):** steer default follows `reference_axis` (+z default,
  +x when set; θ=0→front, θ=90⟂front, unit-norm); the delay-sum note logic; and an end-to-end
  real-data check (front-steered LS "Cardioid" on the reconstructed run2 H → rear null < −12 dB
  in-band; skips if `HDF5/run2` absent). Full GUI smoke suite green (39).

## [1.4.1] — 2026-06-23 — Bug-Fix Chunk 5a: filter-designer WNG normalization (RC1 keystone)

First sub-chunk of Chunk 5 (`docs/Chunk5_Gameplan.md`), the repair of the filter designer that
"fails to maintain any semblance of directivity shaping" on a real two-opposed-driver loudspeaker
(`First_Run_Bugs.txt` #8). Root cause RC1: the white-noise-gain (WNG) floor — the single robustness
knob — was computed on the **absolute** matched-field power `‖c‖²` (Pa²) instead of the dimensionless
array gain, so for real BEM data (`|H| ≈ 5e-3` Pa at 3 m) the WNG ceiling sat at −43…−29 dB, far
below the user's −6/−20 dB floor. Every bin was flagged infeasible and every adaptive engine clamped
to maximum loading → collapse to the omni / delay-sum corner (the exact `engine=delay_sum`, DI≈0,
all-infeasible result in `HDF5/run2/`). CI never caught it because the synthetic `monopole_field`
fixtures have `|H|≈1` (`‖c‖²≈M`), where the metric reads identically. Schema unchanged.

### Fixed
- **WNG metric normalized to the dimensionless array gain (`beamform/regularize.py`).**
  `white_noise_gain_db` now divides by the average per-element power `‖c‖²/M`, so the matched-field
  corner reaches the scale-free ceiling `10·log10(M)` regardless of the absolute level of `H`;
  `max_white_noise_gain_db` returns `10·log10(M)`; the duplicated inline closure in
  `solve_loading_for_wng` now calls the canonical function (advisor-flagged). `‖c‖²=0 → −inf` guard.
  The change is a per-frequency constant offset, so MVDR/max-dir bisection monotonicity and the
  constant-DI τ-search unimodality are preserved; no solver-internal logic changed. On the
  reconstructed real run2 data this restores a cardioid in-band (DI 4.9 dB, rear null −26 dB @100 Hz)
  and `auto` now correctly selects `ls` (was `delay_sum`); 81/81 bins feasible (was 0/81).

### Added
- **V-WNG-SCALE gate** `tests/test_beamform_wng_scale.py` (CI-safe): design `feasible_mask`/`di_db`/
  `wng_db` invariant to a global `H` scale (the exact masked property), ceiling = `10·log10(M)`, and a
  previously-unreachable −6 dB floor now met on faint (1e-3) data.
- **Real-H reconstruction harness** `tests/_fixtures/reconstruct_run2.py` (local-only; `HDF5/` is
  git-ignored): rebuilds the real 2-driver `H_full[2,81,2562]` from `run2/driver_{0,1}/H_full/*.frd`,
  validated lossless (median rel err 4e-7) against the stored `driver_0/H_full`.

### Notes
- Existing beamform suite (54 tests) passes **unchanged** — no thresholds needed updating (monopole
  fixtures have `‖c‖²≈M`, offset ≈ 0 dB). DI/target-error were always scale-invariant; only WNG broke.
- Cardinal rule intact (no per-driver re-zero); V-5 stays green. RC2/RC3 (GUI steering + engine
  guidance) land in 5b; the HDF5 save-error fix in 5c.

## [1.4.0] — 2026-06-22 — Bug-Fix Chunk 4: model-viewer UX & driver interaction (#1, #2, #3)

Fourth chunk of the first-run bug-fix campaign (`docs/Bug_Fix_Proposal.md`), covering the three
model-viewer / driver-interaction bugs (#1, #2, #3 from `First_Run_Bugs.txt`). All three are GUI /
geometry fixes; none touches the solver, the H-tensor, or the phase origin (cardinal rule preserved,
V-5 green). PyVista/VTK cannot run under offscreen CI, so the load-bearing logic was factored into
pure, headless-testable functions (`geometry/faces.py`), with the interactive VTK path verified
manually and via an off-screen PyVista API check. Findings, decisions, and the testing approach are in
`docs/Chunk4_Findings.md`. Schema unchanged (no `schema_version` bump).

### Fixed

- **#3 — driver orientation no longer resets to +z.** `parameters_panel.TSDialog._prefill` now
  restores the "Face normal" combo from `dp.spec.normal` (it previously never set the combo, so the
  editor always re-defaulted to +z and OK-ing it silently re-zeroed the driver's true orientation).
- **#3 — face-normal authority unified.** The two driver-edit paths (`DriversTab._edit_driver` and
  `geometry_view.GeometryTab._on_canvas_driver_edited`) now share ONE rule —
  `geometry.faces.reconcile_placement(chosen_normal, old_fp, radius, w, h, d) → (spec, face_placement)`
  — so a re-orient persists identically wherever it was edited: same face keeps the position, a new
  orientation moves the driver to that face (recentred, radius clamped to fit). `FacePlacement` stays
  the single source of truth, so `spec` and `face_placement` can no longer silently disagree.
- **#2 — drivers place where you click.** The canvas `driverAdded` signal now carries the clicked,
  face-clamped `(u, v)`, so a new driver lands at the click location instead of always the face centre.

### Added

- **#1 — reference-axis (0°) + virtual-microphone indicator in the 3-D editor.** A new pure
  `geometry.faces.reference_axis_indicator(axis, w, h, d) → AxisIndicator` computes the arrow + mic
  placement; `_DriverEditorCanvas._draw_reference_axis` renders an arrow from the box centre along the
  measurement axis to a scaled stand-off, with a microphone glyph and a `0° / on-axis mic` label, so
  the box-vs-mic orientation is unambiguous from any camera angle. The label conveys direction only —
  it deliberately claims no false distance.
- **Settable reference axis, threaded end-to-end.** New `AppState.reference_axis` (default +z) with a
  6-way "Reference (0°) axis" combo on the Geometry tab, threaded through
  `SimulationTab.build_request → SimulationRequest.reference_axis` (already persisted as the dataset's
  `reference_axis` root attr). This closes the Chunk-1 cross-cutting thread: the editor indicator, the
  stored attr, and the Results On-axis/Balloon/Polar/CEA views now all agree on the loudspeaker front.
- **`geometry.faces` helpers (Qt-free, CI-tested):** `FACE_NORMALS`, `face_id_from_normal`,
  `reconcile_placement`, `AxisIndicator` + `reference_axis_indicator`; `AppState.box_dims` as the
  shared dims source both driver editors read.
- **Discovery hint** under the editor ("click a face to add · drag to move · right-click to Edit/Delete")
  and seven headless tests in `tests/test_gui_smoke.py` (combo↔face_id invariant, prefill restore for
  all six normals, reconcile same-/new-face behaviour incl. radius clamp, the edit→reopen persistence,
  the indicator geometry rotating with the axis, and place-at-click `(u, v)`).

### Notes

- **PyVista is already a soft requirement** (in `[project].dependencies`); the Matplotlib fallback is
  retained for headless / no-GL environments, so the kickoff's "make it a default dep" decision was
  already satisfied — no `pyproject` change was needed.
- **Reference axis is display/metadata only** — it never moves the geometry or the phase origin
  (cardinal rule). It reuses `core.sphere.reference_frame` so the editor and every Results view share
  one convention.

## [1.3.0] — 2026-06-22 — Bug-Fix Chunk 3 complete: filter-designer visualization (3e)

Fifth and final sub-chunk of the beamforming/filter-designer rebuild (#8, `docs/Bug_Fix_Proposal.md`)
— **closing Chunk 3** (`v1.2.1`→`v1.2.4` engines + auto/multi-target, now visualized). 3e is **pure
visualization**: it adds the remaining plot views to the Filter Designer tab, all **read-only** over
the returned `DesignResult` — it never recomputes, re-tunes, or re-solves the design, and never
re-zeros / minimum-phase-ifies a driver (cardinal rule). All five proposal deliverables are now on
screen: per-driver responses, filter magnitude/phase, achieved-vs-target directivity, CEA-2034-A
in-room, and beamwidth/DI/WNG vs frequency — reusing the Chunk-2 plotting infrastructure. Findings,
the load-bearing premises, and the review outcome are in `docs/Chunk3e_Findings.md`. Schema unchanged
(no `schema_version` bump).

### Added

- **Five filter-designer plot views (`gui/filter_designer_view.py`).** The right-hand plot panel is
  now a `QTabWidget` of `_MplCanvas` sub-tabs (mirroring `results_view.ResultsTab`): **Polar**
  (achieved vs target at one frequency), **Directivity** (DI / −6 dB beamwidth / WNG vs frequency,
  with the WNG floor, infeasible-bin markers, multi-target DI/beamwidth reference lines, and the
  target-error on a twin axis), **Filters** (per-driver weight magnitude in dB-re-max + unwrapped
  phase), **Per-driver** (filtered `w_m·H_full` on-axis responses + the combined steered beam), and
  **CEA2034 / in-room** (the steered spinorama, with the Estimated-In-Room curve emphasised).
- **Render-gate tests (`tests/test_gui_smoke.py`).** Assert the *correct series* on every view (line
  counts per driver, labels, axes — not merely "did not raise"), a **cardinal-rule snapshot guard**
  (stored `H_bem`/`H_full` byte-equal after plotting), the multi-target reference lines, the
  **CEA `steer_dir`-referencing guard** (steered +x on a +z-front dataset), the WNG `-inf` / infeasible
  edge cases, and the frequency-combo→polar-only wiring.

### Notes

- **CEA reference axis = the beam axis (`steer_dir`), not the dataset front axis.** The new spinorama
  references the listener's on-axis to the steered beam, so its plotted in-room slope agrees with the
  number `orchestrator.design_multi` already reports (using the dataset front axis would silently
  disagree). Guarded by a dedicated test.
- **Reuse over duplication.** The dB-SPL conversion reuses `results_view._db`; no new plotting helper
  was introduced. The frequency combo redraws the polar view alone (the band-spanning views, incl. the
  per-candidate SH resampling, do not re-run on a frequency change).
- Verified by an **adversarial multi-dimension review workflow** (cardinal-rule safety, plotted-series
  correctness, reuse, test/completeness — each finding adversarially verified). The cardinal-rule
  dimension found nothing; the low/nit findings were applied. Full CI-safe suite green.

## [1.2.4] — 2026-06-22 — Bug-Fix Chunk 3d: multi-target objectives

Fourth sub-chunk of the beamforming/filter-designer rebuild (#8, `docs/Bug_Fix_Proposal.md`). A new
`objective="multi"` (dispatched through `engine="auto"`) lets the user target **{directivity index,
−6 dB beamwidth, in-room CEA-2034-A EIR slope} jointly** via a **scalarized weighted-sum** search
over the existing engines + knobs — producing a *sensible Pareto trade-off* and an **honest
per-objective achieved-vs-target report**. It is a principled *search + scoring* layer on top of the
3a/3b/3c engines: it only **calls** `design()` and Chunk-2's `compute_cea2034`; it never re-tunes a
solver and never re-zeros/min-phases a driver. Both flagged decisions were confirmed at kickoff
(scalarized weighted-sum with hard constraints as feasibility gates; search over (engine, knobs), not
a new joint solver). Research-led and **empirically de-risked against the real `design()`** before any
package change (diagnose → conflict map → prototype → synthesize); the methodology, the measured
conflict structure, the in-room target-slope research, and **five settled premises** (incl. the
non-circular minimax reframe of the gate) are in `docs/Chunk3d_Findings.md`. Schema unchanged (no
`schema_version` bump).

### Added

- **Multi-target Auto-Design (`beamform/orchestrator.py`, `design_multi`).** `objective="multi"`
  enumerates a curated `_MULTI_LADDER` (`ls`×{cardioid, supercardioid, hypercardioid}, `delay_sum`,
  `constant_di`×{gdi 10/12/14/None}, `max_directivity`; all index-mode, `mvdr` dropped as ≡ index
  max-directivity) through the real `design()`, scores each candidate on a **weighted sum of
  NORMALIZED per-objective deviations** (DI vs `target_di_db`, beamwidth vs `target_beamwidth_deg`,
  in-room EIR slope vs `target_inroom_slope_db_per_oct`), gated by the honest WNG floor / any `nulls`
  as lexicographically-prior feasibility constraints, and returns the best feasible candidate. The
  choice is recorded honestly in `attrs` (`auto_class="multi"`, `multi_targets`, `multi_weights`,
  `multi_norm`, `multi_trace`, `multi_achieved`, `auto_reason`, `auto_prescreen`, `band_feasible`).
- **`TargetSpec` multi-target fields** — `objective="multi"` plus optional `target_di_db`,
  `target_beamwidth_deg`, `target_inroom_slope_db_per_oct`, `objective_weights` (all default `None`,
  fully back-compatible). Under `"multi"`, `nulls` is a feasibility **gate**, not a class override.
- **V-MULTI gate** `tests/test_beamform_multi_target.py`: the load-bearing **non-circular minimax**
  check (the balanced design's *worst* normalized deviation is lower than every single-objective
  optimum's — a fact about the achieved fields, not the selector), the explicit 3c-style trade, the
  end-to-end Pareto weight-trace, the honest-report shape, and the cardinal-rule collapse/shared-ramp
  controls on the new path. (The kickoff's literal "beats the extremes on the combined score" line is
  also reported, labeled as the trivial corollary.)
- **GUI multi-target** (`gui/filter_designer_view.py`): a "Multi-target (DI/beamwidth/in-room)"
  pattern that locks the engine to Auto-Design and exposes per-objective {use, target, weight}
  controls (in-room slope defaulting to the research-backed −1 dB/oct); the status line appends the
  per-objective achieved-vs-target summary.

### Notes

- **In-room target slope default = −1.0 dB/oct** (Harman/Olive "preferred" in-room, flatter for very
  directive speakers). On a <1-octave band the EIR slope is largely a constant-directivity proxy; the
  genuine multi-octave downtilt axis is documented in `docs/Chunk3d_Findings.md`.
- The normalization scales (`_NORM = {di:3 dB, beamwidth:12°, inroom:1 dB/oct}`) are fixed physical
  constants derived from each objective's measured span; the minimax trade holds structurally because
  the objectives genuinely conflict (r(DI,BW)=−0.89), not because of the scale choice.

## [1.2.3] — 2026-06-22 — Bug-Fix Chunk 3c: Auto-Design orchestrator

Third sub-chunk of the beamforming/filter-designer rebuild (#8, `docs/Bug_Fix_Proposal.md`). A new
`engine="auto"` lets the user pick a **target**, not an algorithm: the **Auto-Design** orchestrator
(`beamform/orchestrator.py`) runs a target-conditioned escalation ladder over the well-posed engines
built in 3a/3b, scores each on the target's **own** objective metric, picks the best feasible one,
and **reports its choice and where the target is infeasible honestly** — never silently stacking
incommensurable objectives. This is the principled realization of the user's "iterate through all
algorithms until it converges" (proposal Open Question 1, confirmed at kickoff). Research-led and
**empirically de-risked against the real `design()`** before any package change (diagnose → engine-
wins-per-class map → synthesize); methodology, the measured engine-selection margins, and **three
settled premises** are in `docs/Chunk3c_Findings.md`. Schema unchanged (no `schema_version` bump).

### Added

- **`engine="auto"` Auto-Design orchestrator (`beamform/orchestrator.py`).** Classifies the target
  (`nulls` → hard-null; else `TargetSpec.objective`), runs a robust→aggressive candidate ladder
  through the real `design()`, and selects the **optimizer of the class's own metric** among
  candidates that meet the honest WNG floor (best-effort + `band_feasible=False` if none do). On the
  real fixtures it picks the engine an expert would, by a decisive margin on that metric:
  `shape` → **`ls`** (lowest shape error), `constant_directivity` → **`constant_di`** (flattest DI),
  `nulls` → **`lcmv`** (deepest null), `max_directivity` → **`max_directivity`** (highest DI within
  the floor). The choice is recorded honestly in `attrs` (`auto_class`, `auto_trace`, `auto_reason`,
  `auto_prescreen`, `band_feasible`); the concrete engine used is reported as `attrs["engine"]`.
- **`TargetSpec.objective`** (`"shape"` default · `"max_directivity"` · `"constant_directivity"`) —
  the cross-frequency target *intent* that no per-frequency shape field encoded (a non-empty `nulls`
  overrides to the null class). Ignored by every concrete engine, so it is fully back-compatible.
- **Auto-Design gate** `tests/test_beamform_auto_design.py`: each scenario asserts the expert engine
  is chosen **and** beats the runner-up on that target's own metric in the recorded trace (a
  non-circular check of engine behavior, not orchestrator wiring), plus honest infeasibility flagging
  and the cardinal-rule collapse/shared-ramp controls on the new auto path.
- **GUI Auto-Design** (`gui/filter_designer_view.py`): "Auto-Design (pick best engine)" leads the
  engine list (one click away) but is **opt-in** — Least-squares stays the active default, so the
  default "Design" is one fast solve, not the auto ladder's up-to-4. Two pattern entries
  ("Constant directivity", "Maximum directivity") carry the objective; the status line names the
  engine Auto-Design **chose** (`Auto → ls`), warns when the result is best-effort, and exposes the
  selection reason as a tooltip.

### Notes

- The orchestrator only **calls** `design()`; it never re-tunes an engine (3a/3b did that) and never
  re-zeros/min-phases a driver — steering stays entirely in H's inter-driver phase (cardinal rule).
- It forces `directivity_mode="index"` on the `constant_di`/`max_directivity` candidates regardless
  of the caller, so the DI-objective engines optimize Luo's proper directivity index (the `"region"`
  default would make `constant_di` lose its own class — an advisor-flagged silent-failure trap).
- Per-band engine *blending* and multi-target scalarization (in-room/CEA2034 + DI + beamwidth) stay
  deferred to 3d; ladder-trace visualization to 3e.

## [1.2.2] — 2026-06-22 — Bug-Fix Chunk 3b: constant-directivity hardening

Second sub-chunk of the beamforming/filter-designer rebuild (#8, `docs/Bug_Fix_Proposal.md`).
The constant-directivity engine now actually holds **constant directivity** end-to-end on a
multi-driver array (new engine-level **V-CBT** gate), with realizable filters and an honest
white-noise-gain floor. Research-led and **empirically de-risked against the real forward model**
before any package change (diagnose → 4-probe research+prototype workflow → synthesize); the
methodology, measured numbers, and **three corrected premises** are recorded in
`docs/Chunk3b_Findings.md`. Schema unchanged (no `schema_version` bump). Kickoff decisions
confirmed: **MSCD-only** (MECD stays a documented stub); **constraint-preserving** frequency
regularization (not 3a's additive penalty — and the evidence showed the constraint-preserving op
is a global-phase alignment, not a smoothing kernel).

### Fixed

- **`constant_di` optimized the wrong objective (the headline defect).** The shipped engine held a
  *front-cap-to-total power ratio* (`A` = accept-cap covariance) constant — but that lets the main
  lobe narrow while the proper directivity index varies 6.7 dB and the −6 dB beamwidth drifts
  (std 17°). Added `TargetSpec.directivity_mode`: **`"index"`** holds **Luo's proper directivity
  index** (`A = c cᴴ`, `R` = whole sphere — matches arXiv:2407.01860) flat to ~1e-11 dB by
  construction; `"region"` (default, back-compat) keeps the old cap-ratio objective. The GUI and the
  V-CBT gate select `"index"`. VERIFIED: constant directivity index + ~constant beamwidth (ptp 7°,
  std 2.3°) across the flat-CBT band on a 50-driver cap.
- **No WNG floor on `constant_di` / `max_directivity` (Chunk-3a defect #3 carry).** `constant_di`
  now picks the largest *single shared* `tau*` whose worst-bin WNG meets the floor (the WNG-vs-`tau`
  curve is **unimodal**, so the search lives on the descending branch — a naïve "lower `tau`" would
  collapse it); infeasible bands are flagged (`band_feasible`), never silent superdirective garbage.
  `max_directivity` now loads the reject covariance to meet the floor (`solve_maxdir_loading_for_wng`),
  taming its −15 dB superdirective blow-up.
- **Ringy, unrealizable constant-DI filters.** Per-bin MSCD weights are ~5 rad rough across
  frequency — almost entirely a spurious per-bin global phase (secular-root sign) plus noise on
  near-silent drivers. Fixed with a **cardinal-safe global-phase continuity alignment + one shared
  modeling delay** (`align_global_phase`, `choose_shared_delay_complex`) — both per-frequency global
  factors, so `|P|`, the inter-driver phase, DI, beamwidth and WNG are exactly invariant. Honest
  realizability is now gated on `magnitude_gated_phase_roughness` (raw `phase_roughness` overcounts
  silent-driver noise); measured 0.107 rad after alignment.
- **Ill-posed edge bins.** A relative `eps_min·I` floor (`floor_covariances`, default 1e-7) on the
  generalized eigenproblem / secular root keeps every bin well-posed without disturbing the achieved
  directivity (`<0.1 dB` at fixed `tau`).

### Validated / unchanged

- **`frac_mu = 1e-2` kept (3a carry-forward re-validated).** 3a flagged it as near-inert on the
  well-posed cardioid; on a genuine under-determined stressor (3-driver near-collinear, 80–500 Hz,
  supercardioid, strict floor) it cuts cross-frequency roughness ~39% at a DI drift of 0.11 dB and is
  beam-safe through 1e-1 (harmful at ≥3e-1). New regression test locks this in.
- **MECD** remains a documented `NotImplementedError` stub (MSCD-only, per kickoff decision 1).

### Gate / tests

- New **V-CBT engine-level gate** `tests/test_beamform_cbt_band.py`: `design(constant_di, "index")`
  on a 50-driver cap holds constant directivity index + ~constant beamwidth, realizable filters, an
  honest WNG floor, with two cardinal-rule proofs (near-collapse → DI ≈ 0; shared-ramp invariant).
- Chunk-3a V-cardioid, V-RT, V-5 phase-origin, V-1/V-2 and all existing constant-DI tests stay green
  (full CI suite: 446 passed).

## [1.2.1] — 2026-06-22 — Bug-Fix Chunk 3a: beamformer core formulation fix

First sub-chunk of the beamforming/filter-designer rebuild (#8, `docs/Bug_Fix_Proposal.md`).
Fixes the three confirmed LS pressure-match defects so a **2-driver dual-opposed cardioid**
holds across a band with realizable filters and an honest robustness floor — the new
cardinal-rule proof. Research-led and **empirically de-risked against the real forward model
before any package change**; the methodology, measured numbers, and two corrected research
premises are recorded in `docs/Chunk3a_Findings.md`. Schema unchanged (no `schema_version`
bump). Kickoff decisions confirmed: Auto-Design = principled escalation ladder (3c);
in-room = CEA-2034-A, with target DI/beamwidth/constant-directivity kept as 3d objectives.

### Fixed
- **LS target was real and frequency-independent (defect #1).** `beamform/targets.py` broadcast
  a real `a + (1-a) cos` pattern across all frequencies and cast custom targets through
  `np.real`. Replaced with a **complex, frequency-dependent virtual-source target**
  (`build_virtual_target`): the field of an ideal first-order source (origin monopole +
  normalized origin dipole) synthesized with the **same** `monopole_field` operator the solver
  inverts, so the target carries the correct complex angular structure and finite-radius radial
  phase. Custom complex targets are preserved (no more `np.real` cast). VERIFIED `b/g_mono =
  a+(1-a)cos` to 6e-8.
- **Per-frequency solves were uncoupled / ringy (defect #2).** Added a frequency-COUPLED LS
  path (`ls_pressure_match_coupled`): all bins solved jointly with a per-driver second-difference
  smoothness penalty across frequency, factoring out ONE shared modeling delay (common latency,
  cardinal-rule safe). Degrades exactly to the per-bin solve for `F < 3`.
- **No honest WNG floor for LS (defect #3).** Replaced the old `lambda_for_ls` Tikhonov-fraction
  heuristic (which collapsed directivity at the default robustness) with an in-solve
  per-frequency WNG-floor **grid search** (LS WNG is non-monotone in λ, so bisection is invalid),
  reusing the existing `white_noise_gain_db` / `max_white_noise_gain_db`. Bins whose WNG ceiling
  is below the floor are flagged in `feasible_mask` and rolled off gracefully — never silent
  garbage.

### Added
- **V-cardioid gate** — `tests/test_beamform_cardioid_band.py`: the dual-opposed cardioid held
  across 150–600 Hz (DI ≈ 4.77 dB, rear null ≤ −22 dB at every bin); filter realizability
  (phase curvature ~0.02 rad vs the pre-3a 0.47 rad); honest WNG floor binds gracefully; the
  **collapse-to-origin cardinal-rule control** (zero inter-driver phase → DI 0); the complex
  frequency-dependent target; and direct frequency-coupling unit tests.
- **New beamform API** — `targets.build_virtual_target` / `_first_order_a`;
  `weights.ls_bricks` / `ls_pressure_match_coupled` / `phase_roughness`;
  `regularize.ls_wng_lambda_grid`. `design()` records `ls_tau_s` / `ls_lambda` provenance.

### Decisions / departures (flagged per CLAUDE.md; see `docs/Chunk3a_Findings.md`)
- **DR-P2-03:** the LS engine (`spec.engine == "ls"`) is now frequency-coupled by default,
  degrading to the per-bin solve for `F < 3` so all existing single-bin LS tests stay green.
- The proposal's premise *"a real LS target cannot make a cardioid"* is **empirically false for
  M=2** — the LS absorbs the global phase, so a real and the complex target give identical
  per-frequency DI/null. The complex target's real, measured payoff is **cross-frequency filter
  realizability** (30× smoother), which is the actual goal.
- **Frequency coupling is not load-bearing for the 2-driver gate** (the target alone delivers
  realizable filters); it is kept as benign, beam-preserving insurance for 3b's
  superdirective / under-determined regimes and is unit-tested directly, not via the gate.
- The cardioid band is **physics-limited** (~150–670 Hz for d = 0.086 m): above kd ≈ π the
  2-element pair spatially aliases (the analytic delay-sum ground truth shows the identical
  taper). Wider bands need more drivers / smaller spacing.

## [1.2.0] — 2026-06-21 — Bug-Fix Chunk 2: results visualization & diagnostics

Second chunk of the First-Run Bug-Fix campaign (`docs/Bug_Fix_Proposal.md`): make the
results views **trustworthy and legible** (#9, #10, #11), add a CEA-2034-A spinorama panel,
and add a far-field display option. Built on Chunk 1's proven `reference_axis` metadata.
Schema unchanged (no `schema_version` bump). Diagnose-first confirmed the proposal's
findings in code; previews of the fixed views are in `docs/chunk2_preview/`.

### Fixed
- **Polar plots were jagged (#10).** `_PolarView` masked ~3–6 scattered Lebedev/icosphere
  points by a crude `|cos θ| < 0.25` band and line-plotted them; it also hardcoded +z. Now
  it **SH-resamples** (`core.sh_transform`) onto a smooth 361-point great-circle arc in the
  horizontal or vertical plane built from the dataset's **reference axis**.
- **Directivity map was misleading (#9).** `_DirectivityMapView` θ-sorted all N points
  (mixing azimuths) and `imshow`'d on a **linear** frequency axis. Replaced with separate
  **horizontal & vertical sonograms** on a **log** frequency axis, SH-resampled per
  frequency (`pcolormesh`, normalised dB, shared 0-dB reference).

### Added
- **CEA-2034-A / spinorama panel (#11)** — new `metrics/cea2034.py`: On-Axis, Listening
  Window, Early Reflections, Sound Power, the two DI curves (SPDI/ERDI), and the Estimated
  In-Room response, computed with the **exact CTA-2034-A angle sets and sound-power area
  weights** (verified against the `pierreaubert/spinorama` master implementation). All
  spatial averages are power-domain; a new "CEA2034" Results sub-tab plots SPL curves (left
  axis) and DI curves (right axis) on a log-f axis.
  - *Departure flagged (CLAUDE.md):* the proposal said "reuse `power_di.directivity_index`",
    but CEA DI (LW−SP, LW−ER) is a different quantity than max/mean intensity; the CTA
    definitions are implemented, with the sphere-quadrature SP kept only as a test cross-check.
- **Far-field display option** — new `core/field_referencing.py`: a dataset-wide,
  **display-only** referencing toggle (Near-field / Far-field: acoustic-center / Far-field:
  SH extrapolation) that every directional view honours. Acoustic-center divides out each
  driver's 1/r spreading + path-length phase about its position; SH extrapolation gives the
  true r→∞ radiating pattern via outgoing spherical-Hankel ratios. Both make a low-frequency
  single driver read near-omni; **neither ever mutates the stored H-tensor** (cardinal rule).
- **H_bem vs H_full in-UI (#11)** — a field selector + data-contract tooltip on every
  directional view, so the user always knows whether raw-BEM (unit cone velocity) or the
  full terminal response is shown.
- **`core.sphere.reference_frame`** — the right-handed (front, right, up) measurement frame
  shared by the polar, sonogram, and CEA2034 orbit construction.
- Tests: `tests/test_cea2034.py` (angle sets + monopole/dipole/cos² vs hand-computed
  spinorama values + reference-axis invariance + SP quadrature cross-check),
  `tests/test_field_referencing.py` (far-field omni behaviour + cardinal-rule guard), and
  extended `tests/test_gui_smoke.py` (polar/sonogram/CEA sub-tabs + referencing combo).

## [1.1.0] — 2026-06-21 — Bug-Fix Chunk 1: data & solver correctness + logging foundation

First chunk of the First-Run Bug-Fix campaign (`docs/Bug_Fix_Proposal.md`): fix the
multi-driver persistence corruption (#7) and lay the logging foundation (#5). Diagnosed
**spike-first** from the user's real `HDF5/Dr1.h5` — a 7-entry `driver_order` with a
duplicated `driver_4` but a single surviving driver group.

### Fixed
- **Duplicate `driver_id` silently corrupted the dataset (#7).** Root cause: the GUI minted
  ids as `f"driver_{len(drivers)}"`, which **reuses an index after a middle driver is
  deleted** → two drivers share an id → `h5py` raises on the second `create_group` mid-write,
  leaving a partial file whose `driver_order` is longer than its group set, which
  `read_dataset` then "reads" by **duplicating the survivor and dropping the rest** (no error).
  Reproduced exactly, then fixed in depth:
  - `core/driver_ids.py` (new): `next_driver_id` (lowest free `driver_N`, never reuses one in
    use) and `make_unique_id` (de-dups a user-typed id) replace the count-based scheme in
    `gui/geometry_view.py` and `gui/parameters_panel.py` (add **and** edit paths).
  - `validate_unique_driver_ids` is enforced at every contract boundary: `run_simulation`
    (before the solve — fails in ms, not after a multi-hour run), `build_dataset`, and
    `write_dataset` (**before** opening `"w"`, so a bad dataset never truncates a good file).
  - `read_dataset` now **refuses** a corrupt file (`driver_order` vs group mismatch) with an
    actionable message, while still reading a valid legacy file that predates the attr.

### Added
- **Reference / measurement axis metadata.** `SimulationRequest.reference_axis` (default `+z`)
  is written as the `reference_axis` root attr; `core.sphere.nearest_direction_index` picks the
  on-axis direction along it. The Results **On-axis** view now uses this instead of a hardcoded
  `argmax(z)`, and the **Balloon** view draws a 0°/on-axis indicator. Additive-optional, so the
  default path is byte-identical to before (`argmax(dot(uv,+z)) == argmax(uv[:,2])`).
- **Logging foundation (#5).** `core/logging_setup.py`: `get_logger` (library code) +
  `configure_logging(file, level)` (app/CLI/tests), with a `NullHandler` on the `beamsim2`
  logger. Wired into `pipeline/run.py` (run summary, per-driver, non-convergence) and the
  NumCalc adapter. The GUI Preferences toggle lands in Chunk 5.
- **`tests/test_solver_correctness.py`** (CI-safe, no NumCalc): a synthetic two-monopole pair
  sharing the global phase origin proves `driver_order` integrity + round-trip, duplicate-id
  rejection (no partial file), corrupt/legacy read handling, `reference_axis` round-trip, and
  low-freq near-omni directivity about the axis that **grows with frequency** (inter-driver
  time-of-flight — V-5's concern). Plus the exact GUI delete-then-add id-collision scenario.

### Notes
- **`schema_version` stays `1.0`.** `reference_axis` is additive and optional (safe `+z`
  default when absent), so no on-disk-format bump — which also avoids warning on every existing
  1.0 file via `_check_schema_version`.
- **Cardinal rule intact.** No change to phase origin or geometry; `tests/test_phase_origin.py`
  (V-5), V-1, V-2, and the full suite stay green.
- **Open question for the user (proposal §2):** whether the low-frequency directivity display
  should switch to a far-field / acoustic-center-referenced convention (loudspeaker directivity
  is conventionally far-field). That is locked-architecture-adjacent — **flagged, not changed.**

## [Unreleased] — Phase 2 kickoff: beamforming filter designer (2026-06-20)

Start of **Phase 2** — the automatic beamforming filter designer that consumes the
Phase-1 `H[M×F×N]` tensor and solves per-driver weights `w_m(f)` to steer/shape the beam.
This entry is the kickoff (docs + package scaffolding); implementation lands stage-by-stage
on `feature/phase2-filter-designer` (build order P2-0…P2-5).

### Added
- **`docs/Phase 2 - Filter Solver.md`** — the authoritative Phase-2 gameplan (DR-P2-01…06,
  pipeline, filter/data contract, verified core math, GUI, validation V-tests, milestones,
  risk register, build order), mirroring `BEAMSIMII_Gameplan.md`.
- **`docs/Research Phase 2.md`** — the deep, adversarially-verified research report (synthesis
  + full per-topic dossier) the gameplan distills.
- **`src/beamsim2/beamform/`** — package scaffold (Qt-free): `covariance.py` (house-convention
  look vector `c=conj(H_look)` and covariance `R=conj(H)·diag(a)·Hᵀ`, fully implemented) and
  `weights.matched_field` (the max-WNG / delay-sum corner, implemented); `targets`, `weights`
  (LS/MVDR/LCMV/Luo), `regularize` (WNG-floor), `forward`, `design`, `realize` are stubbed with
  signatures + docstrings for their stages.

### Notes
- House sign convention pinned (DR-P2-02): the coded forward model is `P=Σ_m w_m·H_m`; the
  microphone-array conjugate convention would silently mirror-steer. A round-trip steering test
  is the arbiter (Stage P2-0a). Cardinal rule preserved — the beamformer never re-zeroes a driver.

### Stage P2-0 — foundation (grid + SH + contract hardening)
- **`core/sphere.py`**: `icosphere(subdivisions)` near-uniform grid (no vendored tables;
  spherical-area weights summing to 4π) scaling to thousands of points (2562 / 10242), plus a
  `make_observation_grid(scheme, n_points)` dispatcher. Resolves DR-P2-06 — the simulator can now
  produce the dense directions beam design/audit needs (previously capped at Lebedev-26).
- **GUI**: new "Balloon (642 / 2562 / 10242 points)" observation-sphere presets;
  `SimulationRequest` gains `sphere_scheme`.
- **`core/sh_transform.py`**: spherical-harmonic forward (least-squares / quadrature) + inverse
  + `resample` to arbitrary directions / regular lat-lon grid / great-circle arcs — the bridge
  from the scattered solve grid to VituixCAD/REW polar arcs, CLF, and CBT beamwidth.
- **Contract hardening**: `pipeline/run.py` writes `diaphragm_area`; `io/hdf5_store.read_dataset`
  guards `schema_version` (warns on missing/minor mismatch, refuses incompatible major).
- Tests: `test_beamform_convention` (V-RT + bug-injection mirror-steer control),
  `test_sphere_dense`, `test_sh_transform` (V-SH round-trip), `test_contract_phase2`.

### Stage P2-1 — LS/pressure-matching engine (engine #1) + WNG robustness
- **`beamform/weights.py`**: `ls_pressure_match` (`w=(conj(H)WHᵀ+λI)⁻¹conj(H)Wb` — house
  convention, not the mirror-steering mic-array form), loaded MVDR, LCMV hard nulls.
- **`beamform/regularize.py`**: the single robustness knob — a white-noise-gain floor solved
  by monotone bisection on the diagonal loading; `lambda_for_ls`.
- **`beamform/targets.py`**: `build_target` for presets (omni/cardioid/super/hyper/fig8/
  wide/narrow), continuous cardioid order, steering, and arbitrary custom patterns.
- **`beamform/forward.py`** + **`design.py`**: achieved DI / −6 dB beamwidth / target error;
  `design(ds, spec) -> DesignResult` with a `feasible_mask` (flags where the array can't meet
  the target/floor — never silent garbage).
- Tests (`test_beamform_engine.py`): first-order DI anchors (cardioid 4.771 / super 5.719 /
  hyper 6.021), LS cardioid in the achievable regime, all engines steer, WNG floor respected
  + flagged above ceiling, LCMV null < −40 dB, WNG-monotone/distortionless invariants.

### Stage P2-2 — Luo constant-directivity engine (engine #2) + V-CBT
- **`beamform/weights.py`**: `max_directivity` (generalized eigenproblem — the per-frequency
  directivity ceiling) and `luo_mscd` (max-sensitivity constant-directivity QCQP via the
  closed-form secular root). `design.py` adds the two-pass `constant_di` engine that holds the
  generalized directivity index constant across frequency (exact, by construction). MECD and
  GRPQ generalized-crossovers are deferred follow-ups.
- Tests (`test_beamform_constant_di.py`): GDI constant across frequency; max-directivity
  varies and dominates; MSCD distortionless with zero quadratic; **V-CBT** — a Legendre-shaded
  spherical-cap CBT holds a constant −6 dB beamwidth ≈ 0.64·(2θ₀) above cutoff (matching Keele)
  while the unshaded cap does not.

### Stage P2-3 — GUI Filter-Designer tab + audit export (v1 usable end-to-end)
- **`gui/filter_designer_view.py`**: a new top-level **"Filter Designer"** tab (5th) — pick a
  pattern preset / cardioid-order / steering direction, an engine, and a robustness (WNG-floor)
  slider; "Design" runs the solver on a background `QThread`; the achieved-vs-target H-plane
  polar and directivity-vs-frequency are plotted; "Export audit…" writes the audit set. Reads
  the in-memory dataset after a solve or an opened HDF5 file. Strict one-way core←gui dependency.
- **`io/filter_export.py`**: `export_filter_design` — the DR-P2-03 audit-first export. Writes
  filtered per-driver `.frd` (design weight baked in) and combined steered `.frd` on matched
  H/V polar arcs (SH-resampled from the scattered solve grid), the raw weights (`.npz`,
  re-loadable), a `manifest.csv`, and a `design.json` summary — openable in VituixCAD/REW.
  `load_design_weights` reloads the weights to reconstruct the beam exactly.
- Tests: `test_filter_export.py` (**V-EXPORT** — weights round-trip reconstructs the designed
  beam to < 1e-12; `.frd`/arc structure) and two new `test_gui_smoke.py` cases (the tab loads a
  dataset, designs inline, replots, and runs the constant-DI engine end-to-end).
- **v1 of the Phase-2 filter designer is now usable end-to-end** (design → view → audit export).
  Deployable FIR/biquad coefficient export remains Stage P2-5 (deferred until a target DSP is
  chosen). MECD and GRPQ generalized-crossovers also remain follow-ups.

## [Unreleased] — Fix click-to-place driver: instant placement + drag (2026-06-19)

### Fixed
- **`TypeError: TSParams.__init__() got an unexpected keyword argument 'Le'`** —
  `_on_canvas_driver_added` (geometry_view.py) previously hand-built a stub `TSParams`
  with a non-existent `Le` kwarg and missing required `Sd`; the crash fired on every
  face click before a driver was ever appended, so nothing placed and Preview showed a
  downstream driver-placement error.  Root-cause: `Le` belongs to `LR2Ladder`
  (the inductance model), not `TSParams`.  Fix: delete the stub/dialog approach and place
  the driver **instantly** using the new `default_terminal_model()` factory — LEAP-style.
- **Dragging a driver rotated the camera instead of moving the driver** — `_on_left_press`
  suppressed camera rotation with `OnLeftButtonDown()` + `iren.CreateTimer(1)`, which is
  unsound and still drove the trackball-camera style.  Fix: swap the interactor style to
  `vtkInteractorStyleUser()` (a no-op) for the duration of a drag; restore the saved style
  in `_on_left_release`.
- **Laggy UI / camera jumping on every driver edit** — `render_scene` called
  `reset_camera()` on every invocation (every driver edit, drag step, etc.), causing the
  viewport to re-fit after each change.  Fix: guard with `_camera_initialized`; call
  `reset_camera()` only on the first render and when box dimensions change.

### Added
- **`beamsim2.driver.terminal.default_terminal_model(name)`** — Qt-free factory returning
  a fully valid `TerminalModel` with canonical woofer defaults (Re=6 Ω, Bl=7 T·m,
  Mms=12 g, Cms=0.8 mm/N, Rms=1 N·s/m, Sd=133 cm², LR-2: Le=0.5 mH / Le2=0.2 mH /
  Re2=3 Ω).  Defaults match TSDialog's spin-box initial values so right-clicking a
  click-placed driver to Edit T/S shows consistent numbers.
- **`tests/test_driver_terminal.py::TestDefaultTerminalModel`** — 4 CI-safe tests
  covering field values, default name, finite audio-band response, and HF sign.

---

## [Unreleased] — Interactive driver placement editor (2026-06-19)

### Added (flagged architecture departure — see below)
- **`src/beamsim2/geometry/faces.py`** — Face-local driver placement model (Qt-free,
  gmsh-free, numpy only).  Defines `FacePlacement(face_id, u, v, radius)` as the
  GUI's source of truth for where a driver sits on a box face.  Provides
  `face_basis`, `face_local_to_spec`, `fits_on_face`, `clamp_uv_to_face`,
  `world_to_face_uv`, `classify_face`, and `validate_spec_on_box`.  The derived
  `DriverSpec.center` is always exactly on the face plane, eliminating the "Mesh
  watertight failure" that occurred when typed coordinates missed the plane.
- **`tests/test_faces.py`** — 35 CI-safe unit tests for the face-local model
  (no VTK, Qt, gmsh, or NumCalc required).
- **`src/beamsim2/gui/geometry_view.py`** — Replaced the static Matplotlib
  preview with a LEAP-style interactive 3-D driver placement editor (`_DriverEditorCanvas`)
  backed by PyVista / VTK (`pyvistaqt.QtInteractor`).  Features:
  - Click any box face to place a driver at the face centroid (Add Driver mode).
  - Drag a driver — movement locked to the face plane, clamped to face bounds.
  - Right-click a driver → context menu: **Delete** / **Edit T/S…**.
  - Box-dimension changes re-derive all face-local driver world-coordinates
    (centroid-tracking, LEAP-compatible).
  - Falls back to the matplotlib `_MeshCanvas` static preview when VTK is absent
    or when running under `QT_QPA_PLATFORM=offscreen` (CI, smoke tests).
- **`src/beamsim2/gui/app.py`** — Cross-tab driver sync: `GeometryTab.driversChanged`
  → `DriversTab.refresh` and `DriversTab.driversChanged` → `GeometryTab.refresh_canvas`.

### Fixed
- **Watertight mesh failure from off-plane driver coordinates** — `assemble_box_driver`
  now validates every `DriverSpec` against the box face planes *before* calling
  `gmsh.initialize`, using `validate_spec_on_box` from `faces.py`.  Previously, the
  docstring promised a `ValueError` but the check was never implemented; the failure
  surfaced only as a cryptic BEM-mesh "open/non-manifold edges" message.  The new error
  message names the offending value and the distance in mm, and also catches disks that
  overflow the face boundary.

### Changed
- **`src/beamsim2/pipeline/run.py`** — `DriverPlacement` gains a trailing optional
  field `face_placement: Optional[FacePlacement] = None`.  All existing 3-arg
  constructions remain valid; V-5 and all test specs unaffected.
- **`src/beamsim2/gui/parameters_panel.py`** — `DriversTab` gains a public `refresh()`
  slot; `_edit_driver` now preserves `face_placement` when editing T/S parameters.

### Architecture departure (flagged, DR-06)
- **PyVista + VTK added as mandatory dependencies** (`pyvista>=0.43`, `pyvistaqt>=0.11`).
  This departs from DR-06's "matplotlib-only visualization" mandate.  Rationale: the
  LEAP-style interactive 3-D drag-and-drop placement editor cannot be done without a
  GPU-accelerated renderer.  Scope is **GUI only** — the core pipeline, backends, and
  all headless solve paths remain VTK-free.  Matplotlib is retained for the Results tab
  (plots) and as the driver-placement fallback when VTK is unavailable.

---

## [Unreleased] — Stage-4 close-the-loop gate (2026-06-19)

### Added
- **`src/beamsim2/validation/closed_loop.py`** — Stage-4 beamforming validation module.
  Provides `monopole_field`, `delay_sum_weights`, `steer_response`, `null_depth_db`,
  and `field_agreement_db`. Analytic point-monopole formula in the engineering
  convention (exp(+jkr)), consistent with NumCalc convention established by V-2.
- **`tests/test_closed_loop.py`** — §8 Stage-4 gate (G3). Two tiers:
  - **CI-safe synthetic** (5 tests, no NumCalc): analytic two-monopole end-fire array.
    Null ≤ −31 dB at −z direction at design frequency (f = c/4d ≈ 1716 Hz, d = 0.05 m).
    Unsteered sum: no null at −z. Bug injection (strip driver B's on-axis phase):
    null completely disappears at −z (fills to 0.0 dB), confirming the null relied on
    correctly preserved inter-driver time-of-flight phase (§3.4 cardinal rule).
  - **Real-BEM** (3 tests, `@pytest.mark.local_only`): V-5 box+2-driver geometry
    (drivers side-by-side in x, d_x = 0.05 m). Null at −x = −24.4 dB at design freq.
    BEM vs analytic monopole-pair RMS error ≤ 2.08 dB at 250–1000 Hz (design freq
    excluded: near-null dB sensitivity amplifies finite-piston vs point-monopole
    difference). Bug injection raises the −x null by 14.2 dB and raises BEM−analytic
    error from 5.88 → 8.28 dB at design freq.
  The real-BEM tier routes through the full data contract: NumCalc → ComplexField →
  build_dataset → HDF5 round-trip → stacked_h_full. Assembly and HDF5 I/O verified
  phase-lossless (max diff = 0.00e+00). This is the §8 Stage-4 gate green.

### Notes
- The cardioid null is at ONE design frequency, not broadband constant-directivity.
  Broadband CD beamforming (CBT, superdirective) belongs to Phase 2.
- Not yet tagging v1.0.0 — remaining blockers G1/G2/G4 (V-3 convergence, V-4 rigor,
  DR-05 timing) remain open per the audit findings.

## [Unreleased] — Phase-1 completion audit (2026-06-19)

Skeptical whole-project review against Gameplan §6/§7/§8/§3/§9. Full findings,
triage, and close-out assessment in `docs/handoffs/HANDOFF_2026-06-19_phase1_audit.md`.

### Fixed (clearly-safe; no behaviour change)
- **Code quality**: `ruff` 68 → 0 and `black` clean across `src/` + `tests/` (removed dead
  locals, reflowed over-long module docstrings, `# noqa: E741` on the SH-degree `l` params).
- **`backends/bempp/adapter.py`**: corrected the top-docstring exterior BIE sign to
  `(K − ½I) p_s = V g_N` / `p_ext = K[p_s] − V[g_N]` (the code was already correct; only the
  docstring stated the interior form).
- **`validation/__init__.py`**: docstring now states truthfully that V-1/V-2/V-4/V-5 are wired
  and V-3/V-6 are not yet implemented.
- **`.gitignore`**: ignore `.serena/` (tool-generated cache).

### Audit verdict (no code change — flagged for decision)
- Full suite green with the NumCalc binary (V-1/V-2/V-5 pass on real BEM; bempp cross-check
  agrees). Cardinal single-phase-origin rule preserved everywhere it executes.
- **Not ready to tag v1.0.0.** Blockers: Stage-4 close-the-loop (beamforming reproduction of a
  CBT/cardioid from the H tensor) is absent; V-3 mesh-convergence test missing; V-4 synthetic-only
  (no reciprocity/energy check); the DR-05 `bem_cap_hz` timing basis is unreliable; several §3.5
  metadata fields are never written by the pipeline; `burton_miller` is ignored by the NumCalc
  backend. Recommend tagging the audit state **v0.2.1** and reserving v1.0.0 for after
  close-the-loop. `schema_version` unchanged.

## [0.2.0] — 2026-06-19 — Stage 1: real single-driver enclosure solve

### Added
- **`tests/test_stage1_enclosure.py`** — `@pytest.mark.local_only` Stage 1 gate test.
  Reference enclosure: 200 × 300 × 200 mm box, 75 mm piston on front face, 100 Hz → 5 kHz
  at 1/3-octave (18 steps), Lebedev-26 sphere, n_epw=6, terminal=None.

### Stage 1 results (2026-06-19, M4 Max 48 GB)

Timing (per-step wall-clock from scheduler, RAM from NumCalc Memory.txt):

| freq (Hz) | n_elem_est | RAM (GB) | wall (s) |
|-----------|-----------|---------|---------|
| 100 | 1 | 0.61 | 28 |
| … | … | 0.61 | 28 |
| 5000 | 2445 | 0.61 | 28 |

Total wall-clock: 56.5 s (0.9 min). All 18 steps converged. HDF5 at `runs/stage1/stage1.h5`.

Physics confirmed:
- On-axis level range: **36.6 dB** (baffle step + diffraction ripple clearly visible; gate: > 3 dB ✓)
- DI at 100 Hz → 5 kHz: **2.1 → 12.6 dB** (rise = 10.5 dB; gate: > 2 dB ✓)
- **Stage 1 gate: PASSED**

DR-05 decision (bem_cap_hz):
- 5 kHz step: 2445 elements, 28 s/step, 0.61 GB RAM
- Extrapolation to 20 kHz (N^1.3 FMM scaling): ~17 min/step, ~39 GB RAM est.
- Full-band 24-step solve estimate: ~2.4 h total
- **DR-05 DECISION: `bem_cap_hz = 20000` (full-band solve is feasible on 48 GB / M4 Max)**
  The top step fits in ~1/3 of available RAM and completes in < 30 min. No splice needed.
  Stage 2 will add the T/S electrical chain (not the HF splice).

### Changed (pipeline instrumentation)
- **`backends/numcalc/scheduler.py`** — `_run_pass()`: records `time.perf_counter()` at
  step launch, emits `{"elapsed_seconds": elapsed}` in the `"step_done"` event (was `{}`).
  Backward-compatible; downstream ignores extra event keys.
- **`pipeline/progress.py`** — `ProgressModel.step_done()` gains optional
  `elapsed_seconds: float = 0.0`; stored in `_step_elapsed`. New property
  `step_elapsed_seconds → dict[(driver_idx, step_idx): float]` exposes per-step timing.
- **`pipeline/run.py`** — `_make_scheduler()` event handler forwards `elapsed_seconds`
  from `"step_done"` event through to `ProgressModel.step_done()`.

### Notes
- `schema_version` unchanged (no on-disk format change).

## [Unreleased] — build-order item 11: bempp-cl validation backend

### Added
- **`backends/bempp/adapter.py`** — `BemppBackend(BEMBackend)`: independent
  Galerkin BEM cross-check on NumCalc via bempp-cl 0.4.2 (Numba JIT on
  Apple Silicon; OpenCL deliberately omitted). Implements the four-method
  `BEMBackend` interface (DR-02) with stateless on-disk serialisation
  (mesh.npz + obs.npz + JSON sidecar) so `prepare()` and `solve()` are
  separate calls with no bempp objects crossing the boundary.
  Physics: exterior Neumann Helmholtz BIE — `(K − ½I) p_s = V g_N` on the
  surface, then `p_ext = K_pot(p_s) − V_pot(g_N)` (Colton & Kress, Thm 3.3
  and 3.22; both signs VERIFIED by V-2 phase gate). Dense LU solve (O(T³));
  convergence_flags all True. Neumann datum `g_N = iωρ v_n` (engineering
  `exp(−iωt)` convention, same as NumCalc and all analytic formulas).
- **`backends/bempp/__init__.py`** — exports `BemppBackend`.
- **`tests/test_bempp_validation.py`** — V-2 sphere benchmark through
  `BemppBackend`, reusing `sphere_benchmark_errors()` unchanged: mean
  magnitude error ≤ 0.5 dB AND phase ≤ 5° at 250/500/1000 Hz (ka ≈
  0.46/0.92/1.83). Actual results: 0.15/0.12/0.09 dB, 0.05°/0.27°/0.92°.
  Two CI-safe conformance tests (no bempp install needed).
- **`pyproject.toml`** — `[dependency-groups] bempp = ["bempp-cl>=0.4.2"]`
  optional group (install with `uv sync --group bempp`; default env unaffected).
  New pytest marker `bempp` registered.

### Notes
- No pipeline wiring (pipeline/run.py stays NumCalcBackend); bempp is
  instantiated explicitly in the validation test only.
- `schema_version` unchanged — no on-disk format change.
- No milestone tag (item 11 is off the Stage-0→4 milestone path).

## [Unreleased] — build-order item 10: headless pipeline orchestrator + PySide6 GUI

### Added
- **`pipeline/run.py`** — headless end-to-end runner: `SimulationRequest`/
  `SimulationResult`/`BoxGeometry`/`DriverPlacement`/`ResourceEstimate`
  dataclasses; `run_simulation(req, backend, progress)` drives Stages A–G
  (geometry → per-driver unit-velocity BEM solve loop → assembly →
  `build_dataset` → `write_dataset`/`write_frd`/`write_sofa`); `estimate_resources`
  calls `backend.estimate` once and scales by M drivers with a coarse heuristic
  fallback for the always-NaN `time_seconds_per_step`.  Qt-free; testable
  headlessly with a fake backend.

- **`pipeline/progress.py`** — Qt-free observable solve-state model:
  `StepState` (QUEUED/RUNNING/DONE/FLAGGED), `ProgressSnapshot` (immutable
  value: `[M, F]` StepState grid, steps_done/total, RAM, ETA, current_driver,
  message), `ProgressModel` (subscribe / driver_started / step_running /
  step_done / driver_finished mutators).  ETA = rolling estimate; RAM = Σ
  est_ram of RUNNING steps.  The GUI subscribes a bound Qt signal as the single
  bridge.

- **`backends/numcalc/scheduler.py`** — minimal, default-preserving `on_event`
  callback hook added to `NumCalcScheduler.__init__` (default no-op → all
  existing tests unaffected).  Emits `step_running`, `step_done`,
  `step_converged` at the existing launch/reap/`read_convergence` points,
  enabling live M×F status grids in the run-monitor (§6 Gameplan).

- **`backends/numcalc/adapter.py`** — `solve()` now honours an injected
  `NumCalcScheduler` (passed via the `scheduler` arg it previously ignored),
  enabling progress-wired solves; falls back to its own internal scheduler when
  `None` or a non-`NumCalcScheduler` is passed.

- **`gui/app.py`** — PySide6 `MainWindow`: four-tab `QTabWidget` (Geometry /
  Drivers / Simulation / Results), `AppState` dataclass, `SolveWorker`
  (`QObject + moveToThread`) with `progressChanged`/`finished`/`failed`
  signals.  The `SolveWorker` builds a `ProgressModel`, subscribes
  `self.progressChanged.emit` as the sole Qt bridge, then calls
  `run_simulation(req, progress=progress)`.  File → Open dataset loads an
  existing HDF5 directly into ResultsTab (no solve needed).

- **`gui/geometry_view.py`** — `GeometryTab`: box dimension spin-boxes + gmsh
  health-check + `mpl_toolkits.mplot3d` mesh preview (no new dependency;
  matplotlib already required).  Emits `geometryChanged` to gate the Run button.

- **`gui/parameters_panel.py`** — `DriversTab` (scrollable list of
  `_DriverRow` + `TSDialog` for T/S / inductance / box-volume entry, derived
  fs/Qts shown read-only) + `SimulationTab` (frequency range, sphere density
  preset — only {6,14,26} offered matching `core.sphere` — Estimate/Run
  buttons) + `RunMonitorWidget` (M×F status grid, progress bar, RAM/ETA
  labels).  Sphere combo deliberately omits "balloon-5°" until finer Lebedev
  tables are vendored.

- **`gui/results_view.py`** — `ResultsTab` with five Matplotlib sub-tabs:
  on-axis (magnitude + phase vs frequency, flagged bins amber), horizontal
  polar, vertical polar, 3-D balloon (scatter coloured by dB SPL), directivity
  map (`imshow` freq × elevation angle).  Export panel: Save HDF5, Export .frd,
  Export SOFA; CLF button present but disabled (SH resampling deferred).

- **`gui/run_monitor.py`** — re-exports `RunMonitorWidget` for discoverability.

- **`tests/test_pipeline_run.py`** — 12 CI-safe orchestrator tests (fake
  backend): per-driver BC correctness (only group m+1 vibrates), unit cone
  velocity, 3-driver group assignment, `stacked_h_full → [M,F,N]` shape,
  H_full identity without terminal, non-convergence flag propagation to
  `flagged_frequencies` and HDF5, HDF5 round-trip, `estimate_resources` shape,
  `work_dirs` populated.

- **`tests/test_progress.py`** — 17 CI-safe `ProgressModel` tests: grid
  initialisation, step-state transitions, RAM accumulation/release, steps_done
  counter, `driver_finished` flag reconciliation, ETA None until first step,
  multiple subscribers, out-of-bounds ignored, snapshot grid is a copy.

- **`tests/test_scheduler.py`** — three new tests for the `on_event` hook:
  `step_running`/`step_done`/`step_converged` events fire correctly; default
  `None` leaves existing behaviour intact; non-converged step emits
  `converged=True` after retry.

- **`tests/test_gui_smoke.py`** — 9 GUI smoke tests (offscreen, no binary):
  `MainWindow` constructs and has four tabs; `ResultsTab.load()` populates from
  a synthetic dataset; `_OnAxisView`, `_BalloonView`, `_DirectivityMapView`
  load without exception; `SolveWorker` emits `finished` and `progressChanged`
  with a fake `run_simulation`; `AppState` defaults; all gui/ modules importable.

- **`tests/test_pipeline_e2e.py`** — `@local_only` end-to-end test: runs
  `run_simulation` on the V-5 box+2-driver geometry (same constants as
  `test_phase_origin.py`), asserts `stacked_h_full → [2,3,26]`, HDF5 round-trip,
  and the V-5 superposition guardrail (`relative_l2 = 1.7e-7`, gate ≤ 1e-3).
  This is item 10's de-facto acceptance gate (no §7 entry exists for the GUI/
  orchestrator).

### Notes
- 3-D rendering uses `mpl_toolkits.mplot3d` (no new dependency); GPU-accelerated
  balloon (`pyqtgraph`) deferred.  Sphere presets limited to {6,14,26}.
  CLF export greyed (SH resampling deferred to a later item).
- GUI does not earn its own semver tag; it rides the Stage-3 `v0.4.0` release
  (§8 Gameplan) when the full multi-driver NumCalc run is completed.

## [Unreleased] — build-order item 9: io/ interoperability exports

### Added
- **`io/frd_export.py`** — `write_frd(out_dir, ds, *, fields, p_ref, driver_ids)`:
  writes one VituixCAD-compatible `.frd` text file per (driver, field, direction) under
  `<out_dir>/<driver_id>/<field>/`.  Exports both `H_full` (measurement-equivalent) and
  `H_bem` (raw BEM at unit cone velocity) by default.  Magnitude = dB SPL re 20 µPa.
  Phase = `np.angle(H)` in degrees — **not re-zeroed** (§3.4 cardinal rule enforced and
  tested).  `manifest.csv` maps every file to its Lebedev direction metadata (index,
  unit-vector x/y/z, θ/φ in degrees).
- **`io/sofa_export.py`** — `write_sofa(path, ds, *, field, driver_ids)`: writes a SOFA
  file (AES69-2022, `GeneralTF` convention via sofar 1.2.3) with M=drivers,
  R=Lebedev-directions, N=frequencies.  Exact complex128 roundtrip verified.
  `GLOBAL_Comment` explicitly records the global-origin phase rule (§3.4).
  `SourcePosition` = cartesian driver positions; `ReceiverPosition` = unit_vectors × r_obs.
  INFERRED: `GeneralTF` chosen over `FreeFieldDirectivityTF` because the latter is for
  rotating-speaker setups (M=directions, R=1 mic) and cannot naturally hold multiple
  drivers in one file (empirically verified with sofar v1.2.3).
- **`io/clf_export.py`** — `write_clf(...)` documented deferred stub (`NotImplementedError`):
  CLF text-data format requires SH-resampling from the Lebedev grid onto a regular lat/lon
  grid; the compiled `.cf2` binary has no open-source writer.  Revisit when a CLF
  balloon consumer is needed.
- **`io/__init__.py`** — public re-exports: `write_frd`, `write_sofa`, `write_clf`,
  `write_dataset`, `read_dataset`.
- **`pyproject.toml`** — `sofar>=1.2.3` added to `dependencies`.
- **`tests/test_frd_export.py`** — 17 pure-Python tests (no `@local_only`): file count,
  manifest existence and row count, frequency/magnitude/phase column values, §3.4
  phase-ramp guardrail (deliberate path-delay ramp survives export exactly), H_full vs
  H_bem difference guard, subset selection, error paths.
- **`tests/test_sofa_export.py`** — 12 pure-Python tests: file write/read, exact complex
  roundtrip for H_full and H_bem, dimension shape [M, R, N], frequency vector,
  ReceiverPosition vs unit_vectors, SourcePosition vs driver attrs, GLOBAL_Comment phase
  note, H_full≠H_bem, driver subset, error paths.

### Notes
- `schema_version` unchanged ("1.0") — on-disk HDF5 contract not affected.
- Stage-3 gate (`v0.4.0`) requires a full multi-driver NumCalc run within ~1–2 days;
  not yet reached.  Item 9 lands in `[Unreleased]`.

## [Unreleased] — build-order item 8: driver/ electrical/terminal chain

### Added
- **`driver/thiele_small.py`** — `TSParams` dataclass (Re, Bl, Mms, Cms, Rms, Sd) with
  `fs`/`Qms`/`Qes`/`Qts` property accessors and `vas(rho, c)` method; `from_datasheet`
  constructor (accepts fs, Qms, Qes|Qts, Vas_m3, Re, Sd); `mechanical_impedance(ts, omega,
  box_volume, rho, c)` → `[F]` complex128 with free-air and sealed-box (Cab air-spring)
  alignments; `cone_velocity(ts, ze, omega, voltage, box_volume)` → `[F]` complex128, textbook
  exp(+jωt) convention.  VERIFIED: Thiele 1971; Small 1972/1973.
- **`driver/inductance.py`** — `PlainLe(Le)` (labeled fallback) and `LR2Ladder(Le, Le2, Re2)`
  (parallel topology: Z_L = jωLe ‖ (Re2 + jωLe2)); `voice_coil_impedance(model, Re, omega)` →
  blocked Ze(ω); `input_impedance(ze, zm, Bl)` → Z_in = Ze + Bl²/Zm (the measurable terminal
  curve).  VERIFIED: Wright, JAES 38(10):749–754, 1990.
- **`driver/terminal.py`** — `TerminalModel(ts, inductance, box_volume, voltage, name)`;
  `terminal_response(model, frequencies, rho, c)` → `[F]` complex128, **engineering exp(−jωt)**
  convention (= conj(u_textbook)); `terminal_responses_for(models, frequencies)` list builder
  wired to `build_dataset(terminal_responses=...)`.  `TerminalModel.to_attrs()` populates §3.5
  per-driver metadata (terminal_response_model, ts_params, box_volume_m3).
- **`tests/test_driver_terminal.py`** — 35 pure-Python tests (no `@local_only`): T/S roundtrip,
  Zm resonance/sealed-box fc, Z_in DC/peak/HF, LR-2 vs plain-Le, **convention lock** (critical:
  asserts Im(Z_in_eng) < 0 at HF and terminal_response = conj(u_textbook) element-by-element),
  sealed-box fc/Qtc shift, output hygiene, list builder, wiring through `build_dataset`.

### Key correctness note — time-convention lock
`H_bem` uses NumCalc's engineering exp(−jωt) convention; the T/S lumped model is textbook
exp(+jωt).  `terminal_response = conj(u_textbook)` performs the one-step conversion in
`terminal.py`.  Without it, per-driver H_full phase would be wrong and inter-driver steering
silently corrupted.  The convention is locked by test assertions on Im sign at HF.

### Deferred (as planned)
- `driver/velocity_profile.py` — spatial BC profiles; deferred (uniform VELO already in ncinp_writer).
- `splice/` — analytic HF tail + blend; gated on Stage-1 timing (not yet run).

## [Unreleased] — build-order item 7: assembly/ + io/hdf5_store + V-5

### Added
- **`assembly/superpose.py`** — `driver_h_bem` (returns raw BEM pressure, no phase
  processing — §3.4 cardinal rule) and `superpose_fields` (linear complex sum of
  per-driver fields; validates shape/dtype).
- **`assembly/phase_origin.py`** — `superposition_residual` (relative_l2,
  max_abs_db, max_phase_deg) and `assert_superposition_matches` (rtol=1e-3 guard
  against accidental per-driver re-zeroing — R-02 mitigation, §3.4 guardrail).
- **`assembly/tensor.py`** — `DriverData` and `RadiationDataset` dataclasses;
  `build_dataset` assembles ComplexField results + `terminal_response` (identity
  `ones[F]` until item 8 implements DR-05) into the `H_bem` / `H_full` triad;
  `stacked_h_full` produces `[M × F × N]` complex128 Phase-2 steering matrix view.
- **`io/hdf5_store.py`** — `write_dataset` / `read_dataset` in the exact §3.6 HDF5
  layout: `/frequencies`, `/directions/`, `/drivers/<id>/H_bem|H_full|terminal_response|convergence_flags`
  plus all §3.5 attrs; complex128 stored natively (exact lossless roundtrip);
  dict/list attrs JSON-encoded; `schema_version = "1.0"`; drivers read in sorted key
  order for determinism.
- **`tests/test_phase_origin.py`** — 19 pure-Python (CI) tests covering superpose
  linearity/guards, positive proof that a simulated per-driver phase-zeroing bug is
  detected by the guardrail (no NumCalc needed), tensor H_full contract and mismatch
  guards; plus **V-5** (`@local_only`): two-driver box superposition vs direct
  two-driver BEM solve, `relative_l2 = 1.7e-7` (gate ≤ 1e-3). V-5 also first real
  exercise of multi-group BC writer `_group_element_runs`.
- **`tests/test_hdf5_roundtrip.py`** — 13 pure-Python tests: bit-exact roundtrip of
  every array (complex128, bool), every §3.5 attr including nested `ts_params` dict,
  `schema_version` present, `stacked_h_full` shape. Stage-3 lossless-export gate.

### Fixed
- **`ncinp_writer` `nelgrp` field** — `chterms[0]` in `NC_Input.cpp` is
  `numElementGroups_` (verified in source); was hardcoded `2`, now
  `max(mesh.group_tags)`. Three-group meshes (driver A / driver B / shell) no longer
  trigger NumCalc's `ielgrp must be <= nelgrp` error at runtime.
- **`ncinp_writer` multi-group BC** — `_validate_bc` previously raised
  `NotImplementedError` for more than one vibrating group (deferred to item 7 per
  docstring). Now supports N scalar vibrating groups; BOUNDARY section loops over all
  of them via `_group_element_runs`. `test_ncinp_writer` updated accordingly.

## [Unreleased] — build-order item 6: RAM-aware NumCalc scheduler

### Added
- **`backends/numcalc/scheduler.py`** — `NumCalcScheduler` and `SchedulerConfig`.
  Launches one `NumCalc -istart S -iend S` process per frequency step; packs
  concurrent processes against a 42 GB RAM budget (48 GB − 6 GB OS headroom),
  highest-frequency-first ordering (R-04), resume on restart (R-08), and a
  single R-07 retry at raised `-niter_max 1000` for non-converged steps.
  Mock-launcher injection point makes the class fully unit-testable without a binary.
- **`tests/test_scheduler.py`** — 18 pure-Python tests: `order_steps` (RAM/freq
  ordering, NaN fallback, ties), `step_completed` (pEvalGrid + "End time:" logic),
  scheduler launch/skip/RAM-gate/retry — all via mock launcher, no binary required.

### Fixed
- **`ncinp_writer` BC leak (non-contiguous vibrating groups)** — replaced
  `_group_element_range` (single over-inclusive lo–hi span) with
  `_group_element_runs` (returns exact contiguous blocks). BOUNDARY section now
  emits one `ELEM lo TO hi VELO …` line per run; rigid elements between driver runs
  are never touched.
- **`adapter._parse_memory_txt`** — rewritten to the real Memory.txt format:
  `<step> <freq_Hz> <ram_GB>` (3 space-separated floats; GB → bytes). Old parser
  expected `"Step N: X MB"` and silently returned all-NaN.
- **`reader.read_convergence`** — detects per-step `NC{S}-{S}.out` log layout
  (written by the scheduler) vs. legacy combined `NC1-{F}.out`, reads each format.
- **`reader.step_completed`** (new) — `be.out/be.{S}/pEvalGrid` exists **and**
  `NC{S}-{S}.out` contains `"End time:"` (crash/partial runs lack the marker).
- **`adapter.solve`** — delegates to `NumCalcScheduler` instead of a single
  blocking `subprocess.run(-istart 1 -iend F)`.

### Tests
- **`tests/test_ncinp_writer.py`** filled in — 16 pure-Python tests covering
  `_group_element_runs` (contiguous/non-contiguous/missing), BC leak proof,
  ELEM velocity encoding, structural section checks, `NotImplementedError` guards.

---

## [Unreleased] — build-order item 5: geometry/ package

### Added
- **`geometry/primitives.py`** — `make_sphere_mesh` and `make_box_mesh` via the
  gmsh OCC kernel. Shared `_extract_tagged_mesh` helper: maps 1-based gmsh node
  tags to 0-based indices, sorts triangles by group_tag for contiguous blocks,
  enforces outward normals.  `make_sphere_mesh` is used directly by the V-2
  physics canary below.
- **`geometry/assemble.py`** — `DriverSpec` dataclass and `assemble_box_driver`:
  fragments a driver disk into a box face via OCC `fragment`, assigns each driver
  its own contiguous element group (1…n; shell = n+1), and asserts contiguity
  at return time.  This closes the open follow-up from item 3 —
  `ncinp_writer._group_element_range` can now reliably use `ELEM lo TO hi` ranges
  without leaking the velocity BC onto adjacent rigid elements.
- **`geometry/health.py`** — `run_health_checks` aggregator plus individual
  checks: `check_watertight` (located plain-English open-edge report),
  `check_normals` (auto-repair inward windings), `check_degenerate`
  (auto-removal of zero-area faces), `check_min_feature` (feature-size warning
  against target edge).  `HealthReport` dataclass.
- **`geometry/mesh.py`** — `target_edge_length(f_max, n_epw, c)` implementing
  DR-03's `c / (f_max · N_epw)` sizing rule; `mesh_geometry` convenience wrapper
  (size → assemble → health-check).  Multi-band routing table deferred to item 6+
  (Stage-3 RAM optimisation; documented TODO).
- **`geometry/import_io.py`** — documented `NotImplementedError` stub (CAD import
  deferred by user decision; parametric path covers all Stage-0/1 use cases).
- **`tests/test_geometry_health.py`** — 28 pure-Python tests (sizing math, health
  checks, gmsh primitives, assembly contiguity) plus one `@local_only` V-2
  physics canary.

### Fixed
- **`ncinp_writer` BC-leak for non-contiguous vibrating groups** — `assemble_box_
  driver` now guarantees contiguous element-index blocks per group and asserts
  this invariant before returning, so the min/max range in `ELEM lo TO hi` is
  always exact.

### Verified (solve-spike, 2026-06-18)
- A box enclosure with a flush disk driver does **not** trigger the
  `NC_GenerateSubelements` MSBE overrun.  The crash is specific to globally-flat
  all-coplanar meshes (the original V-1 piston+baffle geometry); closed 3-D box
  surfaces are safe.  Spike: 542-element box+driver at 500 Hz, converged in 18
  CGS iterations.
- **V-2 physics canary** (`test_gmsh_sphere_v2_gate`, `@local_only`): gmsh OCC
  sphere → NumCalc → magnitude error ≤ 0.5 dB at 250/500/1000 Hz, proving the
  gmsh extraction path is solver-equivalent to the trusted synthetic icosphere.

---

## [0.1.2] — 2026-06-18 — V-1 redesigned (curved geometry); Stage-0 gate passes

### Fixed
- **V-1 redesigned to a spherical-cap-on-rigid-sphere geometry**
  (`validation/analytic_piston.py`, `tests/test_analytic_piston.py`).
  The old flat piston-in-baffle geometry crashes NumCalc (coplanar elements,
  ε = 0 → `NC_GenerateSubelements` overruns the `MSBE` cap). V-1 now uses a 45°
  polar cap vibrating at unit radial velocity on an otherwise-rigid icosphere
  (a = 0.10 m, 1280 triangles) and compares BEM directivity to the **exact**
  spherical-cap closed form `spherical_cap_directivity()` (Legendre /
  spherical-Hankel series, NumCalc engineering convention; VERIFIED against
  Morse & Ingard §7.2, the α→180° omni limit, and the small-cap → flat-piston
  limit). At ka_sphere = 1, 2, 3 (≈546/1093/1639 Hz) mean directivity error is
  0.60/0.75/0.82 dB, inside the 1 dB gate. The residual is BEM discretization
  plus icosahedral azimuthal asymmetry (mesh-independent; not analytic error).
  The flat `make_piston_mesh` / `piston_directivity` are retained for reference.
- Stale `MSBE = 110` docstring literal corrected to the compiled `MSBE = 220`.

### Known issue (documented, not yet fixed)
- **`ncinp_writer._group_element_range` mis-applies the velocity BC for a
  non-contiguous vibrating group.** It emits a single `ELEM lo TO hi` range
  from the group's min to max element index, so if the vibrating elements are
  interleaved with rigid ones the BC silently leaks onto the rigid elements in
  between — wrong physics, not a safe degradation. V-2 never hit this (all
  elements vibrate); the cap mesh did, and `make_spherical_cap_piston_mesh`
  works around it by ordering cap elements contiguously before the rigid
  remainder. **This must be fixed (a fail-loud guard, or per-element BCs)
  before multi-driver meshes in build-order items 6–7.**

## [0.1.1] — 2026-06-17 — V-2 passes; V-1 redesign pending

### Fixed
- **V-2 time-convention correction** (`validation/sphere_benchmark.py`).
  The `pulsating_sphere_pressure()` formula used the Kinsler physics convention
  (exp(+jωt), outgoing wave ∝ exp(−jkr)), but NumCalc uses the **engineering
  convention** (exp(−jωt), outgoing wave ∝ exp(+jkr)). Corrected formula:
  `p(r) = ρc · (jka/(jka−1)) · (a/r) · exp(+jk(r−a))`.
  This equals the complex conjugate of the Kinsler form; magnitude is unchanged,
  phase sign flips. VERIFIED: phase residual after fix < 0.5° at all test
  frequencies.  Previously the phase error was 19°/128°/32° at 250/500/1000 Hz.

- **V-2 mesh resolution** (`tests/test_sphere_benchmark.py`).
  Changed from `subdivisions=1` (80 triangles, 92.8 % sphere area) to
  `subdivisions=2` (320 triangles, ~98 % area). The coarser mesh has a geometric
  amplitude error of 0.57 dB at 250 Hz, just above the 0.5 dB gate; subdiv-2
  brings all three frequencies under 0.15 dB.

- **V-2 phase tracking added** (`validation/sphere_benchmark.py`,
  `tests/test_sphere_benchmark.py`). `sphere_benchmark_errors()` now returns
  `mean_phase_deg` / `max_phase_deg` and the `passed` flag requires both
  magnitude ≤ 0.5 dB **and** phase ≤ 5° at every frequency. With the convention
  fix applied, measured phase errors are [−0.30°, −0.42°, −0.24°] at
  [250, 500, 1000] Hz.

### Changed
- **`make_piston_mesh()` now graded** (`validation/analytic_piston.py`).
  Added optional `h_baffle` parameter. The baffle uses a Distance/Threshold
  gmsh field referenced to the piston boundary circle, coarsening radially
  outward.  Critical addition: `Mesh.CharacteristicLengthExtendFromBoundary=0`
  prevents gmsh from propagating the fine piston-edge size across the entire
  baffle interior.  Result: 86 piston + 918 baffle = 1004 total elements (was
  9245 with uniform sizing).

### Known Issue — V-1 (piston directivity) still failing
NumCalc's `NC_GenerateSubelements` algorithm subdivides each near-field
integration element until the ratio `distance / sqrt(area_subelement) ≥ 1.3`.
For **flat coplanar BEM meshes** (piston + baffle both in z = 0), two adjacent
elements share a plane; the perpendicular distance ε between their planes is
**exactly zero**. This makes `ratdis` a constant (≈ 0.31) that never reaches 1.3,
so subdivision runs until the counter `nsbe` hits the compile-time limit `MSBE`.
The limit is `#define MSBE 220` in `NC_ConstantsVariables.h` (the hardcoded
error string `"MSBE(= 110)"` in `NC_3dFunctions.cpp` is a stale literal — the
actual runtime limit after the rebuild is 220). Increasing `MSBE` further would
not fix the root cause; the subdivision would always eventually crash.

**Implication:** V-1 requires redesigning the BEM geometry to avoid coplanar
elements — e.g., replacing the flat piston+baffle with a spherical-cap piston
on a sphere and comparing to the spherical-cap analytic formula (or to the
flat-piston approximation for small caps on large spheres where the two are
equivalent to < 1 dB in the forward hemisphere). This redesign is the next task.

## [0.1.0] — 2026-06-17

### Added
- **`validation/sphere_benchmark.py`** — `make_pulsating_sphere_mesh` (icosphere
  mesh builder migrated from the roundtrip test; returns `(Mesh, BoundaryConditions)`),
  `pulsating_sphere_pressure` (analytic result using Kinsler physics convention),
  `sphere_benchmark_errors` (mean/max dB error vs. analytic, V-2 pass criterion ≤ 0.5 dB).
- **`validation/analytic_piston.py`** — `piston_directivity` (`D(θ) = 2J₁(ka·sinθ)/(ka·sinθ)`,
  limit → 1 on-axis, VERIFIED Kinsler et al. eq. 7.4.14), `make_piston_mesh` (gmsh flat
  piston + square baffle, group 1 = piston, group 2 = sound-hard ring, +z normals enforced),
  `piston_benchmark_errors` (normalise BEM by on-axis, compare shape to D(θ), V-1 pass
  criterion ≤ 1 dB).
- **`validation/power_di.py`** — `directivity_index` (`DI = 10·log10(max/mean_intensity)`
  via Lebedev quadrature, VERIFIED Benesty et al. §2.3).
- **`tests/test_power_di.py`** (V-4, no `@local_only`) — 4 tests: monopole → 0 dB; cos²θ
  dipole → 10·log10(3) ≈ 4.771 dB (exact on Lebedev-26, which integrates degree-7 poly
  exactly); DI invariant under amplitude scaling; power integral positive and finite. Note:
  a naive half-space step-function test was intentionally replaced with the dipole test —
  the Lebedev-26 grid cannot integrate a step function exactly (~1.7 dB vs. 3.01 dB
  expected), but does integrate cos²θ to full floating-point precision.
- **`tests/test_sphere_benchmark.py`** (V-2, `@local_only`) — pulsating sphere a = 0.10 m,
  subdiv-1, [250, 500, 1000] Hz; asserts mean |magnitude error| ≤ 0.5 dB per frequency.
- **`tests/test_analytic_piston.py`** (V-1, `@local_only`) — piston a = 0.05 m, baffle
  W = 0.40 m, three ka ≈ 1/2/3 frequencies; asserts mean |directivity error| ≤ 1 dB.
- **`tests/test_numcalc_roundtrip.py`** (refactored) — removed duplicated `_pulsating_sphere_mesh`
  and `_subdivide` helpers; now imports `make_pulsating_sphere_mesh` from
  `beamsim2.validation.sphere_benchmark`.

## [Unreleased]

### Added
- **`backends/base.py`** (`BEMBackend` abstract interface) — four-method contract
  (estimate / prepare / solve / extract) using only normalized `core/types` on both
  sides. DR-02 departure approved: `ObservationPoints` added to `prepare()` because
  NumCalc bakes the evaluation grid into `NC.inp` at that stage; the DR-02 essence
  ("only normalized types cross the boundary") is preserved.
- **`backends/numcalc/config.py`** — binary-path resolver (`BEAMSIM2_NUMCALC_BIN`
  env var; explicit arg; `FileNotFoundError` with guidance). Path never hardcoded.
- **`backends/numcalc/ncinp_writer.py`** (minimal) — writes `NC.inp`, boundary-mesh
  `Nodes.txt`/`Elements.txt` (PROPERTY 0), and evaluation-grid `Nodes.txt`/`Elements.txt`
  (ConvexHull triangulation, PROPERTY 2, single group). Supports one vibrating group
  with a uniform scalar `VELO` BC; conventional BEM (method 0); single multi-frequency
  `NC.inp`. Three format facts found in NC_Input.cpp and fixed: (1) `PLANE WAVES`
  keyword must be omitted when `n_planewaves=0` (the parser skips the block entirely
  and chokes on the keyword); (2) frequency-curve y-axis is in Hz directly, not scaled
  by 10 000; (3) log file is `NC1-{F}.out`, not `NC.out`.
- **`backends/numcalc/reader.py`** — parses `be.out/be.N/pEvalGrid` into
  `[F, N] complex128` (asserts eval-node count per file to catch silent desync);
  parses `NC1-{F}.out` for per-step convergence flags.
- **`backends/numcalc/adapter.py`** (`NumCalcBackend`) — full four-method adapter;
  `meta.json` sidecar bridges `frequencies`/`n_obs` from `prepare()` to `extract()`
  without touching `core/types.py`; pressure passed raw (cardinal rule §3.4).
- **`tests/test_numcalc_roundtrip.py`** (`@local_only`) — smoke test with a
  pulsating-sphere mesh (a = 0.10 m, icosphere subdiv-1, [250, 500] Hz, Lebedev N=14
  at 1 m). Asserts `pressure.shape == (2, 14)`, `complex128`, finite, non-zero,
  all-converged. Mesh geometry (origin-centered, outward normals, raw phase) is
  preserved for item 4 analytic validation. Skips without binary; `uv run pytest`
  (without binary) stays green.
- `pyproject.toml`: registered `local_only` pytest marker.
- Initial project skeleton: package structure, pyproject.toml, config files.
- Authoritative design documents in `docs/`.
- `CLAUDE.md`: project-level coding instructions for Claude Code sessions.
- `docs/DATA_CONTRACT.md`: full §3 data contract (H tensor schema, sphere sampling,
  frequency grid, single-phase-origin rule, HDF5 layout), extracted from gameplan.
- `docs/CODING_STANDARDS.md`: full §5.1 coding standards, extracted from gameplan.
- `core/types.py`: normalized data types crossing the solver-abstraction boundary —
  Mesh, BoundaryConditions, FrequencyGrid, ObservationPoints, SolverConfig,
  ComplexField, ResourcePlan, SolveSpec, RawSolveResult.
- `core/sphere.py`: Lebedev–Laikov quadrature grids for n = {6, 14, 26} points,
  with analytically verified weights (sum_4pi convention, exact to algebraic degrees
  3, 5, 7 respectively). Fliege–Maier, t-design, and icosphere raise NotImplementedError.
- `core/units.py`: speed of sound c(T, RH, P), air density ρ(T, RH, P), and
  air-attenuation stub (model="none"). Dry-air ideal-gas formulas with HEURISTIC labels.
- `tests/test_sphere_grids.py`: 39 tests — quadrature weights sum to 4π, ∫1 dΩ = 4π,
  unit norms, Y₀⁰ integral, SH orthonormality diagonal and cross-terms for l ≤ 3,
  θ/φ roundtrip, and error-handling paths.
- `tests/test_core_foundation.py`: 26 tests — dataclass construction and shape
  validation, c(20 °C) ≈ 343.2 m/s, ρ(20 °C) ≈ 1.204 kg/m³, attenuation stub.

### Fixed
- `.gitignore`: added `NC.out`, `NC.log`, `NC*.out` to exclude NumCalc runtime output files.
