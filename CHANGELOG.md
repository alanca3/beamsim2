# Changelog

All notable changes to BeamSimII are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning: [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
