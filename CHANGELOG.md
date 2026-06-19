# Changelog

All notable changes to BeamSimII are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning: [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
