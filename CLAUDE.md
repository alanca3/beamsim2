# CLAUDE.md — BeamSimII

BeamSimII is a macOS desktop app that simulates a loudspeaker's full-3D acoustic
radiation with the Boundary Element Method (BEM), then (Phase 2) designs per-driver
beamforming filters from that simulation. **Current focus: Phase 1** (the radiation
simulator). The user is an acoustics/measurement expert, **not a programmer** — write
all code complete and runnable, and explain numerical/DSP ideas with acoustics analogies.

**Build-order status (2026-06-19):** All 11 items complete and merged to `main`.
Current milestone: **Stage 1** — first real single-driver enclosure solve, RAM/timing
measurement, `bem_cap_hz` decision (DR-05), earns `v0.2.0`.

## Authoritative docs — read the relevant one before non-trivial work
This file does NOT replace the project docs; it points to them. The architecture and the
"why" live in the docs. This file is only the always-on rules + commands + pointers.
- `docs/BEAMSIMII_Gameplan.md` — THE architecture/spec. Decision records (DR-01…DR-06),
  pipeline stages A–G, the data schema (§3), validation tests (§7), milestones (§8),
  build order (§10). Authoritative; flag departures, do not change silently.
- `docs/DATA_CONTRACT.md` — the Phase-1 output schema. Read before any work in
  `assembly/` or `io/`.
- `docs/CODING_STANDARDS.md` — full code standards (summarized below).
- `docs/BEAMSIMII_Project_Overview.md`, `docs/BEAMSIMII_First_Research.md` — background.
Ignore all prior "beamsim" v1–v5 work.

## Cardinal rule — must never break
Every per-driver response shares ONE common spatial phase origin (the global coordinate
origin), preserving each driver's true time-of-flight. NEVER minimum-phase-ify or re-zero
a driver independently — doing so silently mis-steers the Phase-2 beam. The two-driver
superposition test (`tests/test_phase_origin.py`, validation V-5) guards this and must pass.

## Locked architecture (flag before changing — never change silently)
- BEM engine; the app fully abstracts the solver (the user never configures BEM directly).
- Primary backend: NumCalc (Mesh2HRTF), behind the `BEMBackend` interface
  (`src/beamsim2/backends/base.py`). bempp-cl = validation backend; COMSOL = manual fallback.
- Driver model: the cone is a vibrating surface inside the BEM mesh (rigid piston / simple
  cap); T/S parameters + a lossy voice-coil inductance model give the on-axis terminal
  response, applied as a per-driver complex multiplier across all directions. Cone breakup
  is out of scope.
- Output: per-driver complex `H[driver x frequency x direction]` on a near-uniform sphere
  grid (Lebedev default), native HDF5. Designed backward from what the Phase-2 beamformer needs.
- No AKABAK/ABEC. Open-source / self-contained. macOS GUI is PySide6, and is built LAST.

## Environment / hardware
- macOS 15.7.7, Apple M4 Max (12 performance + 4 efficiency cores), 48 GB unified memory.
  Dev machine = solve machine. The binding constraint is RAM, not cores.
- NumCalc is CPU-only and single-threaded per process; parallelism = many frequency-step
  processes scheduled against available RAM (highest-frequency-first, since those are the
  most RAM-hungry). It is a C++ binary built from source with `make`, not a pip package.

## Commands
- Sync dependencies:        `uv sync --group dev`
- Sync incl. bempp backend: `uv sync --group bempp`   (optional; adds numba/llvmlite)
- Run a pipeline tool:      `uv run python -m beamsim2.pipeline.run --help`
- Full test suite:          `uv run pytest`
- CI suite (no hardware):   `uv run pytest -m 'not local_only and not bempp'`
- One test file:            `uv run pytest tests/test_phase_origin.py`
- Format + lint:            `uv run black . && uv run ruff check .`
Prefer running single tests while working; run the full suite before closing a session.

## Code standards (full version in docs/CODING_STANDARDS.md)
- Complete, runnable code only — never pseudocode, never "fill in the rest." If a file is
  too long for one response, say so and split it into explicitly labeled parts.
- Dimensional comment on every significant array, e.g. `# H_bem: [F x N] complex128`.
- Every function: plain-English purpose + parameter/return descriptions (NumPy-style docstrings).
- Label physics/technical claims VERIFIED / INFERRED / HEURISTIC with author/year.
- Type hints on public functions; black + ruff clean.
- A self-test for every subsystem; the full suite must pass before a session closes.

## Workflow
- Build-order items 1–11 are complete. Work is now milestone-driven (Stage 1 → Stage 4).
  Use plan mode before any multi-file or unfamiliar work; skip for one-line fixes.
- Each Stage's acceptance gate (§8) is the finish line — implement, run the gate test,
  iterate until green, then update CHANGELOG and tag the milestone semver.
- The headless core/pipeline gates each milestone; the GUI never on the critical path.

## Git / versioning
- `main` always passes its tests. Non-trivial work goes on a short feature branch
  (e.g. `feature/numcalc-adapter`) and merges when green; tiny fixes can go straight to main.
- One commit = one coherent change with a clear message. Tag milestones with semver
  (Stage 0 -> v0.1.0 … Phase-1 complete -> v1.0.0) and update CHANGELOG.md.
- Bump the data contract's `schema_version` (separate from the app version) ONLY when the
  on-disk format changes; note it in CHANGELOG.md.
- Never commit large solve outputs, meshes, the NumCalc binary, or the Mesh2HRTF checkout
  (see `.gitignore`). Use the `gh` CLI for GitHub operations.

## NumCalc time convention
NumCalc uses the **engineering convention**: exp(−jωt) time factor, outgoing waves propagate
as exp(+jkr). This is the complex conjugate of the Kinsler physics convention (exp(+jωt),
outgoing ~ exp(−jkr)). All analytic formulas checked against NumCalc output must use the
engineering convention or the comparison will show phase errors of tens of degrees.
Pulsating-sphere formula in engineering convention (VERIFIED against NumCalc):
`p(r) = ρc · (jka/(jka−1)) · (a/r) · exp(+jk(r−a))`

## bempp-cl notes (item 11 validation backend)
- Install: `uv sync --group bempp`. The package imports as **`bempp_cl.api`** (underscore),
  not `bempp.api` — `import bempp.api` raises `ModuleNotFoundError`.
- On Apple Silicon, bempp auto-selects the Numba JIT backend (OpenCL unavailable).
  `pyopencl`/`exafmm` are deliberately not installed.
- **Exterior Neumann BIE sign** (easy to get backwards): with normal **n = outward from
  scatterer** (into fluid), the correct exterior BIE is **(K − ½I) p_s = V g_N** and the
  representation formula is **p_ext = K_pot(p_s) − V_pot(g_N)**. The opposite sign
  (½I + K, V − K) silently solves the *interior* problem. VERIFIED: Colton & Kress,
  *Inverse Acoustic and Electromagnetic Scattering Theory*, 3rd ed., Thm 3.3/3.22.

## Gotchas
- NumCalc can fail to converge at the highest frequencies (critical/irregular frequencies).
  Detect non-converged steps, retry with more iterations, then flag + interpolate — never
  emit silent garbage.
- **NumCalc cannot integrate flat coplanar BEM meshes.** The near-field subelement
  subdivision algorithm (`NC_GenerateSubelements` in `NC_3dFunctions.cpp`) terminates when
  `distance / sqrt(area_subelement) ≥ 1.3`.  For two elements in the same plane (z = 0),
  the perpendicular distance ε = 0, so this ratio never grows — the subdivision loops until
  the counter `nsbe` hits `MSBE` and crashes.  **All real-loudspeaker meshes are closed 3-D
  surfaces (curved), so this is not an issue in production.** It only bites flat-mesh
  validation tests.  Fixed (2026-06-18): V-1 replaced the flat piston + flat baffle geometry
  with a spherical-cap piston on a sphere mesh (curved, ε > 0).  Note: the MSBE limit is `#define
  MSBE 220` in `NC_ConstantsVariables.h`; the error message string in `NC_3dFunctions.cpp`
  hardcodes "110" (stale literal — does not reflect the actual compiled limit).
- Imported geometry is often not watertight; the geometry health-check stage must surface
  located, plain-English errors. Driver diaphragms are always app-generated primitives, so
  their elements are auto-tagged for the vibrating boundary condition (no face-guessing).

## Stage-0 status (complete, v0.1.0, 2026-06-18)
V-1 (spherical-cap piston on rigid sphere), V-2 (pulsating sphere), V-4 (DI = 0/4.77 dB) all
green. See `tests/test_analytic_piston.py`, `test_sphere_benchmark.py`, `test_power_di.py`.
The flat piston geometry (`make_piston_mesh`) is retained only as the documented NumCalc-crash
reference. The `ncinp_writer` BC-leak (item 3 open follow-up) was closed in item 5:
`assemble_box_driver` now guarantees contiguous element-index blocks per group and asserts the
invariant at return time — `_group_element_runs` emits exact per-run `ELEM lo TO hi` lines.
