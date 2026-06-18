# CLAUDE.md — BeamSimII

BeamSimII is a macOS desktop app that simulates a loudspeaker's full-3D acoustic
radiation with the Boundary Element Method (BEM), then (Phase 2) designs per-driver
beamforming filters from that simulation. **Current focus: Phase 1** (the radiation
simulator). The user is an acoustics/measurement expert, **not a programmer** — write
all code complete and runnable, and explain numerical/DSP ideas with acoustics analogies.

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
- Run a pipeline tool:      `uv run python -m beamsim2.pipeline.run --help`
- Full test suite:          `uv run pytest`
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
- Work ONE build-order item (Gameplan §10) at a time. Use plan mode to explore and plan
  first on anything multi-file or unfamiliar; skip planning for one-line fixes.
- Each item's matching validation test (§7) is the finish line: implement, run it, iterate
  until green, then move on.
- The headless core/pipeline gates each milestone; the GUI is last and never on the critical path.

## Git / versioning
- `main` always passes its tests. Non-trivial work goes on a short feature branch
  (e.g. `feature/numcalc-adapter`) and merges when green; tiny fixes can go straight to main.
- One commit = one coherent change with a clear message. Tag milestones with semver
  (Stage 0 -> v0.1.0 … Phase-1 complete -> v1.0.0) and update CHANGELOG.md.
- Bump the data contract's `schema_version` (separate from the app version) ONLY when the
  on-disk format changes; note it in CHANGELOG.md.
- Never commit large solve outputs, meshes, the NumCalc binary, or the Mesh2HRTF checkout
  (see `.gitignore`). Use the `gh` CLI for GitHub operations.

## Gotchas
- NumCalc can fail to converge at the highest frequencies (critical/irregular frequencies).
  Detect non-converged steps, retry with more iterations, then flag + interpolate — never
  emit silent garbage.
- Imported geometry is often not watertight; the geometry health-check stage must surface
  located, plain-English errors. Driver diaphragms are always app-generated primitives, so
  their elements are auto-tagged for the vibrating boundary condition (no face-guessing).
