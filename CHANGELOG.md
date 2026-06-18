# Changelog

All notable changes to BeamSimII are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning: [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
