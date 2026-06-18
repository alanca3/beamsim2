# Changelog

All notable changes to BeamSimII are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning: [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
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
