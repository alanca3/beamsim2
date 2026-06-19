# BeamSimII — Setup Notes

## Stage-0 Toolchain Gate: NumCalc arm64 Verification

**Date:** 2026-06-17
**Machine:** MacBook Pro 16" Late 2024, Apple M4 Max (12P+4E cores, 48 GB), macOS 15.7.7

### Result: GATE PASSED

NumCalc built and ran natively on Apple Silicon (arm64) with no errors.

### Build details

**Source:** Mesh2HRTF repository, commit `e45d0436a6fbeca3db13828cbae23ca109225be3`
**Clone location:** `/Users/andy/mesh2hrtf` (outside the beamsim2 repo, never committed)
**Source directory:** `/Users/andy/mesh2hrtf/mesh2hrtf/NumCalc/src/`
**Binary path:** `/Users/andy/mesh2hrtf/mesh2hrtf/NumCalc/bin/NumCalc`

### Architecture check (`file` output)

```
/Users/andy/mesh2hrtf/mesh2hrtf/NumCalc/bin/NumCalc: Mach-O 64-bit executable arm64
```

### Live solve verification

Ran NumCalc against the bundled SHTF test project (1 frequency step, 100 Hz):

```
Step 1, Frequency = 100 Hz
Single level fast multipole BEM
Number of equations = 2412
CGS converged in 29 iterations
Total time: 1 second
Output: be.out/be.1/{pBoundary, pEvalGrid, vBoundary, vEvalGrid}
```

**DR-01 confirmed:** NumCalc builds and runs native arm64 on the M4 Max. The ML-FMM
is active ("Single level fast multipole BEM"). The binary is the production BEM backend
for all Phase 1 solves.

### Notes on build warnings

Clang emitted deprecation warnings for `sprintf` (use `snprintf`) and a VLA extension
warning in `NC_Input.cpp`. These are cosmetic — upstream code, zero effect on
correctness or performance.

### NumCalc on PATH

Binary is at `/Users/andy/mesh2hrtf/mesh2hrtf/NumCalc/bin/NumCalc`. Add to PATH or
symlink to `/usr/local/bin/NumCalc` when needed (requires sudo). The `NumCalcAdapter`
in `backends/numcalc/adapter.py` will accept a configurable binary path so this is not
required for the app to work.

---

## Build-order item 10 — Headless orchestrator + PySide6 GUI (2026-06-18)

### Summary

Item 10 complete and merged to `main` (commit `5d50fbd`). 268 CI tests green.

### What was built

- **`pipeline/run.py`** — headless end-to-end runner (Stages A–G). Key types:
  `SimulationRequest`, `SimulationResult`, `BoxGeometry`, `DriverPlacement`,
  `ResourceEstimate`. Main entry points: `run_simulation(req, backend, progress)` and
  `estimate_resources(req, backend)`. Qt-free; fake backend injectable for CI testing.
- **`pipeline/progress.py`** — Qt-free `ProgressModel` with `StepState`/`ProgressSnapshot`
  observable pattern. GUI bridges via `progress.subscribe(worker.progressChanged.emit)`.
- **`backends/numcalc/scheduler.py`** — added optional `on_event` callback hook
  (`step_running`, `step_done`, `step_converged`; default no-op).
- **`backends/numcalc/adapter.py`** — `solve()` now honours an injected `NumCalcScheduler`.
- **`gui/app.py`** — `MainWindow` (four-tab), `AppState`, `SolveWorker` (`moveToThread`).
- **`gui/geometry_view.py`** — box form + mplot3d mesh preview + health label.
- **`gui/parameters_panel.py`** — `DriversTab` (T/S dialog), `SimulationTab` (freq/sphere/
  Estimate/Run), `RunMonitorWidget` (M×F status grid).
- **`gui/results_view.py`** — On-axis / H polar / V polar / Balloon / Directivity map +
  Export (HDF5 / .frd / SOFA; CLF greyed).

### Tests

- 57 new CI tests (orchestrator, progress model, scheduler hook, GUI smoke/offscreen).
- `@local_only` `tests/test_pipeline_e2e.py`: full `run_simulation` on the V-5 box+2-driver
  geometry → `relative_l2 = 1.692e-07` (gate ≤ 1e-3). PASS.

### To launch the GUI

```bash
uv run python -m beamsim2.gui.app
```

### Notes / deferrals

- 3-D rendering: `mpl_toolkits.mplot3d` (no new dep). GPU balloon deferred.
- Sphere presets: {6, 14, 26} only — larger Lebedev grids not yet vendored.
- CLF export: greyed; needs Lebedev → regular-grid SH resample (future item).
- No semver tag earned — GUI rides the Stage-3 `v0.4.0` milestone, which requires
  the full multi-driver NumCalc timing run (Stage 1 decision: `bem_cap_hz`).
- `time_seconds_per_step` from `backend.estimate()` is always NaN today; `estimate_resources`
  uses a coarse element-count heuristic fallback (0.5 ms/element, approximate).

### Next step

**Stage 1 milestone:** Run a single driver in a real enclosure, measure peak RAM and
wall-clock per step at the top of the band. This sets `bem_cap_hz` (DR-05 decision:
full-band vs splice) and earns `v0.2.0`. The headless pipeline is now ready to do this.

---

## Build-order item 11 — bempp-cl validation backend (2026-06-19)

### Summary

Item 11 complete and merged to `main` (commit `5451e84`). 268 CI tests unchanged.
No semver tag — item 11 is off the Stage 0–4 milestone path (explicitly "when time
allows" in §10).

### What was built

- **`backends/bempp/adapter.py`** — `BemppBackend(BEMBackend)`: independent Galerkin
  BEM cross-check on NumCalc via bempp-cl 0.4.2 (Numba JIT on Apple Silicon; OpenCL
  and ExaFMM deliberately not installed). Stateless on-disk pattern (mesh.npz + obs.npz
  + JSON sidecar) mirrors NumCalc adapter.
- **`backends/bempp/__init__.py`** — exports `BemppBackend`.
- **`tests/test_bempp_validation.py`** — V-2 sphere benchmark through bempp, reusing
  `sphere_benchmark_errors()` unchanged (proves DR-02 abstraction is backend-agnostic).
- **`pyproject.toml`** — new optional `bempp` dependency group; new `bempp` pytest marker.

### Install

```bash
uv sync --group bempp   # pulls bempp-cl 0.4.2, numba, llvmlite, meshio
```

### Test results (V-2 gate: ≤ 0.5 dB, ≤ 5°)

| freq (Hz) | ka    | mean_mag (dB) | mean_phase (°) |
|-----------|-------|--------------|----------------|
| 250       | 0.458 | 0.148        | 0.05           |
| 500       | 0.915 | 0.120        | 0.27           |
| 1000      | 1.831 | 0.087        | 0.92           |

### Key physics note (bug found and fixed)

The exterior Neumann BIE sign with outward-from-scatterer normal n is:
  **(K − ½I) p_s = V g_N**  and  **p_ext = K_pot(p_s) − V_pot(g_N)**

Initial code had the opposite sign (interior-problem form). The V-2 phase gate caught
this immediately (81 dB / 156° error → correct after fix). Reference: Colton & Kress,
*Inverse Acoustic and Electromagnetic Scattering Theory*, 3rd ed., Thm 3.3 and 3.22.

### Notes

- Neumann datum: `g_N = iωρ v_n` (engineering exp(−iωt); VERIFIED by phase gate).
- Dense LU (O(T³)); convergence_flags all True. Not for production solves.
- No pipeline wiring; BemppBackend is instantiated explicitly in the test only.
- All pre-existing 268 CI tests unchanged. Full suite: `uv run pytest -m 'not local_only'`.

### Next step (unchanged)

**Stage 1 milestone:** Run a single driver in a real enclosure and measure RAM/wall-clock.
