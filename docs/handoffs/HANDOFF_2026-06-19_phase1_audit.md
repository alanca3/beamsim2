# HANDOFF — Phase-1 Completion Audit (2026-06-19)

> Skeptical whole-project review before any `v1.0.0` tag. Auditor: Claude (Opus 4.8),
> with 5 read-only deep-dive subagents + the auditor's own reads/runs.
> Checklist: Gameplan §6/§7/§8/§3/§9, DATA_CONTRACT, CODING_STANDARDS, CLAUDE.md.

## Spine of this audit (read first)

**"Build-order items 1–11 implemented" ≠ "Stage gates passed."** Current tag is `v0.2.0`
(Stage 1). Phase-1 "done" → `v1.0.0` requires the **§8 Stage-4 close-the-loop gate**:
reproduce a known constant-directivity pattern (CBT/cardioid) *from the H tensor* via a
beamforming check. **That capability does not exist** (no module, no test — verified by
grep; all "steering/covariance" hits are docstrings describing the Phase-2 contract).

Doc note: the task cited "§6 = what done looks like," but Gameplan §6 is the *GUI sketch*.
The operative done-bar is **§8 Stage-4 gate + all §7 V-tests present & green + a solid data
contract**. Measured against that bar, the project is **not yet at 1.0.0** (see PART 4).

---

## PART 1 — VERIFY (findings)

### 1. Test suite
- CI selection (`-m 'not local_only and not bempp'`): **271 passed, 6 skipped** (skips are
  NumCalc-dependent), 1 `local_only` test "fails" ONLY because `BEAMSIM2_NUMCALC_BIN` is unset.
- With the binary set (`/Users/andy/mesh2hrtf/mesh2hrtf/NumCalc/bin/NumCalc`), the
  `local_only or bempp` selection = **10 passed in 89 s**. Full suite is **green when the binary
  is set**. The earlier "1 failed" was purely the unset env var, not a code defect.

**§7 validation-test matrix** (the user's rule: a green suite MISSING a V-test is not a pass):

| V | What | Module | pytest test | §7 threshold | Verdict |
|---|---|---|---|---|---|
| V-1 | piston/cap directivity | validation/analytic_piston.py | test_analytic_piston::test_cap_benchmark | mean ≤1 dB | **PASS** (real BEM; cap closed-form substituted — documented; ka=1,2,3) |
| V-2 | pulsating sphere | validation/sphere_benchmark.py | test_sphere_benchmark (+ bempp mirror) | ≤0.5 dB + phase | **PASS** (real NumCalc + independent bempp 2nd solver) |
| V-3 | mesh convergence 6→8→10 | **convergence.py = 1-line STUB, zero code** | **ABSENT** | 8→10 ≤0.25 dB | **MISSING** |
| V-4 | power / DI anchors | validation/power_di.py | test_power_di | DI→0 & **→3 dB baffled** **+ reciprocity** | **WEAK** (synthetic only; 3-dB anchor swapped for dipole-4.77; reciprocity check absent) |
| V-5 | two-driver superposition | assembly/phase_origin.py | test_phase_origin::test_v5_two_driver_superposition | within solver tol | **PASS** (strongest: real 3-solve, rtol 1e-3, bug-injection positive controls) |
| V-6 | BEM-vs-analytic diffraction | — | **ABSENT** | diagnostic (not a hard gate) | **MISSING** (low severity) |

`validation/__init__.py` docstring **falsely claims** all of V-1…V-6 are wired as pytest +
harness — untrue for V-3 and V-6.

### 2. Cardinal rule (single phase origin, §3.4) — **PRESERVED everywhere in the live path**
Auditor read `assembly/{superpose,phase_origin,tensor}.py` directly; subagent traced the full
solver→reader→extract→superpose→tensor→export path. No re-zeroing, minimum-phase-ification,
time-alignment, per-driver normalization, or conjugation of the spatial pressure anywhere it
executes. All M drivers solve on one shared mesh at true positions (run.py per-driver loop).
- `core/sh_transform.py` (the only place a min-phase violation could hide) is a **dead 1-line
  stub, never imported** — moot today; must preserve excess-phase if ever implemented.
- The one `conj()` in the driver chain acts on the **scalar terminal response** (EE→engineering
  convention), not spatial/time-of-flight phase. By design.
- V-5 confirmed to exercise a **genuine direct 2-driver BEM solve** vs the per-driver sum.

### 3. Data-contract integrity
- **Complex128 round-trip: LOSSLESS** (compound real/imag; bit-exact `np.array_equal` asserted
  for H_bem/terminal_response/H_full/weights/theta_phi). PASS.
- **Metadata completeness — GAPS (populated in every test fixture but NEVER by the real
  pipeline `run.py`, so the green suite masks the absence):**
  - **MISSING**: `solver_version`, medium `temperature`/`humidity`/`pressure`, `diaphragm_area`.
  - **MIS-POPULATED**: `solver_backend` hardcoded `"numcalc"` (mislabels a bempp run);
    per-driver `name` clobbered by `terminal.to_attrs()` → defaults to `"driver"` when terminal
    name unset; `ts_params`/`terminal_response_model` only written when `terminal is not None`
    (a pure-BEM run emits neither); `profile` constant `"flush_disk"`.
  - **`[3] float64` attrs stored as JSON STRINGS** (`hdf5_store._write_attrs` JSON-encodes lists;
    `run.py` passes `phase_origin`/`position`/`orientation` as Python lists). Round-trips inside
    Python, but a non-Python Phase-2 reader sees a string, not the §3.6-promised float64[3].
- **R-09 schema_version: attribute written but NO READ GUARD** — `read_dataset` never compares
  the file's version to `SCHEMA_VERSION`; a future incompatible file is silently accepted.
  (Also a double-set footgun: `run.py` literal `"1.0"` overrides the `hdf5_store` constant.)
- **interpolated_mask: never written to disk at all** (read fabricates zeros) → §3.6
  non-conformance; never set True by any code.
- **Exports:** `.frd` per-driver/angle text — correct, phase-ramp guardrail tested (PASS).
  **CLF** = clean `NotImplementedError` stub (documented; by design). **SOFA** round-trips
  (rtol 1e-12) but exports only ONE field and omits terminal_response/convergence/medium with no
  "what's lost" warning (Stage-G surfacing intent partly unmet).

### 4. Risk register
- **R-02 (phase origin): MITIGATED** — schema rule + `assert_superposition_matches` +
  V-5 in suite with bug-injection controls. Solid.
- **R-07 (HF non-convergence): PARTIAL.** Detect (reader parses "Maximum iterations") → retry
  once at `-niter_max 1000` → flag → flag reaches HDF5 (`convergence_flags`) + GUI (amber). The
  outcome "**never silent garbage**" holds (flagged + raw pressure retained + visible).
  **BUT (a)** the §2 Stage-F "interpolate flagged bins in SH/min-phase domain" half is
  **unimplemented**; **(b)** `burton_miller` (SolverConfig default `True`) is **NEVER written into
  `NC.inp`** — accepted in the API, read by bempp, but ignored by the NumCalc backend. Since the
  reader only detects the "max-iterations" failure, a *non-unique* solution at a box interior /
  irregular frequency (exactly what Burton–Miller suppresses) could be marked "converged."
  No real-NumCalc non-convergence is exercised by any test (flag plumbing tested with a fake
  backend; log parsing with mock strings — the two halves never join).
- **R-06 (geometry health): PARTIAL but coherent.** 4 checks real, located, plain-English,
  tested, auto-repairs logged (watertight/manifold, normal-orientation+auto-flip, degenerate,
  min-feature). **Self-intersection NOT checked; `import_io` STEP/STL/OBJ are NotImplementedError
  stubs** — but both only affect the *imported* path, which is explicitly deferred; the
  in-scope parametric path is clean.
- **R-05 (splice seam ≤0.5 dB): ABSENT / UNTESTABLE — by design.** `splice/*` are 1-line stubs,
  zero imports, no GUI control, no test. Correct per the recorded DR-05 (`bem_cap_hz=20000`,
  splice off). Flag: the §8 Stage-2 splice gate has no implementation behind it should a cap
  below f_max ever be selected.
- **R-09 (format drift): PARTIAL** — version recorded but unguarded on read (above).

### 5. Cross-backend (bempp) — **AGREES**
V-2 sphere through `BemppBackend` (independent Galerkin BEM, Numba) matches the analytic sphere
to 0.15/0.12/0.09 dB and 0.05°/0.27°/0.92° at ka≈0.46/0.92/1.83 — and NumCalc passes the same
gate. Two independent solvers agreeing is the strongest possible cross-check. One safe-fix: the
bempp adapter **top docstring states the interior-problem sign** `(½I+K)`/`V−K` while the code is
correctly exterior `(K−½I)`/`(K−V)` — doc-only contradiction, code verified correct.

### 6. GUI (item 10) — **SOUND**
`SolveWorker.run` calls `run_simulation(...)` — the **same headless core**, not duplicated logic
(app.py:119). Long solve runs on a background `QThread` (`moveToThread` + queued
`progressChanged`/`finished`/`failed` signals) → window non-freezing. Estimate→Run→Results→Export
wired; Results loads `result.dataset` (the exported tensor) directly; `core`/`pipeline` never
import `gui` (one-way dependency honored). `test_gui_smoke` covers construction/offscreen only —
no real interactive solve is automated (acceptable for a headless-gated project).

### 7. Code quality
- **black: clean** (76 files). **ruff: 68 errors** (34 src / 34 tests): 31 F401 unused-import,
  18 E501, 10 I001 import-order, 5 F841 unused-local, 2 F811 redefine, 2 E741. Inspected the
  F811/F841/E741 individually — **all benign** (test re-imports, dead `uvecs`/`F`/`N` locals,
  `l` as SH-degree symbol); none mask a bug. Violates CLAUDE.md "ruff clean" → safe auto-fix.
- Docstrings / dimensional comments / VERIFIED-INFERRED-HEURISTIC labels: spot-checks across
  assembly/backends/io/validation are consistently present and high quality.

### 8. Stage-1 timing / DR-05 — **INSTRUMENTATION NOT TRUSTWORTHY for the cap decision**
Real `-s` run: **all 18 steps report identical 28.1 s wall and 0.61 GB RAM**, 100 Hz (1 elem) →
5 kHz (2445 elem). Root causes (read from code):
- `wall` = scheduler launch→reap time gated by `poll_seconds=2.0`, reaped in concurrency *waves*;
  each wave's duration is stamped onto every step in it (2 waves × 28.1 s = 56.5 s total). For
  these tiny meshes it is **process/I/O/poll overhead, not compute**.
- `RAM_GB` = NumCalc `-estimate_ram` floor (0.61 GB) — **baseline, not the BEM matrix** (which is
  tens of MB at 2445 elems).
- `n_elem` = a **heuristic** `area/h²` estimate, decoupled from the real gmsh mesh (hence "1" at
  100 Hz). None of the three columns co-vary.
- The `N^1.3`/`N^1.5` extrapolation (test L242/L261) **scales these constants** → meaningless
  17.2 m / 39 GB at 20 kHz. The "feasible" verdict trivially passes on the bogus number.
- The code's printed recommendation is literally `int(_F_MAX)` = **"5000 (full-band feasible)"**;
  the CHANGELOG's **`bem_cap_hz=20000`** is a *human leap*, not the code's output.
- **The Stage-1 *physics* gate IS legitimate and passes on real data**: on-axis range 36.6 dB
  (>3), DI rise 10.5 dB (>2), all-converged. Only the timing/DR-05 basis is unreliable. The §8
  Stage-1 gate's "timing measured" requirement is met only superficially.

---

## PART 2 — TRIAGE

### (a) PASSES — verified working
- Full test suite green with binary; black clean; cardinal rule preserved everywhere live (R-02).
- V-1, V-2 (NumCalc **and** bempp), V-5 present, real-BEM, thresholds met. V-5 has bug-injection
  positive controls — exemplary.
- HDF5 complex round-trip lossless + shape validation; `.frd` export correct; SOFA round-trips.
- GUI drives the headless core on a background thread; one-way core↔gui dependency; results read
  the exported tensor.
- R-06 parametric-path health checks (located, plain-English, logged repairs).
- Cross-backend NumCalc-vs-bempp agreement (independent confirmation).

### (b) BUGS

**b1 — CLEARLY SAFE (auto-fix; touches neither phase-origin, schema, nor a locked decision):**
1. ruff: 68 errors → `ruff check --fix` + manual mop-up of E501/E741 (all benign).
2. bempp adapter top docstring states interior sign; code is correct exterior → fix docstring.
3. `validation/__init__.py` docstring falsely claims V-3/V-6 are wired → correct it.
4. `.serena/` untracked → add to `.gitignore` before close-out.

**b2 — SHOW-FIRST (touches the data-contract / a locked decision / DR-05 — NOT auto-fixing):**
5. **Metadata population gaps** (`run.py` `_root_attrs`/`_driver_attrs`): add `solver_version`,
   medium `temperature`/`humidity`/`pressure`, `diaphragm_area`; fix `solver_backend` to reflect
   the real backend; fix `name`-clobber; write `ts_params`/`terminal_response_model` even for a
   pure-BEM driver. *Fills contract-required fields the pipeline currently omits (a contract
   violation), but writes to the output file → wants your nod.*
6. **`[3] float64` attrs stored as JSON strings** → store as float64 arrays to match §3.6.
   *Changes on-disk dtype → schema-touching → needs approval (+ consider schema_version note).*
7. **R-09 read guard**: have `read_dataset` check `schema_version` and warn/raise on mismatch.
8. **`interpolated_mask`**: write/read it to disk (§3.6 conformance).
9. **`burton_miller` ignored by NumCalc backend** — verify NumCalc's default formulation against
   its source; either wire the flag into `NC.inp` or document that NumCalc applies CBIE/B-M by
   default and make `=False` honest. *Touches DR-01/solver correctness → review, do not auto-fix.*
10. **DR-05 timing**: the `bem_cap_hz=20000` "full-band feasible" basis is unsupported by the
    instrumentation. *DR-05-affecting → your call (re-measure honestly, or re-frame the decision).*

### (c) BY DESIGN / in-scope — do NOT "fix"
- Cone breakup out of scope (R-03). Rigid piston/cap only.
- V-1 cap closed-form (not the 2J₁/x flat piston): documented — flat coplanar meshes crash
  NumCalc (CLAUDE.md gotcha). The cap-on-sphere cross-check is legitimate.
- V-4 dipole-4.77 dB anchor substituting the 3-dB half-space step: documented (discrete grid
  can't integrate the step-function to ≤0.1 dB). *NB: the dipole substitution is defensible, but
  the **missing reciprocity/energy check** is a genuine gap, not by-design — see (d).*
- CLF `NotImplementedError` (no open `.cf2` writer; needs SH resample) — documented out-of-scope.
- Splice stubs + no Stage-2 splice gate: correct per DR-05 (`bem_cap_hz=20000`, splice off).
- `import_io` STEP/STL/OBJ stubs + self-intersection unchecked: imported-geometry path deferred;
  parametric path (the in-scope one) is clean. `profile="flush_disk"` constant likewise.

### (d) DEFER to Phase 2 (or a later Phase-1 hardening pass)
- **V-6** diffraction diagnostic test (§7 says diagnostic, not a hard gate).
- **R-07 interpolation** of flagged bins (SH/min-phase) + `sh_transform.py` implementation.
  Current flag-and-retain is acceptable ("never silent garbage"); decide build-vs-amend-spec.
- STEP/STL/OBJ import + self-intersection check (imported-geometry path).
- SOFA `.verify()` (AES69 strictness) and multi-field/lossy-export warnings.

### GATING ITEMS FOR 1.0.0 (the real blockers, beyond the bug list)
- **G1 — V-3 mesh convergence is entirely missing** (stub). A §7 hard gate. Must be built.
- **G2 — V-4 rigor**: synthetic-only; no real-BEM DI anchor; reciprocity/energy check absent.
- **G3 — Stage-4 close-the-loop ABSENT**: no beamforming reproduction of a CBT/cardioid from H.
  This is the §8 gate that defines Phase-1 "done."
- **G4 — DR-05 cap decision rests on untrustworthy timing** (PART 1 §8).

---

## PART 3 — FIX & RE-VERIFY

### Done autonomously (b1 — clearly safe; touched neither phase-origin, schema, nor a locked decision)
1. **ruff: 68 → 0.** `ruff --fix` (43 auto) + manual: removed dead locals (`run.py` `F`/`N`,
   `results_view.py` `uvecs`, test `rng`/`result`), reflowed 13 over-long module docstrings,
   `# noqa: E741` on the two SH-degree `l` params (conventional math symbol). Inspected every
   F841/F811/E741 first — all benign, none masked a bug. `black` + `ruff` now clean (76 files).
2. **bempp adapter top docstring** corrected to the exterior sign `(K−½I)`/`K[p_s]−V[g_N]`
   (code was already correct; docstring stated the interior form).
3. **`validation/__init__.py`** docstring rewritten to state truthfully that V-1/V-2/V-4/V-5 are
   wired and V-3/V-6 are not yet implemented (was claiming all of V-1…V-6 wired).
4. **`.gitignore`**: added `.serena/` (tool-generated cache).

Re-verified: `black --check` clean, `ruff` clean, CI suite **268 passed**, full validation
suite (local_only+bempp) re-run green after the edits.

### Held for user decision (b2 + gating items) — NOT changed
b2-5…b2-10 (metadata population, JSON-string vs float64[3], R-09 read guard, interpolated_mask,
`burton_miller` wiring, DR-05 timing) and G1–G4 (V-3, V-4 rigor, Stage-4 close-the-loop, DR-05
basis) all touch the data-contract schema, a locked decision, DR-05, or are substantive new
work — surfaced in the triage for your call rather than silently changed.

## PART 4 — CLOSE-OUT ASSESSMENT

### What is solid (the good news)
- **Cardinal single-phase-origin rule is preserved everywhere it executes** (R-02). V-5 passes
  with bug-injection positive controls — the highest-risk rule is genuinely well-guarded. This
  is the thing that most needed to be right, and it is.
- **V-1, V-2 (NumCalc + an independent bempp solver), V-5** pass against real BEM at their §7
  thresholds. Two independent BEM engines agreeing on V-2 is the strongest cross-check available.
- HDF5 complex round-trip is **lossless**; shape validation refuses malformed tensors; `.frd`
  and SOFA exports work; the GUI is a **clean thin shell** over the headless core on a background
  thread (no duplicated logic, one-way core↔gui dependency).
- Full test suite green with the binary; `black`+`ruff` clean after this audit's fixes.
- The DR-02 solver abstraction is real and load-bearing (the same V-2 test runs unchanged on
  NumCalc and bempp).

### Why this is NOT yet the §8/§7/§3 "done" bar for v1.0.0
(The task cited "§6 = done"; §6 is the GUI sketch. The operative bar is §8 Stage-4 + §7 V-tests
+ §3 contract. Measured against that:)

1. **G3 — Stage-4 close-the-loop is ABSENT.** §8 defines Phase-1 "done" as reproducing a known
   constant-directivity pattern (CBT/cardioid) *from the H tensor* via a beamforming check. No
   beamforming module and no such test exist. The project has **not yet demonstrated its core
   purpose** — that the contract actually steers a beam. This is also the ultimate end-to-end
   proof of the phase-origin rule against a known analytic result. **This is THE blocker.**
2. **G1 — a §7 hard-gate test is missing: V-3 mesh convergence** (`convergence.py` is an empty
   stub; nothing re-solves N_epw 6→8→10). The §8 Stage-1 gate's own "stable under mesh
   refinement" clause is therefore unverified. Per the project's rule, a suite missing a V-test
   is not a pass.
3. **G2 — V-4 rigor**: synthetic-only, no real-BEM DI anchor, and the §7 reciprocity/
   energy-conservation check is absent.
4. **G4 — the DR-05 `bem_cap_hz=20000` / full-band decision rests on untrustworthy timing**
   (constant/floor instrumentation, invalid extrapolation). The Stage-3 "full-run within ~1–2
   days for a representative enclosure" claim is likewise undemonstrated (only a 56 s toy run).
   A real 20 kHz (or representative mid-band) solve is needed to set the cap honestly.
5. **Data-contract metadata gaps**: `solver_version`, medium `temperature`/`humidity`/`pressure`,
   `diaphragm_area` never written; `[3] float64` attrs stored as JSON strings; no `schema_version`
   read guard; `interpolated_mask` not persisted. The contract *machinery* is solid; its
   *metadata* is incomplete for a clean Phase-2 handoff.
6. **`burton_miller` is silently ignored by the NumCalc backend** — a latent correctness concern
   (a non-unique solution at an irregular frequency could be marked "converged"). Needs
   verification against NumCalc's default formulation.

### Stage-gate reality (code exists; gates not all met)
All build-order items 1–11 are implemented. But: Stage-0 ✅ (v0.1.0); Stage-1 physics ✅ /
timing-&-cap ⚠️ (v0.2.0); Stage-2 terminal chain ✅, splice N/A by DR-05; Stage-3 V-5 + contract
✅, full-run-timing ❌; **Stage-4 ❌ absent.**

### RECOMMENDATION
**Do NOT tag v1.0.0.** Run the **Stage-4 close-the-loop check first** (reproduce a cardioid/CBT
from H) — it is both the defining §8 gate and the final proof the contract steers correctly. In
the same pass, close **G1 (V-3)**, **G2 (V-4 reciprocity + a real-BEM DI anchor)**, **G4 (one
honest timing measurement to set/confirm `bem_cap_hz`)**, and the **§3.5 metadata gaps**. Verify
**`burton_miller`** against NumCalc.

The current state is a strong, well-validated **Stage-1+ foundation** (core physics correct,
cardinal rule guarded, contract round-trips). Suggest tagging the audit state as a **patch
(v0.2.1)** once the b1 fixes are committed, and reserving **v1.0.0 for after the close-the-loop
check passes**.

### What this audit changed vs. what it only flagged
- **Changed (safe):** ruff 68→0 + black; bempp/validation docstring corrections; `.serena/`
  gitignored; this report. No behavior change; suite still green.
- **Flagged, not changed (your call):** all b2 items (metadata/schema/`burton_miller`/DR-05) and
  the gating items G1–G4. None were touched because each hits the data-contract schema, a locked
  decision, DR-05, or substantive new scope.
