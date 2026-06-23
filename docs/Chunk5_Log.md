# Chunk 5 — Running Log

> Append-only. Each session: what was done, measured numbers, decisions, surprises.
> See `Chunk5_Gameplan.md` for the plan and `Chunk5_Status.md` for current state.

## Session 1 — 2026-06-23 — diagnosis + 5a (WNG normalization)

### Investigation (read-only, against `HDF5/run2/`)
- `design.json`: `engine=delay_sum`, DI 0.38→5.7 dB, `feasible_mask` all `false` (81 bins).
- `weights.npz`: `steer_dir=[0,0,1]` (+z); dataset `reference_axis=[1,0,0]` (+x).
- SOFA `SourcePosition`: drivers at x=0.16 and x=0.0 (separation **d≈0.16 m along x**); y,z ~equal
  → opposed front/back along x. Confirms RC2 (steered broadside to the driver axis).
- `HDF5.h5`: `driver_order=[driver_0, driver_1]` but only `driver_0` group stored (user confirmed a
  reproducible GUI **save error** → partial write). Secondary HDF5 bug.
- Magnitude scale: `|H_full| ≈ 5e-3 Pa` (unit cone velocity at r=3 m).

### RC1 confirmed numerically
- Un-normalized WNG **ceiling** (best possible, delay-sum corner): **−42.8 dB @20 Hz … −19.6 dB**
  near top — entirely below the −6/−20 dB floor.
- Synthetic monopole fixtures (`monopole_field`, r≈1, `|H|≈1`) give `‖c‖²≈M` ⇒ WNG ≈ `10log10 M`,
  which **masked** the bug in CI.

### Reconstruction harness (`tests/_fixtures/reconstruct_run2.py`)
- Rebuilds `H_full[2,81,2562]` (Pa) from `run2/driver_{0,1}/H_full/*.frd` aligned to `HDF5.h5`
  directions. Validated vs the stored `driver_0/H_full`: **median rel err 4e-7, max 9e-7** (lossless).

### Failure reproduced on real data (steer = +x front), then fixed
| metric | before (un-norm WNG) | after (5a normalized) |
|---|---|---|
| WNG ceiling | −42.8 … −19.6 dB | **+3.0 dB** (=10log10 2, constant) |
| `ls` cardioid DI @100/300 Hz | 0.50 / 1.45 dB | **4.88 / 4.24 dB** |
| `ls` rear-null @100/300 Hz | −0.9 / −5.6 dB | **−25.9 / −14.6 dB** |
| `constant_di` DI @100/600 Hz | 0.40 / 2.94 dB | 2.03 / 3.63 dB |
| `auto` chose | `delay_sum` | **`ls`** |
| feasible bins (all engines) | 0/81 | **81/81** |
→ A real cardioid now forms in the low-mid band (DI≈4.9 dB, deep rear null), degrading toward
600 Hz+ as physics dictates (kd→π). `auto` now correctly selects `ls`.

### 5a implementation
- `regularize.py`: normalized `white_noise_gain_db` (÷ `‖c‖²/M`), `max_white_noise_gain_db`=`10log10 M`,
  inline closure in `solve_loading_for_wng` now calls the canonical function; `‖c‖²=0 → -inf` guard.
- New gate `tests/test_beamform_wng_scale.py` (6 tests): scale-invariance of feasibility/DI/WNG,
  ceiling=`10log10 M`, faint-data-now-feasible. **PASS** (6/6).
- Existing beamform suite (54 tests): **pass unchanged** (no threshold edits needed).
- Full CI-safe suite (`-m 'not local_only and not bempp'`): **482 passed, 14 deselected** (4:47).
- black + ruff clean on changed files.

### Decisions / notes
- The achievable cardioid here is a **low-frequency** band (best ~100–300 Hz for d=0.16 m); it
  degrades above ~600 Hz (kd→π). This is the honest band-limited behavior the user signed off on.
- `feasible_mask` (a WNG/robustness flag) is now all-true on this data; the *pattern* band limit is
  read from the DI / rear-null / target-error curves, not from feasibility.

### Git
- 5a committed on `fix/chunk5a-wng-normalization`, merged `--no-ff` to `main`, tagged **v1.4.1**
  (per user). uv.lock + other pre-existing untracked files left alone.

## Session 1 (cont.) — 5b: steer-to-front-axis (RC2) + engine guidance (RC3)

- `gui/filter_designer_view.py`: `_steer_dir()` now builds the steer in the dataset reference frame
  (`core.sphere.reference_frame`), `θ=0`→front; `load()` sets `_front_axis` from
  `ds.attrs["reference_axis"]`, resets steer to (0,0), shows `Front (0°) axis: …`; added a live
  delay-and-sum guidance note (`_update_engine_note`, text-based for headless testability).
- Tests (`tests/test_gui_smoke.py`, 3 new): steer default follows `reference_axis` (+z default, +x
  when set; θ=0→front, θ=90⟂front, unit-norm); delay-sum note logic; end-to-end real-data cardioid
  (front-steered LS rear null < −12 dB in-band; skips if `HDF5/run2` absent). **Test artifact found+
  fixed:** `isVisible()` is False for an unshown offscreen widget → assert on `text()` instead, and
  the note clears its text when hidden.
- Full `tests/test_gui_smoke.py`: **39 passed**. black + ruff clean. +z-front fixtures unaffected
  (reframed steer reduces to the old formula there).
- Findings: `docs/Chunk5b_Findings.md`. Next: 5c (`docs/Chunk5c_Kickoff_Prompt.md`) — HDF5 atomic
  write; **ask user for the exact GUI save-error text first**.
