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
