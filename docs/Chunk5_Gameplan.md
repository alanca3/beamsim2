# Chunk 5 — Filter-Designer Repair (cardioid / constant-DI / auto) + HDF5 save fix

> Master gameplan for Chunk 5 of the First-Run Bug-Fix campaign (`docs/Bug_Fix_Proposal.md`
> item #8). Companion docs: `Chunk5_Log.md` (append-only session log + measured numbers),
> `Chunk5_Status.md` (current state / next step), and per-sub-chunk `Chunk5{a,b,c}_Findings.md`
> + `Chunk5{b,c}_Kickoff_Prompt.md`. Written 2026-06-23.

## Why this work

Chunk 3 rebuilt the filter designer, but on the user's real **two-opposed-driver** loudspeaker
(front + back faces of one box, drivers separated **d ≈ 0.16 m along +x** = the dataset
`reference_axis`) it still **cannot make a cardioid**, nor sane **constant-DI** / **Auto-Design**
results (`First_Run_Bugs.txt` #8). The `HDF5/run2/` export shows it exactly: `engine="delay_sum"`,
DI ≈ 0.38 dB at LF → ~5.7 dB at HF (near-omni, not a flat 4.77 dB cardioid), `feasible_mask`
**all false**, front ≈ rear at LF. Saving the dataset to HDF5 in the GUI also **errors**, leaving a
corrupt `HDF5.h5` (lists 2 drivers, stores 1).

## Diagnosis (confirmed; numbers reproduced on the real reconstructed data)

**RC1 — keystone: the WNG floor is computed on absolute Pa², not the dimensionless array gain, so
it is unreachable for real BEM data and every adaptive engine collapses to omni.**
`regularize.white_noise_gain_db` returned `10·log10(|cᴴw|²/‖w‖²)`; at the matched-field corner this
is `10·log10(‖c‖²) = 10·log10(Σ|H_m|²)` — **absolute pressure²**. `docs/Phase 2 - Filter Solver.md`
§5.3 says the *intended* metric is M-referenced (delay-sum → WNG = M = `10·log10 M`). Real BEM
`|H_full| ≈ 5e-3 Pa` (unit cone velocity at r=3 m) ⇒ measured WNG **ceiling −42.8 … −19.6 dB**,
entirely below the −6/−20 dB floor. Consequences (all reproduced on the reconstructed run2 data):
- `feasible_mask` all `False`.
- `solve_loading_for_wng` / `solve_maxdir_loading_for_wng` / the LS λ-grid see "floor ≥ ceiling"
  and clamp to **maximum** diagonal loading → MVDR / max-dir / constant-DI / LS all degrade to the
  matched-field omni corner; `auto` then picks `delay_sum` (best-effort among collapsed candidates)
  — matching run2's `design.json` exactly.
- CI never caught it: every beamform test uses `monopole_field` (`|H| ≈ 1`, `‖c‖² ≈ M`), where the
  un-normalized metric already ≈ `10·log10 M` — the bug only appears when `|H| ≠ O(1)`.

**RC2 — steering is never tied to the loudspeaker front; the +z default steered broadside to the
±x driver axis.** `weights.npz steer_dir=[0,0,1]`; SOFA `SourcePosition` shows drivers at x=0.16 /
x=0.0 (separation along x); `reference_axis=+x`. The GUI `_steer_dir()` builds θ-from-+z/φ (default
+z) with no link to `reference_axis`. A first-order (endfire) cardioid must steer *along* the driver
axis (±x); steering broadside (+z) cannot form one.

**RC3 — `delay_sum` cannot synthesize a cardioid.** It is the omni / max-robustness corner; run2 used
it (directly or via `auto`'s best-effort fallback). The GUI lets a "Cardioid" pattern pair with
`delay_sum` and gives no steer-axis or engine guidance.

**Secondary — HDF5 save error corrupts the dataset.** `io/hdf5_store.write_dataset` opens
`h5py.File(path,"w")` (truncate), writes `driver_order`=all ids, then writes each driver group in a
loop. If the 2nd driver's write raises (likely an un-h5py-serializable `attr`), `driver_0` is already
on disk → the exact corrupt partial file observed. Chunk-1's `read_dataset` guard detects this on
*read*; the *write* path must not produce it.

## Decisions (locked with the user)
- Fix **RC1 + RC2 + RC3 + the HDF5 save bug**.
- Verify by **reconstructing the real 2-driver H** from `run2/driver_{0,1}/H_full/*.frd`
  (`tests/_fixtures/reconstruct_run2.py`) **and** a synthetic 2-opposed-driver CI fixture.
- Cardioid success bar = **band-limited + honest flagging** (real cardioid across the physically
  achievable sub-band; out-of-band degrades gracefully and is flagged).
- Package as **Chunk 5** with gameplan + log + status docs; **each session ends by writing the next
  session's Kickoff Prompt**.

## Sub-chunks & gates

### 5a — RC1: normalize the WNG metric (keystone)  [this session]
- `regularize.py`: `white_noise_gain_db` → `10·log10(|cᴴw|²/(‖w‖²·‖c‖²/M))`;
  `max_white_noise_gain_db` → `10·log10(M)`; the inline closure in `solve_loading_for_wng` calls the
  canonical function. Guard `‖c‖²=0 → -inf`. (Per-frequency constant offset ⇒ MVDR/max-dir
  monotonicity and constant-DI τ unimodality preserved; no solver-internal logic change.)
- New CI gate `tests/test_beamform_wng_scale.py`: design metrics invariant to a global `H` scale;
  ceiling = `10·log10(M)`; faint (1e-3) data now feasible + a real cardioid.
- Update WNG-referenced thresholds in the existing beamform tests as needed.
- **Gate:** scale test green; reconstructed-real-data cardioid holds in-band (DI ≈ 4.9 dB, rear null
  ≤ −20 dB at LF); `auto` picks a shaping engine; full CI-safe suite green.

### 5b — RC2 + RC3: steer-to-front-axis + engine guidance (GUI)
- `gui/filter_designer_view.py`: default `steer_dir` to `ds.attrs["reference_axis"]` on load; express
  the steer controls relative to the front frame (reuse `core.sphere.reference_frame`); default shape
  patterns to LS/Auto and warn when a first-order shape is paired with `delay_sum`; surface
  achievable-band / infeasible info legibly.
- **Gate:** GUI smoke tests assert steer default follows `reference_axis`, the engine default/warn
  logic, and an in-band cardioid renders on the reconstructed dataset.

### 5c — HDF5 atomic write + attr hardening (data integrity)
- `io/hdf5_store.py`: write to a temp path + atomic rename (a failed write never corrupts/leaves a
  partial file); harden `_write_attrs` to coerce/surface bad attr types with a clear per-driver/key
  error. Get the exact GUI error text from the user to fix the underlying cause.
- **Gate:** a write that fails mid-stream leaves the original file intact; a 2-driver dataset
  round-trips; CHANGELOG + tag.

## Cardinal rule
No per-driver re-zero / min-phase anywhere; steering stays in H's inter-driver phase.
`tests/test_phase_origin.py` (V-5) stays green throughout.

## Honest scope
A 2-opposed-driver array is a first-order gradient source: the cardioid is physically band-limited
(LF gradient roll-off + HF spatial aliasing above kd≈π, ~1 kHz for d=0.16 m). The fix makes the
achievable band a real cardioid and flags the rest honestly; it does **not** make `delay_sum` a
cardioid or extend the band beyond physics (that needs more drivers / different spacing).
