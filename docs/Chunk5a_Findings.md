# Chunk 5a — Findings: WNG normalization (RC1 keystone)

> What the diagnosis and the fix actually showed. Written 2026-06-23 (Session 1).
> See `Chunk5_Gameplan.md` for the full diagnosis and `Chunk5_Log.md` for the raw numbers.

## The fix, in one line
`white_noise_gain_db` measured **absolute** matched-field power `‖c‖²` (Pa²), not the
dimensionless array gain. Normalizing by the average per-element power `‖c‖²/M` (so the
matched-field corner reaches the scale-free ceiling `10·log10(M)`) restores the WNG floor as a
meaningful, reachable robustness knob — which un-collapses every adaptive engine (LS / MVDR /
max-dir / constant-DI / auto) on real BEM data.

## Why it was invisible until real data
The metric only misbehaves when `|H| ≠ O(1)`. Every beamform test builds its array with
`validation.closed_loop.monopole_field`, which returns `exp(jkr)/r ≈ 1` at r≈1 m, giving
`‖c‖² ≈ M`. There the un-normalized and normalized metrics are numerically identical, so the
54-test beamform suite passed before *and after* the fix with **no threshold changes**. Real BEM
`H_full` is ~5e-3 Pa (unit cone velocity at 3 m), where `‖c‖²/M ≈ 2.5e-5` → a −43 dB ceiling.

## Measured (reconstructed real run2 data, steer = +x front, floor −6 dB)
| metric | before | after |
|---|---|---|
| WNG ceiling | −42.8 … −19.6 dB | **+3.0 dB** (=10·log10 2) |
| `ls` cardioid DI @100 / 300 Hz | 0.50 / 1.45 dB | **4.88 / 4.24 dB** |
| `ls` rear-null @100 / 300 Hz | −0.9 / −5.6 dB | **−25.9 / −14.6 dB** |
| `auto` chose | `delay_sum` | **`ls`** |
| feasible bins | 0/81 | 81/81 |

The `auto`→`delay_sum`, all-infeasible, near-omni result **exactly reproduced run2's `design.json`**
before the fix — confirming RC1 is the cause of the user's run, not just a side issue.

## Settled premises
1. **The fix is a pure metric re-scaling, solver-safe.** The normalization is a per-frequency
   constant offset independent of `w`/`eps`/`tau`, so the WNG-vs-loading curve only shifts
   vertically: the MVDR / max-dir bisection monotonicity and the constant-DI τ-search unimodality
   are preserved. No solver-internal logic changed.
2. **DI and target-error were always fine.** `directivity_index` is a power ratio and
   `field_agreement_db` power-normalizes each field — both scale-invariant. Only WNG was broken.
   (The BEM tensor itself is correct; this was never a solver/simulation bug.)
3. **The cardioid here is a low-frequency band.** For d≈0.16 m the first-order endfire cardioid is
   strongest ~100–300 Hz (DI≈4.2–4.9 dB, rear null ≤ −15…−26 dB) and degrades toward 600 Hz+ as
   kd→π. With WNG now meetable, all bins are flagged feasible; the *pattern* band limit is read from
   the DI / rear-null / target-error curves, not from `feasible_mask`. This is the agreed
   band-limited + honest behavior.

## Changed
- `src/beamsim2/beamform/regularize.py` — `white_noise_gain_db` (÷ `‖c‖²/M`, `-inf` guard),
  `max_white_noise_gain_db` → `10·log10(M)`, inline closure in `solve_loading_for_wng` → calls the
  canonical function.
- `tests/test_beamform_wng_scale.py` (new, CI-safe) — scale-invariance + ceiling + faint-feasible.
- `tests/_fixtures/reconstruct_run2.py` (new, local-only) — real-H reconstruction harness.

## Not changed (deliberately)
- No existing test thresholds (they hold; see "Why it was invisible").
- No steering/engine UX (RC2/RC3 → 5b) and no HDF5 writer change (→ 5c).
