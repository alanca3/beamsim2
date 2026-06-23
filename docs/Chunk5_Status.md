# Chunk 5 — Status

_Last updated: 2026-06-23 (Session 1)_

## Current state
- **5a — WNG normalization (RC1 keystone): DONE (verified).**
  - `regularize.py` normalized (3 sites); `‖c‖²=0` guarded.
  - New gate `tests/test_beamform_wng_scale.py` — **6/6 pass**.
  - Existing beamform suite (54 tests) — **pass unchanged** (no thresholds needed updating;
    monopole fixtures have `‖c‖²≈M` so the offset is ~0 there).
  - Full CI-safe suite — **482 passed, 14 deselected** (`-m 'not local_only and not bempp'`).
  - Real-data verification (reconstructed run2): cardioid holds in-band (DI≈4.9 dB, rear null
    −26 dB @100 Hz), `auto` now picks `ls`, 81/81 feasible.
  - `Chunk5a_Findings.md` + CHANGELOG entry written. **Not committed/tagged** (awaiting user).
    Suggested tag when committed: `v1.4.1`.

## Next
- **5b — steer-to-front-axis + engine guidance (GUI).** See `Chunk5b_Kickoff_Prompt.md`.
- **5c — HDF5 atomic write + attr hardening.** Needs the exact GUI save-error text from the user.

## Verification assets
- `tests/_fixtures/reconstruct_run2.py` — rebuilds the real 2-driver H from `HDF5/run2/*.frd`
  (local-only; `HDF5/` is git-ignored). Validated lossless vs the stored `driver_0/H_full`.
- `tests/test_beamform_wng_scale.py` — CI-safe scale-invariance gate (synthetic).

## Gates (definition of done)
- [x] 5a: scale test green; real cardioid in-band; auto picks shaping engine; CI-safe suite green.
- [ ] 5b: GUI steer defaults to `reference_axis`; cardioid + Auto/LS one-click sane; smoke tests green.
- [ ] 5c: failed write leaves original file intact; 2-driver round-trip; save error resolved.
- [ ] Cardinal rule (V-5) green throughout. CHANGELOG + milestone tag.
