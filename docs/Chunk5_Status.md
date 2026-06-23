# Chunk 5 — Status

_Last updated: 2026-06-23 (Session 1)_

## Current state — **Chunk 5 COMPLETE (v1.4.3)**

- **5c — HDF5 atomic write + attr hardening: DONE + COMMITTED (v1.4.3).**
  - `io/hdf5_store.py`: `_write_attrs` skips `None`, JSON-encodes tuples, raises with driver id +
    key + type on unserializable value; `write_dataset` uses temp-file + `os.replace` (atomic).
  - `tests/test_hdf5_atomic.py` — **14/14 pass** (CI-safe, no marker).
  - Full CI-safe suite — **499 passed, 14 deselected**.
  - Root cause: `box_volume_m3=None` (free-air driver) reached h5py raw → object-dtype error.
  - `Chunk5c_Findings.md` written; CHANGELOG + log updated.

- **5b — steer-to-front-axis (RC2) + engine guidance (RC3): DONE + COMMITTED (v1.4.2).**
  - `gui/filter_designer_view.py`: steer measured from `reference_axis` (θ=0→front); delay-sum note.
  - `tests/test_gui_smoke.py` (3 new) + full GUI smoke **39 passed**; black/ruff clean.
  - `docs/Chunk5b_Findings.md` written.

- **5a — WNG normalization (RC1 keystone): DONE + COMMITTED (v1.4.1).**
  - `regularize.py` normalized (3 sites); `‖c‖²=0` guarded.
  - New gate `tests/test_beamform_wng_scale.py` — **6/6 pass**.
  - Full CI-safe suite — **482 passed, 14 deselected**.
  - Real-data verification (reconstructed run2): cardioid holds in-band (DI≈4.9 dB, rear null
    −26 dB @100 Hz), `auto` now picks `ls`, 81/81 feasible.

## Next
- **Chunk 5 closed.** No deferred items within Chunk 5.
- Carry-forward: full-band cardioid on a 2-opposed-driver array is physically band-limited
  (LF roll-off + HF aliasing above kd≈π ≈ 1 kHz for d=0.16 m); extending it requires more
  drivers or a different spacing — out of Chunk 5 scope, record for the next campaign.

## Verification assets
- `tests/_fixtures/reconstruct_run2.py` — rebuilds the real 2-driver H from `HDF5/run2/*.frd`
  (local-only; `HDF5/` is git-ignored). Validated lossless vs the stored `driver_0/H_full`.
- `tests/test_beamform_wng_scale.py` — CI-safe scale-invariance gate (synthetic).

## Gates (definition of done)
- [x] 5a: scale test green; real cardioid in-band; auto picks shaping engine; CI-safe suite green.
- [x] 5b: GUI steer defaults to `reference_axis`; cardioid + LS/Auto one-click sane; smoke tests green.
- [x] 5c: failed write leaves original file intact; 2-driver round-trip; save error resolved.
- [x] Cardinal rule (V-5) green throughout. CHANGELOG + milestone tag (v1.4.3).

## Recommended human check (Chunk-4 lesson)
Before tagging Chunk 5 complete, do a manual GUI pass: open the Filter Designer on a 2-opposed-driver
dataset, confirm the steer defaults to the front axis, "Cardioid" + LS/Auto gives a forward cardioid
balloon + rear null, and the plots populate. (5b touches only Qt widgets, not the GL viewer, so
headless coverage is strong — but a visual confirm is still worth it.)
