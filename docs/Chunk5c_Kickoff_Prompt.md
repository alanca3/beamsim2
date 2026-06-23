# Chunk 5c — Kickoff Prompt: HDF5 atomic write + attr hardening (data integrity)

> Paste this to start the session. Read first: `docs/Chunk5_Gameplan.md` (secondary bug),
> `docs/Chunk5_Status.md`, `docs/Chunk5a_Findings.md`, `docs/Chunk5b_Findings.md`, the memory index.
> **FIRST ASK THE USER for the exact GUI error text shown when saving HDF5 fails** — it pinpoints
> the underlying cause and saves a guessing pass. (User deferred 5c at the end of Session 1.)

## Goal
Saving the dataset to HDF5 in the GUI **errors** and leaves a **corrupt partial file**: the user's
`HDF5/run2/HDF5.h5` lists `driver_order=[driver_0, driver_1]` but stores only `driver_0`. Fix it so
(a) a save never corrupts/loses data, and (b) the underlying per-driver write error is resolved or
surfaced clearly.

## Context (confirmed Session 1)
- `io/hdf5_store.write_dataset` opens `h5py.File(path, "w")` (truncate), writes the `driver_order`
  attr = all ids, then writes each driver group in a loop. If the **2nd** driver's write raises,
  `driver_0` is already on disk → exactly the corrupt partial file observed. The Chunk-1
  `read_dataset` guard *detects* this on read but the *write* path can still produce it.
- The SOFA export (written minutes earlier from the same drivers) has both drivers, so the failure is
  specific to the HDF5 write path / one driver's data or attrs.
- Likely root cause: an un-h5py-serializable value in `d.attrs` (e.g. `None`, a tuple, a
  `FacePlacement`/dataclass) reaching `_write_attrs`, which passes non-dict/list/ndarray values
  straight to `h5py.attrs[key] = val`. (Get the error text to confirm.)

## Scope (do)
1. **Atomic write.** Write to a temp path in the same directory, then `os.replace()` (atomic rename)
   over the target — so a failed/raising write never truncates or corrupts the existing file. Clean
   up the temp file on failure.
2. **Harden `_write_attrs`.** Coerce/skip un-serializable attr values with a clear, actionable error
   that names the driver id and the offending key (and type), instead of a raw h5py TypeError. Decide
   per the error text: coerce (e.g. JSON-encode dataclasses/tuples, drop `None`) vs. raise early with
   a good message. Keep the on-disk contract unchanged for valid attrs.
3. **Fix the underlying cause** once the error text identifies it (e.g. ensure driver `attrs` only
   carry contract-§3.5 serializable fields, or JSON-encode the offender).

## Verify
- New `tests/test_hdf5_atomic.py` (CI-safe): (a) a write that raises mid-stream (inject a bad driver
  attr) leaves the **pre-existing** file byte-unchanged (atomicity); (b) the bad-attr case yields a
  clear error naming driver+key (or is coerced and round-trips); (c) a 2-driver dataset round-trips
  losslessly. Reuse the synthetic-dataset pattern from `tests/test_solver_correctness.py` /
  `test_gui_smoke._synthetic_dataset`.
- Re-run the user's GUI save flow (or the in-memory 2-driver dataset) to confirm the real save now
  succeeds (or fails safely with a clear message and no corruption).

## Constraints
- `schema_version` unchanged (no on-disk format change). Cardinal rule untouched (no H/phase change).
- One coherent commit; update `CHANGELOG.md`, `Chunk5_Log.md`, `Chunk5_Status.md`. This closes Chunk 5
  — write a short `Chunk5_Findings`/close-out note and (with the user) tag the milestone.

## Close-out
Update the campaign docs + memory. If anything in Chunk 5 remains (e.g. the full-band cardioid needs
an array change, or a deferred GUI band-readout), record it as the next pointer.
