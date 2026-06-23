# Chunk 5c — HDF5 atomic write + attr hardening: Findings

_Written 2026-06-23 (Session 2). Closes Chunk 5._

## Root cause (confirmed)

User-reported error: **`Object dtype dtype('O') has no native HDF5 equivalent`**, exact message.

Trace:
1. `TerminalModel.to_attrs()` (`src/beamsim2/driver/terminal.py:118`) emits
   `"box_volume_m3": self.box_volume` where `self.box_volume: float | None`. The default
   (`box_volume=None`) is used for every free-air / infinite-baffle driver (the back-face driver
   in the user's two-driver box was free-air).
2. `_write_attrs` (`io/hdf5_store.py`) only JSON-encoded `dict`/`list`; everything else fell
   through to `h5obj.attrs[key] = val`. For `val = None`, h5py calls `np.asarray(None)` →
   dtype `object` → the crash.
3. `write_dataset` opens `h5py.File(path, "w")` which **truncates the file immediately** before
   any write. The failure on the *second* driver left `driver_order=[driver_0, driver_1]` but
   only `driver_0`'s group on disk — the exact corrupt partial file the user observed.

Note: `box_volume_m3` is **not listed in DATA_CONTRACT §3.5** (it's an extra `to_attrs()`
field); `None` (free-air, no box) has no sensible disk representation and is correctly absent.

## Why the fix belongs in `_write_attrs`, not in `to_attrs()`

The serialization layer is the right boundary: fixing it there makes *any* future `None`-valued
attr safe, without requiring every emitter to defensively filter its output. `to_attrs()`
legitimately conveys "no box = None" in memory — that's its correct semantic.

## What changed in `io/hdf5_store.py`

### `_write_attrs(h5obj, attrs, context="")`

New `context` parameter (e.g. `" for driver 'drv_1'"`, `" (root attrs)"`) included in errors.

Encoding table (DATA_CONTRACT §3.5 extended):

| Value type | Before | After |
|---|---|---|
| `None` | passed to h5py → **crash** | **skipped** (absent attr = unset) |
| `dict` / `list` | JSON-encoded | JSON-encoded (unchanged) |
| `tuple` | passed raw (h5py stored as 1D array, may silently change type) | JSON-encoded as list |
| `np.ndarray` | passed raw | passed raw; on `TypeError`/`ValueError` re-raises with key + context + dtype |
| scalar / str | passed raw | passed raw; on `TypeError`/`ValueError` re-raises with key + context + type name |

### `write_dataset(path, ds)` — atomic write

1. `validate_unique_driver_ids(...)` still runs first (no disk touch on duplicate-id failure).
2. `tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")` creates the temp in
   the **same directory** as the target, guaranteeing `os.replace` is always a same-filesystem
   atomic rename (no cross-device move).
3. All writes go to `tmp_path`. On success: `os.replace(tmp_path, path)`.
4. On *any* exception: `tmp_path.unlink(missing_ok=True)` (best-effort, swallows `OSError`)
   then re-raise. The target file is **never touched** until the rename.

### `os.replace` file-permission caveat

`mkstemp` creates the temp file at mode 0600 (owner-read/write only). After `os.replace`, the
final file inherits this mode. For a single-user desktop application this is acceptable. If
world-readable output is ever needed, add `os.chmod(tmp_path, 0o644)` before `os.replace`.

## What was NOT changed

- `schema_version` — no on-disk format change; absent attrs are exactly absent on read-back.
- `_read_attrs` — unchanged; it JSON-decodes only strings that parse to dict/list (so skipping
  None on write and reading back gives `None` via `.get("box_volume_m3", None)` naturally).
- `TerminalModel.to_attrs()` — intentionally left returning `box_volume=None`.
- GUI `_save_hdf5` (`gui/results_view.py`) — already has `try/except + QMessageBox.critical`;
  the atomic write makes its missing partial-file cleanup moot (original is always intact after
  a failed write).
- Cardinal rule (no per-driver re-zero / min-phase) — untouched. `test_phase_origin.py` (V-5)
  stays green throughout.

## Test coverage (`tests/test_hdf5_atomic.py`, 14 tests, CI-safe)

1. **`TestAsymmetricNone`** (4 tests) — exact-bug reproduction:
   - Write succeeds with `box_volume_m3=float` on driver_0 and `box_volume_m3=None` on driver_1.
   - Float attr on driver_0 survives round-trip.
   - None attr is absent on read-back from driver_1.
   - Both drivers present in the written file (no corrupt partial).

2. **`TestAtomicWrite`** (4 tests) — atomicity:
   - A write with an object-dtype ndarray attr raises `TypeError`.
   - Error message names the offending driver id and attr key.
   - Pre-existing file is byte-identical after a failed write.
   - No `.tmp` file remains on disk after failure.

3. **`TestRoundTrip`** (6 tests) — lossless 2-driver round-trip:
   - Driver count and insertion order preserved.
   - `H_full` arrays equal element-for-element.
   - Frequency grid exact.
   - `driver_order` raw attr is a JSON list.
   - Scalar driver attrs (`box_volume_m3`, `reference_voltage_V`, `terminal_response_model`).
   - Dict driver attrs (`ts_params`) round-trip as dicts.

Full CI-safe suite: **499 passed, 14 deselected** (5b baseline was also 485+14; the increase
comes from the 14 new tests minus the 14-test 5a set which merged into the count).
Regression tests green: `test_hdf5_roundtrip`, `test_solver_correctness`, `test_gui_smoke`,
`test_phase_origin`.

## Carry-forward

- **Full-band cardioid on a 2-opposed-driver array is physically band-limited** — LF gradient
  roll-off and HF spatial aliasing above kd ≈ π (~1 kHz for d = 0.16 m). The Chunk-5 fixes make
  the achievable band a real cardioid. Extending the band requires more drivers or a different
  spacing — out of Chunk 5 scope; record for the next campaign phase.
- **Chunk 5 is closed.** All three sub-chunks (5a WNG-norm, 5b steer-to-front, 5c atomic-write)
  done and merged; tag **v1.4.3**.
