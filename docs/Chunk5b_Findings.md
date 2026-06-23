# Chunk 5b — Findings: steer-to-front-axis (RC2) + engine guidance (RC3)

> What 5b changed and why. Written 2026-06-23 (Session 1, after 5a). GUI-only; no core/solver
> change. See `Chunk5_Gameplan.md` for the diagnosis.

## RC2 — steering is now measured from the loudspeaker front axis
The GUI built the steer from world `θ`-from-`+z` / `φ`, both defaulting to 0 → **+z**, with no link
to the dataset. The user's box front is `reference_axis = +x` and the opposed drivers lie along x,
so the default aimed the beam **broadside** to the pair — no cardioid could form (run2
`weights.npz steer_dir=[0,0,1]`).

Fix (`gui/filter_designer_view.py`):
- `_steer_dir()` now builds the steer in the dataset's reference frame (`core.sphere.reference_frame`
  → `front, right, up`): `steer = cos θ·front + sin θ·(cos φ·right + sin φ·up)`. **`θ = 0` aims
  straight out the front**, so the default cardioid points where the speaker faces.
- `load(ds)` reads `ds.attrs["reference_axis"]` into `self._front_axis`, resets the steer to (0,0),
  and shows the front axis (`Front (0°) axis: +x`). Labels renamed: "Steer θ (off front axis)" /
  "Steer φ (around front)".
- Cardinal-rule safe: this is display/intent only — it never moves geometry or the phase origin.
- Back-compatible for +z-front datasets: `front=+z`, so `(θ,φ)=(0,0)` still gives +z and `θ=90,φ=0`
  still gives +x (`right`), matching the old formula on the existing +z fixtures.

## RC3 — delay-and-sum guidance
`delay_sum` is the omni / max-robustness corner and cannot shape a cardioid; the GUI let a shape
pattern pair with it silently (run2 ended on `delay_sum`). LS is already the default engine, so a
fresh user gets a capable engine. Added a live red note under the engine combo
(`_update_engine_note`, wired to pattern + engine changes): when `engine == delay_sum` and the
pattern is anything but Omni, it reads *"Delay-and-sum only steers — it cannot shape a cardioid or
hold directivity. Use Least-squares or Auto-Design."* The note's text is set/cleared (not just
shown/hidden) so its state is queryable in headless tests (`isVisible()` is False for an unshown
offscreen widget).

## Verified
- `tests/test_gui_smoke.py` (3 new): steer default follows `reference_axis` (+z default, +x when set;
  `θ=0`→front, `θ=90`⟂front, unit-norm); the delay-sum note logic; and an **end-to-end real-data**
  check (skips if `HDF5/run2` absent) — front-steered LS "Cardioid" on the reconstructed 2-driver H
  has a rear null < −12 dB in-band and the bin is feasible.
- Full `test_gui_smoke.py` green; the +z-front fixtures are unaffected by the reframed steering.

## Not changed (deliberately)
- LS stays the default engine (Chunk-3c decision: one fast solve, not the 4-solve auto ladder).
- The Directivity plot already shows DI / −6 dB beamwidth / WNG vs frequency with infeasible markers,
  which is the honest band indicator; no new band readout was added.
- Manual GUI interactive pass still recommended before declaring the milestone done (Chunk-4 lesson:
  VTK/GL interaction is invisible to headless tests) — though 5b touches only Qt widgets, not the GL
  viewer, so the headless coverage is strong here.
