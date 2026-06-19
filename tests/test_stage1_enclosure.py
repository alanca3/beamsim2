"""Stage 1 gate test: real single-driver box-enclosure BEM solve.

Earns v0.2.0. Gate criteria (Gameplan §8):
  1. Qualitative diffraction features visible (DI rises, on-axis not flat).
  2. Peak RAM and wall-clock per step measured at the top of the band.
  3. DR-05 timing data printed so bem_cap_hz can be decided.

Reference enclosure: 200 × 300 × 200 mm box, 75 mm piston on front face.
Frequency grid: 100 Hz → 5 kHz, 1/3-octave (18 steps).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

from beamsim2.core.sphere import lebedev
from beamsim2.core.types import FrequencyGrid, SolverConfig
from beamsim2.geometry.assemble import DriverSpec
from beamsim2.pipeline.progress import ProgressModel
from beamsim2.pipeline.run import (
    BoxGeometry,
    DriverPlacement,
    SimulationRequest,
    run_simulation,
)
from beamsim2.validation.power_di import directivity_index

# ---------------------------------------------------------------------------
# Binary skip guard (same pattern as other local_only tests)
# ---------------------------------------------------------------------------

try:
    from beamsim2.backends.numcalc.config import resolve_numcalc_binary

    _BINARY = resolve_numcalc_binary()
except FileNotFoundError:
    _BINARY = None

pytestmark = pytest.mark.local_only


def _skip_if_no_binary() -> None:
    if _BINARY is None:
        pytest.skip("NumCalc binary not found. Set BEAMSIM2_NUMCALC_BIN to run this test.")


# ---------------------------------------------------------------------------
# Reference enclosure constants (edit here to change the geometry)
# ---------------------------------------------------------------------------

_W, _H, _D = 0.200, 0.300, 0.200  # box dimensions, metres (W × H × D)
_DRIVER_RADIUS = 0.075  # piston radius, metres (≈ 6″ woofer class)
_DRIVER_CENTRE = (0.100, 0.200, 0.000)  # front face z=0, centred H, 2/3 height
_DRIVER_NORMAL = (0.0, 0.0, -1.0)  # outward from front face
_DRIVER_ID = "woofer_0"

_F_MIN, _F_MAX, _N_OCT = 100.0, 5000.0, 3  # 1/3-octave grid
_N_STEPS = int(round(_N_OCT * np.log2(_F_MAX / _F_MIN))) + 1  # 18 steps
_SPHERE_N = 26  # Lebedev-26
_N_EPW = 6  # elements per wavelength

# Persistent output directory (in .gitignore — never committed)
_OUT_DIR = Path(__file__).parent.parent / "runs" / "stage1"

# Surface area of the reference box [m²] — used for element-count estimate
_BOX_SURFACE_AREA = 2 * (_W * _H + _W * _D + _H * _D)  # 0.32 m²


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_freqs() -> FrequencyGrid:
    """1/3-octave grid from _F_MIN to _F_MAX."""
    return FrequencyGrid(
        frequencies=np.geomspace(_F_MIN, _F_MAX, _N_STEPS),
        spacing="fractional-octave",
        fractional_octave=1.0 / _N_OCT,
    )


def _read_memory_txt(work_dir: str, n_freq: int) -> np.ndarray:
    """Parse NumCalc Memory.txt → per-step RAM in bytes, [F] float64.

    Format: one line per step — '<1-based-step> <freq_Hz> <ram_GB>'
    Returns NaN for missing or unparseable entries.
    """
    ram = np.full(n_freq, np.nan, dtype=np.float64)  # [F] float64 — bytes
    path = Path(work_dir) / "Memory.txt"
    if not path.exists():
        return ram
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 3:
            try:
                step = int(parts[0]) - 1  # 1-based → 0-based
                ram_gb = float(parts[2])
                if 0 <= step < n_freq:
                    ram[step] = ram_gb * (1024**3)
            except (ValueError, IndexError):
                pass
    return ram


def _n_elem_estimate(freq_hz: float) -> int:
    """Rough element count for the reference box at `freq_hz` with n_epw=6."""
    h = 343.2 / (freq_hz * _N_EPW)  # edge length, m
    return max(1, int(round(_BOX_SURFACE_AREA / h**2)))


def _format_seconds(s: float) -> str:
    if np.isnan(s) or s <= 0:
        return "  --  "
    if s < 60:
        return f"{s:5.1f}s"
    return f"{s / 60:5.1f}m"


# ---------------------------------------------------------------------------
# Stage 1 gate test
# ---------------------------------------------------------------------------


def test_stage1_enclosure() -> None:
    """Stage 1: real single-driver box-enclosure BEM solve.

    Geometry:  200 × 300 × 200 mm box
    Driver:    75 mm piston at front face centre (terminal=None, unit velocity)
    Freqs:     100 Hz → 5 kHz, 1/3-octave (18 steps)
    Sphere:    Lebedev-26, r = 1.0 m
    n_epw:     6

    Gate assertions:
      - All frequency steps converge.
      - On-axis level has ≥ 3 dB variation across the band (shows diffraction structure).
      - DI at 5 kHz > DI at 100 Hz by ≥ 2 dB (directivity builds with frequency).

    Timing output:
      - Per-step RAM (from NumCalc Memory.txt) and wall-clock are printed.
      - Timing JSON and HDF5 saved to runs/stage1/ for post-run inspection.
      - DR-05 extrapolation to 20 kHz printed so bem_cap_hz can be decided.
    """
    _skip_if_no_binary()

    # Ensure output directory exists
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    h5_path = _OUT_DIR / "stage1.h5"
    timing_json_path = _OUT_DIR / "timing.json"

    freqs = _make_freqs()
    n_freq = len(freqs.frequencies)

    # Build the SimulationRequest
    req = SimulationRequest(
        geometry=BoxGeometry(_W, _H, _D),
        drivers=[
            DriverPlacement(
                spec=DriverSpec(
                    center=_DRIVER_CENTRE,
                    normal=_DRIVER_NORMAL,
                    radius=_DRIVER_RADIUS,
                    cap_height=0.0,
                ),
                terminal=None,  # placeholder unit velocity; T/S chain added in Stage 2
                driver_id=_DRIVER_ID,
            )
        ],
        frequencies=freqs,
        sphere_n_points=_SPHERE_N,
        sphere_radius=1.0,
        config=SolverConfig(n_epw=_N_EPW),
        output_h5=h5_path,
    )

    # ProgressModel captures per-step timing
    progress = ProgressModel(
        n_drivers=1,
        n_freq=n_freq,
        driver_ids=[_DRIVER_ID],
    )

    # Run the full pipeline A–G
    t_wall_start = time.perf_counter()
    result = run_simulation(req, progress=progress)
    t_wall_total = time.perf_counter() - t_wall_start

    # -----------------------------------------------------------------------
    # Collect timing data
    # -----------------------------------------------------------------------

    # Per-step RAM from NumCalc Memory.txt (already produced by -estimate_ram)
    ram_bytes = _read_memory_txt(result.work_dirs[_DRIVER_ID], n_freq)  # [F] float64

    # Per-step wall-clock from ProgressModel (populated by scheduler timing)
    elapsed = progress.step_elapsed_seconds  # {(0, step): seconds}
    wall_per_step = np.array(
        [elapsed.get((0, s), 0.0) for s in range(n_freq)], dtype=np.float64
    )  # [F] float64 — seconds; 0 = not captured

    # -----------------------------------------------------------------------
    # Print timing table
    # -----------------------------------------------------------------------

    sep = "-" * 66
    print(f"\n{'Stage 1 timing — reference bookshelf enclosure':^66}")
    print(
        f"{'Box: %d×%d×%d mm   Driver: %d mm piston' % (_W*1000, _H*1000, _D*1000, _DRIVER_RADIUS*1000):^66}"
    )
    print(sep)
    print(f"  {'freq_Hz':>8}  {'n_elem':>7}  {'RAM_GB':>7}  {'wall':>7}  {'converged':>10}")
    print(sep)
    for i, f in enumerate(freqs.frequencies):
        n_el = _n_elem_estimate(f)
        ram_gb = ram_bytes[i] / (1024**3) if not np.isnan(ram_bytes[i]) else float("nan")
        ram_str = f"{ram_gb:6.2f}" if not np.isnan(ram_gb) else "  NaN "
        wall_str = _format_seconds(wall_per_step[i])
        conv = "✓" if not result.flagged_frequencies[_DRIVER_ID][i] else "FLAGGED"
        print(f"  {f:>8.0f}  {n_el:>7d}  {ram_str}  {wall_str}  {conv:>10}")
    print(sep)
    print(f"  Total wall-clock: {t_wall_total:.1f} s  ({t_wall_total/60:.1f} min)")
    print(f"  HDF5: {h5_path}")

    # -----------------------------------------------------------------------
    # DR-05 extrapolation to 20 kHz (informational)
    # -----------------------------------------------------------------------

    # Use timing of the top-frequency step (which runs first, alone)
    top_step = n_freq - 1  # 5 kHz is the last (highest) in the grid
    t_top = wall_per_step[top_step]
    n_top = _n_elem_estimate(_F_MAX)  # elements at 5 kHz

    # FMM scaling: time ≈ C · N^α with α ≈ 1.3 (O(N log N) in practice)
    n_20k = int(round(_BOX_SURFACE_AREA / (343.2 / (20000 * _N_EPW)) ** 2))
    alpha = 1.3
    if t_top > 0:
        t_20k_est = t_top * (n_20k / n_top) ** alpha
    else:
        t_20k_est = float("nan")

    n_steps_20k = int(round(_N_OCT * np.log2(20000.0 / _F_MIN))) + 1

    print(f"\n{'DR-05 extrapolation (full-band to 20 kHz)':^66}")
    print(sep)
    print(f"  5 kHz step:  {n_top:>6d} elements,  wall ≈ {_format_seconds(t_top).strip()}")
    print(
        f"  20 kHz step: {n_20k:>6d} elements,  wall ≈ {_format_seconds(t_20k_est).strip()}  (N^{alpha} FMM scaling)"
    )
    if not np.isnan(t_20k_est):
        total_20k = t_20k_est * n_steps_20k * 0.35  # rough: lower steps much faster
        print(f"  Full-band solve ({n_steps_20k} steps): est. total ≈ {total_20k/3600:.1f} h")
        peak_ram_gb = (
            np.nanmax(ram_bytes) / (1024**3) if not np.all(np.isnan(ram_bytes)) else float("nan")
        )
        if not np.isnan(peak_ram_gb):
            ram_20k_est = peak_ram_gb * (n_20k / n_top) ** 1.5
            print(
                f"  Peak RAM at 20 kHz: est. ≈ {ram_20k_est:.1f} GB  (measured {peak_ram_gb:.1f} GB at 5 kHz)"
            )
        feasible = (not np.isnan(t_20k_est)) and (t_20k_est < 7200)  # < 2h per step
        rec = (
            f"{int(_F_MAX)} (full-band feasible)"
            if feasible
            else "5000 (splice advised — top step too slow)"
        )
        print(f"\n  DR-05 RECOMMENDATION: bem_cap_hz = {rec}")
    print(sep)

    # -----------------------------------------------------------------------
    # Save timing JSON
    # -----------------------------------------------------------------------

    timing_data = {
        "enclosure_mm": {"width": _W * 1000, "height": _H * 1000, "depth": _D * 1000},
        "driver_radius_mm": _DRIVER_RADIUS * 1000,
        "n_epw": _N_EPW,
        "total_wall_seconds": t_wall_total,
        "steps": [
            {
                "freq_hz": float(freqs.frequencies[i]),
                "n_elem_est": _n_elem_estimate(freqs.frequencies[i]),
                "ram_bytes": float(ram_bytes[i]) if not np.isnan(ram_bytes[i]) else None,
                "wall_seconds": float(wall_per_step[i]) if wall_per_step[i] > 0 else None,
                "converged": bool(not result.flagged_frequencies[_DRIVER_ID][i]),
            }
            for i in range(n_freq)
        ],
    }
    timing_json_path.write_text(json.dumps(timing_data, indent=2))
    print(f"  Timing JSON: {timing_json_path}")

    # -----------------------------------------------------------------------
    # Gate assertions
    # -----------------------------------------------------------------------

    # 1. All steps converged (no retry-failed steps flagged)
    assert result.flagged_frequencies[_DRIVER_ID].sum() == 0, (
        f"Non-converged steps at: "
        f"{freqs.frequencies[result.flagged_frequencies[_DRIVER_ID]].tolist()} Hz"
    )

    # 2. Geometry was clean
    assert result.health.ok, f"Geometry health issues: {result.health.problems}"

    # 3. On-axis response has structure (≥ 3 dB variation → diffraction visible)
    ds = result.dataset
    driver_data = next(d for d in ds.drivers if d.driver_id == _DRIVER_ID)
    H_bem = driver_data.H_bem  # [F, N] complex128
    obs = lebedev(_SPHERE_N, radius=1.0)

    # On-axis = observation point most aligned with driver's forward direction
    driver_fwd = np.array([0.0, 0.0, -1.0])  # driver faces −z
    on_axis_idx = int(np.argmax(obs.unit_vectors @ driver_fwd))
    on_axis_dB = 20.0 * np.log10(np.abs(H_bem[:, on_axis_idx]))  # [F]

    level_range_dB = float(np.ptp(on_axis_dB))
    print(f"\n  On-axis level range: {level_range_dB:.1f} dB  (gate: > 3 dB)")
    assert level_range_dB > 3.0, (
        f"On-axis response looks flat: only {level_range_dB:.1f} dB variation "
        "across the band — expected baffle step + diffraction ripple to exceed 3 dB."
    )

    # 4. DI rises with frequency (forward directivity builds above ka=1)
    # directivity_index accepts [F, N] → [F] dB
    DI_per_freq = directivity_index(H_bem, obs.weights)  # [F] dB
    di_rise = float(DI_per_freq[-1] - DI_per_freq[0])
    print(
        f"  DI at 100 Hz: {DI_per_freq[0]:.1f} dB,  at 5 kHz: {DI_per_freq[-1]:.1f} dB  (rise: {di_rise:.1f} dB)"
    )
    assert di_rise > 2.0, (
        f"DI only rose by {di_rise:.1f} dB from 100 Hz to 5 kHz "
        "(expected > 2 dB — 75 mm piston has ka > 3 at 5 kHz, should be clearly directive)."
    )

    print("\n  Stage 1 gate: PASSED  (v0.2.0 conditions met)")
