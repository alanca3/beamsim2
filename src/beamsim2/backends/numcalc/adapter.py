"""NumCalc BEMBackend adapter — implements estimate/prepare/solve/extract against
the NumCalc C++ executable (Mesh2HRTF, primary backend per DR-01).

This is the MINIMAL adapter for build-order item 3. It supports a single driver,
one vibrating group with a uniform scalar velocity, conventional BEM (method 0),
and a single NC.inp covering all frequencies in one NumCalc invocation. The full
RAM-aware, highest-frequency-first, checkpointed scheduler is deferred to item 6.

Binary-path resolution: explicit constructor argument → BEAMSIM2_NUMCALC_BIN env var →
FileNotFoundError. The path recorded in docs/SETUP_NOTES.md is never hardcoded.

The frequency bridge between prepare() and extract() uses a meta.json sidecar written
into work_dir by prepare(). This keeps core/types.py untouched and survives a future
resume path (item 6).

References
----------
NumCalc invocation: manage_numcalc.py, lines 362–378 (subprocess.Popen pattern).
VERIFIED: Kreuzer et al., Engineering Analysis with Boundary Elements 161:157-178, 2024.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from typing import Optional

import numpy as np

from beamsim2.backends.base import BEMBackend
from beamsim2.backends.numcalc.config import resolve_numcalc_binary
from beamsim2.backends.numcalc.ncinp_writer import write_mesh_files, write_nc_inp
from beamsim2.backends.numcalc.reader import read_convergence, read_eval_pressure
from beamsim2.core.types import (
    BoundaryConditions,
    ComplexField,
    FrequencyGrid,
    Mesh,
    ObservationPoints,
    RawSolveResult,
    ResourcePlan,
    SolverConfig,
    SolveSpec,
)

_META_FILENAME = "beamsim2_meta.json"


class NumCalcBackend(BEMBackend):
    """BEM adapter for the NumCalc C++ solver (Mesh2HRTF, DR-01 primary backend).

    Parameters
    ----------
    binary_path : str or None
        Explicit path to the NumCalc executable. If None, falls back to the
        BEAMSIM2_NUMCALC_BIN environment variable.

    Raises
    ------
    FileNotFoundError
        On construction if the binary cannot be resolved.
    """

    def __init__(self, binary_path: str | None = None) -> None:
        self._binary = resolve_numcalc_binary(binary_path)

    # ── estimate ─────────────────────────────────────────────────────────────

    def estimate(
        self,
        mesh: Mesh,
        bc: BoundaryConditions,
        frequencies: FrequencyGrid,
        config: SolverConfig,
    ) -> ResourcePlan:
        """Return per-frequency RAM/time estimates via NumCalc -estimate_ram.

        NumCalc writes a Memory.txt file into the working directory when invoked
        with -estimate_ram. This minimal implementation runs that flag on a temporary
        copy of the NC.inp and reads Memory.txt. If the file is absent or the binary
        fails, falls back to NaN-filled estimates rather than raising (the scheduler
        must handle unknown estimates gracefully).

        Parameters
        ----------
        mesh : Mesh
            Boundary surface mesh.
        bc : BoundaryConditions
            Boundary conditions.
        frequencies : FrequencyGrid
            Frequencies to estimate, shape [F].
        config : SolverConfig
            Solver parameters.

        Returns
        -------
        ResourcePlan
            ram_bytes_per_step [F] float64, time_seconds_per_step [F] float64.
            Fields are NaN where NumCalc did not produce an estimate.
        """
        n_freq = len(frequencies.frequencies)
        obs_dummy = _dummy_observation_points()

        with tempfile.TemporaryDirectory(prefix="beamsim2_estimate_") as tmpdir:
            counts = write_mesh_files(tmpdir, mesh, obs_dummy)
            write_nc_inp(tmpdir, mesh, bc, frequencies, config, counts)
            try:
                subprocess.run(
                    [self._binary, "-estimate_ram"],
                    cwd=tmpdir,
                    capture_output=True,
                    timeout=60,
                )
                ram = _parse_memory_txt(tmpdir, n_freq)
            except Exception:
                ram = np.full(n_freq, np.nan, dtype=np.float64)

        return ResourcePlan(
            ram_bytes_per_step=ram,
            time_seconds_per_step=np.full(n_freq, np.nan, dtype=np.float64),
        )

    # ── prepare ──────────────────────────────────────────────────────────────

    def prepare(
        self,
        mesh: Mesh,
        bc: BoundaryConditions,
        frequencies: FrequencyGrid,
        observation_points: ObservationPoints,
        config: SolverConfig,
    ) -> SolveSpec:
        """Write NC.inp, mesh files, and eval-grid files into a scratch directory.

        # DR-02: ObservationPoints added here because NumCalc bakes the evaluation
        # grid into NC.inp at prepare() time; cannot be deferred to extract().

        Also writes a beamsim2_meta.json sidecar so extract() can reconstruct
        frequencies and point count without touching core/types.py.

        Parameters
        ----------
        mesh : Mesh
            Boundary surface mesh.
        bc : BoundaryConditions
            Boundary conditions (one vibrating group, scalar velocity, minimal writer).
        frequencies : FrequencyGrid
            Frequencies to solve, shape [F].
        observation_points : ObservationPoints
            Evaluation-sphere directions and radius.
        config : SolverConfig
            Solver parameters.

        Returns
        -------
        SolveSpec
            work_dir, nc_inp_paths, frequency_grid.
        """
        work_dir = tempfile.mkdtemp(prefix="beamsim2_numcalc_")

        counts = write_mesh_files(work_dir, mesh, observation_points)
        nc_inp_path = write_nc_inp(work_dir, mesh, bc, frequencies, config, counts)

        # Sidecar: bridge information that extract() needs but isn't in RawSolveResult.
        meta = {
            "frequencies": frequencies.frequencies.tolist(),
            "n_obs": len(observation_points.unit_vectors),
            "eval_node_base": counts.eval_node_base,
        }
        with open(os.path.join(work_dir, _META_FILENAME), "w") as fh:
            json.dump(meta, fh, indent=2)

        return SolveSpec(
            work_dir=work_dir,
            nc_inp_paths=[nc_inp_path],
            frequency_grid=frequencies,
        )

    # ── solve ─────────────────────────────────────────────────────────────────

    def solve(
        self,
        spec: SolveSpec,
        scheduler: Optional[object] = None,
    ) -> RawSolveResult:
        """Run NumCalc on the prepared NC.inp and return raw result locations.

        Invokes ``NumCalc -istart 1 -iend F`` in spec.work_dir (all frequencies in
        one process). The full RAM-aware per-step scheduler is item 6; the scheduler
        argument is accepted here for interface conformance but ignored.

        NumCalc writes be.out/be.N/ directories and an NC.out log relative to its
        working directory. VERIFIED: manage_numcalc.py lines 362–378.

        Parameters
        ----------
        spec : SolveSpec
            Prepared solve spec from prepare().
        scheduler : object, optional
            Future scheduler (item 6). Ignored in this minimal implementation.

        Returns
        -------
        RawSolveResult
            work_dir, completed_steps, convergence_flags [F] bool.

        Raises
        ------
        RuntimeError
            If the NumCalc process exits with a non-zero return code.
        """
        n_freq = len(spec.frequency_grid.frequencies)
        work_dir = spec.work_dir

        result = subprocess.run(
            [self._binary, "-istart", "1", "-iend", str(n_freq)],
            cwd=work_dir,
            capture_output=False,  # let NumCalc print to terminal for visibility
            timeout=3600,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"NumCalc exited with code {result.returncode}. "
                f"Check {os.path.join(work_dir, 'NC.out')} for details."
            )

        convergence_flags = read_convergence(work_dir, n_freq)  # [F] bool

        completed_steps = {
            i
            for i in range(n_freq)
            if os.path.isdir(os.path.join(work_dir, "be.out", f"be.{i + 1}"))
        }

        return RawSolveResult(
            work_dir=work_dir,
            completed_steps=completed_steps,
            convergence_flags=convergence_flags,
        )

    # ── extract ───────────────────────────────────────────────────────────────

    def extract(
        self,
        raw: RawSolveResult,
        observation_points: ObservationPoints,
    ) -> ComplexField:
        """Parse NumCalc output and return raw complex pressure as ComplexField.

        Reads beamsim2_meta.json to recover frequencies and n_obs, then reads all
        pEvalGrid files. Pressure is passed through **raw** — no re-zeroing, no
        minimum-phase processing (cardinal rule §3.4).

        Parameters
        ----------
        raw : RawSolveResult
            Raw output from solve().
        observation_points : ObservationPoints
            The same grid passed to prepare(). Used to validate the point count.

        Returns
        -------
        ComplexField
            pressure [F, N] complex128, convergence_flags [F] bool, frequencies [F] float64.

        Raises
        ------
        FileNotFoundError
            If beamsim2_meta.json or any pEvalGrid file is missing.
        ValueError
            If observation_points count doesn't match the prepared grid.
        """
        meta_path = os.path.join(raw.work_dir, _META_FILENAME)
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(
                f"beamsim2_meta.json not found in {raw.work_dir}. "
                "Was prepare() called on the same work_dir?"
            )

        with open(meta_path, "r") as fh:
            meta = json.load(fh)

        frequencies = np.array(meta["frequencies"], dtype=np.float64)  # [F] float64
        n_obs_meta = int(meta["n_obs"])
        n_obs = len(observation_points.unit_vectors)
        n_freq = len(frequencies)

        if n_obs != n_obs_meta:
            raise ValueError(
                f"observation_points has {n_obs} points but prepare() was called "
                f"with {n_obs_meta}. Pass the same ObservationPoints to extract()."
            )

        # Parse per-frequency pEvalGrid files.
        pressure = read_eval_pressure(raw.work_dir, n_freq, n_obs)  # [F, N] complex128

        # Pressure passes through raw — cardinal rule §3.4.
        return ComplexField(
            pressure=pressure,
            convergence_flags=raw.convergence_flags,
            frequencies=frequencies,
        )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _dummy_observation_points() -> ObservationPoints:
    """Return a minimal 4-point observation grid for use in estimate().

    estimate() needs to write NC.inp but doesn't care about the eval grid.
    Using 4 points keeps the file small.
    """
    from beamsim2.core.sphere import lebedev

    return lebedev(n_points=6, radius=1.0)


def _parse_memory_txt(work_dir: str, n_freq: int) -> np.ndarray:
    """Parse NumCalc Memory.txt into per-frequency RAM estimates in bytes.

    Memory.txt is written by NumCalc -estimate_ram. Its format is not fully
    documented; this parser reads lines of the form 'Step N: X MB' and maps
    them to frequency indices. Returns NaN for any step not found.

    Parameters
    ----------
    work_dir : str
        Directory where Memory.txt is written.
    n_freq : int
        Number of frequency steps.

    Returns
    -------
    np.ndarray, shape [n_freq], float64
        Estimated RAM per step in bytes. NaN where not parseable.
    """
    import re

    ram = np.full(n_freq, np.nan, dtype=np.float64)
    mem_path = os.path.join(work_dir, "Memory.txt")
    if not os.path.isfile(mem_path):
        return ram

    step_re = re.compile(r"[Ss]tep\s+(\d+).*?(\d+\.?\d*)\s*MB", re.IGNORECASE)
    with open(mem_path) as fh:
        for line in fh:
            m = step_re.search(line)
            if m:
                step = int(m.group(1)) - 1  # 1-based → 0-based
                mb = float(m.group(2))
                if 0 <= step < n_freq:
                    ram[step] = mb * 1024 * 1024  # MB → bytes

    return ram
