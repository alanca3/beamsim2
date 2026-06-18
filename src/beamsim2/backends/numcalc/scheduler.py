"""RAM-aware, highest-frequency-first, resumable job scheduler for NumCalc.

Implements Stage E (Solve / job management / the budget) per DR-01/DR-02 and
build-order item 6. Keeps one NC.inp covering all F frequencies and launches
one NumCalc process per frequency step:

    NumCalc -istart S -iend S

Concurrent processes are packed against a configurable RAM budget
(default 42 GB = 48 GB - 6 GB OS headroom), sorted highest-frequency-first
so the RAM peak occurs when the queue is shortest. Steps already completed
on disk are skipped so a crash or pause never loses work (R-08). Non-converged
steps are retried once with a raised ``-niter_max`` argument (R-07).

In acoustics terms: the scheduler is the session manager for a multi-day
balloon-measurement campaign — it books measurement slots highest-frequency
first (where the "studio" is most expensive), skips shots already in the can,
and re-shoots the few that were technically unsatisfactory.

Mitigated risks: R-04 (RAM pressure), R-07 (HF non-convergence), R-08 (resume).

References
----------
Scheduling pattern: Mesh2HRTF manage_numcalc.py (VERIFIED, commit e45d0436a).
NC.inp invocation: NC_Main.cpp sprintf("NC%d-%d.out", istart_, iend) (VERIFIED).
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from beamsim2.core.types import RawSolveResult, ResourcePlan

# Default budget: 48 GB total − 6 GB OS headroom = 42 GB usable.
_DEFAULT_RAM_BUDGET_BYTES: float = 42 * 1024**3


@dataclass
class SchedulerConfig:
    """Tuning parameters for the NumCalc job scheduler.

    All defaults are calibrated for the M4 Max / 48 GB development machine
    (DR-01). Overriding these is an advanced operation; sensible defaults apply
    to any typical BEM run within the project's scope.

    Parameters
    ----------
    ram_budget_bytes : float
        Maximum RAM to allocate across all concurrent processes, in bytes.
        Default 42 GB (48 GB total − 6 GB OS headroom).
    ram_safety_factor : float
        Multiply each step's estimated RAM by this before comparing against
        the available budget. Default 1.1 (10 % safety margin, matching the
        Mesh2HRTF reference manager).
    max_concurrency : int
        Hard cap on simultaneous NumCalc processes, regardless of RAM budget.
        Default 12 matches the M4 Max performance-core count.
    poll_seconds : float
        Interval between "check for finished processes and launch new ones" cycles.
    retry_max_iterations : int
        ``-niter_max`` value used for the R-07 retry pass. NumCalc's built-in
        default is 250; retrying at 1000 gives a 4x wider convergence window.
    """

    ram_budget_bytes: float = _DEFAULT_RAM_BUDGET_BYTES
    ram_safety_factor: float = 1.1
    max_concurrency: int = 12
    poll_seconds: float = 2.0
    retry_max_iterations: int = 1000


def order_steps(
    frequencies: np.ndarray,
    ram_bytes_per_step: np.ndarray,
) -> list[int]:
    """Return 0-based step indices ordered highest-first by estimated RAM.

    Steps are sorted descending by estimated RAM (the proxy for frequency
    and mesh size). When estimates are NaN the sort key falls back to frequency
    value, so highest-frequency steps still come first without breaking anything.

    In acoustics terms: schedule the most "expensive" measurement slot first,
    so the RAM headroom is largest when the queue has the most remaining steps.
    HEURISTIC: matches Mesh2HRTF manage_numcalc.py default ``starting_order="high"``.

    Parameters
    ----------
    frequencies : np.ndarray, shape [F], float64
        Frequency values in Hz — used as the fallback sort key when RAM
        estimates are NaN.
    ram_bytes_per_step : np.ndarray, shape [F], float64
        Estimated RAM per step in bytes. NaN = unknown estimate.

    Returns
    -------
    list[int]
        0-based step indices, highest-estimated-RAM first.
    """
    # When RAM estimate is NaN, fall back to frequency as the sort key
    # (higher frequency ≈ higher RAM). Both are ascending-sort then reversed.
    sort_key = np.where(np.isfinite(ram_bytes_per_step), ram_bytes_per_step, frequencies)

    # stable argsort ascending, then reverse → highest first
    ascending = np.argsort(sort_key, kind="stable")
    return list(reversed(ascending.tolist()))


class NumCalcScheduler:
    """RAM-aware, highest-frequency-first, resumable NumCalc job manager.

    Keeps one NC.inp (covering all F frequencies) and launches one process per
    step ``NumCalc -istart S -iend S``.  Packs concurrent processes against
    the RAM budget, skips completed steps for resume, and retries non-converged
    steps once with a higher ``-niter_max``.

    The actual subprocess launch is isolated in ``_launch_step()`` so unit tests
    can inject a mock launcher and verify ordering / packing / resume logic
    without a real NumCalc binary.

    Parameters
    ----------
    binary : str
        Absolute path to the NumCalc executable.
    config : SchedulerConfig, optional
        Scheduling parameters. Defaults are M4 Max / 48 GB tuned.
    _launcher : callable, optional
        Inject ``(cmd: list[str], work_dir: str, step: int) -> Popen``
        instead of the real subprocess. Used only in unit tests.
    """

    def __init__(
        self,
        binary: str,
        config: Optional[SchedulerConfig] = None,
        _launcher: Optional[Callable] = None,
    ) -> None:
        self._binary = binary
        self._config = config or SchedulerConfig()
        self._launcher = _launcher

    def run(
        self,
        work_dir: str,
        frequencies: np.ndarray,  # [F] float64 — Hz
        resource_plan: Optional[ResourcePlan] = None,
    ) -> RawSolveResult:
        """Solve all F frequency steps, returning a RawSolveResult.

        Steps already completed on disk are skipped automatically. Non-converged
        steps are retried once at a higher iteration cap (R-07). Returns only
        after all processes have exited.

        Parameters
        ----------
        work_dir : str
            NumCalc working directory (NC.inp + mesh files prepared by adapter).
        frequencies : np.ndarray, shape [F], float64
            Frequency values in Hz — used for ordering and result labeling.
        resource_plan : ResourcePlan, optional
            Per-step RAM/time estimates from backend.estimate(). If None or
            all-NaN, RAM gating is disabled and only max_concurrency applies.

        Returns
        -------
        RawSolveResult
            work_dir, completed_steps (set of 0-based indices), convergence_flags [F] bool.
        """
        from beamsim2.backends.numcalc.reader import read_convergence, step_completed

        n_freq = len(frequencies)
        ram_est: np.ndarray = (
            resource_plan.ram_bytes_per_step
            if resource_plan is not None
            else np.full(n_freq, np.nan, dtype=np.float64)
        )  # [F] float64 — bytes or NaN

        # ── Pass 1: normal solve ─────────────────────────────────────────────
        pending: list[int] = [
            i for i in order_steps(frequencies, ram_est) if not step_completed(work_dir, i + 1)
        ]
        self._run_pass(work_dir, pending, ram_est)

        # ── R-07 retry: one more pass at raised iteration cap ────────────────
        conv_flags = read_convergence(work_dir, n_freq)  # [F] bool
        retry_pending: list[int] = [
            i for i in range(n_freq) if not conv_flags[i] and step_completed(work_dir, i + 1)
        ]
        if retry_pending:
            self._run_pass(
                work_dir,
                retry_pending,
                ram_est,
                extra_args=["-niter_max", str(self._config.retry_max_iterations)],
            )
            conv_flags = read_convergence(work_dir, n_freq)

        # ── Collect results ──────────────────────────────────────────────────
        completed_steps = {i for i in range(n_freq) if step_completed(work_dir, i + 1)}
        return RawSolveResult(
            work_dir=work_dir,
            completed_steps=completed_steps,
            convergence_flags=conv_flags,
        )

    def _run_pass(
        self,
        work_dir: str,
        pending: list[int],  # 0-based indices, already in desired launch order
        ram_est: np.ndarray,  # [F] float64
        extra_args: Optional[list[str]] = None,
    ) -> None:
        """Launch and wait for all steps in `pending` with RAM/concurrency gating.

        Parameters
        ----------
        work_dir : str
            NumCalc working directory.
        pending : list[int]
            0-based step indices to run, in desired launch order.
        ram_est : np.ndarray
            Per-step RAM estimates in bytes; NaN disables RAM gating.
        extra_args : list[str], optional
            Extra CLI args for each invocation, e.g. ``["-niter_max", "1000"]``.
        """
        pending = list(pending)
        in_flight: dict[int, subprocess.Popen] = {}  # 0-based idx → Popen

        while pending or in_flight:
            # Reap finished processes.
            done = [idx for idx, proc in in_flight.items() if proc.poll() is not None]
            for idx in done:
                del in_flight[idx]

            # Launch as many new steps as slots and RAM allow.
            while pending and len(in_flight) < self._config.max_concurrency:
                next_idx = pending[0]
                if not self._fits_in_ram(next_idx, in_flight, ram_est):
                    break  # RAM full — wait for in-flight steps to release it.
                pending.pop(0)
                proc = self._launch_step(work_dir, next_idx + 1, extra_args)
                in_flight[next_idx] = proc

            if pending or in_flight:
                time.sleep(self._config.poll_seconds)

    def _fits_in_ram(
        self,
        step_idx: int,
        in_flight: dict[int, subprocess.Popen],
        ram_est: np.ndarray,
    ) -> bool:
        """Return True if launching step_idx stays within the RAM budget.

        If the step's estimate is NaN or zero, RAM gating is skipped (only the
        max_concurrency cap applies). In-flight steps with NaN estimates
        contribute zero to the in-flight RAM total (conservative: unknown
        costs don't block new launches).
        """
        est = ram_est[step_idx]
        if not np.isfinite(est) or est <= 0.0:
            return True  # No estimate — concurrency cap is the only gate.

        in_flight_ram = sum(float(ram_est[idx]) for idx in in_flight if np.isfinite(ram_est[idx]))
        required = est * self._config.ram_safety_factor
        available = self._config.ram_budget_bytes - in_flight_ram
        return required <= available

    def _launch_step(
        self,
        work_dir: str,
        step: int,
        extra_args: Optional[list[str]] = None,
    ) -> subprocess.Popen:
        """Launch ``NumCalc -istart S -iend S`` as a non-blocking subprocess.

        NumCalc writes its own ``NC{step}-{step}.out`` internally (VERIFIED:
        NC_Main.cpp ``sprintf(filename,"NC%d-%d.out",istart_,iend)``). Stdout
        is discarded since all meaningful solver output goes to that file.

        Parameters
        ----------
        work_dir : str
            NumCalc working directory (NC.inp must be present).
        step : int
            1-based step number.
        extra_args : list[str], optional
            Additional CLI arguments, e.g. ``["-niter_max", "1000"]``.

        Returns
        -------
        subprocess.Popen
        """
        cmd = [self._binary, "-istart", str(step), "-iend", str(step)]
        if extra_args:
            cmd.extend(extra_args)

        if self._launcher is not None:
            return self._launcher(cmd, work_dir, step)

        return subprocess.Popen(
            cmd,
            cwd=work_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
