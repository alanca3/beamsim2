"""Qt-free progress model for multi-driver BEM solve monitoring.

``ProgressModel`` is the single source of truth for solve state: which
frequency steps are queued / running / done / flagged, current estimated RAM
load, steps-done/total, and a rolling ETA. It is entirely Qt-free; the GUI
bridges it to the event loop by subscribing a bound signal:

    progress.subscribe(worker.progressChanged.emit)

The orchestrator (``pipeline/run.py``) calls the mutators; the GUI poll is
driven by the subscriber callback fired on every state change.

Build-order item 10 (Stage E progress display, §6 Gameplan).
"""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np


class StepState(enum.Enum):
    """Per-frequency-step status used by the §6 status grid."""

    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FLAGGED = "flagged"  # non-converged; solver retried, interpolation pending


@dataclass
class ProgressSnapshot:
    """Immutable value object emitted to subscribers on every state change.

    Parameters
    ----------
    grid : np.ndarray
        ``[M, F]`` object array of ``StepState`` values.  M = number of
        drivers, F = number of frequency steps.
    steps_done : int
    steps_total : int
        ``M × F``
    current_ram_bytes : float
        Sum of ``est_ram`` for all RUNNING steps.  Approximation: uses the
        scheduler's pre-computed estimate, not a live psutil sample.
    eta_seconds : float or None
        Rolling estimate of time remaining.  None until at least one step
        has completed.
    current_driver : str or None
        ``driver_id`` of the driver currently being solved.
    message : str
        Human-readable status string for the run-monitor label.
    """

    grid: np.ndarray  # [M, F] object dtype of StepState
    steps_done: int
    steps_total: int  # M × F
    current_ram_bytes: float
    eta_seconds: Optional[float]
    current_driver: Optional[str]
    message: str


ProgressCallback = Callable[[ProgressSnapshot], None]


class ProgressModel:
    """Observable solve-progress model.  Call mutators from the orchestrator;
    subscribers receive a fresh ``ProgressSnapshot`` on every change.

    Parameters
    ----------
    n_drivers : int
    n_freq : int
    driver_ids : list[str]
    """

    def __init__(self, n_drivers: int, n_freq: int, driver_ids: List[str]) -> None:
        if len(driver_ids) != n_drivers:
            raise ValueError("len(driver_ids) must equal n_drivers")
        self._n_drivers = n_drivers
        self._n_freq = n_freq
        self._driver_ids = list(driver_ids)
        # Grid: rows = drivers (0..M-1), cols = freq steps (0..F-1)
        self._grid: np.ndarray = np.full((n_drivers, n_freq), StepState.QUEUED, dtype=object)
        self._running_ram: dict[tuple[int, int], float] = {}  # (m, step) → est_ram
        self._step_elapsed: dict[tuple[int, int], float] = {}  # (m, step) → wall-clock s
        self._steps_done: int = 0
        self._start_time: Optional[float] = None
        self._current_driver: Optional[str] = None
        self._current_m: int = 0  # row offset for the active driver
        self._subscribers: List[ProgressCallback] = []

    # ------------------------------------------------------------------
    # Subscription
    # ------------------------------------------------------------------

    def subscribe(self, callback: ProgressCallback) -> None:
        """Register a callback called with a ``ProgressSnapshot`` on every change."""
        self._subscribers.append(callback)

    # ------------------------------------------------------------------
    # Orchestrator-facing mutators
    # ------------------------------------------------------------------

    def driver_started(self, driver_id: str, m: int, n_total: int) -> None:
        """Called before the orchestrator starts per-driver solve loop for driver m."""
        self._current_driver = driver_id
        self._current_m = m
        if self._start_time is None:
            self._start_time = time.monotonic()
        self._emit(f"Solving driver {m + 1}/{n_total}: {driver_id}")

    def step_running(self, m: int, step: int, est_ram_bytes: float) -> None:
        """Called when a frequency step is launched (scheduler 'step_running' event)."""
        if 0 <= m < self._n_drivers and 0 <= step < self._n_freq:
            self._grid[m, step] = StepState.RUNNING
            self._running_ram[(m, step)] = est_ram_bytes
        self._emit()

    def step_done(self, m: int, step: int, converged: bool, elapsed_seconds: float = 0.0) -> None:
        """Called when a frequency step finishes ('step_done' + 'step_converged' events).

        For the scheduler's two-event pattern (step_done fires on process exit,
        step_converged fires with the convergence result), callers should call this
        once with the final ``converged`` value; or call step_done twice and let the
        second call with the real converged value win.

        Parameters
        ----------
        elapsed_seconds : float
            Actual wall-clock time for this step in seconds (from scheduler timing).
            0.0 if not available.
        """
        if 0 <= m < self._n_drivers and 0 <= step < self._n_freq:
            self._grid[m, step] = StepState.DONE if converged else StepState.FLAGGED
            self._running_ram.pop((m, step), None)
            self._steps_done += 1
            if elapsed_seconds > 0.0:
                self._step_elapsed[(m, step)] = elapsed_seconds
        self._emit()

    @property
    def step_elapsed_seconds(self) -> dict:
        """Wall-clock time per completed step: {(driver_idx, step_idx): seconds}.

        Populated only when the NumCalc scheduler emits timing data (i.e., when
        run_simulation() is called with a ProgressModel). Values are 0.0 for steps
        where timing was unavailable.
        """
        return dict(self._step_elapsed)

    def driver_finished(self, driver_id: str, m: int, flagged: np.ndarray) -> None:
        """Called after the orchestrator's extract() for driver m completes.

        ``flagged`` is ``[F]`` bool where True = non-converged (the meaning from
        ``~field.convergence_flags``).  Reconciles the grid with the final
        authoritative convergence data from the solver output.
        """
        if 0 <= m < self._n_drivers:
            for step, is_flagged in enumerate(flagged):
                if step < self._n_freq:
                    if self._grid[m, step] in (StepState.DONE, StepState.FLAGGED):
                        # Already recorded; reconcile with authoritative flag.
                        self._grid[m, step] = StepState.FLAGGED if is_flagged else StepState.DONE
        self._emit(f"Driver {driver_id} complete")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ram_in_use(self) -> float:
        """Sum of est_ram_bytes for all currently RUNNING steps."""
        return sum(self._running_ram.values())

    def _eta(self) -> Optional[float]:
        """Rolling ETA in seconds; None until at least one step done."""
        if self._steps_done == 0 or self._start_time is None:
            return None
        elapsed = time.monotonic() - self._start_time
        remaining = (self._n_drivers * self._n_freq) - self._steps_done
        return (elapsed / self._steps_done) * remaining

    def _emit(self, message: str = "") -> None:
        snap = ProgressSnapshot(
            grid=self._grid.copy(),
            steps_done=self._steps_done,
            steps_total=self._n_drivers * self._n_freq,
            current_ram_bytes=self._ram_in_use(),
            eta_seconds=self._eta(),
            current_driver=self._current_driver,
            message=message,
        )
        for cb in self._subscribers:
            cb(snap)
