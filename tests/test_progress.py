"""CI tests for pipeline/progress.py — Qt-free progress model.

Covers ProgressModel mutators, snapshot contents, ETA, RAM sum,
grid state transitions, and subscriber callback pattern.
No NumCalc binary, no Qt required.
"""

from __future__ import annotations

import numpy as np
import pytest

from beamsim2.pipeline.progress import ProgressModel, ProgressSnapshot, StepState


def _model(M: int = 2, F: int = 3) -> ProgressModel:
    ids = [f"drv_{m}" for m in range(M)]
    return ProgressModel(n_drivers=M, n_freq=F, driver_ids=ids)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_construction_grid_all_queued() -> None:
    pm = _model(2, 3)
    snapshots: list[ProgressSnapshot] = []
    pm.subscribe(snapshots.append)
    # No events yet → no snapshots
    assert len(snapshots) == 0


def test_bad_driver_ids_raises() -> None:
    with pytest.raises(ValueError):
        ProgressModel(n_drivers=2, n_freq=3, driver_ids=["only_one"])


# ---------------------------------------------------------------------------
# driver_started
# ---------------------------------------------------------------------------


def test_driver_started_sets_current_driver() -> None:
    pm = _model()
    snaps: list[ProgressSnapshot] = []
    pm.subscribe(snaps.append)
    pm.driver_started("drv_0", 0, 2)
    assert len(snaps) == 1
    assert snaps[-1].current_driver == "drv_0"
    assert snaps[-1].steps_done == 0
    assert snaps[-1].steps_total == 6  # 2 × 3


# ---------------------------------------------------------------------------
# step_running / step_done transitions
# ---------------------------------------------------------------------------


def test_step_running_updates_grid() -> None:
    pm = _model()
    snaps: list[ProgressSnapshot] = []
    pm.subscribe(snaps.append)
    pm.step_running(0, 0, 1e9)
    assert snaps[-1].grid[0, 0] == StepState.RUNNING


def test_step_running_adds_ram() -> None:
    pm = _model()
    snaps: list[ProgressSnapshot] = []
    pm.subscribe(snaps.append)
    pm.step_running(0, 0, 1.5e9)
    pm.step_running(0, 1, 2.0e9)
    assert snaps[-1].current_ram_bytes == pytest.approx(3.5e9)


def test_step_done_converged_increments_done() -> None:
    pm = _model()
    snaps: list[ProgressSnapshot] = []
    pm.subscribe(snaps.append)
    pm.step_running(0, 0, 1e9)
    pm.step_done(0, 0, converged=True)
    assert snaps[-1].grid[0, 0] == StepState.DONE
    assert snaps[-1].steps_done == 1
    assert snaps[-1].current_ram_bytes == pytest.approx(0.0)


def test_step_done_not_converged_marks_flagged() -> None:
    pm = _model()
    snaps: list[ProgressSnapshot] = []
    pm.subscribe(snaps.append)
    pm.step_running(0, 2, 1e9)
    pm.step_done(0, 2, converged=False)
    assert snaps[-1].grid[0, 2] == StepState.FLAGGED
    assert snaps[-1].steps_done == 1


def test_steps_done_accumulates_across_drivers() -> None:
    pm = _model(M=2, F=2)
    snaps: list[ProgressSnapshot] = []
    pm.subscribe(snaps.append)
    pm.step_done(0, 0, True)
    pm.step_done(0, 1, True)
    pm.step_done(1, 0, True)
    assert snaps[-1].steps_done == 3


def test_grid_shape() -> None:
    M, F = 3, 5
    pm = ProgressModel(n_drivers=M, n_freq=F, driver_ids=[f"d{i}" for i in range(M)])
    snaps: list[ProgressSnapshot] = []
    pm.subscribe(snaps.append)
    pm.step_running(0, 0, 0.0)
    grid = snaps[-1].grid
    assert grid.shape == (M, F)


# ---------------------------------------------------------------------------
# driver_finished reconciles flags
# ---------------------------------------------------------------------------


def test_driver_finished_reconciles_flagged() -> None:
    pm = _model()
    snaps: list[ProgressSnapshot] = []
    pm.subscribe(snaps.append)
    # Mark step 2 converged tentatively
    pm.step_running(0, 2, 1e9)
    pm.step_done(0, 2, converged=True)
    # Authoritative flags from extract(): step 2 was actually non-converged
    flagged = np.array([False, False, True])
    pm.driver_finished("drv_0", 0, flagged)
    assert snaps[-1].grid[0, 2] == StepState.FLAGGED


def test_driver_finished_no_change_when_all_converged() -> None:
    pm = _model()
    snaps: list[ProgressSnapshot] = []
    pm.subscribe(snaps.append)
    pm.step_running(0, 0, 0.0)
    pm.step_done(0, 0, converged=True)
    pm.driver_finished("drv_0", 0, np.array([False, False, False]))
    assert snaps[-1].grid[0, 0] == StepState.DONE


# ---------------------------------------------------------------------------
# ETA
# ---------------------------------------------------------------------------


def test_eta_none_before_any_step_done() -> None:
    pm = _model()
    snaps: list[ProgressSnapshot] = []
    pm.subscribe(snaps.append)
    pm.driver_started("drv_0", 0, 2)
    assert snaps[-1].eta_seconds is None


def test_eta_non_none_after_step_done() -> None:
    pm = _model()
    snaps: list[ProgressSnapshot] = []
    pm.subscribe(snaps.append)
    pm.driver_started("drv_0", 0, 2)
    pm.step_done(0, 0, converged=True)
    assert snaps[-1].eta_seconds is not None
    assert snaps[-1].eta_seconds >= 0.0


# ---------------------------------------------------------------------------
# Multiple subscribers
# ---------------------------------------------------------------------------


def test_multiple_subscribers_all_called() -> None:
    pm = _model()
    calls_a: list[ProgressSnapshot] = []
    calls_b: list[ProgressSnapshot] = []
    pm.subscribe(calls_a.append)
    pm.subscribe(calls_b.append)
    pm.driver_started("drv_0", 0, 2)
    assert len(calls_a) == 1
    assert len(calls_b) == 1


# ---------------------------------------------------------------------------
# Out-of-bounds steps are silently ignored (defensive)
# ---------------------------------------------------------------------------


def test_out_of_bounds_step_ignored() -> None:
    pm = _model(M=2, F=3)
    snaps: list[ProgressSnapshot] = []
    pm.subscribe(snaps.append)
    # Should not raise even if step or m is out of range
    pm.step_running(99, 99, 0.0)
    pm.step_done(99, 99, True)
    assert snaps[-1].steps_done == 0  # no valid step was done


# ---------------------------------------------------------------------------
# ProgressSnapshot is an immutable value copy (grid copy)
# ---------------------------------------------------------------------------


def test_snapshot_grid_is_copy() -> None:
    """Mutating the model after emitting must not change the previous snapshot."""
    pm = _model()
    snaps: list[ProgressSnapshot] = []
    pm.subscribe(snaps.append)
    pm.step_running(0, 0, 0.0)
    first_snap = snaps[-1]
    pm.step_done(0, 0, True)
    # The first snapshot's grid must still show RUNNING
    assert first_snap.grid[0, 0] == StepState.RUNNING
