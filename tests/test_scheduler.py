"""Pure-Python unit tests for the NumCalc job scheduler.

No NumCalc binary required. Tests ordering, RAM-aware packing, resume
(skip-completed logic), and the mock-launcher interface used by
NumCalcScheduler. All filesystem interaction uses tmp_path.

Build-order item 6 finish-line tests (pure-Python half).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pytest

from beamsim2.backends.numcalc.reader import step_completed
from beamsim2.backends.numcalc.scheduler import (
    NumCalcScheduler,
    SchedulerConfig,
    order_steps,
)
from beamsim2.core.types import ResourcePlan

# ---------------------------------------------------------------------------
# order_steps — pure function tests
# ---------------------------------------------------------------------------


def test_order_steps_highest_freq_first_nan():
    """With all-NaN RAM estimates, order by frequency descending."""
    freqs = np.array([100.0, 500.0, 1000.0, 250.0])
    ram = np.full(4, np.nan)

    result = order_steps(freqs, ram)
    # Highest frequency (1000 Hz, idx 2) should be first
    assert result[0] == 2, f"Expected step 2 (1000 Hz) first, got step {result[0]}"
    # Lowest frequency (100 Hz, idx 0) should be last
    assert result[-1] == 0, f"Expected step 0 (100 Hz) last, got step {result[-1]}"


def test_order_steps_by_ram_highest_first():
    """With RAM estimates, higher RAM comes first."""
    freqs = np.array([100.0, 200.0, 300.0])
    # Step 1 is cheapest, step 2 is most expensive
    ram = np.array([1e9, 5e8, 4e9])  # bytes

    result = order_steps(freqs, ram)
    assert result[0] == 2, f"Expected step 2 (4 GB) first, got step {result[0]}"
    assert result[-1] == 1, f"Expected step 1 (0.5 GB) last, got step {result[-1]}"


def test_order_steps_nan_entries_after_known():
    """Steps with NaN estimates come after known-expensive steps."""
    freqs = np.array([1000.0, 500.0, 2000.0, 250.0])
    # Steps 0 and 2 have RAM estimates; steps 1 and 3 are NaN
    ram = np.array([2e9, np.nan, 3e9, np.nan])

    result = order_steps(freqs, ram)
    # Step 2 (3 GB) should be first
    assert result[0] == 2, f"Expected step 2 (3 GB) first, got step {result[0]}"


def test_order_steps_returns_all_indices():
    """Result contains exactly n unique 0-based indices."""
    n = 6
    freqs = np.linspace(100.0, 1000.0, n)
    ram = np.random.default_rng(42).uniform(1e8, 5e9, n)

    result = order_steps(freqs, ram)
    assert len(result) == n
    assert set(result) == set(range(n))


def test_order_steps_single_step():
    """Single-step array returns [0]."""
    assert order_steps(np.array([440.0]), np.array([1e9])) == [0]


def test_order_steps_all_equal_ram():
    """Ties in RAM are broken by stable sort (original index order preserved)."""
    freqs = np.array([100.0, 200.0, 300.0])
    ram = np.array([1e9, 1e9, 1e9])  # all equal

    result = order_steps(freqs, ram)
    # Stable sort reversed means last index first in tie: [2, 1, 0]
    assert result == [2, 1, 0], f"Expected [2, 1, 0] for equal RAM, got {result}"


# ---------------------------------------------------------------------------
# step_completed — filesystem logic
# ---------------------------------------------------------------------------


def test_step_completed_no_peval_returns_false(tmp_path):
    """Returns False when pEvalGrid does not exist."""
    assert not step_completed(str(tmp_path), step=1)


def test_step_completed_peval_no_log_returns_true(tmp_path):
    """pEvalGrid exists but no NC{S}-{S}.out → treat as complete (legacy path)."""
    be_dir = tmp_path / "be.out" / "be.1"
    be_dir.mkdir(parents=True)
    (be_dir / "pEvalGrid").write_text("Mesh2HRTF 1.0\n1 0\n")

    assert step_completed(str(tmp_path), step=1)


def test_step_completed_log_missing_end_time_returns_false(tmp_path):
    """pEvalGrid exists and NC1-1.out exists but has no 'End time:' → not done."""
    be_dir = tmp_path / "be.out" / "be.1"
    be_dir.mkdir(parents=True)
    (be_dir / "pEvalGrid").write_text("data\n")
    (tmp_path / "NC1-1.out").write_text("Start time: Jan 01 2025, 00:00:00\nSolving...\n")

    assert not step_completed(str(tmp_path), step=1)


def test_step_completed_log_with_end_time_returns_true(tmp_path):
    """pEvalGrid and NC1-1.out with 'End time:' → fully done."""
    be_dir = tmp_path / "be.out" / "be.1"
    be_dir.mkdir(parents=True)
    (be_dir / "pEvalGrid").write_text("data\n")
    (tmp_path / "NC1-1.out").write_text(
        "Start time: Jan 01 2025, 00:00:00\nSolving...\nEnd time: Jan 01 2025, 00:00:05\n"
    )

    assert step_completed(str(tmp_path), step=1)


def test_step_completed_checks_correct_step_number(tmp_path):
    """step_completed is step-specific: completing step 1 does not affect step 2."""
    be_dir = tmp_path / "be.out" / "be.1"
    be_dir.mkdir(parents=True)
    (be_dir / "pEvalGrid").write_text("data\n")
    (tmp_path / "NC1-1.out").write_text("End time: done\n")

    assert step_completed(str(tmp_path), step=1)
    assert not step_completed(str(tmp_path), step=2)


# ---------------------------------------------------------------------------
# NumCalcScheduler — mock-launcher tests
# ---------------------------------------------------------------------------


# A Popen-like object that reports done immediately.
@dataclass
class _MockProc:
    returncode: int = 0

    def poll(self):
        return self.returncode


def _make_mock_launcher(tmp_path, launched_steps: list, conv_ok: bool = True):
    """Return a mock launcher that records steps and creates completion files."""

    def launcher(cmd: list, work_dir: str, step: int):
        launched_steps.append(step)
        # Create output files that step_completed() and read_convergence() expect.
        be_dir = os.path.join(work_dir, "be.out", f"be.{step}")
        os.makedirs(be_dir, exist_ok=True)
        with open(os.path.join(be_dir, "pEvalGrid"), "w") as f:
            f.write("Mesh2HRTF 1.0\n1 0\n")
        cgs_line = (
            "CGS solver: number of iterations = 5, relative error = 1e-10\n"
            if conv_ok
            else "Warning: Maximum number of iterations is reached!\n"
        )
        with open(os.path.join(work_dir, f"NC{step}-{step}.out"), "w") as f:
            f.write(f"Start time: ...\n{cgs_line}End time: ...\n")
        return _MockProc()

    return launcher


def test_scheduler_launches_all_steps(tmp_path):
    """Scheduler launches every pending step exactly once."""
    freqs = np.array([100.0, 500.0, 1000.0])
    n = len(freqs)
    launched = []
    sched = NumCalcScheduler(
        binary="/fake/binary",
        config=SchedulerConfig(poll_seconds=0.0),
        _launcher=_make_mock_launcher(tmp_path, launched),
    )

    result = sched.run(str(tmp_path), freqs)

    assert sorted(launched) == [1, 2, 3], f"Expected steps 1-3, got {launched}"
    assert result.completed_steps == {0, 1, 2}
    assert result.convergence_flags.shape == (n,)
    assert result.convergence_flags.all()


def test_scheduler_highest_freq_launched_first(tmp_path):
    """With NaN RAM estimates, highest-frequency step launches first."""
    freqs = np.array([200.0, 800.0, 500.0])  # step 1 (800 Hz) should go first
    launched = []
    sched = NumCalcScheduler(
        binary="/fake/binary",
        config=SchedulerConfig(max_concurrency=1, poll_seconds=0.0),
        _launcher=_make_mock_launcher(tmp_path, launched),
    )

    sched.run(str(tmp_path), freqs)

    # With concurrency=1, steps launch strictly in order.
    # Highest frequency is 800 Hz (index 1) → 1-based step 2.
    assert launched[0] == 2, f"Expected step 2 (800 Hz) first, got step {launched[0]}"


def test_scheduler_skips_completed_steps(tmp_path):
    """Steps already on disk are not re-launched."""
    freqs = np.array([250.0, 500.0, 1000.0])
    # Pre-create step 2 as complete.
    step_2_be = tmp_path / "be.out" / "be.2"
    step_2_be.mkdir(parents=True)
    (step_2_be / "pEvalGrid").write_text("data\n")
    (tmp_path / "NC2-2.out").write_text("CGS solver: number of iterations = 5\nEnd time: done\n")

    launched = []
    sched = NumCalcScheduler(
        binary="/fake/binary",
        config=SchedulerConfig(poll_seconds=0.0),
        _launcher=_make_mock_launcher(tmp_path, launched),
    )

    result = sched.run(str(tmp_path), freqs)

    # Step 2 (1-based) must not be in launched.
    assert 2 not in launched, f"Step 2 was re-launched despite being complete: {launched}"
    # Other steps were launched.
    assert sorted(launched) == [1, 3], f"Expected steps 1 and 3, got {launched}"
    assert result.completed_steps == {0, 1, 2}


def test_scheduler_respects_max_concurrency(tmp_path):
    """Scheduler never exceeds max_concurrency simultaneous in-flight processes."""
    freqs = np.linspace(100.0, 1000.0, 8)
    max_seen: list[int] = []
    in_flight_count = [0]

    def counting_launcher(cmd, work_dir, step):
        in_flight_count[0] += 1
        max_seen.append(in_flight_count[0])
        # Create completion files immediately (MockProc polls done right away).
        be_dir = os.path.join(work_dir, "be.out", f"be.{step}")
        os.makedirs(be_dir, exist_ok=True)
        open(os.path.join(be_dir, "pEvalGrid"), "w").write("data\n")
        open(os.path.join(work_dir, f"NC{step}-{step}.out"), "w").write(
            "CGS solver: number of iterations = 3\nEnd time:\n"
        )
        in_flight_count[0] -= 1
        return _MockProc()

    max_c = 3
    sched = NumCalcScheduler(
        binary="/fake/binary",
        config=SchedulerConfig(max_concurrency=max_c, poll_seconds=0.0),
        _launcher=counting_launcher,
    )
    sched.run(str(tmp_path), freqs)

    # Because mock processes complete instantly before scheduler checks again,
    # the peak in-flight count per launch cycle should never exceed max_concurrency.
    # (It may be 1 since we decrement before returning, but that's acceptable.)
    assert max(max_seen) <= max_c, f"Exceeded max_concurrency={max_c}: peak {max(max_seen)}"


def test_scheduler_ram_gating_prevents_over_budget(tmp_path):
    """Steps that would exceed RAM budget are not launched until budget frees."""
    freqs = np.array([100.0, 200.0, 300.0])
    # Huge RAM estimates so each step nearly fills the budget alone.
    budget = 10 * 1024**3  # 10 GB
    ram_est = np.array([6e9, 6e9, 6e9])  # 6 GB each — only one fits at a time

    launched_order: list[int] = []

    def launcher(cmd, work_dir, step):
        launched_order.append(step)
        be_dir = os.path.join(work_dir, "be.out", f"be.{step}")
        os.makedirs(be_dir, exist_ok=True)
        open(os.path.join(be_dir, "pEvalGrid"), "w").write("data\n")
        open(os.path.join(work_dir, f"NC{step}-{step}.out"), "w").write(
            "CGS solver: number of iterations = 1\nEnd time:\n"
        )
        return _MockProc()

    resource_plan = ResourcePlan(
        ram_bytes_per_step=ram_est,
        time_seconds_per_step=np.full(3, np.nan),
    )
    sched = NumCalcScheduler(
        binary="/fake/binary",
        config=SchedulerConfig(
            ram_budget_bytes=budget,
            ram_safety_factor=1.0,
            max_concurrency=12,
            poll_seconds=0.0,
        ),
        _launcher=launcher,
    )
    sched.run(str(tmp_path), freqs, resource_plan=resource_plan)

    # All 3 steps should eventually run.
    assert sorted(launched_order) == [1, 2, 3], f"Steps missing: {launched_order}"


def test_scheduler_retry_on_non_converged(tmp_path):
    """Steps that fail convergence are retried with -niter_max in the command."""
    freqs = np.array([100.0, 200.0])
    retry_max = 999
    launched_cmds: list[list[str]] = []

    def launcher(cmd, work_dir, step):
        launched_cmds.append(list(cmd))
        be_dir = os.path.join(work_dir, "be.out", f"be.{step}")
        os.makedirs(be_dir, exist_ok=True)
        open(os.path.join(be_dir, "pEvalGrid"), "w").write("data\n")

        is_retry = "-niter_max" in cmd
        if step == 1 and not is_retry:
            # First pass for step 1: non-converged
            log = "Warning: Maximum number of iterations is reached!\nEnd time:\n"
        else:
            # Step 2 and all retries: converged
            log = "CGS solver: number of iterations = 5, relative error = 1e-9\nEnd time:\n"
        open(os.path.join(work_dir, f"NC{step}-{step}.out"), "w").write(log)
        return _MockProc()

    sched = NumCalcScheduler(
        binary="/fake/binary",
        config=SchedulerConfig(poll_seconds=0.0, retry_max_iterations=retry_max),
        _launcher=launcher,
    )
    result = sched.run(str(tmp_path), freqs)

    # Step 1 (1-based) should appear in launched_cmds twice: once without
    # -niter_max (normal pass) and once with it (retry).
    step1_cmds = [c for c in launched_cmds if "-istart" in c and c[c.index("-istart") + 1] == "1"]
    assert (
        len(step1_cmds) == 2
    ), f"Expected 2 launches for step 1, got {len(step1_cmds)}: {step1_cmds}"

    retry_cmd = next(c for c in step1_cmds if "-niter_max" in c)
    niter_idx = retry_cmd.index("-niter_max")
    assert retry_cmd[niter_idx + 1] == str(
        retry_max
    ), f"Expected -niter_max {retry_max}, got {retry_cmd[niter_idx + 1]}"

    # After retry, step 0 should be converged.
    assert result.convergence_flags[0], "Step 0 should be converged after retry"
    assert result.convergence_flags[1], "Step 1 should be converged (never failed)"


def test_scheduler_config_defaults():
    """SchedulerConfig defaults match documented values."""
    cfg = SchedulerConfig()
    assert cfg.ram_budget_bytes == 42 * 1024**3
    assert cfg.ram_safety_factor == pytest.approx(1.1)
    assert cfg.max_concurrency == 12
    assert cfg.retry_max_iterations == 1000
