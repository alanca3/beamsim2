"""GUI smoke tests: widget construction + dataset load + worker-thread plumbing.

Runs headlessly via QT_QPA_PLATFORM=offscreen.  No window is shown; no pixels
are compared.  These tests verify:
  - MainWindow and all four tabs construct without exception
  - ResultsTab.load() populates from a synthetic RadiationDataset
  - SolveWorker emits 'finished' and progressChanged with a fake backend

No NumCalc binary required.  No pytest-qt dependency — we use QApplication
directly and call processEvents manually.

Build-order item 10 (GUI construction and wiring, §6 Gameplan).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

# Force offscreen rendering before any Qt import
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QThread
from PySide6.QtWidgets import QApplication


@pytest.fixture(scope="module")
def qapp():
    """Module-scoped QApplication (one per test run)."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv[:1])
    yield app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

F = 3  # frequency steps
N = 14  # Lebedev-14 directions


def _synthetic_dataset():
    """Build a minimal RadiationDataset without any NumCalc solve."""
    from beamsim2.assembly.tensor import build_dataset
    from beamsim2.core.sphere import lebedev
    from beamsim2.core.types import ComplexField

    freqs = np.array([250.0, 500.0, 1000.0])
    obs = lebedev(n_points=N, radius=1.0)

    def _field(seed: int) -> tuple:
        r = np.random.default_rng(seed)
        pressure = r.standard_normal((F, N)) + 1j * r.standard_normal((F, N))
        return ComplexField(
            pressure=pressure.astype(np.complex128),
            convergence_flags=np.ones(F, dtype=bool),
            frequencies=freqs,
        )

    driver_inputs = [
        (
            "drv_a",
            _field(0),
            {
                "name": "drv_a",
                "position": [0, 0, 0],
                "orientation": [0, 0, 1],
                "radius": 0.02,
                "profile": "flush_disk",
            },
        ),
        (
            "drv_b",
            _field(1),
            {
                "name": "drv_b",
                "position": [0, 0, 0],
                "orientation": [0, 0, 1],
                "radius": 0.02,
                "profile": "flush_disk",
            },
        ),
    ]
    return build_dataset(
        driver_inputs=driver_inputs,
        directions=obs,
        freq_grid_spacing="log",
        root_attrs={"phase_origin": [0, 0, 0]},
    )


def _fake_backend():
    """Minimal fake BEMBackend that returns synthetic ComplexField."""
    from beamsim2.backends.base import BEMBackend
    from beamsim2.core.types import (
        ComplexField,
        RawSolveResult,
        ResourcePlan,
        SolveSpec,
    )

    class _Fake(BEMBackend):
        def __init__(self):
            self._call = 0

        def estimate(self, mesh, bc, frequencies, config):
            F = len(frequencies.frequencies)
            return ResourcePlan(np.full(F, 1e9), np.full(F, np.nan))

        def prepare(self, mesh, bc, frequencies, obs, config):
            self._call += 1
            return SolveSpec(f"/tmp/fake_{self._call}", [], frequencies)

        def solve(self, spec, scheduler=None):
            F = len(spec.frequency_grid.frequencies)
            return RawSolveResult(spec.work_dir, set(range(F)), np.ones(F, bool))

        def extract(self, raw, obs):
            F = len(raw.convergence_flags)
            N = obs.unit_vectors.shape[0]
            rng = np.random.default_rng(self._call)
            p = (rng.standard_normal((F, N)) + 1j * rng.standard_normal((F, N))).astype(
                np.complex128
            )
            freqs = np.array([250.0, 500.0, 1000.0])[:F]
            return ComplexField(p, np.ones(F, bool), freqs)

    return _Fake()


# ---------------------------------------------------------------------------
# Test 1: MainWindow construction
# ---------------------------------------------------------------------------


def test_main_window_constructs(qapp):
    """MainWindow must construct without exception in offscreen mode."""
    from beamsim2.gui.app import MainWindow

    win = MainWindow()
    assert win is not None
    assert win.centralWidget() is not None  # the QTabWidget
    win.close()


def test_main_window_has_five_tabs(qapp):
    """MainWindow must expose exactly 5 tabs (incl. the Phase-2 Filter Designer)."""
    from PySide6.QtWidgets import QTabWidget

    from beamsim2.gui.app import MainWindow

    win = MainWindow()
    tabs = win.findChild(QTabWidget)
    assert tabs is not None
    assert tabs.count() == 5
    labels = [tabs.tabText(i) for i in range(5)]
    assert labels == ["Geometry", "Drivers", "Simulation", "Results", "Filter Designer"]
    win.close()


# ---------------------------------------------------------------------------
# Test 2: ResultsTab loads a synthetic dataset
# ---------------------------------------------------------------------------


def test_results_tab_loads_dataset(qapp):
    """ResultsTab.load() must populate without exception and the sub-tabs appear."""
    from beamsim2.gui.app import AppState
    from beamsim2.gui.results_view import ResultsTab

    state = AppState()
    tab = ResultsTab(state)
    ds = _synthetic_dataset()
    tab.load(ds)  # must not raise
    # Canvas should have been drawn — just assert no exception propagated
    assert tab._ds is ds
    tab.close()


def test_results_on_axis_view_loads(qapp):
    """_OnAxisView.load() must handle 2-driver 3-freq-14-dir dataset."""
    from beamsim2.gui.results_view import _OnAxisView

    v = _OnAxisView()
    ds = _synthetic_dataset()
    v.load(ds)
    assert v._ds is ds
    assert v._drv_combo.count() == 2
    v.close()


def test_filter_designer_tab_loads_and_designs(qapp):
    """FilterDesignerTab loads a dataset, runs a design (inline), and replots without raising."""
    from beamsim2.beamform.design import design
    from beamsim2.gui.app import AppState
    from beamsim2.gui.filter_designer_view import FilterDesignerTab

    state = AppState()
    tab = FilterDesignerTab(state)
    ds = _synthetic_dataset()
    tab.load(ds)  # must not raise
    assert tab._ds is ds
    assert tab._freq_combo.count() == F
    assert tab._design_btn.isEnabled()
    assert not tab._export_btn.isEnabled()

    # Run the solver inline (avoid the worker thread) and feed the result back to the tab.
    spec = tab._build_spec()
    result = design(ds, spec)
    tab._on_design_done(result)  # exercises metrics text + both plots
    assert tab._result is result
    assert tab._export_btn.isEnabled()
    assert "Engine" in tab._metrics.text()
    tab.close()


def test_filter_designer_constant_di_engine(qapp):
    """The constant-DI engine path runs end-to-end through the tab's spec builder."""
    from beamsim2.beamform.design import design
    from beamsim2.gui.app import AppState
    from beamsim2.gui.filter_designer_view import _ENGINES, FilterDesignerTab

    state = AppState()
    tab = FilterDesignerTab(state)
    tab.load(_synthetic_dataset())
    tab._engine.setCurrentIndex([e for _, e in _ENGINES].index("constant_di"))
    result = design(tab._ds, tab._build_spec())
    tab._on_design_done(result)
    assert "constant_gdi_db" in result.attrs
    tab.close()


def test_results_balloon_view_loads(qapp):
    """_BalloonView.load() must not raise for a 14-direction dataset."""
    from beamsim2.gui.results_view import _BalloonView

    v = _BalloonView()
    ds = _synthetic_dataset()
    v.load(ds)
    assert v._ds is ds
    v.close()


def test_results_di_map_view_loads(qapp):
    """_DirectivityMapView.load() must not raise."""
    from beamsim2.gui.results_view import _DirectivityMapView

    v = _DirectivityMapView()
    ds = _synthetic_dataset()
    v.load(ds)
    assert v._ds is ds
    v.close()


# ---------------------------------------------------------------------------
# Test 3: SolveWorker thread plumbing
# ---------------------------------------------------------------------------


def test_solve_worker_emits_finished(qapp):
    """SolveWorker must emit 'finished' and multiple 'progressChanged' on a fake solve."""
    from beamsim2.core.types import FrequencyGrid
    from beamsim2.geometry.assemble import DriverSpec
    from beamsim2.gui.app import SolveWorker
    from beamsim2.pipeline.run import BoxGeometry, DriverPlacement, SimulationRequest

    req = SimulationRequest(
        geometry=BoxGeometry(0.12, 0.10, 0.08),
        drivers=[
            DriverPlacement(
                DriverSpec((0.035, 0.05, 0.08), (0.0, 0.0, 1.0), 0.020),
                terminal=None,
                driver_id="drv_a",
            ),
            DriverPlacement(
                DriverSpec((0.085, 0.05, 0.08), (0.0, 0.0, 1.0), 0.020),
                terminal=None,
                driver_id="drv_b",
            ),
        ],
        frequencies=FrequencyGrid(np.array([250.0, 500.0, 1000.0]), spacing="log"),
        sphere_n_points=14,
    )

    results_received = []
    progress_received = []
    failures_received = []

    # Monkey-patch run_simulation WHERE SolveWorker uses it: the import in app.py.
    # SolveWorker holds a reference to run_simulation imported at app.py module scope,
    # so patching beamsim2.pipeline.run.run_simulation has no effect; we must patch
    # the name binding inside beamsim2.gui.app.
    #
    # Use a FULLY fake run_simulation that avoids gmsh (not thread-safe) and
    # the BEM backend entirely.  Still drives ProgressModel so progressChanged fires.
    import beamsim2.gui.app as app_mod
    from beamsim2.geometry.health import HealthReport
    from beamsim2.pipeline.run import SimulationResult

    def _totally_fake_run(r, backend=None, progress=None):
        ds = _synthetic_dataset()
        if progress is not None:
            for m, dp in enumerate(r.drivers):
                progress.driver_started(dp.driver_id, m, len(r.drivers))
                for step in range(len(r.frequencies.frequencies)):
                    progress.step_done(m, step, True)
                progress.driver_finished(
                    dp.driver_id, m, np.zeros(len(r.frequencies.frequencies), dtype=bool)
                )
        health = HealthReport(
            is_watertight=True, open_edge_count=0, problems=[], repairs=[], warnings=[]
        )
        flagged = {dp.driver_id: np.zeros(len(r.frequencies.frequencies), bool) for dp in r.drivers}
        work_dirs = {dp.driver_id: f"/tmp/fake_{dp.driver_id}" for dp in r.drivers}
        return SimulationResult(
            dataset=ds,
            h5_path=None,
            health=health,
            flagged_frequencies=flagged,
            work_dirs=work_dirs,
        )

    original_run = app_mod.run_simulation
    app_mod.run_simulation = _totally_fake_run  # patch the imported name SolveWorker calls

    try:
        thread = QThread()
        worker = SolveWorker(req)
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.finished.connect(results_received.append)
        worker.progressChanged.connect(progress_received.append)
        worker.failed.connect(failures_received.append)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)

        thread.start()

        # Process events while waiting so queued cross-thread signals are delivered.
        # thread.wait() blocks the event loop; instead poll with processEvents().
        import time

        deadline = time.monotonic() + 10.0
        while thread.isRunning() and time.monotonic() < deadline:
            qapp.processEvents()
            time.sleep(0.05)
        qapp.processEvents()  # one final drain

    finally:
        app_mod.run_simulation = original_run  # restore

    if failures_received:
        pytest.fail(f"SolveWorker.failed emitted: {failures_received[0]}")

    assert len(results_received) == 1, f"Expected 1 'finished' signal; got {len(results_received)}"
    assert len(progress_received) > 0, "Expected at least one progressChanged signal"

    assert isinstance(results_received[0], SimulationResult)


def _drive_until_stopped(qapp, thread, timeout_s=10.0):
    """Pump the Qt event loop until a worker thread quits (deliver queued signals)."""
    import time

    deadline = time.monotonic() + timeout_s
    while thread.isRunning() and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.02)
    qapp.processEvents()


def test_design_worker_emits_finished_on_thread(qapp):
    """DesignWorker must emit 'finished' with a DesignResult when run on a real QThread.

    Closes the P2-3 gate 'design ... through the GUI worker' — the smoke tests above call
    the slot inline, so this is the only coverage of moveToThread + signal wiring + quit().
    """
    from beamsim2.beamform.design import DesignResult
    from beamsim2.beamform.targets import TargetSpec
    from beamsim2.gui.filter_designer_view import DesignWorker

    finished, failed = [], []
    thread = QThread()
    worker = DesignWorker(_synthetic_dataset(), TargetSpec(engine="ls"))
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    worker.finished.connect(finished.append)
    worker.failed.connect(failed.append)
    worker.finished.connect(thread.quit)
    worker.failed.connect(thread.quit)

    thread.start()
    _drive_until_stopped(qapp, thread)

    assert not thread.isRunning(), "worker thread did not quit after finishing"
    assert not failed, f"DesignWorker.failed emitted: {failed[:1]}"
    assert len(finished) == 1
    assert isinstance(finished[0], DesignResult)


def test_design_worker_emits_failed_on_bad_spec(qapp):
    """A bad spec makes design() raise; DesignWorker must surface it via 'failed' (not crash)."""
    from beamsim2.beamform.targets import TargetSpec
    from beamsim2.gui.filter_designer_view import DesignWorker

    finished, failed = [], []
    thread = QThread()
    worker = DesignWorker(_synthetic_dataset(), TargetSpec(engine="does_not_exist"))
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    worker.finished.connect(finished.append)
    worker.failed.connect(failed.append)
    worker.finished.connect(thread.quit)
    worker.failed.connect(thread.quit)

    thread.start()
    _drive_until_stopped(qapp, thread)

    assert not thread.isRunning()
    assert not finished
    assert len(failed) == 1 and "does_not_exist" in failed[0]


# ---------------------------------------------------------------------------
# Test 4: AppState dataclass
# ---------------------------------------------------------------------------


def test_app_state_defaults():
    """AppState must construct with sensible defaults."""
    from beamsim2.gui.app import AppState

    state = AppState()
    assert state.geometry is None
    assert state.drivers == []
    assert state.sphere_n_points == 26
    assert state.result is None


# ---------------------------------------------------------------------------
# Test 5: GUI shell package imports
# ---------------------------------------------------------------------------


def test_all_gui_modules_importable():
    """All gui/ modules must import without exception."""
    import beamsim2.gui.app  # noqa: F401
    import beamsim2.gui.filter_designer_view  # noqa: F401
    import beamsim2.gui.geometry_view  # noqa: F401
    import beamsim2.gui.parameters_panel  # noqa: F401
    import beamsim2.gui.results_view  # noqa: F401
    import beamsim2.gui.run_monitor  # noqa: F401
