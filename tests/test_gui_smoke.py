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


def test_results_views_honor_reference_axis(qapp):
    """On-axis pick follows the dataset's reference_axis (default +z, settable +x)."""
    import numpy as np

    from beamsim2.core.sphere import nearest_direction_index
    from beamsim2.gui.results_view import _BalloonView, _OnAxisView, _reference_axis

    ds = _synthetic_dataset()
    uvecs = ds.directions.unit_vectors

    # Default (no attr): +z, identical to the old argmax(z) behaviour.
    assert np.allclose(_reference_axis(ds), [0.0, 0.0, 1.0])
    v = _OnAxisView()
    v.load(ds)
    assert v._last_on_axis_idx == int(np.argmax(uvecs[:, 2]))
    v.close()

    # Settable: +x reference axis must move the on-axis pick to the +x direction,
    # and the balloon (with its axis indicator) must replot without raising.
    ds.attrs["reference_axis"] = [1.0, 0.0, 0.0]
    assert np.allclose(_reference_axis(ds), [1.0, 0.0, 0.0])
    v2 = _OnAxisView()
    v2.load(ds)
    assert v2._last_on_axis_idx == nearest_direction_index(uvecs, (1.0, 0.0, 0.0))
    assert v2._last_on_axis_idx != int(np.argmax(uvecs[:, 2]))
    v2.close()
    b = _BalloonView()
    b.load(ds)  # exercises the reference-axis indicator draw path
    b.close()


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
    # The GUI selects Luo's proper directivity-index objective (Chunk 3b), so it reports the
    # held-constant directivity index, not the cap-ratio GDI.
    assert result.attrs["directivity_mode"] == "index"
    assert "constant_di_db" in result.attrs
    tab.close()


def test_filter_designer_multi_target(qapp):
    """The Multi-target pattern (Chunk 3d) forces Auto-Design and runs end-to-end in the tab."""
    from beamsim2.beamform.design import design
    from beamsim2.gui.app import AppState
    from beamsim2.gui.filter_designer_view import _MULTI_LABEL, _PATTERNS, FilterDesignerTab

    state = AppState()
    tab = FilterDesignerTab(state)
    tab.load(_synthetic_dataset())
    tab._pattern.setCurrentIndex([lbl for lbl, _, _ in _PATTERNS].index(_MULTI_LABEL))
    # Selecting Multi-target locks the engine to Auto-Design and enables the objective controls.
    assert tab._engine.currentText().startswith("Auto-Design")
    assert not tab._engine.isEnabled()
    assert tab._mt_group.isEnabled()

    spec = tab._build_spec()
    assert spec.objective == "multi" and spec.engine == "auto"
    assert spec.target_di_db is not None and spec.target_inroom_slope_db_per_oct is not None

    result = design(tab._ds, spec)
    tab._on_design_done(result)
    assert result.attrs["auto_class"] == "multi"
    assert "multi:" in tab._metrics.text()  # the per-objective achieved-vs-target summary
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


# ---------------------------------------------------------------------------
# Test 6: Chunk-2 results views (polar SH arcs, sonograms, CEA2034, referencing)
# ---------------------------------------------------------------------------


def _analytic_dataset():
    """Two offset monopoles on an icosphere grid — a real, smooth radiation dataset."""
    from beamsim2.assembly.tensor import build_dataset
    from beamsim2.core.sphere import icosphere
    from beamsim2.core.types import ComplexField
    from beamsim2.validation.closed_loop import monopole_field

    obs = icosphere(2, radius=2.0)  # 162 points
    freqs = np.array([100.0, 500.0, 2000.0])
    positions = np.array([[0.10, 0.0, 0.05], [-0.10, 0.0, 0.05]])
    driver_inputs = []
    for i, p in enumerate(positions):
        pressure = monopole_field(p[None, :], obs, freqs)[0].astype(np.complex128)  # [F, N]
        field = ComplexField(pressure, np.ones(len(freqs), bool), freqs)
        driver_inputs.append((f"driver_{i + 1}", field, {"position": p.tolist()}))
    root_attrs = {"reference_axis": [0.0, 0.0, 1.0], "speed_of_sound": 343.2}
    return build_dataset(driver_inputs, obs, root_attrs=root_attrs)


def test_results_polar_view_sh_resamples(qapp):
    """_PolarView must SH-resample to a smooth arc without raising on a real dataset."""
    from beamsim2.gui.results_view import _PolarView

    for plane in ("Horizontal", "Vertical"):
        v = _PolarView(plane)
        v.load(_analytic_dataset())  # must not raise
        v._freq_combo.setCurrentIndex(2)  # exercise a replot at a different frequency
        v.close()


def test_results_sonogram_view_loads(qapp):
    """_DirectivityMapView renders H and V sonograms on a log-f axis without raising."""
    from beamsim2.gui.results_view import _DirectivityMapView

    v = _DirectivityMapView()
    v.load(_analytic_dataset())  # must not raise
    # Two sonogram subplots (H and V) are created on the figure.
    assert len(v._canvas.fig.axes) >= 2
    v.close()


def test_results_cea2034_view_loads(qapp):
    """_Cea2034View computes and plots the spinorama curves without raising."""
    from beamsim2.gui.results_view import _Cea2034View

    v = _Cea2034View()
    v.load(_analytic_dataset())  # must not raise
    v.close()


def test_results_referencing_combo_switches_all_modes(qapp):
    """The dataset-wide referencing combo drives every view through all modes."""
    from beamsim2.core.field_referencing import REFERENCING_MODES
    from beamsim2.gui.app import AppState
    from beamsim2.gui.results_view import ResultsTab

    tab = ResultsTab(AppState())
    tab.load(_analytic_dataset())
    # Cardinal rule: the stored tensors must be byte-for-byte unchanged across all the
    # display-only referencing modes (guard the invariant at the GUI integration boundary).
    snaps = [(d.H_bem.copy(), d.H_full.copy()) for d in tab._ds.drivers]
    for mode in REFERENCING_MODES:
        tab._ref_combo.setCurrentText(mode)  # triggers _on_referencing_changed -> replots
        assert tab._on_axis._mode == mode
        assert tab._cea._mode == mode
    for d, (bem0, full0) in zip(tab._ds.drivers, snaps):
        assert np.array_equal(d.H_bem, bem0), "referencing mutated stored H_bem (cardinal rule)"
        assert np.array_equal(d.H_full, full0), "referencing mutated stored H_full (cardinal rule)"
    tab.close()


def test_results_field_selector_distinguishes_h_bem_and_h_full(qapp):
    """The H_bem/H_full selector actually routes the chosen field (non-trivial terminal resp)."""
    from beamsim2.assembly.tensor import build_dataset
    from beamsim2.core.field_referencing import NEAR_FIELD
    from beamsim2.core.sphere import icosphere
    from beamsim2.core.types import ComplexField
    from beamsim2.gui.results_view import _referenced_field
    from beamsim2.validation.closed_loop import monopole_field

    obs = icosphere(2, radius=2.0)
    freqs = np.array([100.0, 500.0, 2000.0])
    pressure = monopole_field(np.array([[0.1, 0.0, 0.0]]), obs, freqs)[0].astype(np.complex128)
    g = np.array([2.0, 0.5, 4.0], dtype=np.complex128)  # non-trivial terminal response |g|≠1
    field = ComplexField(pressure, np.ones(len(freqs), bool), freqs)
    ds = build_dataset(
        [("driver_1", field, {"position": [0.1, 0.0, 0.0]})], obs, terminal_responses=[g]
    )
    drv = ds.drivers[0]
    h_full = _referenced_field(ds, drv, "H_full", NEAR_FIELD)
    h_bem = _referenced_field(ds, drv, "H_bem", NEAR_FIELD)
    delta_db = 20.0 * np.log10(np.abs(h_full) / np.abs(h_bem))  # [F, N]
    expected = (20.0 * np.log10(np.abs(g)))[:, None]  # [F, 1] — broadcast over directions
    assert np.allclose(delta_db, expected, atol=1e-9), "field selector does not route H_full/H_bem"


def test_results_views_honor_rotated_reference_axis_numerically(qapp):
    """Polar + CEA views built off a +x reference axis sample +x as on-axis, not +z.

    A field peaked at +x: if a view ignored reference_axis and hardcoded +z, its 0°/on-axis
    sample would land in the pattern's flank, not at the peak.
    """
    from beamsim2.assembly.tensor import build_dataset
    from beamsim2.core.sphere import icosphere
    from beamsim2.core.types import ComplexField
    from beamsim2.gui.results_view import _Cea2034View, _PolarView

    obs = icosphere(3, radius=2.0)
    freqs = np.array([500.0])
    # Forward-peaked real pattern about +x: |H| = (0.5 + 0.5 cosθ_x)  (max at +x, min at −x).
    pattern = 0.5 + 0.5 * obs.unit_vectors[:, 0]
    field = ComplexField(np.tile(pattern.astype(np.complex128), (1, 1)), np.ones(1, bool), freqs)
    ds = build_dataset(
        [("d1", field, {"position": [0.0, 0.0, 0.0]})],
        obs,
        root_attrs={"reference_axis": [1.0, 0.0, 0.0]},
    )

    pv = _PolarView("Horizontal")
    pv.load(ds)
    ang, norm_db = pv._last_arc
    on_axis_idx = int(np.argmin(np.abs(ang)))  # 0° sample
    assert norm_db[on_axis_idx] > -0.5, "polar 0° is not the peak → reference axis ignored"
    assert float(norm_db.max()) <= 1e-6  # normalised to 0 dB at the loudest direction
    pv.close()

    cea = _Cea2034View()
    cea.load(ds)
    c = cea._last_curves
    # On-axis (+x, the peak) must be the loudest CEA curve → SPDI/ERDI strictly positive.
    assert float(c["sound_power_di"][0]) > 0.5, "CEA on-axis not at +x peak → axis ignored"
    cea.close()


def test_results_tab_has_cea_subtab(qapp):
    """The Results tab exposes the new CEA2034 sub-tab."""
    from beamsim2.gui.app import AppState
    from beamsim2.gui.results_view import ResultsTab

    tab = ResultsTab(AppState())
    titles = [tab._sub_tabs.tabText(i) for i in range(tab._sub_tabs.count())]
    assert "CEA2034" in titles
    assert "Sonograms" in titles
    tab.close()
