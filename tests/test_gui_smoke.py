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


# ---------------------------------------------------------------------------
# Chunk 3e: filter-designer visualization (render gate — correct series + cardinal rule)
# ---------------------------------------------------------------------------


def _ax_labels(ax):
    """Line labels on one matplotlib axis (the discriminating 'correct series' check)."""
    return [ln.get_label() for ln in ax.lines]


def test_filter_designer_3e_views_render_correct_series(qapp):
    """Chunk 3e: every new filter-designer plot view builds with the correct series.

    Asserts the *series* (per-deliverable line counts / labels), not merely that nothing raised,
    and guards the cardinal rule: plotting reads the stored H-tensor but never mutates it.
    """
    from beamsim2.beamform.design import design
    from beamsim2.gui.app import AppState
    from beamsim2.gui.filter_designer_view import FilterDesignerTab
    from beamsim2.metrics.cea2034 import DI_CURVES, SPL_CURVES

    state = AppState()
    tab = FilterDesignerTab(state)
    ds = _synthetic_dataset()
    n_drivers = len(ds.drivers)  # 2
    tab.load(ds)

    # Nine plot sub-tabs: the beam-axis Polar, the full-system pattern views (H/V Polar, Balloon,
    # Sonograms), and the realization views (Directivity, Filters, Per-driver, CEA2034 / in-room).
    titles = [tab._plot_tabs.tabText(i) for i in range(tab._plot_tabs.count())]
    assert titles == [
        "Polar",
        "H Polar",
        "V Polar",
        "Balloon",
        "Sonograms",
        "Directivity",
        "Filters",
        "Per-driver",
        "CEA2034 / in-room",
    ]

    # Cardinal rule: snapshot the stored tensors; plotting must leave them byte-for-byte equal.
    snaps = [(d.H_bem.copy(), d.H_full.copy()) for d in ds.drivers]

    result = design(ds, tab._build_spec())  # default = cardioid / least-squares
    tab._on_design_done(result)  # refreshes ALL views at once

    for d, (bem0, full0) in zip(ds.drivers, snaps):
        assert np.array_equal(d.H_bem, bem0), "plotting mutated stored H_bem (cardinal rule)"
        assert np.array_equal(d.H_full, full0), "plotting mutated stored H_full (cardinal rule)"

    # Polar (achieved-vs-target directivity): both curves present.
    assert "achieved" in _ax_labels(tab._polar.ax) and "target" in _ax_labels(tab._polar.ax)

    # Directivity dashboard: DI + -6 dB beamwidth + WNG panels, with the WNG floor reference.
    metrics_axes = tab._metrics_canvas.fig.axes
    assert len(metrics_axes) == 4  # 3 panels + the twin axis carrying the target-error series
    metric_labels = {lbl for ax in metrics_axes for lbl in _ax_labels(ax)}
    assert {
        "achieved DI",
        "-6 dB beamwidth",
        "achieved WNG",
        "WNG floor",
        "target err",
    } <= metric_labels

    # Filters: per-driver weight magnitude + phase, one line per driver on each panel.
    f_axes = tab._filter_canvas.fig.axes
    assert len(f_axes) == 2
    assert len(f_axes[0].lines) == n_drivers and len(f_axes[1].lines) == n_drivers

    # Per-driver responses: filtered on-axis per driver + the combined beam (M+1 lines).
    d_axes = tab._driver_canvas.fig.axes
    assert len(d_axes) == 2
    assert len(d_axes[0].lines) == n_drivers + 1
    assert "combined" in _ax_labels(d_axes[0])

    # CEA2034 / in-room: SPL spinorama (left axis) + two DI curves (twin); in-room curve present.
    cea_axes = tab._cea_canvas.fig.axes
    assert len(cea_axes) == 2
    assert len(cea_axes[0].lines) == len(SPL_CURVES)
    assert len(cea_axes[1].lines) == len(DI_CURVES)
    assert "Estimated In-Room" in {lbl for ax in cea_axes for lbl in _ax_labels(ax)}
    tab.close()


def test_filter_designer_3e_multi_target_reference_lines(qapp):
    """3e: multi-target objectives appear as dashed target reference lines on the metrics view."""
    from beamsim2.beamform.design import design
    from beamsim2.gui.app import AppState
    from beamsim2.gui.filter_designer_view import _MULTI_LABEL, _PATTERNS, FilterDesignerTab

    state = AppState()
    tab = FilterDesignerTab(state)
    tab.load(_synthetic_dataset())
    tab._pattern.setCurrentIndex([lbl for lbl, _, _ in _PATTERNS].index(_MULTI_LABEL))
    result = design(tab._ds, tab._build_spec())
    tab._on_design_done(result)
    metric_labels = {lbl for ax in tab._metrics_canvas.fig.axes for lbl in _ax_labels(ax)}
    # The default multi-target spec sets a DI and a beamwidth target -> dashed reference lines.
    assert "target DI" in metric_labels and "target BW" in metric_labels
    tab.close()


def test_filter_designer_3e_cea_references_steer_axis(qapp):
    """3e (load-bearing): the CEA spinorama is referenced to the BEAM axis, not the dataset front.

    All other 3e tests steer +z on a default +z-front dataset, where the two axes coincide and a
    regression to ``_reference_axis(ds)`` would pass unnoticed. Here the beam is steered to +x while
    the dataset front stays +z, so the steer-referenced and front-referenced spinoramas differ — and
    the plotted On-Axis curve must match the steer-referenced one (consistency with the in-room
    slope the orchestrator/metrics line reports).
    """
    from dataclasses import replace

    from beamsim2.beamform.design import design
    from beamsim2.gui.app import AppState
    from beamsim2.gui.filter_designer_view import FilterDesignerTab
    from beamsim2.gui.results_view import _CEA_LABELS
    from beamsim2.metrics.cea2034 import compute_cea2034

    state = AppState()
    tab = FilterDesignerTab(state)
    ds = _synthetic_dataset()  # reference_axis defaults to +z (front = +z)
    tab.load(ds)
    spec = replace(tab._build_spec(), steer_dir=np.array([1.0, 0.0, 0.0]))  # beam +x != front +z
    # Drive the real finished-design path (sets the design current, refreshes every view incl. CEA).
    tab._on_design_done(design(ds, spec))

    onaxis = [
        ln
        for ax in tab._cea_canvas.fig.axes
        for ln in ax.lines
        if ln.get_label() == _CEA_LABELS["on_axis"]
    ][0]
    plotted = np.asarray(onaxis.get_ydata())
    steer_ref = compute_cea2034(
        tab._result.steered_field, ds.frequencies, ds.directions, np.array([1.0, 0.0, 0.0])
    )["on_axis"]
    front_ref = compute_cea2034(
        tab._result.steered_field, ds.frequencies, ds.directions, np.array([0.0, 0.0, 1.0])
    )["on_axis"]
    assert np.allclose(plotted, steer_ref, atol=1e-6)  # plotted spinorama uses the beam axis
    assert not np.allclose(plotted, front_ref, atol=0.1)  # ... which genuinely differs from front
    tab.close()


def test_filter_designer_3e_metrics_handles_inf_wng_and_infeasible(qapp):
    """3e: the metrics view survives -inf WNG (collapsed bins) and flags infeasible bins in red."""
    from beamsim2.beamform.design import design
    from beamsim2.gui.app import AppState
    from beamsim2.gui.filter_designer_view import FilterDesignerTab

    state = AppState()
    tab = FilterDesignerTab(state)
    ds = _synthetic_dataset()
    tab.load(ds)
    result = design(ds, tab._build_spec())
    floor = float(result.spec.wng_floor_db)
    # Inject the documented honest edge cases into the FROZEN metrics (display-only mutation of
    # copies of the result's dicts — never the stored H): bin 0 = a finite sub-floor INFEASIBLE bin
    # (the common case -> a *visible* red marker); bin 1 = a collapsed bin (-inf) that stays
    # feasible (proves -inf is masked to a gap, not crashing the log axis).
    wng = np.asarray(result.metrics["wng_db"], dtype=float).copy()
    wng[0] = floor - 3.0
    wng[1] = -np.inf
    result.metrics["wng_db"] = wng
    feas = np.asarray(result.metrics["feasible_mask"], dtype=bool).copy()
    feas[0] = False
    result.metrics["feasible_mask"] = feas
    tab._on_design_done(result)  # must not raise on -inf / nan
    marker = [
        ln
        for ax in tab._metrics_canvas.fig.axes
        for ln in ax.lines
        if ln.get_label() == "infeasible bin"
    ][0]
    ydata = np.asarray(marker.get_ydata(), dtype=float)
    assert ydata.size >= 1 and np.all(np.isfinite(ydata))  # a real drawn marker, not plotted at nan
    tab.close()


def test_filter_designer_3e_freq_combo_redraws_polar(qapp):
    """The 'Frequency' combo redraws the per-frequency views (the cuts + balloon span one bin)."""
    from beamsim2.beamform.design import design
    from beamsim2.gui.app import AppState
    from beamsim2.gui.filter_designer_view import FilterDesignerTab

    state = AppState()
    tab = FilterDesignerTab(state)
    ds = _synthetic_dataset()
    tab.load(ds)
    result = design(ds, tab._build_spec())
    tab._on_design_done(result)
    tab._freq_combo.setCurrentIndex(0)  # fires _replot_freq_views (beam-axis + H/V Polar + Balloon)
    assert f"{ds.frequencies[0]:.0f} Hz" in tab._polar.ax.get_title()
    assert f"{ds.frequencies[0]:.0f} Hz" in tab._hpolar.ax.get_title()
    tab.close()


# ---------------------------------------------------------------------------
# Full-system pattern views + live target preview (this chunk)
# ---------------------------------------------------------------------------


def _labels_of(ax):
    """Set of line labels on one matplotlib axis."""
    return {ln.get_label() for ln in ax.lines}


def test_filter_designer_target_preview_before_design(qapp):
    """The target response is shown live BEFORE any design, on every target-aware view.

    Guards Task 2 (target visible pre-design) and the cardinal rule (the preview build/draw reads
    but never mutates the stored H-tensor).
    """
    from beamsim2.gui.app import AppState
    from beamsim2.gui.filter_designer_view import FilterDesignerTab
    from beamsim2.metrics.cea2034 import SPL_CURVES

    tab = FilterDesignerTab(AppState())
    ds = _synthetic_dataset()
    snaps = [(d.H_bem.copy(), d.H_full.copy()) for d in ds.drivers]

    tab.load(ds)  # builds + draws the preview — no design() called

    assert tab._result is None  # nothing solved yet
    assert tab._target is not None  # but the target field is cached + drawn
    assert not tab._export_btn.isEnabled()
    for d, (bem0, full0) in zip(ds.drivers, snaps):  # cardinal rule
        assert np.array_equal(d.H_bem, bem0) and np.array_equal(d.H_full, full0)

    # Beam-axis Polar + the full-system H/V cuts show the target, with no achieved overlay yet.
    assert "target" in _labels_of(tab._polar.ax) and "achieved" not in _labels_of(tab._polar.ax)
    for canvas in (tab._hpolar, tab._vpolar):
        labels = _labels_of(canvas.ax)
        assert "target" in labels and "achieved (system)" not in labels

    # Balloon (3-D scatter) + Sonograms (pcolormesh) + CEA (target spinorama) all render — and the
    # balloon/sonogram titles must say "target preview" (pins the achieved-vs-target source, since
    # those two views swap the whole field rather than overlaying a labelled curve).
    assert len(tab._balloon.ax.collections) >= 1
    assert "target preview" in tab._balloon.ax.get_title()
    sono_titles = " ".join(ax.get_title() for ax in tab._sonogram_canvas.fig.axes)
    assert len(tab._sonogram_canvas.fig.axes) >= 2 and "target preview" in sono_titles
    assert len(tab._cea_canvas.fig.axes[0].lines) == len(SPL_CURVES)
    # Directivity preview: the WNG-floor reference line is drawn; achieved curves await a design.
    metric_labels = {lbl for ax in tab._metrics_canvas.fig.axes for lbl in _labels_of(ax)}
    assert "WNG floor" in metric_labels and "achieved DI" not in metric_labels
    tab.close()


def test_filter_designer_target_preview_updates_on_param_change(qapp):
    """Changing a target parameter redraws the live target (different shape) without designing."""
    from beamsim2.gui.app import AppState
    from beamsim2.gui.filter_designer_view import _PATTERNS, FilterDesignerTab

    tab = FilterDesignerTab(AppState())
    tab.load(_synthetic_dataset())
    labels = [lbl for lbl, _, _ in _PATTERNS]

    def _polar_target():
        line = [ln for ln in tab._polar.ax.lines if ln.get_label() == "target"][0]
        return np.asarray(line.get_ydata()).copy()

    tab._pattern.setCurrentIndex(labels.index("Omni"))
    omni = _polar_target()  # omni: a flat (normalized) target lobe
    tab._pattern.setCurrentIndex(labels.index("Figure-8"))
    fig8 = _polar_target()  # figure-8: a deep null off-axis
    assert tab._result is None  # still no design — this is the live preview
    assert not np.allclose(omni, fig8)  # the target followed the pattern change
    tab.close()


def test_filter_designer_full_system_views_after_design(qapp):
    """After a design the full-system cuts overlay achieved+target; balloon/sonograms render."""
    from beamsim2.beamform.design import design
    from beamsim2.gui.app import AppState
    from beamsim2.gui.filter_designer_view import FilterDesignerTab

    tab = FilterDesignerTab(AppState())
    ds = _synthetic_dataset()
    tab.load(ds)
    snaps = [(d.H_bem.copy(), d.H_full.copy()) for d in ds.drivers]

    tab._on_design_done(design(ds, tab._build_spec()))

    for d, (bem0, full0) in zip(ds.drivers, snaps):  # cardinal rule on the full-system paths too
        assert np.array_equal(d.H_bem, bem0) and np.array_equal(d.H_full, full0)
    for canvas in (tab._hpolar, tab._vpolar):
        labels = _labels_of(canvas.ax)
        assert "achieved (system)" in labels and "target" in labels
    # Balloon + Sonograms now draw the achieved system field — their titles must say so (the
    # discriminating check that the field source flipped from the preview, not just "rendered").
    assert len(tab._balloon.ax.collections) >= 1
    assert "achieved (system)" in tab._balloon.ax.get_title()
    sono_titles = " ".join(ax.get_title() for ax in tab._sonogram_canvas.fig.axes)
    assert len(tab._sonogram_canvas.fig.axes) >= 2 and "achieved (system)" in sono_titles
    tab.close()


def test_filter_designer_multi_target_preview_reference_lines(qapp):
    """The multi-target DI/beamwidth target lines show in the PRE-design preview, not just achieved.

    Pins both conditional branches of ``_replot_metrics_preview`` (target DI + target beamwidth)
    in the no-design state — the achieved metrics path is covered separately.
    """
    from beamsim2.gui.app import AppState
    from beamsim2.gui.filter_designer_view import _MULTI_LABEL, _PATTERNS, FilterDesignerTab

    tab = FilterDesignerTab(AppState())
    tab.load(_synthetic_dataset())
    tab._pattern.setCurrentIndex([lbl for lbl, _, _ in _PATTERNS].index(_MULTI_LABEL))
    assert tab._result is None  # no design — this is the live preview
    metric_labels = {lbl for ax in tab._metrics_canvas.fig.axes for lbl in _labels_of(ax)}
    assert {"target DI", "target BW", "WNG floor"} <= metric_labels
    assert "achieved DI" not in metric_labels  # achieved curves await a design
    tab.close()


def test_filter_designer_param_change_after_design_is_non_destructive(qapp):
    """A param change after a design goes stale (reverts to preview, disables export), keeps it."""
    from beamsim2.beamform.design import design
    from beamsim2.gui.app import AppState
    from beamsim2.gui.filter_designer_view import FilterDesignerTab

    tab = FilterDesignerTab(AppState())
    ds = _synthetic_dataset()
    tab.load(ds)
    tab._on_design_done(design(ds, tab._build_spec()))
    assert tab._show_achieved() and tab._export_btn.isEnabled()
    assert "achieved (system)" in _labels_of(tab._hpolar.ax)

    tab._steer_theta.setValue(tab._steer_theta.value() + 10.0)  # nudge a target parameter

    assert tab._stale and not tab._show_achieved()  # views revert to the live target preview
    assert not tab._export_btn.isEnabled()  # exporting a stale design would be wrong
    assert "achieved (system)" not in _labels_of(tab._hpolar.ax)
    assert tab._result is not None  # the design itself is retained (non-destructive)
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
# Chunk 4: model-viewer UX (#1 reference-axis indicator, #2 place-at-click,
# #3 orientation round-trip & face-normal authority).  PyVista can't run under
# offscreen Qt, so these test the GL-free logic: the orientation round-trip, the
# shared placement-reconcile rule, and the pure indicator-placement geometry.
# ---------------------------------------------------------------------------


def test_face_normal_combo_invariant(qapp):
    """The driver editor's 'Face normal' combo is index-aligned with face_id.

    The whole #3 fix (mapping a stored normal back to a combo index, and detecting a
    re-orient onto a new face) rests on: combo item i == FACE_NORMALS[i] == the outward
    normal of face_id i == the inverse map face_id_from_normal.  If anyone reorders the
    combo or the normal table, this breaks silently — so assert it.
    """
    import numpy as np

    from beamsim2.geometry.faces import FACE_NAMES, FACE_NORMALS, face_basis, face_id_from_normal
    from beamsim2.gui.parameters_panel import TSDialog

    dlg = TSDialog()
    assert dlg._normal_combo.count() == len(FACE_NORMALS) == len(FACE_NAMES) == 6
    for i in range(6):
        dlg._normal_combo.setCurrentIndex(i)
        assert np.allclose(dlg._normal_from_combo(), FACE_NORMALS[i])
        assert np.allclose(face_basis(i, 0.12, 0.10, 0.08).normal, FACE_NORMALS[i])
        assert face_id_from_normal(FACE_NORMALS[i]) == i
    dlg.close()


def test_ts_dialog_prefill_restores_orientation(qapp):
    """#3 (the reported bug): _prefill must set the combo from dp.spec.normal, not +z.

    Before the fix _prefill never touched the combo, so it always showed index 0 (+z) and
    OK-ing the dialog silently re-zeroed the driver's true orientation.
    """
    import numpy as np

    from beamsim2.driver.terminal import default_terminal_model
    from beamsim2.geometry.assemble import DriverSpec
    from beamsim2.geometry.faces import FACE_NORMALS
    from beamsim2.gui.parameters_panel import TSDialog
    from beamsim2.pipeline.run import DriverPlacement

    for i, normal in enumerate(FACE_NORMALS):
        dp = DriverPlacement(
            spec=DriverSpec(center=(0.06, 0.05, 0.08), normal=normal, radius=0.03),
            terminal=default_terminal_model("d"),
            driver_id="d",
        )
        dlg = TSDialog(placement=dp)
        assert dlg._normal_combo.currentIndex() == i  # combo shows the true orientation
        assert np.allclose(dlg._normal_from_combo(), normal)
        dlg.close()


def test_reconcile_placement_same_and_new_face():
    """The shared reconcile rule: keep position on the same face; recentre+clamp on a new one."""
    import numpy as np

    from beamsim2.geometry.faces import (
        FacePlacement,
        face_id_from_normal,
        fits_on_face,
        reconcile_placement,
    )

    w, h, d = 0.12, 0.10, 0.08  # +z face half-extents (0.06, 0.05); +x face (0.05, 0.04)

    # Same face (+z stays +z): position preserved, normal unchanged.
    fp0 = FacePlacement(face_id=0, u=0.02, v=-0.01, radius=0.03)
    spec, fp = reconcile_placement((0.0, 0.0, 1.0), fp0, 0.03, w, h, d)
    assert fp.face_id == 0 and (fp.u, fp.v) == (0.02, -0.01)
    assert np.allclose(spec.normal, (0.0, 0.0, 1.0))

    # Same face but an enlarged radius forces the position to re-clamp inward (exercise the
    # same-face clamp branch): +z half_u = 0.06, so a 0.05 m radius at u = 0.055 -> u = 0.01.
    fp_grow = FacePlacement(face_id=0, u=0.055, v=0.0, radius=0.03)
    _, fp_c = reconcile_placement((0.0, 0.0, 1.0), fp_grow, 0.05, w, h, d)
    assert fp_c.face_id == 0 and fp_c.radius == 0.05
    assert np.isclose(fp_c.u, 0.01) and fits_on_face(fp_c, w, h, d)  # re-clamped from 0.055

    # Re-orient onto a new (+x) face: recentre to the face centroid and clamp radius to fit.
    fp_big = FacePlacement(face_id=0, u=0.0, v=0.0, radius=0.045)  # 0.045 > +x half_v (0.04)
    spec2, fp2 = reconcile_placement((1.0, 0.0, 0.0), fp_big, 0.045, w, h, d)
    assert fp2.face_id == face_id_from_normal((1.0, 0.0, 0.0)) == 2
    assert (fp2.u, fp2.v) == (0.0, 0.0)
    assert fp2.radius == 0.04  # clamped to min(half_u, half_v) of the new face
    assert fits_on_face(fp2, w, h, d)
    assert np.allclose(spec2.normal, (1.0, 0.0, 0.0))


def test_drivers_tab_edit_reorients_and_persists(qapp, monkeypatch):
    """#3 end-to-end: editing a face-placed driver's orientation moves it AND survives reopen.

    The discriminating test the prefill round-trip alone can't catch: a face-placed driver's
    normal is otherwise re-derived from its (unchanged) face_placement, so a combo change would
    revert.  Drives the real DriversTab.edit path with a stubbed dialog, then reopens the REAL
    dialog and asserts the combo shows the persisted (+x) orientation.
    """
    import numpy as np
    from PySide6.QtWidgets import QDialog

    from beamsim2.driver.terminal import default_terminal_model
    from beamsim2.geometry.assemble import DriverSpec
    from beamsim2.geometry.faces import FacePlacement, face_id_from_normal, face_local_to_spec
    from beamsim2.gui import parameters_panel as pp
    from beamsim2.gui.app import AppState
    from beamsim2.pipeline.run import DriverPlacement

    state = AppState()
    state.box_dims = (0.12, 0.10, 0.08)
    fp = FacePlacement(face_id=0, u=0.01, v=0.0, radius=0.03)  # on the +z face
    dp = DriverPlacement(
        spec=face_local_to_spec(fp, *state.box_dims),
        terminal=default_terminal_model("d0"),
        driver_id="d0",
        face_placement=fp,
    )
    state.drivers.append(dp)
    tab = pp.DriversTab(state)
    real_ts_dialog = pp.TSDialog  # keep the real class for the reopen check

    class _StubDialog:
        """Stand-in for TSDialog: returns Accepted with the orientation flipped to +x."""

        def __init__(self, placement=None, parent=None):
            self._dp = placement

        def exec(self):
            return QDialog.DialogCode.Accepted

        @property
        def placement(self):
            old = self._dp
            new_spec = DriverSpec(center=old.spec.center, normal=(1.0, 0.0, 0.0), radius=0.03)
            return DriverPlacement(
                spec=new_spec,
                terminal=old.terminal,
                driver_id=old.driver_id,
                face_placement=old.face_placement,
            )

    monkeypatch.setattr(pp, "TSDialog", _StubDialog)
    tab._edit_driver(0)

    edited = state.drivers[0]
    # Persisted: BOTH the placement face and the derived spec normal are now +x.
    assert edited.face_placement.face_id == face_id_from_normal((1.0, 0.0, 0.0)) == 2
    assert np.allclose(edited.spec.normal, (1.0, 0.0, 0.0))

    # Survives reopen: the REAL dialog's combo now shows +x (index 2), not the +z default.
    reopened = real_ts_dialog(placement=edited)
    assert reopened._normal_combo.currentIndex() == 2
    reopened.close()


def test_reference_axis_indicator_geometry():
    """#1: the pure indicator placement points along the reference axis and rotates with it."""
    import numpy as np

    from beamsim2.geometry.faces import reference_axis_indicator

    w, h, d = 0.12, 0.10, 0.08
    center = np.array([w / 2, h / 2, d / 2])

    ind_z = reference_axis_indicator((0.0, 0.0, 1.0), w, h, d)
    assert np.allclose(ind_z.origin, center)
    assert np.allclose(ind_z.direction, (0.0, 0.0, 1.0))
    assert np.isclose(ind_z.length, 1.6 * max(w, h, d))
    assert np.allclose(ind_z.tip, center + np.array([0.0, 0.0, 1.0]) * ind_z.length)
    assert np.allclose(ind_z.mic_pos, ind_z.tip)

    # Rotating the axis to +x rotates the whole indicator (arrow + mic) to +x.
    ind_x = reference_axis_indicator((1.0, 0.0, 0.0), w, h, d)
    assert np.allclose(ind_x.direction, (1.0, 0.0, 0.0))
    assert ind_x.tip[0] > center[0] and np.isclose(ind_x.tip[2], center[2])
    assert not np.allclose(ind_x.tip, ind_z.tip)

    # A non-unit axis is normalised; a zero axis falls back to +z (via reference_frame).
    ind_scaled = reference_axis_indicator((0.0, 5.0, 0.0), w, h, d)
    assert np.allclose(ind_scaled.direction, (0.0, 1.0, 0.0))
    assert np.allclose(
        reference_axis_indicator((0.0, 0.0, 0.0), w, h, d).direction, (0.0, 0.0, 1.0)
    )


def test_geometry_tab_reference_axis_control(qapp):
    """#1 wiring: the Geometry tab exposes a 6-way reference-axis combo that drives AppState."""
    import numpy as np

    from beamsim2.geometry.faces import FACE_NORMALS
    from beamsim2.gui.app import AppState
    from beamsim2.gui.geometry_view import GeometryTab

    state = AppState()
    tab = GeometryTab(state)
    assert tab._ref_axis_combo.count() == 6
    assert tab._ref_axis_combo.currentIndex() == 0  # default +z front
    assert np.allclose(state.reference_axis, (0.0, 0.0, 1.0))
    assert state.box_dims == (tab._w.value(), tab._h.value(), tab._d.value())  # dims mirrored

    tab._ref_axis_combo.setCurrentIndex(2)  # +x
    assert np.allclose(state.reference_axis, FACE_NORMALS[2])

    # Changing a box dimension must re-mirror box_dims (the basis the Drivers-list edit
    # reconcile reads) — otherwise a re-orient there would reconcile against stale dims.
    tab._w.setValue(0.20)  # fires _on_dims_changed
    assert state.box_dims == (tab._w.value(), tab._h.value(), tab._d.value())
    assert state.box_dims[0] == 0.20
    tab.close()


def test_canvas_driver_added_places_at_click(qapp):
    """#2: a driver added from the canvas lands at the clicked (u, v), not the face centre."""
    import numpy as np

    from beamsim2.geometry.faces import face_local_to_center
    from beamsim2.gui.app import AppState
    from beamsim2.gui.geometry_view import GeometryTab

    state = AppState()
    tab = GeometryTab(state)
    w, h, d = tab._w.value(), tab._h.value(), tab._d.value()

    tab._on_canvas_driver_added(0, 0.02, -0.015, 0.03)  # +z face, off-centre click
    assert len(state.drivers) == 1
    dp = state.drivers[0]
    assert dp.face_placement.face_id == 0
    assert (dp.face_placement.u, dp.face_placement.v) == (0.02, -0.015)
    assert dp.face_placement.u != 0.0  # genuinely off-centre, not forced to centroid
    assert np.allclose(dp.spec.center, face_local_to_center(dp.face_placement, w, h, d))
    tab.close()


def test_canvas_edit_reorients_face_placed_driver(qapp, monkeypatch):
    """#3: the canvas right-click 'Edit T/S' path reconciles a re-orient too (dims from spinboxes).

    Mirrors the DriversTab edit test for the geometry_view path, which sources box dims from its own
    spin-boxes (not AppState.box_dims) — so both edit entry points are guarded independently.
    """
    import numpy as np
    from PySide6.QtWidgets import QDialog

    from beamsim2.driver.terminal import default_terminal_model
    from beamsim2.geometry.assemble import DriverSpec
    from beamsim2.geometry.faces import FacePlacement, face_id_from_normal, face_local_to_spec
    from beamsim2.gui import geometry_view as gv
    from beamsim2.gui.app import AppState

    state = AppState()
    tab = gv.GeometryTab(state)
    w, h, d = tab._w.value(), tab._h.value(), tab._d.value()
    fp = FacePlacement(face_id=0, u=0.0, v=0.0, radius=0.03)  # +z face
    from beamsim2.pipeline.run import DriverPlacement

    state.drivers.append(
        DriverPlacement(
            spec=face_local_to_spec(fp, w, h, d),
            terminal=default_terminal_model("d0"),
            driver_id="d0",
            face_placement=fp,
        )
    )

    class _StubDialog:
        def __init__(self, placement=None, parent=None):
            self._dp = placement

        def exec(self):
            return QDialog.DialogCode.Accepted

        @property
        def placement(self):
            old = self._dp
            return DriverPlacement(
                spec=DriverSpec(center=old.spec.center, normal=(0.0, 1.0, 0.0), radius=0.03),
                terminal=old.terminal,
                driver_id=old.driver_id,
                face_placement=old.face_placement,
            )

    # _on_canvas_driver_edited imports TSDialog from parameters_panel inside the method.
    import beamsim2.gui.parameters_panel as pp

    monkeypatch.setattr(pp, "TSDialog", _StubDialog)
    tab._on_canvas_driver_edited(0)

    edited = state.drivers[0]
    assert edited.face_placement.face_id == face_id_from_normal((0.0, 1.0, 0.0)) == 4  # +y
    assert np.allclose(edited.spec.normal, (0.0, 1.0, 0.0))
    tab.close()


def test_build_request_threads_reference_axis(qapp):
    """#1 wiring: SimulationTab.build_request carries AppState.reference_axis into the request.

    Closes the Chunk-1 cross-cutting loop: the editor indicator and the solved dataset's
    reference_axis attr must come from the same place, else they would silently disagree.
    """
    import numpy as np

    from beamsim2.geometry.assemble import DriverSpec
    from beamsim2.gui.app import AppState
    from beamsim2.gui.parameters_panel import SimulationTab
    from beamsim2.pipeline.run import BoxGeometry, DriverPlacement

    state = AppState()
    state.geometry = BoxGeometry(0.12, 0.10, 0.08)
    state.drivers.append(
        DriverPlacement(
            spec=DriverSpec((0.06, 0.05, 0.08), (0.0, 0.0, 1.0), 0.02),
            terminal=None,
            driver_id="d0",
        )
    )
    state.reference_axis = (1.0, 0.0, 0.0)  # +x front

    tab = SimulationTab(state)
    assert tab.build_request(state) is True
    assert np.allclose(tab.current_request.reference_axis, (1.0, 0.0, 0.0))
    tab.close()


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


# ---------------------------------------------------------------------------
# Chunk 5b: steer-to-front-axis (RC2) + delay-and-sum engine guidance (RC3)
# ---------------------------------------------------------------------------


def test_filter_designer_steers_from_front_axis(qapp):
    """Steering is measured from the dataset's reference_axis; theta=0 aims out the front (RC2).

    The run2 failure: the GUI defaulted the steer to world +z while the loudspeaker front (and the
    opposed-driver axis) was +x, so the beam aimed broadside and no cardioid could form. The fix
    defaults the steer to the front axis.
    """
    import numpy as np

    from beamsim2.gui.app import AppState
    from beamsim2.gui.filter_designer_view import FilterDesignerTab

    tab = FilterDesignerTab(AppState())
    ds = _synthetic_dataset()  # no reference_axis -> +z front
    tab.load(ds)
    assert np.allclose(tab._front_axis, [0.0, 0.0, 1.0])
    assert np.allclose(tab._steer_dir(), [0.0, 0.0, 1.0], atol=1e-9)  # theta=0 -> front

    # +x-facing speaker (the run2 geometry): the default beam must aim +x, not +z.
    ds.attrs["reference_axis"] = [1.0, 0.0, 0.0]
    tab.load(ds)
    assert np.allclose(tab._front_axis, [1.0, 0.0, 0.0])
    assert "+x" in tab._front_lbl.text()
    assert np.allclose(tab._steer_dir(), [1.0, 0.0, 0.0], atol=1e-9)  # cardioid aims out the front

    # An off-axis steer stays a unit vector orthogonal-component-correct about the front.
    tab._steer_theta.setValue(90.0)
    tab._steer_phi.setValue(0.0)
    s = tab._steer_dir()
    assert np.isclose(np.linalg.norm(s), 1.0)
    assert abs(float(s @ np.array([1.0, 0.0, 0.0]))) < 1e-9  # 90 deg off the +x front
    tab.close()


def test_filter_designer_delay_sum_warning(qapp):
    """Delay-and-sum + a shaping target shows the guidance note; omni or LS/Auto does not (RC3)."""
    from beamsim2.gui.app import AppState
    from beamsim2.gui.filter_designer_view import _ENGINES, _PATTERNS, FilterDesignerTab

    tab = FilterDesignerTab(AppState())
    tab.load(_synthetic_dataset())
    p = {lbl: i for i, (lbl, _, _) in enumerate(_PATTERNS)}
    e = {eng: i for i, (_, eng) in enumerate(_ENGINES)}

    # Check the note's text (isVisible() is unreliable for an unshown offscreen widget).
    tab._pattern.setCurrentIndex(p["Cardioid"])
    tab._engine.setCurrentIndex(e["delay_sum"])
    assert tab._engine_note.text()  # delay-sum can't make a cardioid -> warn

    tab._engine.setCurrentIndex(e["ls"])
    assert not tab._engine_note.text()  # LS can -> no warning

    tab._pattern.setCurrentIndex(p["Omni"])
    tab._engine.setCurrentIndex(e["delay_sum"])
    assert not tab._engine_note.text()  # delay-sum is fine for omni
    tab.close()


def test_filter_designer_cardioid_on_real_run2_data(qapp):
    """End-to-end on the reconstructed real 2-driver data: front-steered cardioid has a rear null.

    Local-only (HDF5/ is git-ignored): skipped when the run2 export is absent. Guards the whole
    RC1+RC2 chain through the GUI spec builder on real data.
    """
    import numpy as np

    try:
        from _fixtures.reconstruct_run2 import load_run2_dataset, run2_available
    except ImportError:
        import pytest

        pytest.skip("run2 reconstruction fixture not importable")
    import pytest

    if not run2_available():
        pytest.skip("HDF5/run2 export not present (local-only verification)")

    from beamsim2.beamform.design import design
    from beamsim2.gui.app import AppState
    from beamsim2.gui.filter_designer_view import _ENGINES, _PATTERNS, FilterDesignerTab

    ds = load_run2_dataset()
    tab = FilterDesignerTab(AppState())
    tab.load(ds)  # steer defaults to the +x front axis
    tab._pattern.setCurrentIndex([lbl for lbl, _, _ in _PATTERNS].index("Cardioid"))
    tab._engine.setCurrentIndex([eng for _, eng in _ENGINES].index("ls"))
    result = design(ds, tab._build_spec())

    uv = ds.directions.unit_vectors
    look = int(np.argmax(uv @ np.array([1.0, 0.0, 0.0])))
    rear = int(np.argmax(uv @ np.array([-1.0, 0.0, 0.0])))
    freqs = np.asarray(ds.frequencies)
    fi = int(np.argmin(np.abs(freqs - 150.0)))  # inside the achievable cardioid band
    p = result.steered_field[fi]
    rear_db = 20.0 * np.log10(np.abs(p[rear]) / np.abs(p[look]))
    assert rear_db < -12.0, f"no rear null in-band: {rear_db:.1f} dB"
    assert bool(result.metrics["feasible_mask"][fi]), "in-band bin should be feasible"
    tab.close()
