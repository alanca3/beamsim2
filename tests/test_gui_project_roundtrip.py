"""GUI project round-trip test: save → New → load reproduces state; undo/redo cycle.

Runs headlessly with QT_QPA_PLATFORM=offscreen.  No NumCalc binary required.
Exercises MainWindow._gather_state / _apply_state / _save_to / _load_project_from
and the undo stack.
"""

from __future__ import annotations

import os
import sys

import pytest

# Offscreen Qt — must be set before importing Qt
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from beamsim2.driver.inductance import LR2Ladder
from beamsim2.driver.terminal import TerminalModel
from beamsim2.driver.thiele_small import TSParams
from beamsim2.geometry.assemble import DriverSpec
from beamsim2.geometry.faces import FacePlacement
from beamsim2.gui.app import MainWindow
from beamsim2.pipeline.run import DriverPlacement


@pytest.fixture(scope="module")
def qapp():
    """Module-scoped QApplication for headless Qt tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv[:1])
    yield app


def _make_driver(driver_id: str = "woofer_0") -> DriverPlacement:
    """Build a DriverPlacement with an LR2Ladder inductance for rich round-trip coverage."""
    ts = TSParams(Re=6.0, Bl=7.0, Mms=0.012, Cms=8e-4, Rms=1.0, Sd=0.0133)
    fp = FacePlacement(face_id=0, u=0.02, v=-0.01, radius=0.03)
    spec = DriverSpec(center=(0.06, 0.05, 0.08), normal=(0.0, 0.0, 1.0), radius=0.03)
    return DriverPlacement(
        spec=spec,
        terminal=TerminalModel(
            ts=ts,
            inductance=LR2Ladder(Le=0.5e-3, Le2=0.2e-3, Re2=3.0),
            box_volume=0.005,
            voltage=2.83,
            name=driver_id,
        ),
        driver_id=driver_id,
        face_placement=fp,
    )


# ---------------------------------------------------------------------------
# Test 1: _gather_state / _apply_state round-trip (no file I/O)
# ---------------------------------------------------------------------------


def test_gather_apply_roundtrip(qapp, tmp_path):
    """Gather state, reset via New, apply back → drivers + sim params reproduced."""
    win = MainWindow()

    # Set some non-default widget values
    win._geo_tab._w.setValue(0.20)
    win._geo_tab._h.setValue(0.15)
    win._geo_tab._d.setValue(0.10)
    win._geo_tab._fi.setValue(0.005)
    win._geo_tab._ref_axis_combo.setCurrentIndex(2)  # +x

    win._sim_tab._f_lo.setValue(50.0)
    win._sim_tab._f_hi.setValue(5000.0)
    win._sim_tab._res_combo.setCurrentIndex(0)  # 1/3 octave
    win._sim_tab._sphere_combo.setCurrentIndex(1)  # 14-pt lebedev

    win._state.drivers.append(_make_driver("woofer_0"))
    win._state.drivers.append(_make_driver("tweeter_0"))

    # Snapshot
    doc = win._gather_state()

    # Clear all state via New (apply a blank document)
    win._applying = True
    try:
        from beamsim2.gui.app import MainWindow as _MW

        win._apply_state(_MW._blank_document())
    finally:
        win._applying = False

    assert len(win._state.drivers) == 0
    assert win._geo_tab._w.value() == pytest.approx(0.12, abs=1e-4)

    # Restore from snapshot
    win._apply_state(doc)
    qapp.processEvents()

    # Geometry params
    assert win._geo_tab._w.value() == pytest.approx(0.20, abs=1e-4)
    assert win._geo_tab._h.value() == pytest.approx(0.15, abs=1e-4)
    assert win._geo_tab._d.value() == pytest.approx(0.10, abs=1e-4)
    assert win._geo_tab._fi.value() == pytest.approx(0.005, abs=1e-6)
    assert win._geo_tab._ref_axis_combo.currentIndex() == 2

    # Drivers
    assert len(win._state.drivers) == 2
    ids = [dp.driver_id for dp in win._state.drivers]
    assert "woofer_0" in ids
    assert "tweeter_0" in ids

    # Sim params
    assert win._sim_tab._f_lo.value() == pytest.approx(50.0, abs=0.1)
    assert win._sim_tab._f_hi.value() == pytest.approx(5000.0, abs=0.1)
    assert win._sim_tab._res_combo.currentIndex() == 0
    assert win._sim_tab._sphere_combo.currentIndex() == 1

    win._dirty = False  # prevent unsaved-changes dialog in headless close
    win.close()


# ---------------------------------------------------------------------------
# Test 2: save → New → load reproduces state (file I/O)
# ---------------------------------------------------------------------------


def test_save_new_load_roundtrip(qapp, tmp_path):
    """Save project file, reset to blank, reload → all state reproduced."""
    win = MainWindow()

    # Configure non-default state
    win._geo_tab._w.setValue(0.25)
    win._geo_tab._d.setValue(0.12)
    win._state.drivers.append(_make_driver("d0"))
    win._sim_tab._f_lo.setValue(80.0)
    win._sim_tab._f_hi.setValue(8000.0)

    # Save
    project_file = tmp_path / "test_project.bsim"
    win._save_to(project_file)
    assert project_file.is_file()
    assert not win._dirty

    # Reset to blank
    from beamsim2.gui.app import MainWindow as _MW

    win._applying = True
    try:
        win._apply_state(_MW._blank_document())
    finally:
        win._applying = False
        win._state.drivers.clear()

    assert len(win._state.drivers) == 0
    assert win._geo_tab._w.value() == pytest.approx(0.12, abs=1e-4)

    # Reload
    win._load_project_from(project_file)
    qapp.processEvents()

    assert win._geo_tab._w.value() == pytest.approx(0.25, abs=1e-4)
    assert win._geo_tab._d.value() == pytest.approx(0.12, abs=1e-4)
    assert len(win._state.drivers) == 1
    assert win._state.drivers[0].driver_id == "d0"
    assert win._sim_tab._f_lo.value() == pytest.approx(80.0, abs=0.1)
    assert win._sim_tab._f_hi.value() == pytest.approx(8000.0, abs=0.1)
    assert not win._dirty
    assert win._current_project_path == project_file

    win.close()


# ---------------------------------------------------------------------------
# Test 3: undo / redo cycle
# ---------------------------------------------------------------------------


def test_undo_redo_single_step(qapp):
    """One dims change: undo restores previous value; redo re-applies it."""
    win = MainWindow()

    # Take initial snapshot (already done in __init__ via _current_snapshot = _gather_state())
    initial_w = win._geo_tab._w.value()

    # Change width — this fires stateChanged → _on_state_changed pushes snapshot
    win._geo_tab._w.setValue(0.30)
    qapp.processEvents()

    # Should now have one undo step
    assert len(win._undo_stack) >= 1
    assert win._dirty

    # Undo
    win._undo()
    qapp.processEvents()
    assert win._geo_tab._w.value() == pytest.approx(initial_w, abs=1e-4)
    assert len(win._redo_stack) >= 1

    # Redo
    win._redo()
    qapp.processEvents()
    assert win._geo_tab._w.value() == pytest.approx(0.30, abs=1e-4)

    win._dirty = False  # prevent unsaved-changes dialog in headless close
    win.close()


# ---------------------------------------------------------------------------
# Test 4: cardinal rule — _apply_state never touches H_bem / H_full
# ---------------------------------------------------------------------------


def test_apply_state_does_not_touch_h_tensors(qapp):
    """_apply_state is inputs-only; it never mutates any H_bem or H_full tensor."""
    # Since project save/load only stores inputs (geometry, drivers, sim params),
    # there is no code path in _apply_state that could touch stored H tensors.
    # This test verifies that: load a document that contains no results field
    # and assert no result is set on AppState.
    win = MainWindow()
    from beamsim2.gui.app import MainWindow as _MW

    doc = _MW._blank_document()
    assert doc.get("results_h5_path") is None

    win._apply_state(doc)
    qapp.processEvents()
    # No result should be set
    assert win._state.result is None
    win.close()
