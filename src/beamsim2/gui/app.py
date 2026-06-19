"""BeamSimII desktop application: main window, AppState, and Qt worker thread.

DR-04 mandates PySide6 (LGPL, first-class Apple Silicon support).

The core (pipeline/, assembly/, backends/, …) never imports Qt.  Only this
module and the four view modules depend on PySide6.  The GUI is a thin shell
over the headless pipeline (pipeline/run.py).

Worker-thread pattern
---------------------
Long solves (minutes to days) run on a ``SolveWorker`` via
``QThread.moveToThread``.  The worker calls ``run_simulation(progress=…)`` and
emits ``progressChanged`` (a ``ProgressSnapshot``) and ``finished``
(a ``SimulationResult``) — cross-thread queued connections marshal data to the
GUI thread.  The core never imports Qt; the single bridge is
``progress.subscribe(self.progressChanged.emit)``.

Build-order item 10 (DR-04, §6 Gameplan).
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtCore import QObject, QThread, Signal, Slot
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QStatusBar,
    QTabWidget,
    QWidget,
)

from beamsim2.backends.numcalc.config import _write_numcalc_config, resolve_numcalc_binary
from beamsim2.core.types import FrequencyGrid, SolverConfig
from beamsim2.pipeline.progress import ProgressModel
from beamsim2.pipeline.run import (
    BoxGeometry,
    DriverPlacement,
    SimulationRequest,
    SimulationResult,
    run_simulation,
)

# ---------------------------------------------------------------------------
# Application state (Qt-free)
# ---------------------------------------------------------------------------


@dataclass
class AppState:
    """Mutable application state shared across all tabs.

    Tabs read and write this object; the main window subscribes to tab signals
    that indicate state changes so it can gate controls (e.g. Run is disabled
    until health is green and ≥1 driver exists).
    """

    geometry: Optional[BoxGeometry] = None
    drivers: list[DriverPlacement] = field(default_factory=list)
    frequencies: Optional[FrequencyGrid] = None
    sphere_n_points: int = 26
    sphere_radius: float = 1.0
    config: SolverConfig = field(default_factory=SolverConfig)
    result: Optional[SimulationResult] = None
    h5_path: Optional[Path] = None  # last opened or saved dataset


# ---------------------------------------------------------------------------
# Background solve worker
# ---------------------------------------------------------------------------


class SolveWorker(QObject):
    """Runs ``run_simulation`` on a background QThread.

    Signals
    -------
    progressChanged : ProgressSnapshot
        Emitted for every ``ProgressModel`` state change.  Connected to
        the run-monitor widget via a queued (cross-thread) connection.
    finished : SimulationResult
        Emitted when the solve completes successfully.
    failed : str
        Emitted with an error string if an exception is raised.
    """

    progressChanged = Signal(object)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, req: SimulationRequest) -> None:
        super().__init__()
        self._req = req

    @Slot()
    def run(self) -> None:
        """Entry point on the background thread.  Never call directly."""
        n_drivers = len(self._req.drivers)
        n_freq = len(self._req.frequencies.frequencies)
        driver_ids = [dp.driver_id for dp in self._req.drivers]

        progress = ProgressModel(
            n_drivers=n_drivers,
            n_freq=n_freq,
            driver_ids=driver_ids,
        )
        # Single Qt bridge: the Qt-free ProgressModel calls this subscriber,
        # which emits a queued signal — marshalled to the GUI thread automatically.
        progress.subscribe(self.progressChanged.emit)

        try:
            result = run_simulation(self._req, progress=progress)
            self.finished.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------


class MainWindow(QMainWindow):
    """Four-tab PySide6 main window.

    Tab order matches §6 screen flow:
      0 — Geometry   (BoxGeometry builder + 3-D mesh preview)
      1 — Drivers    (T/S parameter entry, driver list)
      2 — Simulation (frequency, sphere, Estimate/Run, progress monitor)
      3 — Results    (on-axis, polar, balloon, directivity map, Export)
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("BeamSimII — Acoustic Radiation Simulator")
        self.resize(1100, 780)

        self._state = AppState()
        self._thread: Optional[QThread] = None
        self._worker: Optional[SolveWorker] = None

        self._tabs = QTabWidget()
        self.setCentralWidget(self._tabs)

        # Lazy import to avoid circular dependency at module level
        from beamsim2.gui.geometry_view import GeometryTab
        from beamsim2.gui.parameters_panel import DriversTab, SimulationTab
        from beamsim2.gui.results_view import ResultsTab

        self._geo_tab = GeometryTab(self._state)
        self._drv_tab = DriversTab(self._state)
        self._sim_tab = SimulationTab(self._state)
        self._res_tab = ResultsTab(self._state)

        self._tabs.addTab(self._geo_tab, "Geometry")
        self._tabs.addTab(self._drv_tab, "Drivers")
        self._tabs.addTab(self._sim_tab, "Simulation")
        self._tabs.addTab(self._res_tab, "Results")

        # Wire cross-tab signals
        self._geo_tab.geometryChanged.connect(self._on_geometry_changed)
        self._drv_tab.driversChanged.connect(self._on_drivers_changed)
        self._sim_tab.runRequested.connect(self._on_run_requested)
        self._sim_tab.estimateRequested.connect(self._on_estimate_requested)

        # Cross-tab driver sync: canvas edits update the Drivers list tab and vice versa
        self._geo_tab.driversChanged.connect(self._drv_tab.refresh)
        self._geo_tab.driversChanged.connect(self._on_drivers_changed)
        self._drv_tab.driversChanged.connect(self._geo_tab.refresh_canvas)

        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status.showMessage("Ready")

        self._build_menu()
        self._refresh_run_enabled()

    # ------------------------------------------------------------------
    # Menu bar
    # ------------------------------------------------------------------

    def _build_menu(self) -> None:
        mb = self.menuBar()
        file_menu = mb.addMenu("File")

        open_act = file_menu.addAction("Open dataset…")
        open_act.triggered.connect(self._open_dataset)

        file_menu.addSeparator()
        quit_act = file_menu.addAction("Quit")
        quit_act.triggered.connect(self.close)

        help_menu = mb.addMenu("Help")
        about_act = help_menu.addAction("About BeamSimII")
        about_act.triggered.connect(self._about)

    # ------------------------------------------------------------------
    # Cross-tab slots
    # ------------------------------------------------------------------

    def _on_geometry_changed(self) -> None:
        self._refresh_run_enabled()

    def _on_drivers_changed(self) -> None:
        self._refresh_run_enabled()

    def _on_estimate_requested(self) -> None:
        """Run estimate_resources and display results in the status bar."""
        if not self._sim_tab.build_request(self._state):
            return
        req = self._sim_tab.current_request
        if req is None:
            return
        from beamsim2.pipeline.run import estimate_resources

        try:
            est = estimate_resources(req)
            gb = est.peak_ram_bytes / 1e9
            mins = est.total_wall_seconds / 60
            msg = (
                f"Estimate: peak RAM ≈ {gb:.1f} GB, "
                f"wall-clock ≈ {mins:.0f} min (approx.), "
                f"{est.n_steps_total} solver steps"
            )
            self._status.showMessage(msg, 10000)
            self._sim_tab.show_estimate(est)
        except Exception as exc:
            QMessageBox.warning(self, "Estimate failed", str(exc))

    def _on_run_requested(self) -> None:
        """Validate inputs, build SimulationRequest, launch the solve worker."""
        if not self._sim_tab.build_request(self._state):
            return
        req = self._sim_tab.current_request
        if req is None:
            return

        self._thread = QThread()
        self._worker = SolveWorker(req)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.progressChanged.connect(self._sim_tab.monitor.update_snapshot)
        self._worker.finished.connect(self._on_solve_finished)
        self._worker.failed.connect(self._on_solve_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)

        self._sim_tab.set_running(True)
        self._thread.start()
        self._status.showMessage("Solve running…")

    def _on_solve_finished(self, result: SimulationResult) -> None:
        self._state.result = result
        if result.h5_path:
            self._state.h5_path = result.h5_path
        self._sim_tab.set_running(False)
        self._res_tab.load(result.dataset)
        self._tabs.setCurrentIndex(3)  # jump to Results
        flagged = sum(int(np.any(v)) for v in result.flagged_frequencies.values())
        msg = "Solve complete"
        if flagged:
            msg += f" — {flagged} driver(s) have flagged (non-converged) frequencies"
        self._status.showMessage(msg)

    def _on_solve_failed(self, err: str) -> None:
        self._sim_tab.set_running(False)
        QMessageBox.critical(self, "Solve failed", err)
        self._status.showMessage("Solve failed")

    def _refresh_run_enabled(self) -> None:
        ok = self._state.geometry is not None and len(self._state.drivers) > 0
        self._sim_tab.set_run_enabled(ok)

    # ------------------------------------------------------------------
    # File menu actions
    # ------------------------------------------------------------------

    def _open_dataset(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open BeamSimII dataset",
            "",
            "HDF5 datasets (*.h5 *.bsim);;All files (*)",
        )
        if not path:
            return
        try:
            from beamsim2.io.hdf5_store import read_dataset

            ds = read_dataset(path)
            self._state.h5_path = Path(path)
            self._state.result = None
            self._res_tab.load(ds)
            self._tabs.setCurrentIndex(3)
            self._status.showMessage(f"Loaded {Path(path).name}")
        except Exception as exc:
            QMessageBox.critical(self, "Open failed", str(exc))

    def _about(self) -> None:
        QMessageBox.about(
            self,
            "About BeamSimII",
            "BeamSimII — Loudspeaker BEM Radiation Simulator\n"
            "Phase 1: Full-sphere directivity via NumCalc (Mesh2HRTF)\n\n"
            "Build-order item 10: GUI (DR-04 / Gameplan §6)",
        )


# ---------------------------------------------------------------------------
# Application entry point
# ---------------------------------------------------------------------------


def _prompt_numcalc_binary() -> None:
    """One-time file-picker dialog to locate NumCalc when it is not configured.

    Called from ``main()`` before ``MainWindow`` is constructed.  Writes the
    chosen path to ``~/.config/beamsim2/settings.toml`` permanently.  If the
    user cancels, the main window still opens — they can browse geometry and
    load saved datasets without running a solve.
    """
    try:
        resolve_numcalc_binary()
        return  # already configured — nothing to do
    except FileNotFoundError:
        pass

    QMessageBox.information(
        None,
        "NumCalc Binary Not Found",
        "BeamSimII needs the NumCalc binary to run BEM simulations.\n\n"
        "Please locate the NumCalc executable in the next dialog.\n"
        "Your choice will be saved — you won't be asked again.\n\n"
        "(Cancel to open BeamSimII without simulation capability.)",
    )

    path, _ = QFileDialog.getOpenFileName(
        None,
        "Locate NumCalc Binary",
        os.path.expanduser("~"),
        "NumCalc binary (NumCalc);;All files (*)",
    )

    if not path:
        return  # user cancelled; main window still shows

    try:
        resolve_numcalc_binary(path)  # validate the selection exists on disk
    except FileNotFoundError:
        QMessageBox.warning(
            None,
            "Invalid Selection",
            f"File not found:\n{path}\n\nPlease relaunch BeamSimII and try again.",
        )
        return

    _write_numcalc_config(path)
    # Belt-and-suspenders: set env var so current session works immediately
    # without re-reading the config file from disk.
    os.environ["BEAMSIM2_NUMCALC_BIN"] = path


def main() -> None:
    """Launch the BeamSimII desktop application."""
    app = QApplication.instance() or QApplication(sys.argv)
    _prompt_numcalc_binary()  # one-time binary location prompt
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
