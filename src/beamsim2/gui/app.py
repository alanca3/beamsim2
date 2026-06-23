"""BeamSimII desktop application: main window, AppState, and Qt worker thread.

DR-04 mandates PySide6 (LGPL, first-class Apple Silicon support).

The core (pipeline/, assembly/, backends/, …) never imports Qt.  Only this
module and the five view modules depend on PySide6.  The GUI is a thin shell
over the headless pipeline (pipeline/run.py).

Worker-thread pattern
---------------------
Long solves (minutes to days) run on a ``SolveWorker`` via
``QThread.moveToThread``.  The worker calls ``run_simulation(progress=…)`` and
emits ``progressChanged`` (a ``ProgressSnapshot``) and ``finished``
(a ``SimulationResult``) — cross-thread queued connections marshal data to the
GUI thread.  The core never imports Qt; the single bridge is
``progress.subscribe(self.progressChanged.emit)``.

Project system (App-Shell Chunk, v1.5.0)
-----------------------------------------
A ``.bsim`` JSON file stores all *input* state (box, drivers, sim params,
solver config, optional results-HDF5 path) via ``io.project_io``.  Authoritative
state lives partly in AppState and partly in tab widgets (GeometryTab spin-boxes,
SimulationTab combos); a single ``_gather_state`` / ``_apply_state`` pair
collects / distributes both sources, powering project save/load *and* undo/redo.

Build-order item 10 (DR-04, §6 Gameplan); App-Shell Chunk (Bug_Fix_Proposal.md §5).
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtCore import QObject, QThread, Signal, Slot
from PySide6.QtGui import QAction, QActionGroup, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QMainWindow,
    QMenu,
    QMessageBox,
    QStatusBar,
    QTabWidget,
    QWidget,
)

from beamsim2.backends.numcalc.config import _write_numcalc_config, resolve_numcalc_binary
from beamsim2.core.types import FrequencyGrid, SolverConfig
from beamsim2.io.project_io import (
    PROJECT_SCHEMA,
    PROJECT_VERSION,
    document_from_json,
    document_to_json,
    driver_from_dict,
    driver_to_dict,
)
from beamsim2.pipeline.progress import ProgressModel
from beamsim2.pipeline.run import (
    BoxGeometry,
    DriverPlacement,
    SimulationRequest,
    SimulationResult,
    run_simulation,
)

# Maximum undo history depth.
_UNDO_CAP = 50

# ---------------------------------------------------------------------------
# Application state (Qt-free)
# ---------------------------------------------------------------------------


@dataclass
class AppState:
    """Mutable application state shared across all tabs.

    Tabs read and write this object; the main window subscribes to tab signals
    that indicate state changes so it can gate controls (e.g. Run is disabled
    until health is green and ≥1 driver exists).

    Notes
    -----
    ``frequencies``, ``sphere_n_points``, and ``sphere_radius`` are carried for
    forward-compatibility but are **never written by the GUI** — the authoritative
    values live in SimulationTab's widget state and are assembled on demand in
    ``SimulationTab.build_request``.  Do not rely on them for project save/load.
    """

    geometry: Optional[BoxGeometry] = None
    drivers: list[DriverPlacement] = field(default_factory=list)
    frequencies: Optional[FrequencyGrid] = None
    sphere_n_points: int = 26
    sphere_radius: float = 1.0
    config: SolverConfig = field(default_factory=SolverConfig)
    result: Optional[SimulationResult] = None
    h5_path: Optional[Path] = None  # last opened or saved dataset

    # Live box dimensions (x, y, z) in metres — the single dims source both driver
    # editors read so a re-orient reconciles identically whether it came from the
    # canvas right-click or the Drivers-list "Edit" button.  Mirrors the Geometry
    # tab's spin-boxes (which remain the interactive truth) and defaults to them.
    box_dims: tuple[float, float, float] = (0.12, 0.10, 0.08)

    # Measurement / 0°-on-axis reference direction (loudspeaker front) in the global
    # Cartesian frame.  Default +z.  Shown as the reference-axis + virtual-mic glyph in
    # the 3-D editor (Bug #1) and threaded into SimulationRequest so the solved dataset's
    # ``reference_axis`` attr — and every Results view — agrees with what the editor shows.
    # Display/metadata only: never moves the geometry or phase origin (cardinal rule).
    reference_axis: tuple[float, float, float] = (0.0, 0.0, 1.0)


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
    """Five-tab PySide6 main window.

    Tab order matches §6 screen flow:
      0 — Geometry        (BoxGeometry builder + 3-D mesh preview)
      1 — Drivers         (T/S parameter entry, driver list)
      2 — Simulation      (frequency, sphere, Estimate/Run, progress monitor)
      3 — Results         (on-axis, polar, balloon, directivity map, Export)
      4 — Filter Designer (beamforming filter design)

    Project system
    --------------
    ``_gather_state()`` snapshots all editable inputs (AppState + tab-widget
    values) into a dict matching the ``.bsim`` schema.  ``_apply_state(doc)``
    distributes a dict back to AppState and tab widgets (blocking signals to
    prevent cascading captures), then triggers an explicit canvas refresh.
    Both methods are shared by project save/load and undo/redo.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("BeamSimII — Acoustic Radiation Simulator")
        self.resize(1100, 780)

        self._state = AppState()
        self._thread: Optional[QThread] = None
        self._worker: Optional[SolveWorker] = None

        # Project bookkeeping
        self._current_project_path: Optional[Path] = None
        self._dirty: bool = False
        self._applying: bool = False  # guard: suppress undo capture during _apply_state

        # Undo / redo stacks (dicts matching the .bsim schema)
        self._undo_stack: list[dict] = []
        self._redo_stack: list[dict] = []
        self._current_snapshot: Optional[dict] = None

        self._tabs = QTabWidget()
        self.setCentralWidget(self._tabs)

        # Lazy import to avoid circular dependency at module level
        from beamsim2.gui.filter_designer_view import FilterDesignerTab
        from beamsim2.gui.geometry_view import GeometryTab
        from beamsim2.gui.parameters_panel import DriversTab, SimulationTab
        from beamsim2.gui.results_view import ResultsTab

        self._geo_tab = GeometryTab(self._state)
        self._drv_tab = DriversTab(self._state)
        self._sim_tab = SimulationTab(self._state)
        self._res_tab = ResultsTab(self._state)
        self._fd_tab = FilterDesignerTab(self._state)

        self._tabs.addTab(self._geo_tab, "Geometry")
        self._tabs.addTab(self._drv_tab, "Drivers")
        self._tabs.addTab(self._sim_tab, "Simulation")
        self._tabs.addTab(self._res_tab, "Results")
        self._tabs.addTab(self._fd_tab, "Filter Designer")

        # Wire cross-tab signals → _on_state_changed (undo capture + run-enable refresh)
        self._geo_tab.stateChanged.connect(self._on_state_changed)
        self._geo_tab.geometryChanged.connect(self._on_state_changed)
        self._geo_tab.driversChanged.connect(self._on_state_changed)
        self._drv_tab.driversChanged.connect(self._on_state_changed)
        self._sim_tab.stateChanged.connect(self._on_state_changed)

        # Cross-tab driver sync: canvas edits update the Drivers list tab and vice versa
        self._geo_tab.driversChanged.connect(self._drv_tab.refresh)
        self._drv_tab.driversChanged.connect(self._geo_tab.refresh_canvas)

        # Solve signals
        self._sim_tab.runRequested.connect(self._on_run_requested)
        self._sim_tab.estimateRequested.connect(self._on_estimate_requested)

        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status.showMessage("Ready")

        self._build_menu()
        self._refresh_run_enabled()

        # Seed the initial undo snapshot (empty project)
        self._current_snapshot = self._gather_state()

    # ------------------------------------------------------------------
    # Menu bar
    # ------------------------------------------------------------------

    def _build_menu(self) -> None:
        """Build the full menu bar (File / Edit / View / Settings / Help)."""
        mb = self.menuBar()

        # ── File ──────────────────────────────────────────────────────────
        file_menu = mb.addMenu("&File")

        new_act = QAction("&New", self)
        new_act.setShortcut(QKeySequence.StandardKey.New)
        new_act.setStatusTip("Start a new empty project")
        new_act.triggered.connect(self._new_project)
        file_menu.addAction(new_act)

        open_act = QAction("&Open Project…", self)
        open_act.setShortcut(QKeySequence.StandardKey.Open)
        open_act.setStatusTip("Open a .bsim project file")
        open_act.triggered.connect(self._open_project)
        file_menu.addAction(open_act)

        self._save_act = QAction("&Save", self)
        self._save_act.setShortcut(QKeySequence.StandardKey.Save)
        self._save_act.setStatusTip("Save project")
        self._save_act.triggered.connect(self._save_project)
        file_menu.addAction(self._save_act)

        saveas_act = QAction("Save &As…", self)
        saveas_act.setShortcut(QKeySequence("Ctrl+Shift+S"))
        saveas_act.setStatusTip("Save project to a new file")
        saveas_act.triggered.connect(self._save_project_as)
        file_menu.addAction(saveas_act)

        # Recent Projects submenu
        self._recent_menu = QMenu("Recent Projects", self)
        file_menu.addMenu(self._recent_menu)
        self._recent_menu.aboutToShow.connect(self._rebuild_recent_menu)

        file_menu.addSeparator()

        open_ds_act = QAction("Open &Dataset…", self)
        open_ds_act.setStatusTip("Open an existing HDF5 radiation dataset (.h5)")
        open_ds_act.triggered.connect(self._open_dataset)
        file_menu.addAction(open_ds_act)

        file_menu.addSeparator()

        quit_act = QAction("&Quit", self)
        quit_act.setShortcut(QKeySequence.StandardKey.Quit)
        quit_act.triggered.connect(self.close)
        file_menu.addAction(quit_act)

        # ── Edit ──────────────────────────────────────────────────────────
        edit_menu = mb.addMenu("&Edit")

        self._undo_act = QAction("&Undo", self)
        self._undo_act.setShortcut(QKeySequence.StandardKey.Undo)
        self._undo_act.setEnabled(False)
        self._undo_act.triggered.connect(self._undo)
        edit_menu.addAction(self._undo_act)

        self._redo_act = QAction("&Redo", self)
        self._redo_act.setShortcut(QKeySequence.StandardKey.Redo)
        self._redo_act.setEnabled(False)
        self._redo_act.triggered.connect(self._redo)
        edit_menu.addAction(self._redo_act)

        # ── View ──────────────────────────────────────────────────────────
        view_menu = mb.addMenu("&View")

        reset_view_act = QAction("&Reset View", self)
        reset_view_act.setShortcut(QKeySequence("R"))
        reset_view_act.setStatusTip("Reset the 3-D camera to fit the model")
        reset_view_act.triggered.connect(self._geo_tab.reset_view)
        view_menu.addAction(reset_view_act)

        view_menu.addSeparator()

        # Projection toggle — two checkable actions in an exclusive group
        proj_group = QActionGroup(self)
        proj_group.setExclusive(True)

        self._persp_act = QAction("&Perspective", self)
        self._persp_act.setCheckable(True)
        self._persp_act.setChecked(True)
        self._persp_act.triggered.connect(lambda: self._geo_tab.set_parallel_projection(False))
        proj_group.addAction(self._persp_act)
        view_menu.addAction(self._persp_act)

        self._ortho_act = QAction("&Orthographic", self)
        self._ortho_act.setCheckable(True)
        self._ortho_act.setChecked(False)
        self._ortho_act.triggered.connect(lambda: self._geo_tab.set_parallel_projection(True))
        proj_group.addAction(self._ortho_act)
        view_menu.addAction(self._ortho_act)

        view_menu.addSeparator()

        # Preset views
        _preset_views = [
            ("&Front", "front", "F"),
            ("&Back", "back", ""),
            ("&Left", "left", ""),
            ("&Right", "right", ""),
            ("&Top", "top", ""),
            ("B&ottom", "bottom", ""),
            ("&Isometric", "isometric", "I"),
        ]
        for label, name, shortcut in _preset_views:
            act = QAction(label, self)
            if shortcut:
                act.setShortcut(QKeySequence(shortcut))
            act.triggered.connect(lambda checked=False, n=name: self._geo_tab.set_view(n))
            view_menu.addAction(act)

        # ── Settings ──────────────────────────────────────────────────────
        settings_menu = mb.addMenu("&Settings")

        prefs_act = QAction("&Preferences…", self)
        prefs_act.setShortcut(QKeySequence("Ctrl+,"))
        prefs_act.setStatusTip("Open the Preferences dialog")
        prefs_act.triggered.connect(self._open_preferences)
        settings_menu.addAction(prefs_act)

        # ── Help ──────────────────────────────────────────────────────────
        help_menu = mb.addMenu("&Help")
        about_act = QAction("About BeamSimII", self)
        about_act.triggered.connect(self._about)
        help_menu.addAction(about_act)

    # ------------------------------------------------------------------
    # State change capture (undo / dirty tracking)
    # ------------------------------------------------------------------

    def _on_state_changed(self) -> None:
        """Called by any user-visible state change signal.

        Pushes the previous snapshot onto the undo stack, recomputes the
        current snapshot, marks the project dirty, and refreshes gated controls.
        """
        if self._applying:
            return

        # Push current snapshot as an undo step
        if self._current_snapshot is not None:
            self._undo_stack.append(self._current_snapshot)
            if len(self._undo_stack) > _UNDO_CAP:
                self._undo_stack.pop(0)

        self._redo_stack.clear()
        self._current_snapshot = self._gather_state()
        self._dirty = True
        self._update_window_title()
        self._update_edit_actions()
        self._refresh_run_enabled()

    # ------------------------------------------------------------------
    # Undo / redo
    # ------------------------------------------------------------------

    def _undo(self) -> None:
        """Undo the last user action by restoring the previous snapshot."""
        if not self._undo_stack:
            return
        if self._current_snapshot is not None:
            self._redo_stack.append(self._current_snapshot)
        self._current_snapshot = self._undo_stack.pop()
        self._apply_state(self._current_snapshot)
        self._update_edit_actions()

    def _redo(self) -> None:
        """Redo the last undone action."""
        if not self._redo_stack:
            return
        if self._current_snapshot is not None:
            self._undo_stack.append(self._current_snapshot)
        self._current_snapshot = self._redo_stack.pop()
        self._apply_state(self._current_snapshot)
        self._update_edit_actions()

    # ------------------------------------------------------------------
    # Project gather / apply (shared by save/load and undo/redo)
    # ------------------------------------------------------------------

    def _gather_state(self) -> dict:
        """Snapshot the full editable application state into a ``.bsim`` dict.

        Reads from AppState AND from tab-widget values (GeometryTab spin-boxes,
        SimulationTab combos), since some authoritative values live only in the
        widgets (see class docstring of AppState for details).

        Returns
        -------
        dict
            A project document dict as defined by ``io.project_io`` schema v1.
        """
        geo_params = self._geo_tab.get_project_params()
        sim_params = self._sim_tab.get_project_params()

        cfg = self._state.config
        solver_config: dict = {
            "n_epw": cfg.n_epw,
            "tolerance": cfg.tolerance,
            "max_iterations": cfg.max_iterations,
            "burton_miller": cfg.burton_miller,
            "speed_of_sound": cfg.speed_of_sound,
            "air_density": cfg.air_density,
            "air_attenuation_model": cfg.air_attenuation_model,
        }

        return {
            "schema": PROJECT_SCHEMA,
            "project_version": PROJECT_VERSION,
            "box": {
                "width": geo_params["width"],
                "height": geo_params["height"],
                "depth": geo_params["depth"],
                "fillet_radius": geo_params["fillet_radius"],
            },
            "reference_axis": geo_params["reference_axis"],
            "drivers": [driver_to_dict(dp) for dp in self._state.drivers],
            "simulation": sim_params,
            "solver_config": solver_config,
            "results_h5_path": (str(self._state.h5_path) if self._state.h5_path else None),
        }

    def _apply_state(self, doc: dict) -> None:
        """Distribute a ``.bsim`` document dict back to AppState and tab widgets.

        Load order:
          1. Geometry (dims + fillet + ref-axis): signals blocked.
          2. Drivers: populate AppState list (after dims, so specs are valid).
          3. Simulation params: signals blocked.
          4. Solver config + h5 path.
          5. Explicit canvas + driver-list refresh.

        After load, ``state.geometry is None`` (same as fresh state), so the Run
        button stays disabled until the user clicks "Preview mesh" — this is
        intentional and matches normal fresh-project behaviour.

        Parameters
        ----------
        doc : dict
            A project document dict produced by ``_gather_state`` or loaded from
            a ``.bsim`` file via ``document_from_json``.
        """
        self._applying = True
        try:
            # 1. Box geometry (dims, fillet, reference axis)
            box = doc.get("box", {})
            geo_params = {
                "width": box.get("width", 0.12),
                "height": box.get("height", 0.10),
                "depth": box.get("depth", 0.08),
                "fillet_radius": box.get("fillet_radius", 0.0),
                "reference_axis": doc.get("reference_axis", [0.0, 0.0, 1.0]),
            }
            self._geo_tab.apply_project_params(geo_params)

            # 2. Drivers (after dims so spec.center/normal are derived correctly)
            self._state.drivers.clear()
            for d in doc.get("drivers", []):
                self._state.drivers.append(driver_from_dict(d))

            # 3. Simulation parameters
            sim_params = doc.get("simulation", {})
            if sim_params:
                self._sim_tab.apply_project_params(sim_params)

            # 4. Solver config
            cfg_d = doc.get("solver_config", {})
            self._state.config = SolverConfig(
                n_epw=int(cfg_d.get("n_epw", 6)),
                tolerance=float(cfg_d.get("tolerance", 1e-6)),
                max_iterations=int(cfg_d.get("max_iterations", 1000)),
                burton_miller=bool(cfg_d.get("burton_miller", True)),
                speed_of_sound=float(cfg_d.get("speed_of_sound", 343.2)),
                air_density=float(cfg_d.get("air_density", 1.2041)),
                air_attenuation_model=str(cfg_d.get("air_attenuation_model", "none")),
            )

            # 5. Results path
            rh5 = doc.get("results_h5_path")
            self._state.h5_path = Path(rh5) if rh5 else None

        finally:
            self._applying = False

        # 6. Explicit refresh — signals were blocked so no auto-refresh fired
        self._geo_tab.refresh_canvas()
        self._drv_tab.refresh()
        self._refresh_run_enabled()

    # ------------------------------------------------------------------
    # Project operations
    # ------------------------------------------------------------------

    def _new_project(self) -> None:
        """Reset to a blank project (with unsaved-changes guard)."""
        if not self._maybe_discard():
            return
        self._apply_state(self._blank_document())
        self._current_project_path = None
        self._dirty = False
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._current_snapshot = self._gather_state()
        self._update_window_title()
        self._update_edit_actions()
        self._status.showMessage("New project")

    def _open_project(self) -> None:
        """Open a .bsim project file (with unsaved-changes guard)."""
        if not self._maybe_discard():
            return
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open BeamSimII Project",
            "",
            "BeamSimII projects (*.bsim);;All files (*)",
        )
        if not path:
            return
        self._load_project_from(Path(path))

    def _load_project_from(self, path: Path) -> None:
        """Load a project from *path*, with error handling and recent-list update."""
        try:
            doc = document_from_json(path)
        except Exception as exc:
            QMessageBox.critical(
                self, "Open failed", f"Could not open project:\n{path.name}\n\n{exc}"
            )
            return

        self._apply_state(doc)
        self._current_project_path = path
        self._dirty = False
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._current_snapshot = self._gather_state()
        self._update_window_title()
        self._update_edit_actions()

        # Push to recent list
        from beamsim2.backends.numcalc.config import push_recent_project

        push_recent_project(path)

        self._status.showMessage(f"Opened {path.name}")

        # If results h5 exists on disk, offer to load it
        rh5 = self._state.h5_path
        if rh5 and rh5.is_file():
            try:
                from beamsim2.io.hdf5_store import read_dataset

                ds = read_dataset(str(rh5))
                self._res_tab.load(ds)
                self._fd_tab.load(ds)
                self._tabs.setCurrentIndex(3)
                self._status.showMessage(f"Opened {path.name} (results loaded)")
            except Exception:
                pass  # don't block project load on a stale h5 reference

    def _save_project(self) -> None:
        """Save the project, prompting for a file name if not yet saved."""
        if self._current_project_path is None:
            self._save_project_as()
        else:
            self._save_to(self._current_project_path)

    def _save_project_as(self) -> None:
        """Prompt for a file name and save."""
        default = str(self._current_project_path) if self._current_project_path else ""
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save BeamSimII Project",
            default,
            "BeamSimII projects (*.bsim);;All files (*)",
        )
        if not path:
            return
        p = Path(path)
        if p.suffix.lower() != ".bsim":
            p = p.with_suffix(".bsim")
        self._save_to(p)

    def _save_to(self, path: Path) -> None:
        """Write the current state to *path* and record it as the current project."""
        try:
            doc = self._gather_state()
            document_to_json(doc, path)
        except Exception as exc:
            QMessageBox.critical(
                self, "Save failed", f"Could not save project:\n{path.name}\n\n{exc}"
            )
            return

        self._current_project_path = path
        self._dirty = False
        self._current_snapshot = self._gather_state()
        self._update_window_title()

        from beamsim2.backends.numcalc.config import push_recent_project

        push_recent_project(path)

        self._status.showMessage(f"Saved {path.name}")

    # ------------------------------------------------------------------
    # Recent projects menu
    # ------------------------------------------------------------------

    def _rebuild_recent_menu(self) -> None:
        """Repopulate the Recent Projects submenu from the settings store."""
        from beamsim2.backends.numcalc.config import read_recent_projects

        self._recent_menu.clear()
        recent = read_recent_projects()
        if not recent:
            no_act = QAction("(none)", self)
            no_act.setEnabled(False)
            self._recent_menu.addAction(no_act)
            return
        for path_str in recent:
            p = Path(path_str)
            act = QAction(p.name, self)
            act.setStatusTip(path_str)
            act.setToolTip(path_str)
            act.triggered.connect(lambda checked=False, ps=path_str: self._open_recent(ps))
            self._recent_menu.addAction(act)

    def _open_recent(self, path_str: str) -> None:
        """Load a project from the recent list (with unsaved-changes guard)."""
        if not self._maybe_discard():
            return
        p = Path(path_str)
        if not p.is_file():
            QMessageBox.warning(
                self,
                "File not found",
                f"The project file no longer exists:\n{path_str}",
            )
            return
        self._load_project_from(p)

    # ------------------------------------------------------------------
    # Unsaved-changes guard
    # ------------------------------------------------------------------

    def _maybe_discard(self) -> bool:
        """Ask the user whether to save unsaved changes.

        Returns
        -------
        bool
            True if the caller should proceed (saved or discarded); False if
            the user chose Cancel and the operation should be aborted.
        """
        if not self._dirty:
            return True

        name = self._current_project_path.name if self._current_project_path else "Untitled"
        btn = QMessageBox.question(
            self,
            "Unsaved changes",
            f'The project "{name}" has unsaved changes.\n\nSave before continuing?',
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )
        if btn == QMessageBox.StandardButton.Cancel:
            return False
        if btn == QMessageBox.StandardButton.Save:
            self._save_project()
            # If save was aborted (e.g. user cancelled Save As), _dirty is still True
            if self._dirty:
                return False
        return True

    def closeEvent(self, event) -> None:  # type: ignore[override]
        """Intercept close to guard against unsaved changes."""
        if self._maybe_discard():
            event.accept()
        else:
            event.ignore()

    # ------------------------------------------------------------------
    # Window title / action state helpers
    # ------------------------------------------------------------------

    def _update_window_title(self) -> None:
        """Reflect project name and dirty flag in the window title."""
        name = self._current_project_path.stem if self._current_project_path else "Untitled"
        dirty_marker = " *" if self._dirty else ""
        self.setWindowTitle(f"BeamSimII — {name}{dirty_marker}")

    def _update_edit_actions(self) -> None:
        """Enable / disable Undo and Redo based on stack depths."""
        self._undo_act.setEnabled(bool(self._undo_stack))
        self._redo_act.setEnabled(bool(self._redo_stack))
        # Update undo/redo text to hint at what will be undone/redone
        self._undo_act.setText(f"&Undo ({len(self._undo_stack)} steps)")
        self._redo_act.setText(f"&Redo ({len(self._redo_stack)} steps)")

    # ------------------------------------------------------------------
    # Cross-tab slots
    # ------------------------------------------------------------------

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
        self._fd_tab.load(result.dataset)
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
        """Open an existing HDF5 radiation dataset (not a .bsim project file)."""
        if not self._maybe_discard():
            return
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open BeamSimII dataset",
            "",
            "HDF5 datasets (*.h5);;All files (*)",
        )
        if not path:
            return
        try:
            from beamsim2.io.hdf5_store import read_dataset

            ds = read_dataset(path)
            self._state.h5_path = Path(path)
            self._state.result = None
            self._res_tab.load(ds)
            self._fd_tab.load(ds)
            self._tabs.setCurrentIndex(3)
            self._status.showMessage(f"Loaded {Path(path).name}")
        except Exception as exc:
            QMessageBox.critical(self, "Open failed", str(exc))

    def _open_preferences(self) -> None:
        """Open the Preferences dialog."""
        from beamsim2.gui.preferences_dialog import PreferencesDialog

        dlg = PreferencesDialog(parent=self)
        dlg.exec()

    def _about(self) -> None:
        QMessageBox.about(
            self,
            "About BeamSimII",
            "BeamSimII — Loudspeaker BEM Radiation Simulator\n"
            "Phase 1: Full-sphere directivity via NumCalc (Mesh2HRTF)\n\n"
            "App-Shell Chunk v1.5.0: project save/load, undo/redo,\n"
            "view manager, full menu bar, GUI logging.\n\n"
            "Build-order item 10: GUI (DR-04 / Gameplan §6)",
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _blank_document() -> dict:
        """Return the schema-header dict for an empty project."""
        return {
            "schema": PROJECT_SCHEMA,
            "project_version": PROJECT_VERSION,
            "box": {"width": 0.12, "height": 0.10, "depth": 0.08, "fillet_radius": 0.0},
            "reference_axis": [0.0, 0.0, 1.0],
            "drivers": [],
            "simulation": {
                "f_lo": 100.0,
                "f_hi": 2000.0,
                "frac_oct": 1 / 12,
                "sphere_scheme": "lebedev",
                "sphere_n_points": 26,
                "sphere_radius": 1.0,
                "output_path": "",
            },
            "solver_config": {
                "n_epw": 6,
                "tolerance": 1e-6,
                "max_iterations": 1000,
                "burton_miller": True,
                "speed_of_sound": 343.2,
                "air_density": 1.2041,
                "air_attenuation_model": "none",
            },
            "results_h5_path": None,
        }


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
