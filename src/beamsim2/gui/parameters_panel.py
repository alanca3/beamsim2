"""Parameters panel: Drivers tab (T/S entry) and Simulation tab (freq / sphere / Estimate / Run).

Build-order item 10, Tabs 2 and 3 (§6 Gameplan — parameter entry and run controls).

Sphere-density presets are limited to Lebedev {6, 14, 26} — the only orders
implemented in ``core.sphere``.  Higher-order options ("balloon-5°") are deferred
until finer Lebedev tables are vendored.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from beamsim2.core.types import FrequencyGrid
from beamsim2.driver.inductance import LR2Ladder, PlainLe
from beamsim2.driver.terminal import TerminalModel
from beamsim2.driver.thiele_small import TSParams
from beamsim2.geometry.assemble import DriverSpec
from beamsim2.pipeline.run import (
    DriverPlacement,
    ResourceEstimate,
    SimulationRequest,
)

# Only these Lebedev orders are implemented in core.sphere
_SPHERE_PRESETS = [
    ("Coarse (6 points)", 6),
    ("Standard (14 points)", 14),
    ("Fine (26 points)", 26),
]

# Fractional-octave resolution presets → step size in Hz for FrequencyGrid
_RESOLUTION_PRESETS = [
    ("1/3 octave", "1/3-oct", 1 / 3),
    ("1/6 octave", "1/6-oct", 1 / 6),
    ("1/12 octave (default)", "1/12-oct", 1 / 12),
]


def _spin(
    lo: float, hi: float, value: float, decimals: int = 4, step: float = 0.001
) -> QDoubleSpinBox:
    sb = QDoubleSpinBox()
    sb.setRange(lo, hi)
    sb.setValue(value)
    sb.setDecimals(decimals)
    sb.setSingleStep(step)
    return sb


def _build_freq_grid(f_lo: float, f_hi: float, frac_oct: float) -> FrequencyGrid:
    """Build a log-spaced FrequencyGrid from low/high/frac-octave resolution."""
    n_per_oct = int(round(1.0 / frac_oct))
    # Number of steps spanning the octave range
    n_oct = np.log2(f_hi / f_lo)
    n_steps = max(int(round(n_oct * n_per_oct)) + 1, 2)
    freqs = np.geomspace(f_lo, f_hi, n_steps)
    return FrequencyGrid(
        frequencies=freqs,
        spacing="fractional-octave",
        fractional_octave=frac_oct,
    )


# ---------------------------------------------------------------------------
# T/S parameter dialog
# ---------------------------------------------------------------------------


class TSDialog(QDialog):
    """Dialog for entering the six irreducible TSParams and inductance model.

    Shows derived fs / Qts as read-only calculated fields.
    """

    def __init__(
        self, placement: Optional[DriverPlacement] = None, parent: Optional[QWidget] = None
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("T/S Parameters")
        self.setMinimumWidth(360)
        self._result: Optional[DriverPlacement] = None

        layout = QVBoxLayout(self)

        # Identity
        id_row = QHBoxLayout()
        id_row.addWidget(QLabel("Driver ID:"))
        self._id_edit = QLineEdit()
        self._id_edit.setPlaceholderText("e.g. woofer_0")
        id_row.addWidget(self._id_edit)
        layout.addLayout(id_row)

        # Geometry
        geo_box = QGroupBox("Geometry")
        geo_form = QFormLayout(geo_box)
        self._cx = _spin(-2, 2, 0.06, decimals=4)
        geo_form.addRow("Center x (m):", self._cx)
        self._cy = _spin(-2, 2, 0.05, decimals=4)
        geo_form.addRow("Center y (m):", self._cy)
        self._cz = _spin(-2, 2, 0.08, decimals=4)
        geo_form.addRow("Center z (m):", self._cz)
        self._radius = _spin(0.005, 0.5, 0.040, decimals=4)
        geo_form.addRow("Radius (m):", self._radius)
        self._normal_combo = QComboBox()
        for label, _ in [
            ("+z (top/front)", (0, 0, 1)),
            ("-z (back)", (0, 0, -1)),
            ("+x (right)", (1, 0, 0)),
            ("-x (left)", (-1, 0, 0)),
            ("+y (top)", (0, 1, 0)),
            ("-y (bottom)", (0, -1, 0)),
        ]:
            self._normal_combo.addItem(label)
        geo_form.addRow("Face normal:", self._normal_combo)
        layout.addWidget(geo_box)

        # T/S parameters (the six irreducible small-signal parameters)
        ts_box = QGroupBox("Small-signal parameters (T/S)")
        ts_form = QFormLayout(ts_box)
        self._Re = _spin(0.01, 100, 6.0, decimals=3, step=0.1)
        ts_form.addRow("Re (Ω):", self._Re)
        self._Bl = _spin(0.01, 50, 7.0, decimals=3, step=0.1)
        ts_form.addRow("Bl (T·m):", self._Bl)
        self._Mms = _spin(1e-6, 0.5, 0.012, decimals=6, step=0.001)
        ts_form.addRow("Mms (kg):", self._Mms)
        self._Cms = _spin(1e-6, 5e-3, 8e-4, decimals=7, step=1e-5)
        ts_form.addRow("Cms (m/N):", self._Cms)
        self._Rms = _spin(0.001, 50, 1.0, decimals=4, step=0.01)
        ts_form.addRow("Rms (N·s/m):", self._Rms)
        self._Sd = _spin(1e-5, 1.0, 0.0133, decimals=5, step=0.001)
        ts_form.addRow("Sd (m²):", self._Sd)
        # Derived read-only
        self._fs_label = QLabel("—")
        self._qts_label = QLabel("—")
        ts_form.addRow("fs (Hz, derived):", self._fs_label)
        ts_form.addRow("Qts (derived):", self._qts_label)
        layout.addWidget(ts_box)

        # Inductance model
        ind_box = QGroupBox("Voice-coil inductance")
        ind_form = QFormLayout(ind_box)
        self._ind_combo = QComboBox()
        self._ind_combo.addItem("LR-2 ladder (recommended)")
        self._ind_combo.addItem("Plain Le (simple ideal inductor)")
        ind_form.addRow("Model:", self._ind_combo)
        self._Le = _spin(0, 0.1, 0.5e-3, decimals=7, step=1e-5)
        ind_form.addRow("Le (H):", self._Le)
        self._Le2 = _spin(0, 0.1, 0.2e-3, decimals=7, step=1e-5)
        ind_form.addRow("Le2 (H, LR-2):", self._Le2)
        self._Re2 = _spin(0, 200, 3.0, decimals=3, step=0.1)
        ind_form.addRow("Re2 (Ω, LR-2):", self._Re2)
        layout.addWidget(ind_box)

        # Box volume
        bv_box = QGroupBox("Enclosure (optional)")
        bv_form = QFormLayout(bv_box)
        self._use_box = QCheckBox("Sealed enclosure back-volume")
        bv_form.addRow(self._use_box)
        self._vol = _spin(0.001, 100, 10.0, decimals=3, step=0.5)
        self._vol.setSuffix(" L")
        bv_form.addRow("Box volume:", self._vol)
        layout.addWidget(bv_box)

        # Buttons
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self._on_ok)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

        # Update derived on parameter change
        for sb in (self._Mms, self._Cms, self._Rms, self._Re, self._Bl):
            sb.valueChanged.connect(self._update_derived)

        # Pre-fill from existing placement
        if placement is not None:
            self._prefill(placement)
        self._update_derived()

    def _prefill(self, dp: DriverPlacement) -> None:
        self._id_edit.setText(dp.driver_id)
        self._cx.setValue(dp.spec.center[0])
        self._cy.setValue(dp.spec.center[1])
        self._cz.setValue(dp.spec.center[2])
        self._radius.setValue(dp.spec.radius)
        if dp.terminal is not None:
            ts = dp.terminal.ts
            self._Re.setValue(ts.Re)
            self._Bl.setValue(ts.Bl)
            self._Mms.setValue(ts.Mms)
            self._Cms.setValue(ts.Cms)
            self._Rms.setValue(ts.Rms)
            self._Sd.setValue(ts.Sd)

    def _update_derived(self) -> None:
        try:
            ts = TSParams(
                Re=self._Re.value(),
                Bl=self._Bl.value(),
                Mms=self._Mms.value(),
                Cms=self._Cms.value(),
                Rms=self._Rms.value(),
                Sd=self._Sd.value(),
            )
            self._fs_label.setText(f"{ts.fs:.1f} Hz")
            self._qts_label.setText(f"{ts.Qts:.3f}")
        except Exception:
            self._fs_label.setText("error")
            self._qts_label.setText("error")

    def _normal_from_combo(self) -> tuple[float, float, float]:
        normals = [(0, 0, 1), (0, 0, -1), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)]
        return normals[self._normal_combo.currentIndex()]

    def _on_ok(self) -> None:
        driver_id = self._id_edit.text().strip() or "driver_0"
        spec = DriverSpec(
            center=(self._cx.value(), self._cy.value(), self._cz.value()),
            normal=self._normal_from_combo(),
            radius=self._radius.value(),
        )
        ts = TSParams(
            Re=self._Re.value(),
            Bl=self._Bl.value(),
            Mms=self._Mms.value(),
            Cms=self._Cms.value(),
            Rms=self._Rms.value(),
            Sd=self._Sd.value(),
        )
        if self._ind_combo.currentIndex() == 0:
            inductance = LR2Ladder(
                Le=self._Le.value(), Le2=self._Le2.value(), Re2=self._Re2.value()
            )
        else:
            inductance = PlainLe(Le=self._Le.value())

        box_vol = (self._vol.value() * 1e-3) if self._use_box.isChecked() else None
        terminal = TerminalModel(ts=ts, inductance=inductance, box_volume=box_vol, name=driver_id)
        self._result = DriverPlacement(spec=spec, terminal=terminal, driver_id=driver_id)
        self.accept()

    @property
    def placement(self) -> Optional[DriverPlacement]:
        return self._result


# ---------------------------------------------------------------------------
# Drivers tab
# ---------------------------------------------------------------------------


class _DriverRow(QWidget):
    """One driver row: label + Edit + Remove buttons."""

    editRequested = Signal(int)
    removeRequested = Signal(int)

    def __init__(self, index: int, dp: DriverPlacement, parent=None) -> None:
        super().__init__(parent)
        self._index = index
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 2, 0, 2)
        label = QLabel(
            f"{dp.driver_id}  "
            f"r={dp.spec.radius*100:.1f} cm  "
            f"@ ({dp.spec.center[0]:.3f}, {dp.spec.center[1]:.3f}, {dp.spec.center[2]:.3f}) m"
        )
        row.addWidget(label, stretch=1)
        edit_btn = QPushButton("Edit")
        edit_btn.setFixedWidth(50)
        edit_btn.clicked.connect(lambda: self.editRequested.emit(self._index))
        row.addWidget(edit_btn)
        rm_btn = QPushButton("Remove")
        rm_btn.setFixedWidth(60)
        rm_btn.clicked.connect(lambda: self.removeRequested.emit(self._index))
        row.addWidget(rm_btn)


class DriversTab(QWidget):
    """Tab 2 — Drivers: list + T/S entry dialog.

    Signals
    -------
    driversChanged
        Emitted when the driver list changes (add / edit / remove).
    """

    driversChanged = Signal()

    def __init__(self, state, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._state = state

        layout = QVBoxLayout(self)

        info = QLabel(
            "Add one or more drivers.  Each driver gets its own T/S parameters\n"
            "and geometric placement on the enclosure."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        add_btn = QPushButton("+ Add driver")
        add_btn.clicked.connect(self._add_driver)
        layout.addWidget(add_btn)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._list_widget = QWidget()
        self._list_layout = QVBoxLayout(self._list_widget)
        self._list_layout.addStretch()
        self._scroll.setWidget(self._list_widget)
        layout.addWidget(self._scroll)

    def _rebuild_rows(self) -> None:
        # Clear existing rows (all except the trailing stretch)
        while self._list_layout.count() > 1:
            item = self._list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        for i, dp in enumerate(self._state.drivers):
            row = _DriverRow(i, dp)
            row.editRequested.connect(self._edit_driver)
            row.removeRequested.connect(self._remove_driver)
            self._list_layout.insertWidget(i, row)

    def _add_driver(self) -> None:
        dlg = TSDialog(parent=self)
        # Pre-number the new driver
        dlg._id_edit.setText(f"driver_{len(self._state.drivers)}")
        if dlg.exec() == QDialog.DialogCode.Accepted and dlg.placement:
            self._state.drivers.append(dlg.placement)
            self._rebuild_rows()
            self.driversChanged.emit()

    def _edit_driver(self, index: int) -> None:
        dp = self._state.drivers[index]
        dlg = TSDialog(placement=dp, parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted and dlg.placement:
            self._state.drivers[index] = dlg.placement
            self._rebuild_rows()
            self.driversChanged.emit()

    def _remove_driver(self, index: int) -> None:
        self._state.drivers.pop(index)
        self._rebuild_rows()
        self.driversChanged.emit()


# ---------------------------------------------------------------------------
# Run-monitor widget (used by SimulationTab)
# ---------------------------------------------------------------------------


class RunMonitorWidget(QWidget):
    """Live M × F status grid + progress bar + RAM/ETA labels.

    Receives ``ProgressSnapshot`` objects from ``update_snapshot``.
    Connected as a slot to ``SolveWorker.progressChanged`` (queued).
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)

        from PySide6.QtGui import QColor
        from PySide6.QtWidgets import QTableWidget, QTableWidgetItem

        from beamsim2.pipeline.progress import StepState

        self._StepState = StepState
        self._QTableWidget = QTableWidget
        self._QTableWidgetItem = QTableWidgetItem
        self._QColor = QColor

        self._grid = QTableWidget(0, 0)
        self._grid.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._grid.horizontalHeader().setDefaultSectionSize(28)
        self._grid.verticalHeader().setDefaultSectionSize(22)
        layout.addWidget(self._grid)

        self._progress_bar = __import__(
            "PySide6.QtWidgets", fromlist=["QProgressBar"]
        ).QProgressBar()
        layout.addWidget(self._progress_bar)

        meta_row = QHBoxLayout()
        self._ram_label = QLabel("RAM: —")
        self._eta_label = QLabel("ETA: —")
        self._drv_label = QLabel("Driver: —")
        meta_row.addWidget(self._drv_label)
        meta_row.addStretch()
        meta_row.addWidget(self._ram_label)
        meta_row.addWidget(self._eta_label)
        layout.addLayout(meta_row)

        self._flagged_label = QLabel("")
        self._flagged_label.setStyleSheet("color: darkorange;")
        self._flagged_label.setWordWrap(True)
        layout.addWidget(self._flagged_label)

    def update_snapshot(self, snap) -> None:
        """Called (via queued signal) with each ProgressSnapshot from the worker."""
        from beamsim2.pipeline.progress import StepState

        M, F = snap.grid.shape
        if self._grid.rowCount() != M or self._grid.columnCount() != F:
            self._grid.setRowCount(M)
            self._grid.setColumnCount(F)

        _COLORS = {
            StepState.QUEUED: "#d0d0d0",
            StepState.RUNNING: "#5294e0",
            StepState.DONE: "#52c052",
            StepState.FLAGGED: "#e09c52",
        }
        for m in range(M):
            for f in range(F):
                state = snap.grid[m, f]
                item = self._QTableWidgetItem()
                item.setBackground(self._QColor(_COLORS.get(state, "#ffffff")))
                self._grid.setItem(m, f, item)

        self._progress_bar.setMaximum(snap.steps_total)
        self._progress_bar.setValue(snap.steps_done)

        gb = snap.current_ram_bytes / 1e9
        self._ram_label.setText(f"RAM: {gb:.1f} GB")
        if snap.eta_seconds is not None:
            mins = snap.eta_seconds / 60
            self._eta_label.setText(f"ETA: {mins:.0f} min")
        else:
            self._eta_label.setText("ETA: —")
        if snap.current_driver:
            self._drv_label.setText(f"Driver: {snap.current_driver}")

        if snap.message:
            self._flagged_label.setText(snap.message)


# ---------------------------------------------------------------------------
# Simulation tab
# ---------------------------------------------------------------------------


class SimulationTab(QWidget):
    """Tab 3 — Simulation: frequency range, sphere preset, Estimate/Run, monitor.

    Signals
    -------
    estimateRequested
        Emitted when the user clicks [Estimate].
    runRequested
        Emitted when the user clicks [Run].
    """

    estimateRequested = Signal()
    runRequested = Signal()

    def __init__(self, state, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._state = state
        self.current_request: Optional[SimulationRequest] = None

        layout = QVBoxLayout(self)

        # ── Frequency ───────────────────────────────────────────────────────
        freq_box = QGroupBox("Frequency range")
        freq_form = QFormLayout(freq_box)
        self._f_lo = _spin(20, 20000, 100, decimals=1, step=10)
        self._f_lo.setSuffix(" Hz")
        self._f_hi = _spin(20, 20000, 2000, decimals=1, step=100)
        self._f_hi.setSuffix(" Hz")
        freq_form.addRow("Low (Hz):", self._f_lo)
        freq_form.addRow("High (Hz):", self._f_hi)

        self._res_combo = QComboBox()
        for label, _, _ in _RESOLUTION_PRESETS:
            self._res_combo.addItem(label)
        self._res_combo.setCurrentIndex(2)  # default 1/12 oct
        freq_form.addRow("Resolution:", self._res_combo)
        layout.addWidget(freq_box)

        # ── Sphere ──────────────────────────────────────────────────────────
        sphere_box = QGroupBox("Observation sphere")
        sphere_form = QFormLayout(sphere_box)
        self._sphere_combo = QComboBox()
        for label, _ in _SPHERE_PRESETS:
            self._sphere_combo.addItem(label)
        self._sphere_combo.setCurrentIndex(2)  # default fine-26
        sphere_form.addRow("Density:", self._sphere_combo)
        self._sphere_radius = _spin(0.1, 10.0, 1.0, decimals=2, step=0.1)
        self._sphere_radius.setSuffix(" m")
        sphere_form.addRow("Radius:", self._sphere_radius)
        layout.addWidget(sphere_box)

        # ── Output ──────────────────────────────────────────────────────────
        out_box = QGroupBox("Output")
        out_form = QFormLayout(out_box)
        self._out_path = QLineEdit()
        self._out_path.setPlaceholderText("(optional) path/to/output.h5")
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse_output)
        out_row = QHBoxLayout()
        out_row.addWidget(self._out_path)
        out_row.addWidget(browse_btn)
        out_form.addRow("Save HDF5:", out_row)
        layout.addWidget(out_box)

        # ── Estimate + Run ───────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        self._est_btn = QPushButton("Estimate…")
        self._est_btn.clicked.connect(self.estimateRequested.emit)
        btn_row.addWidget(self._est_btn)

        self._run_btn = QPushButton("Run")
        self._run_btn.setEnabled(False)
        self._run_btn.clicked.connect(self.runRequested.emit)
        btn_row.addWidget(self._run_btn)
        layout.addLayout(btn_row)

        self._estimate_label = QLabel("")
        self._estimate_label.setWordWrap(True)
        layout.addWidget(self._estimate_label)

        # ── Run monitor ──────────────────────────────────────────────────────
        self.monitor = RunMonitorWidget()
        layout.addWidget(self.monitor)

    # ------------------------------------------------------------------
    # Public interface (called by MainWindow)
    # ------------------------------------------------------------------

    def set_run_enabled(self, ok: bool) -> None:
        self._run_btn.setEnabled(ok)

    def set_running(self, running: bool) -> None:
        self._run_btn.setEnabled(not running)
        self._est_btn.setEnabled(not running)

    def show_estimate(self, est: ResourceEstimate) -> None:
        gb = est.peak_ram_bytes / 1e9 if est.peak_ram_bytes == est.peak_ram_bytes else "?"
        mins = est.total_wall_seconds / 60
        self._estimate_label.setText(
            f"Estimate (approx.): peak RAM ≈ {gb:.1f} GB, "
            f"wall-clock ≈ {mins:.0f} min, {est.n_steps_total} steps"
        )

    def build_request(self, state) -> bool:
        """Validate GUI inputs and populate ``self.current_request``.

        Returns ``True`` if valid, ``False`` with a dialog if not.
        """
        from PySide6.QtWidgets import QMessageBox

        if not state.drivers:
            QMessageBox.warning(self, "No drivers", "Add at least one driver.")
            return False
        if state.geometry is None:
            QMessageBox.warning(
                self,
                "No geometry",
                "Click 'Preview mesh' on the Geometry tab to validate the enclosure.",
            )
            return False

        f_lo = self._f_lo.value()
        f_hi = self._f_hi.value()
        if f_lo >= f_hi:
            QMessageBox.warning(self, "Invalid frequencies", "Low must be < High.")
            return False

        _, _, frac_oct = _RESOLUTION_PRESETS[self._res_combo.currentIndex()]
        freqs = _build_freq_grid(f_lo, f_hi, frac_oct)

        _, n_pts = _SPHERE_PRESETS[self._sphere_combo.currentIndex()]

        out_h5 = self._out_path.text().strip() or None

        self.current_request = SimulationRequest(
            geometry=state.geometry,
            drivers=list(state.drivers),
            frequencies=freqs,
            sphere_n_points=n_pts,
            sphere_radius=self._sphere_radius.value(),
            config=state.config,
            output_h5=out_h5,
        )
        return True

    def _browse_output(self) -> None:
        from PySide6.QtWidgets import QFileDialog

        path, _ = QFileDialog.getSaveFileName(
            self, "Save HDF5 output", "", "HDF5 files (*.h5 *.bsim)"
        )
        if path:
            self._out_path.setText(path)
