"""Filter Designer tab (Stage P2-3): drive the Phase-2 beamformer from the GUI.

A thin shell over the Qt-free core (:mod:`beamsim2.beamform.design`). The user picks a target
beam (pattern preset, optional cardioid-order, steering direction), an engine, and a
robustness (white-noise-gain) floor; "Design" runs the solver on a background thread; the
result is plotted (achieved vs target H-plane polar + directivity-vs-frequency) and can be
exported for audit in VituixCAD/REW (:func:`beamsim2.io.filter_export.export_filter_design`).

Follows the Phase-1 GUI conventions: ``AppState``, the matplotlib ``_MplCanvas`` pattern, a
background ``QThread`` worker, and a strict one-way core<-gui dependency.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from beamsim2.assembly.tensor import RadiationDataset
from beamsim2.beamform.design import design
from beamsim2.beamform.targets import TargetSpec, build_target
from beamsim2.core.sh_transform import great_circle_arc, resample, safe_order_for_grid
from beamsim2.gui.results_view import _MplCanvas

# Pattern combo entries -> (mode, preset). "Cardioid order (slider)" enables the order slider.
# The last two are Auto-Design *objectives* (not first-order shapes): with the Auto-Design engine
# they route to the constant-DI / max-directivity target classes; with a concrete engine the mode
# is a harmless steering target (the engine ignores the objective). See _PATTERN_OBJECTIVE.
_PATTERNS = [
    ("Omni", "preset", "omni"),
    ("Cardioid", "preset", "cardioid"),
    ("Supercardioid", "preset", "supercardioid"),
    ("Hypercardioid", "preset", "hypercardioid"),
    ("Figure-8", "preset", "figure8"),
    ("Wide", "preset", "wide"),
    ("Narrow", "preset", "narrow"),
    ("Cardioid order (slider)", "cardioid_order", None),
    ("Constant directivity", "steering_only", None),
    ("Maximum directivity", "steering_only", None),
    ("Multi-target (DI/beamwidth/in-room)", "steering_only", None),
]
# Pattern label -> Auto-Design objective (target class). Everything not listed is "shape".
_PATTERN_OBJECTIVE = {
    "Constant directivity": "constant_directivity",
    "Maximum directivity": "max_directivity",
    "Multi-target (DI/beamwidth/in-room)": "multi",
}
# The multi-target pattern label (Chunk 3d) — needs the Auto-Design engine + the objective controls.
_MULTI_LABEL = "Multi-target (DI/beamwidth/in-room)"
# Auto-Design leads the list (the user picks a target, not an algorithm) but is opt-in; the
# concrete engines remain selectable, with Least-squares the active default (set in _build_controls
# so the default Design is one fast solve). Auto reports which engine it chose (see below).
_ENGINES = [
    ("Auto-Design (pick best engine)", "auto"),
    ("Least-squares (shape)", "ls"),
    ("Delay-and-sum (steer)", "delay_sum"),
    ("MVDR (superdirective)", "mvdr"),
    ("LCMV (steer + nulls)", "lcmv"),
    ("Max directivity", "max_directivity"),
    ("Constant directivity", "constant_di"),
]


class DesignWorker(QObject):
    """Runs :func:`beamsim2.beamform.design.design` on a background QThread."""

    finished = Signal(object)  # DesignResult
    failed = Signal(str)

    def __init__(self, ds: RadiationDataset, spec: TargetSpec) -> None:
        super().__init__()
        self._ds = ds
        self._spec = spec

    @Slot()
    def run(self) -> None:
        try:
            self.finished.emit(design(self._ds, self._spec))
        except Exception as exc:  # surface any solver error to the GUI
            self.failed.emit(str(exc))


class FilterDesignerTab(QWidget):
    """Tab 5 — design per-driver beamforming weights and export an audit set."""

    def __init__(self, state, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._state = state
        self._ds: Optional[RadiationDataset] = None
        self._result = None
        self._thread: Optional[QThread] = None
        self._worker: Optional[DesignWorker] = None

        root = QHBoxLayout(self)
        root.addWidget(self._build_controls(), 0)
        root.addWidget(self._build_plots(), 1)
        self._set_enabled(False)

    # ------------------------------------------------------------------ build
    def _build_controls(self) -> QWidget:
        box = QGroupBox("Target beam")
        form = QFormLayout(box)

        self._pattern = QComboBox()
        for label, _, _ in _PATTERNS:
            self._pattern.addItem(label)
        self._pattern.setCurrentIndex(1)  # cardioid
        self._pattern.currentIndexChanged.connect(self._on_pattern_changed)
        form.addRow("Pattern:", self._pattern)

        self._order = QSlider(Qt.Orientation.Horizontal)
        self._order.setRange(0, 100)  # a in [0, 1] -> /100
        self._order.setValue(50)
        self._order.setEnabled(False)
        form.addRow("Cardioid order a:", self._order)

        self._steer_theta = self._spin(0.0, 180.0, 0.0, " deg")
        self._steer_phi = self._spin(0.0, 360.0, 0.0, " deg")
        form.addRow("Steer θ (from +z):", self._steer_theta)
        form.addRow("Steer φ (azimuth):", self._steer_phi)

        self._engine = QComboBox()
        for label, _ in _ENGINES:
            self._engine.addItem(label)
        # Auto-Design is the first (most prominent) item, one click away, but Least-squares stays
        # the active default so the default "Design" runs one fast solve, not the auto ladder's
        # up-to-4 solves (the user opts into Auto-Design deliberately).
        self._engine.setCurrentIndex([e for _, e in _ENGINES].index("ls"))
        form.addRow("Engine:", self._engine)

        self._accept = self._spin(5.0, 90.0, 60.0, " deg")
        form.addRow("Accept half-angle:", self._accept)

        self._wng = QSlider(Qt.Orientation.Horizontal)
        self._wng.setRange(-20, 10)  # WNG floor in dB
        self._wng.setValue(-6)
        self._wng.valueChanged.connect(lambda v: self._wng_lbl.setText(f"{v} dB"))
        self._wng_lbl = QLabel("-6 dB")
        wrow = QWidget()
        wlay = QHBoxLayout(wrow)
        wlay.setContentsMargins(0, 0, 0, 0)
        wlay.addWidget(self._wng)
        wlay.addWidget(self._wng_lbl)
        form.addRow("Robustness (WNG floor):", wrow)

        # Multi-target objectives (Chunk 3d): each row = [use] target + relative weight. Active only
        # for the Multi-target pattern (which forces the Auto-Design engine). Defaults match the
        # research-backed in-room slope (-1 dB/oct) and a mid directive/beamwidth point.
        self._mt_group = self._build_multi_group()
        form.addRow(self._mt_group)

        self._design_btn = QPushButton("Design")
        self._design_btn.clicked.connect(self._on_design)
        form.addRow(self._design_btn)

        self._export_btn = QPushButton("Export audit (.frd + weights)…")
        self._export_btn.clicked.connect(self._on_export)
        self._export_btn.setEnabled(False)
        form.addRow(self._export_btn)

        self._metrics = QLabel("No design yet.")
        self._metrics.setWordWrap(True)
        form.addRow("Result:", self._metrics)
        return box

    def _build_plots(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        topbar = QHBoxLayout()
        topbar.addWidget(QLabel("Polar frequency:"))
        self._freq_combo = QComboBox()
        self._freq_combo.currentIndexChanged.connect(self._replot)
        topbar.addWidget(self._freq_combo)
        topbar.addStretch(1)
        lay.addLayout(topbar)
        self._polar = _MplCanvas(projection="polar")
        self._di = _MplCanvas()
        lay.addWidget(self._polar, 1)
        lay.addWidget(self._di, 1)
        return w

    @staticmethod
    def _spin(lo: float, hi: float, val: float, suffix: str) -> QDoubleSpinBox:
        s = QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setValue(val)
        s.setSuffix(suffix)
        return s

    def _build_multi_group(self) -> QGroupBox:
        """The Multi-target objective controls (Chunk 3d): per-objective use/target/weight rows."""
        box = QGroupBox("Multi-target objectives (Auto-Design)")
        form = QFormLayout(box)

        def row(
            target: QDoubleSpinBox, default_on: bool = True
        ) -> tuple[QWidget, QCheckBox, QDoubleSpinBox]:
            use = QCheckBox()
            use.setChecked(default_on)
            weight = self._spin(0.0, 10.0, 1.0, "")
            weight.setSingleStep(0.5)
            wrap = QWidget()
            lay = QHBoxLayout(wrap)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.addWidget(use)
            lay.addWidget(target, 1)
            lay.addWidget(QLabel("weight:"))
            lay.addWidget(weight)
            return wrap, use, weight

        self._mt_di = self._spin(0.0, 25.0, 12.0, " dB")
        r1, self._mt_di_on, self._mt_di_w = row(self._mt_di)
        form.addRow("Directivity index:", r1)

        self._mt_bw = self._spin(5.0, 180.0, 45.0, " deg")
        r2, self._mt_bw_on, self._mt_bw_w = row(self._mt_bw)
        form.addRow("-6 dB beamwidth:", r2)

        # In-room (CEA-2034-A Estimated-In-Room) spectral slope; the Harman/Olive preferred neutral
        # value is ~ -1 dB/oct (flatter for very directive speakers). See docs/Chunk3d_Findings.md.
        self._mt_slope = self._spin(-6.0, 2.0, -1.0, " dB/oct")
        r3, self._mt_slope_on, self._mt_slope_w = row(self._mt_slope)
        form.addRow("In-room slope:", r3)
        return box

    # --------------------------------------------------------------- dataset
    def load(self, ds: RadiationDataset) -> None:
        """Attach a dataset (from a finished solve or an opened HDF5 file)."""
        self._ds = ds
        self._result = None
        self._freq_combo.blockSignals(True)
        self._freq_combo.clear()
        for f in ds.frequencies:
            self._freq_combo.addItem(f"{f:.0f} Hz")
        self._freq_combo.setCurrentIndex(len(ds.frequencies) // 2)
        self._freq_combo.blockSignals(False)
        self._set_enabled(True)
        self._export_btn.setEnabled(False)
        self._metrics.setText("Dataset loaded — choose a target and press Design.")

    def _set_enabled(self, on: bool) -> None:
        for wdg in (
            self._pattern,
            self._steer_theta,
            self._steer_phi,
            self._engine,
            self._accept,
            self._wng,
            self._mt_group,
            self._design_btn,
            self._freq_combo,
        ):
            wdg.setEnabled(on)
        if on:
            self._on_pattern_changed()  # restore the multi-group / engine-lock state per pattern

    # ---------------------------------------------------------------- design
    def _steer_dir(self) -> np.ndarray:
        th = np.deg2rad(self._steer_theta.value())
        ph = np.deg2rad(self._steer_phi.value())
        return np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])

    def _build_spec(self) -> TargetSpec:
        label, mode, preset = _PATTERNS[self._pattern.currentIndex()]
        objective = _PATTERN_OBJECTIVE.get(label, "shape")
        common = dict(
            mode=mode,
            preset=preset,
            order_a=self._order.value() / 100.0 if mode == "cardioid_order" else None,
            steer_dir=self._steer_dir(),
            wng_floor_db=float(self._wng.value()),
            accept_halfangle_deg=float(self._accept.value()),
            # The constant_di / max_directivity engines use Luo's proper directivity INDEX
            # (constant directivity in the loudspeaker sense); see docs/Chunk3b_Findings.md.
            directivity_mode="index",
            objective=objective,
        )
        if objective == "multi":
            # Multi-target (Chunk 3d): scalarized {DI, beamwidth, in-room} search on Auto-Design.
            # An unchecked objective -> target None (dropped); the rest are the active objectives.
            return TargetSpec(
                **common,
                engine="auto",
                target_di_db=float(self._mt_di.value()) if self._mt_di_on.isChecked() else None,
                target_beamwidth_deg=(
                    float(self._mt_bw.value()) if self._mt_bw_on.isChecked() else None
                ),
                target_inroom_slope_db_per_oct=(
                    float(self._mt_slope.value()) if self._mt_slope_on.isChecked() else None
                ),
                objective_weights={
                    "di": float(self._mt_di_w.value()),
                    "beamwidth": float(self._mt_bw_w.value()),
                    "inroom": float(self._mt_slope_w.value()),
                },
            )
        # Auto-Design (engine="auto") reads `objective` (the target class) to pick the engine;
        # concrete engines ignore it. The pattern label conveys the constant-/max-DI objectives.
        return TargetSpec(**common, engine=_ENGINES[self._engine.currentIndex()][1])

    def _on_pattern_changed(self) -> None:
        label, mode, _ = _PATTERNS[self._pattern.currentIndex()]
        self._order.setEnabled(mode == "cardioid_order")
        is_multi = label == _MULTI_LABEL
        self._mt_group.setEnabled(is_multi)
        if is_multi:
            # Multi-target is a search -> it must run on Auto-Design; lock the engine combo there.
            self._engine.setCurrentIndex([e for _, e in _ENGINES].index("auto"))
        self._engine.setEnabled(not is_multi)

    def _on_design(self) -> None:
        if self._ds is None:
            return
        spec = self._build_spec()
        self._design_btn.setEnabled(False)
        self._design_btn.setText("Designing…")
        self._thread = QThread()
        self._worker = DesignWorker(self._ds, spec)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_design_done)
        self._worker.failed.connect(self._on_design_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.start()

    def _on_design_done(self, result) -> None:
        self._result = result
        self._design_btn.setEnabled(True)
        self._design_btn.setText("Design")
        self._export_btn.setEnabled(True)
        m = result.metrics
        fi = max(self._freq_combo.currentIndex(), 0)
        di = float(m["di_db"][fi])
        wng = float(m["wng_db"][fi])
        feas = bool(np.all(m["feasible_mask"]))
        extra = ""
        if "constant_di_db" in result.attrs:  # directivity_mode="index" (Luo directivity index)
            extra = f"  constant DI = {result.attrs['constant_di_db']:.2f} dB"
        elif "constant_gdi_db" in result.attrs:  # directivity_mode="region" (cap-ratio GDI)
            extra = f"  constant GDI = {result.attrs['constant_gdi_db']:.2f} dB"
        # Multi-target: append a per-objective achieved-vs-target summary (Chunk 3d honest report).
        if result.attrs.get("auto_class") == "multi":
            ma = result.attrs.get("multi_achieved", {})
            bits = []
            if "di" in ma:
                bits.append(f"DI {ma['di']['achieved']:.1f}/{ma['di']['target']:.0f} dB")
            if "beamwidth" in ma:
                bw_a = ma["beamwidth"]["achieved"]
                bw_s = "n/a" if not np.isfinite(bw_a) else f"{bw_a:.0f}"
                bits.append(f"BW {bw_s}/{ma['beamwidth']['target']:.0f}°")
            if "inroom" in ma:
                bits.append(
                    f"in-room {ma['inroom']['achieved']:+.1f}/{ma['inroom']['target']:+.1f} dB/oct"
                )
            if bits:
                extra += "  ·  multi: " + ", ".join(bits)
        # Auto-Design: name the engine it CHOSE and surface its honest reason / infeasibility.
        engine_label = result.attrs["engine"]
        if result.attrs.get("auto_selected"):
            engine_label = f"Auto → {result.attrs['engine']}"
            if result.attrs.get("band_feasible", True) is False:
                extra += "  ⚠ target not fully feasible (best-effort)"
        self._metrics.setText(
            f"Engine: {engine_label}  ·  DI ≈ {di:.1f} dB  ·  WNG ≈ {wng:.1f} dB  ·  "
            f"all bins feasible: {feas}{extra}"
        )
        if result.attrs.get("auto_selected"):
            self._metrics.setToolTip(result.attrs.get("auto_reason", ""))
        self._replot()

    def _on_design_failed(self, err: str) -> None:
        self._design_btn.setEnabled(True)
        self._design_btn.setText("Design")
        QMessageBox.critical(self, "Design failed", err)

    # ----------------------------------------------------------------- plots
    def _replot(self) -> None:
        if self._result is None or self._ds is None:
            return
        fi = max(self._freq_combo.currentIndex(), 0)
        spec = self._result.spec
        obs = self._ds.directions
        order = min(20, safe_order_for_grid(obs.unit_vectors.shape[0]))

        # Achieved + target on the horizontal great-circle through the steer axis.
        angle, arc_uv = great_circle_arc(np.asarray(spec.steer_dir, float), 361)
        achieved = resample(self._result.steered_field[fi], obs, arc_uv, order)
        # Pass the dataset's speed of sound so the (now c-dependent) target phase matches design().
        c_sound = float(self._ds.attrs.get("speed_of_sound", 343.2))
        target = build_target(spec, obs, self._ds.frequencies, c_sound=c_sound)
        tgt_arc = resample(target.b_field[fi], obs, arc_uv, order)

        def norm_db(x):
            mag = np.abs(x)
            return 20.0 * np.log10(mag / (np.max(mag) + 1e-300) + 1e-6)

        ax = self._polar.ax
        ax.clear()
        ax.set_theta_zero_location("N")
        ax.plot(angle, np.clip(norm_db(achieved), -40, 0), label="achieved", lw=2)
        ax.plot(angle, np.clip(norm_db(tgt_arc), -40, 0), "--", label="target", lw=1)
        ax.set_ylim(-40, 0)
        ax.set_title(f"H-plane polar @ {self._ds.frequencies[fi]:.0f} Hz (dB, normalized)")
        ax.legend(loc="lower left", fontsize=8)
        self._polar.redraw()

        ax2 = self._di.ax
        ax2.clear()
        ax2.semilogx(self._ds.frequencies, self._result.metrics["di_db"], "-o", ms=3)
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("Directivity index (dB)")
        ax2.set_title("Achieved directivity vs frequency")
        ax2.grid(True, which="both", alpha=0.3)
        self._di.redraw()

    # ---------------------------------------------------------------- export
    def _on_export(self) -> None:
        if self._result is None or self._ds is None:
            return
        out = QFileDialog.getExistingDirectory(self, "Choose an export directory")
        if not out:
            return
        try:
            from beamsim2.io.filter_export import export_filter_design

            path = export_filter_design(Path(out) / "beamsim_filter_design", self._ds, self._result)
            QMessageBox.information(self, "Export complete", f"Audit set written to:\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
