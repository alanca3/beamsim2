"""Filter Designer tab (Stage P2-3): drive the Phase-2 beamformer from the GUI.

A thin shell over the Qt-free core (:mod:`beamsim2.beamform.design`). The user picks a target
beam (pattern preset, optional cardioid-order, steering direction), an engine, and a
robustness (white-noise-gain) floor; "Design" runs the solver on a background thread; the
result is plotted and can be exported for audit in VituixCAD/REW
(:func:`beamsim2.io.filter_export.export_filter_design`).

The plot panel is a sub-tab per view, all **read-only** (never recomputing or re-tuning the
design). Two families of view:

* **Target-aware pattern views** — **Polar** (beam-axis achieved-vs-target cut), **H Polar** /
  **V Polar** (full-system horizontal / vertical cuts referenced to the loudspeaker front, like
  the Results tab), **Balloon** (full-system 3-D pattern), **Sonograms** (full-system H/V
  directivity maps). These now show the **target response live as the parameters change, before
  any design** (the cut views overlay the achieved beam once a design exists); the full-system
  views show the combined response of all drivers with their filters applied (``steered_field``).
* **Realization views** — **Directivity** (DI / -6 dB beamwidth / WNG vs frequency), **Filters**
  (per-driver weight magnitude + phase), **Per-driver** (filtered on-axis responses + combined),
  **CEA2034 / in-room** (the steered spinorama). These need a finished design (the Directivity
  panel shows target reference lines in the preview; CEA previews the target spinorama).

A design is *stale* once any target parameter changes after a solve: the views revert to the
live target preview and export is disabled until the next ``Design``. See
:meth:`FilterDesignerTab._build_plots` and :meth:`FilterDesignerTab._on_target_changed`.

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
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from beamsim2.assembly.tensor import RadiationDataset, stacked_h_full
from beamsim2.beamform.design import design
from beamsim2.beamform.targets import TargetSpec, build_target
from beamsim2.core.sh_transform import great_circle_arc, resample, safe_order_for_grid
from beamsim2.core.sphere import reference_frame
from beamsim2.gui.results_view import (
    _CEA_COLORS,
    _CEA_LABELS,
    _arc_order,
    _db,
    _MplCanvas,
    _plane_arc,
    _quiet_redraw_warnings,
    _reference_axis,
)
from beamsim2.metrics.cea2034 import DI_CURVES, SPL_CURVES, compute_cea2034

# dB-SPL (re 20 µPa) conversion is reused from results_view (`_db`); only the filter-weight
# magnitude path needs its own zero-guard, as it references |w| to the loudest weight (not 20 µPa).
_MAG_FLOOR = 1e-300  # guards log10 of a zero magnitude (degenerate / collapsed weights)

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
        # The loudspeaker front (0deg) axis the steer angles are measured from; set per dataset
        # (RC2 fix, Chunk 5b). Defaults to +z until a dataset is loaded.
        self._front_axis: np.ndarray = np.array([0.0, 0.0, 1.0])
        # Live target preview state. ``_target`` is the current spec's target field (cached so
        # all views share one build_target call); ``_stale`` is True whenever the target params
        # have changed since the last design (so the views show the target preview, not the now-
        # outdated achieved beam, and export is disabled). ``_suspend_preview`` gates the param-
        # change handler during ``load`` so the preview is rebuilt once, at the end of load.
        self._target = None
        self._stale: bool = True
        self._suspend_preview: bool = False
        self._balloon_cbar = None

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
        self._order.valueChanged.connect(self._on_target_changed)
        form.addRow("Cardioid order a:", self._order)

        # Steering is measured from the loudspeaker FRONT axis (the dataset's reference_axis),
        # not world +z (RC2 fix, Chunk 5b): theta=0 aims the beam straight out the front, so the
        # default design points where the speaker faces. _front_lbl shows which world axis that is.
        self._steer_theta = self._spin(0.0, 180.0, 0.0, " deg")
        self._steer_phi = self._spin(0.0, 360.0, 0.0, " deg")
        self._steer_theta.valueChanged.connect(self._on_target_changed)
        self._steer_phi.valueChanged.connect(self._on_target_changed)
        form.addRow("Steer θ (off front axis):", self._steer_theta)
        form.addRow("Steer φ (around front):", self._steer_phi)
        self._front_lbl = QLabel("Front (0°) axis: +z")
        self._front_lbl.setStyleSheet("color: gray;")
        form.addRow("", self._front_lbl)

        self._engine = QComboBox()
        for label, _ in _ENGINES:
            self._engine.addItem(label)
        # Auto-Design is the first (most prominent) item, one click away, but Least-squares stays
        # the active default so the default "Design" runs one fast solve, not the auto ladder's
        # up-to-4 solves (the user opts into Auto-Design deliberately).
        self._engine.setCurrentIndex([e for _, e in _ENGINES].index("ls"))
        self._engine.currentIndexChanged.connect(self._update_engine_note)
        # The engine does not change the target shape, but choosing a different one makes any prior
        # design stale (a different achieved beam) — mark stale + drop the now-outdated achieved.
        self._engine.currentIndexChanged.connect(self._on_target_changed)
        form.addRow("Engine:", self._engine)
        # Live guidance: delay-and-sum only steers and cannot shape a cardioid/directivity (RC3).
        self._engine_note = QLabel("")
        self._engine_note.setWordWrap(True)
        self._engine_note.setStyleSheet("color: #b22; font-size: 11px;")
        self._engine_note.setVisible(False)
        form.addRow("", self._engine_note)

        self._accept = self._spin(5.0, 90.0, 60.0, " deg")
        self._accept.valueChanged.connect(self._on_target_changed)
        form.addRow("Accept half-angle:", self._accept)

        self._wng = QSlider(Qt.Orientation.Horizontal)
        self._wng.setRange(-20, 10)  # WNG floor in dB
        self._wng.setValue(-6)
        self._wng.valueChanged.connect(lambda v: self._wng_lbl.setText(f"{v} dB"))
        self._wng.valueChanged.connect(self._on_target_changed)
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
        """Right-hand plot panel: a sub-tab per view, all read-only over the target / DesignResult.

        Pattern views (target-aware): *Polar* (beam-axis achieved-vs-target cut), *H Polar* /
        *V Polar* (full-system cuts referenced to the loudspeaker front, like the Results tab),
        *Balloon* (full-system 3-D pattern), *Sonograms* (full-system H/V directivity maps).
        Realization views (need a design): *Directivity* (DI / -6 dB beamwidth / WNG vs f),
        *Filters* (per-driver weight magnitude/phase), *Per-driver* (filtered on-axis + combined),
        *CEA2034 / in-room* (the steered spinorama). Each reuses the ``_MplCanvas`` pattern and the
        Results-tab helpers; none recompute or re-tune the design.
        """
        w = QWidget()
        lay = QVBoxLayout(w)
        topbar = QHBoxLayout()
        topbar.addWidget(QLabel("Frequency:"))
        self._freq_combo = QComboBox()
        # The per-frequency cut/3-D views (Polar, H/V Polar, Balloon) depend on the selected
        # frequency; changing it redraws those alone (the vs-frequency / sonogram / spinorama
        # views span the whole band and are unaffected).
        self._freq_combo.currentIndexChanged.connect(self._replot_freq_views)
        topbar.addWidget(self._freq_combo)
        topbar.addStretch(1)
        lay.addLayout(topbar)

        self._plot_tabs = QTabWidget()
        self._polar = _MplCanvas(projection="polar")  # beam-axis achieved vs target, one frequency
        self._hpolar = _MplCanvas(projection="polar")  # full-system horizontal cut (front-ref)
        self._vpolar = _MplCanvas(projection="polar")  # full-system vertical cut (front-ref)
        self._balloon = _MplCanvas(projection="3d")  # full-system 3-D pattern, one frequency
        self._sonogram_canvas = _MplCanvas()  # full-system H/V directivity maps vs frequency
        self._metrics_canvas = _MplCanvas()  # DI / beamwidth / WNG vs frequency (+ target error)
        self._filter_canvas = _MplCanvas()  # per-driver filter weight magnitude + phase
        self._driver_canvas = _MplCanvas()  # filtered per-driver on-axis responses + combined
        self._cea_canvas = _MplCanvas()  # CEA-2034-A spinorama of the steered field (in-room)
        self._plot_tabs.addTab(self._polar, "Polar")
        self._plot_tabs.addTab(self._hpolar, "H Polar")
        self._plot_tabs.addTab(self._vpolar, "V Polar")
        self._plot_tabs.addTab(self._balloon, "Balloon")
        self._plot_tabs.addTab(self._sonogram_canvas, "Sonograms")
        self._plot_tabs.addTab(self._metrics_canvas, "Directivity")
        self._plot_tabs.addTab(self._filter_canvas, "Filters")
        self._plot_tabs.addTab(self._driver_canvas, "Per-driver")
        self._plot_tabs.addTab(self._cea_canvas, "CEA2034 / in-room")
        lay.addWidget(self._plot_tabs, 1)
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
            # A multi-target objective change updates the preview's target reference lines.
            use.toggled.connect(self._on_target_changed)
            target.valueChanged.connect(self._on_target_changed)
            weight.valueChanged.connect(self._on_target_changed)
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
        # Suspend the live-preview handler while we mutate controls (each setValue would otherwise
        # fire _on_target_changed); rebuild the preview once, at the end.
        self._suspend_preview = True
        self._ds = ds
        self._result = None
        self._stale = True
        # Steer from the loudspeaker front (RC2): default the beam straight out the speaker face.
        self._front_axis = self._front_axis_from(ds)
        self._steer_theta.setValue(0.0)
        self._steer_phi.setValue(0.0)
        self._front_lbl.setText(f"Front (0°) axis: {self._axis_label(self._front_axis)}")
        self._freq_combo.blockSignals(True)
        self._freq_combo.clear()
        for f in ds.frequencies:
            self._freq_combo.addItem(f"{f:.0f} Hz")
        self._freq_combo.setCurrentIndex(len(ds.frequencies) // 2)
        self._freq_combo.blockSignals(False)
        self._set_enabled(True)
        self._export_btn.setEnabled(False)
        self._metrics.setText(
            "Dataset loaded — choose a target and press Design (target shown live)."
        )
        # Build and draw the live target preview for the default target (before any design).
        self._suspend_preview = False
        self._rebuild_target()
        self._replot()

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
        """Steering unit vector, measured from the loudspeaker FRONT axis (RC2 fix).

        ``theta`` is the angle OFF the front axis (``reference_axis``); ``theta = 0`` aims the
        beam straight out the front. ``phi`` rotates around the front axis in the (right, up)
        plane of the dataset's reference frame. So the default ``(0, 0)`` points where the
        speaker faces — the correct cardioid axis for an opposed-driver box (front +x, drivers
        along x), which the old world-+z default could never hit.
        """
        front, right, up = reference_frame(self._front_axis)
        th = np.deg2rad(self._steer_theta.value())
        ph = np.deg2rad(self._steer_phi.value())
        return np.cos(th) * front + np.sin(th) * (np.cos(ph) * right + np.sin(ph) * up)

    @staticmethod
    def _front_axis_from(ds: RadiationDataset) -> np.ndarray:
        """The dataset's loudspeaker front axis (``reference_axis``), normalized; +z fallback."""
        ax = ds.attrs.get("reference_axis", [0.0, 0.0, 1.0])
        if isinstance(ax, str):
            import json

            try:
                ax = json.loads(ax)
            except (json.JSONDecodeError, ValueError):
                ax = [0.0, 0.0, 1.0]
        a = np.asarray(ax, dtype=np.float64).reshape(3)
        n = float(np.linalg.norm(a))
        return a / n if n > 0 else np.array([0.0, 0.0, 1.0])

    @staticmethod
    def _axis_label(axis: np.ndarray) -> str:
        """Short label for a unit axis (``+x`` / ``-y`` / ``[..]`` for off-axis fronts)."""
        names = {
            (1, 0, 0): "+x",
            (-1, 0, 0): "-x",
            (0, 1, 0): "+y",
            (0, -1, 0): "-y",
            (0, 0, 1): "+z",
            (0, 0, -1): "-z",
        }
        key = tuple(int(round(v)) for v in axis)
        if key in names and np.allclose(axis, key, atol=1e-6):
            return names[key]
        return "[" + ", ".join(f"{v:.2f}" for v in axis) + "]"

    def _update_engine_note(self) -> None:
        """Show the delay-and-sum guidance note when it can't do the chosen target (RC3 fix)."""
        label, _, _ = _PATTERNS[self._pattern.currentIndex()]
        engine = _ENGINES[self._engine.currentIndex()][1]
        show = engine == "delay_sum" and label != "Omni"  # every non-omni target needs shaping
        # Set text even when hidden empty so the state is queryable headlessly (isVisible() is
        # False for an unshown widget regardless of setVisible).
        self._engine_note.setText(
            "⚠ Delay-and-sum only steers — it cannot shape a cardioid or hold directivity. "
            "Use Least-squares or Auto-Design."
            if show
            else ""
        )
        self._engine_note.setVisible(show)

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
        self._update_engine_note()  # refresh the delay-and-sum guidance for the new pattern
        self._on_target_changed()  # refresh the live target preview for the new pattern

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
        self._stale = False  # the achieved beam now matches the displayed target
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
        self._rebuild_target()  # target now reflects the design's own spec (not stale controls)
        self._replot()

    def _on_design_failed(self, err: str) -> None:
        self._design_btn.setEnabled(True)
        self._design_btn.setText("Design")
        QMessageBox.critical(self, "Design failed", err)

    # ----------------------------------------------------------------- plots
    def _show_achieved(self) -> bool:
        """True when a finished, still-current design exists (vs the live target preview)."""
        return self._result is not None and not self._stale

    def _display_spec(self) -> TargetSpec:
        """The spec the views currently render: the design's spec when fresh, else the controls'."""
        if self._show_achieved():
            return self._result.spec
        return self._build_spec()

    def _steer_unit(self) -> np.ndarray:
        """Normalized steering/look direction of the displayed spec (the beam's on-axis)."""
        s = np.asarray(self._display_spec().steer_dir, dtype=np.float64).reshape(3)
        n = float(np.linalg.norm(s))
        return s / n if n > 0 else np.array([0.0, 0.0, 1.0])

    def _rebuild_target(self) -> None:
        """Cache the target field for the displayed spec (one build_target shared by all views).

        The target is engine-independent (built from the pattern / steer / accept-angle), so this
        is the *requested* beam — shown live as the parameters change, before any design exists.
        """
        if self._ds is None:
            self._target = None
            return
        c_sound = float(self._ds.attrs.get("speed_of_sound", 343.2))
        self._target = build_target(
            self._display_spec(), self._ds.directions, self._ds.frequencies, c_sound=c_sound
        )

    def _on_target_changed(self) -> None:
        """Live target preview: any target-parameter change rebuilds + redraws the target (Task 2).

        Marks any prior design stale (its achieved beam no longer matches the requested target),
        disables export until the next solve, and redraws every view from the fresh target.
        """
        if self._ds is None or self._suspend_preview:
            return
        self._stale = True
        self._export_btn.setEnabled(False)
        if self._result is not None:
            self._metrics.setText(
                "Target preview — parameters changed since the last design; "
                "press Design to update the results."
            )
        self._rebuild_target()
        self._replot()

    def _resample_order(self) -> int:
        """Capped SH order for grid→arc / grid→axis resampling (matches the polar/export caps)."""
        return min(20, safe_order_for_grid(self._ds.directions.unit_vectors.shape[0]))

    def _replot(self) -> None:
        """Refresh every plot view from the current target (preview) and/or ``DesignResult``."""
        if self._ds is None:
            return
        if self._target is None:
            self._rebuild_target()
        self._replot_polar()
        self._replot_hpolar()
        self._replot_vpolar()
        self._replot_balloon()
        self._replot_sonogram()
        self._replot_metrics()
        self._replot_filters()
        self._replot_drivers()
        self._replot_cea()

    def _replot_freq_views(self) -> None:
        """Redraw only the per-frequency views (the cuts + the 3-D balloon)."""
        self._replot_polar()
        self._replot_hpolar()
        self._replot_vpolar()
        self._replot_balloon()

    def _placeholder(self, canvas, text: str) -> None:
        """Draw a centered 'press Design' message on a realization view with no target preview."""
        fig = canvas.fig
        with _quiet_redraw_warnings():
            fig.clear()
            ax = fig.add_subplot(1, 1, 1)
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            text,
            ha="center",
            va="center",
            fontsize=11,
            color="0.4",
            transform=ax.transAxes,
        )
        canvas.redraw()

    def _replot_polar(self) -> None:
        """Beam-axis polar cut at the selected frequency: target always, achieved when designed.

        The cut is the great-circle through the *steer* axis (the beam's on-axis); the target is
        shown live before any design, and the achieved beam is overlaid once a (fresh) design
        exists. Both are normalized to their own peak (dB).
        """
        if self._ds is None or self._target is None:
            return
        fi = max(self._freq_combo.currentIndex(), 0)
        obs = self._ds.directions
        order = self._resample_order()

        # Target + (when designed) achieved on the great-circle through the steer axis.
        angle, arc_uv = great_circle_arc(self._steer_unit(), 361)
        tgt_arc = resample(self._target.b_field[fi], obs, arc_uv, order)

        def norm_db(x):
            mag = np.abs(x)
            return 20.0 * np.log10(mag / (np.max(mag) + 1e-300) + 1e-6)

        ax = self._polar.ax
        ax.clear()
        ax.set_theta_zero_location("N")
        if self._show_achieved():
            achieved = resample(self._result.steered_field[fi], obs, arc_uv, order)
            ax.plot(angle, np.clip(norm_db(achieved), -40, 0), label="achieved", lw=2)
        ax.plot(angle, np.clip(norm_db(tgt_arc), -40, 0), "--", label="target", lw=1)
        ax.set_ylim(-40, 0)
        mode = "achieved vs target" if self._show_achieved() else "target preview"
        ax.set_title(
            f"Beam-axis polar @ {self._ds.frequencies[fi]:.0f} Hz ({mode}, dB norm.)", fontsize=9
        )
        ax.legend(loc="lower left", fontsize=8)
        self._polar.redraw()

    # ---- full-system pattern views (referenced to the loudspeaker front, like the Results tab) --
    def _system_field(self) -> np.ndarray:
        """The full-system field to draw: the achieved steered beam when fresh, else the target.

        Both are ``[F, N]`` complex on the dataset grid: ``steered_field = sum_m w_m H_m`` (all
        drivers with their filters applied) when a current design exists, otherwise the requested
        ``target.b_field``. A fresh allocation either way — the stored H-tensor is never touched.
        """
        if self._show_achieved():
            return self._result.steered_field
        return self._target.b_field

    def _replot_cut(self, canvas: _MplCanvas, plane_code: str, plane_name: str) -> None:
        """Full-system H or V polar cut at the selected frequency (front-axis referenced).

        Mirrors the Results-tab polar: the cut is the great-circle in the loudspeaker's horizontal
        (``"H"``) or vertical (``"V"``) plane about the dataset front axis, SH-resampled and
        normalized to its own peak. The target is overlaid (dashed) so the requested vs achieved
        shape is visible even before / after a design.
        """
        if self._ds is None or self._target is None:
            return
        fi = max(self._freq_combo.currentIndex(), 0)
        obs = self._ds.directions
        ang, arc_uv = _plane_arc(self._ds, plane_code)  # [A], [A, 3]
        order = _arc_order(obs.unit_vectors.shape[0])

        def cut_norm_db(field_fn: np.ndarray) -> np.ndarray:
            p = resample(field_fn[fi], obs, arc_uv, order)  # [A] complex
            mag_db = _db(p)
            return mag_db - mag_db.max()  # 0 dB at the loudest direction on the cut

        ax = canvas.ax
        ax.cla()
        ax.set_theta_zero_location("N")  # 0° (front axis) at the top
        ax.set_theta_direction(-1)
        ax.plot(
            np.deg2rad(ang),
            np.clip(cut_norm_db(self._target.b_field), -40, 1),
            "--",
            color="0.35",
            lw=1.2,
            label="target",
        )
        if self._show_achieved():
            ax.plot(
                np.deg2rad(ang),
                np.clip(cut_norm_db(self._result.steered_field), -40, 1),
                "b-",
                lw=1.6,
                label="achieved (system)",
            )
        ax.set_ylim(-40.0, 1.0)
        ax.set_yticks([-30, -20, -10, 0])
        ax.grid(True, alpha=0.4)
        mode = "achieved vs target" if self._show_achieved() else "target preview"
        ax.set_title(
            f"{plane_name} polar — full system @ {self._ds.frequencies[fi]:.0f} Hz\n{mode}",
            fontsize=9,
        )
        ax.legend(loc="lower left", fontsize=8)
        canvas.redraw()

    def _replot_hpolar(self) -> None:
        """Full-system horizontal polar cut (front-axis referenced)."""
        self._replot_cut(self._hpolar, "H", "Horizontal")

    def _replot_vpolar(self) -> None:
        """Full-system vertical polar cut (front-axis referenced)."""
        self._replot_cut(self._vpolar, "V", "Vertical")

    def _replot_balloon(self) -> None:
        """Full-system 3-D balloon at the selected frequency (achieved when designed, else target).

        Magnitude as radial distance on the sphere directions, colour = dB; the dataset front
        (0°) axis is drawn as the reference indicator. Mirrors the Results-tab balloon but for the
        combined steered field (or the requested target before a design).
        """
        if self._ds is None or self._target is None:
            return
        fi = max(self._freq_combo.currentIndex(), 0)
        field = self._system_field()[fi]  # [N] complex
        uvecs = self._ds.directions.unit_vectors  # [N, 3]
        mag = np.abs(field)
        mag_norm = mag / (mag.max() + 1e-300)
        xyz = uvecs * mag_norm[:, None]  # [N, 3]

        ax = self._balloon.ax
        ax.cla()
        sc = ax.scatter(
            xyz[:, 0], xyz[:, 1], xyz[:, 2], c=_db(field), cmap="jet", s=60, depthshade=True
        )
        if self._balloon_cbar is None:
            self._balloon_cbar = self._balloon.fig.colorbar(sc, ax=ax, shrink=0.6, label="dB SPL")
        else:
            self._balloon_cbar.update_normal(sc)

        axis = _reference_axis(self._ds)
        norm = float(np.linalg.norm(axis))
        if norm > 0:
            a = axis / norm * 1.15
            ax.plot([0, a[0]], [0, a[1]], [0, a[2]], "k--", linewidth=1.5)
            ax.text(a[0], a[1], a[2], "  0° axis", fontsize=8)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")  # type: ignore[attr-defined]
        mode = "achieved (system)" if self._show_achieved() else "target preview"
        ax.set_title(
            f"Balloon — full system @ {self._ds.frequencies[fi]:.0f} Hz · {mode}", fontsize=9
        )
        self._balloon.redraw()

    def _sonogram_data(self, field: np.ndarray, plane: str) -> tuple[np.ndarray, np.ndarray]:
        """Return (angles_deg [A], mag_db [F, A]) for one plane's SH-resampled sonogram."""
        ang, arc_uv = _plane_arc(self._ds, plane)  # [A], [A, 3]
        order = _arc_order(self._ds.directions.unit_vectors.shape[0])
        p_arc = resample(field, self._ds.directions, arc_uv, order)  # [F, A] complex
        return ang, _db(p_arc)

    def _replot_sonogram(self) -> None:
        """Full-system H/V directivity sonograms (achieved when designed, else target).

        Two stacked maps (horizontal + vertical), log frequency × angle, colour = normalized dB,
        SH-resampled per frequency. Mirrors the Results-tab sonograms for the combined field.
        """
        if self._ds is None or self._target is None:
            return
        field = self._system_field()  # [F, N]
        freqs = self._ds.frequencies  # [F]
        ang_h, db_h = self._sonogram_data(field, "H")
        ang_v, db_v = self._sonogram_data(field, "V")
        vmax = float(max(db_h.max(), db_v.max()))  # shared 0-dB reference across both planes

        fig = self._sonogram_canvas.fig
        with _quiet_redraw_warnings():
            fig.clear()
            ax_h = fig.add_subplot(2, 1, 1)
            ax_v = fig.add_subplot(2, 1, 2, sharex=ax_h)
        mesh = None
        for ax, ang, db, title in (
            (ax_h, ang_h, db_h, "Horizontal"),
            (ax_v, ang_v, db_v, "Vertical"),
        ):
            FF, AA = np.meshgrid(freqs, ang)  # [A, F]
            mesh = ax.pcolormesh(
                FF, AA, (db - vmax).T, shading="gouraud", cmap="jet", vmin=-30.0, vmax=0.0
            )
            ax.set_xscale("log")
            ax.set_ylabel(f"{title}\nangle (°)")
            ax.set_yticks([-180, -90, 0, 90, 180])
        ax_v.set_xlabel("Frequency (Hz)")
        mode = "achieved (system)" if self._show_achieved() else "target preview"
        ax_h.set_title(f"Directivity sonograms — full system · {mode}", fontsize=9)
        if mesh is not None:
            fig.colorbar(mesh, ax=[ax_h, ax_v], label="normalised dB")
        self._sonogram_canvas.redraw()

    def _replot_metrics(self) -> None:
        """Directivity dashboard: DI, -6 dB beamwidth, and WNG vs frequency (read from metrics).

        Three stacked panels share the log-frequency axis. The DI panel carries the band's
        magnitude target-error on a twin axis (the most direct achieved-vs-target number, since
        the polar only shows one frequency); multi-target DI / beamwidth targets are dashed
        reference lines. The WNG panel draws the requested floor and flags infeasible bins (where
        the solver could not meet that floor) in red — the honest "where physics won" markers.

        Before a design (preview), only the *target* reference lines are knowable (target DI /
        beamwidth where set, the WNG floor); the achieved curves require the solved weights.
        """
        if self._ds is None:
            return
        if not self._show_achieved():
            self._replot_metrics_preview()
            return
        freqs = np.asarray(self._ds.frequencies, dtype=np.float64)  # [F]
        m = self._result.metrics
        spec = self._result.spec
        di = np.asarray(m["di_db"], dtype=np.float64)  # [F]
        bw = np.asarray(m["beamwidth_deg"], dtype=np.float64)  # [F] (nan where lobe not closed)
        wng = np.asarray(m["wng_db"], dtype=np.float64)  # [F]
        wng_plot = np.where(np.isfinite(wng), wng, np.nan)  # -inf (collapsed bins) -> gap
        feas = np.asarray(m["feasible_mask"], dtype=bool)  # [F]
        te = m.get("target_error_db")  # [F] | None

        fig = self._metrics_canvas.fig
        with _quiet_redraw_warnings():
            fig.clear()
            ax_di = fig.add_subplot(3, 1, 1)
            ax_bw = fig.add_subplot(3, 1, 2, sharex=ax_di)
            ax_wng = fig.add_subplot(3, 1, 3, sharex=ax_di)

        ax_di.semilogx(freqs, di, "-o", ms=3, color="tab:blue", label="achieved DI")
        if spec.target_di_db is not None:  # multi-target DI objective
            ax_di.axhline(spec.target_di_db, ls="--", color="tab:blue", lw=1, label="target DI")
        if te is not None:  # achieved-vs-target magnitude error on a twin axis
            ax_te = ax_di.twinx()
            ax_te.semilogx(
                freqs, np.asarray(te, float), ":", color="0.5", lw=1.2, label="target err"
            )
            ax_te.set_ylabel("target err (dB)", color="0.4", fontsize=8)
            ax_te.tick_params(axis="y", labelsize=7, colors="0.4")
        ax_di.set_ylabel("DI (dB)")
        ax_di.set_title("Achieved directivity / beamwidth / robustness vs frequency", fontsize=9)
        ax_di.legend(fontsize=7, loc="best")

        ax_bw.semilogx(freqs, bw, "-o", ms=3, color="tab:green", label="-6 dB beamwidth")
        if spec.target_beamwidth_deg is not None:  # multi-target beamwidth objective
            ax_bw.axhline(
                spec.target_beamwidth_deg, ls="--", color="tab:green", lw=1, label="target BW"
            )
            ax_bw.legend(fontsize=7, loc="best")
        ax_bw.set_ylabel("beamwidth (°)")

        ax_wng.semilogx(freqs, wng_plot, "-o", ms=3, color="tab:purple", label="achieved WNG")
        ax_wng.axhline(spec.wng_floor_db, ls="--", color="tab:red", lw=1, label="WNG floor")
        if np.any(~feas):  # honest: bins where the WNG floor could not be met
            ax_wng.plot(
                freqs[~feas], wng_plot[~feas], "x", color="red", ms=7, label="infeasible bin"
            )
        ax_wng.set_ylabel("WNG (dB)")
        ax_wng.set_xlabel("Frequency (Hz)")
        ax_wng.legend(fontsize=7, loc="best")

        for ax in (ax_di, ax_bw, ax_wng):
            ax.grid(True, which="both", alpha=0.3)
        self._metrics_canvas.redraw()

    def _replot_metrics_preview(self) -> None:
        """Preview of the directivity dashboard: the target reference lines only (no achieved yet).

        Draws what the spec alone fixes — the WNG floor, plus the multi-target DI / beamwidth
        targets where set — on the same three-panel layout, with a 'press Design' note. The
        achieved DI / beamwidth / WNG curves need the solved weights and appear after Design.
        """
        spec = self._display_spec()
        freqs = np.asarray(self._ds.frequencies, dtype=np.float64)  # [F] — for a meaningful x-range
        fig = self._metrics_canvas.fig
        with _quiet_redraw_warnings():
            fig.clear()
            ax_di = fig.add_subplot(3, 1, 1)
            ax_bw = fig.add_subplot(3, 1, 2, sharex=ax_di)
            ax_wng = fig.add_subplot(3, 1, 3, sharex=ax_di)

        if spec.target_di_db is not None:  # multi-target DI objective
            ax_di.axhline(spec.target_di_db, ls="--", color="tab:blue", lw=1, label="target DI")
            ax_di.legend(fontsize=7, loc="best")
        ax_di.set_ylabel("DI (dB)")
        ax_di.set_title(
            "Target preview — press Design for achieved DI / beamwidth / WNG", fontsize=9
        )

        if spec.target_beamwidth_deg is not None:  # multi-target beamwidth objective
            ax_bw.axhline(
                spec.target_beamwidth_deg, ls="--", color="tab:green", lw=1, label="target BW"
            )
            ax_bw.legend(fontsize=7, loc="best")
        ax_bw.set_ylabel("beamwidth (°)")

        ax_wng.axhline(spec.wng_floor_db, ls="--", color="tab:red", lw=1, label="WNG floor")
        ax_wng.legend(fontsize=7, loc="best")
        ax_wng.set_ylabel("WNG (dB)")
        ax_wng.set_xlabel("Frequency (Hz)")

        for ax in (ax_di, ax_bw, ax_wng):
            ax.set_xscale("log")
            ax.set_xlim(freqs[0], freqs[-1])  # the real band, not the empty-axis 1–10 Hz default
            ax.grid(True, which="both", alpha=0.3)
        self._metrics_canvas.redraw()

    def _replot_filters(self) -> None:
        """Per-driver filter weights ``w_m(f)``: magnitude (dB, re the loudest weight) + phase.

        This is the *filter* itself — the complex multiplier the beamformer applies to each
        driver — read straight from ``result.weights`` (never recomputed). Magnitude is shown
        relative to the largest weight in the design (weights are dimensionless gains); the phase
        is unwrapped per driver for legibility. Phase is the filter's own phase and is plotted
        as-is — the steering lives in H's inter-driver phase, not re-zeroed here (cardinal rule).
        """
        if self._ds is None:
            return
        if not self._show_achieved():  # the filter weights only exist once a design is solved
            self._placeholder(
                self._filter_canvas, "Per-driver filter weights\n\nPress “Design” to compute them."
            )
            return
        freqs = np.asarray(self._ds.frequencies, dtype=np.float64)  # [F]
        weights = self._result.weights  # [M, F] complex128
        ids = [d.driver_id for d in self._ds.drivers]
        ref = float(np.max(np.abs(weights))) + _MAG_FLOOR  # 0 dB at the loudest driver/bin

        fig = self._filter_canvas.fig
        with _quiet_redraw_warnings():
            fig.clear()
            ax_mag = fig.add_subplot(2, 1, 1)
            ax_ph = fig.add_subplot(2, 1, 2, sharex=ax_mag)
        for mi, did in enumerate(ids):
            w = weights[mi]  # [F]
            mag_db = 20.0 * np.log10(np.abs(w) / ref + _MAG_FLOOR)
            ax_mag.semilogx(freqs, mag_db, "-o", ms=3, label=did)
            ax_ph.semilogx(freqs, np.degrees(np.unwrap(np.angle(w))), "-o", ms=3, label=did)
        ax_mag.set_ylabel("|w| (dB re max)")
        ax_mag.set_title("Per-driver filter weights (magnitude + phase)", fontsize=9)
        ax_mag.legend(fontsize=7, ncol=2, loc="best")
        ax_ph.set_ylabel("phase (°, unwrapped)")
        ax_ph.set_xlabel("Frequency (Hz)")
        for ax in (ax_mag, ax_ph):
            ax.grid(True, which="both", alpha=0.3)
        self._filter_canvas.redraw()

    def _replot_drivers(self) -> None:
        """Filtered per-driver on-axis responses ``w_m(f)·H_full[m]`` + the combined steered beam.

        Each driver's *radiated* on-axis contribution (its filter baked into its measured H_full),
        SH-resampled to the steer direction, plus the achieved combined response — so the user sees
        how the units sum to the beam. Phase is referenced to the global origin exactly as stored
        (cardinal rule): ``stacked_h_full`` is read-only and nothing is re-zeroed per driver.
        """
        if self._ds is None:
            return
        if not self._show_achieved():  # the per-driver filtered contributions need solved weights
            self._placeholder(
                self._driver_canvas,
                "Filtered per-driver responses + combined beam\n\nPress “Design” to compute them.",
            )
            return
        obs = self._ds.directions
        freqs = np.asarray(self._ds.frequencies, dtype=np.float64)  # [F]
        order = self._resample_order()
        steer_uv = self._steer_unit()[None, :]  # [1, 3]
        h = stacked_h_full(self._ds)  # [M, F, N] complex128 (fresh stack; stored H untouched)
        weights = self._result.weights  # [M, F]
        ids = [d.driver_id for d in self._ds.drivers]

        fig = self._driver_canvas.fig
        with _quiet_redraw_warnings():
            fig.clear()
            ax_mag = fig.add_subplot(2, 1, 1)
            ax_ph = fig.add_subplot(2, 1, 2, sharex=ax_mag)
        for mi, did in enumerate(ids):
            filtered = weights[mi, :, None] * h[mi]  # [F, N] = w_m(f) * H_full[m]
            on_axis = resample(filtered, obs, steer_uv, order)[:, 0]  # [F] on the steer axis
            ax_mag.semilogx(freqs, _db(on_axis), "-", lw=1, label=did)
            ax_ph.semilogx(freqs, np.degrees(np.angle(on_axis)), "-", lw=1, label=did)
        combined = resample(self._result.steered_field, obs, steer_uv, order)[:, 0]  # [F]
        ax_mag.semilogx(freqs, _db(combined), "k-", lw=2.2, label="combined")
        ax_ph.semilogx(freqs, np.degrees(np.angle(combined)), "k-", lw=2.2, label="combined")
        ax_mag.set_ylabel("on-axis (dB SPL)")
        ax_mag.set_title("Filtered per-driver responses + combined (on the steer axis)", fontsize=9)
        ax_mag.legend(fontsize=7, ncol=2, loc="best")
        ax_ph.set_ylabel("phase (°)")
        ax_ph.set_xlabel("Frequency (Hz)")
        for ax in (ax_mag, ax_ph):
            ax.grid(True, which="both", alpha=0.3)
        self._driver_canvas.redraw()

    def _replot_cea(self) -> None:
        """CEA-2034-A spinorama of the steered field — the in-room deliverable.

        Computed from the *frozen* ``steered_field`` (a display derivation, not a re-solve), with
        the spinorama referenced to the BEAM axis (``steer_dir``) — on-axis for the listener and
        consistent with the in-room slope the multi-target report already shows. SPL curves on the
        left axis, the two directivity indices on the right; the Estimated In-Room curve is bold.

        Before a design, the *target* spinorama is previewed (the requested beam's predicted
        in-room); once a design exists, the achieved ``steered_field`` is shown instead. Either way
        the spinorama is referenced to the BEAM axis (``steer_dir``), consistent with the in-room
        slope the multi-target report shows.
        """
        if self._ds is None or self._target is None:
            return
        field = self._result.steered_field if self._show_achieved() else self._target.b_field
        curves = compute_cea2034(
            field,
            self._ds.frequencies,
            self._ds.directions,
            self._steer_unit(),  # reference the spinorama to the BEAM axis, not the dataset front
        )
        freqs = curves["frequencies"]

        fig = self._cea_canvas.fig
        with _quiet_redraw_warnings():
            fig.clear()
            ax = fig.add_subplot(1, 1, 1)
            ax2 = ax.twinx()
        for key in SPL_CURVES:
            lw = 2.4 if key == "estimated_in_room" else 1.3
            ax.semilogx(freqs, curves[key], color=_CEA_COLORS[key], lw=lw, label=_CEA_LABELS[key])
        for key in DI_CURVES:
            ax2.semilogx(
                freqs, curves[key], color=_CEA_COLORS[key], lw=1.1, ls="--", label=_CEA_LABELS[key]
            )
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("dB SPL (re 20 µPa)")
        ax2.set_ylabel("Directivity Index (dB)")
        ax.grid(True, which="both", alpha=0.3)
        which = "steered beam" if self._show_achieved() else "target preview"
        ax.set_title(f"CEA-2034-A spinorama — {which} (in-room on beam axis)", fontsize=9)
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, fontsize=6, loc="lower center", ncol=4)
        self._cea_canvas.redraw()

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
