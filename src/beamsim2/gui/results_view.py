"""Results tab: on-axis, polar, 3-D balloon, directivity sonograms, CEA2034, export.

All plots use ``matplotlib`` via ``FigureCanvasQTAgg`` — no new dependency.

The tab can be used as a standalone HDF5 viewer (File → Open dataset) without running a
new solve — this is also the GUI smoke-test entry point.

Chunk 2 (`docs/Bug_Fix_Proposal.md` #9/#10/#11 + far-field display) reworked the
directional views to be **trustworthy**, building on Chunk 1's proven `reference_axis`:

* **Polar** (#10): SH-resampled 361-point great-circle arcs (smooth), in the horizontal
  or vertical plane built from the reference axis — not a handful of scattered points.
* **Directivity sonograms** (#9): separate horizontal & vertical maps, **log** frequency
  axis, angle −180..180°, colour = normalised dB — SH-resampled per frequency.
* **CEA2034 / spinorama** (#11): on-axis, listening window, early reflections, sound
  power, the two DI curves, and the estimated in-room response (`metrics.cea2034`).
* **H_bem vs H_full** (#11): a field selector + tooltip on every directional view.
* **Field referencing** (far-field option): a display-only Near-field / acoustic-center /
  SH-extrapolation toggle (`core.field_referencing`) that NEVER mutates the stored tensor.

Build-order item 10, Tab 4 (§6 Gameplan — results display and export).
"""

from __future__ import annotations

import contextlib
import warnings
from typing import Optional

import matplotlib
import numpy as np
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from beamsim2.assembly.tensor import RadiationDataset
from beamsim2.core.field_referencing import (
    NEAR_FIELD,
    REFERENCING_MODES,
    apply_referencing,
)
from beamsim2.core.sh_transform import resample, safe_order_for_grid
from beamsim2.core.sphere import (
    DEFAULT_REFERENCE_AXIS,
    nearest_direction_index,
    reference_frame,
)
from beamsim2.metrics.cea2034 import DI_CURVES, SPL_CURVES, compute_cea2034

_P_REF = 20e-6  # Pa — reference SPL
_ARC_POINTS = 361  # great-circle arc resolution for polar / sonogram resampling

_FIELD_TOOLTIP = (
    "Which per-driver response is plotted (DATA_CONTRACT.md §3.1):\n"
    "• H_full = H_bem × terminal_response (T/S low-freq + voice-coil inductance).\n"
    "    The Phase-2 default — what the driver actually radiates at its terminals.\n"
    "• H_bem = raw BEM at unit cone velocity — geometry / baffle / box / diffraction\n"
    "    only, with no electrical/mechanical driver response applied."
)
_REF_TOOLTIP = (
    "Display referencing — a view transform that NEVER alters the stored H-tensor:\n"
    "• Near-field (as solved): the BEM field on the observation sphere, phase referenced\n"
    "    to the global origin (the form Phase-2 beamforming consumes).\n"
    "• Far-field: acoustic-center: removes 1/r spreading + path-length phase about each\n"
    "    driver's position, so an offset source reads as if measured about its own centre.\n"
    "• Far-field: SH extrapolation: the true r→∞ radiating pattern via spherical-harmonic\n"
    "    Hankel ratios.\n"
    "Both far-field modes make a low-frequency single driver read near-omni."
)


@contextlib.contextmanager
def _quiet_redraw_warnings():
    """Suppress benign matplotlib transient-redraw warnings.

    Clearing/redrawing a log-scaled axis momentarily autoscales it to a non-positive
    range, and a colorbar spanning multiple Axes is not ``tight_layout``-compatible —
    both are cosmetic and resolve once the real data is drawn.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*non-positive xlim on a log-scaled axis.*")
        warnings.filterwarnings("ignore", message=".*not compatible with tight_layout.*")
        yield


def _db(pressure: np.ndarray) -> np.ndarray:
    """Magnitude in dB SPL re 20 µPa."""
    return 20.0 * np.log10(np.abs(pressure) / _P_REF + 1e-300)


def _reference_axis(ds: RadiationDataset) -> np.ndarray:
    """Return the dataset's measurement/reference axis (default +z).

    Reads the ``reference_axis`` root attr written by the pipeline; falls back to
    +z for legacy files that predate it.  Used so the directional views pick the
    loudspeaker front instead of hardcoding +z.
    """
    axis = ds.attrs.get("reference_axis", DEFAULT_REFERENCE_AXIS)
    return np.asarray(axis, dtype=np.float64).reshape(3)


def _referenced_field(ds: RadiationDataset, drv, field_name: str, mode: str) -> np.ndarray:
    """Pick ``H_full``/``H_bem`` for a driver and apply the display referencing mode.

    Returns a NEW ``[F, N]`` array; the stored ``DriverData`` is never modified
    (cardinal rule — DATA_CONTRACT.md §3.4).
    """
    H = drv.H_full if field_name == "H_full" else drv.H_bem  # [F, N]
    pos = drv.attrs.get("position")
    position = np.asarray(pos, dtype=np.float64).reshape(3) if pos is not None else None
    c = float(ds.attrs.get("speed_of_sound", 343.2))
    return apply_referencing(
        H, mode, frequencies=ds.frequencies, obs=ds.directions, position=position, c=c
    )


def _plane_arc(ds: RadiationDataset, plane: str) -> tuple[np.ndarray, np.ndarray]:
    """Great-circle arc (angles_deg [A], unit_vectors [A, 3]) in the H or V plane.

    The plane is built from the dataset's reference axis: ``plane == "H"`` sweeps the
    horizontal (front × right) great circle, ``"V"`` the vertical (front × up) one.
    """
    front, right, up = reference_frame(_reference_axis(ds))
    inplane = right if plane == "H" else up
    ang = np.linspace(-180.0, 180.0, _ARC_POINTS)  # [A]
    rad = np.deg2rad(ang)
    uv = np.cos(rad)[:, None] * front[None, :] + np.sin(rad)[:, None] * inplane[None, :]  # [A, 3]
    return ang, uv


def _arc_order(n_points: int) -> int:
    """SH order for grid→arc resampling (capped for speed on dense grids)."""
    return min(safe_order_for_grid(n_points), 19)


class _MplCanvas(QWidget):
    """A matplotlib Figure embedded as a Qt widget."""

    def __init__(
        self,
        nrows: int = 1,
        ncols: int = 1,
        projection: Optional[str] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._fig = Figure(tight_layout=True)
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._canvas)
        if projection:
            self._ax = self._fig.add_subplot(nrows, ncols, 1, projection=projection)
        else:
            self._ax = self._fig.add_subplot(nrows, ncols, 1)

    @property
    def ax(self):
        return self._ax

    @property
    def fig(self):
        return self._fig

    def redraw(self) -> None:
        with _quiet_redraw_warnings():
            self._fig.tight_layout()
            self._canvas.draw()


class _ReferencedView(QWidget):
    """Base for views that honour the dataset, a field selector, and a referencing mode."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._ds: Optional[RadiationDataset] = None
        self._mode: str = NEAR_FIELD

    def set_referencing(self, mode: str) -> None:
        """Set the display referencing mode and replot (called by the ResultsTab combo)."""
        self._mode = mode
        if self._ds is not None:
            self._replot()

    def _populate_drivers(self, combo: QComboBox, ds: RadiationDataset) -> None:
        combo.blockSignals(True)
        combo.clear()
        for d in ds.drivers:
            combo.addItem(d.driver_id)
        combo.blockSignals(False)

    def _populate_freqs(self, combo: QComboBox, ds: RadiationDataset) -> None:
        combo.blockSignals(True)
        combo.clear()
        for f in ds.frequencies:
            combo.addItem(f"{f:.0f} Hz")
        combo.setCurrentIndex(len(ds.frequencies) // 2)
        combo.blockSignals(False)

    def _field_combo_widget(self) -> QComboBox:
        combo = QComboBox()
        combo.addItems(["H_full", "H_bem"])
        combo.setToolTip(_FIELD_TOOLTIP)
        combo.currentIndexChanged.connect(self._replot)
        return combo

    def _replot(self) -> None:  # pragma: no cover - overridden
        raise NotImplementedError


# ---------------------------------------------------------------------------
# On-axis sub-tab
# ---------------------------------------------------------------------------


class _OnAxisView(_ReferencedView):
    """On-axis frequency response (magnitude + phase) for one selected driver."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Driver:"))
        self._drv_combo = QComboBox()
        self._drv_combo.currentIndexChanged.connect(self._replot)
        ctrl.addWidget(self._drv_combo)
        ctrl.addWidget(QLabel("Field:"))
        self._field_combo = self._field_combo_widget()
        ctrl.addWidget(self._field_combo)
        ctrl.addStretch()
        layout.addLayout(ctrl)

        self._canvas = _MplCanvas(nrows=2)
        self._ax_mag = self._canvas.fig.add_subplot(2, 1, 1)
        self._ax_ph = self._canvas.fig.add_subplot(2, 1, 2, sharex=self._ax_mag)
        self._canvas.fig.delaxes(self._canvas.ax)
        layout.addWidget(self._canvas)

    def load(self, ds: RadiationDataset) -> None:
        self._ds = ds
        self._populate_drivers(self._drv_combo, ds)
        self._replot()

    def _replot(self) -> None:
        if self._ds is None:
            return
        idx = self._drv_combo.currentIndex()
        if idx < 0 or idx >= len(self._ds.drivers):
            return

        drv = self._ds.drivers[idx]
        field_name = self._field_combo.currentText()
        H = _referenced_field(self._ds, drv, field_name, self._mode)  # [F, N]
        freqs = self._ds.frequencies  # [F]
        conv = drv.convergence_flags  # [F] bool

        # On-axis: pick the direction nearest the dataset's reference axis (never +z).
        uvecs = self._ds.directions.unit_vectors  # [N, 3]
        on_ax_idx = nearest_direction_index(uvecs, _reference_axis(self._ds))
        self._last_on_axis_idx = on_ax_idx  # exposed for tests / debugging
        h_onaxis = H[:, on_ax_idx]  # [F] complex128

        with _quiet_redraw_warnings():
            self._ax_mag.cla()
            self._ax_ph.cla()

        converged = conv
        flagged = ~conv
        if np.any(converged):
            self._ax_mag.semilogx(freqs[converged], _db(h_onaxis[converged]), "b-", linewidth=1.5)
            self._ax_ph.semilogx(
                freqs[converged], np.degrees(np.angle(h_onaxis[converged])), "b-", linewidth=1.5
            )
        if np.any(flagged):
            self._ax_mag.semilogx(
                freqs[flagged],
                _db(h_onaxis[flagged]),
                "o",
                color="darkorange",
                label="non-converged",
                markersize=5,
            )
            self._ax_ph.semilogx(
                freqs[flagged],
                np.degrees(np.angle(h_onaxis[flagged])),
                "o",
                color="darkorange",
                markersize=5,
            )
            self._ax_mag.legend(fontsize=8)

        self._ax_mag.set_ylabel("dB SPL (re 20 µPa)")
        self._ax_mag.set_title(f"On-axis — {drv.driver_id} [{field_name}] · {self._mode}")
        self._ax_mag.grid(True, which="both", alpha=0.4)

        self._ax_ph.set_xlabel("Frequency (Hz)")
        self._ax_ph.set_ylabel("Phase (°)")
        self._ax_ph.grid(True, which="both", alpha=0.4)

        self._canvas.fig.tight_layout()
        self._canvas.redraw()


# ---------------------------------------------------------------------------
# Polar sub-tab (SH-resampled arcs)
# ---------------------------------------------------------------------------


class _PolarView(_ReferencedView):
    """Horizontal or vertical polar plot at a selectable frequency (SH-resampled arc)."""

    def __init__(self, plane: str = "Horizontal", parent=None):
        super().__init__(parent)
        self._plane = plane  # "Horizontal" or "Vertical"
        self._plane_code = "H" if plane == "Horizontal" else "V"
        layout = QVBoxLayout(self)

        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Driver:"))
        self._drv_combo = QComboBox()
        self._drv_combo.currentIndexChanged.connect(self._replot)
        ctrl.addWidget(self._drv_combo)
        ctrl.addWidget(QLabel("Field:"))
        self._field_combo = self._field_combo_widget()
        ctrl.addWidget(self._field_combo)
        ctrl.addWidget(QLabel("Frequency:"))
        self._freq_combo = QComboBox()
        self._freq_combo.currentIndexChanged.connect(self._replot)
        ctrl.addWidget(self._freq_combo)
        ctrl.addStretch()
        layout.addLayout(ctrl)

        self._canvas = _MplCanvas(projection="polar")
        layout.addWidget(self._canvas)

    def load(self, ds: RadiationDataset) -> None:
        self._ds = ds
        self._populate_drivers(self._drv_combo, ds)
        self._populate_freqs(self._freq_combo, ds)
        self._replot()

    def _replot(self) -> None:
        if self._ds is None:
            return
        drv_idx = self._drv_combo.currentIndex()
        freq_idx = self._freq_combo.currentIndex()
        if drv_idx < 0 or freq_idx < 0:
            return

        drv = self._ds.drivers[drv_idx]
        field_name = self._field_combo.currentText()
        H = _referenced_field(self._ds, drv, field_name, self._mode)  # [F, N]
        H_f = H[freq_idx, :]  # [N] complex128

        ang, arc_uv = _plane_arc(self._ds, self._plane_code)  # [A], [A, 3]
        order = _arc_order(self._ds.directions.unit_vectors.shape[0])
        p_arc = resample(H_f, self._ds.directions, arc_uv, order)  # [A] complex
        mag_db = _db(p_arc)
        mag_db_norm = mag_db - mag_db.max()  # 0 dB at the loudest direction
        self._last_arc = (ang, mag_db_norm)  # exposed for tests / debugging

        ax = self._canvas.ax
        ax.cla()
        ax.set_theta_zero_location("N")  # 0° (on-axis) at the top
        ax.set_theta_direction(-1)
        ax.plot(np.deg2rad(ang), mag_db_norm, "b-", linewidth=1.5)
        ax.set_ylim(-40.0, 1.0)
        ax.set_yticks([-30, -20, -10, 0])
        ax.grid(True, alpha=0.4)
        ax.set_title(
            f"{self._plane} polar — {drv.driver_id} [{field_name}] @ "
            f"{self._ds.frequencies[freq_idx]:.0f} Hz\n{self._mode}",
            fontsize=9,
        )
        self._canvas.redraw()


# ---------------------------------------------------------------------------
# 3-D balloon sub-tab
# ---------------------------------------------------------------------------


class _BalloonView(_ReferencedView):
    """3-D balloon: magnitude as radial distance on the sphere directions."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Driver:"))
        self._drv_combo = QComboBox()
        self._drv_combo.currentIndexChanged.connect(self._replot)
        ctrl.addWidget(self._drv_combo)
        ctrl.addWidget(QLabel("Field:"))
        self._field_combo = self._field_combo_widget()
        ctrl.addWidget(self._field_combo)
        ctrl.addWidget(QLabel("Frequency:"))
        self._freq_combo = QComboBox()
        self._freq_combo.currentIndexChanged.connect(self._replot)
        ctrl.addWidget(self._freq_combo)
        ctrl.addStretch()
        layout.addLayout(ctrl)

        self._canvas = _MplCanvas(projection="3d")
        layout.addWidget(self._canvas)
        self._cbar = None

    def load(self, ds: RadiationDataset) -> None:
        self._ds = ds
        self._populate_drivers(self._drv_combo, ds)
        self._populate_freqs(self._freq_combo, ds)
        self._replot()

    def _replot(self) -> None:
        if self._ds is None:
            return
        drv_idx = self._drv_combo.currentIndex()
        freq_idx = self._freq_combo.currentIndex()
        if drv_idx < 0 or freq_idx < 0:
            return

        drv = self._ds.drivers[drv_idx]
        field_name = self._field_combo.currentText()
        H = _referenced_field(self._ds, drv, field_name, self._mode)  # [F, N]
        H_f = H[freq_idx, :]  # [N]
        uvecs = self._ds.directions.unit_vectors  # [N, 3]

        mag = np.abs(H_f)
        mag_norm = mag / (mag.max() + 1e-300)
        xyz = uvecs * mag_norm[:, None]  # [N, 3]

        ax = self._canvas.ax
        ax.cla()
        sc = ax.scatter(
            xyz[:, 0], xyz[:, 1], xyz[:, 2], c=_db(H_f), cmap="jet", s=60, depthshade=True
        )
        if self._cbar is None:
            self._cbar = self._canvas.fig.colorbar(sc, ax=ax, shrink=0.6, label="dB SPL")
        else:
            self._cbar.update_normal(sc)

        # Reference-axis (0°/on-axis) indicator along the dataset's measurement axis.
        axis = _reference_axis(self._ds)
        norm = float(np.linalg.norm(axis))
        if norm > 0:
            a = axis / norm * 1.15
            ax.plot([0, a[0]], [0, a[1]], [0, a[2]], "k--", linewidth=1.5)
            ax.text(a[0], a[1], a[2], "  0° axis", fontsize=8)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")  # type: ignore[attr-defined]
        ax.set_title(
            f"Balloon — {drv.driver_id} [{field_name}] @ "
            f"{self._ds.frequencies[freq_idx]:.0f} Hz · {self._mode}",
            fontsize=9,
        )
        self._canvas.redraw()


# ---------------------------------------------------------------------------
# Directivity sonograms sub-tab (H + V, log frequency)
# ---------------------------------------------------------------------------


class _DirectivityMapView(_ReferencedView):
    """Directivity sonograms: horizontal & vertical planes, log frequency × angle, dB colour.

    Each plane is SH-resampled onto a fine great-circle arc per frequency, so the maps are
    smooth and physically meaningful (not a θ-sort of scattered points on a linear axis).
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Driver:"))
        self._drv_combo = QComboBox()
        self._drv_combo.currentIndexChanged.connect(self._replot)
        ctrl.addWidget(self._drv_combo)
        ctrl.addWidget(QLabel("Field:"))
        self._field_combo = self._field_combo_widget()
        ctrl.addWidget(self._field_combo)
        ctrl.addStretch()
        layout.addLayout(ctrl)

        self._canvas = _MplCanvas()
        layout.addWidget(self._canvas)

    def load(self, ds: RadiationDataset) -> None:
        self._ds = ds
        self._populate_drivers(self._drv_combo, ds)
        self._replot()

    def _sonogram(self, H: np.ndarray, plane: str) -> tuple[np.ndarray, np.ndarray]:
        """Return (angles_deg [A], mag_db [F, A]) for one plane's SH-resampled sonogram."""
        ang, arc_uv = _plane_arc(self._ds, plane)  # [A], [A, 3]
        order = _arc_order(self._ds.directions.unit_vectors.shape[0])
        p_arc = resample(H, self._ds.directions, arc_uv, order)  # [F, A]
        return ang, _db(p_arc)

    def _replot(self) -> None:
        if self._ds is None:
            return
        drv_idx = self._drv_combo.currentIndex()
        if drv_idx < 0:
            return

        drv = self._ds.drivers[drv_idx]
        field_name = self._field_combo.currentText()
        H = _referenced_field(self._ds, drv, field_name, self._mode)  # [F, N]
        freqs = self._ds.frequencies  # [F]

        ang_h, db_h = self._sonogram(H, "H")
        ang_v, db_v = self._sonogram(H, "V")
        vmax = float(max(db_h.max(), db_v.max()))  # shared 0-dB reference across both planes

        fig = self._canvas.fig
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
        ax_h.set_title(
            f"Directivity sonograms — {drv.driver_id} [{field_name}] · {self._mode}", fontsize=9
        )
        if mesh is not None:
            fig.colorbar(mesh, ax=[ax_h, ax_v], label="normalised dB")
        self._canvas.redraw()


# ---------------------------------------------------------------------------
# CEA2034 / spinorama sub-tab
# ---------------------------------------------------------------------------

_CEA_LABELS = {
    "on_axis": "On-Axis",
    "listening_window": "Listening Window",
    "early_reflections": "Early Reflections",
    "sound_power": "Sound Power",
    "estimated_in_room": "Estimated In-Room",
    "sound_power_di": "Sound Power DI",
    "early_reflections_di": "Early Refl. DI",
}
_CEA_COLORS = {
    "on_axis": "k",
    "listening_window": "tab:green",
    "early_reflections": "tab:orange",
    "sound_power": "tab:blue",
    "estimated_in_room": "tab:red",
    "sound_power_di": "tab:purple",
    "early_reflections_di": "tab:brown",
}


class _Cea2034View(_ReferencedView):
    """CEA-2034-A spinorama: SPL curves (left axis) + DI curves (right axis), log frequency."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Driver:"))
        self._drv_combo = QComboBox()
        self._drv_combo.currentIndexChanged.connect(self._replot)
        ctrl.addWidget(self._drv_combo)
        ctrl.addWidget(QLabel("Field:"))
        self._field_combo = self._field_combo_widget()
        ctrl.addWidget(self._field_combo)
        ctrl.addStretch()
        layout.addLayout(ctrl)

        self._canvas = _MplCanvas()
        layout.addWidget(self._canvas)

    def load(self, ds: RadiationDataset) -> None:
        self._ds = ds
        self._populate_drivers(self._drv_combo, ds)
        self._replot()

    def _replot(self) -> None:
        if self._ds is None:
            return
        drv_idx = self._drv_combo.currentIndex()
        if drv_idx < 0:
            return

        drv = self._ds.drivers[drv_idx]
        field_name = self._field_combo.currentText()
        H = _referenced_field(self._ds, drv, field_name, self._mode)  # [F, N]
        curves = compute_cea2034(
            H, self._ds.frequencies, self._ds.directions, _reference_axis(self._ds)
        )
        self._last_curves = curves  # exposed for tests / debugging
        freqs = curves["frequencies"]

        fig = self._canvas.fig
        with _quiet_redraw_warnings():
            fig.clear()
        ax = fig.add_subplot(1, 1, 1)
        ax2 = ax.twinx()
        for key in SPL_CURVES:
            ax.semilogx(
                freqs, curves[key], color=_CEA_COLORS[key], linewidth=1.5, label=_CEA_LABELS[key]
            )
        for key in DI_CURVES:
            ax2.semilogx(
                freqs,
                curves[key],
                color=_CEA_COLORS[key],
                linewidth=1.2,
                linestyle="--",
                label=_CEA_LABELS[key],
            )
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("dB SPL (re 20 µPa)")
        ax2.set_ylabel("Directivity Index (dB)")
        ax.grid(True, which="both", alpha=0.3)
        ax.set_title(
            f"CEA-2034-A spinorama — {drv.driver_id} [{field_name}] · {self._mode}", fontsize=9
        )
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, fontsize=7, loc="lower center", ncol=4)
        self._canvas.redraw()


# ---------------------------------------------------------------------------
# Results tab (outer container with referencing control + export)
# ---------------------------------------------------------------------------


class ResultsTab(QWidget):
    """Tab 4 — Results.

    Six sub-tabs (on-axis, H/V polar, balloon, sonograms, CEA2034) + a dataset-wide
    field-referencing control + Export panel.  Call ``load(dataset)`` to populate all
    views from a ``RadiationDataset`` (a completed solve or File → Open dataset).
    """

    def __init__(self, state, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._state = state
        self._ds: Optional[RadiationDataset] = None

        layout = QVBoxLayout(self)

        # ── Dataset-wide field referencing (display-only) ─────────────────────
        ref_row = QHBoxLayout()
        ref_lbl = QLabel("Field referencing:")
        ref_lbl.setToolTip(_REF_TOOLTIP)
        ref_row.addWidget(ref_lbl)
        self._ref_combo = QComboBox()
        self._ref_combo.addItems(REFERENCING_MODES)
        self._ref_combo.setToolTip(_REF_TOOLTIP)
        self._ref_combo.currentTextChanged.connect(self._on_referencing_changed)
        ref_row.addWidget(self._ref_combo)
        ref_row.addStretch()
        layout.addLayout(ref_row)

        self._sub_tabs = QTabWidget()
        self._on_axis = _OnAxisView()
        self._h_polar = _PolarView("Horizontal")
        self._v_polar = _PolarView("Vertical")
        self._balloon = _BalloonView()
        self._di_map = _DirectivityMapView()
        self._cea = _Cea2034View()
        self._sub_tabs.addTab(self._on_axis, "On-axis")
        self._sub_tabs.addTab(self._h_polar, "H Polar")
        self._sub_tabs.addTab(self._v_polar, "V Polar")
        self._sub_tabs.addTab(self._balloon, "Balloon")
        self._sub_tabs.addTab(self._di_map, "Sonograms")
        self._sub_tabs.addTab(self._cea, "CEA2034")
        layout.addWidget(self._sub_tabs)

        self._views = [
            self._on_axis,
            self._h_polar,
            self._v_polar,
            self._balloon,
            self._di_map,
            self._cea,
        ]

        # ── Export controls ──────────────────────────────────────────────────
        exp_box = QGroupBox("Export")
        exp_row = QHBoxLayout(exp_box)
        self._exp_h5_btn = QPushButton("Save HDF5…")
        self._exp_frd_btn = QPushButton("Export .frd…")
        self._exp_sofa_btn = QPushButton("Export SOFA…")
        self._exp_clf_btn = QPushButton("Export CLF…")
        self._exp_clf_btn.setToolTip("CLF export requires SH resampling — deferred")
        self._exp_clf_btn.setEnabled(False)
        for btn in (self._exp_h5_btn, self._exp_frd_btn, self._exp_sofa_btn, self._exp_clf_btn):
            exp_row.addWidget(btn)
        exp_row.addStretch()
        layout.addWidget(exp_box)

        self._exp_h5_btn.clicked.connect(self._save_hdf5)
        self._exp_frd_btn.clicked.connect(self._export_frd)
        self._exp_sofa_btn.clicked.connect(self._export_sofa)

        self._set_export_enabled(False)

    def _on_referencing_changed(self, mode: str) -> None:
        for v in self._views:
            v.set_referencing(mode)

    def load(self, ds: RadiationDataset) -> None:
        """Populate all result views from ``ds``."""
        self._ds = ds
        mode = self._ref_combo.currentText()
        for v in self._views:
            v._mode = mode
            v.load(ds)
        self._set_export_enabled(True)

    def _set_export_enabled(self, ok: bool) -> None:
        for btn in (self._exp_h5_btn, self._exp_frd_btn, self._exp_sofa_btn):
            btn.setEnabled(ok)

    # ------------------------------------------------------------------
    # Export actions
    # ------------------------------------------------------------------

    def _save_hdf5(self) -> None:
        if self._ds is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save HDF5", "", "HDF5 (*.h5 *.bsim)")
        if not path:
            return
        from beamsim2.io.hdf5_store import write_dataset

        try:
            write_dataset(path, self._ds)
        except Exception as exc:
            from PySide6.QtWidgets import QMessageBox

            QMessageBox.critical(self, "Save failed", str(exc))

    def _export_frd(self) -> None:
        if self._ds is None:
            return
        directory = QFileDialog.getExistingDirectory(self, "Export .frd to directory")
        if not directory:
            return
        from beamsim2.io.frd_export import write_frd

        try:
            write_frd(directory, self._ds)
        except Exception as exc:
            from PySide6.QtWidgets import QMessageBox

            QMessageBox.critical(self, "Export failed", str(exc))

    def _export_sofa(self) -> None:
        if self._ds is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export SOFA", "", "SOFA AES69 (*.sofa)")
        if not path:
            return
        from beamsim2.io.sofa_export import write_sofa

        try:
            write_sofa(path, self._ds)
        except Exception as exc:
            from PySide6.QtWidgets import QMessageBox

            QMessageBox.critical(self, "Export failed", str(exc))
