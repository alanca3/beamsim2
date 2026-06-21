"""Results tab: on-axis, polar, 3-D balloon, directivity map, and export controls.

All plots use ``matplotlib`` via ``FigureCanvasQTAgg`` — no new dependency.
mplot3d is used for the 3-D balloon; adequate for Lebedev-26 (26 points).

The tab can be used as a standalone HDF5 viewer (File → Open dataset) without
running a new solve — this is also the GUI smoke-test entry point.

Build-order item 10, Tab 4 (§6 Gameplan — results display and export).
"""

from __future__ import annotations

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
from beamsim2.core.sphere import DEFAULT_REFERENCE_AXIS, nearest_direction_index

_P_REF = 20e-6  # Pa — reference SPL


def _db(pressure: np.ndarray) -> np.ndarray:
    """Magnitude in dB SPL re 20 µPa."""
    return 20.0 * np.log10(np.abs(pressure) / _P_REF + 1e-300)


def _reference_axis(ds: RadiationDataset) -> np.ndarray:
    """Return the dataset's measurement/reference axis (default +z).

    Reads the ``reference_axis`` root attr written by the pipeline; falls back to
    +z for legacy files that predate it.  Used so the on-axis/balloon views pick
    the loudspeaker front instead of hardcoding +z.
    """
    axis = ds.attrs.get("reference_axis", DEFAULT_REFERENCE_AXIS)
    return np.asarray(axis, dtype=np.float64).reshape(3)


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
        self._fig.tight_layout()
        self._canvas.draw()


# ---------------------------------------------------------------------------
# On-axis sub-tab
# ---------------------------------------------------------------------------


class _OnAxisView(QWidget):
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
        self._field_combo = QComboBox()
        self._field_combo.addItems(["H_full", "H_bem"])
        self._field_combo.currentIndexChanged.connect(self._replot)
        ctrl.addWidget(self._field_combo)
        ctrl.addStretch()
        layout.addLayout(ctrl)

        self._canvas = _MplCanvas(nrows=2)
        self._ax_mag = self._canvas.fig.add_subplot(2, 1, 1)
        self._ax_ph = self._canvas.fig.add_subplot(2, 1, 2, sharex=self._ax_mag)
        self._canvas.fig.delaxes(self._canvas.ax)
        layout.addWidget(self._canvas)

        self._ds: Optional[RadiationDataset] = None

    def load(self, ds: RadiationDataset) -> None:
        self._ds = ds
        self._drv_combo.blockSignals(True)
        self._drv_combo.clear()
        for d in ds.drivers:
            self._drv_combo.addItem(d.driver_id)
        self._drv_combo.blockSignals(False)
        self._replot()

    def _replot(self) -> None:
        if self._ds is None:
            return
        idx = self._drv_combo.currentIndex()
        if idx < 0 or idx >= len(self._ds.drivers):
            return

        drv = self._ds.drivers[idx]
        field_name = self._field_combo.currentText()
        H = drv.H_full if field_name == "H_full" else drv.H_bem  # [F, N]
        freqs = self._ds.frequencies  # [F]
        conv = drv.convergence_flags  # [F] bool

        # On-axis: pick the direction nearest the dataset's reference axis
        # (default +z, but settable to the loudspeaker front) — never hardcode +z.
        uvecs = self._ds.directions.unit_vectors  # [N, 3]
        on_ax_idx = nearest_direction_index(uvecs, _reference_axis(self._ds))
        self._last_on_axis_idx = on_ax_idx  # exposed for tests / debugging
        h_onaxis = H[:, on_ax_idx]  # [F] complex128

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
        self._ax_mag.set_title(f"On-axis — {drv.driver_id} [{field_name}]")
        self._ax_mag.grid(True, which="both", alpha=0.4)

        self._ax_ph.set_xlabel("Frequency (Hz)")
        self._ax_ph.set_ylabel("Phase (°)")
        self._ax_ph.grid(True, which="both", alpha=0.4)

        self._canvas.fig.tight_layout()
        self._canvas.redraw()


# ---------------------------------------------------------------------------
# Polar sub-tab
# ---------------------------------------------------------------------------


class _PolarView(QWidget):
    """Horizontal or vertical polar plot at a selectable frequency."""

    def __init__(self, plane: str = "Horizontal", parent=None):
        super().__init__(parent)
        self._plane = plane  # "Horizontal" or "Vertical"
        layout = QVBoxLayout(self)

        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Driver:"))
        self._drv_combo = QComboBox()
        self._drv_combo.currentIndexChanged.connect(self._replot)
        ctrl.addWidget(self._drv_combo)
        ctrl.addWidget(QLabel("Frequency:"))
        self._freq_combo = QComboBox()
        self._freq_combo.currentIndexChanged.connect(self._replot)
        ctrl.addWidget(self._freq_combo)
        ctrl.addStretch()
        layout.addLayout(ctrl)

        self._canvas = _MplCanvas(projection="polar")
        layout.addWidget(self._canvas)

        self._ds: Optional[RadiationDataset] = None

    def load(self, ds: RadiationDataset) -> None:
        self._ds = ds
        self._drv_combo.blockSignals(True)
        self._drv_combo.clear()
        for d in ds.drivers:
            self._drv_combo.addItem(d.driver_id)
        self._drv_combo.blockSignals(False)

        self._freq_combo.blockSignals(True)
        self._freq_combo.clear()
        for f in ds.frequencies:
            self._freq_combo.addItem(f"{f:.0f} Hz")
        self._freq_combo.setCurrentIndex(len(ds.frequencies) // 2)
        self._freq_combo.blockSignals(False)
        self._replot()

    def _replot(self) -> None:
        if self._ds is None:
            return
        drv_idx = self._drv_combo.currentIndex()
        freq_idx = self._freq_combo.currentIndex()
        if drv_idx < 0 or freq_idx < 0:
            return

        drv = self._ds.drivers[drv_idx]
        H_f = drv.H_full[freq_idx, :]  # [N] complex128
        theta_phi = self._ds.directions.theta_phi  # [N, 2] — theta, phi in rad

        # Plane selection: horizontal = equatorial (theta near π/2),
        # vertical = xz-plane (phi near 0 or π)
        if self._plane == "Horizontal":
            equator = np.abs(np.cos(theta_phi[:, 0])) < 0.25  # near sin(theta)≈1
            mask = equator if np.sum(equator) >= 3 else np.ones(len(H_f), bool)
        else:
            near_xz = (np.abs(np.sin(theta_phi[:, 1])) < 0.25) | (
                np.abs(np.sin(theta_phi[:, 1]) + 1) < 0.25
            )
            mask = near_xz if np.sum(near_xz) >= 3 else np.ones(len(H_f), bool)

        angles = theta_phi[mask, 1] if self._plane == "Horizontal" else theta_phi[mask, 0]
        mag_db = _db(H_f[mask])
        mag_db_norm = mag_db - mag_db.max()

        ax = self._canvas.ax
        ax.cla()
        # Sort by angle for a clean outline
        order = np.argsort(angles)
        ax.plot(
            np.append(angles[order], angles[order[0]]),
            np.append(mag_db_norm[order], mag_db_norm[order[0]]),
            "b-",
            linewidth=1.5,
        )
        ax.set_title(
            f"{self._plane} polar — {drv.driver_id} @ {self._ds.frequencies[freq_idx]:.0f} Hz"
        )
        self._canvas.redraw()


# ---------------------------------------------------------------------------
# 3-D balloon sub-tab
# ---------------------------------------------------------------------------


class _BalloonView(QWidget):
    """3-D balloon: magnitude as radial distance on the sphere directions."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Driver:"))
        self._drv_combo = QComboBox()
        self._drv_combo.currentIndexChanged.connect(self._replot)
        ctrl.addWidget(self._drv_combo)
        ctrl.addWidget(QLabel("Frequency:"))
        self._freq_combo = QComboBox()
        self._freq_combo.currentIndexChanged.connect(self._replot)
        ctrl.addWidget(self._freq_combo)
        ctrl.addStretch()
        layout.addLayout(ctrl)

        self._canvas = _MplCanvas(projection="3d")
        layout.addWidget(self._canvas)

        self._ds: Optional[RadiationDataset] = None

    def load(self, ds: RadiationDataset) -> None:
        self._ds = ds
        self._drv_combo.blockSignals(True)
        self._drv_combo.clear()
        for d in ds.drivers:
            self._drv_combo.addItem(d.driver_id)
        self._drv_combo.blockSignals(False)

        self._freq_combo.blockSignals(True)
        self._freq_combo.clear()
        for f in ds.frequencies:
            self._freq_combo.addItem(f"{f:.0f} Hz")
        self._freq_combo.setCurrentIndex(len(ds.frequencies) // 2)
        self._freq_combo.blockSignals(False)
        self._replot()

    def _replot(self) -> None:
        if self._ds is None:
            return
        drv_idx = self._drv_combo.currentIndex()
        freq_idx = self._freq_combo.currentIndex()
        if drv_idx < 0 or freq_idx < 0:
            return

        drv = self._ds.drivers[drv_idx]
        H_f = drv.H_full[freq_idx, :]  # [N]
        uvecs = self._ds.directions.unit_vectors  # [N, 3]

        # Scale unit vectors by normalised magnitude for the balloon
        mag = np.abs(H_f)
        mag_norm = mag / (mag.max() + 1e-300)
        xyz = uvecs * mag_norm[:, None]  # [N, 3]

        ax = self._canvas.ax
        ax.cla()
        sc = ax.scatter(
            xyz[:, 0], xyz[:, 1], xyz[:, 2], c=_db(H_f), cmap="jet", s=60, depthshade=True
        )
        if not hasattr(self, "_cbar") or self._cbar is None:
            self._cbar = self._canvas.fig.colorbar(sc, ax=ax, shrink=0.6, label="dB SPL")
        else:
            self._cbar.update_normal(sc)

        # Reference-axis (0°/on-axis) indicator: a dashed line from the origin
        # along the dataset's measurement axis, so "on-axis" is unambiguous.
        axis = _reference_axis(self._ds)
        norm = float(np.linalg.norm(axis))
        if norm > 0:
            a = axis / norm * 1.15
            ax.plot([0, a[0]], [0, a[1]], [0, a[2]], "k--", linewidth=1.5)
            ax.text(a[0], a[1], a[2], "  0° axis", fontsize=8)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")  # type: ignore[attr-defined]
        ax.set_title(f"Balloon — {drv.driver_id} @ {self._ds.frequencies[freq_idx]:.0f} Hz")
        self._canvas.redraw()


# ---------------------------------------------------------------------------
# Directivity map sub-tab
# ---------------------------------------------------------------------------


class _DirectivityMapView(QWidget):
    """Directivity map: frequency × angle, level as colour (imshow).

    Angle axis = elevation angle θ (co-latitude from +z axis).
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Driver:"))
        self._drv_combo = QComboBox()
        self._drv_combo.currentIndexChanged.connect(self._replot)
        ctrl.addWidget(self._drv_combo)
        ctrl.addStretch()
        layout.addLayout(ctrl)

        self._canvas = _MplCanvas()
        layout.addWidget(self._canvas)
        self._im = None

        self._ds: Optional[RadiationDataset] = None

    def load(self, ds: RadiationDataset) -> None:
        self._ds = ds
        self._drv_combo.blockSignals(True)
        self._drv_combo.clear()
        for d in ds.drivers:
            self._drv_combo.addItem(d.driver_id)
        self._drv_combo.blockSignals(False)
        self._replot()

    def _replot(self) -> None:
        if self._ds is None:
            return
        drv_idx = self._drv_combo.currentIndex()
        if drv_idx < 0:
            return

        drv = self._ds.drivers[drv_idx]
        H = drv.H_full  # [F, N]
        freqs = self._ds.frequencies  # [F]
        theta_phi = self._ds.directions.theta_phi  # [N, 2]

        # Sort directions by elevation θ for a smooth map
        theta = theta_phi[:, 0]  # [N]
        order = np.argsort(theta)
        theta_sorted = theta[order]
        mag_db = _db(H[:, order])  # [F, N_sorted]

        ax = self._canvas.ax
        ax.cla()
        # imshow: y = frequency index (top=low), x = angle
        ext = [np.degrees(theta_sorted[0]), np.degrees(theta_sorted[-1]), freqs[-1], freqs[0]]
        im = ax.imshow(mag_db, aspect="auto", origin="upper", extent=ext, cmap="jet")
        if self._im is None:
            self._canvas.fig.colorbar(im, ax=ax, label="dB SPL")
        self._im = im
        ax.set_xlabel("Elevation θ (°)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_title(f"Directivity map — {drv.driver_id}")
        self._canvas.redraw()


# ---------------------------------------------------------------------------
# Results tab (outer container with export controls)
# ---------------------------------------------------------------------------


class ResultsTab(QWidget):
    """Tab 4 — Results.

    Four sub-tabs (on-axis, polar, balloon, directivity map) + Export panel.
    Call ``load(dataset)`` to populate all views from a ``RadiationDataset``
    (from a completed solve or a File → Open dataset load).
    """

    def __init__(self, state, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._state = state
        self._ds: Optional[RadiationDataset] = None

        layout = QVBoxLayout(self)

        self._sub_tabs = QTabWidget()
        self._on_axis = _OnAxisView()
        self._h_polar = _PolarView("Horizontal")
        self._v_polar = _PolarView("Vertical")
        self._balloon = _BalloonView()
        self._di_map = _DirectivityMapView()
        self._sub_tabs.addTab(self._on_axis, "On-axis")
        self._sub_tabs.addTab(self._h_polar, "H Polar")
        self._sub_tabs.addTab(self._v_polar, "V Polar")
        self._sub_tabs.addTab(self._balloon, "Balloon")
        self._sub_tabs.addTab(self._di_map, "Dir. Map")
        layout.addWidget(self._sub_tabs)

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

    def load(self, ds: RadiationDataset) -> None:
        """Populate all result views from ``ds``."""
        self._ds = ds
        self._on_axis.load(ds)
        self._h_polar.load(ds)
        self._v_polar.load(ds)
        self._balloon.load(ds)
        self._di_map.load(ds)
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
