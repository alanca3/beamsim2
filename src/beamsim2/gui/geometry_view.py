"""Geometry tab: box-dimension form + Matplotlib 3-D mesh preview + health status.

3-D rendering uses ``mpl_toolkits.mplot3d`` via ``FigureCanvasQTAgg`` — no new
dependency; matplotlib is already mandated (DR-06 / pyproject.toml).  The
canvas is adequate for a coarse placement preview and a Lebedev-26 balloon.

Build-order item 10, Tab 1 (§6 Gameplan — geometry input view).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from beamsim2.geometry.assemble import DriverSpec, assemble_box_driver
from beamsim2.geometry.health import HealthReport

# Inline import keeps startup fast (Matplotlib is slow to import)
try:
    import matplotlib

    matplotlib.use("QtAgg")
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # type: ignore[import-untyped]

    _MPL_OK = True
except Exception:
    _MPL_OK = False


def _spin(lo: float, hi: float, value: float, step: float = 0.01) -> QDoubleSpinBox:
    """Helper: create a metres spin-box."""
    sb = QDoubleSpinBox()
    sb.setRange(lo, hi)
    sb.setValue(value)
    sb.setSingleStep(step)
    sb.setSuffix(" m")
    sb.setDecimals(4)
    return sb


class _MeshCanvas(QWidget):
    """Matplotlib 3-D mesh preview embedded in Qt.

    Shows the box wireframe with driver discs coloured by group tag.
    Software-rendered via mplot3d; fine for a coarse placement preview.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if not _MPL_OK:
            layout.addWidget(QLabel("Matplotlib not available — 3-D preview disabled."))
            self._ax = None
            self._canvas = None
            return

        fig = Figure(figsize=(5, 4), tight_layout=True)
        self._canvas = FigureCanvasQTAgg(fig)
        self._canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._ax = fig.add_subplot(111, projection="3d")
        layout.addWidget(self._canvas)

    def render(self, mesh, health: Optional[HealthReport] = None) -> None:
        """Render the mesh triangles coloured by group_tag."""
        if self._ax is None:
            return

        ax = self._ax
        ax.cla()

        verts = mesh.vertices  # [V, 3]
        tris = mesh.triangles  # [T, 3]
        tags = mesh.group_tags  # [T]

        unique_tags = np.unique(tags)
        # Map tags to colours: drivers = warm colours, shell = light grey
        cmap = ["#e05252", "#5294e0", "#52c052", "#c09652"]  # up to 4 drivers
        for t in unique_tags:
            mask = tags == t
            polys = verts[tris[mask]]  # [k, 3, 3]
            is_shell = t == tags.max()
            color = "#cccccc" if is_shell else cmap[(t - 1) % len(cmap)]
            col = Poly3DCollection(polys, alpha=0.3 if is_shell else 0.7)
            col.set_facecolor(color)
            col.set_edgecolor("#888888")
            col.set_linewidth(0.2)
            ax.add_collection3d(col)

        # Auto-scale axes
        xyz_min = verts.min(axis=0)
        xyz_max = verts.max(axis=0)
        center = (xyz_min + xyz_max) / 2
        span = (xyz_max - xyz_min).max() / 2 * 1.1
        ax.set_xlim(center[0] - span, center[0] + span)
        ax.set_ylim(center[1] - span, center[1] + span)
        ax.set_zlim(center[2] - span, center[2] + span)  # type: ignore[attr-defined]
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")  # type: ignore[attr-defined]
        ax.set_title("Geometry preview")

        self._canvas.draw()


class GeometryTab(QWidget):
    """Tab 1 — Geometry.

    Controls
    --------
    - Width / Height / Depth / Fillet spin-boxes (metres)
    - [Preview] button: builds mesh and renders in the 3-D canvas
    - Health label: ✔ watertight / ✗ + error text
    - [Add driver] button: delegates to the Drivers tab by signalling

    Signals
    -------
    geometryChanged
        Emitted when the user edits dimensions or accepts a valid mesh.
    """

    geometryChanged = Signal()

    def __init__(self, state, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._state = state
        self._mesh = None
        self._bc = None

        main_layout = QHBoxLayout(self)

        # ── Left panel: form + health ────────────────────────────────────────
        left = QVBoxLayout()

        dims_box = QGroupBox("Box enclosure")
        form = QFormLayout(dims_box)
        self._w = _spin(0.05, 2.0, 0.12)
        self._h = _spin(0.05, 2.0, 0.10)
        self._d = _spin(0.05, 2.0, 0.08)
        self._fi = _spin(0.0, 0.05, 0.0, step=0.001)
        form.addRow("Width (x):", self._w)
        form.addRow("Height (y):", self._h)
        form.addRow("Depth (z):", self._d)
        form.addRow("Fillet radius:", self._fi)
        left.addWidget(dims_box)

        self._preview_btn = QPushButton("Preview mesh")
        self._preview_btn.clicked.connect(self._on_preview)
        left.addWidget(self._preview_btn)

        self._health_label = QLabel("Click 'Preview mesh' to validate geometry.")
        self._health_label.setWordWrap(True)
        left.addWidget(self._health_label)

        left.addStretch()
        main_layout.addLayout(left, stretch=1)

        # ── Right panel: 3-D canvas ──────────────────────────────────────────
        self._canvas = _MeshCanvas()
        main_layout.addWidget(self._canvas, stretch=3)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_preview(self) -> None:
        from beamsim2.pipeline.run import BoxGeometry
        from beamsim2.geometry.mesh import mesh_geometry

        # Use drivers from state (may be empty → just box geometry)
        drivers = [dp.spec for dp in self._state.drivers] or [
            # Placeholder: flush disk at box centre of +z face
            DriverSpec(
                center=(self._w.value() / 2, self._h.value() / 2, self._d.value()),
                normal=(0.0, 0.0, 1.0),
                radius=0.020,
            )
        ]

        from beamsim2.core.types import SolverConfig

        config = self._state.config or SolverConfig()

        try:
            mesh, bc, health = mesh_geometry(
                width=self._w.value(),
                height=self._h.value(),
                depth=self._d.value(),
                drivers=drivers,
                config=config,
                f_max=1000.0,  # 1 kHz target for preview mesh
                fillet_radius=self._fi.value(),
            )
        except ValueError as exc:
            self._health_label.setText(f"✗ {exc}")
            self._health_label.setStyleSheet("color: red;")
            return

        self._mesh = mesh
        self._bc = bc

        # Update app state
        self._state.geometry = BoxGeometry(
            self._w.value(), self._h.value(), self._d.value(), self._fi.value()
        )

        if health.is_watertight and not health.problems:
            self._health_label.setText("✔ Watertight · normals OK")
            self._health_label.setStyleSheet("color: green;")
        else:
            summary = "; ".join(health.problems) or "warnings present"
            self._health_label.setText(f"⚠ {summary}")
            self._health_label.setStyleSheet("color: darkorange;")

        self._canvas.render(mesh, health)
        self.geometryChanged.emit()
