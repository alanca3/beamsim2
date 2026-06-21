"""Geometry tab: box-dimension form + interactive 3-D driver placement editor.

3-D editor uses PyVista / VTK embedded via ``pyvistaqt.QtInteractor`` (flagged
departure from DR-06 matplotlib-only mandate; noted in CHANGELOG).  Provides
LEAP-style interactive driver placement: click a face to place a driver, drag it
(movement locked to the face plane), right-click for Delete / Edit T/S.

If PyVista / VTK is not installed, falls back to the original Matplotlib
``_MeshCanvas`` static preview (``_PV_OK = False``).  The app launches either way.

Coordinate convention
---------------------
The parametric box is built at the origin corner: ``[0,w] × [0,h] × [0,d]``.
Driver placement is stored face-local (``FacePlacement``) and only ever converted
to world coordinates via ``face_local_to_spec`` — so drivers are always on-plane.

Build-order item 10 follow-up (interactive driver placement, GUI usability).
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from beamsim2.geometry.assemble import DriverSpec
from beamsim2.geometry.faces import (
    FacePlacement,
    clamp_uv_to_face,
    classify_face,
    face_basis,
    face_local_to_spec,
    fits_on_face,
    world_to_face_uv,
)
from beamsim2.geometry.health import HealthReport

if TYPE_CHECKING:
    from beamsim2.pipeline.run import DriverPlacement

# Driver cap colour palette (matches the existing mesh preview palette)
_DRIVER_COLORS = ["#e05252", "#5294e0", "#52c052", "#c09652"]
_SELECT_COLORS = ["#ff8080", "#80b8ff", "#80ff80", "#ffcc80"]
_FIT_FAIL_COLOR = "#ff4400"

# ── Lazy PyVista import ───────────────────────────────────────────────────────
# PyVista requires a real OpenGL context; it cannot work under QT_QPA_PLATFORM=offscreen
# (used by CI / test_gui_smoke.py). Treat headless Qt as "no VTK" so the app
# degrades gracefully without a crash.
_HEADLESS_QT = os.environ.get("QT_QPA_PLATFORM") == "offscreen"

try:
    if not _HEADLESS_QT:
        import pyvista as pv
        from pyvistaqt import QtInteractor

        _PV_OK = True
    else:
        _PV_OK = False
except Exception:
    _PV_OK = False

# ── Lazy Matplotlib import (fallback) ────────────────────────────────────────
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
    """Helper: metres spin-box."""
    sb = QDoubleSpinBox()
    sb.setRange(lo, hi)
    sb.setValue(value)
    sb.setSingleStep(step)
    sb.setSuffix(" m")
    sb.setDecimals(4)
    return sb


# ---------------------------------------------------------------------------
# Matplotlib fallback canvas (used when _PV_OK = False)
# ---------------------------------------------------------------------------


class _MeshCanvas(QWidget):
    """Matplotlib 3-D mesh preview embedded in Qt.

    Shows the meshed box triangles coloured by group tag.  Drivers appear as
    distinct warm-colour patches.  Software-rendered mplot3d; adequate for a
    coarse placement preview when PyVista / VTK is not available.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        if _MPL_OK:
            self._fig = Figure(figsize=(4, 4))
            self._canvas = FigureCanvasQTAgg(self._fig)
            layout.addWidget(self._canvas)
        else:
            layout.addWidget(QLabel("Matplotlib 3-D preview not available."))

    def render(self, mesh, health: HealthReport) -> None:  # type: ignore[type-arg]
        """Draw the mesh in the 3-D axes, coloured by group tag."""
        if not _MPL_OK:
            return
        self._fig.clear()
        ax = self._fig.add_subplot(111, projection="3d")

        verts = mesh.vertices  # [V, 3]
        tris = mesh.triangles  # [T, 3]
        tags = mesh.group_tags  # [T]

        unique_tags = np.unique(tags)
        cmap = ["#e05252", "#5294e0", "#52c052", "#c09652"]
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


# ---------------------------------------------------------------------------
# PyVista interactive driver-placement editor
# ---------------------------------------------------------------------------


class _DriverEditorCanvas(QWidget):
    """Interactive 3-D enclosure and driver placement editor using PyVista / VTK.

    Interaction model (LEAP EnclosureShop reference, manual pp. 330–352):

    * Left-click on an empty box face → place a new driver at the face centroid
      (in "add" mode) or deselect.
    * Left-click on an existing driver → select it (highlighted).
    * Left-drag on a selected driver → slide it along the face plane (movement
      locked to the plane, clamped to the face boundary).
    * Right-click on a driver → context menu: Delete / Edit T/S.

    Signals
    -------
    driverAdded(face_id, radius)
        Emitted when the user clicks an empty face in add mode.
        The caller should open TSDialog and then call ``commit_driver``.
    driverDeleted(index)
        Emitted when the user deletes a driver via context menu.
    driverEdited(index)
        Emitted when the user selects "Edit T/S" from context menu.
    driverMoved(index, face_placement)
        Emitted when a drag is released, carrying the new FacePlacement.
    """

    driverAdded = Signal(int, float)  # face_id, default_radius
    driverDeleted = Signal(int)  # driver index
    driverEdited = Signal(int)  # driver index
    driverMoved = Signal(int, object)  # index, FacePlacement

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if not _PV_OK:
            layout.addWidget(
                QLabel(
                    "PyVista / VTK not installed — interactive 3-D editor disabled.\n"
                    "Run: uv add pyvista pyvistaqt"
                )
            )
            return

        # Embed the QtInteractor
        self._plotter = QtInteractor(parent=self)
        self._plotter.set_background("#1e1e2e")
        layout.addWidget(self._plotter)

        # State
        self._w: float = 0.12
        self._h: float = 0.10
        self._d: float = 0.08
        self._mode: str = "select"  # "select" | "add"
        self._selected_idx: Optional[int] = None
        self._driver_actors: dict[int, object] = {}  # idx → vtkActor
        self._placements: list[DriverPlacement] = []  # current driver list copy

        # Drag state
        self._dragging: bool = False
        self._drag_idx: Optional[int] = None
        self._drag_face_id: Optional[int] = None
        self._saved_style: Optional[object] = None  # interactor style saved before drag

        # Camera: only reset_camera on first render or after box dims change
        self._camera_initialized: bool = False

        # Wire VTK observers
        iren = self._plotter.iren.interactor
        iren.AddObserver("LeftButtonPressEvent", self._on_left_press)
        iren.AddObserver("MouseMoveEvent", self._on_mouse_move)
        iren.AddObserver("LeftButtonReleaseEvent", self._on_left_release)
        iren.AddObserver("RightButtonPressEvent", self._on_right_press)

        # Show an empty box placeholder
        self._render_empty()

    # ── Public API ────────────────────────────────────────────────────────────

    def set_box_dims(self, w: float, h: float, d: float) -> None:
        """Update the box dimensions and re-render.  Called by the dim spinboxes."""
        self._w = w
        self._h = h
        self._d = d

    def render_scene(
        self,
        placements: list[DriverPlacement],
        w: float,
        h: float,
        d: float,
        selected_idx: Optional[int] = None,
    ) -> None:
        """Rebuild all actors: box shell + one disc per driver.

        Parameters
        ----------
        placements : list[DriverPlacement]
        w, h, d : float
        selected_idx : int or None
            Index of the currently selected driver (highlighted).
        """
        if not _PV_OK:
            return

        # Reset camera only on first render or when box dimensions change
        dims_changed = (w, h, d) != (self._w, self._h, self._d)

        self._w, self._h, self._d = w, h, d
        self._placements = list(placements)
        self._selected_idx = selected_idx
        self._driver_actors.clear()
        self._plotter.clear_actors()

        # ── box ──────────────────────────────────────────────────────────────
        box_mesh = pv.Box(bounds=(0, w, 0, h, 0, d))
        self._plotter.add_mesh(
            box_mesh,
            color="#9999bb",
            opacity=0.18,
            show_edges=True,
            edge_color="#aaaacc",
            line_width=1.0,
            name="box_shell",
        )

        # ── drivers ──────────────────────────────────────────────────────────
        for i, dp in enumerate(placements):
            self._add_driver_actor(i, dp, selected=(i == selected_idx))

        # ── axis labels ──────────────────────────────────────────────────────
        self._plotter.add_axes(xlabel="x", ylabel="y", zlabel="z", interactive=False)
        if not self._camera_initialized or dims_changed:
            self._plotter.reset_camera()
            self._camera_initialized = True
        self._plotter.render()

    def set_mode(self, mode: str) -> None:
        """Set interaction mode: 'select' or 'add'."""
        self._mode = mode

    # ── Internal rendering ────────────────────────────────────────────────────

    def _render_empty(self) -> None:
        """Show an empty unit box as a placeholder."""
        if not _PV_OK:
            return
        self._plotter.clear_actors()
        box_mesh = pv.Box(bounds=(0, self._w, 0, self._h, 0, self._d))
        self._plotter.add_mesh(
            box_mesh,
            color="#9999bb",
            opacity=0.18,
            show_edges=True,
            edge_color="#aaaacc",
            line_width=1.0,
            name="box_shell",
        )
        self._plotter.add_axes(xlabel="x", ylabel="y", zlabel="z", interactive=False)
        self._plotter.reset_camera()
        self._plotter.render()

    def _add_driver_actor(
        self,
        idx: int,
        dp: DriverPlacement,
        selected: bool = False,
        fit_fail: bool = False,
    ) -> None:
        """Add (or replace) the actor for driver idx."""
        if not _PV_OK:
            return

        center = dp.spec.center  # (x, y, z) world
        normal = dp.spec.normal  # outward face normal
        r = dp.spec.radius

        # Build a thin disc on the face
        disc = pv.Disc(
            center=center,
            normal=normal,
            inner=0.0,
            outer=r,
            r_res=1,
            c_res=40,
        )

        if fit_fail:
            color = _FIT_FAIL_COLOR
        elif selected:
            color = _SELECT_COLORS[idx % len(_SELECT_COLORS)]
        else:
            color = _DRIVER_COLORS[idx % len(_DRIVER_COLORS)]

        actor = self._plotter.add_mesh(
            disc,
            color=color,
            opacity=0.85,
            show_edges=True,
            edge_color="#222222",
            line_width=1.0,
            name=f"driver_{idx}",
        )
        self._driver_actors[idx] = actor

    def _pick_world_point(
        self, x: int, y: int
    ) -> tuple[Optional[tuple], Optional[tuple], Optional[int]]:
        """Pick a world-space point and normal at display coords (x, y).

        Returns
        -------
        (point, normal, driver_idx)
            point is (px, py, pz) or None if missed.
            normal is the picked cell normal or None.
            driver_idx is the matched driver index or None if the box was hit.
        """
        import vtk

        picker = vtk.vtkCellPicker()
        picker.SetTolerance(0.001)
        renderer = self._plotter.renderer
        picker.Pick(x, y, 0, renderer)
        pos = picker.GetPickPosition()
        if pos == (0.0, 0.0, 0.0) and picker.GetCellId() == -1:
            return None, None, None

        point = tuple(pos)

        # Determine picked normal
        normal = None
        actor = picker.GetActor()
        if actor is not None:
            pd = picker.GetDataSet()
            if pd is not None:
                cell_id = picker.GetCellId()
                if cell_id >= 0 and pd.GetCellData().GetNormals() is not None:
                    n = pd.GetCellData().GetNormals().GetTuple(cell_id)
                    normal = tuple(n)

        # Check if a driver actor was hit
        driver_idx: Optional[int] = None
        for idx, drv_actor in self._driver_actors.items():
            if actor is drv_actor:
                driver_idx = idx
                break

        return point, normal, driver_idx

    def _ray_plane_hit(self, x: int, y: int, face_id: int) -> Optional[tuple[float, float, float]]:
        """Ray–plane intersection for constrained drag.

        Builds a world-space ray from camera through display point (x, y) and
        intersects it with the fixed face plane.  Returns None if the ray is
        parallel to the plane (shouldn't happen for sensible views).
        """
        renderer = self._plotter.renderer

        # Two display depths → two world points → ray direction
        p0 = [0.0, 0.0, 0.0]
        p1 = [0.0, 0.0, 0.0]
        renderer.SetDisplayPoint(x, y, 0.0)
        renderer.DisplayToWorld()
        renderer.GetWorldPoint(p0)

        renderer.SetDisplayPoint(x, y, 1.0)
        renderer.DisplayToWorld()
        renderer.GetWorldPoint(p1)

        # Handle homogeneous coordinate (w component)
        if abs(p0[3]) > 1e-12:
            p0 = [p0[0] / p0[3], p0[1] / p0[3], p0[2] / p0[3]]
        if abs(p1[3]) > 1e-12:
            p1 = [p1[0] / p1[3], p1[1] / p1[3], p1[2] / p1[3]]

        ray_origin = np.array(p0[:3], dtype=float)
        ray_dir = np.array(p1[:3], dtype=float) - ray_origin
        dir_len = np.linalg.norm(ray_dir)
        if dir_len < 1e-12:
            return None
        ray_dir /= dir_len

        b = face_basis(face_id, self._w, self._h, self._d)
        centroid = np.array(b.centroid, dtype=float)
        normal = np.array(b.normal, dtype=float)

        denom = float(np.dot(ray_dir, normal))
        if abs(denom) < 1e-10:
            return None  # ray parallel to plane
        t = float(np.dot(centroid - ray_origin, normal)) / denom
        hit = ray_origin + t * ray_dir
        return (float(hit[0]), float(hit[1]), float(hit[2]))

    # ── VTK event observers ───────────────────────────────────────────────────

    def _on_left_press(self, obj, event) -> None:
        """Handle left mouse button press: select driver or start place/drag."""
        if not _PV_OK:
            return

        iren = self._plotter.iren.interactor
        x, y = iren.GetEventPosition()

        point, normal, driver_idx = self._pick_world_point(x, y)
        if point is None:
            # Missed everything → deselect
            self._selected_idx = None
            self._redraw_driver_colors()
            return

        if driver_idx is not None:
            # Hit a driver actor → select it and start drag
            self._selected_idx = driver_idx
            self._redraw_driver_colors()
            self._dragging = True
            self._drag_idx = driver_idx
            self._drag_face_id = self._placements[driver_idx].spec.normal
            # Resolve face_id from spec normal
            dp = self._placements[driver_idx]
            if dp.face_placement is not None:
                self._drag_face_id = dp.face_placement.face_id
            else:
                self._drag_face_id = classify_face(
                    dp.spec.center, dp.spec.normal, self._w, self._h, self._d
                )
            # Swap to a no-op interactor style so camera doesn't rotate during drag;
            # restored in _on_left_release.
            import vtk  # noqa: PLC0415  (local import — VTK only available when _PV_OK)

            self._saved_style = iren.GetInteractorStyle()
            iren.SetInteractorStyle(vtk.vtkInteractorStyleUser())
        else:
            # Hit the box face
            face_id = classify_face(point, normal, self._w, self._h, self._d)
            if self._mode == "add":
                self.driverAdded.emit(face_id, 0.040)  # default r=40mm
            else:
                # Deselect
                self._selected_idx = None
                self._redraw_driver_colors()

    def _on_mouse_move(self, obj, event) -> None:
        """During drag: reproject hit to face plane and slide the driver."""
        if not _PV_OK or not self._dragging or self._drag_idx is None:
            return
        if self._drag_face_id is None:
            return

        iren = self._plotter.iren.interactor
        x, y = iren.GetEventPosition()

        hit = self._ray_plane_hit(x, y, self._drag_face_id)
        if hit is None:
            return

        u, v = world_to_face_uv(self._drag_face_id, hit, self._w, self._h, self._d)
        idx = self._drag_idx
        dp = self._placements[idx]
        r = dp.spec.radius
        u, v = clamp_uv_to_face(self._drag_face_id, u, v, r, self._w, self._h, self._d)

        # Update the FacePlacement in our local copy
        if dp.face_placement is not None:
            dp.face_placement.u = u
            dp.face_placement.v = v
        else:
            # Create a face_placement on the fly so drag works
            dp.face_placement = FacePlacement(face_id=self._drag_face_id, u=u, v=v, radius=r)

        # Re-derive spec from face_placement
        new_spec = face_local_to_spec(dp.face_placement, self._w, self._h, self._d)
        dp.spec = new_spec  # type: ignore[misc]

        # Redraw just the driver disc (re-add replaces the named actor)
        fp_check = dp.face_placement
        fail = not fits_on_face(fp_check, self._w, self._h, self._d) if fp_check else False
        self._add_driver_actor(idx, dp, selected=True, fit_fail=fail)
        self._plotter.render()

    def _on_left_release(self, obj, event) -> None:
        """End drag: restore camera interactor style and commit the updated placement."""
        if not _PV_OK or not self._dragging:
            return
        self._dragging = False
        idx = self._drag_idx
        self._drag_idx = None
        self._drag_face_id = None

        # Restore normal camera interaction (was swapped to no-op during drag)
        if self._saved_style is not None:
            iren = self._plotter.iren.interactor
            iren.SetInteractorStyle(self._saved_style)
            self._saved_style = None

        if idx is not None and idx < len(self._placements):
            dp = self._placements[idx]
            if dp.face_placement is not None:
                self.driverMoved.emit(idx, dp.face_placement)

    def _on_right_press(self, obj, event) -> None:
        """Right-click: show context menu if a driver is under the cursor."""
        if not _PV_OK:
            return
        iren = self._plotter.iren.interactor
        x, y = iren.GetEventPosition()

        _, _, driver_idx = self._pick_world_point(x, y)
        if driver_idx is None:
            return

        # Build Qt context menu
        from PySide6.QtGui import QCursor

        menu = QMenu(self)
        edit_act = menu.addAction("Edit T/S…")
        del_act = menu.addAction("Delete")
        chosen = menu.exec(QCursor.pos())
        if chosen is del_act:
            self.driverDeleted.emit(driver_idx)
        elif chosen is edit_act:
            self.driverEdited.emit(driver_idx)

    def _redraw_driver_colors(self) -> None:
        """Repaint all driver actors with correct selection highlight."""
        if not _PV_OK:
            return
        for i, dp in enumerate(self._placements):
            fp = dp.face_placement
            fail = not fits_on_face(fp, self._w, self._h, self._d) if fp else False
            self._add_driver_actor(i, dp, selected=(i == self._selected_idx), fit_fail=fail)
        self._plotter.render()

    def closeEvent(self, event) -> None:
        """Clean up the VTK interactor on widget close."""
        if _PV_OK and hasattr(self, "_plotter"):
            self._plotter.close()
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# Geometry tab
# ---------------------------------------------------------------------------


class GeometryTab(QWidget):
    """Tab 0 — Geometry.

    Controls
    --------
    - Width / Height / Depth / Fillet spin-boxes (metres).
    - [Add Driver] toggle: click a box face to place a driver.
    - [Preview mesh] button: builds the BEM mesh and runs health checks.
    - 3-D interactive canvas (PyVista if available, matplotlib fallback).

    Signals
    -------
    geometryChanged
        Emitted when the mesh is rebuilt and accepted by health checks.
    driversChanged
        Emitted when the driver list changes via the canvas (add/move/delete).
    """

    geometryChanged = Signal()
    driversChanged = Signal()

    def __init__(self, state, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._state = state
        main_layout = QHBoxLayout(self)

        # ── left panel: dimension controls ───────────────────────────────────
        left = QVBoxLayout()
        left.setSpacing(6)

        dims_box = QGroupBox("Enclosure dimensions")
        form = QFormLayout(dims_box)
        self._w = _spin(0.01, 2.0, 0.12, step=0.01)
        self._h = _spin(0.01, 2.0, 0.10, step=0.01)
        self._d = _spin(0.01, 2.0, 0.08, step=0.01)
        self._fi = _spin(0.0, 0.1, 0.0, step=0.002)
        form.addRow("Width (x):", self._w)
        form.addRow("Height (y):", self._h)
        form.addRow("Depth (z):", self._d)
        form.addRow("Fillet radius:", self._fi)
        left.addWidget(dims_box)

        # Dimension changes → update 3-D canvas live
        for sb in (self._w, self._h, self._d):
            sb.valueChanged.connect(self._on_dims_changed)

        # Add Driver toggle
        self._add_drv_btn = QPushButton("+ Add Driver")
        self._add_drv_btn.setCheckable(True)
        self._add_drv_btn.setToolTip("Click a box face in the 3-D view to place a driver there")
        self._add_drv_btn.toggled.connect(self._on_add_mode_toggled)
        left.addWidget(self._add_drv_btn)

        # Preview mesh button
        self._preview_btn = QPushButton("Preview mesh")
        self._preview_btn.clicked.connect(self._on_preview)
        left.addWidget(self._preview_btn)

        self._health_label = QLabel("Click 'Preview mesh' to validate geometry.")
        self._health_label.setWordWrap(True)
        left.addWidget(self._health_label)

        left.addStretch()
        main_layout.addLayout(left, stretch=1)

        # ── right panel: 3-D canvas ───────────────────────────────────────────
        if _PV_OK:
            self._editor: _DriverEditorCanvas | _MeshCanvas = _DriverEditorCanvas()
            # Wire canvas signals
            self._editor.driverAdded.connect(self._on_canvas_driver_added)
            self._editor.driverDeleted.connect(self._on_canvas_driver_deleted)
            self._editor.driverEdited.connect(self._on_canvas_driver_edited)
            self._editor.driverMoved.connect(self._on_canvas_driver_moved)
        else:
            self._editor = _MeshCanvas()

        self._editor.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        main_layout.addWidget(self._editor, stretch=3)

        # Initial render (empty box)
        self._refresh_canvas()

    # ── Public slots ──────────────────────────────────────────────────────────

    def refresh_canvas(self) -> None:
        """Called by other tabs when their driver edits change the list."""
        self._refresh_canvas()

    # ── Private helpers ───────────────────────────────────────────────────────

    def _refresh_canvas(self) -> None:
        """Rebuild the 3-D scene from current state."""
        if _PV_OK and isinstance(self._editor, _DriverEditorCanvas):
            self._editor.render_scene(
                placements=list(self._state.drivers),
                w=self._w.value(),
                h=self._h.value(),
                d=self._d.value(),
            )

    def _on_dims_changed(self) -> None:
        """Box dimensions changed: re-derive all face-local drivers and re-render."""
        w, h, d = self._w.value(), self._h.value(), self._d.value()
        for dp in self._state.drivers:
            if dp.face_placement is not None:
                new_spec = face_local_to_spec(dp.face_placement, w, h, d)
                dp.spec = new_spec  # type: ignore[misc]
        self._refresh_canvas()
        # Invalidate any existing mesh geometry (dims changed)
        self._state.geometry = None
        self._health_label.setText("Dimensions changed — click 'Preview mesh' to rebuild.")
        self._health_label.setStyleSheet("")
        self.geometryChanged.emit()

    def _on_add_mode_toggled(self, checked: bool) -> None:
        """Toggle 'add driver' mode on the canvas."""
        if _PV_OK and isinstance(self._editor, _DriverEditorCanvas):
            self._editor.set_mode("add" if checked else "select")
        if checked:
            self._add_drv_btn.setText("✕ Cancel (click face to place)")
        else:
            self._add_drv_btn.setText("+ Add Driver")

    def _on_canvas_driver_added(self, face_id: int, default_radius: float) -> None:
        """User clicked an empty face in add mode → place driver instantly with defaults.

        LEAP-style: the driver appears immediately at the face centre with a standard
        woofer T/S.  The user can right-click → Edit T/S to tune parameters afterward.
        No modal dialog on click — the previous stub-TSParams/TSDialog approach crashed
        because TSParams has no 'Le' field and required 'Sd' was missing.
        """
        from beamsim2.driver.terminal import default_terminal_model
        from beamsim2.pipeline.run import DriverPlacement

        # Deactivate add mode
        self._add_drv_btn.setChecked(False)

        # Face-local placement at the face centre (u=v=0)
        fp = FacePlacement(
            face_id=face_id,
            u=0.0,
            v=0.0,
            radius=default_radius,
        )
        w, h, d = self._w.value(), self._h.value(), self._d.value()
        new_spec = face_local_to_spec(fp, w, h, d)

        # Unique driver_id: lowest free "driver_N" — never reuses an id still in
        # use (a count-based scheme collides after a middle driver is deleted).
        from beamsim2.core.driver_ids import next_driver_id

        driver_id = next_driver_id(dp.driver_id for dp in self._state.drivers)

        # Build a fully valid DriverPlacement using the canonical default T/S factory
        dp = DriverPlacement(
            spec=new_spec,
            terminal=default_terminal_model(driver_id),
            driver_id=driver_id,
            face_placement=fp,
        )
        self._state.drivers.append(dp)
        self._refresh_canvas()
        self.driversChanged.emit()

    def _on_canvas_driver_deleted(self, idx: int) -> None:
        """User deleted a driver via context menu."""
        if 0 <= idx < len(self._state.drivers):
            self._state.drivers.pop(idx)
            self._refresh_canvas()
            self.driversChanged.emit()

    def _on_canvas_driver_edited(self, idx: int) -> None:
        """User requested 'Edit T/S' from context menu → open TSDialog."""
        if idx < 0 or idx >= len(self._state.drivers):
            return
        from beamsim2.gui.parameters_panel import TSDialog

        dp = self._state.drivers[idx]
        dlg = TSDialog(dp, parent=self)
        if dlg.exec():
            result = dlg.placement
            if result is not None:
                # Preserve the face_placement; re-derive spec
                result.face_placement = dp.face_placement
                if result.face_placement is not None:
                    w, h, d = self._w.value(), self._h.value(), self._d.value()
                    result.spec = face_local_to_spec(result.face_placement, w, h, d)  # type: ignore[misc]
                self._state.drivers[idx] = result
                self._refresh_canvas()
                self.driversChanged.emit()

    def _on_canvas_driver_moved(self, idx: int, face_placement: FacePlacement) -> None:
        """Drag released: commit the new FacePlacement to AppState."""
        if idx < 0 or idx >= len(self._state.drivers):
            return
        dp = self._state.drivers[idx]
        dp.face_placement = face_placement  # type: ignore[misc]
        w, h, d = self._w.value(), self._h.value(), self._d.value()
        dp.spec = face_local_to_spec(face_placement, w, h, d)  # type: ignore[misc]
        self.driversChanged.emit()

    # ── Preview mesh slot ──────────────────────────────────────────────────────

    def _on_preview(self) -> None:
        """Build the BEM mesh for the current geometry + drivers and show health."""
        from beamsim2.core.types import SolverConfig
        from beamsim2.geometry.mesh import mesh_geometry
        from beamsim2.pipeline.run import BoxGeometry

        w, h, d = self._w.value(), self._h.value(), self._d.value()

        # Build driver specs — use face_placement if present, else use spec directly
        if self._state.drivers:
            drivers = [dp.spec for dp in self._state.drivers]
        else:
            # Placeholder: a small driver at the +z face centroid so mesh_geometry works
            drivers = [
                DriverSpec(
                    center=(w / 2, h / 2, d),
                    normal=(0.0, 0.0, 1.0),
                    radius=0.020,
                )
            ]

        config = self._state.config or SolverConfig()

        try:
            mesh, bc, health = mesh_geometry(
                width=w,
                height=h,
                depth=d,
                drivers=drivers,
                config=config,
                f_max=1000.0,
                fillet_radius=self._fi.value(),
            )
        except ValueError as exc:
            self._health_label.setText(f"✗ {exc}")
            self._health_label.setStyleSheet("color: #cc0000;")
            return

        # Update AppState geometry

        self._state.geometry = BoxGeometry(
            width=w,
            height=h,
            depth=d,
            fillet_radius=self._fi.value(),
        )
        self.geometryChanged.emit()

        # Health report
        if health.is_watertight and not health.problems:
            self._health_label.setText(
                f"✔ Watertight — {len(mesh.triangles)} triangles, " f"{len(mesh.vertices)} vertices"
            )
            self._health_label.setStyleSheet("color: #006600;")
        else:
            problems_text = "\n".join(health.problems)
            self._health_label.setText(f"⚠ {problems_text}")
            self._health_label.setStyleSheet("color: #cc6600;")

        # Show the mesh in the fallback matplotlib canvas (if PV not available)
        if not _PV_OK and isinstance(self._editor, _MeshCanvas):
            self._editor.render(mesh, health)
