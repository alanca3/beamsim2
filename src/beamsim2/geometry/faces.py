"""Face-local driver placement model for the parametric box enclosure.

The box built by ``assemble_box_driver`` spans ``[0,w] × [0,h] × [0,d]`` (origin corner,
NOT centred).  A driver's placement is stored as face-local offsets ``(u, v)`` from the
face centroid, which guarantees the derived world ``center`` is exactly on the face plane —
eliminating the "Mesh watertight failure" caused by slightly-off typed coordinates.

This module is **Qt-free and gmsh-free** (numpy/dataclass only) so it can be CI-tested
without a display, and shared between:

- ``geometry.assemble`` — the headless backstop ``ValueError`` guard.
- ``gui.geometry_view`` — live pick-and-drag placement editor.
- ``pipeline.run`` — ``DriverPlacement.face_placement`` field.

Coordinate convention
---------------------
face_id → (normal, centroid, u_hat, v_hat, half_u, half_v):

  id  name       normal      centroid             u_hat  v_hat  half_u  half_v
  --  ---------  ----------  -------------------  -----  -----  ------  ------
   0  +z front   (0,0, 1)    (w/2, h/2,  d)       +x     +y     w/2     h/2
   1  −z back    (0,0,−1)    (w/2, h/2,  0)       +x     +y     w/2     h/2
   2  +x right   (1, 0, 0)   (w,   h/2, d/2)      +y     +z     h/2     d/2
   3  −x left    (−1,0, 0)   (0,   h/2, d/2)      +y     +z     h/2     d/2
   4  +y top     (0, 1, 0)   (w/2,  h,  d/2)      +x     +z     w/2     d/2
   5  −y bottom  (0,−1, 0)   (w/2,  0,  d/2)      +x     +z     w/2     d/2

u_hat/v_hat are the two world axes that are NOT the normal axis, in ascending
order (+x before +y before +z), always positive direction.  The face centroid is
the midpoint of the face rectangle.  Derived center = centroid + u·u_hat + v·v_hat.

Build-order item 10 follow-up (interactive driver placement, GUI usability).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

Vec3 = tuple[float, float, float]

# face_id → combo label in the exact order of TSDialog's _normal_combo
FACE_NAMES = [
    "+z (front/top)",
    "-z (back)",
    "+x (right)",
    "-x (left)",
    "+y (top)",
    "-y (bottom)",
]

_NORMALS: list[Vec3] = [
    (0.0, 0.0, 1.0),
    (0.0, 0.0, -1.0),
    (1.0, 0.0, 0.0),
    (-1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, -1.0, 0.0),
]

# Public alias.  By construction ``FACE_NORMALS[i]`` is the outward normal of face
# ``i`` and matches ``FACE_NAMES[i]`` — so a UI combo built from these is index-aligned
# with ``face_id`` (relied on by the orientation round-trip in the driver editor).
FACE_NORMALS: list[Vec3] = _NORMALS


@dataclass
class FacePlacement:
    """GUI source of truth for one driver's placement.

    Parameters
    ----------
    face_id : int
        0..5 — selects one of the six axis-aligned box faces (see module table).
    u : float
        Offset along ``u_hat`` from the face centroid in metres.
    v : float
        Offset along ``v_hat`` from the face centroid in metres.
    radius : float
        Piston radius in metres.  Also determines the fit predicate.
    """

    face_id: int
    u: float
    v: float
    radius: float


@dataclass(frozen=True)
class FaceBasis:
    """Geometric constants for one face of the box.

    All fields derived from the box dimensions ``(w, h, d)`` and the face_id.
    Immutable so it can be cached freely.

    Parameters
    ----------
    face_id : int
    normal : Vec3
        Outward unit normal of the face.
    centroid : Vec3
        World-space midpoint of the face rectangle.
    u_hat : Vec3
        First in-plane unit direction (one of ±x/y/z).
    v_hat : Vec3
        Second in-plane unit direction (orthogonal to normal and u_hat).
    half_u : float
        Half the face width along u_hat in metres.
    half_v : float
        Half the face height along v_hat in metres.
    """

    face_id: int
    normal: Vec3
    centroid: Vec3
    u_hat: Vec3
    v_hat: Vec3
    half_u: float
    half_v: float


# ---------------------------------------------------------------------------
# Core construction
# ---------------------------------------------------------------------------


def face_basis(face_id: int, w: float, h: float, d: float) -> FaceBasis:
    """Return the geometric constants for one face of a box at origin corner.

    Parameters
    ----------
    face_id : int
        0..5 — see the module-level coordinate table.
    w, h, d : float
        Box width (x), height (y), depth (z) in metres.

    Returns
    -------
    FaceBasis

    Raises
    ------
    ValueError
        If face_id is not in 0..5.
    """
    if face_id == 0:  # +z front
        return FaceBasis(
            face_id=0,
            normal=(0.0, 0.0, 1.0),
            centroid=(w / 2, h / 2, d),
            u_hat=(1.0, 0.0, 0.0),
            v_hat=(0.0, 1.0, 0.0),
            half_u=w / 2,
            half_v=h / 2,
        )
    elif face_id == 1:  # -z back
        return FaceBasis(
            face_id=1,
            normal=(0.0, 0.0, -1.0),
            centroid=(w / 2, h / 2, 0.0),
            u_hat=(1.0, 0.0, 0.0),
            v_hat=(0.0, 1.0, 0.0),
            half_u=w / 2,
            half_v=h / 2,
        )
    elif face_id == 2:  # +x right
        return FaceBasis(
            face_id=2,
            normal=(1.0, 0.0, 0.0),
            centroid=(w, h / 2, d / 2),
            u_hat=(0.0, 1.0, 0.0),
            v_hat=(0.0, 0.0, 1.0),
            half_u=h / 2,
            half_v=d / 2,
        )
    elif face_id == 3:  # -x left
        return FaceBasis(
            face_id=3,
            normal=(-1.0, 0.0, 0.0),
            centroid=(0.0, h / 2, d / 2),
            u_hat=(0.0, 1.0, 0.0),
            v_hat=(0.0, 0.0, 1.0),
            half_u=h / 2,
            half_v=d / 2,
        )
    elif face_id == 4:  # +y top
        return FaceBasis(
            face_id=4,
            normal=(0.0, 1.0, 0.0),
            centroid=(w / 2, h, d / 2),
            u_hat=(1.0, 0.0, 0.0),
            v_hat=(0.0, 0.0, 1.0),
            half_u=w / 2,
            half_v=d / 2,
        )
    elif face_id == 5:  # -y bottom
        return FaceBasis(
            face_id=5,
            normal=(0.0, -1.0, 0.0),
            centroid=(w / 2, 0.0, d / 2),
            u_hat=(1.0, 0.0, 0.0),
            v_hat=(0.0, 0.0, 1.0),
            half_u=w / 2,
            half_v=d / 2,
        )
    else:
        raise ValueError(f"face_id must be 0..5, got {face_id!r}")


def all_face_bases(w: float, h: float, d: float) -> list[FaceBasis]:
    """Return FaceBasis for all six faces in face_id order (0..5).

    Parameters
    ----------
    w, h, d : float
        Box dimensions in metres.

    Returns
    -------
    list[FaceBasis]
        Length 6, indexed by face_id.
    """
    return [face_basis(i, w, h, d) for i in range(6)]


# ---------------------------------------------------------------------------
# Derived quantities
# ---------------------------------------------------------------------------


def face_local_to_center(fp: FacePlacement, w: float, h: float, d: float) -> Vec3:
    """Convert face-local (u, v) offsets to a world-space center point.

    The returned point is guaranteed to lie on the face plane within floating-point
    precision, so ``DriverSpec.center`` derived from it will not cause a watertight
    failure in ``assemble_box_driver``.

    Parameters
    ----------
    fp : FacePlacement
        Source-of-truth placement (face_id, u, v, radius).
    w, h, d : float
        Box dimensions in metres.

    Returns
    -------
    Vec3
        (x, y, z) world-space center of the driver cap.
    """
    b = face_basis(fp.face_id, w, h, d)
    cx = b.centroid[0] + fp.u * b.u_hat[0] + fp.v * b.v_hat[0]
    cy = b.centroid[1] + fp.u * b.u_hat[1] + fp.v * b.v_hat[1]
    cz = b.centroid[2] + fp.u * b.u_hat[2] + fp.v * b.v_hat[2]
    return (cx, cy, cz)


def face_local_to_spec(fp: FacePlacement, w: float, h: float, d: float):
    """Convert a FacePlacement to a DriverSpec.

    The DriverSpec.center is derived analytically (no off-plane rounding), so the
    BEM mesh fragment will succeed.  DriverSpec.normal matches the face outward normal.

    Parameters
    ----------
    fp : FacePlacement
    w, h, d : float

    Returns
    -------
    DriverSpec
        Solver-facing descriptor guaranteed on-plane and within the box face.
    """
    # Lazy import avoids circular dependency (assemble imports faces for the backstop).
    from beamsim2.geometry.assemble import DriverSpec

    b = face_basis(fp.face_id, w, h, d)
    return DriverSpec(
        center=face_local_to_center(fp, w, h, d),
        normal=b.normal,
        radius=fp.radius,
    )


# ---------------------------------------------------------------------------
# Fit / validation
# ---------------------------------------------------------------------------


def fits_on_face(fp: FacePlacement, w: float, h: float, d: float) -> bool:
    """Return True if the driver disk lies entirely within the face rectangle.

    LEAP rule: "All edges of the transducer must fit within the face."

    Parameters
    ----------
    fp : FacePlacement
    w, h, d : float

    Returns
    -------
    bool
    """
    b = face_basis(fp.face_id, w, h, d)
    return (abs(fp.u) + fp.radius <= b.half_u + 1e-12) and (
        abs(fp.v) + fp.radius <= b.half_v + 1e-12
    )


def clamp_uv_to_face(
    face_id: int, u: float, v: float, radius: float, w: float, h: float, d: float
) -> tuple[float, float]:
    """Clamp (u, v) so the driver disk (of given radius) stays inside the face.

    Used by the drag handler to lock a driver to its face rectangle (LEAP-style).

    Parameters
    ----------
    face_id : int
    u, v : float
        Current (possibly out-of-bounds) face-local offsets in metres.
    radius : float
        Driver piston radius in metres.
    w, h, d : float

    Returns
    -------
    (u_clamped, v_clamped) : tuple[float, float]
    """
    b = face_basis(face_id, w, h, d)
    lim_u = max(0.0, b.half_u - radius)
    lim_v = max(0.0, b.half_v - radius)
    return (max(-lim_u, min(lim_u, u)), max(-lim_v, min(lim_v, v)))


def world_to_face_uv(
    face_id: int, point: Vec3, w: float, h: float, d: float
) -> tuple[float, float]:
    """Project a world-space point onto the face plane and return (u, v) offsets.

    Used by the drag handler each ``MouseMoveEvent``: after intersecting the camera
    ray with the face plane, convert the hit point to local u/v for clamping.

    Parameters
    ----------
    face_id : int
    point : Vec3
        World-space point (should be on or near the face plane).
    w, h, d : float

    Returns
    -------
    (u, v) : tuple[float, float]
        Offsets from the face centroid along (u_hat, v_hat).
    """
    b = face_basis(face_id, w, h, d)
    dx = point[0] - b.centroid[0]
    dy = point[1] - b.centroid[1]
    dz = point[2] - b.centroid[2]
    u = dx * b.u_hat[0] + dy * b.u_hat[1] + dz * b.u_hat[2]
    v = dx * b.v_hat[0] + dy * b.v_hat[1] + dz * b.v_hat[2]
    return (u, v)


def classify_face(
    point: Vec3,
    normal_hint: Optional[Vec3],
    w: float,
    h: float,
    d: float,
) -> int:
    """Return the face_id (0..5) of the face nearest to a picked world point.

    Two strategies:

    1. If ``normal_hint`` is provided (e.g. from a VTK cell-picker normal), pick
       the face whose outward normal best aligns with the hint.
    2. Otherwise, pick the face whose plane the point is closest to.

    Parameters
    ----------
    point : Vec3
        World-space picked position.
    normal_hint : Vec3 or None
        Picked cell normal (approximate), or None.
    w, h, d : float

    Returns
    -------
    int
        face_id in 0..5.
    """
    if normal_hint is not None:
        n = np.array(normal_hint, dtype=float)
        norm = np.linalg.norm(n)
        if norm > 1e-12:
            n /= norm
            dots = [np.dot(n, np.array(nb, dtype=float)) for nb in _NORMALS]
            return int(np.argmax(dots))

    # Fallback: minimum distance from point to each of the 6 face planes.
    x, y, z = point
    # Distance from the plane (which has a known coordinate value):
    # face 0: z = d,  face 1: z = 0,  face 2: x = w,  face 3: x = 0,
    # face 4: y = h,  face 5: y = 0
    distances = [
        abs(z - d),  # face 0 (+z)
        abs(z - 0),  # face 1 (-z)
        abs(x - w),  # face 2 (+x)
        abs(x - 0),  # face 3 (-x)
        abs(y - h),  # face 4 (+y)
        abs(y - 0),  # face 5 (-y)
    ]
    return int(np.argmin(distances))


def face_id_from_normal(normal: Vec3) -> int:
    """Return the face_id (0..5) whose outward normal best matches ``normal``.

    The driver editor's "Face normal" combo is index-aligned with ``face_id`` (its
    i-th entry is :data:`FACE_NORMALS` ``[i]``), so this is the inverse map used to
    pre-fill the combo from a stored ``DriverSpec.normal`` and to detect when an
    edit re-orients a driver onto a different box face.

    Parameters
    ----------
    normal : Vec3
        A (not necessarily unit) outward normal in world coordinates.

    Returns
    -------
    int
        face_id in 0..5 (the closest axis-aligned face by dot product).  A zero
        vector falls back to face 0 (+z front).
    """
    n = np.asarray(normal, dtype=float).reshape(3)
    nn = float(np.linalg.norm(n))
    if nn < 1e-12:
        return 0
    n = n / nn
    dots = [float(np.dot(n, np.asarray(nb, dtype=float))) for nb in _NORMALS]
    return int(np.argmax(dots))


def reconcile_placement(
    chosen_normal: Vec3,
    old_fp: Optional[FacePlacement],
    radius: float,
    w: float,
    h: float,
    d: float,
) -> tuple["object", FacePlacement]:
    """Reconcile a driver-editor orientation choice into a consistent (spec, placement).

    The single rule shared by *both* editor paths (the Drivers-list "Edit" button and
    the 3-D canvas right-click "Edit T/S"), so an edited orientation persists identically
    no matter where it was edited (Bug #3 — face-normal authority).  ``FacePlacement`` is
    the source of truth: ``DriverSpec`` is always derived from it, so the two can never
    silently disagree (which is what made an edited orientation revert).

    Behaviour
    ---------
    - **Same face** (chosen normal still points at the driver's current face): keep the
      face-local ``(u, v)`` position, re-clamp it so a possibly-enlarged radius still fits.
    - **New face** (the user picked a different orientation): move the driver to that face,
      placed at the face centroid (``u = v = 0``), with the radius clamped to fit the new
      (possibly smaller) face — so a re-orient never produces an off-face / overflowing disc.

    Parameters
    ----------
    chosen_normal : Vec3
        The outward normal selected in the editor's "Face normal" combo.
    old_fp : FacePlacement or None
        The driver's existing face placement (``None`` is treated as "no prior face").
    radius : float
        The driver piston radius (metres) from the editor.
    w, h, d : float
        Current box dimensions in metres.

    Returns
    -------
    (spec, fp) : tuple[DriverSpec, FacePlacement]
        A guaranteed-consistent pair: ``fp`` is the new face placement and ``spec`` is
        ``face_local_to_spec(fp, w, h, d)`` (on-plane, within the face).
    """
    new_face = face_id_from_normal(chosen_normal)
    b = face_basis(new_face, w, h, d)

    if old_fp is not None and old_fp.face_id == new_face:
        # Same face: keep position, re-clamp in case the radius changed.
        u, v = clamp_uv_to_face(new_face, old_fp.u, old_fp.v, radius, w, h, d)
        fp = FacePlacement(face_id=new_face, u=u, v=v, radius=radius)
    else:
        # Re-oriented onto a new face: centre it and shrink to fit the new face.
        r = min(radius, b.half_u, b.half_v)
        fp = FacePlacement(face_id=new_face, u=0.0, v=0.0, radius=r)

    return face_local_to_spec(fp, w, h, d), fp


# ---------------------------------------------------------------------------
# Reference-axis (0° measurement / virtual-microphone) indicator
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AxisIndicator:
    """Placement geometry for the 3-D editor's reference-axis / virtual-mic glyph.

    Pure (numpy-only) so the placement can be CI-tested without an OpenGL context;
    the GUI turns it into VTK actors in ``geometry_view``.  All points are world
    coordinates in the box frame ``[0,w] × [0,h] × [0,d]``.

    Parameters
    ----------
    origin : np.ndarray
        ``[3]`` box geometric centre — where the axis arrow starts.
    direction : np.ndarray
        ``[3]`` unit reference axis (the loudspeaker's 0°/on-axis "front").
    tip : np.ndarray
        ``[3]`` arrow tip = ``origin + direction * length`` (the mic stand-off point).
    mic_pos : np.ndarray
        ``[3]`` centre of the virtual-microphone glyph (== ``tip``).
    length : float
        Arrow length in metres (a view-scaled stand-off, NOT the true mic distance).
    """

    origin: np.ndarray
    direction: np.ndarray
    tip: np.ndarray
    mic_pos: np.ndarray
    length: float


def reference_axis_indicator(
    axis: Vec3,
    w: float,
    h: float,
    d: float,
    standoff_factor: float = 1.6,
) -> AxisIndicator:
    """Compute the reference-axis arrow + virtual-mic placement for the 3-D editor.

    Answers Bug #1 ("no way of telling which axis is the 0° measurement axis"): the
    arrow points from the box centre along the measurement axis, with a microphone
    glyph at its tip, so the box-vs-mic orientation is unambiguous from any camera angle.

    The arrow length is a *view-scaled* stand-off proportional to the box size (so the
    glyph stays visible for any enclosure); it deliberately does NOT encode the true
    observation distance — the on-screen label conveys direction only.

    Parameters
    ----------
    axis : Vec3
        The measurement / reference axis (need not be unit; ``(0,0,0)`` falls back to +z).
    w, h, d : float
        Box dimensions in metres.
    standoff_factor : float, optional
        Arrow length as a multiple of the box's largest dimension (default 1.6).

    Returns
    -------
    AxisIndicator
        World-space placement of the arrow and microphone glyph.
    """
    # Reuse the canonical reference frame so the zero-axis fallback (+z) and
    # normalisation match every Results view (core.sphere.reference_frame).
    from beamsim2.core.sphere import reference_frame

    front, _right, _up = reference_frame(axis)  # front = unit reference axis
    origin = np.array([w / 2.0, h / 2.0, d / 2.0], dtype=float)
    length = float(standoff_factor * max(w, h, d))
    tip = origin + front * length
    return AxisIndicator(
        origin=origin,
        direction=front,
        tip=tip,
        mic_pos=tip,
        length=length,
    )


# ---------------------------------------------------------------------------
# Headless validation backstop
# ---------------------------------------------------------------------------


def validate_spec_on_box(
    center: Vec3,
    normal: Vec3,
    radius: float,
    w: float,
    h: float,
    d: float,
    tol: float = 1e-6,
) -> Optional[str]:
    """Check that a DriverSpec is on a box face and fits within it.

    Called by ``assemble_box_driver`` as a backstop before ``gmsh.initialize``,
    so callers get a clear, located error instead of a cryptic "Mesh watertight
    failure" deep inside the BEM mesher.

    Parameters
    ----------
    center : Vec3
        Driver cap centre in world coordinates.
    normal : Vec3
        Declared face outward normal (unit vector).
    radius : float
        Driver piston radius in metres.
    w, h, d : float
        Box dimensions in metres.
    tol : float
        Tolerance for the on-plane check in metres (default 1 µm).

    Returns
    -------
    str or None
        A plain-English error message if validation fails, else ``None``.
    """
    nx, ny, nz = normal

    # ── 1. Find the matching face (normal must be one of the 6 axis-aligned ones) ──
    best_id: Optional[int] = None
    best_dot = -2.0
    n_arr = np.array([nx, ny, nz], dtype=float)
    n_norm = np.linalg.norm(n_arr)
    if n_norm < 1e-12:
        return f"Driver normal {(nx,ny,nz)} is a zero vector — cannot determine face."
    n_arr /= n_norm

    for fid, n_ref in enumerate(_NORMALS):
        dot = float(np.dot(n_arr, np.array(n_ref, dtype=float)))
        if dot > best_dot:
            best_dot = dot
            best_id = fid

    if best_dot < 0.99:
        return (
            f"Driver normal ({nx:.4f}, {ny:.4f}, {nz:.4f}) does not match any of the "
            f"six axis-aligned box faces (best match was {FACE_NAMES[best_id]} with "
            f"dot={best_dot:.4f}).  Only axis-aligned normals are supported."
        )

    face_id = best_id  # type: ignore[assignment]
    b = face_basis(face_id, w, h, d)

    # ── 2. Check the center lies on the face plane ──────────────────────────
    cx, cy, cz = center
    c_arr = np.array([cx, cy, cz], dtype=float)
    centroid_arr = np.array(b.centroid, dtype=float)
    normal_arr = np.array(b.normal, dtype=float)
    dist_from_plane = float(abs(np.dot(c_arr - centroid_arr, normal_arr)))

    if dist_from_plane > tol:
        return (
            f"Driver center ({cx:.6f}, {cy:.6f}, {cz:.6f}) is {dist_from_plane*1000:.4f} mm "
            f"off the {FACE_NAMES[face_id]} face plane.  The BEM mesh would not be "
            f"watertight (OCC fragment needs the driver disk exactly on the face).  "
            f"Correct the center so it lies on the face plane within {tol*1e6:.0f} µm."
        )

    # ── 3. Check the disk fits within the face ──────────────────────────────
    u, v = world_to_face_uv(face_id, (cx, cy, cz), w, h, d)
    overflow_u = abs(u) + radius - b.half_u
    overflow_v = abs(v) + radius - b.half_v

    if overflow_u > 1e-12 or overflow_v > 1e-12:
        msg_parts = []
        if overflow_u > 1e-12:
            msg_parts.append(
                f"|u|+r = {abs(u)*100:.2f}+{radius*100:.2f} = "
                f"{(abs(u)+radius)*100:.2f} cm > half-width {b.half_u*100:.2f} cm"
            )
        if overflow_v > 1e-12:
            msg_parts.append(
                f"|v|+r = {abs(v)*100:.2f}+{radius*100:.2f} = "
                f"{(abs(v)+radius)*100:.2f} cm > half-height {b.half_v*100:.2f} cm"
            )
        detail = ";  ".join(msg_parts)
        return (
            f"Driver (r={radius*100:.2f} cm) on the {FACE_NAMES[face_id]} face overflows "
            f"the face boundary: {detail}.  "
            f"Reduce the radius or move the driver toward the face centroid."
        )

    return None  # all checks passed
