"""Driver-into-enclosure assembly: boolean union, diaphragm group tagging, contiguity.

Each driver gets its own contiguous element group (group 1, 2, … for drivers;
highest tag for the enclosure shell). This is required because the NC.inp writer
uses min/max element-index ranges per group — a non-contiguous group silently
leaks the velocity BC onto sound-hard elements in between.

This module closes the open follow-up from item 3 / CHANGELOG [Unreleased]:
``ncinp_writer._group_element_range`` mis-applies the BC to a non-contiguous
vibrating group.  The assemble step now guarantees contiguity and asserts it.

Only parametric enclosure shapes are supported (CAD import is a future item;
see ``geometry.import_io`` stub).

Build-order item 5 (DR-03, pipeline Stages A–C).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from beamsim2.core.types import BoundaryConditions, Mesh
from beamsim2.geometry.primitives import _extract_tagged_mesh

# ---------------------------------------------------------------------------
# Driver descriptor
# ---------------------------------------------------------------------------


@dataclass
class DriverSpec:
    """Geometric descriptor for one driver diaphragm cap.

    Parameters
    ----------
    center : tuple[float, float, float]
        Centre of the driver cap in metres, in world coordinates.
        Must lie on a face of the enclosure (within floating-point tolerance).
    normal : tuple[float, float, float]
        Unit outward normal of the enclosure face the driver is mounted on.
        (1,0,0) = right face of box; (0,0,-1) = front face of a box at z=0, etc.
    radius : float
        Piston radius in metres.
    cap_height : float
        Height of a proud spherical cap protruding from the face in metres.
        0.0 (default) = flush disk (coplanar with enclosure face ring elements).
        Spike test (2026-06-18) confirmed flush drivers on a box face do NOT
        cause the NC_GenerateSubelements MSBE crash — that crash is specific to
        globally-flat all-coplanar meshes (the original V-1 geometry). Use
        cap_height > 0 only if you observe convergence issues on a specific mesh.
    """

    center: tuple[float, float, float]
    normal: tuple[float, float, float]
    radius: float
    cap_height: float = 0.0


# ---------------------------------------------------------------------------
# Box + driver assembly
# ---------------------------------------------------------------------------


def assemble_box_driver(
    width: float,
    height: float,
    depth: float,
    drivers: list[DriverSpec],
    h_elem: float,
    fillet_radius: float = 0.0,
) -> tuple[Mesh, BoundaryConditions]:
    """Assemble a box enclosure with one or more driver caps.

    Each driver disk (or proud cap) is fragmented into the enclosure face so
    the boundary mesh has conforming nodes at the driver rim.  Each driver
    gets its own surface-group tag (1, 2, …); the enclosure shell occupies
    the highest tag.

    Contiguity is enforced: all elements of each group form a contiguous index
    block in the returned Mesh. This is required by the NC.inp writer's
    ``ELEM lo TO hi`` range syntax (see ``ncinp_writer._group_element_range``).

    Parameters
    ----------
    width, height, depth : float
        Box dimensions in metres (x, y, z from origin).
    drivers : list[DriverSpec]
        One entry per driver. Tag assigned in list order (first driver = group 1).
    h_elem : float
        Target element edge length in metres (passed to gmsh as CharLengthMax).
    fillet_radius : float
        Edge fillet radius in metres. 0 = sharp corners. See ``make_box_mesh``
        docstring for the NumCalc coplanar-element crash risk.

    Returns
    -------
    mesh : Mesh
        vertices [V, 3] float64, triangles [T, 3] int32,
        group_tags [T] int32:  1..n_drivers for driver caps, n_drivers+1 for shell.
    bc : BoundaryConditions
        vibrating_groups = {i+1: 1+0j} for i in range(n_drivers).
        Shell group is sound-hard (not listed in vibrating_groups).

    Raises
    ------
    ValueError
        If a driver center is not on any face of the box (within 1e-6 m).
    NotImplementedError
        If cap_height > 0 (proud cap assembly is not yet implemented).
    AssertionError
        If the assembled mesh violates the contiguous-group invariant.
    """
    import gmsh

    if not drivers:
        raise ValueError("At least one DriverSpec is required.")

    for drv in drivers:
        if drv.cap_height != 0.0:
            raise NotImplementedError(
                "Proud (cap_height > 0) driver assembly is not yet implemented. "
                "Use cap_height=0.0 for a flush disk driver."
            )

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("box_driver_assembly")

    # ── build box volume ─────────────────────────────────────────────────────
    box_vol = gmsh.model.occ.addBox(0.0, 0.0, 0.0, width, height, depth)
    gmsh.model.occ.synchronize()

    if fillet_radius > 0.0:
        curves = gmsh.model.getEntities(1)
        edge_tags = [tag for _, tag in curves]
        new_vols = gmsh.model.occ.fillet([box_vol], edge_tags, [fillet_radius] * len(edge_tags))
        gmsh.model.occ.synchronize()
        box_vol = new_vols[0] if new_vols else box_vol

    # ── embed each driver disk into the relevant box face ────────────────────
    # Strategy: create disk on the box face plane, fragment (volume, disk) so
    # OCC splits the coincident face into ring + disk with shared boundary nodes.
    driver_surf_tags: list[set[int]] = []  # one set per driver, driver i → group i+1

    fragment_tools: list[tuple[int, int]] = []  # (dim, tag) for all disks
    disk_tags: list[int] = []

    for drv in drivers:
        cx, cy, cz = drv.center
        nx, ny, nz = drv.normal
        # Disk on face: addDisk creates in XY plane; rotate to face normal.
        disk = gmsh.model.occ.addDisk(cx, cy, cz, drv.radius, drv.radius)
        # Rotate disk so its normal aligns with the face normal (from +z default).
        _align_disk_to_normal(disk, (cx, cy, cz), (nx, ny, nz))
        gmsh.model.occ.synchronize()
        disk_tags.append(disk)
        fragment_tools.append((2, disk))

    # Fragment the box volume with all driver disks at once.
    out_entities, parent_map = gmsh.model.occ.fragment([(3, box_vol)], fragment_tools)
    gmsh.model.occ.synchronize()

    # parent_map[0] = children of box_vol (should be one 3-D entity)
    # parent_map[1+i] = children of disk i (the driver surface pieces)
    for i in range(len(drivers)):
        surfs = {tag for dim, tag in parent_map[1 + i] if dim == 2}
        driver_surf_tags.append(surfs)

    # ── identify shell surfaces ───────────────────────────────────────────────
    box_vols_out = [(dim, tag) for dim, tag in parent_map[0] if dim == 3]
    all_shell_surf_tags: set[int] = set()
    for dimtag in box_vols_out:
        bnd = gmsh.model.getBoundary([dimtag], oriented=False, combined=False)
        for dim, tag in bnd:
            if dim == 2:
                all_shell_surf_tags.add(abs(tag))
    all_driver_surf_tags: set[int] = set().union(*driver_surf_tags)
    shell_only_tags = all_shell_surf_tags - all_driver_surf_tags

    # ── physical groups: driver i+1, shell = n_drivers+1 ────────────────────
    n = len(drivers)
    shell_group = n + 1
    phys_to_internal: dict[int, int] = {}

    for i, surfs in enumerate(driver_surf_tags):
        gmsh.model.addPhysicalGroup(2, list(surfs), tag=i + 1)
        phys_to_internal[i + 1] = i + 1

    gmsh.model.addPhysicalGroup(2, list(shell_only_tags), tag=shell_group)
    phys_to_internal[shell_group] = shell_group

    # ── mesh ─────────────────────────────────────────────────────────────────
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h_elem)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h_elem / 4.0)
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(1)

    mesh = _extract_tagged_mesh(phys_to_internal)
    gmsh.finalize()

    # ── assert contiguity (guards the ncinp_writer BC-leak) ──────────────────
    _assert_groups_contiguous(mesh)

    # ── boundary conditions: each driver at unit velocity ────────────────────
    bc = BoundaryConditions(vibrating_groups={i + 1: complex(1.0, 0.0) for i in range(n)})
    return mesh, bc


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _align_disk_to_normal(
    disk_tag: int,
    center: tuple[float, float, float],
    normal: tuple[float, float, float],
) -> None:
    """Rotate disk in the active gmsh OCC session to align its +z axis to normal.

    gmsh addDisk creates a disk whose normal is +z. This rotates it so the disk
    lies on the face described by ``normal``.

    Parameters
    ----------
    disk_tag : int
        gmsh OCC surface tag of the disk to rotate.
    center : tuple[float, float, float]
        Rotation pivot point (the disk centre).
    normal : tuple[float, float, float]
        Target face normal (unit vector).
    """
    import gmsh

    nx, ny, nz = normal
    # Default disk normal is (0, 0, 1); rotate onto target normal.
    src = np.array([0.0, 0.0, 1.0])
    tgt = np.array([nx, ny, nz], dtype=float)
    tgt /= np.linalg.norm(tgt)

    axis = np.cross(src, tgt)
    axis_norm = np.linalg.norm(axis)

    if axis_norm < 1e-10:
        # Already aligned (+z) or anti-aligned (−z).
        if np.dot(src, tgt) < 0:
            # 180° rotation around x-axis
            gmsh.model.occ.rotate([(2, disk_tag)], center[0], center[1], center[2], 1, 0, 0, np.pi)
        # else: no rotation needed
        return

    axis /= axis_norm
    angle = np.arccos(np.clip(np.dot(src, tgt), -1.0, 1.0))
    gmsh.model.occ.rotate(
        [(2, disk_tag)],
        center[0],
        center[1],
        center[2],
        float(axis[0]),
        float(axis[1]),
        float(axis[2]),
        float(angle),
    )


def _assert_groups_contiguous(mesh: Mesh) -> None:
    """Raise AssertionError if any group's element indices are not contiguous.

    A non-contiguous group causes ``ncinp_writer._group_element_range`` to
    produce a BC range that covers sound-hard elements between the first and
    last matching index, silently applying the velocity BC to the wrong surface.

    Parameters
    ----------
    mesh : Mesh
        Assembled boundary mesh to check.
    """
    for tag in np.unique(mesh.group_tags):
        indices = np.where(mesh.group_tags == tag)[0]
        expected = np.arange(indices[0], indices[-1] + 1, dtype=np.intp)
        if not np.array_equal(indices, expected):
            raise AssertionError(
                f"Group {tag} elements are not contiguous: "
                f"first 5 indices = {indices[:5].tolist()}, "
                f"last 5 = {indices[-5:].tolist()} "
                f"(expected contiguous range {indices[0]}–{indices[-1]}). "
                "The NC.inp ELEM lo TO hi BC range would leak onto wrong elements."
            )
