"""Parametric geometry builders for enclosure shells and driver diaphragm caps.

All builders use the gmsh OpenCASCADE kernel (DR-03) and return a Mesh plus
BoundaryConditions ready for the health-check and mesh-sizing pipeline.

The private helper `_extract_tagged_mesh` factored out of `make_piston_mesh`
so every builder shares one extraction path: getNodes/getElements, 1-based →
0-based index map, sort by group_tag for contiguous blocks, outward-normal check.
"""

from __future__ import annotations

import numpy as np

from beamsim2.core.types import BoundaryConditions, Mesh

# ---------------------------------------------------------------------------
# Internal extraction helper
# ---------------------------------------------------------------------------


def _extract_tagged_mesh(phys_to_internal: dict[int, int]) -> Mesh:
    """Extract a triangulated surface Mesh from the active gmsh session.

    Reads nodes and first-order triangular elements for each listed physical
    group, maps 1-based gmsh node tags to 0-based indices, sorts triangles by
    internal group_tag (contiguous-block property), and enforces outward normals.

    Parameters
    ----------
    phys_to_internal : dict[int, int]
        Maps gmsh physical-group tag (dim=2) → internal mesh group_tag to embed
        in Mesh.group_tags. Only dim-2 (surface) entities are read.

    Returns
    -------
    Mesh
        vertices [V, 3] float64, triangles [T, 3] int32,
        group_tags [T] int32 (sorted ascending → contiguous blocks).

    Raises
    ------
    ValueError
        If no triangular elements are found in the specified physical groups.
    """
    import gmsh

    # ── nodes ────────────────────────────────────────────────────────────────
    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    node_tag_to_idx: dict[int, int] = {int(t): i for i, t in enumerate(node_tags)}
    vertices = np.array(coords, dtype=np.float64).reshape(-1, 3)  # [V, 3] float64

    tri_list: list[list[int]] = []
    tag_list: list[int] = []

    for phys_tag, int_tag in phys_to_internal.items():
        ents = gmsh.model.getEntitiesForPhysicalGroup(2, phys_tag)
        for ent in ents:
            etype_list, _, enode_tags_list = gmsh.model.mesh.getElements(2, ent)
            for etype, enodes in zip(etype_list, enode_tags_list):
                if etype != 2:  # 2 = first-order triangle in gmsh
                    continue
                tris = np.array(enodes, dtype=np.int64).reshape(-1, 3)
                for row in tris:
                    tri_list.append([node_tag_to_idx[int(n)] for n in row])
                    tag_list.append(int_tag)

    if not tri_list:
        raise ValueError(
            f"No triangular elements found in physical groups {list(phys_to_internal)}."
        )

    # ── sort by group_tag → contiguous blocks ────────────────────────────────
    tag_arr = np.array(tag_list, dtype=np.int32)  # [T]
    order = np.argsort(tag_arr, kind="stable")
    triangles = np.array(tri_list, dtype=np.int32)[order]  # [T, 3] int32
    group_tags = tag_arr[order]  # [T] int32

    # ── outward-normal check ─────────────────────────────────────────────────
    # Outward means the face normal points away from the surface's centre of mass.
    # Works for all convex shapes; adequate for the parametric primitives here.
    center = vertices.mean(axis=0)  # [3] float64
    v0 = vertices[triangles[:, 0]]  # [T, 3]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)  # [T, 3] unnormalised
    centroids = (v0 + v1 + v2) / 3.0  # [T, 3]
    dots = np.einsum("ij,ij->i", normals, centroids - center)  # [T]
    flip = dots < 0
    if flip.any():
        triangles[flip] = triangles[flip][:, [0, 2, 1]]  # reverse winding

    return Mesh(vertices=vertices, triangles=triangles, group_tags=group_tags)


# ---------------------------------------------------------------------------
# Sphere mesh (pulsating sphere — V-2 physics canary)
# ---------------------------------------------------------------------------


def make_sphere_mesh(
    radius: float,
    h_elem: float,
    all_vibrating: bool = True,
) -> tuple[Mesh, BoundaryConditions]:
    """Build a gmsh OCC sphere mesh for use as a pulsating-sphere BEM source.

    This is the gmsh-path equivalent of ``validation.sphere_benchmark.make_pulsating_
    sphere_mesh``.  Passing the result through ``sphere_benchmark_errors`` at the
    same ≤ 0.5 dB gate proves the gmsh extraction path is solver-equivalent to the
    trusted synthetic icosphere path (V-2 physics canary, ``@local_only``).

    Parameters
    ----------
    radius : float
        Sphere radius in metres.
    h_elem : float
        Target element edge length in metres.
    all_vibrating : bool
        If True (default) all elements are vibrating (group 1, VELO = 1+0j),
        matching the pulsating-sphere BC from ``make_pulsating_sphere_mesh``.
        If False, all elements are sound-hard (group 1 in sound_hard_groups).

    Returns
    -------
    mesh : Mesh
        vertices [V, 3] float64, triangles [T, 3] int32, group_tags [T] int32 = all 1.
    bc : BoundaryConditions
        vibrating_groups = {1: 1+0j} if all_vibrating, else sound_hard_groups = {1}.
    """
    import gmsh

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("sphere_mesh")

    sphere_vol = gmsh.model.occ.addSphere(0.0, 0.0, 0.0, radius)
    gmsh.model.occ.synchronize()

    bnd = gmsh.model.getBoundary([(3, sphere_vol)], oriented=False, combined=False)
    surf_tags = [abs(tag) for dim, tag in bnd if dim == 2]
    gmsh.model.addPhysicalGroup(2, surf_tags, tag=1)

    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h_elem)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h_elem / 4.0)
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(1)

    mesh = _extract_tagged_mesh({1: 1})
    gmsh.finalize()

    if all_vibrating:
        bc = BoundaryConditions(vibrating_groups={1: complex(1.0, 0.0)})
    else:
        bc = BoundaryConditions(vibrating_groups={}, sound_hard_groups={1})

    return mesh, bc


# ---------------------------------------------------------------------------
# Box mesh (sound-hard enclosure shell, no driver)
# ---------------------------------------------------------------------------


def make_box_mesh(
    width: float,
    height: float,
    depth: float,
    fillet_radius: float = 0.0,
    h_elem: float = 0.01,
) -> tuple[Mesh, BoundaryConditions]:
    """Build a closed box surface mesh, all elements sound-hard.

    Provides a standalone box mesh without a driver for health-check and
    timing tests. Use ``geometry.assemble.assemble_box_driver`` to add a
    vibrating driver cap.

    When ``fillet_radius > 0`` the 12 box edges are rounded, giving every
    pair of elements on adjacent faces a non-zero perpendicular offset ε.
    This avoids the ``NC_GenerateSubelements`` MSBE crash that occurs for
    fully coplanar element pairs (see CLAUDE.md Gotchas).

    Parameters
    ----------
    width, height, depth : float
        Box dimensions in metres (x, y, z extents from origin).
    fillet_radius : float
        Edge fillet radius in metres. 0 (default) = sharp corners.
    h_elem : float
        Target element edge length in metres.

    Returns
    -------
    mesh : Mesh
        vertices [V, 3] float64, triangles [T, 3] int32, group_tags [T] int32 = all 1.
    bc : BoundaryConditions
        sound_hard_groups = {1}, vibrating_groups = {} (no driver).
    """
    import gmsh

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("box_mesh")

    box_vol = gmsh.model.occ.addBox(0.0, 0.0, 0.0, width, height, depth)
    gmsh.model.occ.synchronize()

    if fillet_radius > 0.0:
        curves = gmsh.model.getEntities(1)
        edge_tags = [tag for _, tag in curves]
        new_vols = gmsh.model.occ.fillet([box_vol], edge_tags, [fillet_radius] * len(edge_tags))
        gmsh.model.occ.synchronize()
        box_vol = new_vols[0] if new_vols else box_vol

    bnd = gmsh.model.getBoundary([(3, box_vol)], oriented=False, combined=False)
    surf_tags = [abs(tag) for dim, tag in bnd if dim == 2]
    gmsh.model.addPhysicalGroup(2, surf_tags, tag=1)

    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h_elem)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h_elem / 4.0)
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(1)

    mesh = _extract_tagged_mesh({1: 1})
    gmsh.finalize()

    bc = BoundaryConditions(vibrating_groups={}, sound_hard_groups={1})
    return mesh, bc
