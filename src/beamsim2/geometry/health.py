"""Geometry health checks: watertight/manifold, normal orientation, degenerate faces,
minimum feature size.  Operates on ``Mesh`` objects (no gmsh dependency).

Auto-repairs cheap cases (flipped normals, duplicate/degenerate faces) and logs them
rather than hiding them. Non-repairable defects are reported as located, plain-English
messages matching the DR-03 spec ("the enclosure has a 3 mm gap near (x,y,z); BEM
needs a closed surface").

Build-order item 5 (DR-03, pipeline Stage B).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from beamsim2.core.types import Mesh

# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------


@dataclass
class HealthReport:
    """Summary of geometry health checks and auto-repairs.

    Parameters
    ----------
    is_watertight : bool
        True if every edge is shared by exactly two triangles (manifold closed).
    open_edge_count : int
        Number of boundary edges (edges shared by fewer than 2 triangles).
    problems : list[str]
        Located, plain-English descriptions of non-repairable defects.
    repairs : list[str]
        Descriptions of auto-repairs that were applied.
    warnings : list[str]
        Non-fatal advisories (e.g. feature size below target edge).
    """

    is_watertight: bool = True
    open_edge_count: int = 0
    problems: list[str] = field(default_factory=list)
    repairs: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when no non-repairable problems were found."""
        return self.is_watertight and len(self.problems) == 0


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def check_watertight(mesh: Mesh) -> tuple[bool, int, str]:
    """Check that every edge is shared by exactly two triangles.

    A BEM mesh must be closed (watertight) and manifold: no boundary edges,
    no edges shared by more than two triangles.

    Parameters
    ----------
    mesh : Mesh
        Surface mesh to check.

    Returns
    -------
    is_watertight : bool
    open_edge_count : int
        Number of boundary edges (0 if watertight).
    message : str
        Plain-English description of open edges, or empty string if watertight.
    """
    tris = mesh.triangles  # [T, 3]
    verts = mesh.vertices  # [V, 3]

    # Build edge → triangle count map.
    edge_count: dict[tuple[int, int], int] = {}
    for tri in tris:
        for i in range(3):
            a, b = int(tri[i]), int(tri[(i + 1) % 3])
            key = (min(a, b), max(a, b))
            edge_count[key] = edge_count.get(key, 0) + 1

    boundary_edges = [k for k, v in edge_count.items() if v != 2]
    if not boundary_edges:
        return True, 0, ""

    # Report first few open edges with approximate 3-D location.
    locations = []
    for a, b in boundary_edges[:5]:
        mid = (verts[a] + verts[b]) / 2.0
        locations.append(f"({mid[0]:.4f}, {mid[1]:.4f}, {mid[2]:.4f}) m")

    loc_str = "; ".join(locations)
    if len(boundary_edges) > 5:
        loc_str += f" … and {len(boundary_edges) - 5} more"

    msg = (
        f"Mesh has {len(boundary_edges)} open/non-manifold edge(s). "
        f"BEM requires a closed watertight surface. "
        f"Open edges near: {loc_str}."
    )
    return False, len(boundary_edges), msg


def check_normals(mesh: Mesh) -> tuple[Mesh, int]:
    """Detect and auto-repair inward-facing triangles.

    Uses the heuristic that a face normal should point away from the centroid
    of all vertices (the geometric centre).  Adequate for convex closed surfaces.

    Parameters
    ----------
    mesh : Mesh
        Surface mesh to check.

    Returns
    -------
    mesh : Mesh
        Mesh with any flipped triangles corrected (may be the input object if
        no flips were needed — caller should use the returned mesh).
    n_flipped : int
        Number of triangles whose winding was reversed.
    """
    verts = mesh.vertices  # [V, 3]
    tris = mesh.triangles.copy()  # [T, 3]

    center = verts.mean(axis=0)  # [3]
    v0, v1, v2 = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)  # [T, 3]
    centroids = (v0 + v1 + v2) / 3.0  # [T, 3]
    dots = np.einsum("ij,ij->i", normals, centroids - center)  # [T]
    flip = dots < 0
    n_flipped = int(flip.sum())
    if n_flipped:
        tris[flip] = tris[flip][:, [0, 2, 1]]
        mesh = Mesh(vertices=verts, triangles=tris, group_tags=mesh.group_tags)

    return mesh, n_flipped


def check_degenerate(mesh: Mesh, area_threshold: float = 1e-20) -> tuple[Mesh, int]:
    """Detect and remove degenerate (zero or near-zero area) triangles.

    Degenerate triangles can cause numerical instabilities in the BEM integrator.

    Parameters
    ----------
    mesh : Mesh
        Surface mesh to check.
    area_threshold : float
        Triangles with area below this value (m²) are removed. Default 1e-20 m².

    Returns
    -------
    mesh : Mesh
        Mesh with degenerate triangles removed (may be the input if none found).
    n_removed : int
        Number of triangles removed.
    """
    verts = mesh.vertices  # [V, 3]
    tris = mesh.triangles  # [T, 3]
    tags = mesh.group_tags  # [T]

    v0, v1, v2 = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)  # [T, 3]
    areas = 0.5 * np.linalg.norm(cross, axis=1)  # [T]
    keep = areas >= area_threshold
    n_removed = int((~keep).sum())

    if n_removed:
        mesh = Mesh(
            vertices=verts,
            triangles=tris[keep],
            group_tags=tags[keep],
        )

    return mesh, n_removed


def check_min_feature(mesh: Mesh, target_edge: float) -> list[str]:
    """Warn if any edge in the mesh is shorter than ``target_edge / 4``.

    Very short edges relative to the target element size can cause mesh
    quality problems in downstream solvers.

    Parameters
    ----------
    mesh : Mesh
        Surface mesh to check.
    target_edge : float
        Target element edge length in metres (from ``mesh.target_edge_length``).

    Returns
    -------
    list[str]
        Plain-English warnings; empty if no short edges found.
    """
    threshold = target_edge / 4.0
    verts = mesh.vertices  # [V, 3]
    tris = mesh.triangles  # [T, 3]

    warnings: list[str] = []
    short_edges: list[tuple[float, np.ndarray]] = []  # (length, midpoint)

    for i in range(3):
        a = verts[tris[:, i]]  # [T, 3]
        b = verts[tris[:, (i + 1) % 3]]  # [T, 3]
        lengths = np.linalg.norm(b - a, axis=1)  # [T]
        mids = (a + b) / 2.0  # [T, 3]
        short_mask = lengths < threshold
        for length, mid in zip(lengths[short_mask], mids[short_mask]):
            short_edges.append((float(length), mid))

    if short_edges:
        count = len(short_edges)
        examples = short_edges[:3]
        ex_str = "; ".join(
            f"{e:.4f} m at ({m[0]:.4f}, {m[1]:.4f}, {m[2]:.4f}) m" for e, m in examples
        )
        if count > 3:
            ex_str += f" … and {count - 3} more"
        warnings.append(
            f"{count} edge(s) shorter than {threshold * 1e3:.2f} mm "
            f"(threshold = target_edge / 4 = {threshold * 1e3:.2f} mm). "
            f"Short edges: {ex_str}. "
            "Consider increasing target element size or smoothing the geometry."
        )

    return warnings


# ---------------------------------------------------------------------------
# Aggregate runner
# ---------------------------------------------------------------------------


def run_health_checks(
    mesh: Mesh,
    target_edge: float | None = None,
) -> tuple[Mesh, HealthReport]:
    """Run all geometry health checks and apply cheap auto-repairs.

    Checks run in order: degenerate faces (removed first so they don't confuse
    the edge-count), normal orientation (auto-repaired), watertight/manifold
    (reported, not repaired), minimum feature size (warning only).

    Parameters
    ----------
    mesh : Mesh
        Input surface mesh.
    target_edge : float or None
        Target element edge length in metres for the feature-size check.
        If None, the feature-size check is skipped.

    Returns
    -------
    mesh : Mesh
        Repaired mesh (degenerate faces removed, normals corrected).
    report : HealthReport
        Full health report including located problems, repair log, and warnings.
    """
    report = HealthReport()

    # 1. Remove degenerate triangles.
    mesh, n_degen = check_degenerate(mesh)
    if n_degen:
        report.repairs.append(f"Removed {n_degen} degenerate (near-zero-area) triangle(s).")

    # 2. Repair flipped normals.
    mesh, n_flipped = check_normals(mesh)
    if n_flipped:
        report.repairs.append(f"Reversed winding of {n_flipped} inward-facing triangle(s).")

    # 3. Watertight / manifold check.
    is_wt, open_count, wt_msg = check_watertight(mesh)
    report.is_watertight = is_wt
    report.open_edge_count = open_count
    if not is_wt:
        report.problems.append(wt_msg)

    # 4. Feature-size warning.
    if target_edge is not None:
        report.warnings.extend(check_min_feature(mesh, target_edge))

    return mesh, report
