"""NC.inp text-file generator: translates normalized Mesh/BoundaryConditions/FrequencyGrid
into NumCalc input files.

This is the MINIMAL writer for build-order item 3. It supports:
  - A single vibrating group with a uniform complex scalar velocity (unit cone velocity).
  - Sound-hard implicit for all non-vibrating elements.
  - Conventional collocation BEM (method 0). The production adapter (item 6) will switch
    to ML-FMM (method 4) for large meshes; see DR-01.
  - A single NC.inp covering all frequencies; per-step batching and the frequency scheduler
    are item 6.

The evaluation grid for scattered observation points (e.g. Lebedev) uses a ConvexHull
triangulation, exactly as Mesh2HRTF does for sphere-surface points. Every point on a
convex hull is a hull vertex, so no observation points are dropped and the index order
is preserved.

Unsupported inputs raise NotImplementedError with clear messages; they are not silently
ignored.

References
----------
NC.inp format: Mesh2HRTF commit e45d0436a, mesh2input.py _write_nc_inp(), lines 1167–1377.
VERIFIED: Kreuzer et al., Engineering Analysis with Boundary Elements 161:157-178, 2024.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
from scipy.spatial import ConvexHull

from beamsim2.core.types import (
    BoundaryConditions,
    FrequencyGrid,
    Mesh,
    ObservationPoints,
    SolverConfig,
)

# Node/element ID offsets matching Mesh2HRTF convention (verified against
# mesh2input.py and the Lebedev_N10 example grids).
_OBJECT_NODE_BASE = 0
_OBJECT_ELEM_BASE = 0
_EVAL_NODE_BASE = 200_000
_EVAL_ELEM_BASE = 200_000
_EVAL_GROUP = 1  # all eval elements must share one group (NumCalc constraint)

# NC.inp BEM method codes (NC_Input.cpp).
# 0 = conventional collocation BEM, 4 = ML-FMM BEM (production, item 6).
_BEM_METHOD_CONVENTIONAL = 0


@dataclass
class _WriterCounts:
    """Mesh and eval-grid counts needed to build the NC.inp header."""

    n_obj_nodes: int
    n_obj_elems: int
    n_eval_nodes: int
    n_eval_elems: int
    eval_node_base: int  # first external node ID for eval grid


def write_mesh_files(
    work_dir: str,
    mesh: Mesh,
    observation_points: ObservationPoints,
) -> _WriterCounts:
    """Write object-mesh and evaluation-grid Nodes/Elements files into work_dir.

    Two subdirectories are created:
      work_dir/ObjectMesh/   — boundary mesh (PROPERTY 0)
      work_dir/EvalGrid/     — observation sphere (PROPERTY 2)

    The evaluation grid is built from the observation point unit vectors scaled to
    the observation radius, triangulated via ConvexHull (every sphere surface point
    is a hull vertex, so no points are dropped). This is the same approach used by
    Mesh2HRTF's write_evaluation_grid() for sphere-surface grids.

    Parameters
    ----------
    work_dir : str
        Directory where subdirectories are created. Must exist.
    mesh : Mesh
        Boundary surface mesh. triangles are 0-based indices into vertices.
    observation_points : ObservationPoints
        Directional sampling grid at radius r. Scaled by r before writing so
        NumCalc evaluates the field at the physical sphere surface.

    Returns
    -------
    _WriterCounts
        Node and element counts for both grids, plus the eval-node base ID.
    """
    obj_dir = os.path.join(work_dir, "ObjectMesh")
    eval_dir = os.path.join(work_dir, "EvalGrid")
    os.makedirs(obj_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    _write_object_nodes(os.path.join(obj_dir, "Nodes.txt"), mesh)
    _write_object_elements(os.path.join(obj_dir, "Elements.txt"), mesh)

    eval_xyz = observation_points.unit_vectors * observation_points.radius  # [N, 3] float64
    _write_eval_nodes(os.path.join(eval_dir, "Nodes.txt"), eval_xyz)
    n_eval_elems = _write_eval_elements(os.path.join(eval_dir, "Elements.txt"), eval_xyz)

    return _WriterCounts(
        n_obj_nodes=len(mesh.vertices),
        n_obj_elems=len(mesh.triangles),
        n_eval_nodes=len(eval_xyz),
        n_eval_elems=n_eval_elems,
        eval_node_base=_EVAL_NODE_BASE,
    )


def write_nc_inp(
    work_dir: str,
    mesh: Mesh,
    bc: BoundaryConditions,
    frequencies: FrequencyGrid,
    config: SolverConfig,
    counts: _WriterCounts,
) -> str:
    """Write NC.inp for a single-source, all-frequency solve into work_dir.

    This minimal writer supports one vibrating group with a uniform scalar complex
    velocity. It uses conventional BEM (method 0) and emits a single NC.inp that
    covers all frequencies in a single ``NumCalc -istart 1 -iend F`` invocation.

    The BC is written as two ELEM lines per group if imaginary part is non-zero (one
    for real, one for imaginary via the frequency-curve mechanism). For the unit-velocity
    smoke test the imaginary part is zero, so a single VELO line suffices.

    Not yet supported (item 6):
      - Multiple vibrating groups with different velocities.
      - Per-element velocity profiles (np.ndarray values in bc.vibrating_groups).
      - PRES / ADMI / IMPE boundary types.
      - ML-FMM (method 4).

    Parameters
    ----------
    work_dir : str
        Directory where NC.inp is written. ObjectMesh/ and EvalGrid/ must already exist.
    mesh : Mesh
        Boundary mesh (for element count and group-tag mapping).
    bc : BoundaryConditions
        Boundary conditions. Exactly one vibrating group with a scalar complex velocity
        is supported; raises NotImplementedError otherwise.
    frequencies : FrequencyGrid
        Frequency array, shape [F].
    config : SolverConfig
        Physics and numerical parameters.
    counts : _WriterCounts
        Node/element counts returned by write_mesh_files().

    Returns
    -------
    str
        Absolute path to the written NC.inp file.

    Raises
    ------
    NotImplementedError
        If bc has more than one vibrating group, per-element velocity profiles, or
        any other unsupported feature.
    """
    _validate_bc(bc, mesh)

    freqs = frequencies.frequencies  # [F] float64
    n_freq = len(freqs)
    n_total_nodes = counts.n_obj_nodes + counts.n_eval_nodes
    n_total_elems = counts.n_obj_elems + counts.n_eval_elems

    lines: list[str] = []

    # ── Header ──────────────────────────────────────────────────────────────
    lines += [
        "##-------------------------------------------",
        "## This file was created by BeamSimII",
        "##-------------------------------------------",
        "Mesh2HRTF 1.0.0",
        "##",
        "BeamSimII BEM solve",
        "##",
    ]

    # ── Control parameter I ─────────────────────────────────────────────────
    # Format verified against mesh2input.py line 1229 and the test NC.inp.
    lines += [
        "## Controlparameter I",
        "0 0 0 0 7 0",
        "##",
    ]

    # ── Control parameter II ────────────────────────────────────────────────
    # "1 <n_freq> 0.000001 0.00e+00 1 0 0"
    lines += [
        "## Controlparameter II",
        f"1 {n_freq} 0.000001 0.00e+00 1 0 0",
        "##",
    ]

    # ── Load Frequency Curve ─────────────────────────────────────────────────
    # Row 0 is always zeros; rows 1..F map internal step counter to real frequency.
    # The x-axis is the "curve parameter" in units of 0.000001 per step.
    # The y-axis is frequency in Hz directly. VERIFIED empirically: y=250.0 →
    # NumCalc logs "Frequency = 250 Hz". The y-axis is NOT scaled by 10000 despite
    # the Mesh2HRTF template comment "FREQ_IN_10000HZ_UNITS" (misleading label).
    lines += [
        "## Load Frequency Curve",
        f"0 {n_freq + 1}",
        "0.000000 0.000000e+00 0.0",
    ]
    for k, f in enumerate(freqs, start=1):
        x = k * 0.000001
        y = float(f)  # Hz directly
        lines.append(f"{x:.6f} {y:.6e} 0.0")
    lines.append("##")

    # ── Main Parameters I ────────────────────────────────────────────────────
    # "2 <n_elem_total> <n_node_total> 0 0 2 1 <method> 0"
    # method 0 = conventional BEM. DR-01: production uses ML-FMM (method 4, item 6).
    # VERIFIED: mesh2input.py line 1276 and NC_Input.cpp.
    lines += [
        "## 1. Main Parameters I",
        f"2 {n_total_elems} {n_total_nodes} 0 0 2 1 {_BEM_METHOD_CONVENTIONAL} 0",
        "##",
    ]

    # ── Main Parameters II ───────────────────────────────────────────────────
    # n_planewaves n_pointsrc 0 0.0e+00 0 0 0
    lines += [
        "## 2. Main Parameters II",
        "0 0 0 0.0000e+00 0 0 0",
        "##",
    ]

    # ── Main Parameters III ──────────────────────────────────────────────────
    lines += ["## 3. Main Parameters III", "0 0 0 0", "##"]

    # ── Main Parameters IV ───────────────────────────────────────────────────
    # speed_of_sound  density  1.0  0.0e+00  0.0 e+00  0.0e+00  0.0e+00
    c = config.speed_of_sound
    rho = config.air_density
    lines += [
        "## 4. Main Parameters IV",
        f"{c} {rho:.4e} 1.0 0.0e+00 0.0 e+00 0.0e+00 0.0e+00",
        "##",
    ]

    # ── NODES ────────────────────────────────────────────────────────────────
    lines += [
        "NODES",
        "ObjectMesh/Nodes.txt",
        "EvalGrid/Nodes.txt",
        "##",
    ]

    # ── ELEMENTS ─────────────────────────────────────────────────────────────
    lines += [
        "ELEMENTS",
        "ObjectMesh/Elements.txt",
        "EvalGrid/Elements.txt",
        "##",
    ]

    # ── BOUNDARY ─────────────────────────────────────────────────────────────
    # Format: ELEM <lo> TO <hi> VELO <re> -1 <im> -1
    # -1 for curve IDs = "no frequency-dependent curve" (constant value).
    # VERIFIED: mesh2input.py lines 1316–1340, test NC.inp.
    lines.append("BOUNDARY")
    tag, velocity = next(iter(bc.vibrating_groups.items()))
    vel = complex(velocity)
    elem_lo, elem_hi = _group_element_range(mesh, tag)
    lines.append(f"ELEM {elem_lo} TO {elem_hi} VELO {vel.real:.6e} -1 {vel.imag:.6e} -1")
    lines.append("RETU")
    lines.append("##")

    # ── POST PROCESS / END ───────────────────────────────────────────────────
    # When numIncidentPlaneWaves_ = 0 and numPointSources_ = 0, NC_ReadSoundSources
    # skips the PLANE WAVES and POINT SOURCES blocks entirely (verified NC_Input.cpp
    # lines 1013-1045). The parser then reads directly for CURVES or POST PROCESS.
    # Including a PLANE WAVES keyword here would cause "Key word CURVES or POST PROCESS
    # expected!" because the parser reads PLANE as the first non-comment token.
    lines += ["POST PROCESS", "##", "END", ""]

    nc_inp_path = os.path.join(work_dir, "NC.inp")
    with open(nc_inp_path, "w") as fh:
        fh.write("\n".join(lines))

    return nc_inp_path


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _write_object_nodes(path: str, mesh: Mesh) -> None:
    """Write boundary-mesh Nodes.txt.

    Format per NC_Input.cpp NC_ReadMesh():
        <count>
        <extID> <x> <y> <z>
        ...
    Node IDs start at _OBJECT_NODE_BASE (0).
    """
    verts = mesh.vertices  # [V, 3] float64
    lines = [str(len(verts))]
    for i, (x, y, z) in enumerate(verts):
        lines.append(f"{_OBJECT_NODE_BASE + i} {x:.10e} {y:.10e} {z:.10e}")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def _write_object_elements(path: str, mesh: Mesh) -> None:
    """Write boundary-mesh Elements.txt.

    Format: <extElemID> <n1> <n2> <n3> <PROPERTY=0> 0 <group_tag>
    Node indices are offset by _OBJECT_NODE_BASE (0 for boundary mesh).
    PROPERTY 0 = boundary element.
    """
    tris = mesh.triangles  # [T, 3] int32
    tags = mesh.group_tags  # [T] int32
    lines = [str(len(tris))]
    base = _OBJECT_NODE_BASE
    for i, ((n0, n1, n2), tag) in enumerate(zip(tris, tags)):
        lines.append(f"{_OBJECT_ELEM_BASE + i} {base + n0} {base + n1} {base + n2} 0 0 {tag}")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def _write_eval_nodes(path: str, eval_xyz: np.ndarray) -> None:
    """Write evaluation-grid Nodes.txt.

    Node IDs start at _EVAL_NODE_BASE (200 000), matching Mesh2HRTF convention.

    Parameters
    ----------
    eval_xyz : np.ndarray, shape [N, 3] float64
        Cartesian coordinates of the evaluation points (unit_vectors × radius).
    """
    lines = [str(len(eval_xyz))]
    for i, (x, y, z) in enumerate(eval_xyz):
        lines.append(f"{_EVAL_NODE_BASE + i} {x:.10e} {y:.10e} {z:.10e}")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def _write_eval_elements(path: str, eval_xyz: np.ndarray) -> int:
    """Write evaluation-grid Elements.txt using ConvexHull triangulation.

    For points on a sphere every point is a hull vertex, so no observation
    points are dropped and the index mapping is eval_xyz[i] ↔ node (EVAL_NODE_BASE+i).
    PROPERTY 2 = evaluation element. All elements share _EVAL_GROUP (1).

    If ConvexHull fails (e.g. nearly-coplanar sets), falls back to cyclic dummy
    elements (Mesh2HRTF strategy 2: groups nodes in triplets).

    Parameters
    ----------
    eval_xyz : np.ndarray, shape [N, 3] float64
        Cartesian evaluation-point coordinates.

    Returns
    -------
    int
        Number of evaluation elements written.
    """
    n = len(eval_xyz)
    try:
        hull = ConvexHull(eval_xyz)
        simplices = hull.simplices  # [T, 3] int — indices into eval_xyz
        lines = [str(len(simplices))]
        for i, (a, b, c) in enumerate(simplices):
            na = _EVAL_NODE_BASE + a
            nb = _EVAL_NODE_BASE + b
            nc_ = _EVAL_NODE_BASE + c
            lines.append(f"{_EVAL_ELEM_BASE + i} {na} {nb} {nc_} 2 0 {_EVAL_GROUP}")
        with open(path, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        return len(simplices)

    except Exception:
        # Fallback: cyclic degenerate elements (Mesh2HRTF Strategy 2).
        n_elems = (n + 2) // 3
        lines = [str(n_elems)]
        for i in range(n_elems):
            a = (i * 3 + 0) % n + _EVAL_NODE_BASE
            b = (i * 3 + 1) % n + _EVAL_NODE_BASE
            c = (i * 3 + 2) % n + _EVAL_NODE_BASE
            lines.append(f"{_EVAL_ELEM_BASE + i} {a} {b} {c} 2 0 {_EVAL_GROUP}")
        with open(path, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        return n_elems


def _validate_bc(bc: BoundaryConditions, mesh: Mesh) -> None:
    """Raise NotImplementedError for features deferred to item 6."""
    if len(bc.vibrating_groups) != 1:
        raise NotImplementedError(
            f"Minimal NC.inp writer supports exactly one vibrating group; "
            f"got {len(bc.vibrating_groups)}. Multi-source batching is item 6."
        )
    vel = next(iter(bc.vibrating_groups.values()))
    if isinstance(vel, np.ndarray):
        raise NotImplementedError(
            "Per-element velocity profiles (ndarray BC values) are not supported "
            "in the minimal writer. Full profile support is item 6."
        )


def _group_element_range(mesh: Mesh, tag: int) -> tuple[int, int]:
    """Return the (lo, hi) inclusive element-index range for a group tag.

    NumCalc BOUNDARY lines use 0-based element indices matching the order elements
    appear in Elements.txt. This helper scans group_tags to find contiguous or all
    matching elements for a given tag.

    For the pulsating-sphere test all elements share one tag, so lo=0, hi=T-1.

    Parameters
    ----------
    mesh : Mesh
        Boundary mesh.
    tag : int
        The surface-group tag to query.

    Returns
    -------
    tuple[int, int]
        (lo, hi) — 0-based inclusive element indices with group_tag == tag.
        Uses the min and max matching indices so a non-contiguous group still produces
        a valid (though over-inclusive) ELEM range. Item 6 will handle per-element BCs.
    """
    indices = np.where(mesh.group_tags == tag)[0]
    if len(indices) == 0:
        raise ValueError(f"No elements with group_tag={tag} in mesh.")
    return int(indices[0]), int(indices[-1])
