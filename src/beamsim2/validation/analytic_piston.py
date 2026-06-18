"""V-1: piston-in-a-large-baffle BEM vs. closed-form D(θ)=2·J₁(ka·sinθ)/(ka·sinθ) directivity cross-check (≤1 dB acceptance)."""

from __future__ import annotations

import numpy as np
from scipy.special import j1

from beamsim2.core.types import BoundaryConditions, Mesh, ObservationPoints

# ---------------------------------------------------------------------------
# Analytic piston directivity
# ---------------------------------------------------------------------------


def piston_directivity(theta_rad: np.ndarray, ka: float) -> np.ndarray:
    """Normalized far-field directivity of a rigid piston in an infinite baffle.

    D(θ) = 2·J₁(ka·sinθ) / (ka·sinθ),  with limit D → 1 as ka·sinθ → 0.

    VERIFIED: Kinsler, Frey, Coppens, Sanders,
    *Fundamentals of Acoustics*, 4th ed., eq. 7.4.14 (2000).

    D is real and equals 1 on-axis (θ = 0). It models the bending of the
    radiation pattern as the piston becomes large relative to the wavelength
    (ka > 1 → side-lobes appear).

    Parameters
    ----------
    theta_rad : np.ndarray
        Colatitude angle(s) from the piston axis (+z) in radians, shape [N].
    ka : float
        Wave number × piston radius (dimensionless frequency).

    Returns
    -------
    np.ndarray, shape [N], float64
        Normalized directivity D(θ), range [−0.4, 1].
    """
    theta_rad = np.asarray(theta_rad, dtype=np.float64)
    x = ka * np.sin(theta_rad)  # [N] — argument of the J₁ function
    # Avoid 0/0: for |x| < threshold use the Taylor limit D → 1 − (ka·sinθ)²/8 + …
    safe = np.abs(x) > 1e-12
    result = np.where(safe, 2.0 * j1(x) / x, 1.0)
    return result.astype(np.float64)


# ---------------------------------------------------------------------------
# Piston mesh builder (gmsh)
# ---------------------------------------------------------------------------


def make_piston_mesh(
    a_piston: float = 0.05,
    baffle_half_width: float = 0.40,
    h_elem: float = 0.013,
) -> tuple[Mesh, BoundaryConditions]:
    """Build a flat piston-in-baffle BEM mesh using gmsh.

    Geometry: a square baffle of half-width W centred at the origin in the
    z = 0 plane, with a circular hole of radius a.  Inside the hole is the
    piston disk (group 1, vibrating).  The surrounding ring (group 2) is
    sound-hard (rigid baffle).

    All surface normals are oriented in the +z direction (toward the
    radiating half-space).  NumCalc requires outward normals to get the
    sign of the VELO boundary condition right.

    Parameters
    ----------
    a_piston : float
        Piston radius in metres (default 0.05 m).
    baffle_half_width : float
        Half-width of the square baffle in metres (default 0.40 m).
    h_elem : float
        Target element edge length in metres (default 0.013 m).

    Returns
    -------
    mesh : Mesh
        vertices [V, 3] float64 (z ≈ 0),
        triangles [T, 3] int32,
        group_tags [T] int32  (1 = piston, 2 = baffle ring).
    bc : BoundaryConditions
        group 1 vibrating at VELO = 1+0j; group 2 implicitly sound-hard.
    """
    import gmsh

    a = a_piston
    W = baffle_half_width

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)  # suppress console output
    gmsh.model.add("piston_mesh")

    # Create piston disk and baffle square as OCC surfaces
    piston_s = gmsh.model.occ.addDisk(0.0, 0.0, 0.0, a, a)
    baffle_s = gmsh.model.occ.addRectangle(-W, -W, 0.0, 2 * W, 2 * W)

    # Fragment: splits surfaces at their shared circular boundary so the mesh
    # nodes are conforming at the piston/baffle interface.
    _, emap = gmsh.model.occ.fragment([(2, baffle_s)], [(2, piston_s)])
    # emap[0]: new surface(s) derived from the baffle rectangle (now a ring)
    # emap[1]: new surface(s) derived from the piston disk
    gmsh.model.occ.synchronize()

    ring_tags = [tag for _, tag in emap[0]]
    disk_tags = [tag for _, tag in emap[1]]

    gmsh.model.addPhysicalGroup(2, disk_tags, tag=1)  # piston  → group 1
    gmsh.model.addPhysicalGroup(2, ring_tags, tag=2)  # baffle  → group 2

    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h_elem)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h_elem / 4.0)
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(1)  # linear (first-order) triangles

    # ── Extract nodes ─────────────────────────────────────────────────────
    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    coords = np.array(coords, dtype=np.float64).reshape(-1, 3)  # [V, 3]
    # Build 1-based node tag → 0-based local index
    node_tag_to_idx: dict[int, int] = {int(t): i for i, t in enumerate(node_tags)}
    vertices = coords  # [V, 3]

    # ── Extract triangular elements per physical group ────────────────────
    all_triangles: list[list[int]] = []
    all_group_tags: list[int] = []

    for pg_tag in (1, 2):
        ents = gmsh.model.getEntitiesForPhysicalGroup(2, pg_tag)
        for ent in ents:
            etype_list, _, enode_tags_list = gmsh.model.mesh.getElements(dim=2, tag=ent)
            for etype, enodes in zip(etype_list, enode_tags_list):
                if etype == 2:  # first-order triangle (gmsh element type 2)
                    tris = np.array(enodes, dtype=np.int64).reshape(-1, 3)
                    for tri in tris:
                        local_tri = [node_tag_to_idx[int(n)] for n in tri]
                        all_triangles.append(local_tri)
                        all_group_tags.append(pg_tag)

    gmsh.finalize()

    triangles = np.array(all_triangles, dtype=np.int32)  # [T, 3]
    group_tags = np.array(all_group_tags, dtype=np.int32)  # [T]

    # ── Ensure all normals point +z (toward the radiating half-space) ─────
    v0 = vertices[triangles[:, 0]]  # [T, 3]
    v1 = vertices[triangles[:, 1]]  # [T, 3]
    v2 = vertices[triangles[:, 2]]  # [T, 3]
    normal_z = np.cross(v1 - v0, v2 - v0)[:, 2]  # [T] — z-component of normal
    flip = normal_z < 0
    if flip.any():
        triangles[flip] = triangles[flip][:, [0, 2, 1]]

    mesh = Mesh(
        vertices=vertices.astype(np.float64),
        triangles=triangles,
        group_tags=group_tags,
    )
    bc = BoundaryConditions(vibrating_groups={1: complex(1.0, 0.0)})
    return mesh, bc


# ---------------------------------------------------------------------------
# Error computation for V-1
# ---------------------------------------------------------------------------


def piston_benchmark_errors(
    H_bem: np.ndarray,
    frequencies: np.ndarray,
    obs_points: ObservationPoints,
    a_piston: float,
    c: float,
) -> dict:
    """Compare BEM directivity shape to the analytic piston formula.

    For each frequency, the BEM pressure field is normalised by its on-axis
    value (direction closest to θ = 0), giving a dimensionless shape. This
    is compared to ``piston_directivity(θ, ka)`` at each forward-hemisphere
    direction (z ≥ 0). The error is the dB difference of the magnitudes.

    Pass criterion V-1: mean |error| ≤ 1 dB at every frequency.

    Parameters
    ----------
    H_bem : np.ndarray, shape [F, N], complex128
        BEM complex pressure at the observation points.
    frequencies : np.ndarray, shape [F], float64
        Frequencies in Hz.
    obs_points : ObservationPoints
        Observation sphere. The (0,0,1) direction must be present (Lebedev-26
        and larger grids always include the axis points).
    a_piston : float
        Piston radius in metres.
    c : float
        Speed of sound in m/s.

    Returns
    -------
    dict with keys:
        mag_error_db : np.ndarray, shape [F, N_fwd], float64
            Signed dB error at each forward-hemisphere direction and frequency.
        mean_mag_db : np.ndarray, shape [F], float64
            Mean of |mag_error_db| per frequency.
        max_mag_db : np.ndarray, shape [F], float64
            Max of |mag_error_db| per frequency.
        passed : bool
            True if all mean_mag_db ≤ 1.0 dB.
    """
    uvec = obs_points.unit_vectors  # [N, 3]

    # Forward hemisphere: z ≥ 0
    fwd_mask = uvec[:, 2] >= 0  # [N] bool
    uvec_fwd = uvec[fwd_mask]  # [N_fwd, 3]

    # On-axis index: direction with the largest z-component (closest to θ = 0)
    on_axis_idx = int(np.argmax(uvec[:, 2]))  # global index in full sphere

    # Colatitude of each forward-hemisphere direction
    theta_fwd = np.arccos(np.clip(uvec_fwd[:, 2], -1.0, 1.0))  # [N_fwd] radians

    mag_error_all: list[np.ndarray] = []

    for fi, f in enumerate(frequencies):
        ka = 2.0 * np.pi * f * a_piston / c
        p_on_axis = H_bem[fi, on_axis_idx]  # complex scalar

        # Normalised magnitude at forward-hemisphere directions
        p_fwd = H_bem[fi, fwd_mask]  # [N_fwd] complex
        mag_norm = np.abs(p_fwd) / np.abs(p_on_axis)  # [N_fwd] — 1.0 on axis

        # Analytic directivity at same angles
        D = piston_directivity(theta_fwd, ka)  # [N_fwd] float64

        # dB error — use absolute value of D to handle back-lobe nulls
        mag_error = 20.0 * np.log10(mag_norm / np.maximum(np.abs(D), 1e-6))  # [N_fwd]
        mag_error_all.append(mag_error)

    mag_error_db = np.array(mag_error_all, dtype=np.float64)  # [F, N_fwd]
    mean_mag_db = np.mean(np.abs(mag_error_db), axis=1)  # [F]
    max_mag_db = np.max(np.abs(mag_error_db), axis=1)  # [F]
    passed = bool(np.all(mean_mag_db <= 1.0))

    return {
        "mag_error_db": mag_error_db,
        "mean_mag_db": mean_mag_db,
        "max_mag_db": max_mag_db,
        "passed": passed,
    }
