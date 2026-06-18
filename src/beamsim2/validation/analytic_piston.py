"""V-1: spherical-cap piston on a rigid sphere — BEM directivity vs. the exact
spherical-cap closed form (Legendre/Hankel series); ≤1 dB acceptance.

The original V-1 used a flat coplanar piston-in-baffle (``make_piston_mesh``)
compared to the infinite-baffle formula ``2·J₁(ka·sinθ)/(ka·sinθ)``. That
geometry crashes NumCalc: for two coplanar elements the perpendicular offset
ε = 0, so the near-field subelement subdivision (``NC_GenerateSubelements``)
never satisfies its stopping ratio and overruns the compile-time ``MSBE`` cap.
All real loudspeaker meshes are closed, curved surfaces (ε > 0), so this only
ever bit the flat validation case. V-1 is therefore validated on a curved
geometry — a polar cap on a rigid sphere — against its own exact solution
(``spherical_cap_directivity``). The flat ``piston_directivity`` /
``make_piston_mesh`` are retained for reference and as the small-cap limit
cross-check.
"""

from __future__ import annotations

import numpy as np
from scipy.special import eval_legendre, j1, spherical_jn, spherical_yn

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


def spherical_cap_directivity(
    theta_rad: np.ndarray,
    ka_sphere: float,
    cap_half_angle_rad: float,
    n_terms: int | None = None,
) -> np.ndarray:
    """Exact normalized far-field magnitude directivity of a vibrating polar
    cap on an otherwise-rigid sphere.

    A sphere of radius a has a cap of half-angle α (measured from the +z axis)
    moving radially at uniform velocity U₀; the rest of the surface is rigid.
    The exterior field is the spherical-harmonic (Legendre) series

        p(r,θ) = Σ_n a_n · h_n⁽¹⁾(kr) · P_n(cosθ),
        a_n    = i·ρc · U_n / h_n⁽¹⁾′(ka),

    where the cap's uniform-velocity Legendre coefficients are

        U_0 = (U₀/2)(1 − cosα),
        U_n = (U₀/2)[P_{n−1}(cosα) − P_{n+1}(cosα)]   (n ≥ 1),

    obtained from ∫_{cosα}^{1} P_n(x) dx = [P_{n−1}(cosα) − P_{n+1}(cosα)]/(2n+1).
    The far-field angular factor is g(θ) = Σ_n a_n (−i)^{n+1} P_n(cosθ); this
    function returns |g(θ)| / |g(0)| (unity on axis). The common i·ρc cancels.

    Convention: NumCalc engineering convention exp(−jωt), outgoing wave
    ∝ exp(+jkr), so h_n⁽¹⁾ = j_n + i·y_n is the correct (outgoing) radial
    solution. Using h_n⁽²⁾ would conjugate the phase and is wrong for NumCalc.

    VERIFIED derivation: Morse & Ingard, *Theoretical Acoustics* §7.2
    (radiation from a sphere with prescribed surface velocity), McGraw-Hill
    1968; Williams, *Fourier Acoustics* ch. 6, 1999. HEURISTIC cross-check:
    in the small-cap, locally-flat limit (α → 0, ka·sinα fixed) this reduces
    to the flat-piston directivity ``piston_directivity`` with ka_cap = ka·sinα
    in the forward hemisphere — used only to sanity the series, not in the
    pass criterion.

    In acoustics terms: the cap is a "porthole" radiator set into a rigid ball.
    At low ka it is nearly omnidirectional; as ka·sinα grows the cap beams
    forward just like a piston, but the closed sphere (not an infinite baffle)
    sets the rear field — which is exactly the curved-surface case BEM must get
    right.

    Parameters
    ----------
    theta_rad : np.ndarray
        Colatitude(s) from the cap axis (+z) in radians, shape [N].
    ka_sphere : float
        Wave number × sphere radius, k·a (dimensionless).
    cap_half_angle_rad : float
        Cap half-angle α in radians.
    n_terms : int or None
        Number of Legendre/Hankel terms. None → int(ka_sphere) + 20, which
        over-resolves the series (the coefficients decay super-exponentially
        for n ≫ ka because h_n⁽¹⁾′ grows like (2n−1)!!/(ka)^{n+1}).

    Returns
    -------
    np.ndarray, shape [N], float64
        Normalized magnitude directivity |D(θ)|, equal to 1 at θ = 0.
    """
    theta_rad = np.asarray(theta_rad, dtype=np.float64)
    if n_terms is None:
        n_terms = int(ka_sphere) + 20

    cos_alpha = float(np.cos(cap_half_angle_rad))
    cos_theta = np.cos(theta_rad)  # [N]

    g_theta = np.zeros_like(theta_rad, dtype=np.complex128)  # [N] — Σ a_n(−i)^{n+1}P_n(cosθ)
    g_axis = 0.0 + 0.0j  # value at θ = 0, where P_n(1) = 1

    # Suppress the harmless overflow in y_n′ at large n: it sends |h_n⁽¹⁾′| → ∞,
    # so the coefficient → 0 (a high mode the cap cannot drive), which is correct.
    with np.errstate(over="ignore", invalid="ignore"):
        for n in range(n_terms + 1):
            if n == 0:
                u_n = 0.5 * (1.0 - cos_alpha)
            else:
                u_n = 0.5 * (eval_legendre(n - 1, cos_alpha) - eval_legendre(n + 1, cos_alpha))

            # h_n⁽¹⁾′(ka) = j_n′(ka) + i·y_n′(ka)  (spherical-Hankel derivative)
            dh = spherical_jn(n, ka_sphere, derivative=True) + 1j * spherical_yn(
                n, ka_sphere, derivative=True
            )

            coeff = u_n * ((-1j) ** (n + 1)) / dh  # a_n(−i)^{n+1}, up to the iρc constant
            if not np.isfinite(coeff):
                continue

            g_theta = g_theta + coeff * eval_legendre(n, cos_theta)  # [N]
            g_axis = g_axis + coeff  # P_n(1) = 1

    directivity = np.abs(g_theta) / np.abs(g_axis)  # [N]
    return directivity.astype(np.float64)


# ---------------------------------------------------------------------------
# Piston mesh builder (gmsh)
# ---------------------------------------------------------------------------


def make_piston_mesh(
    a_piston: float = 0.05,
    baffle_half_width: float = 0.40,
    h_elem: float = 0.013,
    h_baffle: float | None = None,
) -> tuple[Mesh, BoundaryConditions]:
    """Build a flat piston-in-baffle BEM mesh using gmsh.

    .. deprecated::
        This flat coplanar geometry **crashes NumCalc** and is no longer used
        by V-1 (see ``make_spherical_cap_piston_mesh`` instead). It is kept
        only to document the failure: for two elements in the same z = 0 plane
        the perpendicular offset ε = 0, so ``NC_GenerateSubelements`` never
        satisfies its ``distance/√area ≥ 1.3`` stopping ratio and overruns the
        compile-time element-neighbourhood cap. (The cap is ``#define MSBE 220``
        in ``NC_ConstantsVariables.h``; the runtime error string still prints a
        stale literal ``110`` that does not reflect the compiled value.) Real
        loudspeaker meshes are closed, curved surfaces (ε > 0), so this never
        bites in production.

    Geometry: a square baffle of half-width W centred at the origin in the
    z = 0 plane, with a circular hole of radius a.  Inside the hole is the
    piston disk (group 1, vibrating).  The surrounding ring (group 2) is
    sound-hard (rigid baffle).

    All surface normals are oriented in the +z direction (toward the
    radiating half-space).  NumCalc requires outward normals to get the
    sign of the VELO boundary condition right.

    The mesh is graded: the piston disk uses ``h_elem`` throughout; the
    baffle ring uses ``h_baffle`` (default 4 × h_elem) at the outer rim,
    with a smooth transition near the piston edge.

    Parameters
    ----------
    a_piston : float
        Piston radius in metres (default 0.05 m).
    baffle_half_width : float
        Half-width of the square baffle in metres (default 0.40 m).
    h_elem : float
        Target element edge length on the piston in metres (default 0.013 m).
    h_baffle : float or None
        Target element edge length on the outer baffle rim in metres.
        None (default) → 4 × h_elem.

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
    h_coarse = 4.0 * h_elem if h_baffle is None else h_baffle

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

    # Graded mesh: h_elem at the piston boundary circle, coarsening radially
    # outward on the baffle ring and toward the piston centre.
    # NumCalc's compile-time MSBE=220 caps elements per integration
    # neighbourhood; a uniform h_elem mesh on a 0.8×0.8 m baffle generates
    # ~9000 elements. Referencing distance from the piston boundary CURVE keeps
    # the local count low, but the flat coplanar layout crashes regardless
    # (ε = 0 in NC_GenerateSubelements) — see the function docstring.
    piston_bdry = gmsh.model.getBoundary(
        [(2, t) for t in disk_tags], oriented=False, combined=False
    )
    piston_curve_tags = [abs(t) for _, t in piston_bdry]

    f_dist = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(f_dist, "CurvesList", piston_curve_tags)
    gmsh.model.mesh.field.setNumber(f_dist, "Sampling", 200)

    f_thresh = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(f_thresh, "InField", f_dist)
    gmsh.model.mesh.field.setNumber(f_thresh, "SizeMin", h_elem)
    gmsh.model.mesh.field.setNumber(f_thresh, "SizeMax", h_coarse)
    gmsh.model.mesh.field.setNumber(f_thresh, "DistMin", 0.0)
    gmsh.model.mesh.field.setNumber(f_thresh, "DistMax", 3.0 * h_coarse)
    gmsh.model.mesh.field.setAsBackgroundMesh(f_thresh)

    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h_coarse)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h_elem / 4.0)
    # Disable boundary-size propagation; the background field sets all interior
    # sizes directly. Without this flag h_elem on the piston boundary circle
    # propagates to the entire piston interior and keeps the element count high.
    gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 0)
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


def make_spherical_cap_piston_mesh(
    sphere_radius: float = 0.10,
    cap_half_angle_deg: float = 45.0,
    subdivisions: int = 3,
) -> tuple[Mesh, BoundaryConditions]:
    """Build a rigid-sphere BEM mesh with a vibrating polar cap (the V-1 source).

    A spherical cap of half-angle α about the +z axis is tagged group 1 and
    vibrates with uniform radial velocity (VELO = 1+0j); the remainder of the
    sphere is group 2 (sound-hard).  Because the surface is curved, the
    perpendicular offset ε between any pair of elements is > 0, so NumCalc's
    near-field subelement subdivision (``NC_GenerateSubelements``) terminates
    normally — unlike ``make_piston_mesh``, whose coplanar elements (ε = 0)
    overrun the ``MSBE`` cap and crash.

    The geometry reuses the icosphere builder
    (``sphere_benchmark.make_pulsating_sphere_mesh``); each triangle is assigned
    to the cap by the colatitude of its centroid (θ_c ≤ α).  On a sphere the
    outward normal is radial, so a uniform normal velocity on the cap is exactly
    the uniform-radial-velocity boundary condition assumed by the closed form
    ``spherical_cap_directivity`` — the two describe the same physical source.

    Parameters
    ----------
    sphere_radius : float
        Sphere radius a in metres (default 0.10 m).
    cap_half_angle_deg : float
        Cap half-angle α in degrees, measured from +z (default 45°).
    subdivisions : int
        Icosphere subdivision level (default 3 → 1280 triangles, edge ≈ a/8).
        On a 0.10 m sphere this gives ≳ 8 elements/wavelength up to ~2 kHz and
        ~190 elements inside a 45° cap, so the staircased cap edge is fine.

    Returns
    -------
    mesh : Mesh
        vertices [V, 3] float64, triangles [T, 3] int32,
        group_tags [T] int32 (1 = vibrating cap, 2 = rigid remainder).
    bc : BoundaryConditions
        group 1 vibrating with VELO = 1+0j; group 2 implicitly sound-hard.
    """
    from beamsim2.validation.sphere_benchmark import make_pulsating_sphere_mesh

    # Reuse the icosphere geometry (outward winding already enforced); we only
    # re-tag which elements belong to the cap.
    base_mesh, _ = make_pulsating_sphere_mesh(radius=sphere_radius, subdivisions=subdivisions)
    verts = base_mesh.vertices  # [V, 3] float64
    tris = base_mesh.triangles  # [T, 3] int32

    centroids = verts[tris].mean(axis=1)  # [T, 3] — triangle centroids
    r_c = np.linalg.norm(centroids, axis=1)  # [T]
    cos_theta_c = centroids[:, 2] / r_c  # [T] — cos of colatitude (z / |centroid|)
    theta_c = np.arccos(np.clip(cos_theta_c, -1.0, 1.0))  # [T] radians

    alpha = np.deg2rad(cap_half_angle_deg)
    is_cap = theta_c <= alpha  # [T] bool — cap membership by centroid colatitude

    # The minimal NC.inp writer imposes the cap velocity as a single contiguous
    # "ELEM lo TO hi" range (ncinp_writer._group_element_range takes min/max
    # indices), so the vibrating elements MUST form a contiguous block. The
    # icosphere orders elements by subdivision, scattering the cap through the
    # array; reorder so every cap element precedes every rigid element. Without
    # this the BC leaks onto rigid elements between the first and last cap
    # element and the solve models a far larger vibrating region.
    order = np.argsort(~is_cap, kind="stable")  # cap (False→0) first, rigid after
    tris = tris[order]  # [T, 3]
    group_tags = np.where(is_cap[order], 1, 2).astype(np.int32)  # [T] — 1s then 2s

    mesh = Mesh(
        vertices=verts.astype(np.float64),
        triangles=tris.astype(np.int32),
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


def cap_benchmark_errors(
    H_bem: np.ndarray,
    frequencies: np.ndarray,
    obs_points: ObservationPoints,
    sphere_radius: float,
    cap_half_angle_deg: float,
    c: float,
) -> dict:
    """Compare BEM directivity of a vibrating spherical cap to its exact
    closed form (the V-1 acceptance computation).

    Mirrors ``piston_benchmark_errors`` but references the **exact**
    spherical-cap solution (``spherical_cap_directivity``) for the *same*
    geometry NumCalc actually solves, so the only expected disagreement is BEM
    discretization plus a little cap-edge staircasing — there is no
    geometry/baffle approximation as there would be against the flat-piston
    formula. For each frequency the BEM field is normalized by its on-axis (+z)
    value and compared, over the forward hemisphere (z ≥ 0), to |D(θ)| at
    ka_sphere = 2π·f·a/c.

    Pass criterion V-1: mean |error| ≤ 1 dB at every frequency.

    Parameters
    ----------
    H_bem : np.ndarray, shape [F, N], complex128
        BEM complex pressure at the observation points.
    frequencies : np.ndarray, shape [F], float64
        Frequencies in Hz.
    obs_points : ObservationPoints
        Observation sphere (unit vectors used for direction/colatitude).
    sphere_radius : float
        BEM sphere radius a in metres (drives ka_sphere).
    cap_half_angle_deg : float
        Cap half-angle α in degrees (must match the mesh's cap).
    c : float
        Speed of sound in m/s.

    Returns
    -------
    dict with keys
        mag_error_db : np.ndarray, shape [F, N_fwd], float64
            Signed dB error per frequency and forward-hemisphere direction.
        mean_mag_db : np.ndarray, shape [F], float64
            Mean |mag_error_db| per frequency.
        max_mag_db : np.ndarray, shape [F], float64
            Max |mag_error_db| per frequency.
        passed : bool
            True iff all mean_mag_db ≤ 1.0 dB.
    """
    uvec = obs_points.unit_vectors  # [N, 3]

    # Forward hemisphere: z ≥ 0
    fwd_mask = uvec[:, 2] >= 0  # [N] bool
    uvec_fwd = uvec[fwd_mask]  # [N_fwd, 3]

    # On-axis index: direction with the largest z-component (closest to θ = 0)
    on_axis_idx = int(np.argmax(uvec[:, 2]))  # global index in the full sphere

    # Colatitude of each forward-hemisphere direction
    theta_fwd = np.arccos(np.clip(uvec_fwd[:, 2], -1.0, 1.0))  # [N_fwd] radians
    alpha = np.deg2rad(cap_half_angle_deg)

    mag_error_all: list[np.ndarray] = []

    for fi, f in enumerate(frequencies):
        ka_sphere = 2.0 * np.pi * f * sphere_radius / c

        p_on_axis = H_bem[fi, on_axis_idx]  # complex scalar
        p_fwd = H_bem[fi, fwd_mask]  # [N_fwd] complex
        mag_norm = np.abs(p_fwd) / np.abs(p_on_axis)  # [N_fwd], 1.0 on axis

        # Exact cap directivity at the same angles
        D = spherical_cap_directivity(theta_fwd, ka_sphere, alpha)  # [N_fwd]

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
