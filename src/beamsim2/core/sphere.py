"""Sphere sampling grids with integration weights for directional BEM work.

The primary scheme is Lebedev–Laikov quadrature, which provides exact quadrature
weights for integrating polynomials up to a given algebraic degree over the sphere.
In loudspeaker terms: you can compute radiated power, directivity index, and
Phase-2 covariance matrices as simple weighted sums Σ wᵢ f(directionᵢ) with no
ad-hoc angular-density correction.

The near-uniform **icosphere** grid (geodesic subdivision) is also implemented for the
hundreds-to-thousands of directions Phase-2 beam design/audit needs; it carries
spherical-area quadrature weights (sum = 4π) rather than exact polynomial weights. The
remaining alternatives (Fliege–Maier, t-design) raise NotImplementedError to prevent
silent wrong results. ``make_observation_grid`` dispatches by scheme name.

References
----------
Lebedev, V. I., Laikov, D. N. (1999). A quadrature formula for the sphere of the
  131st algebraic order of accuracy. *Doklady Mathematics*, 59(3), 477–481.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .types import ObservationPoints

_TWO_PI = 2.0 * math.pi
_FOUR_PI = 4.0 * math.pi

# Default measurement / reference axis: +z.  A loudspeaker's "on-axis" response is
# conventionally measured along the axis it faces; the dataset records this as
# ``reference_axis`` (root attr) so views never hardcode +z.  Default keeps every
# legacy file and the existing +z-facing test geometry byte-identical.
DEFAULT_REFERENCE_AXIS = (0.0, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Reference-axis / on-axis direction selection
# ---------------------------------------------------------------------------


def nearest_direction_index(
    unit_vectors: np.ndarray, axis: tuple[float, float, float] | np.ndarray
) -> int:
    """Index of the sampled direction closest to a reference axis.

    "On-axis" is the direction whose unit vector has the largest projection
    (dot product) onto the (normalised) reference ``axis``.  Used by the results
    views to pick the on-axis response for a defined measurement axis instead of
    hardcoding ``argmax(unit_vectors[:, 2])`` (which is only correct when the
    axis happens to be +z).

    Parameters
    ----------
    unit_vectors : np.ndarray
        ``[N × 3]`` float — the sphere sampling directions (rows ~unit length).
    axis : tuple of 3 float or np.ndarray
        Reference axis in the same Cartesian frame.  Normalised internally;
        a zero vector falls back to +z.

    Returns
    -------
    int
        Row index into ``unit_vectors`` of the on-axis direction.

    Notes
    -----
    With ``axis = (0, 0, 1)`` this is exactly ``argmax(unit_vectors[:, 2])`` —
    the prior hardcoded behaviour — so the default path is unchanged.
    """
    a = np.asarray(axis, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(a))
    if norm == 0.0:
        a = np.array(DEFAULT_REFERENCE_AXIS, dtype=np.float64)
    else:
        a = a / norm
    projections = np.asarray(unit_vectors, dtype=np.float64) @ a  # [N]
    return int(np.argmax(projections))


def reference_frame(
    axis: tuple[float, float, float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Orthonormal (front, right, up) frame for a loudspeaker's reference axis.

    Builds the right-handed measurement frame the polar, sonogram and CEA-2034-A
    views use to define their horizontal and vertical great circles.  ``front`` is
    the loudspeaker's on-axis/look direction; ``up`` is the in-frame vertical (so a
    vertical orbit angle of ``+β`` points *up* — the CEA ceiling bounces land at +β,
    floor at −β); ``right`` completes a right-handed triad.

    Construction
    ------------
    - ``front`` = normalised ``axis`` (falls back to +z if ``axis`` is zero).
    - The world "up" reference is +z, unless ``front`` is (anti)parallel to +z (e.g.
      the default +z-facing test geometry), in which case +y is used so the frame is
      still well defined.
    - ``up`` = the world-up reference projected ⟂ ``front`` and normalised.
    - ``right`` = ``up × front`` (unit), so ``{front, right, up}`` is right-handed and
      ``front = right × up``.

    Parameters
    ----------
    axis : tuple of 3 float or np.ndarray
        The reference / front axis in the dataset's Cartesian frame.

    Returns
    -------
    front, right, up : np.ndarray
        Three ``[3]`` float64 unit vectors forming a right-handed orthonormal frame.

    Notes
    -----
    For a ``front`` that is not vertical (e.g. a +x-facing speaker), this yields the
    natural studio convention: a floor-parallel horizontal orbit and a vertical orbit
    containing world-up.  For a +z-facing speaker (no distinct world-up), the H/V
    assignment is conventional but consistent and stable.
    """
    front = np.asarray(axis, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(front))
    front = np.array(DEFAULT_REFERENCE_AXIS, dtype=np.float64) if n == 0.0 else front / n

    world_up = np.array([0.0, 0.0, 1.0])
    if abs(float(front @ world_up)) > 0.999:  # front ∥ +z → pick +y as the up reference
        world_up = np.array([0.0, 1.0, 0.0])

    up = world_up - float(world_up @ front) * front
    up /= np.linalg.norm(up)
    right = np.cross(up, front)
    right /= np.linalg.norm(right)
    return front, right, up


# ---------------------------------------------------------------------------
# Internal grid type (unit-sphere only; no radius)
# ---------------------------------------------------------------------------


@dataclass
class _SphereGrid:
    """Unit-sphere grid with quadrature weights. Internal use only.

    Public callers receive ObservationPoints (which adds a radius).
    """

    unit_vectors: np.ndarray  # [N, 3] float64
    weights: np.ndarray  # [N] float64, sum == 4π
    theta_phi: np.ndarray  # [N, 2] float64, (θ, φ) in radians
    scheme: str
    order: int
    weight_convention: str = "sum_4pi"

    def to_observation_points(self, radius: float) -> ObservationPoints:
        """Attach an observation radius to produce an ObservationPoints."""
        return ObservationPoints(
            unit_vectors=self.unit_vectors,
            radius=radius,
            weights=self.weights,
            scheme=self.scheme,
            order=self.order,
            weight_convention=self.weight_convention,
            theta_phi=self.theta_phi,
        )


# ---------------------------------------------------------------------------
# Coordinate conversion helpers
# ---------------------------------------------------------------------------


def _cartesian_to_theta_phi(xyz: np.ndarray) -> np.ndarray:
    """Convert [N, 3] unit vectors to [N, 2] (colatitude θ, azimuth φ) in radians.

    Physics convention: θ ∈ [0, π] is the angle from the +z axis (north pole),
    φ ∈ [0, 2π) is the azimuth measured from +x toward +y.

    This matches NumCalc's evaluation-grid convention and scipy.special.sph_harm's
    expected (theta, phi) arguments.
    """
    theta = np.arccos(np.clip(xyz[:, 2], -1.0, 1.0))  # [N] colatitude in [0, π]
    phi = np.arctan2(xyz[:, 1], xyz[:, 0]) % _TWO_PI  # [N] azimuth in [0, 2π)
    return np.column_stack([theta, phi])  # [N, 2]


# ---------------------------------------------------------------------------
# Lebedev–Laikov orbit generators (octahedral group Oh)
#
# The octahedral group partitions points on the unit sphere into orbits.
# Each orbit shares a single quadrature weight.  Three orbit types suffice for
# the implemented grids:
#
#   Type 1 (axis): (±1, 0, 0) and coordinate permutations → 6 points
#   Type 2 (edge): (±a, ±a, 0) and coordinate permutations → 12 points
#   Type 3 (body diagonal): (±a, ±a, ±a) → 8 points
#
# Larger grids (≥ 38 points) require orbits with two or three free parameters,
# whose numerical values come from the Lebedev–Laikov 1999 paper and cannot be
# reliably derived by hand.  Those orders raise NotImplementedError.
# ---------------------------------------------------------------------------


def _axis_orbit(w: float) -> tuple[np.ndarray, np.ndarray]:
    """6 axis-intersection points, all with weight w."""
    pts = np.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=np.float64,
    )
    return pts, np.full(6, w)


def _edge_orbit(a: float, w: float) -> tuple[np.ndarray, np.ndarray]:
    """12 edge-midpoint points from orbit (a, a, 0), all with weight w.

    For a = 1/√2 the points lie on the unit sphere.  The orbit contains all
    sign combinations of the two non-zero coordinates and all three coordinate
    planes, giving 4 × 3 = 12 points total.
    """
    b = a
    pts = np.array(
        [
            # z = 0 plane
            [a, b, 0.0],
            [a, -b, 0.0],
            [-a, b, 0.0],
            [-a, -b, 0.0],
            # y = 0 plane
            [a, 0.0, b],
            [a, 0.0, -b],
            [-a, 0.0, b],
            [-a, 0.0, -b],
            # x = 0 plane
            [0.0, a, b],
            [0.0, a, -b],
            [0.0, -a, b],
            [0.0, -a, -b],
        ],
        dtype=np.float64,
    )
    return pts, np.full(12, w)


def _body_orbit(a: float, w: float) -> tuple[np.ndarray, np.ndarray]:
    """8 body-diagonal points from orbit (a, a, a), all with weight w.

    For a = 1/√3 the points lie on the unit sphere.
    """
    pts = np.array(
        [
            [a, a, a],
            [a, a, -a],
            [a, -a, a],
            [a, -a, -a],
            [-a, a, a],
            [-a, a, -a],
            [-a, -a, a],
            [-a, -a, -a],
        ],
        dtype=np.float64,
    )
    return pts, np.full(8, w)


def _assemble_grid(scheme: str, order: int, *orbits: tuple) -> _SphereGrid:
    """Stack orbit (points, weights) tuples into a _SphereGrid."""
    pts_list, w_list = zip(*orbits)
    xyz = np.vstack(pts_list)  # [N, 3]
    weights = np.concatenate(w_list)  # [N]
    return _SphereGrid(
        unit_vectors=xyz,
        weights=weights,
        theta_phi=_cartesian_to_theta_phi(xyz),
        scheme=scheme,
        order=order,
    )


# ---------------------------------------------------------------------------
# Implemented Lebedev–Laikov grids
#
# Weights are in the "sum_4pi" convention: Σ wᵢ = 4π.
#
# VERIFIED against Lebedev & Laikov (1999) Table 1:
#   n=6:  single orbit, all weights = 4π/6
#   n=14: Σ wᵢ = 6 × (4π/15) + 8 × (3π/10) = 8π/5 + 12π/5 = 4π ✓
#   n=26: Σ wᵢ = 6 × (4π/21) + 12 × (16π/105) + 8 × (9π/70)
#               = 120π/105 + 192π/105 + 108π/105 = 420π/105 = 4π ✓
#
# Cross-checked: weights reproduce ∫ Y_l^m dΩ = 0 for l = 2, 4 (n=14) and
# l = 2, 4, 6 (n=26) to floating-point precision.
# ---------------------------------------------------------------------------

_A_EDGE = 1.0 / math.sqrt(2.0)  # 1/√2  — edge-midpoint orbit radius
_A_BODY = 1.0 / math.sqrt(3.0)  # 1/√3  — body-diagonal orbit radius


def _lebedev_006() -> _SphereGrid:
    """6-point Lebedev grid, exact for algebraic degree 3."""
    return _assemble_grid("lebedev", 6, _axis_orbit(_FOUR_PI / 6.0))


def _lebedev_014() -> _SphereGrid:
    """14-point Lebedev grid, exact for algebraic degree 5.

    Quadrature conditions (normalization + Y₄⁰ = 0) yield:
      w_axis = 4π/15,  w_body = 3π/10.
    Derived analytically; confirmed against Lebedev & Laikov (1999).
    """
    return _assemble_grid(
        "lebedev",
        14,
        _axis_orbit(_FOUR_PI / 15.0),
        _body_orbit(_A_BODY, 3.0 * math.pi / 10.0),
    )


def _lebedev_026() -> _SphereGrid:
    """26-point Lebedev grid, exact for algebraic degree 7.

    Three orbits; weights from Lebedev & Laikov (1999):
      w_axis = 4π/21,  w_edge = 16π/105,  w_body = 9π/70.
    Confirmed analytically: integrates Y_l^m dΩ = 0 for l = 2, 4, 6 ✓.
    """
    return _assemble_grid(
        "lebedev",
        26,
        _axis_orbit(_FOUR_PI / 21.0),
        _edge_orbit(_A_EDGE, 16.0 * math.pi / 105.0),
        _body_orbit(_A_BODY, 9.0 * math.pi / 70.0),
    )


_LEBEDEV_BUILDERS: dict[int, object] = {
    6: _lebedev_006,
    14: _lebedev_014,
    26: _lebedev_026,
}

LEBEDEV_AVAILABLE: list[int] = sorted(_LEBEDEV_BUILDERS)
"""Lebedev grid sizes currently implemented."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def lebedev(n_points: int = 26, *, radius: float = 1.0) -> ObservationPoints:
    """Lebedev–Laikov quadrature grid on an observation sphere.

    Returns a set of directions on the unit sphere together with exact
    quadrature weights. The weights satisfy Σ wᵢ = 4π, so integrating any
    function f over the sphere is:

        ∫ f dΩ  ≈  Σ wᵢ · f(unit_vectorsᵢ)

    For loudspeaker work, this means radiated power is Σ wᵢ |H(directionᵢ)|²
    (times the acoustic normalization factor), with no extra cos(θ) term needed.

    VERIFIED: Lebedev & Laikov (1999), "A quadrature formula for the sphere of
    the 131st algebraic order of accuracy", *Doklady Mathematics*, 59(3), 477–481.

    Parameters
    ----------
    n_points : int
        Number of quadrature points. Must be in ``LEBEDEV_AVAILABLE`` (currently
        {6, 14, 26}).  Production grids (≥ 38 points) raise NotImplementedError
        until vendored Lebedev–Laikov tables are added.
    radius : float
        Observation sphere radius in metres (default 1.0).

    Returns
    -------
    ObservationPoints
        Unit vectors [N, 3], weights [N] summing to 4π, theta/phi [N, 2], and
        scheme metadata.

    Raises
    ------
    ValueError
        If ``n_points`` is not a valid Lebedev order.
    NotImplementedError
        If ``n_points`` is larger than the largest implemented order.
    """
    if n_points in _LEBEDEV_BUILDERS:
        grid = _LEBEDEV_BUILDERS[n_points]()  # type: ignore[operator]
        return grid.to_observation_points(radius)
    max_implemented = max(LEBEDEV_AVAILABLE)
    if n_points > max_implemented:
        raise NotImplementedError(
            f"Lebedev grids with n_points > {max_implemented} require vendored "
            "Lebedev–Laikov tables (not yet implemented). "
            f"Available now: {LEBEDEV_AVAILABLE}."
        )
    raise ValueError(
        f"n_points={n_points} is not a valid Lebedev order. " f"Available: {LEBEDEV_AVAILABLE}."
    )


def fliege_maier(degree: int, *, radius: float = 1.0) -> ObservationPoints:
    """Fliege–Maier spherical quadrature grid (not yet implemented).

    Parameters
    ----------
    degree : int
        Spherical polynomial degree to integrate exactly.
    radius : float
        Observation sphere radius in metres.

    Raises
    ------
    NotImplementedError
        Always; use ``lebedev()`` instead.
    """
    raise NotImplementedError("Fliege–Maier grids are not yet implemented. Use lebedev() instead.")


def t_design(degree: int, *, radius: float = 1.0) -> ObservationPoints:
    """Spherical t-design grid (equal-weight quadrature, not yet implemented).

    Parameters
    ----------
    degree : int
        Degree of polynomial exactness.
    radius : float
        Observation sphere radius in metres.

    Raises
    ------
    NotImplementedError
        Always; use ``lebedev()`` instead.
    """
    raise NotImplementedError("Spherical t-designs are not yet implemented. Use lebedev() instead.")


# ---------------------------------------------------------------------------
# Icosphere (geodesic) grid — algorithmic, scales to thousands of points
#
# Built by recursively subdividing a unit icosahedron and projecting new
# vertices to the sphere.  Unlike Lebedev it has no closed-form polynomial-exact
# quadrature weights, but it is *near-uniform* (avoiding the lat/long pole
# clustering the data contract warns against) and is generated entirely in code
# (no vendored tables).  Vertex quadrature weights are the spherical-triangle
# areas of the surrounding faces (Girard's theorem), distributed 1/3 to each
# vertex, so they sum to exactly 4π — a first-order quadrature whose accuracy
# improves with point count.  This is the Phase-2 "Balloon" grid: it supplies the
# hundreds-to-thousands of directions beam design and audit need.
#
# Point count after s subdivisions: N = 10·4^s + 2
#   s=0:12  s=1:42  s=2:162  s=3:642  s=4:2562  s=5:10242  s=6:40962
# ---------------------------------------------------------------------------

# Canonical unit-icosahedron vertices (golden-ratio construction) and the 20
# triangular faces (consistent outward winding).
_PHI = (1.0 + math.sqrt(5.0)) / 2.0
_ICOSA_VERTS = np.array(
    [
        [-1.0, _PHI, 0.0],
        [1.0, _PHI, 0.0],
        [-1.0, -_PHI, 0.0],
        [1.0, -_PHI, 0.0],
        [0.0, -1.0, _PHI],
        [0.0, 1.0, _PHI],
        [0.0, -1.0, -_PHI],
        [0.0, 1.0, -_PHI],
        [_PHI, 0.0, -1.0],
        [_PHI, 0.0, 1.0],
        [-_PHI, 0.0, -1.0],
        [-_PHI, 0.0, 1.0],
    ],
    dtype=np.float64,
)
_ICOSA_FACES = np.array(
    [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ],
    dtype=np.int64,
)  # fmt: skip


def _subdivide_icosphere(subdivisions: int) -> tuple[np.ndarray, np.ndarray]:
    """Subdivide the unit icosahedron ``subdivisions`` times.

    Returns
    -------
    verts : np.ndarray
        ``[V, 3]`` float64 unit vectors (projected to the sphere).
    faces : np.ndarray
        ``[T, 3]`` int64 triangle vertex indices.
    """
    verts: list[np.ndarray] = [v / np.linalg.norm(v) for v in _ICOSA_VERTS]
    faces = _ICOSA_FACES.tolist()
    midpoint_cache: dict[tuple[int, int], int] = {}

    def _midpoint(i: int, j: int) -> int:
        key = (i, j) if i < j else (j, i)
        cached = midpoint_cache.get(key)
        if cached is not None:
            return cached
        m = verts[i] + verts[j]
        m /= np.linalg.norm(m)  # project to the unit sphere
        idx = len(verts)
        verts.append(m)
        midpoint_cache[key] = idx
        return idx

    for _ in range(subdivisions):
        new_faces: list[list[int]] = []
        for a, b, c in faces:
            ab = _midpoint(a, b)
            bc = _midpoint(b, c)
            ca = _midpoint(c, a)
            new_faces.extend([[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]])
        faces = new_faces

    return np.array(verts, dtype=np.float64), np.array(faces, dtype=np.int64)


def _spherical_triangle_areas(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Spherical excess (area, steradians) of each triangle (Van Oosterom–Strackee).

    ``tan(E/2) = |a·(b×c)| / (1 + a·b + b·c + c·a)``; the excesses sum to 4π.
    """
    a = verts[faces[:, 0]]  # [T,3]
    b = verts[faces[:, 1]]
    c = verts[faces[:, 2]]
    triple = np.abs(np.einsum("ij,ij->i", a, np.cross(b, c)))  # [T]
    denom = (
        1.0
        + np.einsum("ij,ij->i", a, b)
        + np.einsum("ij,ij->i", b, c)
        + np.einsum("ij,ij->i", c, a)
    )
    return 2.0 * np.arctan2(triple, denom)  # [T] steradians


def _icosphere_grid(subdivisions: int) -> _SphereGrid:
    """Build a near-uniform icosphere grid with area-based vertex weights."""
    verts, faces = _subdivide_icosphere(subdivisions)  # [V,3], [T,3]
    areas = _spherical_triangle_areas(verts, faces)  # [T]
    # Distribute each triangle's area equally to its three vertices -> sum == 4π.
    weights = np.zeros(verts.shape[0], dtype=np.float64)
    np.add.at(weights, faces[:, 0], areas / 3.0)
    np.add.at(weights, faces[:, 1], areas / 3.0)
    np.add.at(weights, faces[:, 2], areas / 3.0)
    return _SphereGrid(
        unit_vectors=verts,
        weights=weights,
        theta_phi=_cartesian_to_theta_phi(verts),
        scheme="icosphere",
        order=subdivisions,
        weight_convention="sum_4pi",
    )


def icosphere(subdivisions: int, *, radius: float = 1.0) -> ObservationPoints:
    """Near-uniform geodesic (icosphere) grid scaling to thousands of points.

    The Phase-2 "Balloon" grid for beam design / audit. Generated entirely in code
    (no vendored tables) by subdividing a unit icosahedron; vertex weights are the
    spherical-triangle areas of the surrounding faces (sum = 4π). Weights are a
    first-order quadrature (not polynomial-exact like Lebedev) whose accuracy
    improves with point count — adequate for covariance/DI integrals at the
    hundreds-to-thousands point counts used here.

    Parameters
    ----------
    subdivisions : int
        Number of icosahedron subdivision levels (>= 0). Point count
        ``N = 10 * 4**subdivisions + 2`` (s=3 -> 642, s=4 -> 2562, s=5 -> 10242).
    radius : float
        Observation sphere radius in metres.

    Returns
    -------
    ObservationPoints
        With ``sum(weights) == 4π`` and ``scheme == "icosphere"``.

    Raises
    ------
    ValueError
        If ``subdivisions`` is negative.
    """
    if subdivisions < 0:
        raise ValueError(f"subdivisions must be >= 0, got {subdivisions}.")
    return _icosphere_grid(subdivisions).to_observation_points(radius)


def make_observation_grid(scheme: str, n_points: int, *, radius: float = 1.0) -> ObservationPoints:
    """Build an observation grid by scheme name, ``n_points`` interpreted as a target.

    The single entry point the pipeline/GUI use so a request can ask for an exact
    Lebedev order or a near-uniform icosphere with *at least* ``n_points`` directions.

    Parameters
    ----------
    scheme : str
        ``"lebedev"`` (exact quadrature, orders {6, 14, 26}) or ``"icosphere"``
        (near-uniform, thousands of points; the Phase-2 "Balloon" grid).
    n_points : int
        For ``"lebedev"``: the exact order (must be implemented). For ``"icosphere"``:
        a target count — the smallest subdivision with ``>= n_points`` is chosen.
    radius : float
        Observation sphere radius in metres.

    Returns
    -------
    ObservationPoints

    Raises
    ------
    ValueError
        If ``scheme`` is unknown.
    """
    if scheme == "lebedev":
        return lebedev(n_points, radius=radius)
    if scheme == "icosphere":
        subdivisions = 0
        while 10 * 4**subdivisions + 2 < n_points:
            subdivisions += 1
        return icosphere(subdivisions, radius=radius)
    raise ValueError(f"Unknown sphere scheme {scheme!r}. Use 'lebedev' or 'icosphere'.")
