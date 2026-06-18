"""Sphere sampling grids with integration weights for directional BEM work.

The primary scheme is Lebedev–Laikov quadrature, which provides exact quadrature
weights for integrating polynomials up to a given algebraic degree over the sphere.
In loudspeaker terms: you can compute radiated power, directivity index, and
Phase-2 covariance matrices as simple weighted sums Σ wᵢ f(directionᵢ) with no
ad-hoc angular-density correction.

Three alternative schemes (Fliege–Maier, t-design, icosphere) are reserved for
future implementation; they raise NotImplementedError to prevent silent wrong results.

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


def icosphere(subdivisions: int, *, radius: float = 1.0) -> ObservationPoints:
    """Icosphere grid (no exact quadrature weights, not yet implemented).

    Parameters
    ----------
    subdivisions : int
        Number of icosahedron subdivision levels (controls point count).
    radius : float
        Observation sphere radius in metres.

    Raises
    ------
    NotImplementedError
        Always; use ``lebedev()`` instead.
    """
    raise NotImplementedError("Icosphere grids are not yet implemented. Use lebedev() instead.")
