"""Spherical-harmonic (SH) transform and grid resampling (Stage P2-0c).

The Phase-1 BEM results live on a *scattered* near-uniform sphere grid (Lebedev or
icosphere). Several Phase-2 deliverables need the field on a *regular* angular grid or a
specific set of directions:

* filtered ``.frd`` export on matched horizontal/vertical polar arcs (VituixCAD/REW),
* CLF balloon export (regular lat/lon grid),
* the V-CBT beamwidth measurement (a fine great-circle arc),
* GUI balloon / polar visualization.

The bridge is a band-limited spherical-harmonic model: fit complex SH coefficients to the
field on the source grid (``forward_sh``), then evaluate them anywhere (``inverse_sh``).
``resample`` chains the two. Because the icosphere weights are only a first-order
quadrature, the forward fit defaults to **least squares** (robust on any grid); a
quadrature projection is offered for exact-weight Lebedev grids.

Conventions
-----------
Complex SH ``Y_l^m(theta, phi)`` via :func:`scipy.special.sph_harm_y` with ``theta`` the
colatitude in ``[0, pi]`` and ``phi`` the azimuth in ``[0, 2*pi)`` — matching
``ObservationPoints.theta_phi`` (see ``core.sphere._cartesian_to_theta_phi``). Coefficients
are indexed by the flattened ``(l, m)`` order ``l = 0..L, m = -l..l`` (length ``(L+1)**2``).

References
----------
Rafaely, *Fundamentals of Spherical Array Processing*, Springer, 2015 (SH fitting).
docs/Phase 2 - Filter Solver.md §5.5; DATA_CONTRACT.md §3.2.
"""

from __future__ import annotations

import numpy as np
from scipy.special import sph_harm_y

from beamsim2.core.types import ObservationPoints


def n_coeffs(order: int) -> int:
    """Number of SH coefficients for max degree ``order``: ``(order + 1) ** 2``."""
    return (order + 1) ** 2


def sh_design_matrix(order: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Complex SH design matrix ``Y[(order+1)^2, N]`` evaluated at ``(theta, phi)``.

    Row ``k`` (flattened ``l = 0..order, m = -l..l``) is ``Y_l^m`` sampled at the N
    directions. ``field[n] = sum_k coeff[k] * Y[k, n]`` reconstructs a band-limited field.

    Parameters
    ----------
    order : int
        Maximum SH degree ``L``.
    theta : np.ndarray
        ``[N]`` colatitude in radians ``[0, pi]``.
    phi : np.ndarray
        ``[N]`` azimuth in radians ``[0, 2*pi)``.

    Returns
    -------
    np.ndarray
        ``[(order+1)**2, N]`` complex128.
    """
    rows = []
    for l in range(order + 1):  # noqa: E741 — conventional SH degree symbol
        for m in range(-l, l + 1):
            rows.append(sph_harm_y(l, m, theta, phi))
    return np.asarray(rows, dtype=np.complex128)  # [K, N]


def forward_sh(
    field: np.ndarray,
    obs: ObservationPoints,
    order: int,
    *,
    method: str = "lstsq",
) -> np.ndarray:
    """Fit SH coefficients to a field sampled on a sphere grid.

    Parameters
    ----------
    field : np.ndarray
        ``[N]`` or ``[F, N]`` complex — the field on ``obs`` (last axis = directions).
    obs : ObservationPoints
        Source grid (provides ``theta_phi`` and quadrature ``weights``).
    order : int
        Maximum SH degree ``L``. Requires ``(L+1)**2 <= N`` for a determined least-squares
        fit; raises otherwise.
    method : str
        ``"lstsq"`` (default; robust on any grid, incl. icosphere) solves the
        least-squares system ``Y^T c = field``. ``"quadrature"`` uses
        ``c = (conj(Y) * w) @ field`` — exact only for polynomial-exact (Lebedev) weights.

    Returns
    -------
    np.ndarray
        ``[(L+1)**2]`` or ``[F, (L+1)**2]`` complex128 coefficients.
    """
    theta, phi = obs.theta_phi[:, 0], obs.theta_phi[:, 1]
    n = theta.shape[0]
    k = n_coeffs(order)
    if k > n:
        raise ValueError(
            f"SH order {order} needs (L+1)^2={k} coefficients but the grid has only "
            f"N={n} points; reduce the order or use a denser grid."
        )
    Y = sh_design_matrix(order, theta, phi)  # [K, N]

    if method == "quadrature":
        w = obs.weights  # [N]
        if field.ndim == 2:
            return ((np.conj(Y) * w[None, :]) @ field.T).T  # [F, K]
        return (np.conj(Y) * w[None, :]) @ field  # [K]
    if method == "lstsq":
        # field = Y^T @ c  ->  solve [N, K] c = field for c (multi-RHS for [F, N]).
        A = Y.T  # [N, K]
        rhs = field.T if field.ndim == 2 else field  # [N, F] or [N]
        coeffs, *_ = np.linalg.lstsq(A, rhs, rcond=None)  # [K, F] or [K]
        return coeffs.T if field.ndim == 2 else coeffs
    raise ValueError(f"Unknown method {method!r}; use 'lstsq' or 'quadrature'.")


def inverse_sh(coeffs: np.ndarray, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Evaluate an SH coefficient set at directions ``(theta, phi)``.

    Parameters
    ----------
    coeffs : np.ndarray
        ``[K]`` or ``[F, K]`` complex (``K = (L+1)**2``).
    theta, phi : np.ndarray
        ``[M]`` colatitude / azimuth in radians.

    Returns
    -------
    np.ndarray
        ``[M]`` or ``[F, M]`` complex128 field at the requested directions.
    """
    k = coeffs.shape[-1]
    order = int(round(np.sqrt(k))) - 1
    if n_coeffs(order) != k:
        raise ValueError(f"coeffs length {k} is not a perfect (L+1)^2.")
    Y = sh_design_matrix(order, theta, phi)  # [K, M]
    return coeffs @ Y  # [M] or [F, M]


def resample(
    field: np.ndarray,
    src_obs: ObservationPoints,
    target_unit_vectors: np.ndarray,
    order: int,
    *,
    method: str = "lstsq",
) -> np.ndarray:
    """Resample a field from ``src_obs`` to arbitrary ``target_unit_vectors`` via SH.

    Parameters
    ----------
    field : np.ndarray
        ``[N]`` or ``[F, N]`` complex on ``src_obs``.
    src_obs : ObservationPoints
        Source grid.
    target_unit_vectors : np.ndarray
        ``[M, 3]`` unit directions to evaluate at.
    order : int
        Maximum SH degree for the intermediate model.
    method : str
        Forward-fit method (see :func:`forward_sh`).

    Returns
    -------
    np.ndarray
        ``[M]`` or ``[F, M]`` complex128 field at the target directions.
    """
    coeffs = forward_sh(field, src_obs, order, method=method)
    t = target_unit_vectors
    theta = np.arccos(np.clip(t[:, 2], -1.0, 1.0))  # [M]
    phi = np.arctan2(t[:, 1], t[:, 0]) % (2.0 * np.pi)  # [M]
    return inverse_sh(coeffs, theta, phi)


def safe_order_for_grid(n_points: int, *, margin: float = 0.5) -> int:
    """A conservative SH order whose least-squares fit is over-determined on ``n_points``.

    Picks the largest ``L`` with ``(L+1)**2 <= margin * n_points`` (default keeps the system
    at least 2x over-determined), so the fit is well-conditioned. Physically the field's own
    bandlimit (``L ~ ceil(k * r_source)``) should also cap the order.
    """
    target = max(1.0, margin * n_points)
    order = 0
    while n_coeffs(order + 1) <= target:
        order += 1
    return order


def regular_lat_lon_grid(
    n_theta: int = 181, n_phi: int = 360
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Regular colatitude/azimuth grid for balloon / CLF export.

    Parameters
    ----------
    n_theta : int
        Number of colatitude samples in ``[0, pi]`` (inclusive of both poles).
    n_phi : int
        Number of azimuth samples in ``[0, 2*pi)`` (excluding the wrap point).

    Returns
    -------
    unit_vectors : np.ndarray
        ``[n_theta * n_phi, 3]`` Cartesian directions (row-major over theta then phi).
    theta : np.ndarray
        ``[n_theta * n_phi]`` colatitude per point.
    phi : np.ndarray
        ``[n_theta * n_phi]`` azimuth per point.
    """
    th = np.linspace(0.0, np.pi, n_theta)
    ph = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    TH, PH = np.meshgrid(th, ph, indexing="ij")  # [n_theta, n_phi]
    theta = TH.ravel()
    phi = PH.ravel()
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.column_stack([x, y, z]), theta, phi


def great_circle_arc(axis: np.ndarray, n_points: int = 361) -> tuple[np.ndarray, np.ndarray]:
    """A fine great-circle arc through an ``axis`` direction (for polar / beamwidth).

    Returns ``angle`` in radians measured from ``axis`` and the ``[n_points, 3]`` unit
    vectors sweeping the great circle that contains ``axis`` and (preferably) ``+z``.

    Parameters
    ----------
    axis : np.ndarray
        ``[3]`` direction the arc passes through at angle 0.
    n_points : int
        Number of samples over ``[-pi, pi]``.

    Returns
    -------
    angle : np.ndarray
        ``[n_points]`` angle from ``axis`` in radians, in ``[-pi, pi]``.
    unit_vectors : np.ndarray
        ``[n_points, 3]`` directions sweeping the great circle.
    """
    a = np.asarray(axis, dtype=np.float64)
    a = a / np.linalg.norm(a)
    # An in-plane reference orthogonal to the axis (use +z unless (anti)parallel).
    ref = np.array([0.0, 0.0, 1.0])
    if abs(a @ ref) > 0.999:
        ref = np.array([1.0, 0.0, 0.0])
    perp = ref - (ref @ a) * a
    perp /= np.linalg.norm(perp)
    angle = np.linspace(-np.pi, np.pi, n_points)
    uv = np.cos(angle)[:, None] * a[None, :] + np.sin(angle)[:, None] * perp[None, :]
    return angle, uv
