"""Element sizing utilities and mesh-generation convenience wrapper.

The core function is ``target_edge_length``, which implements DR-03's
automatic sizing rule: edge = λ_min / N_epw = c / (f_max · N_epw).

``mesh_geometry`` is a thin orchestration wrapper that computes the target
edge, calls ``assemble_box_driver`` (the only supported shape in item 5),
runs health checks, and returns a verified (Mesh, BoundaryConditions).

Deferred (item 6+): multi-band remeshing and per-frequency mesh routing table.
At Stage-1 mid-band frequencies a single finest mesh is correct and memory-safe;
the routing-table logic is a Stage-3 RAM optimisation.

Build-order item 5 (DR-03, pipeline Stage C).
"""

from __future__ import annotations

from beamsim2.core.types import BoundaryConditions, Mesh, SolverConfig
from beamsim2.geometry.assemble import DriverSpec, assemble_box_driver
from beamsim2.geometry.health import HealthReport, run_health_checks


def target_edge_length(
    f_max: float,
    n_epw: int = 6,
    c: float = 343.2,
) -> float:
    """Compute the target BEM element edge length for a given frequency ceiling.

    Implements the DR-03 / CLAUDE.md automatic-sizing rule:
    edge = λ_min / N_epw = c / (f_max · N_epw).

    HEURISTIC/VERIFIED: N_epw = 6–8 for NumCalc's constant-collocation elements
    (Kreuzer et al., Engineering Analysis Boundary Elements 161:157-178, 2024).

    Parameters
    ----------
    f_max : float
        Upper frequency in Hz. Sets the finest wavelength λ_min = c / f_max.
    n_epw : int
        Elements per wavelength at f_max (default 6 — minimum acceptable).
    c : float
        Speed of sound in m/s (default 343.2 m/s ≈ dry air at 20 °C).

    Returns
    -------
    float
        Target edge length in metres.

    Examples
    --------
    >>> round(target_edge_length(20000, n_epw=6), 5)
    0.00286
    """
    if f_max <= 0:
        raise ValueError(f"f_max must be positive, got {f_max}")
    if n_epw < 1:
        raise ValueError(f"n_epw must be ≥ 1, got {n_epw}")
    return c / (f_max * n_epw)


def mesh_geometry(
    width: float,
    height: float,
    depth: float,
    drivers: list[DriverSpec],
    config: SolverConfig,
    fillet_radius: float = 0.0,
    f_max: float | None = None,
    h_elem: float | None = None,
) -> tuple[Mesh, BoundaryConditions, HealthReport]:
    """Build a box-driver assembly with automatic element sizing and health checks.

    Convenience wrapper for the Stage-C pipeline step:
    1. Compute target edge from ``config.n_epw`` and ``f_max`` (or use ``h_elem``
       directly if supplied).
    2. Call ``assemble_box_driver`` to produce a tagged Mesh.
    3. Run ``run_health_checks``; raise ``ValueError`` if non-repairable defects
       are found.

    TODO (item 6+): Multi-band remeshing + routing table.  Currently produces
    a single mesh at the finest required edge length.

    Parameters
    ----------
    width, height, depth : float
        Box dimensions in metres.
    drivers : list[DriverSpec]
        Driver descriptors (see ``assemble.DriverSpec``).
    config : SolverConfig
        Solver settings; ``config.n_epw`` and ``config.speed_of_sound`` are used
        for automatic sizing.
    fillet_radius : float
        Box edge fillet in metres (see ``assemble_box_driver``).
    f_max : float or None
        Upper frequency in Hz for automatic sizing. Required unless ``h_elem``
        is provided.
    h_elem : float or None
        Override target edge length in metres (skips automatic sizing).

    Returns
    -------
    mesh : Mesh
        Health-checked, tagged surface mesh.
    bc : BoundaryConditions
        Boundary conditions (driver groups vibrating at unit velocity).
    report : HealthReport
        Health report (repairs logged; problems → ValueError raised first).

    Raises
    ------
    ValueError
        If both ``f_max`` and ``h_elem`` are None, or if health checks find
        non-repairable defects (report.problems is non-empty).
    """
    if h_elem is None:
        if f_max is None:
            raise ValueError("Provide either f_max (Hz) or h_elem (m) for sizing.")
        h_elem = target_edge_length(f_max, n_epw=config.n_epw, c=config.speed_of_sound)

    mesh, bc = assemble_box_driver(
        width=width,
        height=height,
        depth=depth,
        drivers=drivers,
        h_elem=h_elem,
        fillet_radius=fillet_radius,
    )

    mesh, report = run_health_checks(mesh, target_edge=h_elem)

    if report.problems:
        problem_str = "\n  ".join(report.problems)
        raise ValueError(
            f"Geometry health check failed ({len(report.problems)} problem(s)):\n"
            f"  {problem_str}"
        )

    return mesh, bc, report
