"""Geometry: parametric primitives, health checks, driver assembly, and gmsh meshing (DR-03).

Public API
----------
assemble.DriverSpec          — driver cap descriptor
assemble.assemble_box_driver — box + driver(s) → tagged Mesh + BoundaryConditions
health.run_health_checks     — watertight/normal/degenerate checks + auto-repair
health.HealthReport          — check result dataclass
mesh.target_edge_length      — c / (f_max * n_epw) sizing rule
mesh.mesh_geometry           — full pipeline: size + assemble + health-check
primitives.make_sphere_mesh  — gmsh OCC sphere (V-2 physics canary)
primitives.make_box_mesh     — standalone box mesh (no driver)
"""
