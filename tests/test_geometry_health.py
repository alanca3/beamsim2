"""Geometry subsystem self-tests: primitives, assembly, health checks, sizing.

Tests are organised in five groups:

1. target_edge_length — pure arithmetic, no gmsh.
2. health checks — hand-built Mesh objects, no gmsh.
3. gmsh primitives — make_sphere_mesh, make_box_mesh (no NumCalc).
4. assembly — assemble_box_driver, one and two drivers (no NumCalc).
5. Physics canary — gmsh sphere through V-2 (local_only, needs NumCalc).
"""

from __future__ import annotations

import numpy as np
import pytest

from beamsim2.core.types import Mesh, SolverConfig
from beamsim2.geometry.assemble import (
    DriverSpec,
    _assert_groups_contiguous,
    assemble_box_driver,
)
from beamsim2.geometry.health import (
    check_degenerate,
    check_min_feature,
    check_normals,
    check_watertight,
    run_health_checks,
)
from beamsim2.geometry.mesh import target_edge_length
from beamsim2.geometry.primitives import make_box_mesh, make_sphere_mesh

# ===========================================================================
# 1. target_edge_length
# ===========================================================================


def test_target_edge_length_formula():
    h = target_edge_length(20000, n_epw=6, c=343.2)
    assert abs(h - 343.2 / (20000 * 6)) < 1e-10


def test_target_edge_length_n8():
    h = target_edge_length(10000, n_epw=8)
    assert abs(h - 343.2 / (10000 * 8)) < 1e-10


def test_target_edge_length_errors():
    with pytest.raises(ValueError):
        target_edge_length(0)
    with pytest.raises(ValueError):
        target_edge_length(1000, n_epw=0)


# ===========================================================================
# 2. Health checks (hand-built meshes, no gmsh)
# ===========================================================================


def _tetrahedron_mesh() -> Mesh:
    """Four-triangle closed mesh of a tetrahedron (watertight, all group 1).

    Winding chosen so face normals point outward from origin:
    for each face [A,B,C], cross(B-A, C-A) · centroid > 0.
    """
    s = 1.0 / np.sqrt(2.0)
    vertices = np.array([[1, 0, -s], [-1, 0, -s], [0, 1, s], [0, -1, s]], dtype=np.float64)
    # Verified: each face's cross product points outward from origin.
    triangles = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]], dtype=np.int32)
    group_tags = np.ones(4, dtype=np.int32)
    return Mesh(vertices=vertices, triangles=triangles, group_tags=group_tags)


def test_check_watertight_closed():
    ok, count, msg = check_watertight(_tetrahedron_mesh())
    assert ok is True
    assert count == 0
    assert msg == ""


def test_check_watertight_open():
    mesh = _tetrahedron_mesh()
    mesh_open = Mesh(
        vertices=mesh.vertices,
        triangles=mesh.triangles[:3],
        group_tags=mesh.group_tags[:3],
    )
    ok, count, msg = check_watertight(mesh_open)
    assert ok is False
    assert count > 0
    assert "BEM" in msg


def test_check_normals_already_correct():
    # Tetrahedron already has outward normals; second pass must find 0 flips.
    repaired, _ = check_normals(_tetrahedron_mesh())
    _, n_second = check_normals(repaired)
    assert n_second == 0


def test_check_normals_repairs_inverted():
    mesh = _tetrahedron_mesh()
    flipped_tris = mesh.triangles[:, [0, 2, 1]]
    mesh_inv = Mesh(vertices=mesh.vertices, triangles=flipped_tris, group_tags=mesh.group_tags)
    repaired, n_flipped = check_normals(mesh_inv)
    assert n_flipped > 0
    _, n_second = check_normals(repaired)
    assert n_second == 0


def test_check_degenerate_removes():
    mesh = _tetrahedron_mesh()
    degen_tri = np.array([[0, 0, 0]], dtype=np.int32)
    tris = np.vstack([mesh.triangles, degen_tri])
    tags = np.concatenate([mesh.group_tags, [1]])
    mesh_d = Mesh(vertices=mesh.vertices, triangles=tris, group_tags=tags)
    repaired, n_removed = check_degenerate(mesh_d)
    assert n_removed == 1
    assert len(repaired.triangles) == 4


def test_check_degenerate_clean():
    _, n = check_degenerate(_tetrahedron_mesh())
    assert n == 0


def test_check_min_feature_no_warning():
    warnings = check_min_feature(_tetrahedron_mesh(), target_edge=0.001)
    assert warnings == []


def test_check_min_feature_warns():
    warnings = check_min_feature(_tetrahedron_mesh(), target_edge=100.0)
    assert len(warnings) > 0
    assert "mm" in warnings[0]


def test_run_health_checks_clean():
    _, report = run_health_checks(_tetrahedron_mesh(), target_edge=0.001)
    assert report.is_watertight is True
    assert report.ok is True
    assert len(report.problems) == 0


def test_run_health_checks_open_mesh():
    mesh = _tetrahedron_mesh()
    mesh_open = Mesh(
        vertices=mesh.vertices,
        triangles=mesh.triangles[:3],
        group_tags=mesh.group_tags[:3],
    )
    _, report = run_health_checks(mesh_open)
    assert report.is_watertight is False
    assert len(report.problems) == 1
    assert "m" in report.problems[0]  # includes metre coordinates


def test_run_health_checks_logs_repairs():
    mesh = _tetrahedron_mesh()
    flipped_tris = mesh.triangles[:, [0, 2, 1]]
    mesh_inv = Mesh(vertices=mesh.vertices, triangles=flipped_tris, group_tags=mesh.group_tags)
    _, report = run_health_checks(mesh_inv)
    assert any("winding" in r.lower() or "normal" in r.lower() for r in report.repairs)


# ===========================================================================
# 3. gmsh primitives (no NumCalc)
# ===========================================================================


def test_make_sphere_mesh_watertight():
    mesh, bc = make_sphere_mesh(radius=0.1, h_elem=0.025)
    ok, count, _ = check_watertight(mesh)
    assert ok, f"Sphere mesh not watertight: {count} open edges"


def test_make_sphere_mesh_bc():
    _, bc = make_sphere_mesh(radius=0.1, h_elem=0.025)
    assert bc.vibrating_groups == {1: complex(1, 0)}


def test_make_sphere_mesh_outward_normals():
    mesh, _ = make_sphere_mesh(radius=0.1, h_elem=0.025)
    v0 = mesh.vertices[mesh.triangles[:, 0]]
    v1 = mesh.vertices[mesh.triangles[:, 1]]
    v2 = mesh.vertices[mesh.triangles[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    centroids = (v0 + v1 + v2) / 3.0
    dots = np.einsum("ij,ij->i", normals, centroids)
    assert np.all(dots > 0), f"{(dots <= 0).sum()} inward normals on sphere"


def test_make_sphere_mesh_scales():
    coarse, _ = make_sphere_mesh(radius=0.1, h_elem=0.04)
    fine, _ = make_sphere_mesh(radius=0.1, h_elem=0.02)
    # Halving h_elem ≈ 4× by area, but gmsh OCC isn't exact; require ≥ 2×.
    assert len(fine.triangles) >= 2 * len(coarse.triangles)


def test_make_box_mesh_watertight():
    mesh, bc = make_box_mesh(width=0.2, height=0.15, depth=0.10, h_elem=0.03)
    ok, count, _ = check_watertight(mesh)
    assert ok, f"Box mesh not watertight: {count} open edges"


def test_make_box_mesh_bc_sound_hard():
    _, bc = make_box_mesh(width=0.2, height=0.15, depth=0.10, h_elem=0.03)
    assert len(bc.vibrating_groups) == 0
    assert 1 in bc.sound_hard_groups


def test_make_box_mesh_outward_normals():
    mesh, _ = make_box_mesh(width=0.2, height=0.15, depth=0.10, h_elem=0.03)
    center = np.array([0.1, 0.075, 0.05])
    v0 = mesh.vertices[mesh.triangles[:, 0]]
    v1 = mesh.vertices[mesh.triangles[:, 1]]
    v2 = mesh.vertices[mesh.triangles[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    centroids = (v0 + v1 + v2) / 3.0
    dots = np.einsum("ij,ij->i", normals, centroids - center)
    frac = (dots > 0).mean()
    assert frac > 0.95, f"Only {frac:.1%} outward normals on box"


# ===========================================================================
# 4. Assembly (no NumCalc)
# ===========================================================================


def test_assemble_single_driver_watertight():
    drv = DriverSpec(center=(0.1, 0.075, 0.0), normal=(0, 0, -1), radius=0.03)
    mesh, _ = assemble_box_driver(width=0.2, height=0.15, depth=0.10, drivers=[drv], h_elem=0.015)
    ok, count, msg = check_watertight(mesh)
    assert ok, f"Assembled mesh not watertight: {count} open edges. {msg}"


def test_assemble_single_driver_groups():
    drv = DriverSpec(center=(0.1, 0.075, 0.0), normal=(0, 0, -1), radius=0.03)
    mesh, bc = assemble_box_driver(width=0.2, height=0.15, depth=0.10, drivers=[drv], h_elem=0.015)
    assert 1 in bc.vibrating_groups
    assert bc.vibrating_groups[1] == complex(1, 0)
    assert 2 not in bc.vibrating_groups  # shell is sound-hard


def test_assemble_driver_elements_contiguous():
    drv = DriverSpec(center=(0.1, 0.075, 0.0), normal=(0, 0, -1), radius=0.03)
    mesh, _ = assemble_box_driver(width=0.2, height=0.15, depth=0.10, drivers=[drv], h_elem=0.015)
    _assert_groups_contiguous(mesh)  # must not raise


def test_assemble_contiguity_violation_raises():
    """Non-contiguous group_tags must trigger AssertionError."""
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    triangles = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int32)
    # group_tags 1,2,1,2 — group 1 at indices 0 and 2 (not contiguous)
    group_tags = np.array([1, 2, 1, 2], dtype=np.int32)
    mesh = Mesh(vertices=vertices, triangles=triangles, group_tags=group_tags)
    with pytest.raises(AssertionError, match="not contiguous"):
        _assert_groups_contiguous(mesh)


def test_assemble_driver_elements_before_shell():
    drv = DriverSpec(center=(0.1, 0.075, 0.0), normal=(0, 0, -1), radius=0.03)
    mesh, _ = assemble_box_driver(width=0.2, height=0.15, depth=0.10, drivers=[drv], h_elem=0.015)
    driver_idx = np.where(mesh.group_tags == 1)[0]
    shell_idx = np.where(mesh.group_tags == 2)[0]
    assert len(driver_idx) > 0 and len(shell_idx) > 0
    assert (
        driver_idx[-1] < shell_idx[0]
    ), "Driver elements must precede shell elements for correct NC.inp BC range."


def test_assemble_two_drivers_contiguous():
    drv1 = DriverSpec(center=(0.1, 0.075, 0.0), normal=(0, 0, -1), radius=0.025)
    drv2 = DriverSpec(center=(0.1, 0.075, 0.1), normal=(0, 0, 1), radius=0.025)
    mesh, bc = assemble_box_driver(
        width=0.2, height=0.15, depth=0.10, drivers=[drv1, drv2], h_elem=0.015
    )
    assert set(bc.vibrating_groups.keys()) == {1, 2}
    assert set(np.unique(mesh.group_tags).tolist()) == {1, 2, 3}
    _assert_groups_contiguous(mesh)


def test_assemble_proud_cap_raises():
    drv = DriverSpec(center=(0.1, 0.075, 0.0), normal=(0, 0, -1), radius=0.03, cap_height=0.005)
    with pytest.raises(NotImplementedError, match="cap_height"):
        assemble_box_driver(width=0.2, height=0.15, depth=0.10, drivers=[drv], h_elem=0.015)


# ===========================================================================
# 5. Physics canary — gmsh sphere through V-2 (local_only, needs NumCalc)
# ===========================================================================


@pytest.mark.local_only
def test_gmsh_sphere_v2_gate():
    """V-2 gate through the gmsh extraction path (physics canary).

    Builds a pulsating sphere via make_sphere_mesh (gmsh OCC path) and runs the
    full NumCalc BEM pipeline at 250/500/1000 Hz.  Asserts the same ≤ 0.5 dB
    gate the synthetic icosphere passes, proving the gmsh extraction path is
    solver-equivalent.

    Requires NumCalc binary (BEAMSIM2_NUMCALC_BIN env var).
    """
    from beamsim2.backends.numcalc.adapter import NumCalcBackend
    from beamsim2.core.sphere import lebedev
    from beamsim2.core.types import FrequencyGrid
    from beamsim2.validation.sphere_benchmark import sphere_benchmark_errors

    A = 0.10  # m — sphere radius (same as V-2)
    C = 343.2  # m/s
    RHO = 1.2041  # kg/m³
    H_ELEM = 0.025  # ≈ subdivisions=2 icosphere edge length

    mesh, bc = make_sphere_mesh(radius=A, h_elem=H_ELEM, all_vibrating=True)

    freqs = FrequencyGrid(
        frequencies=np.array([250.0, 500.0, 1000.0], dtype=np.float64),
        spacing="linear",
        fractional_octave=None,
    )
    obs = lebedev(26, radius=1.0)
    config = SolverConfig(n_epw=6, speed_of_sound=C, air_density=RHO, burton_miller=False)

    backend = NumCalcBackend()
    spec = backend.prepare(mesh, bc, freqs, obs, config)
    raw = backend.solve(spec)
    field = backend.extract(raw, obs)

    assert (
        field.convergence_flags.all()
    ), f"V-2 canary: NumCalc did not converge at all steps. Flags: {field.convergence_flags}"

    result = sphere_benchmark_errors(field.pressure, field.frequencies, obs, a=A, c=C, rho=RHO)

    assert result["passed"], (
        f"V-2 canary FAILED: mag error = {result['mean_mag_db']} dB "
        f"(threshold 0.5 dB), phase = {result['mean_phase_deg']} deg (threshold 5 deg). "
        "gmsh sphere path diverges from analytic — check mesh resolution or extraction."
    )
