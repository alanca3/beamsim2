"""Tests for NC.inp file generation.

Pure-Python — no NumCalc binary required. Verifies that normalised
Mesh / BoundaryConditions / FrequencyGrid round-trip to valid NumCalc
input text, and that the boundary-condition emission covers exactly the
vibrating elements (no BC leak onto adjacent rigid elements).
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from beamsim2.backends.numcalc.ncinp_writer import (
    _group_element_runs,
    write_mesh_files,
    write_nc_inp,
)
from beamsim2.core.sphere import lebedev
from beamsim2.core.types import (
    BoundaryConditions,
    FrequencyGrid,
    Mesh,
    ObservationPoints,
    SolverConfig,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _simple_mesh(group_tags: list[int]) -> Mesh:
    """Build a minimal valid Mesh with the given per-element group tags.

    Uses small non-degenerate triangles (no vertex sharing). Geometry
    coordinates don't affect writer unit tests.
    """
    n_tri = len(group_tags)
    verts = np.zeros((n_tri * 3, 3), dtype=np.float64)
    for i in range(n_tri):
        verts[i * 3 + 0] = [float(i), 0.0, 0.0]
        verts[i * 3 + 1] = [float(i), 1.0, 0.0]
        verts[i * 3 + 2] = [float(i), 0.0, 1.0]
    tris = np.array([[i * 3, i * 3 + 1, i * 3 + 2] for i in range(n_tri)], dtype=np.int32)
    return Mesh(
        vertices=verts,
        triangles=tris,
        group_tags=np.array(group_tags, dtype=np.int32),
    )


def _minimal_obs() -> ObservationPoints:
    return lebedev(n_points=6, radius=1.0)


def _minimal_bc(tag: int = 1, vel: complex = 1.0 + 0j) -> BoundaryConditions:
    return BoundaryConditions(vibrating_groups={tag: vel})


def _minimal_freqs() -> FrequencyGrid:
    return FrequencyGrid(
        frequencies=np.array([250.0, 500.0]),
        spacing="fractional-octave",
        fractional_octave=1 / 3,
    )


def _minimal_config() -> SolverConfig:
    return SolverConfig()


# ---------------------------------------------------------------------------
# _group_element_runs
# ---------------------------------------------------------------------------


def test_group_element_runs_all_same_tag():
    """All elements share one tag → single run covering everything."""
    mesh = _simple_mesh([1, 1, 1, 1])
    runs = _group_element_runs(mesh, tag=1)
    assert runs == [(0, 3)], f"Expected [(0, 3)], got {runs}"


def test_group_element_runs_two_disjoint_runs():
    """Tag 1 at positions 0-1 and 4-5, with tag 2 at 2-3 in between."""
    mesh = _simple_mesh([1, 1, 2, 2, 1, 1])
    runs = _group_element_runs(mesh, tag=1)
    assert runs == [(0, 1), (4, 5)], f"Expected [(0,1),(4,5)], got {runs}"


def test_group_element_runs_single_element():
    """Single matching element → one singleton run."""
    mesh = _simple_mesh([2, 1, 2])
    runs = _group_element_runs(mesh, tag=1)
    assert runs == [(1, 1)], f"Expected [(1,1)], got {runs}"


def test_group_element_runs_missing_tag_raises():
    mesh = _simple_mesh([1, 1, 1])
    with pytest.raises(ValueError, match="group_tag=99"):
        _group_element_runs(mesh, tag=99)


def test_group_element_runs_three_runs_no_overlap():
    """Three separate runs — none should overlap."""
    # tags: [1 1 2 1 2 2 1] → runs for 1: (0,1), (3,3), (6,6)
    mesh = _simple_mesh([1, 1, 2, 1, 2, 2, 1])
    runs = _group_element_runs(mesh, tag=1)
    assert runs == [(0, 1), (3, 3), (6, 6)], f"Unexpected runs: {runs}"
    for i, (lo1, hi1) in enumerate(runs):
        for j, (lo2, hi2) in enumerate(runs):
            if i != j:
                assert hi1 < lo2 or hi2 < lo1, f"Runs {runs[i]} and {runs[j]} overlap"


# ---------------------------------------------------------------------------
# write_nc_inp — BOUNDARY section (BC leak test)
# ---------------------------------------------------------------------------


def test_write_nc_inp_boundary_contiguous_single_run():
    """Contiguous driver group → exactly one ELEM line with correct lo/hi."""
    mesh = _simple_mesh([1, 1, 1, 2, 2])  # driver = 0..2, shell = 3..4
    bc = _minimal_bc(tag=1)
    freqs = _minimal_freqs()
    obs = _minimal_obs()
    config = _minimal_config()

    with tempfile.TemporaryDirectory() as tmpdir:
        counts = write_mesh_files(tmpdir, mesh, obs)
        nc_path = write_nc_inp(tmpdir, mesh, bc, freqs, config, counts)
        with open(nc_path) as f:
            text = f.read()

    elem_lines = [ln.strip() for ln in text.splitlines() if ln.strip().startswith("ELEM ")]
    assert len(elem_lines) == 1, f"Expected 1 ELEM line, got {elem_lines}"
    assert "0 TO 2" in elem_lines[0], f"Expected 'ELEM 0 TO 2 ...', got {elem_lines[0]}"


def test_write_nc_inp_boundary_non_contiguous_no_leak():
    """Non-contiguous driver group → two ELEM lines; rigid elements never covered."""
    # Driver at 0-1 and 4-5; shell (rigid) at 2-3
    mesh = _simple_mesh([1, 1, 2, 2, 1, 1])
    bc = _minimal_bc(tag=1)
    freqs = _minimal_freqs()
    obs = _minimal_obs()
    config = _minimal_config()

    with tempfile.TemporaryDirectory() as tmpdir:
        counts = write_mesh_files(tmpdir, mesh, obs)
        nc_path = write_nc_inp(tmpdir, mesh, bc, freqs, config, counts)
        with open(nc_path) as f:
            text = f.read()

    elem_lines = [ln.strip() for ln in text.splitlines() if ln.strip().startswith("ELEM ")]
    assert len(elem_lines) == 2, f"Expected 2 ELEM lines, got {elem_lines}"

    assert "0 TO 1" in elem_lines[0], f"First ELEM line wrong: {elem_lines[0]}"
    assert "4 TO 5" in elem_lines[1], f"Second ELEM line wrong: {elem_lines[1]}"

    # Rigid elements 2-3 must NOT appear anywhere in any ELEM line's range.
    for line in elem_lines:
        tokens = line.split()
        lo = int(tokens[1])
        hi = int(tokens[3])
        for rigid_idx in [2, 3]:
            assert not (
                lo <= rigid_idx <= hi
            ), f"Rigid element {rigid_idx} is covered by ELEM line: {line}"


def test_write_nc_inp_velo_values_in_boundary():
    """Complex velocity encodes real and imaginary parts in the ELEM line."""
    mesh = _simple_mesh([1, 1])
    bc = BoundaryConditions(vibrating_groups={1: 0.5 + 0.25j})
    freqs = _minimal_freqs()
    obs = _minimal_obs()
    config = _minimal_config()

    with tempfile.TemporaryDirectory() as tmpdir:
        counts = write_mesh_files(tmpdir, mesh, obs)
        nc_path = write_nc_inp(tmpdir, mesh, bc, freqs, config, counts)
        with open(nc_path) as f:
            text = f.read()

    elem_line = next(ln for ln in text.splitlines() if ln.strip().startswith("ELEM "))
    # Format: ELEM lo TO hi VELO <re> -1 <im> -1
    assert "5.000000e-01" in elem_line, f"Real part missing in: {elem_line}"
    assert "2.500000e-01" in elem_line, f"Imag part missing in: {elem_line}"


# ---------------------------------------------------------------------------
# write_nc_inp — structural checks
# ---------------------------------------------------------------------------


def test_write_nc_inp_element_and_node_counts():
    """n_total_elems and n_total_nodes in Main Parameters I are correct."""
    mesh = _simple_mesh([1, 1, 2, 2])
    bc = _minimal_bc(tag=1)
    freqs = _minimal_freqs()
    obs = _minimal_obs()
    config = _minimal_config()

    with tempfile.TemporaryDirectory() as tmpdir:
        counts = write_mesh_files(tmpdir, mesh, obs)
        nc_path = write_nc_inp(tmpdir, mesh, bc, freqs, config, counts)
        with open(nc_path) as f:
            text = f.read()

    n_total_nodes = len(mesh.vertices) + counts.n_eval_nodes
    n_total_elems = len(mesh.triangles) + counts.n_eval_elems

    # Main Parameters I line: "2 {n_elems} {n_nodes} 0 0 2 1 0 0"
    params_line = next(
        ln for ln in text.splitlines() if ln.startswith(f"2 {n_total_elems} {n_total_nodes}")
    )
    assert (
        params_line is not None
    ), f"Main Parameters I line not found; expected '2 {n_total_elems} {n_total_nodes} ...'"


def test_write_nc_inp_frequency_curve_row_count():
    """Load Frequency Curve has n_freq + 1 rows (row 0 is the zero sentinel)."""
    mesh = _simple_mesh([1])
    bc = _minimal_bc(tag=1)
    freqs = FrequencyGrid(
        frequencies=np.array([100.0, 200.0, 400.0]),
        spacing="fractional-octave",
        fractional_octave=1 / 3,
    )
    obs = _minimal_obs()
    config = _minimal_config()

    with tempfile.TemporaryDirectory() as tmpdir:
        counts = write_mesh_files(tmpdir, mesh, obs)
        nc_path = write_nc_inp(tmpdir, mesh, bc, freqs, config, counts)
        with open(nc_path) as f:
            text = f.read()

    lines = text.splitlines()
    curve_header_idx = next(i for i, ln in enumerate(lines) if "## Load Frequency Curve" in ln)
    count_line = lines[curve_header_idx + 1].strip()
    n_rows_str = count_line.split()[1]
    assert int(n_rows_str) == 4, f"Expected 4 rows (3 freqs + sentinel), got {n_rows_str}"


def test_write_nc_inp_required_sections_present():
    """NC.inp contains all required section keywords in order."""
    mesh = _simple_mesh([1, 2])
    bc = _minimal_bc(tag=1)
    freqs = _minimal_freqs()
    obs = _minimal_obs()
    config = _minimal_config()

    with tempfile.TemporaryDirectory() as tmpdir:
        counts = write_mesh_files(tmpdir, mesh, obs)
        nc_path = write_nc_inp(tmpdir, mesh, bc, freqs, config, counts)
        with open(nc_path) as f:
            text = f.read()

    required = [
        "Controlparameter I",
        "Controlparameter II",
        "Load Frequency Curve",
        "1. Main Parameters I",
        "NODES",
        "ELEMENTS",
        "BOUNDARY",
        "RETU",
        "POST PROCESS",
        "END",
    ]
    for kw in required:
        assert kw in text, f"Required keyword '{kw}' missing from NC.inp"


# ---------------------------------------------------------------------------
# write_nc_inp — NotImplementedError guards
# ---------------------------------------------------------------------------


def test_write_nc_inp_raises_on_multiple_vibrating_groups():
    """More than one vibrating group raises NotImplementedError (deferred to item 7)."""
    mesh = _simple_mesh([1, 1, 2, 2])
    bc = BoundaryConditions(vibrating_groups={1: 1.0 + 0j, 2: 0.5 + 0j})
    freqs = _minimal_freqs()
    obs = _minimal_obs()
    config = _minimal_config()

    with tempfile.TemporaryDirectory() as tmpdir:
        counts = write_mesh_files(tmpdir, mesh, obs)
        with pytest.raises(NotImplementedError):
            write_nc_inp(tmpdir, mesh, bc, freqs, config, counts)


def test_write_nc_inp_raises_on_ndarray_velocity():
    """Per-element ndarray velocity raises NotImplementedError (deferred to item 8)."""
    mesh = _simple_mesh([1, 1])
    per_elem_vel = np.array([1.0 + 0j, 0.9 + 0j])
    bc = BoundaryConditions(vibrating_groups={1: per_elem_vel})
    freqs = _minimal_freqs()
    obs = _minimal_obs()
    config = _minimal_config()

    with tempfile.TemporaryDirectory() as tmpdir:
        counts = write_mesh_files(tmpdir, mesh, obs)
        with pytest.raises(NotImplementedError):
            write_nc_inp(tmpdir, mesh, bc, freqs, config, counts)


# ---------------------------------------------------------------------------
# write_mesh_files — structural checks
# ---------------------------------------------------------------------------


def test_write_mesh_files_creates_expected_dirs_and_files():
    """write_mesh_files creates ObjectMesh/ and EvalGrid/ with required files."""
    mesh = _simple_mesh([1, 2])
    obs = _minimal_obs()

    with tempfile.TemporaryDirectory() as tmpdir:
        write_mesh_files(tmpdir, mesh, obs)
        for relpath in [
            os.path.join("ObjectMesh", "Nodes.txt"),
            os.path.join("ObjectMesh", "Elements.txt"),
            os.path.join("EvalGrid", "Nodes.txt"),
            os.path.join("EvalGrid", "Elements.txt"),
        ]:
            assert os.path.isfile(os.path.join(tmpdir, relpath)), f"Missing {relpath}"


def test_write_mesh_files_node_count_header():
    """Nodes.txt first line matches the actual number of nodes written."""
    mesh = _simple_mesh([1, 1, 2])
    obs = _minimal_obs()

    with tempfile.TemporaryDirectory() as tmpdir:
        write_mesh_files(tmpdir, mesh, obs)
        with open(os.path.join(tmpdir, "ObjectMesh", "Nodes.txt")) as f:
            lines = f.read().splitlines()
        header_count = int(lines[0])
        data_lines = [ln for ln in lines[1:] if ln.strip()]
        assert header_count == len(
            data_lines
        ), f"Header says {header_count} nodes but {len(data_lines)} data lines"


def test_write_mesh_files_eval_node_base_offset():
    """EvalGrid node IDs start at 200_000 (the _EVAL_NODE_BASE constant)."""
    mesh = _simple_mesh([1])
    obs = _minimal_obs()

    with tempfile.TemporaryDirectory() as tmpdir:
        write_mesh_files(tmpdir, mesh, obs)
        with open(os.path.join(tmpdir, "EvalGrid", "Nodes.txt")) as f:
            lines = f.read().splitlines()
        first_id = int(lines[1].split()[0])
        assert first_id == 200_000, f"EvalGrid base node ID expected 200000, got {first_id}"
