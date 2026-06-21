"""V-EXPORT — Phase-2 audit export round-trip (Stage P2-3).

The audit-first export writes filtered per-driver / combined ``.frd`` (on H/V polar arcs)
plus the raw weights. The load-bearing guarantee: the exported weights re-load and
reconstruct exactly the designed beam (so the audit artifacts and the design agree).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from beamsim2.assembly.tensor import build_dataset, stacked_h_full
from beamsim2.beamform.design import design
from beamsim2.beamform.targets import TargetSpec
from beamsim2.core.sphere import icosphere
from beamsim2.core.types import ComplexField
from beamsim2.io.filter_export import export_filter_design, load_design_weights, polar_arcs
from beamsim2.validation.closed_loop import monopole_field, steer_response

_C = 343.2


def _design_and_dataset():
    obs = icosphere(4)
    freqs = np.array([500.0, 1000.0, 2000.0])
    pos = 0.05 * np.array([[1, 0, 0], [-1, 0, 0], [0, 0, 1]], float)
    H = monopole_field(pos, obs, freqs, c=_C)
    inputs = [
        (
            f"d{i}",
            ComplexField(frequencies=freqs, pressure=H[i], convergence_flags=np.ones(3, bool)),
            {"name": f"d{i}", "position": list(pos[i])},
        )
        for i in range(3)
    ]
    ds = build_dataset(inputs, obs, root_attrs={"phase_origin": [0, 0, 0], "speed_of_sound": _C})
    r = design(
        ds,
        TargetSpec(
            steer_dir=np.array([0, 0, 1.0]), engine="ls", preset="cardioid", wng_floor_db=-30
        ),
    )
    return ds, r


def test_polar_arcs_through_steer():
    """Both arcs pass through the steer axis at angle 0 and are orthogonal great circles."""
    steer = np.array([0.0, 0.0, 1.0])
    arcs = polar_arcs(steer, step_deg=15.0)
    for plane in ("H", "V"):
        angles, uv = arcs[plane]
        assert np.max(np.abs(np.linalg.norm(uv, axis=1) - 1.0)) < 1e-12
        on_axis = uv[int(np.argmin(np.abs(angles)))]
        assert np.allclose(on_axis, steer, atol=1e-12)
    # H and V sweep orthogonal planes (their +90deg points are orthogonal).
    _, uvh = arcs["H"]
    _, uvv = arcs["V"]
    a90 = int(np.argmin(np.abs(arcs["H"][0] - 90.0)))
    assert abs(uvh[a90] @ uvv[a90]) < 1e-9


def test_export_creates_expected_artifacts(tmp_path):
    ds, r = _design_and_dataset()
    out = export_filter_design(tmp_path / "exp", ds, r, step_deg=15.0)
    assert (out / "weights.npz").exists()
    assert (out / "manifest.csv").exists()
    assert (out / "design.json").exists()
    # 3 drivers x 2 planes x 25 angles + 2 planes x 25 combined = 200 .frd files.
    frd = list(out.rglob("*.frd"))
    assert len(frd) == 3 * 2 * 25 + 2 * 25


def test_weights_roundtrip_reconstructs_beam(tmp_path):
    """V-EXPORT: reloaded weights reconstruct the designed steered field exactly."""
    ds, r = _design_and_dataset()
    out = export_filter_design(tmp_path / "exp", ds, r, step_deg=30.0)
    w = load_design_weights(out / "weights.npz")
    p_reloaded = steer_response(stacked_h_full(ds), w["weights"])
    assert np.max(np.abs(p_reloaded - r.steered_field)) < 1e-12
    assert list(w["driver_ids"]) == [d.driver_id for d in ds.drivers]
    assert w["engine"] == "ls"


def test_frd_rows_match_frequencies(tmp_path):
    """A written .frd has comment lines then one freq/mag/phase row per frequency."""
    ds, r = _design_and_dataset()
    out = export_filter_design(tmp_path / "exp", ds, r, step_deg=30.0)
    sample = next(iter((out / "combined").glob("combined_H_+000.frd")))
    lines = Path(sample).read_text().splitlines()
    comments = [ln for ln in lines if ln.startswith("*")]
    rows = [ln for ln in lines if ln and not ln.startswith("*")]
    assert len(comments) >= 4
    assert len(rows) == len(ds.frequencies)
    freq, mag, phase = rows[0].split("\t")
    assert float(freq) == pytest.approx(500.0)
    assert -200.0 < float(mag) < 200.0
    assert -180.0 <= float(phase) <= 180.0
