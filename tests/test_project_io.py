"""Tests for io.project_io: .bsim round-trip, schema validation, inductance union.

Pure Python — no Qt, no NumCalc, no display required.  Runs in CI.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from beamsim2.driver.inductance import LR2Ladder, PlainLe
from beamsim2.driver.terminal import TerminalModel
from beamsim2.driver.thiele_small import TSParams
from beamsim2.geometry.assemble import DriverSpec
from beamsim2.geometry.faces import FacePlacement
from beamsim2.io.project_io import (
    PROJECT_SCHEMA,
    PROJECT_VERSION,
    document_from_json,
    document_to_json,
    driver_from_dict,
    driver_to_dict,
)
from beamsim2.pipeline.run import DriverPlacement

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_TS = TSParams(Re=6.0, Bl=7.0, Mms=0.012, Cms=8e-4, Rms=1.0, Sd=0.0133)
_FP = FacePlacement(face_id=0, u=0.02, v=-0.01, radius=0.03)
_SPEC = DriverSpec(
    center=(0.06, 0.05, 0.08),
    normal=(0.0, 0.0, 1.0),
    radius=0.03,
    cap_height=0.002,
)


def _make_dp_plain() -> DriverPlacement:
    return DriverPlacement(
        spec=_SPEC,
        terminal=TerminalModel(
            ts=_TS,
            inductance=PlainLe(Le=0.5e-3),
            box_volume=0.005,
            voltage=2.83,
            name="woofer",
        ),
        driver_id="woofer_0",
        face_placement=_FP,
    )


def _make_dp_lr2() -> DriverPlacement:
    return DriverPlacement(
        spec=_SPEC,
        terminal=TerminalModel(
            ts=_TS,
            inductance=LR2Ladder(Le=0.5e-3, Le2=0.2e-3, Re2=3.0),
            box_volume=None,  # free-air
            voltage=2.83,
            name="tweeter",
        ),
        driver_id="tweeter_0",
        face_placement=None,
    )


# ---------------------------------------------------------------------------
# driver_to_dict / driver_from_dict round-trips
# ---------------------------------------------------------------------------


def test_driver_plain_le_round_trip():
    """PlainLe inductance survives driver_to_dict → driver_from_dict."""
    dp = _make_dp_plain()
    d = driver_to_dict(dp)
    assert d["terminal"]["inductance"]["kind"] == "PlainLe"
    dp2 = driver_from_dict(d)

    assert dp2.driver_id == "woofer_0"
    assert isinstance(dp2.terminal.inductance, PlainLe)
    assert dp2.terminal.inductance.Le == pytest.approx(0.5e-3)
    assert dp2.terminal.box_volume == pytest.approx(0.005)
    assert dp2.terminal.name == "woofer"
    # Tuples restored correctly from JSON lists
    assert isinstance(dp2.spec.center, tuple)
    assert dp2.spec.center == pytest.approx((0.06, 0.05, 0.08))
    assert isinstance(dp2.spec.normal, tuple)
    assert dp2.spec.normal == pytest.approx((0.0, 0.0, 1.0))
    assert dp2.spec.radius == pytest.approx(0.03)
    assert dp2.spec.cap_height == pytest.approx(0.002)
    # FacePlacement preserved
    assert dp2.face_placement is not None
    assert dp2.face_placement.face_id == 0
    assert dp2.face_placement.u == pytest.approx(0.02)
    assert dp2.face_placement.v == pytest.approx(-0.01)
    assert dp2.face_placement.radius == pytest.approx(0.03)


def test_driver_lr2ladder_round_trip():
    """LR2Ladder inductance survives driver_to_dict → driver_from_dict."""
    dp = _make_dp_lr2()
    d = driver_to_dict(dp)
    assert d["terminal"]["inductance"]["kind"] == "LR2Ladder"
    dp2 = driver_from_dict(d)

    assert isinstance(dp2.terminal.inductance, LR2Ladder)
    ind = dp2.terminal.inductance
    assert ind.Le == pytest.approx(0.5e-3)
    assert ind.Le2 == pytest.approx(0.2e-3)
    assert ind.Re2 == pytest.approx(3.0)
    # box_volume=None preserved
    assert dp2.terminal.box_volume is None
    # face_placement=None preserved
    assert dp2.face_placement is None


def test_driver_no_terminal():
    """terminal=None round-trips to None."""
    dp = DriverPlacement(
        spec=_SPEC,
        terminal=None,
        driver_id="bare_0",
        face_placement=None,
    )
    d = driver_to_dict(dp)
    assert d["terminal"] is None
    dp2 = driver_from_dict(d)
    assert dp2.terminal is None
    assert dp2.driver_id == "bare_0"


def test_ts_params_only_primitives_stored():
    """Only the six fundamental TSParams fields appear in the dict (no Qts etc)."""
    dp = _make_dp_plain()
    d = driver_to_dict(dp)
    ts_dict = d["terminal"]["ts"]
    assert set(ts_dict.keys()) == {"Re", "Bl", "Mms", "Cms", "Rms", "Sd"}
    # Derived properties are NOT present
    assert "fs" not in ts_dict
    assert "Qts" not in ts_dict


def test_multi_driver_list():
    """Multiple drivers serialise and restore in order."""
    dps = [_make_dp_plain(), _make_dp_lr2()]
    restored = [driver_from_dict(driver_to_dict(dp)) for dp in dps]
    assert restored[0].driver_id == "woofer_0"
    assert restored[1].driver_id == "tweeter_0"
    assert isinstance(restored[0].terminal.inductance, PlainLe)
    assert isinstance(restored[1].terminal.inductance, LR2Ladder)


def test_bad_inductance_kind_raises():
    """Unknown inductance kind raises a clear ValueError."""
    d = driver_to_dict(_make_dp_plain())
    d["terminal"]["inductance"]["kind"] = "FancyLe"
    with pytest.raises(ValueError, match="Unknown inductance kind"):
        driver_from_dict(d)


# ---------------------------------------------------------------------------
# document_to_json / document_from_json round-trips
# ---------------------------------------------------------------------------


def _make_document() -> dict:
    return {
        "schema": PROJECT_SCHEMA,
        "project_version": PROJECT_VERSION,
        "box": {"width": 0.12, "height": 0.10, "depth": 0.08, "fillet_radius": 0.005},
        "reference_axis": [0.0, 0.0, 1.0],
        "drivers": [driver_to_dict(_make_dp_plain()), driver_to_dict(_make_dp_lr2())],
        "simulation": {
            "f_lo": 100.0,
            "f_hi": 2000.0,
            "frac_oct": 1 / 12,
            "sphere_scheme": "lebedev",
            "sphere_n_points": 26,
            "sphere_radius": 1.0,
            "output_path": "",
        },
        "solver_config": {
            "n_epw": 6,
            "tolerance": 1e-6,
            "max_iterations": 1000,
            "burton_miller": True,
            "speed_of_sound": 343.2,
            "air_density": 1.2041,
            "air_attenuation_model": "none",
        },
        "results_h5_path": None,
    }


def test_document_write_read_round_trip(tmp_path: Path):
    """document_to_json → document_from_json reproduces all top-level keys."""
    doc = _make_document()
    p = tmp_path / "test.bsim"
    document_to_json(doc, p)
    assert p.exists()

    loaded = document_from_json(p)
    assert loaded["schema"] == PROJECT_SCHEMA
    assert loaded["project_version"] == PROJECT_VERSION
    assert loaded["box"]["width"] == pytest.approx(0.12)
    assert loaded["reference_axis"] == pytest.approx([0.0, 0.0, 1.0])
    assert len(loaded["drivers"]) == 2
    assert loaded["simulation"]["f_lo"] == pytest.approx(100.0)
    assert loaded["simulation"]["sphere_n_points"] == 26
    assert loaded["results_h5_path"] is None


def test_document_file_is_valid_json(tmp_path: Path):
    """The .bsim file is valid JSON (parseable without project_io)."""
    p = tmp_path / "test.bsim"
    document_to_json(_make_document(), p)
    data = json.loads(p.read_text())
    assert data["schema"] == PROJECT_SCHEMA


def test_document_wrong_schema_raises(tmp_path: Path):
    """document_from_json raises ValueError on wrong schema marker."""
    doc = _make_document()
    doc["schema"] = "not.a.beamsim.file"
    p = tmp_path / "wrong.bsim"
    document_to_json(doc, p)
    with pytest.raises(ValueError, match="Not a BeamSimII project file"):
        document_from_json(p)


def test_document_wrong_version_raises(tmp_path: Path):
    """document_from_json raises ValueError on unsupported version."""
    doc = _make_document()
    doc["project_version"] = 99
    p = tmp_path / "v99.bsim"
    document_to_json(doc, p)
    with pytest.raises(ValueError, match="Unsupported project file version"):
        document_from_json(p)


def test_driver_round_trip_through_document(tmp_path: Path):
    """Drivers loaded from a saved .bsim restore inductance type correctly."""
    doc = _make_document()
    p = tmp_path / "test.bsim"
    document_to_json(doc, p)
    loaded = document_from_json(p)

    dp0 = driver_from_dict(loaded["drivers"][0])
    dp1 = driver_from_dict(loaded["drivers"][1])
    assert isinstance(dp0.terminal.inductance, PlainLe)
    assert isinstance(dp1.terminal.inductance, LR2Ladder)
    assert dp0.terminal.box_volume == pytest.approx(0.005)
    assert dp1.terminal.box_volume is None
