"""BeamSimII project save / load: .bsim JSON file format.

A ``.bsim`` file stores the *input* state needed to reproduce a session:
box geometry, drivers (with T/S models), simulation parameters, solver
config, and (optionally) the path to a previously saved HDF5 results file.
It does NOT embed the H-tensor — that always lives in a separate .h5 file.

Schema version 1 (``project_version`` in the file header).  Bump only when
the on-disk structure changes in a breaking way; see CHANGELOG.md.

This module is **Qt-free**: it can be imported and unit-tested without a
display.  The GUI (``gui.app.MainWindow._gather_state`` / ``_apply_state``)
owns the widget ↔ dict translation; this module owns the dict ↔ JSON file
translation and the driver-level (de)serialisation.

Notes
-----
- Tuple fields (``DriverSpec.center/normal``, ``reference_axis``) become JSON
  arrays; coerced back to ``tuple[float, float, float]`` on load.
- ``TerminalModel.to_attrs()`` is intentionally **not** used — it is an HDF5
  metadata helper and is lossy (flattens inductance to a string).
- The inductance union (``PlainLe`` | ``LR2Ladder``) is distinguished only by
  ``isinstance`` at runtime; a ``"kind"`` discriminator field is written into
  the JSON so the type can be restored unambiguously.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from beamsim2.driver.inductance import LR2Ladder, PlainLe
from beamsim2.driver.terminal import TerminalModel
from beamsim2.driver.thiele_small import TSParams
from beamsim2.geometry.assemble import DriverSpec
from beamsim2.geometry.faces import FacePlacement
from beamsim2.pipeline.run import DriverPlacement

PROJECT_SCHEMA = "beamsim2.project"
PROJECT_VERSION = 1


# ---------------------------------------------------------------------------
# Primitive (de)serialisers
# ---------------------------------------------------------------------------


def _escape_str_toml(s: str) -> str:
    """Escape backslashes and double-quotes for a TOML basic string."""
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _ts_to_dict(ts: TSParams) -> dict[str, float]:
    """Serialise the six fundamental Thiele/Small primitives.

    Derived quantities (fs, Qms, Qes, Qts, vas) are ``@property`` on
    ``TSParams`` and are intentionally **not** stored — they would be
    redundant and would create a round-trip ambiguity.

    Parameters
    ----------
    ts : TSParams
        Thiele/Small parameter set.

    Returns
    -------
    dict
        JSON-safe dict with keys Re, Bl, Mms, Cms, Rms, Sd.
    """
    return {
        "Re": ts.Re,
        "Bl": ts.Bl,
        "Mms": ts.Mms,
        "Cms": ts.Cms,
        "Rms": ts.Rms,
        "Sd": ts.Sd,
    }


def _ts_from_dict(d: dict) -> TSParams:
    """Restore a ``TSParams`` from its serialised dict.

    Parameters
    ----------
    d : dict
        A dict produced by ``_ts_to_dict``.

    Returns
    -------
    TSParams
    """
    return TSParams(
        Re=float(d["Re"]),
        Bl=float(d["Bl"]),
        Mms=float(d["Mms"]),
        Cms=float(d["Cms"]),
        Rms=float(d["Rms"]),
        Sd=float(d["Sd"]),
    )


def _inductance_to_dict(ind: PlainLe | LR2Ladder) -> dict[str, Any]:
    """Serialise the inductance model, adding an explicit ``"kind"`` discriminator.

    The discriminator is required because ``PlainLe`` and ``LR2Ladder`` share the
    ``Le`` field and are otherwise indistinguishable in JSON.

    Parameters
    ----------
    ind : PlainLe or LR2Ladder
        The voice-coil inductance model.

    Returns
    -------
    dict
        JSON-safe dict with at minimum ``{"kind": ..., "Le": ...}``.

    Raises
    ------
    TypeError
        If ``ind`` is not a recognised inductance type.
    """
    if isinstance(ind, LR2Ladder):
        return {"kind": "LR2Ladder", "Le": ind.Le, "Le2": ind.Le2, "Re2": ind.Re2}
    if isinstance(ind, PlainLe):
        return {"kind": "PlainLe", "Le": ind.Le}
    raise TypeError(f"Unknown inductance type: {type(ind).__name__}")


def _inductance_from_dict(d: dict) -> PlainLe | LR2Ladder:
    """Restore an inductance model from its serialised dict.

    Parameters
    ----------
    d : dict
        A dict produced by ``_inductance_to_dict``.

    Returns
    -------
    PlainLe or LR2Ladder

    Raises
    ------
    ValueError
        If the ``"kind"`` field is missing or unrecognised.
    """
    kind = d.get("kind")
    if kind == "LR2Ladder":
        return LR2Ladder(Le=float(d["Le"]), Le2=float(d["Le2"]), Re2=float(d["Re2"]))
    if kind == "PlainLe":
        return PlainLe(Le=float(d["Le"]))
    raise ValueError(f"Unknown inductance kind {kind!r}. " "Expected 'PlainLe' or 'LR2Ladder'.")


def _terminal_to_dict(t: TerminalModel) -> dict[str, Any]:
    """Serialise a ``TerminalModel`` to a JSON-safe dict.

    Parameters
    ----------
    t : TerminalModel
        The terminal model to serialise.

    Returns
    -------
    dict
        JSON-safe dict with keys ts, inductance, box_volume, voltage, name.
    """
    return {
        "ts": _ts_to_dict(t.ts),
        "inductance": _inductance_to_dict(t.inductance),
        "box_volume": t.box_volume,  # float or None — both JSON-safe
        "voltage": t.voltage,
        "name": t.name,
    }


def _terminal_from_dict(d: dict) -> TerminalModel:
    """Restore a ``TerminalModel`` from its serialised dict.

    Parameters
    ----------
    d : dict
        A dict produced by ``_terminal_to_dict``.

    Returns
    -------
    TerminalModel
    """
    bv = d.get("box_volume")
    return TerminalModel(
        ts=_ts_from_dict(d["ts"]),
        inductance=_inductance_from_dict(d["inductance"]),
        box_volume=float(bv) if bv is not None else None,
        voltage=float(d.get("voltage", 2.83)),
        name=str(d.get("name", "driver")),
    )


def _spec_to_dict(s: DriverSpec) -> dict[str, Any]:
    """Serialise a ``DriverSpec``.  Tuples → lists (JSON array).

    Parameters
    ----------
    s : DriverSpec
        The geometric driver descriptor.

    Returns
    -------
    dict
        JSON-safe dict with keys center, normal, radius, cap_height.
    """
    return {
        "center": list(s.center),
        "normal": list(s.normal),
        "radius": s.radius,
        "cap_height": s.cap_height,
    }


def _spec_from_dict(d: dict) -> DriverSpec:
    """Restore a ``DriverSpec``, coercing JSON arrays back to tuples.

    Parameters
    ----------
    d : dict
        A dict produced by ``_spec_to_dict``.

    Returns
    -------
    DriverSpec
    """
    return DriverSpec(
        center=tuple(float(v) for v in d["center"]),  # type: ignore[arg-type]
        normal=tuple(float(v) for v in d["normal"]),  # type: ignore[arg-type]
        radius=float(d["radius"]),
        cap_height=float(d.get("cap_height", 0.0)),
    )


def _face_placement_to_dict(fp: FacePlacement) -> dict[str, Any]:
    """Serialise a ``FacePlacement``.

    Parameters
    ----------
    fp : FacePlacement
        The GUI source-of-truth driver placement.

    Returns
    -------
    dict
        JSON-safe dict with keys face_id, u, v, radius.
    """
    return {"face_id": fp.face_id, "u": fp.u, "v": fp.v, "radius": fp.radius}


def _face_placement_from_dict(d: dict) -> FacePlacement:
    """Restore a ``FacePlacement`` from its serialised dict.

    Parameters
    ----------
    d : dict
        A dict produced by ``_face_placement_to_dict``.

    Returns
    -------
    FacePlacement
    """
    return FacePlacement(
        face_id=int(d["face_id"]),
        u=float(d["u"]),
        v=float(d["v"]),
        radius=float(d["radius"]),
    )


# ---------------------------------------------------------------------------
# Public driver (de)serialisers
# ---------------------------------------------------------------------------


def driver_to_dict(dp: DriverPlacement) -> dict[str, Any]:
    """Serialise one ``DriverPlacement`` to a JSON-safe dict.

    Handles the inductance-type discriminator, tuple→list coercion,
    ``box_volume=None``, and optional ``face_placement``.

    Parameters
    ----------
    dp : DriverPlacement
        The driver placement to serialise.

    Returns
    -------
    dict
        JSON-safe dictionary representation suitable for embedding in the
        project document ``"drivers"`` list.
    """
    return {
        "driver_id": dp.driver_id,
        "spec": _spec_to_dict(dp.spec),
        "face_placement": (
            _face_placement_to_dict(dp.face_placement) if dp.face_placement is not None else None
        ),
        "terminal": (_terminal_to_dict(dp.terminal) if dp.terminal is not None else None),
    }


def driver_from_dict(d: dict) -> DriverPlacement:
    """Deserialise one ``DriverPlacement`` from a dict.

    Restores tuples from JSON arrays; resolves the inductance-kind discriminator.

    Parameters
    ----------
    d : dict
        A dict produced by ``driver_to_dict``.

    Returns
    -------
    DriverPlacement
    """
    fp_raw = d.get("face_placement")
    fp = _face_placement_from_dict(fp_raw) if fp_raw is not None else None

    tm_raw = d.get("terminal")
    tm = _terminal_from_dict(tm_raw) if tm_raw is not None else None

    return DriverPlacement(
        spec=_spec_from_dict(d["spec"]),
        terminal=tm,
        driver_id=str(d["driver_id"]),
        face_placement=fp,
    )


# ---------------------------------------------------------------------------
# Whole-document file I/O
# ---------------------------------------------------------------------------


def document_to_json(doc: dict, path: str | Path) -> None:
    """Write a project document dict to a ``.bsim`` JSON file.

    The document must have been produced by ``MainWindow._gather_state()``
    which inserts the schema/version header and all required top-level keys.

    Parameters
    ----------
    doc : dict
        Project document dict; ``"schema"`` and ``"project_version"`` keys
        must already be present.
    path : str or Path
        Destination file path (parent directories are created if needed;
        file is created or overwritten).
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")


def document_from_json(path: str | Path) -> dict:
    """Read a ``.bsim`` project file and return its document dict.

    Validates the schema marker and version before returning.

    Parameters
    ----------
    path : str or Path
        Path to a ``.bsim`` file.

    Returns
    -------
    dict
        The project document dict.  Keys are as produced by
        ``MainWindow._gather_state()``.

    Raises
    ------
    ValueError
        If the file is not a valid BeamSimII project file (wrong schema
        marker, missing version, or unsupported version number).
    json.JSONDecodeError
        If the file is not valid JSON.
    FileNotFoundError
        If the file does not exist.
    """
    p = Path(path)
    raw = p.read_text(encoding="utf-8")
    doc = json.loads(raw)

    schema = doc.get("schema")
    if schema != PROJECT_SCHEMA:
        raise ValueError(
            f"Not a BeamSimII project file: expected schema={PROJECT_SCHEMA!r}, "
            f"got {schema!r} in {p.name}"
        )

    v = doc.get("project_version")
    if v != PROJECT_VERSION:
        raise ValueError(
            f"Unsupported project file version {v} "
            f"(this build supports version {PROJECT_VERSION}). "
            f"File: {p.name}"
        )

    return doc
