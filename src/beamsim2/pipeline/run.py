"""Headless end-to-end pipeline runner: drives Stages A (geometry) through G (export).

``run_simulation`` is the single entry point for the GUI and for the
``@local_only`` end-to-end test.  It is entirely Qt-free; the GUI supplies a
``ProgressModel`` via the optional ``progress`` parameter.

Per-driver solve loop
---------------------
Each of the M drivers is solved **once, independently, at unit cone velocity**,
with all other driver groups and the shell left sound-hard.  This is the
superposition principle that makes Phase-2 filter design cheap: every driver's
``H_bem[m]`` is reusable without re-solving (§2 Stage F, §3.4 cardinal rule).

Phase origin
------------
All M solves use the **same mesh** (same node coordinates, same spatial reference).
So every ``H_bem[m]`` is already referenced to the global phase origin [0,0,0]
with its true time-of-flight phase.  The orchestrator never re-zeros phase.
The V-5 guard (``assert_superposition_matches``) catches any violation.

Build-order item 10 (orchestration layer, §2 Stages A–G, §6 Gameplan).

References
----------
BEAMSIMII_Gameplan.md §2, §3.4, §6.
DATA_CONTRACT.md §3.
"""

from __future__ import annotations

import datetime
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

from beamsim2.assembly.superpose import driver_h_bem
from beamsim2.assembly.tensor import RadiationDataset, build_dataset
from beamsim2.backends.base import BEMBackend
from beamsim2.core.sphere import lebedev
from beamsim2.core.types import (
    BoundaryConditions,
    ComplexField,
    FrequencyGrid,
    ObservationPoints,
    SolverConfig,
)
from beamsim2.driver.terminal import TerminalModel, terminal_responses_for
from beamsim2.geometry.assemble import DriverSpec, assemble_box_driver
from beamsim2.geometry.health import HealthReport
from beamsim2.geometry.mesh import mesh_geometry

if TYPE_CHECKING:
    from beamsim2.pipeline.progress import ProgressModel


# ---------------------------------------------------------------------------
# Request / result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class BoxGeometry:
    """Parametric box enclosure dimensions.

    Only the parametric box is supported today (``geometry.assemble``).
    CAD import (STEP/STL) is a future item (``geometry.import_io`` stub).

    Parameters
    ----------
    width, height, depth : float
        Box dimensions in metres (x, y, z from origin).
    fillet_radius : float
        Edge fillet radius in metres.  0 → sharp corners.
    """

    width: float
    height: float
    depth: float
    fillet_radius: float = 0.0


@dataclass
class DriverPlacement:
    """One driver's geometry, T/S model, and identifier.

    Parameters
    ----------
    spec : DriverSpec
        Geometry: ``center``, ``normal``, ``radius`` (metres).
    terminal : TerminalModel or None
        Electrical chain. ``None`` → terminal_response = ones(F), so
        ``H_full == H_bem`` (useful for pure-BEM analysis without a
        specific driver model).
    driver_id : str
        Unique string key used for the HDF5 group (e.g. ``"woofer_0"``).
    """

    spec: DriverSpec
    terminal: Optional[TerminalModel]
    driver_id: str


@dataclass
class SimulationRequest:
    """Everything needed to run one full headless simulation.

    Parameters
    ----------
    geometry : BoxGeometry
    drivers : list[DriverPlacement]
        M drivers.  Solve order matches list order.
    frequencies : FrequencyGrid
        [F] float64, Hz.
    sphere_n_points : int
        Lebedev quadrature order.  Only {6, 14, 26} are implemented
        (``core.sphere.lebedev``); offering anything else raises
        ``NotImplementedError`` in ``lebedev()``.
    sphere_radius : float
        Observation sphere radius in metres.
    config : SolverConfig
        Solver and medium parameters (n_epw, c, rho, …).
    output_h5 : str or Path or None
        If set, write the native HDF5 output here after assembly.
    export_frd_dir : str or Path or None
        If set, write VituixCAD .frd files to this directory.
    export_sofa_path : str or Path or None
        If set, write a SOFA AES69 GeneralTF file to this path.
    """

    geometry: BoxGeometry
    drivers: list[DriverPlacement]
    frequencies: FrequencyGrid
    sphere_n_points: int = 26
    sphere_radius: float = 1.0
    config: SolverConfig = field(default_factory=SolverConfig)
    output_h5: Optional[str | Path] = None
    export_frd_dir: Optional[str | Path] = None
    export_sofa_path: Optional[str | Path] = None


@dataclass
class ResourceEstimate:
    """Coarse RAM and time forecast for the UI's "Estimate" button.

    Parameters
    ----------
    ram_bytes_per_step : np.ndarray
        [F] float64 per-frequency RAM estimate for ONE driver solve.
        NaN where NumCalc gave no estimate.
    peak_ram_bytes : float
        ``max(ram_bytes_per_step)``, NaN-safe.
    total_wall_seconds : float
        Estimated total wall-clock seconds for M × F steps.  Uses a coarse
        element-count heuristic when NumCalc's time estimate is NaN (which
        it always is today — adapter returns all-NaN).
    n_steps_total : int
        F × M (total solver calls).
    """

    ram_bytes_per_step: np.ndarray  # [F] float64, bytes
    peak_ram_bytes: float
    total_wall_seconds: float
    n_steps_total: int


@dataclass
class SimulationResult:
    """Output of a completed ``run_simulation`` call.

    Parameters
    ----------
    dataset : RadiationDataset
        Assembled H tensor in memory.
    h5_path : Path or None
        Written HDF5 file path (``None`` if ``output_h5`` was not set).
    health : HealthReport
        Geometry health report (repairs logged, problems already raised).
    flagged_frequencies : dict[str, np.ndarray]
        ``driver_id → [F] bool`` where ``True`` = non-converged (after retry).
    work_dirs : dict[str, str]
        ``driver_id → NumCalc scratch directory`` (kept on disk for resumable
        re-runs; the scheduler's checkpointing skips completed steps on restart).
    """

    dataset: RadiationDataset
    h5_path: Optional[Path]
    health: HealthReport
    flagged_frequencies: dict[str, np.ndarray]  # driver_id → [F] bool
    work_dirs: dict[str, str]  # driver_id → work_dir string


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def estimate_resources(
    req: SimulationRequest,
    backend: Optional[BEMBackend] = None,
) -> ResourceEstimate:
    """Coarse RAM and wall-clock forecast without running a full solve.

    Builds the mesh once, calls ``backend.estimate()`` against the full
    all-drivers boundary condition, and scales by M drivers.

    Parameters
    ----------
    req : SimulationRequest
    backend : BEMBackend or None
        ``None`` → construct ``NumCalcBackend()`` lazily.

    Returns
    -------
    ResourceEstimate
    """
    backend = backend or _default_backend()
    geom = req.geometry
    f_max = float(req.frequencies.frequencies.max())

    mesh, bc_all, _health = mesh_geometry(
        width=geom.width,
        height=geom.height,
        depth=geom.depth,
        drivers=[dp.spec for dp in req.drivers],
        config=req.config,
        f_max=f_max,
        fillet_radius=geom.fillet_radius,
    )

    plan = backend.estimate(mesh, bc_all, req.frequencies, req.config)
    # [F] float64 RAM per step, NaN-safe
    ram = plan.ram_bytes_per_step
    peak_ram = float(np.nanmax(ram)) if not np.all(np.isnan(ram)) else math.nan

    # time_seconds_per_step is always NaN today (adapter returns all-NaN).
    # Fall back to a coarse element-count heuristic: assume ~0.5 s/element/step
    # as a rough order-of-magnitude placeholder.  Labeled "approximate" in the UI.
    time_per_step = plan.time_seconds_per_step
    if np.all(np.isnan(time_per_step)):
        n_elements = len(mesh.triangles)
        seconds_each = n_elements * 0.5e-3  # HEURISTIC: 0.5 ms/element, coarse
        time_per_step = np.full(len(req.frequencies.frequencies), seconds_each)

    M = len(req.drivers)
    total_s = float(np.nansum(time_per_step)) * M
    n_steps = len(req.frequencies.frequencies) * M

    return ResourceEstimate(
        ram_bytes_per_step=ram,
        peak_ram_bytes=peak_ram,
        total_wall_seconds=total_s,
        n_steps_total=n_steps,
    )


def run_simulation(
    req: SimulationRequest,
    backend: Optional[BEMBackend] = None,
    progress: Optional["ProgressModel"] = None,
) -> SimulationResult:
    """Run the full Phase-1 pipeline (Stages A–G) headlessly.

    Parameters
    ----------
    req : SimulationRequest
    backend : BEMBackend or None
        ``None`` → construct ``NumCalcBackend()`` lazily.
    progress : ProgressModel or None
        If supplied, mutators are called to report per-step status.
        The GUI bridges this to Qt signals; CI tests pass ``None``.

    Returns
    -------
    SimulationResult
    """
    backend = backend or _default_backend()
    geom = req.geometry
    M = len(req.drivers)
    F = len(req.frequencies.frequencies)
    f_max = float(req.frequencies.frequencies.max())

    # ── Stage A/B/C: geometry + health check + mesh ─────────────────────────
    # Mesh is built ONCE and shared across all M per-driver solves.  This is the
    # foundation of the single-phase-origin rule: all M solves reference the
    # exact same node coordinates, so their H_bem arrays share a common origin.
    mesh, _bc_all, health = mesh_geometry(
        width=geom.width,
        height=geom.height,
        depth=geom.depth,
        drivers=[dp.spec for dp in req.drivers],
        config=req.config,
        f_max=f_max,
        fillet_radius=geom.fillet_radius,
    )
    # mesh.group_tags: 1..M = driver groups, M+1 = shell (sound-hard).
    # bc_all (all vibrating) is only used for estimate, not for per-driver solves.

    obs: ObservationPoints = lebedev(n_points=req.sphere_n_points, radius=req.sphere_radius)
    N = obs.unit_vectors.shape[0]

    # ── Stage D/E: per-driver BEM solves ─────────────────────────────────────
    # For each driver m, build a BoundaryConditions with ONLY group m+1 vibrating;
    # all other groups (including the other drivers' groups and the shell) are
    # sound-hard (the default in BoundaryConditions for unlisted groups).
    driver_inputs: list[tuple[str, ComplexField, dict]] = []
    flagged: dict[str, np.ndarray] = {}  # driver_id → [F] bool (True = non-converged)
    work_dirs: dict[str, str] = {}

    for m, dp in enumerate(req.drivers):
        group_tag = m + 1  # assemble_box_driver assigns tags 1..M, M+1=shell
        bc_m = BoundaryConditions(vibrating_groups={group_tag: complex(1.0, 0.0)})

        if progress is not None:
            progress.driver_started(dp.driver_id, m, M)

        # Stage D: write NC.inp files
        spec = backend.prepare(mesh, bc_m, req.frequencies, obs, req.config)
        work_dirs[dp.driver_id] = spec.work_dir

        # Stage E: solve — inject a progress-wired scheduler if progress is set
        sched = _make_scheduler(backend, progress, m, req.frequencies.frequencies)
        raw = backend.solve(spec, scheduler=sched)

        # Extract complex pressure [F, N] complex128
        field = backend.extract(raw, obs)  # ComplexField

        # ~field.convergence_flags: True = non-converged (i.e. flagged)
        flagged[dp.driver_id] = ~field.convergence_flags  # [F] bool

        if progress is not None:
            progress.driver_finished(dp.driver_id, m, flagged[dp.driver_id])

        attrs = _driver_attrs(dp)
        driver_inputs.append((dp.driver_id, field, attrs))

    # ── Stage F: terminal responses + assembly ───────────────────────────────
    terminal_responses: Optional[list[np.ndarray]] = None
    if all(dp.terminal is not None for dp in req.drivers):
        terminal_responses = terminal_responses_for(
            [dp.terminal for dp in req.drivers],  # type: ignore[arg-type]
            req.frequencies.frequencies,
            rho=req.config.air_density,
            c=req.config.speed_of_sound,
        )  # list of [F] complex128, engineering exp(-jωt) convention

    dataset = build_dataset(
        driver_inputs=driver_inputs,
        directions=obs,
        freq_grid_spacing=req.frequencies.spacing,
        freq_grid_fractional_octave=req.frequencies.fractional_octave,
        terminal_responses=terminal_responses,
        root_attrs=_root_attrs(req, flagged),
    )
    # dataset.drivers[m].H_bem  [F, N] complex128
    # dataset.drivers[m].H_full [F, N] complex128
    # stacked_h_full(dataset) → [M, F, N] complex128 (Phase-2 steering matrix)

    # ── Stage G: exports ─────────────────────────────────────────────────────
    h5_path: Optional[Path] = None
    if req.output_h5 is not None:
        from beamsim2.io.hdf5_store import write_dataset

        h5_path = Path(req.output_h5)
        h5_path.parent.mkdir(parents=True, exist_ok=True)
        write_dataset(h5_path, dataset)

    if req.export_frd_dir is not None:
        from beamsim2.io.frd_export import write_frd

        write_frd(req.export_frd_dir, dataset)

    if req.export_sofa_path is not None:
        from beamsim2.io.sofa_export import write_sofa

        write_sofa(req.export_sofa_path, dataset)

    return SimulationResult(
        dataset=dataset,
        h5_path=h5_path,
        health=health,
        flagged_frequencies=flagged,
        work_dirs=work_dirs,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _default_backend() -> BEMBackend:
    """Lazily construct a NumCalcBackend; raise FileNotFoundError if binary missing."""
    from beamsim2.backends.numcalc.adapter import NumCalcBackend

    return NumCalcBackend()


def _make_scheduler(
    backend: BEMBackend,
    progress: Optional["ProgressModel"],
    m: int,
    frequencies: np.ndarray,
) -> Optional[object]:
    """Build a progress-wired NumCalcScheduler, or None for headless/non-NumCalc."""
    if progress is None:
        return None
    try:
        from beamsim2.backends.numcalc.adapter import NumCalcBackend
        from beamsim2.backends.numcalc.scheduler import NumCalcScheduler

        if not isinstance(backend, NumCalcBackend):
            return None
        binary = backend._binary

        # Provisional step_done tracking: we emit step_done(converged=True) when
        # the scheduler fires "step_done", then reconcile with the authoritative
        # convergence flags when "step_converged" fires.
        _provisional: dict[int, bool] = {}

        def on_event(event: str, step: int, info: dict) -> None:
            if event == "step_running":
                progress.step_running(m, step, info.get("est_ram", 0.0))
            elif event == "step_done":
                _provisional[step] = True  # converged assumed until step_converged fires
            elif event == "step_converged":
                converged = bool(info.get("converged", True))
                progress.step_done(m, step, converged)
                _provisional.pop(step, None)

        return NumCalcScheduler(binary=binary, on_event=on_event)
    except Exception:
        return None


def _driver_attrs(dp: DriverPlacement) -> dict:
    """Build the §3.5 per-driver metadata dict for build_dataset."""
    attrs: dict = {
        "name": dp.driver_id,
        "position": list(dp.spec.center),
        "orientation": list(dp.spec.normal),
        "radius": dp.spec.radius,
        "profile": "flush_disk",
    }
    if dp.terminal is not None:
        attrs.update(dp.terminal.to_attrs())
    return attrs


def _root_attrs(req: SimulationRequest, flagged: dict[str, np.ndarray]) -> dict:
    """Build the §3.5 root-level metadata dict for build_dataset."""
    n_flagged = {did: int(np.sum(flags)) for did, flags in flagged.items() if np.any(flags)}
    return {
        "schema_version": "1.0",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "solver_backend": "numcalc",
        "phase_origin": [0.0, 0.0, 0.0],
        "axis_convention": "right-hand xyz",
        "length_units": "metres",
        "observation_radius": req.sphere_radius,
        "far_field": False,
        "pressure_convention": "Pa at r_obs for unit cone velocity",
        "speed_of_sound": req.config.speed_of_sound,
        "air_density": req.config.air_density,
        "air_attenuation_model": req.config.air_attenuation_model,
        "convergence_summary": n_flagged if n_flagged else "all_converged",
    }


# ---------------------------------------------------------------------------
# Self-test (quick smoke, no NumCalc)
# ---------------------------------------------------------------------------


def _self_test() -> None:
    """Verify dataclass construction — no solve."""
    geom = BoxGeometry(0.12, 0.10, 0.08)
    assert geom.width == 0.12

    from beamsim2.core.types import FrequencyGrid

    req = SimulationRequest(
        geometry=geom,
        drivers=[
            DriverPlacement(
                spec=DriverSpec((0.06, 0.05, 0.08), (0.0, 0.0, 1.0), 0.020),
                terminal=None,
                driver_id="drv_0",
            )
        ],
        frequencies=FrequencyGrid(np.array([250.0, 500.0]), spacing="log"),
    )
    assert len(req.drivers) == 1
    print("pipeline/run.py self-test: PASS")


if __name__ == "__main__":
    _self_test()
