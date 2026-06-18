"""Shared normalized data types that cross every module boundary.

These are the types defined by DR-02 (BEMBackend contract) and §3 (data contract).
No solver-specific objects appear here — every backend translates into and out of
these types via its adapter. The GUI, pipeline, and validation code all speak only
these types, never raw NumPy arrays with implicit shapes.

In acoustics terms: think of these as the "connector spec" that lets you swap the
BEM engine (NumCalc → bempp → COMSOL) without touching the rest of the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


@dataclass
class Mesh:
    """Triangulated surface mesh with per-element group tags.

    Produced by the geometry pipeline (Stage B/C) and consumed by backend.prepare().
    The solver never sees face-level details beyond what is in this dataclass.

    Parameters
    ----------
    vertices : np.ndarray, shape [V, 3], float64
        XYZ vertex coordinates in metres.
    triangles : np.ndarray, shape [T, 3], int32
        Vertex index triples (0-based).
    group_tags : np.ndarray, shape [T], int32
        Per-element surface-group tag. Tags identify diaphragm elements (vibrating
        velocity BC) versus enclosure elements (sound-hard Neumann BC). Assigned by
        geometry.assemble when the driver cap is unioned into the enclosure shell.
    """

    vertices: np.ndarray  # [V, 3] float64 — XYZ in metres
    triangles: np.ndarray  # [T, 3] int32 — vertex indices, 0-based
    group_tags: np.ndarray  # [T] int32 — per-element surface-group tag

    def __post_init__(self) -> None:
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise ValueError(f"vertices must be [V, 3], got {self.vertices.shape}")
        if self.triangles.ndim != 2 or self.triangles.shape[1] != 3:
            raise ValueError(f"triangles must be [T, 3], got {self.triangles.shape}")
        if self.group_tags.ndim != 1 or len(self.group_tags) != len(self.triangles):
            raise ValueError(
                f"group_tags must be [T], got {self.group_tags.shape} "
                f"(expected [{len(self.triangles)}])"
            )


# ---------------------------------------------------------------------------
# Boundary conditions
# ---------------------------------------------------------------------------


@dataclass
class BoundaryConditions:
    """Boundary conditions for a BEM solve.

    In the BEM context (DR-02): elements in `vibrating_groups` carry a prescribed
    complex normal velocity (the cone velocity); all other elements default to
    sound-hard (zero normal velocity = Neumann BC with v_n = 0). This matches
    the sealed or vented box model where the cone is the only vibrating surface.

    VERIFIED: standard BEM loudspeaker convention (Mesh2HRTF / NumCalc docs;
    Kreuzer et al., Engineering Analysis with Boundary Elements 161:157-178, 2024).

    Parameters
    ----------
    vibrating_groups : dict
        Keys are surface-group tags (int). Values are the prescribed complex normal
        velocity in m/s — either a scalar (uniform across the group) or a 1-D
        ndarray of length equal to the number of elements in that group (per-element
        velocity profile for cone breakup or tapering surround).
    sound_hard_groups : set of int, optional
        Tags explicitly marked sound-hard. Default is the empty set; all tags not in
        `vibrating_groups` are implicitly sound-hard.
    """

    vibrating_groups: dict[int, complex | np.ndarray]
    sound_hard_groups: set[int] = field(default_factory=set)


# ---------------------------------------------------------------------------
# Frequency grid
# ---------------------------------------------------------------------------


@dataclass
class FrequencyGrid:
    """Explicit frequency array plus spacing metadata.

    The `frequencies` array is the single source of truth. The `spacing` and
    `fractional_octave` fields record how it was generated (for HDF5 metadata and
    for regenerating a matching grid at a different resolution).

    Parameters
    ----------
    frequencies : np.ndarray, shape [F], float64
        Explicit frequency values in Hz. Must be strictly positive and increasing.
    spacing : str
        How the array was generated: "fractional-octave", "log", or "linear".
    fractional_octave : float or None
        Octave fraction (e.g. 1/12 for 1/12-octave). None when spacing != "fractional-octave".
    interpolated_mask : np.ndarray, shape [F], bool or None
        True where a frequency bin was filled by SH + minimum-phase interpolation
        rather than solved directly. None if no sparse simulation was used.
    """

    frequencies: np.ndarray  # [F] float64 — Hz
    spacing: str = "fractional-octave"
    fractional_octave: Optional[float] = 1 / 12
    interpolated_mask: Optional[np.ndarray] = None  # [F] bool

    def __post_init__(self) -> None:
        if self.frequencies.ndim != 1:
            raise ValueError(f"frequencies must be 1-D, got shape {self.frequencies.shape}")
        if len(self.frequencies) == 0:
            raise ValueError("frequencies array must not be empty")


# ---------------------------------------------------------------------------
# Observation points (sphere grid + radius)
# ---------------------------------------------------------------------------


@dataclass
class ObservationPoints:
    """Directional sampling grid on an observation sphere.

    Combines the directional part (unit vectors + quadrature weights) with an
    observation radius. This is exactly what the solver backend needs to place
    evaluation points in its field-point list.

    In acoustics terms: these are the microphone positions on an imaginary sphere
    of radius `r` centred on the global origin, at which the BEM solver evaluates
    the pressure. They mirror a real anechoic balloon measurement.

    Produced by core.sphere.lebedev() and related factory functions.

    Parameters
    ----------
    unit_vectors : np.ndarray, shape [N, 3], float64
        Cartesian unit direction cosines. Each row has L2-norm 1.0.
    radius : float
        Observation sphere radius in metres (e.g. 1.0).
    weights : np.ndarray, shape [N], float64
        Quadrature weights. If weight_convention="sum_4pi" then sum(weights) == 4π.
        Allows integrating any function f over the sphere as simply Σ wᵢ f(xᵢ).
    scheme : str
        Sampling scheme name, e.g. "lebedev".
    order : int
        Scheme-specific resolution parameter (Lebedev: number of points).
    weight_convention : str
        "sum_4pi" (weights sum to 4π, the surface area of the unit sphere) or
        "sum_1" (normalized weights; multiply by 4π to integrate).
    theta_phi : np.ndarray, shape [N, 2], float64 or None
        Convenience spherical coordinates: column 0 is colatitude θ ∈ [0, π],
        column 1 is azimuth φ ∈ [0, 2π], both in radians.
    """

    unit_vectors: np.ndarray  # [N, 3] float64
    radius: float
    weights: np.ndarray  # [N] float64 — quadrature weights, sum = 4π
    scheme: str = "lebedev"
    order: int = 0
    weight_convention: str = "sum_4pi"
    theta_phi: Optional[np.ndarray] = None  # [N, 2] float64 — (θ, φ) in radians

    def __post_init__(self) -> None:
        if self.unit_vectors.ndim != 2 or self.unit_vectors.shape[1] != 3:
            raise ValueError(f"unit_vectors must be [N, 3], got {self.unit_vectors.shape}")
        n = len(self.unit_vectors)
        if self.weights.ndim != 1 or len(self.weights) != n:
            raise ValueError(f"weights must be [N], got {self.weights.shape} (expected [{n}])")
        if self.theta_phi is not None:
            if self.theta_phi.shape != (n, 2):
                raise ValueError(f"theta_phi must be [N, 2], got {self.theta_phi.shape}")


# ---------------------------------------------------------------------------
# Solver configuration
# ---------------------------------------------------------------------------


@dataclass
class SolverConfig:
    """Physics and numerical parameters for a BEM solve.

    Passed to every BEMBackend method. The backend is responsible for translating
    these into its own solver's input format (e.g. NC.inp parameters for NumCalc).

    Parameters
    ----------
    n_epw : int
        Elements per wavelength at the highest frequency in the solve.
        HEURISTIC: 6–8 for NumCalc collocation BEM (Kreuzer et al. 2024).
    tolerance : float
        Iterative-solver convergence tolerance. NumCalc CGS default is 1e-6.
    max_iterations : int
        Iteration cap for the iterative solver.
    burton_miller : bool
        Whether to use Burton–Miller formulation to suppress spurious interior
        resonances. VERIFIED must be True for exterior acoustics (Kreuzer et al.
        2024). Only set False for debugging on simple geometries.
    speed_of_sound : float
        Speed of sound in m/s for the medium. See core.units.speed_of_sound().
    air_density : float
        Air density in kg/m³. See core.units.air_density().
    air_attenuation_model : str
        "none" (no atmospheric absorption) or "iso9613-1" (when implemented).
    """

    n_epw: int = 6
    tolerance: float = 1e-6
    max_iterations: int = 1000
    burton_miller: bool = True
    speed_of_sound: float = 343.2  # m/s — dry air, ~20°C
    air_density: float = 1.2041  # kg/m³ — standard dry air, 20°C, 101325 Pa
    air_attenuation_model: str = "none"


# ---------------------------------------------------------------------------
# Backend resource plan
# ---------------------------------------------------------------------------


@dataclass
class ResourcePlan:
    """Per-frequency RAM and time estimates from backend.estimate().

    Used by the scheduler to launch frequency-step processes without exceeding
    available RAM. NumCalc provides these via its -estimate_ram flag.

    Parameters
    ----------
    ram_bytes_per_step : np.ndarray, shape [F], float64
        Estimated peak RAM for each frequency step in bytes.
    time_seconds_per_step : np.ndarray, shape [F], float64
        Estimated wall-clock time per frequency step in seconds.
    """

    ram_bytes_per_step: np.ndarray  # [F] float64 — bytes
    time_seconds_per_step: np.ndarray  # [F] float64 — seconds


# ---------------------------------------------------------------------------
# Solve specification (output of backend.prepare())
# ---------------------------------------------------------------------------


@dataclass
class SolveSpec:
    """Everything the solver needs to execute a prepared job.

    Produced by BEMBackend.prepare(); consumed by BEMBackend.solve().
    For NumCalc this is a directory of NC.inp files plus bookkeeping metadata.

    Parameters
    ----------
    work_dir : str
        Absolute path to the scratch directory containing solver input files.
    nc_inp_paths : list of str
        Ordered list of NC.inp file paths, one per frequency step.
    frequency_grid : FrequencyGrid
        The frequency grid this solve covers.
    """

    work_dir: str
    nc_inp_paths: list[str]  # one path per frequency step
    frequency_grid: FrequencyGrid


# ---------------------------------------------------------------------------
# Raw solve result (output of backend.solve())
# ---------------------------------------------------------------------------


@dataclass
class RawSolveResult:
    """Raw output from a completed (or partially completed) BEM solve.

    Produced by BEMBackend.solve(); consumed by BEMBackend.extract().
    Holds enough information to resume a partial solve or extract results.

    Parameters
    ----------
    work_dir : str
        Absolute path to the scratch directory with solver output files.
    completed_steps : set of int
        Indices (into FrequencyGrid.frequencies) of successfully completed steps.
    convergence_flags : np.ndarray, shape [F], bool
        True where the iterative solver converged within tolerance and iterations.
    """

    work_dir: str
    completed_steps: set[int] = field(default_factory=set)
    convergence_flags: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=bool)
    )  # [F] bool


# ---------------------------------------------------------------------------
# Complex pressure field (output of backend.extract())
# ---------------------------------------------------------------------------


@dataclass
class ComplexField:
    """Complex pressure field at the observation points for one driver.

    This is the direct output of BEMBackend.extract() and the raw material for
    assembly/superpose.py. In acoustics terms: the complex transfer function
    pressure / (unit cone velocity) evaluated at every direction on the observation
    sphere, at every frequency.

    For the single-phase-origin rule (§3.4), this field must carry the natural
    path-length phase from the driver's position — never re-zeroed or minimum-phased.

    Parameters
    ----------
    pressure : np.ndarray, shape [F, N], complex128
        Complex pressure at r_obs for unit cone velocity (Pa).
    convergence_flags : np.ndarray, shape [F], bool
        True where the solver converged for that frequency step.
    frequencies : np.ndarray, shape [F], float64
        Frequency values in Hz (mirrors FrequencyGrid.frequencies).
    """

    pressure: np.ndarray  # [F, N] complex128 — Pa at r_obs, unit cone velocity
    convergence_flags: np.ndarray  # [F] bool
    frequencies: np.ndarray  # [F] float64 — Hz

    def __post_init__(self) -> None:
        f = len(self.frequencies)
        if self.pressure.ndim != 2 or self.pressure.shape[0] != f:
            raise ValueError(f"pressure must be [F, N] with F={f}, got {self.pressure.shape}")
        if self.convergence_flags.ndim != 1 or len(self.convergence_flags) != f:
            raise ValueError(
                f"convergence_flags must be [F={f}], got {self.convergence_flags.shape}"
            )
