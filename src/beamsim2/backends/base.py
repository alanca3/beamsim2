"""Abstract BEMBackend interface — the normalized contract between the solver-abstraction
layer and every concrete BEM adapter (NumCalc, bempp-cl, COMSOL).

DR-01/DR-02 constraint: nothing solver-specific crosses this boundary. Every method
parameter and every return value is a type from core.types. No NC.inp paths, no bempp
GridFunctions, no COMSOL handles appear here or upstream.

The single-phase-origin rule (§3.4) is enforced at this interface: ComplexField.pressure
must be returned raw — never minimum-phased or re-zeroed per-driver. The natural path-length
phase from each driver's physical position is the beamforming steering information; stripping
it silently mis-steers the Phase-2 beam.

In acoustics terms: this is the standard XLR connector between the mixing desk (pipeline)
and whatever amplifier (BEM engine) sits behind it. Swapping NumCalc → bempp → COMSOL
means swapping the amp box, not re-wiring the desk.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from beamsim2.core.types import (
    BoundaryConditions,
    ComplexField,
    FrequencyGrid,
    Mesh,
    ObservationPoints,
    RawSolveResult,
    ResourcePlan,
    SolverConfig,
    SolveSpec,
)


class BEMBackend(ABC):
    """Abstract base class for all BEM solver backends.

    Concrete subclasses implement estimate / prepare / solve / extract for one
    specific solver. The pipeline always calls these four methods in order; each
    adapter translates the normalized core types into whatever its solver needs
    on disk or in memory.

    DR-02 note on prepare(): the Gameplan originally placed ObservationPoints only
    on extract(). NumCalc physically requires the evaluation-grid to be baked into
    NC.inp at prepare() time, so ObservationPoints is added to prepare(). The DR-02
    essence — "only normalized core types cross the boundary" — is preserved.
    Future backends that genuinely evaluate post-hoc (e.g. bempp-cl) receive the
    points at extract() and may ignore the prepare() argument.
    """

    @abstractmethod
    def estimate(
        self,
        mesh: Mesh,
        bc: BoundaryConditions,
        frequencies: FrequencyGrid,
        config: SolverConfig,
    ) -> ResourcePlan:
        """Return per-frequency RAM and wall-clock estimates without solving.

        In acoustics terms: "how heavy is this job before we commit the machine?"
        The scheduler uses these estimates to pack concurrent frequency-step processes
        into available RAM (48 GB target machine) without thrashing.

        For NumCalc this wraps the ``-estimate_ram`` flag; for bempp it can estimate
        from dense-matrix sizing (N² × 16 bytes for complex128).

        Parameters
        ----------
        mesh : Mesh
            Triangulated surface mesh (boundary geometry only; no observation sphere).
        bc : BoundaryConditions
            Which surface groups vibrate, at what complex velocity, and which are
            sound-hard.
        frequencies : FrequencyGrid
            Explicit frequency array to solve over, shape [F].
        config : SolverConfig
            Physics and numerical parameters (n_epw, tolerance, medium properties).

        Returns
        -------
        ResourcePlan
            ram_bytes_per_step [F] float64 and time_seconds_per_step [F] float64.
        """

    @abstractmethod
    def prepare(
        self,
        mesh: Mesh,
        bc: BoundaryConditions,
        frequencies: FrequencyGrid,
        observation_points: ObservationPoints,
        config: SolverConfig,
    ) -> SolveSpec:
        """Translate normalized inputs into solver-ready files on disk.

        No solving happens here. For NumCalc this writes the mesh Nodes/Elements files,
        the evaluation-grid files, and the NC.inp control file. For bempp it would
        assemble the BEM grid and operator objects in memory.

        # DR-02: ObservationPoints added to prepare() because NumCalc bakes the
        # evaluation grid into NC.inp at this stage; it cannot be deferred to extract().
        # The DR-02 essence ("only normalized types cross the boundary") is preserved.

        Parameters
        ----------
        mesh : Mesh
            Triangulated surface mesh with per-element group tags.
        bc : BoundaryConditions
            Boundary conditions (vibrating velocity groups + sound-hard groups).
        frequencies : FrequencyGrid
            Frequency array, shape [F].
        observation_points : ObservationPoints
            Directional sampling grid at which the solver evaluates the pressure field.
            These become the evaluation-mesh nodes in NC.inp.
        config : SolverConfig
            Physics and numerical parameters.

        Returns
        -------
        SolveSpec
            work_dir: scratch directory with all solver input files.
            nc_inp_paths: ordered list of NC.inp paths, one per (set of) frequency steps.
            frequency_grid: the FrequencyGrid this spec covers.
        """

    @abstractmethod
    def solve(
        self,
        spec: SolveSpec,
        scheduler: Optional[object] = None,
    ) -> RawSolveResult:
        """Execute the prepared solve and return raw output locations.

        For NumCalc: runs the NumCalc binary in spec.work_dir and waits for completion.
        The full RAM-aware, highest-frequency-first, resumable scheduler is implemented
        in item 6; here the scheduler argument is accepted but may be ignored.

        Parameters
        ----------
        spec : SolveSpec
            Prepared solve specification from prepare().
        scheduler : object, optional
            Future scheduler object (item 6). Ignored in minimal implementations.

        Returns
        -------
        RawSolveResult
            work_dir: same scratch directory.
            completed_steps: set of frequency-step indices that finished successfully.
            convergence_flags [F] bool: True where the iterative solver converged.
        """

    @abstractmethod
    def extract(
        self,
        raw: RawSolveResult,
        observation_points: ObservationPoints,
    ) -> ComplexField:
        """Extract complex pressure at the observation points from raw solver output.

        Pressure is returned **raw** — no re-zeroing, no minimum-phase processing.
        The natural path-length phase from the driver's physical location is the
        beamforming steering information and must not be stripped (cardinal rule §3.4).

        Parameters
        ----------
        raw : RawSolveResult
            Raw output from solve().
        observation_points : ObservationPoints
            The same grid passed to prepare(). Used to validate that the output
            point count matches and to order the pressure rows.

        Returns
        -------
        ComplexField
            pressure [F, N] complex128: Pa at r_obs for unit cone velocity.
            convergence_flags [F] bool: True where the solver converged.
            frequencies [F] float64: Hz.
        """
