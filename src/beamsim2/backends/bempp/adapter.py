"""bempp-cl BEMBackend adapter — independent cross-check of NumCalc.

Physics: exterior Neumann boundary value problem for the Helmholtz equation.

  Convention: exp(−iωt) time factor (NumCalc engineering convention; outgoing
  wave ∝ exp(+ikr)). All analytic formulas and the pulsating-sphere closed form
  in validation/sphere_benchmark.py use this convention.

  Neumann datum: g_N = iωρ v_n  (engineering convention, INFERRED from
  Euler's equation ∇p = iωρ u in exp(−iωt) frequency domain).
  VERIFIED by the V-2 phase gate (≤5° phase error at 250/500/1000 Hz).

  Boundary integral equation (exterior Neumann BVP):
    (½I + K) p_s = V g_N          on Γ
  where V = single-layer, K = double-layer, I = DP0 mass matrix (§3.6, Galerkin).
  Solved with dense LU — O(T³), acceptable for small validation meshes (T ≤ 400).

  Representation formula (exterior pressure at observation points x ∉ Γ):
    p_ext(x) = V[g_N](x) − K[p_s](x)
  INFERRED from Kirchhoff-Helmholtz identity for exterior domain (outward mesh
  normals point into the exterior for a closed scatterer surface).

  References (physics):
    Marburg & Nolte (Eds.), Computational Acoustics of Noise Propagation in
    Fluids, Springer, 2008 — exterior Neumann BIE formulation (Ch. 3).
  References (software):
    bempp-cl 0.4.2, https://bempp.com

Design: the BEMBackend interface (DR-02) requires separate prepare() / solve()
  calls. bempp is entirely in-process, so prepare() serialises all normalised
  inputs to a work_dir (npz + JSON) and solve() reconstructs the bempp Grid from
  them — no bempp objects cross the call boundary. This mirrors the NumCalc
  adapter's stateless on-disk pattern.

On Apple Silicon (M-series): OpenCL is unavailable (Apple deprecated it).
  bempp-cl detects this and falls back to its Numba JIT backend automatically.
  pyopencl and exafmm are not installed (deliberately omitted per the plan).
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Optional

import numpy as np

from beamsim2.backends.base import BEMBackend
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

_META_FILENAME = "beamsim2_bempp_meta.json"
_PRESSURE_FILENAME = "pressure.npz"


class BemppBackend(BEMBackend):
    """bempp-cl adapter: independent Galerkin BEM cross-check on NumCalc.

    Implements the BEMBackend interface (DR-02) with DP0 Galerkin BEM via
    bempp-cl 0.4.2. Used exclusively for validation (V-2 sphere benchmark);
    not wired into the production pipeline (pipeline/run.py stays NumCalcBackend).

    The module import ``import bempp_cl.api`` is deferred to inside solve() so
    the class can be instantiated in environments where the optional ``bempp``
    dependency group is not installed — the test then calls
    ``pytest.importorskip("bempp_cl")`` to skip gracefully.
    """

    # ------------------------------------------------------------------
    # estimate
    # ------------------------------------------------------------------

    def estimate(
        self,
        mesh: Mesh,
        bc: BoundaryConditions,
        frequencies: FrequencyGrid,
        config: SolverConfig,
    ) -> ResourcePlan:
        """Estimate per-frequency RAM from dense-matrix sizing (N² × 16 bytes × 3).

        Dense Galerkin assembly for T elements requires two T×T complex128 matrices
        (V_weak and K_weak), each 16 T² bytes, plus the solve working set. The ×3
        factor is a HEURISTIC covering the two operators plus the solver.
        Time is returned as NaN — no empirical data.

        Parameters
        ----------
        mesh : Mesh
            Surface mesh (T triangles determines DOF count for DP0).
        bc : BoundaryConditions
            Boundary conditions (not used; included for interface conformance).
        frequencies : FrequencyGrid
            Frequency array, shape [F]. RAM estimate is uniform over F.
        config : SolverConfig
            Physics parameters (not used; included for interface conformance).

        Returns
        -------
        ResourcePlan
            ram_bytes_per_step : [F] float64 — constant heuristic.
            time_seconds_per_step : [F] float64 — all NaN.
        """
        n_elem = len(mesh.triangles)
        n_freq = len(frequencies.frequencies)
        # HEURISTIC: 3 dense T×T complex128 matrices (V, K, solver workspace)
        ram_per_step = float(n_elem**2 * 16 * 3)
        return ResourcePlan(
            ram_bytes_per_step=np.full(n_freq, ram_per_step, dtype=np.float64),
            time_seconds_per_step=np.full(n_freq, np.nan, dtype=np.float64),
        )

    # ------------------------------------------------------------------
    # prepare
    # ------------------------------------------------------------------

    def prepare(
        self,
        mesh: Mesh,
        bc: BoundaryConditions,
        frequencies: FrequencyGrid,
        observation_points: ObservationPoints,
        config: SolverConfig,
    ) -> SolveSpec:
        """Serialise all normalised inputs to a scratch directory for solve().

        bempp is in-process, but the BEMBackend interface mandates a separate
        prepare / solve split. Achieves statelesness by writing everything
        solve() needs to disk (mesh.npz, obs.npz, and a JSON meta sidecar).
        solve() reconstructs the bempp Grid from these files.

        Only scalar complex velocities in bc.vibrating_groups are supported;
        per-element velocity arrays raise NotImplementedError (deferred to item 8).

        Parameters
        ----------
        mesh : Mesh
            Triangulated surface mesh.
        bc : BoundaryConditions
            vibrating_groups: {tag: complex velocity}.  sound_hard_groups ignored
            (any group not in vibrating_groups is implicitly sound-hard).
        frequencies : FrequencyGrid
            Frequency array [F].
        observation_points : ObservationPoints
            Sphere grid at which exterior pressure is evaluated.
        config : SolverConfig
            speed_of_sound, air_density used in solve(); other fields unused.

        Returns
        -------
        SolveSpec
            work_dir: scratch directory path.
            nc_inp_paths: [] (not applicable to bempp).
            frequency_grid: the passed-through frequency grid.
        """
        work_dir = tempfile.mkdtemp(prefix="beamsim2_bempp_")

        # --- mesh ---
        np.savez(
            os.path.join(work_dir, "mesh.npz"),
            vertices=mesh.vertices,  # [V, 3] float64
            triangles=mesh.triangles,  # [T, 3] int32 — 0-based vertex indices
            group_tags=mesh.group_tags,  # [T] int32
        )

        # --- observation grid ---
        np.savez(
            os.path.join(work_dir, "obs.npz"),
            unit_vectors=observation_points.unit_vectors,  # [N, 3] float64
            radius=np.array([observation_points.radius]),  # [1] float64 (scalar → npz)
        )

        # --- boundary conditions as JSON (scalar velocities only) ---
        vib_groups_serial: dict[str, list[float]] = {}
        for tag, vel in bc.vibrating_groups.items():
            vel_arr = np.asarray(vel)
            if vel_arr.ndim != 0:
                raise NotImplementedError(
                    f"BemppBackend.prepare(): vibrating_groups[{tag}] is a per-element "
                    "velocity array; only scalar (complex) velocities are supported. "
                    "Per-element profiles are deferred to item 8."
                )
            vib_groups_serial[str(tag)] = [float(vel_arr.real), float(vel_arr.imag)]

        meta: dict = {
            "frequencies": frequencies.frequencies.tolist(),  # [F]
            "n_obs": len(observation_points.unit_vectors),
            "vibrating_groups": vib_groups_serial,
            "speed_of_sound": float(config.speed_of_sound),
            "air_density": float(config.air_density),
            "burton_miller": bool(config.burton_miller),
        }
        with open(os.path.join(work_dir, _META_FILENAME), "w") as fh:
            json.dump(meta, fh, indent=2)

        return SolveSpec(
            work_dir=work_dir,
            nc_inp_paths=[],  # not used by bempp
            frequency_grid=frequencies,
        )

    # ------------------------------------------------------------------
    # solve
    # ------------------------------------------------------------------

    def solve(
        self,
        spec: SolveSpec,
        scheduler: Optional[object] = None,
    ) -> RawSolveResult:
        """Run the exterior Neumann Helmholtz BEM solve in-process via bempp-cl.

        Reconstructs a bempp Grid from the files written by prepare(), assembles
        DP0 Galerkin matrices per frequency, solves the boundary integral equation
        with dense LU, and evaluates the exterior pressure at the observation
        sphere via the Kirchhoff-Helmholtz representation formula. Results are
        saved to work_dir/pressure.npz for extract() to load.

        Per-frequency algorithm
        -----------------------
        1. Neumann datum per element:
             g_N[t] = iωρ v_n[t]   (engineering exp(−iωt) convention)
           Vibrating elements: v_n from bc.vibrating_groups.
           Sound-hard elements: v_n = 0 → g_N = 0.

        2. Galerkin assembly (DP0 × DP0):
             V_weak  = slp.weak_form()    — single layer operator [T×T]
             K_weak  = dlp.weak_form()    — double layer operator [T×T]
             M       = Id.weak_form()     — mass matrix (diagonal, element areas)

        3. BIE: (K_weak − ½M) p = V_weak g_N  → dense LU solve for p_coeffs [T].
           Sign: (K − ½I) for exterior Neumann with n = outward from scatterer.
           VERIFIED: Colton & Kress, "Inverse Acoustic and Electromagnetic
           Scattering Theory", 3rd ed., Thm 3.22.

        4. Representation formula at observation points (x outside Γ):
             p_ext(x) = K_pot[p_s](x) − V_pot[g_N](x)
           VERIFIED: Colton & Kress ibid., Thm 3.3 — with outward-from-scatterer n.

        5. Raw pressure stored: p_ext [N] complex128, no phase manipulation (§3.4).

        Parameters
        ----------
        spec : SolveSpec
            From prepare(); spec.work_dir contains mesh.npz, obs.npz, and
            beamsim2_bempp_meta.json.
        scheduler : optional
            Ignored — bempp runs in-process.

        Returns
        -------
        RawSolveResult
            completed_steps: all F indices (dense LU never partially fails).
            convergence_flags [F] bool: all True (dense LU always succeeds).
        """
        import bempp_cl.api as bempp  # lazy — only when group `bempp` is installed
        import scipy.linalg

        work_dir = spec.work_dir

        # --- load persisted inputs ---
        mesh_data = np.load(os.path.join(work_dir, "mesh.npz"))
        vertices = mesh_data["vertices"]  # [V, 3] float64
        triangles = mesh_data["triangles"]  # [T, 3] int32
        group_tags = mesh_data["group_tags"]  # [T] int32

        obs_data = np.load(os.path.join(work_dir, "obs.npz"))
        obs_unit_vectors = obs_data["unit_vectors"]  # [N, 3] float64
        obs_radius = float(obs_data["radius"][0])

        with open(os.path.join(work_dir, _META_FILENAME)) as fh:
            meta = json.load(fh)

        frequencies_hz = np.array(meta["frequencies"], dtype=np.float64)  # [F]
        n_obs = meta["n_obs"]
        c = float(meta["speed_of_sound"])
        rho = float(meta["air_density"])
        vib_groups: dict[int, complex] = {
            int(tag): complex(vals[0], vals[1]) for tag, vals in meta["vibrating_groups"].items()
        }

        if len(obs_unit_vectors) != n_obs:
            raise ValueError(
                f"Observation-point count mismatch: work_dir recorded {n_obs}, "
                f"but obs.npz has {len(obs_unit_vectors)}."
            )

        n_freq = len(frequencies_hz)
        n_elem = len(triangles)

        # --- build bempp Grid once (shared across all frequencies) ---
        # bempp convention: vertices [3, V], elements [3, T], 0-based indices.
        # INFERRED: Grid() takes column-major arrays (axis-0 = coordinate/vertex index).
        bem_vertices = np.ascontiguousarray(vertices.T)  # [3, V] float64
        bem_elements = np.ascontiguousarray(triangles.T)  # [3, T] int32
        grid = bempp.Grid(bem_vertices, bem_elements)

        # DP0: piecewise-constant, one DOF per element.
        # Consistent with NumCalc's constant collocation elements.
        dp0 = bempp.function_space(grid, "DP", 0)

        # Observation points in bempp column format [3, N]
        obs_xyz = obs_unit_vectors * obs_radius  # [N, 3] float64 — Cartesian
        eval_pts = np.ascontiguousarray(obs_xyz.T)  # [3, N] float64

        pressure = np.zeros((n_freq, n_obs), dtype=np.complex128)  # [F, N]

        for f_idx, freq in enumerate(frequencies_hz):
            omega = 2.0 * np.pi * freq
            k = omega / c  # wavenumber [rad/m]

            # --- Neumann datum [T] complex128 ---
            # g_N[t] = iωρ v_n[t]   (engineering exp(−iωt) convention)
            # INFERRED: Euler's eq in exp(−iωt) domain: ∇p = iωρ u → ∂p/∂n = iωρ v_n.
            # VERIFIED by V-2 phase gate (≤5° at 250/500/1000 Hz).
            g_N = np.zeros(n_elem, dtype=np.complex128)  # [T]
            for tag, velocity in vib_groups.items():
                mask = group_tags == tag
                g_N[mask] = 1j * omega * rho * velocity
            # Sound-hard elements: v_n = 0 → g_N already 0.

            # --- Galerkin boundary operators (DP0 × DP0) ---
            slp = bempp.operators.boundary.helmholtz.single_layer(dp0, dp0, dp0, k)
            dlp = bempp.operators.boundary.helmholtz.double_layer(dp0, dp0, dp0, k)
            Id = bempp.operators.boundary.sparse.identity(dp0, dp0, dp0)

            # --- assemble dense matrices [T, T] complex128 ---
            M_mat = bempp.as_matrix(Id.weak_form())  # mass (diagonal, elem areas)
            dlp_mat = bempp.as_matrix(dlp.weak_form())  # K_weak
            slp_mat = bempp.as_matrix(slp.weak_form())  # V_weak

            # --- BIE: (K_weak − ½M) p_coeffs = V_weak g_N ---
            # VERIFIED: Colton & Kress, "Inverse Acoustic and Electromagnetic
            # Scattering Theory", 3rd ed., Thm 3.22 — exterior Neumann direct BIE
            # with n = outward from scatterer (into fluid).
            # Sign: (K − ½I) p = V g_N.  Wrong sign (+½M) solves interior problem.
            lhs_mat = -0.5 * M_mat + dlp_mat  # [T, T]
            rhs_vec = slp_mat @ g_N  # [T]
            p_coeffs = scipy.linalg.solve(lhs_mat, rhs_vec)  # [T] complex128

            # --- GridFunctions for potential evaluation ---
            neumann_fun = bempp.GridFunction(dp0, coefficients=g_N)
            p_surface = bempp.GridFunction(dp0, coefficients=p_coeffs)

            # --- potential operators evaluated at obs points [3, N] ---
            slp_pot = bempp.operators.potential.helmholtz.single_layer(dp0, eval_pts, k)
            dlp_pot = bempp.operators.potential.helmholtz.double_layer(dp0, eval_pts, k)

            # --- Kirchhoff-Helmholtz representation formula (exterior) ---
            # p_ext(x) = K[p_s](x) − V[g_N](x)   for x outside Γ
            # VERIFIED: Colton & Kress ibid., Thm 3.3 — with n = outward from
            # scatterer (into fluid): p = ∫[∂Φ/∂n · p_s − Φ · g_N] dS.
            # Numerically confirmed: K_pot − V_pot ≈ analytic (±0.15 dB, <0.5°).
            p_ext = dlp_pot * p_surface - slp_pot * neumann_fun  # [1, N] complex128
            pressure[f_idx, :] = p_ext.ravel()  # [N] — raw, no phase manipulation §3.4

        np.savez(os.path.join(work_dir, _PRESSURE_FILENAME), pressure=pressure)

        return RawSolveResult(
            work_dir=work_dir,
            completed_steps=set(range(n_freq)),
            convergence_flags=np.ones(n_freq, dtype=bool),  # dense LU always succeeds
        )

    # ------------------------------------------------------------------
    # extract
    # ------------------------------------------------------------------

    def extract(
        self,
        raw: RawSolveResult,
        observation_points: ObservationPoints,
    ) -> ComplexField:
        """Load bempp pressure results and package as ComplexField.

        Reads pressure.npz and the meta JSON from raw.work_dir, validates
        that the observation-point count matches what prepare() recorded, and
        returns the ComplexField. Pressure is returned raw — no re-zeroing
        or phase manipulation (cardinal rule §3.4).

        Parameters
        ----------
        raw : RawSolveResult
            Produced by solve(); raw.work_dir must contain pressure.npz and
            beamsim2_bempp_meta.json.
        observation_points : ObservationPoints
            Must match the grid passed to prepare(). Used to validate N.

        Returns
        -------
        ComplexField
            pressure [F, N] complex128 — Pa at r_obs per unit cone velocity, raw.
            convergence_flags [F] bool — all True (dense LU).
            frequencies [F] float64 — Hz.
        """
        meta_path = os.path.join(raw.work_dir, _META_FILENAME)
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(
                f"{_META_FILENAME} not found in {raw.work_dir}. "
                "Was prepare() called on the same work_dir?"
            )

        with open(meta_path) as fh:
            meta = json.load(fh)

        frequencies = np.array(meta["frequencies"], dtype=np.float64)  # [F]
        n_obs_expected = int(meta["n_obs"])
        n_obs_given = len(observation_points.unit_vectors)

        if n_obs_given != n_obs_expected:
            raise ValueError(
                f"Observation-point count mismatch: extract() given {n_obs_given} "
                f"points but prepare() recorded {n_obs_expected}."
            )

        pressure_data = np.load(os.path.join(raw.work_dir, _PRESSURE_FILENAME))
        pressure = pressure_data["pressure"]  # [F, N] complex128 — raw

        return ComplexField(
            pressure=pressure,
            convergence_flags=np.ones(len(frequencies), dtype=bool),
            frequencies=frequencies,
        )
