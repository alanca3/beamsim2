# BeamSimII — Project Gameplan (Phase 1)

> **Status:** Planning deliverable. No implementation code. This document is the architecture/handoff reference for BeamSimII. Save it into the Claude Project's knowledge alongside `BEAMSIMII_Project_Overview.md` and `BEAMSIMII_First_Research.md`. Suggested filename: `BEAMSIMII_Gameplan.md`.
>
> **Supersedes:** nothing. **Superseded by:** nothing yet. **Authoritative inputs:** the Overview and the First Research report. Ignore all prior "beamsim" v1–v5 work.
>
> **Claim labels used throughout:** **VERIFIED** (confirmed from a cited source), **INFERRED** (reasoned from evidence but not directly confirmed for this exact case), **HEURISTIC** (rule of thumb / engineering judgment).

---

## 0. How to read this document, and the assumptions it is built on

This is a plan, not a build. It resolves the open architecture questions, lays out the Phase-1 pipeline, defines the data contract with Phase 2, proposes the project's file layout and tooling, sketches the GUI, defines the validation tests, sets the milestones, registers the risks, and gives a build order. It ends with two things specific to your situation: a from-zero **version-control and GitHub** plan, and a **pre-flight setup checklist** to run before any code is written.

Two facts you supplied are now load-bearing, so they are recorded here as fixed inputs:

- **Compute host:** Late-2024 MacBook Pro 16", Apple **M4 Max (16-core: 12 performance + 4 efficiency)**, **48 GB** unified memory, macOS 15.7.7. This is both the development machine and the solve machine. **INFERRED** core split: the 16-core M4 Max is 12P+4E (Apple's published configuration for the 16-core M4 Max).
- **Geometry input:** You can produce STEP/STL/OBJ, but expect mesh-quality trouble from imports. You want a **basic in-app builder** (simple primitives — box, cylinder, sphere, simple driver caps) *and* import. The builder can be basic; import must be robust enough to fail gracefully.

One process decision is folded in regardless of the technical plan: because you have a GitHub account but have never used it, and because you've previously lost track of changes/versions, **Section 11 is a complete Git + GitHub setup and versioning strategy written for someone with no formal software-development background.** Implementation will happen in Claude Code, and the version-control plan is designed to mesh with that.

A note on what I am *not* relitigating. The Overview's locked decisions — BEM as the engine; the app fully abstracts the solver; the modeled-cone-in-BEM driver plus a T/S-and-inductance-derived on-axis terminal response; macOS GUI; open-source/self-contained; no AKABAK/ABEC; the per-driver complex `H[driver × frequency × direction]` data contract with a single phase origin; the ~1–2 day compute budget — are treated as foundation. Where I push on one of them, I flag it explicitly (see DR-05 on the high-frequency efficiency fork).

---

## 1. Architecture decisions

Each decision is written as a short **Decision Record (DR)**: the decision, why, what was rejected, and the consequences. These are meant to be quotable later and revised only deliberately.

### DR-01 — BEM backend: **NumCalc (Mesh2HRTF) is primary**, behind a solver-abstraction layer, with bempp-cl as a validation backend and COMSOL as a manual escape hatch

**Decision.** Use **NumCalc** (the standalone BEM solver from the Mesh2HRTF project) as the default and primary backend for all production solves. Build a thin **solver-abstraction layer** (DR-02) so a second backend can be added without touching the rest of the app. Designate **bempp-cl** as the second backend, used chiefly for *cross-validation* and as a clean-Python reference, not as the high-frequency workhorse. Keep **COMSOL** (which you own) as a *manual* high-fidelity fallback for special cases, not a routinely-scripted backend on this machine.

**Why NumCalc wins on this specific hardware.**
- **macOS is a first-class platform for it.** The Mesh2HRTF team's preferred platforms are Linux and macOS, and on those platforms NumCalc is compiled from source with a single `make`. **VERIFIED** (Mesh2HRTF Wiki, *Installation* / *Basic HRTF NumCalc Simulation*, 2024). That it builds cleanly on your specific M4 Max / macOS 15.7.7 is **INFERRED** (strong evidence, but it must be confirmed — it is the Stage-0 gate in §8).
- **Fast-multipole acceleration is built into the core.** NumCalc implements a 3-D Burton–Miller collocation BEM coupled with the **multi-level fast multipole method (ML-FMM)** in its own C++. **VERIFIED** (OEAW/ARI Mesh2HRTF page; Kreuzer, Pollack, Brinkmann, Majdak, "NumCalc," *Engineering Analysis with Boundary Elements* 161:157–178, 2024). This is the decisive point: FMM is what makes high-frequency solves tractable (it turns the dense O(N²) matrix–vector product into roughly O(N log N) with near-O(N) memory — see §8), and because it ships *inside* the code that `make` builds, there is **no separate FMM library to port to Apple Silicon.** That is precisely the dependency that sinks the alternatives on Mac.
- **It models a real vibrating cone, not just a point source.** NumCalc supports active vibrating mesh elements via velocity boundary conditions, in addition to point sources and plane waves. **VERIFIED** (mesh2hrtf.org). This is exactly the diaphragm boundary condition the Overview's driver model needs.
- **Its input is plain text the app can generate.** The solver consumes a simple text file (`NC.inp`); the project documents generating it programmatically (e.g., an Octave script for the sphere benchmark). **VERIFIED** (Kreuzer et al. 2024). This satisfies the "the app sets up BEM for the user" requirement directly — our NumCalc backend's job is to write `NC.inp`.
- **It already solves your 48 GB problem.** NumCalc is single-threaded per process; you parallelize by running many frequency steps as separate processes. The shipped manager (`NumCalcManager.py`) estimates per-step RAM via a `-estimate_ram` flag and only launches a new process when enough memory is free (`max_ram_load`, `ram_safety_factor`, `max_cpu_load`). **VERIFIED** (Mesh2HRTF Python API docs; Wiki *NumCalc*). We can build our job manager on this exact mechanism (DR-02, §2 stage 4).

**Why the alternatives were rejected as *primary* (but kept around).**
- **bempp-cl** — a pure-Python JIT BEM library. Its fast path uses **OpenCL**, which Apple **deprecated** (since macOS 10.14) and which "is not available or has some features unavailable" on recent macOS; the Apple CPU OpenCL runtime is explicitly **incompatible** with bempp-cl, leaving a slower **Numba** fallback for native runs, and its **FMM depends on the external ExaFMM library** (a separate C++ build, uncertain on arm64). **VERIFIED** (bempp.com *Installation* and *Assembling Operators*). Net: on your Mac, bempp-cl realistically does **dense assembly via Numba**, which is O(N²) memory — fine for low/mid-frequency cross-checks, unworkable for whole-enclosure 20 kHz. **Kept** as the validation backend because its Python API is clean and an independent implementation is the best possible check on NumCalc.
- **COMSOL** — runs **natively on Apple Silicon** (M1+) and uses Apple's **ArmPL** BLAS by default since 6.2u2; PARDISO auto-falls-back to MUMPS. **VERIFIED** (COMSOL KB 1294/1307/1335; System Requirements 6.4). But the *automation* route is the catch: **LiveLink for MATLAB requires Rosetta 2** and **external MATLAB-function calls are not supported** on the native arm64 build. **VERIFIED** (COMSOL KB 1300/1307). So "COMSOL-driven-via-API" — the thing we'd need for a scripted backend — is the part that is awkward on this exact machine. (The COMSOL **Java API** is the cleaner programmatic route and does work, but wiring Python→Java→COMSOL is a heavier integration than this project should take on now.) Practical tip baked into §9: COMSOL on Apple Silicon defaults to performance cores but on SoCs newer than the KB articles (your M4 Max qualifies) you may need to **set the performance-core count manually** (Settings → Multicore and Cluster Computing) or it can land on efficiency cores. **Kept** as a manual fallback for two things NumCalc handles poorly: rigorous coupled lumped-T/S→acoustic driver modeling, and any case that wants structural cone behavior via hybrid FEM-BEM. Also note COMSOL's infinite-baffle BEM limitation (the Infinite Sound Hard Boundary "cannot have a hole in it," so a baffled driver wants FEM+PML there) — relevant to validation, not production.

**Consequences.**
- The Stage-0 gate is "does NumCalc build and run native arm64 on this Mac." If it unexpectedly fails, the abstraction layer means we pivot to bempp-cl (Numba, accepting lower frequency reach) or COMSOL-manual without rearchitecting. This is why DR-02 exists.
- We accept **CPU-only** solving (no GPU). On a 12-P-core M4 Max this is fine; the binding constraint is RAM, not FLOPs.
- We inherit a known NumCalc behavior: **non-convergence can appear at the highest frequencies** (the critical/irregular-frequency problem; mesh quality is the main cause). The post-processing has detection and workarounds. **VERIFIED** (Mesh2HRTF Wiki). Our assembly stage must detect non-converged steps and flag/interpolate them rather than silently emitting garbage (§2 stage 5, §9 R-04).

### DR-02 — Solver-abstraction layer: a narrow, data-only interface (`BEMBackend`)

**Decision.** Define one abstract interface that every backend implements. The rest of the app speaks only to this interface and never imports a solver directly. The interface is *data-in, data-out* — no solver-specific objects cross it.

**The contract (described, not coded).** A backend exposes four capabilities:
1. **`estimate(mesh, bc, frequencies, config) → ResourcePlan`** — return a per-frequency RAM/time estimate (NumCalc backend wraps `-estimate_ram`). Lets the job manager schedule against 48 GB.
2. **`prepare(mesh, bc, frequencies, config) → SolveSpec`** — translate the normalized inputs into whatever the backend needs on disk (NumCalc backend writes `NC.inp` files; bempp backend builds grid + operators in memory). Pure setup, no solving.
3. **`solve(SolveSpec, scheduler) → RawSolveResult`** — run the solve, honoring the scheduler's concurrency/RAM limits and writing checkpoints. Resumable.
4. **`extract(RawSolveResult, observation_points) → ComplexField`** — return complex pressure as a `[frequency × observation_point]` array at the requested sphere points, plus convergence flags per frequency.

**What crosses the boundary, in normalized form:** a `Mesh` (vertices, triangles, per-element surface-group tags); a `BoundaryConditions` object (which element groups are vibrating, with their prescribed complex normal velocity, and which are sound-hard); a `FrequencyGrid` (explicit array of Hz plus spacing metadata); `ObservationPoints` (the sphere grid unit vectors and radius); a `SolverConfig` (elements-per-wavelength target, solver tolerances, iteration cap, Burton–Miller on/off, medium properties). Out comes complex pressure plus diagnostics. **Nothing else.** No `NC.inp` paths, no bempp `GridFunction`s, leak upward.

**Why.** This is the one structural decision that protects the whole project from DR-01 being wrong, and from the Phase-2 future. It also makes the validation suite (§7) backend-agnostic: the same piston/sphere tests run against any backend.

**Rejected.** Letting the GUI/pipeline call NumCalc directly (faster to write, but welds the app to one solver and one OS quirk-set). Also rejected: a "lowest-common-denominator" interface so thin it can't express velocity BCs or convergence flags — those are exactly the things we need.

**Consequences.** Slightly more upfront design. Each new backend is a self-contained adapter. The contract doubles as documentation of "what BeamSimII needs from any physics engine."

### DR-03 — Meshing & geometry pipeline: **gmsh** core, driver-as-app-primitive, automatic element sizing, frequency-band meshes

**Decision.** Use **gmsh** (via its Python API) as the meshing engine. Treat the **enclosure** and the **driver diaphragm(s)** as *separate* objects: the enclosure may be imported or built; **each driver diaphragm is always an app-generated parametric surface** (a flat disc or a shallow cap, with radius, position, orientation, and a chosen velocity profile). The app unions the driver(s) into the enclosure shell at the chosen mounting locations, then meshes the combined watertight surface, **tagging the diaphragm elements automatically** because the app created them.

**Why this split is the single most important meshing decision.** The hardest, most failure-prone part of "import arbitrary CAD" is *interpreting* which face is the cone, which is the baffle, and whether the surface is watertight and correctly oriented. By making the diaphragm an app object rather than something we must recognize in your CAD, we sidestep the worst of it: the app *always* knows which elements get the vibrating-velocity boundary condition versus the sound-hard condition, with zero guessing. **HEURISTIC**, but it directly matches the Overview's driver model ("the cone is modeled as a simple vibrating surface inside the BEM mesh") and your stated worry about import meshing. The residual hard part shrinks to two localized problems: (a) the enclosure shell being watertight/manifold, and (b) the boolean union of driver cap and shell producing clean geometry. Those we attack with explicit health checks rather than hope.

**Automatic element sizing.** The user never sets element size. The app computes a target edge length from the top frequency of the run: with sound speed `c` and top frequency `f_max`, the wavelength is `λ_min = c / f_max`, and the target edge is `λ_min / N_epw`, where `N_epw` is elements-per-wavelength. **HEURISTIC/VERIFIED:** use **N_epw = 6–8** for NumCalc's constant collocation elements (the project's own rule of thumb; Kreuzer et al. 2024). Worked number: at 20 kHz, `λ ≈ 343/20000 ≈ 17.2 mm`, so edge ≈ 17.2/6 ≈ **2.9 mm** to 17.2/8 ≈ 2.1 mm. **VERIFIED** consistency with COMSOL's meshing guidance (~3.2 mm at 17 mm/5).

**Frequency-band remeshing (a memory/time lever, not a nicety).** Meshing the whole enclosure at 2.9 mm to solve a 200 Hz step is wasteful — a 200 Hz wavelength is ~1.7 m, so its elements could be ~24 cm. The app generates **a small set of band-appropriate meshes** (e.g., one for each octave band, or a few coarse/medium/fine tiers), each sized to the top of its band, and routes each frequency step to the coarsest mesh that resolves it. This is how we keep low-frequency steps cheap and reserve big meshes (and big RAM) for the genuinely high frequencies. **HEURISTIC**, aligned with the First Research report's "frequency-band remeshing" tactic and COMSOL/Simcenter precedent.

**Velocity-profile assignment.** On the diaphragm elements, the prescribed normal velocity is **constant across the cone, tapering to zero across the surround** (the conventional BEM cone boundary condition). **VERIFIED** as the standard convention (First Research report; standard practice). The single scalar level of that velocity comes from the T/S electrical model (DR-05); its *spatial* shape is the rigid-piston (or simple cap) profile. Cone breakup is explicitly out of scope (Overview).

**Geometry health checks (a real pipeline stage, see §2).** Before meshing, run: watertight/manifold check, self-intersection check, normal-orientation check, degenerate-face check, and a minimum-feature-size check against the target element edge. Auto-repair the cheap cases (duplicate faces, flipped normals); for the rest, surface a specific, plain-English error ("the enclosure has a 3 mm gap near the top-left edge; BEM needs a closed surface") rather than letting NumCalc fail cryptically at high frequency.

**Rejected.** (a) Requiring the user to paint the cone faces in the GUI for every import — fragile and tedious; we keep face-painting only as an *optional override*. (b) Name-tag-based cone identification from STEP metadata — depends on disciplined CAD naming we can't assume. (c) A heavier mesh stack (e.g., Netgen/MeshLab as the primary) — gmsh covers STEP via its built-in OpenCASCADE kernel, remeshes STL/OBJ, has a clean Python API, and runs on Apple Silicon via its pip wheel. **INFERRED** (gmsh ships arm64 macOS wheels; to be confirmed in setup).

**Consequences.** The parametric builder and the driver-overlay share one code path (drivers are always parametric), which simplifies the app. Imported enclosures get a "place driver here" step (position + orientation + radius) rather than a "find the cone" step. The boolean-union robustness becomes a tracked risk (§9 R-06).

### DR-04 — GUI framework: **PySide6 (Qt for Python)**

**Decision.** Build the desktop GUI with **PySide6**, the official Qt-for-Python binding (LGPL).

**Why.** It is the strongest fit for *this* app on *this* platform: it is the official, actively maintained Qt binding; it is **LGPL** (compatible with the open-source/self-contained goal and with shipping a Mac app); it has first-class **Apple Silicon** support; it produces genuinely native-feeling macOS windows, menus, file dialogs, and progress UI; and — critically for this project — it integrates cleanly with long-running background work (Qt's signal/slot + `QThread`/worker model) so a multi-day solve can report progress without freezing the window. It also embeds scientific plots well (Matplotlib via its Qt backend; and OpenGL/3-D views for the geometry and balloon displays). **INFERRED/HEURISTIC** (mature, widely used for exactly this class of scientific desktop tool).

**Rejected.** **Tkinter** (ships with Python, but dated look, weak 3-D, clumsy threading); **wxPython** (capable, smaller ecosystem, less momentum); **Dear PyGui / imgui** (great for tools but less "native macOS app," weaker file-dialog/menu integration); a **web/Electron** stack (heaviest dependency story, contradicts "self-contained," and pulls in a browser runtime for no benefit here); **Toga/BeeWare** (promising native-Python story but less mature for a plotting-heavy 3-D app). PySide6 dominates on the axes that matter: native feel, 3-D + plotting, long-task ergonomics, license, and Apple-Silicon maturity.

**Consequences.** Some Qt-specific patterns (the event loop, worker threads, signals/slots) become part of the codebase's vocabulary; Claude Code will write them. The GUI is deliberately the **last** thing built (§10) — it wraps a headless core that is already correct and tested. PyInstaller can later bundle a PySide6 app into a `.app` (§5).

### DR-05 — Driver model and the high-frequency efficiency fork

**Decision (model).** Confirm the Overview's approach unchanged: the **diaphragm is a vibrating surface inside the BEM mesh** (rigid piston or simple cap), so BEM computes aperture directivity *and* box/baffle interaction together without double-counting; the **T/S parameters plus a lossy-inductance model supply the per-driver on-axis terminal frequency response** (LF rolloff + box alignment from T/S; HF rolloff from voice-coil semi-inductance), applied as a **single complex multiplier across all directions** for that driver. Cone breakup is out of scope; approximate off-axis HF behavior is accepted.

**The terminal-response model, concretely (for later implementation).** The low-frequency end comes from the standard sealed/vented alignment driven by the small-signal parameters (`fs, Qts/Qes/Qms, Vas, Re, Le, Bl, Mms, Cms, Sd`), producing the cone-velocity magnitude/phase versus frequency and the system rolloff. **VERIFIED** as standard (Small/Thiele; COMSOL Lumped Loudspeaker Driver tutorial; First Research report). The high-frequency end must use a **lossy (semi-)inductance** model, not a plain `Le`, because the voice coil is a lossy inductor (eddy/skin effects in the pole piece). Recommended: the **Thorborg–Futtrup** semi-inductance + shorting model (the one REW uses), or an **LR-2 ladder**, fit to impedance ideally measured to 20 kHz. **VERIFIED** (Thorborg & Futtrup, *JAES* 59(9):612–627, 2011; Wright, *JAES* 38(10):749–754, 1990; First Research report). This terminal response is **stored separately** in the data file (DR-06/§3) so it is auditable and never silently entangled with the geometric BEM response.

**Decision (efficiency fork) — and a flag.** The Overview treats the analytic high-frequency splice as *optional* and says to decide from real Stage-1 timing. I am keeping that decision contingent as instructed, but I want to **flag, with reasoning, that on 48 GB I expect the splice to earn its keep in practice** above roughly a few kHz for whole-enclosure solves — so we should **build the splice as a first-class capability from the start**, not bolt it on later. Here is the reasoning:

- Element count grows as **N ∝ f²** (fixed elements/wavelength). **VERIFIED** (First Research report; standard BEM scaling). Per-step RAM at the top of the band reaches **8–20+ GB** for high-resolution whole-enclosure meshes. **VERIFIED**-ish (Mesh2HRTF wiki/grading paper figures; treat as order-of-magnitude).
- With ~40 GB usable after OS headroom, you can run **~12 cheap low-frequency steps in parallel** (you have 12 P-cores) but only **~2–3 of the most expensive top-octave steps at once**. So the **top octave serializes and dominates wall-clock**, exactly where the analytic piston model is *most* accurate (the response there is dominated by each driver's own `ka` beaming, which the closed form captures well). **HEURISTIC**, but well-supported.
- Therefore: capping BEM at a few kHz and splicing analytic above it is the highest-leverage single tactic for staying inside 1–2 days — and it costs little accuracy *for directivity* in the spliced region.

**The honest counter-point** (why it stays contingent): the M4 Max with ML-FMM may make full-band-to-20 kHz genuinely fit in an overnight-to-two-day run for typical enclosures, in which case full-band is simpler and avoids any splice artifact. We will **measure at Stage 1** (single driver, real enclosure) and set the production BEM cap from that measurement. The architecture supports both with a single tunable (`bem_cap_hz`): set it to `f_max` for full-band, or lower it to enable the splice.

**If splicing, the blend rule (to avoid magnitude/phase discontinuities).** Choose a crossover region about an octave wide centered near the cap. In that band, compute *both* the BEM result and the analytic piston+diffraction result, align them in **both magnitude and phase** at the lower edge (the BEM result is the reference there), and cross-fade with a smooth (e.g., raised-cosine) weight from BEM-only below to analytic-only above. Validate continuity: the magnitude step across the seam must stay small and the group delay must not jump (§7, §8 Stage-2 acceptance: seam discontinuity ≤ 0.5 dB; no audible group-delay step). **HEURISTIC**, aligned with the First Research report's Stage-2 guidance. The analytic diffraction term in the splice is **planar-only** (Vanderkooy/DED) — acceptable in the spliced HF region where each driver's own beaming dominates, but *not* a substitute for BEM in the diffraction-dominated mid-band. **VERIFIED** limitation (Urban et al., *JAES* 52(10), 2004).

**Rejected.** (a) Plain `Le` for the HF terminal response — known to be inaccurate. (b) Structural-FEM cone breakup — explicitly out of scope; would require a whole FEM subsystem and structural-acoustic coupling. (c) Per-driver minimum-phase-only terminal responses that discard inter-driver delay — would violate the phase-origin rule (see §3).

**Consequences.** The driver subsystem has two cleanly separated parts: a **geometric** part (velocity BC → BEM) and an **electrical/terminal** part (T/S + inductance → complex multiplier), combined only at assembly and stored both separately and combined. The splice is a configurable stage gated on Stage-1 timing.

### DR-06 — Language, runtime, and on-disk working format

**Decision.** **Python 3.12** as the implementation language and the floor version; **HDF5** as the native on-disk working format for results; complex arrays stored as native `complex128`. (Full tooling in §5; full schema in §3.)

**Why Python 3.12.** It has the broadest binary-wheel coverage across the whole stack we need (NumPy, SciPy, h5py, gmsh, PySide6, Matplotlib) and avoids the risk of a missing wheel stalling setup. **HEURISTIC** (conservative compatibility choice as of mid-2026; 3.13 is fine to revisit once every dependency ships arm64 wheels for it).

**Why HDF5.** One file holds the hierarchical, attribute-rich, complex-valued dataset; it is read natively by both **h5py** (Python) and **MATLAB**, both of which are in your workflow; and it scales to large tensors with chunking/compression. **VERIFIED** (HDF5/h5py and MATLAB both support it). SOFA (an AES69 standard that is itself HDF5-based) is offered as an *optional* export, not the working format, because the working format needs fields SOFA doesn't natively carry (e.g., our per-driver terminal-response split and BEM provenance).

**Consequences.** The data contract (§3) is expressed as an HDF5 layout. Phase 2 can be written against it directly in Python or MATLAB.

---

## 2. End-to-end Phase 1 pipeline

The pipeline is a sequence of **discrete, independently testable stages**, each with defined inputs, outputs, and failure surfacing. Stages communicate through plain data objects (the same normalized types DR-02 defines), so any stage can be unit-tested with a hand-made input and Claude Code can build/test the whole chain headless, before the GUI exists.

**Stage A — Geometry acquisition.**
*In:* either an imported file (STEP/STL/OBJ) or parametric-builder parameters (primitive type + dimensions). *Out:* an enclosure surface (B-rep for STEP, triangle soup for STL/OBJ) plus a list of driver placements (each: radius, position, orientation, profile).
*Failures & surfacing:* unreadable/empty file → "couldn't read this file; supported formats are STEP, STL, OBJ"; unit ambiguity (mm vs m) → prompt the user to confirm scale with a shown bounding-box size.

**Stage B — Geometry health check & repair** (DR-03).
*In:* enclosure surface + driver placements. *Out:* a single watertight, correctly-oriented, manifold combined surface with driver-diaphragm element groups tagged; or a specific error. *Failures & surfacing:* non-watertight, self-intersecting, bad normals, features smaller than the target element edge, or a failed driver/shell boolean → each reported as a located, plain-English problem with a suggested fix. Auto-repaired cases are logged, not hidden.

**Stage C — Frequency planning & multi-resolution meshing** (DR-03).
*In:* combined geometry + frequency range + N_epw + `bem_cap_hz`. *Out:* a `FrequencyGrid` and a small set of band-appropriate meshes, each with its diaphragm tags, plus the routing table (which frequency → which mesh). *Failures & surfacing:* a band's required element size produces a mesh too large to fit the RAM budget even at minimum concurrency → warn and recommend lowering `bem_cap_hz` (engage the splice) or coarsening; meshing failure on a band → report which band and why.

**Stage D — BEM setup** (DR-01/DR-02, `prepare`).
*In:* meshes + boundary conditions (diaphragm velocity from DR-05's terminal model at unit reference, sound-hard elsewhere) + medium properties + solver config. *Out:* a backend-specific `SolveSpec` (for NumCalc: a tree of `NC.inp` jobs, one set per driver per frequency-step range). *Failures & surfacing:* invalid BC (e.g., a driver group with zero elements) → caught here, before any solve burns time.

**Stage E — Solve (job management, parallelism, the budget)** (DR-01/DR-02, `solve`).
*In:* `SolveSpec` + the host's resource limits (48 GB, 12 P-cores). *Out:* raw per-frequency complex surface/field results on disk, plus per-step convergence flags. *How:* the scheduler calls `estimate` (NumCalc `-estimate_ram`), then launches frequency-step processes **highest-frequency-first** (most RAM-hungry first, so the memory peak happens when the queue is shortest), packing as many concurrent processes as fit in `usable_RAM = (48 GB − OS_headroom) × safety_factor`, capped at 12 by core count. Each completed step is **checkpointed** so a crash or a deliberate pause never loses finished work; the run is **resumable**. *Failures & surfacing:* a step that won't converge (the known top-frequency issue) is flagged, retried with more iterations, and if still failing is marked for interpolation rather than aborting the run; a step that OOMs is rescheduled at lower concurrency; a progress model (steps done / total, est. time remaining, current RAM) feeds the GUI (§6).

**Stage F — Per-driver superposition & assembly of `H`** (the contract producer).
*In:* raw field results (per driver, per frequency, at the sphere observation points) + the per-driver terminal responses (DR-05) + (if enabled) the analytic HF splice. *Out:* the `H[driver × frequency × direction]` complex tensor with all metadata, in memory, ready to write. *Key correctness steps:* (1) every driver was solved **independently at unit cone velocity**, so its slab of `H` is reusable and filter-able without re-solving (this is the superposition principle that makes Phase 2 cheap); (2) all responses are referenced to the **one global phase origin** with each driver's true time-of-flight preserved (no re-zeroing — §3); (3) the terminal multiplier is applied to form the "measurement-equivalent" response while the raw BEM response is retained separately; (4) flagged non-converged frequencies are interpolated in a controlled way (SH/min-phase domain) and **marked** in metadata. *Failures & surfacing:* a missing/failed driver solve → assembly refuses to emit a silently-incomplete tensor; it reports which driver/frequencies are missing.

**Stage G — Export.**
*In:* the assembled dataset. *Out:* the native **HDF5** working file (always), plus on request **`.frd`** per driver/per angle (VituixCAD), **CLF** balloon (interop), and optionally **SOFA**. *Failures & surfacing:* an export target that can't represent something (e.g., CLF angular resolution) → warn about what is lost, still write the native file.

Each stage boundary is a natural **self-test seam** (§7): feed Stage F a synthetic two-driver field and check the phase-origin handling; feed Stage C a known geometry and check element sizes; etc.

---

## 3. The output data schema — the contract with Phase 2

This is the most important artifact in the project. It is designed **backward from what the Phase-2 beamformer consumes**: complex per-driver directional responses that become the columns of a steering matrix, integrated over the sphere to build "accept" and "reject" covariance matrices. **VERIFIED** need (Luo, *Constant Directivity Loudspeaker Beamforming*, EUSIPCO 2024, arXiv:2407.01860; Feistel et al., AES Paper 7254, 2007).

### 3.1 The core tensor

```
H : [ driver  ×  frequency  ×  direction ]   complex128
      M             F            N
```

- **M** = number of drivers. **F** = number of frequency bins. **N** = number of sphere directions.
- Each `H[m, :, :]` is driver *m*'s full directional response — literally the block of columns Phase 2 assembles.
- Stored physically as **per-driver groups** on disk (one group per driver), because drivers are solved and reused independently; Phase 2 stacks them into the `[M × F × N]` view. Two companion arrays per driver are stored so nothing is entangled or un-auditable:
  - `H_bem` `[F × N]` — the **raw geometric response** at unit cone velocity (BEM only; baffle/box/diffraction/aperture directivity). This is the reusable, filter-able physics.
  - `terminal_response` `[F]` — the **per-driver complex on-axis multiplier** (T/S LF + semi-inductance HF, DR-05).
  - `H_full` `[F × N]` = `H_bem × terminal_response` (broadcast over directions) — the **measurement-equivalent** response Phase 2 uses by default.
  - `splice_applied` (bool) and, if true, `bem_cap_hz` and the blend description, so the spliced region is explicit.

### 3.2 Sphere sampling (directions) and integration weights

**Decision:** default to a **Lebedev grid** as the primary scheme, with **Fliege–Maier** and **spherical t-designs** and **icosphere** as selectable alternatives. **Why Lebedev:** it comes with exact **quadrature weights**, which makes the power and directivity-index integrals (and the Phase-2 covariance integrals) clean and well-conditioned — you integrate any function over the sphere as `Σ w_i · f(direction_i)`. **VERIFIED** rationale (First Research report; standard spherical-quadrature practice). **Never** a naive latitude/longitude grid (it oversamples the poles and integrates poorly).

Stored under a `directions` group:
- `unit_vectors` `[N × 3]` float64 — Cartesian direction cosines.
- `weights` `[N]` float64 — quadrature weights (sum to 4π or to 1; the convention is recorded in an attribute).
- `theta_phi` `[N × 2]` float64 — convenience spherical coordinates (convention recorded).
- attributes: `scheme` (e.g., "lebedev"), `order`, `weight_convention`.

**Resolution-versus-frequency reasoning.** The angular detail of a source's directivity grows with frequency and source size, governed by `k·r_source` (with `k = 2π f / c`). The spherical-harmonic order `Nsh` needed to represent the pattern rises roughly as `Nsh ≈ ⌈k·r_source⌉` plus a margin. **HEURISTIC/VERIFIED** (spherical-array/SH literature; First Research report). To keep `H` a clean rectangular tensor, we use **one fixed grid for all frequencies, sized for the worst case at `f_max`** (or at `bem_cap_hz` if splicing and the analytic tail is evaluated on the same grid). The grid is chosen to **over-resolve** the SH order the optimizer will use, to avoid spatial aliasing — concretely, a Lebedev order supporting degree ≥ `Nsh(f_max) + margin`. Practical point count is commonly a few hundred to ~2,000 directions; for direct GLL/CLF interoperability a ≤5° balloon is the target. **VERIFIED** reference points (NextGenAudio used 1,850 directions for 0–20 kHz; GLL/CLF use 5°; First Research report). Oversampling at low frequencies is cheap and harmless.

### 3.3 Frequency grid

- `frequencies` `[F]` float64 (Hz), explicit, plus attribute `spacing` ("log"/"linear"/"fractional-octave") and the fractional-octave value if applicable.
- **Default:** logarithmic, ~1/12-octave for directivity work (coarser than the 1/24-octave you'd want for crossover detail, because directivity varies more smoothly with frequency — the First Research report notes a coarser grid suffices for directivity). User-settable.
- **Sparse-simulate + interpolate** is supported: simulate a sparse set, interpolate to the stored grid in the **SH + minimum-phase domain**, and **mark interpolated bins** in metadata. **VERIFIED** that this saves large compute (~86% in a cited study) but must be done carefully to avoid artifacts (First Research report).

### 3.4 The single-phase-origin convention (the highest-risk rule)

This is stated at length because getting it wrong silently mis-steers every Phase-2 beam.

**The rule.** Every value in `H_bem[m]` is the complex pressure at the observation point (at radius `r_obs`, in the given direction) produced by driver *m* vibrating with unit normal velocity, **with phase referenced to a single common time origin** (the excitation at `t = 0`) and **a single common spatial origin** (the global coordinate origin, conventionally `(0,0,0)`). Because driver *m* physically sits at position `p_m ≠ origin`, its responses **naturally carry the extra path-length phase** corresponding to its real location. **We do not remove that phase. We do not minimum-phase-ify or re-zero any driver independently.** The inter-driver phase differences *are the information* Phase 2 uses to steer.

**Why it's easy in simulation (an advantage over measurement).** In a real anechoic measurement, the acoustic center rarely coincides with the turntable pivot, producing an artificial phase error that grows with frequency (the "egg-shaped pattern" problem) that must be corrected. **VERIFIED** (Trott 1977 acoustic-center definition; VituixCAD's dual-channel note; First Research report). In simulation we simply define one origin and keep every driver referenced to it — the geometry gives us the correct time-of-flight for free.

**Optional decomposition (allowed, but the origin is sacred).** If useful, a driver's response may be decomposed into minimum-phase × excess-phase (pure delay / time-of-flight), as long as the common origin and the true delay are preserved and recombine exactly. Stored fields supporting this are optional.

**Enforcement in code (for the build).** A dedicated assertion in Stage F: assemble a synthetic two-source field, compare the summed `H` against a direct two-source BEM solve, and require agreement (this is the Stage-3 acceptance test, §8). Any refactor that re-zeros a driver fails this test loudly.

### 3.5 Required metadata (HDF5 attributes / groups)

Root-level and per-driver, everything needed to interpret the data without external context:

- **Schema/provenance:** `schema_version`, `beamsim_version`, `created_utc`, `solver_backend`, `solver_version`, mesh stats per band (element counts, N_epw), convergence summary (which frequencies were flagged/interpolated).
- **Coordinate system & phase origin:** axis convention, `phase_origin` `[3]`, units of length.
- **Drivers (`drivers/<id>/attrs`):** `name`, `position` `[3]`, `orientation` (axis/normal), `radius`, `profile` (piston/cap + parameters), `ts_params` (the full small-signal set), `terminal_response_model` (e.g., "thorborg-futtrup"), `diaphragm_area` (for normalization).
- **Observation:** `observation_radius`, `far_field` (bool), and how pressure is normalized (see below).
- **Sphere:** scheme, order, weights convention (in the `directions` group).
- **Frequency:** grid + spacing + interpolated-bin mask.
- **Medium / reference conditions:** `speed_of_sound`, `air_density`, `temperature`, `humidity`, `pressure`, air-attenuation model used (or "none").
- **Units / reference level:** the pressure convention — recommended **pressure at `r_obs` for unit cone velocity** in the raw `H_bem`, and an SPL reference (e.g., SPL @ 1 m / 2.83 V) attached to `H_full` once the terminal model and sensitivity are applied. The convention is recorded explicitly so Phase 2 never guesses.

**Observation convention decision.** Store pressure at a **fixed finite radius `r_obs`** (e.g., 1 m or 2 m) with a `far_field` flag, rather than a far-field directivity factor. This mirrors a real anechoic measurement (the project's stated goal — "mimic real anechoic measurements") and matches what `.frd`/VituixCAD expect; Phase 2 can convert to far-field if it wants. **HEURISTIC.**

### 3.6 On-disk formats

- **Native working format: HDF5** (`.h5` or a branded `.bsim` extension that is HDF5 inside). The layout:

```
/                                  (root; all root attributes above)
  /frequencies                     [F] float64
  /directions/
      unit_vectors                 [N×3] float64
      weights                      [N]   float64
      theta_phi                    [N×2] float64
  /drivers/
      /<driver_id>/
          H_bem                    [F×N] complex128
          terminal_response        [F]   complex128
          H_full                   [F×N] complex128
          (attrs: name, position, orientation, radius, profile, ts_params, ...)
```

- **`.frd` per driver / per angle** (text: frequency, magnitude dB, phase deg) — the pragmatic **VituixCAD** path; VituixCAD expects per-single-driver responses (it sums them itself) and ingests BEM directivity. **VERIFIED** (First Research report).
- **CLF** balloon — the open balloon format for cross-checking against room-acoustics tools. **VERIFIED** as the open interop choice (First Research report).
- **SOFA** (optional) — AES69, HDF5-based; a natural later add for the directivity convention.

This is concrete enough that Phase 2 can be written against it today.

---

## 4. Project layout and module organization

A clean separation of concerns, so the physics core, the solver backends, the geometry/meshing, the IO, the validation, and the GUI evolve independently. The GUI imports the core; the core never imports the GUI.

```
beamsim2/                              ← the git repository root
├── README.md                         ← what this is, how to set up, how to run
├── CHANGELOG.md                      ← human-readable history of versions (§11)
├── LICENSE                           ← open-source license of your choice
├── pyproject.toml                    ← project metadata + dependencies (§5)
├── uv.lock                           ← exact pinned dependency versions (§5)
├── .gitignore                        ← what git must NOT track (§11)
├── .python-version                   ← pins Python 3.12 for the repo
│
├── docs/
│   ├── BEAMSIMII_Project_Overview.md     ← copies of the authoritative docs
│   ├── BEAMSIMII_First_Research.md
│   ├── BEAMSIMII_Gameplan.md             ← this file
│   ├── DATA_CONTRACT.md                  ← §3, extracted as the standalone contract
│   ├── CODING_STANDARDS.md               ← §5
│   └── handoffs/
│       └── HANDOFF_<date>_<topic>.md     ← one per major session (§ Continuity)
│
├── src/
│   └── beamsim2/
│       ├── __init__.py
│       │
│       ├── core/                     ← physics-agnostic shared types & math
│       │   ├── types.py              ← Mesh, BoundaryConditions, FrequencyGrid,
│       │   │                            ObservationPoints, SolverConfig, ComplexField
│       │   ├── sphere.py             ← Lebedev/Fliege/t-design grids + weights
│       │   ├── units.py             ← medium properties, c(T,RH,P), air attenuation
│       │   └── sh.py                 ← spherical-harmonic transforms / interpolation
│       │
│       ├── geometry/                 ← DR-03
│       │   ├── primitives.py         ← parametric box/cylinder/sphere/driver-cap
│       │   ├── import_io.py          ← STEP/STL/OBJ loading
│       │   ├── health.py             ← watertight/manifold/orientation/feature checks
│       │   ├── assemble.py           ← driver↔enclosure boolean union + tagging
│       │   └── mesh.py               ← gmsh driver: sizing, band meshes, routing
│       │
│       ├── driver/                   ← DR-05
│       │   ├── thiele_small.py       ← LF alignment → cone velocity & sensitivity
│       │   ├── inductance.py         ← Thorborg–Futtrup / LR-2 semi-inductance (HF)
│       │   ├── velocity_profile.py   ← piston/cap normal-velocity BC shapes
│       │   └── terminal.py           ← assemble the per-driver complex multiplier
│       │
│       ├── backends/                 ← DR-01/DR-02 (the abstraction)
│       │   ├── base.py               ← the BEMBackend interface (estimate/prepare/
│       │   │                            solve/extract) + the data contract types
│       │   ├── numcalc/              ← PRIMARY
│       │   │   ├── adapter.py        ← implements BEMBackend
│       │   │   ├── ncinp_writer.py   ← generate NC.inp from normalized inputs
│       │   │   ├── scheduler.py      ← RAM-aware, highest-freq-first, resumable
│       │   │   └── reader.py         ← parse be.out → ComplexField + convergence
│       │   ├── bempp/                ← SECONDARY (validation; Numba on Mac)
│       │   │   └── adapter.py
│       │   └── comsol/               ← MANUAL fallback stub (Java-API later)
│       │       └── adapter.py
│       │
│       ├── splice/                   ← DR-05 fork
│       │   ├── piston.py             ← analytic 2J1(ka sinθ)/(ka sinθ) + DED edge
│       │   └── blend.py              ← magnitude+phase-matched crossfade at the cap
│       │
│       ├── assembly/                 ← Stage F
│       │   ├── superpose.py          ← per-driver unit-velocity → H_bem
│       │   ├── phase_origin.py       ← enforce single origin; assertions
│       │   └── tensor.py             ← build H_full, attach metadata
│       │
│       ├── io/                       ← Stage G
│       │   ├── hdf5_store.py         ← native read/write (the contract)
│       │   ├── frd_export.py         ← VituixCAD
│       │   ├── clf_export.py         ← balloon
│       │   └── sofa_export.py        ← optional
│       │
│       ├── validation/               ← §7 (wired as tests + a callable harness)
│       │   ├── analytic_piston.py
│       │   ├── sphere_benchmark.py
│       │   ├── convergence.py
│       │   └── power_di.py
│       │
│       ├── pipeline/                 ← orchestration of Stages A–G
│       │   ├── run.py                ← the headless end-to-end runner
│       │   └── progress.py           ← progress model the GUI subscribes to
│       │
│       └── gui/                      ← DR-04 (built LAST; imports core, never vice-versa)
│           ├── app.py
│           ├── geometry_view.py      ← 3-D enclosure + driver placement
│           ├── parameters_panel.py   ← T/S, sphere, frequency inputs
│           ├── run_monitor.py        ← long-solve progress
│           └── results_view.py       ← polars, balloons, directivity maps, on-axis
│
└── tests/
    ├── test_sphere_grids.py
    ├── test_geometry_health.py
    ├── test_ncinp_writer.py
    ├── test_phase_origin.py          ← the two-driver superposition test (critical)
    ├── test_analytic_piston.py
    ├── test_sphere_benchmark.py
    ├── test_power_di.py
    └── test_hdf5_roundtrip.py        ← contract stability
```

Ownership in one line each: **core** owns shared types and sphere/medium/SH math; **geometry** owns everything from raw shapes to tagged band meshes; **driver** owns the electrical/terminal model and the velocity-BC shapes; **backends** own the physics engines behind one interface; **splice** owns the analytic HF tail and its blend; **assembly** owns superposition, phase-origin discipline, and the tensor; **io** owns all on-disk formats; **validation** owns the trust tests; **pipeline** orchestrates; **gui** is a thin shell over the proven core.

---

## 5. Coding standards and macOS tooling

### 5.1 Coding standards (the short version; lives in `docs/CODING_STANDARDS.md`)

- **Complete and runnable, never pseudocode, never "fill in the rest."** If something is too long for one response, say so and split into explicitly labeled parts.
- **Dimensional comments on every significant array**, in the project's notation: e.g. `# H_bem: [F x N] complex128` , `# unit_vectors: [N x 3] float64`. Shapes are part of the type.
- **Every function** carries a plain-English block explanation of what it does, plus parameter and return descriptions (Google- or NumPy-style docstrings).
- **Label physics/technical claims** in comments and docs as **VERIFIED / INFERRED / HEURISTIC** with author/year citations, matching this document.
- **Bridge to acoustics** in explanations wherever a DSP/numerical concept has an acoustic analogue (the reader is an acoustics expert, not a programmer).
- **A self-test for every subsystem.** The full `tests/` suite must pass before a session closes. Validation tests (§7) are part of the suite.
- **Type hints** on public functions; **black** for formatting and **ruff** for linting so style is automatic and never a discussion.
- **No silent behavior changes** to locked decisions; flag with reasoning.

### 5.2 macOS tooling plan (Apple Silicon)

- **Python:** 3.12, pinned in `.python-version`. Install via the official python.org arm64 installer or `pyenv`.
- **Environment & dependencies: `uv`.** Use `uv` to create the virtual environment and to manage dependencies via `pyproject.toml` + a committed **`uv.lock`**. This gives exact, reproducible versions — which directly addresses your past "lost track of versions" pain *at the dependency level*: anyone (or any future session) recreates the identical environment from the lockfile. **HEURISTIC** (modern best practice; fast and simple, and Claude Code works well with it).
- **Core libraries:** `numpy`, `scipy`, `h5py` (HDF5), `gmsh` (meshing; arm64 wheel), `matplotlib` (plots; Qt backend), `PySide6` (GUI). For sphere grids, a Lebedev/quadrature source (a small dependency or vendored tables). For `.frd`/CLF/SOFA, lightweight writers (SOFA via `python-sofa`/`sofar` if/when added).
- **The BEM solver build (NumCalc) — the one non-pip dependency.** NumCalc is **not** a Python package; it is a C++ executable compiled from the Mesh2HRTF source. The plan:
  1. Install Apple's command-line tools (`xcode-select --install`) to get `clang`/`make`.
  2. Clone Mesh2HRTF; in `…/NumCalc/Source` run `make`. **VERIFIED** this is the documented Mac build path (Mesh2HRTF Wiki).
  3. Confirm it runs native arm64 (`file NumCalc` should report `arm64`); this is the **Stage-0 gate**.
  4. **Bundle/locate it:** for development, place the binary on `PATH` (or record an absolute path in app config); for a distributable `.app`, vendor the compiled `arm64` binary inside the bundle's `Resources/` and have the NumCalc adapter resolve that path. Because NumCalc has no exotic runtime dependencies and FMM is statically in the binary, bundling is just shipping one executable. **INFERRED.**
- **App packaging (later):** **PyInstaller** to produce a self-contained `.app` for Apple Silicon, with the NumCalc binary and any data tables included as bundled resources. **INFERRED** (standard path; verify at packaging time).
- **Continuous testing:** run `pytest` locally; optionally add a **GitHub Actions** workflow (§11) that at minimum lints and runs the pure-Python tests on push (the NumCalc-dependent tests run locally, since CI won't have the compiled solver unless we add a build step later).

---

## 6. GUI design (sketch)

The GUI's whole job is to let an acoustics expert drive a BEM simulator **without ever seeing BEM**. The user thinks in enclosures, drivers, T/S parameters, observation spheres, and polar plots; the app silently turns those into meshes, boundary conditions, frequency lists, and solver jobs.

**Inputs the user provides (and nothing about BEM):**
- **Enclosure geometry** — import a STEP/STL/OBJ, or build a primitive (box/cylinder/sphere with dimensions) in the builder.
- **Driver(s)** — for each: a rough shape/size (radius, flat disc or shallow cap), a **position** and **orientation** on/in the enclosure, and its **T/S parameters** (entered in a familiar form, like a driver datasheet).
- **Observation sphere** — radius, and a sampling density chosen from presets ("standard / fine / balloon-5°") that map internally to Lebedev orders.
- **Frequency range** — low and high limits, and a resolution preset (mapped to fractional-octave spacing). Optionally a "fast (sparse + interpolate)" toggle.
- Everything else (element size, N_epw, solver tolerances, band meshing, job concurrency, the BEM cap / splice) has **sensible defaults** and lives behind an "Advanced" disclosure for the curious.

**Rough screen flow:**

```
┌─ BeamSimII ────────────────────────────────────────────────┐
│ [1] Geometry      [2] Drivers     [3] Simulation   [4] Results │
├────────────────────────────────────────────────────────────┤
│  [1] GEOMETRY                                                 │
│   ( ) Import file…      (•) Build primitive                   │
│   ┌──────────────┐   Type:[Box ▼]  W[ ] H[ ] D[ ]            │
│   │  3-D view of  │   Health: ✔ watertight  ✔ normals         │
│   │  enclosure +  │                                            │
│   │  driver discs │   [+ Add driver]                           │
│   └──────────────┘                                            │
│  [2] DRIVERS                                                  │
│   Driver 1: radius[ ]  pos[x,y,z]  aim[ ]  [T/S params…]      │
│   Driver 2: …                                                 │
│  [3] SIMULATION                                              │
│   Freq: [20]–[20000] Hz   Resolution:[1/12 oct ▼]            │
│   Sphere: radius[1 m]  density:[Standard ▼]                  │
│   ▸ Advanced (N_epw, BEM cap, concurrency, splice)           │
│   [Estimate]→ "≈ X GB peak, ≈ Y hours"     [Run]             │
│  [4] RESULTS  (after solve)                                  │
│   tabs: On-axis | Horizontal polar | Vertical polar |        │
│         Balloon (3-D) | Directivity map (freq×angle)         │
│   [Export…] → HDF5 / .frd / CLF / SOFA                       │
└────────────────────────────────────────────────────────────┘
```

**Launching and monitoring a multi-day solve.** The "Estimate" button calls the backend's `estimate` and shows a predicted **peak RAM and wall-clock** *before* committing — so the user can dial the BEM cap or resolution if the estimate is too big. "Run" starts the solve on a **background worker** (Qt thread) so the window stays responsive; a monitor view shows steps-done/total, a frequency-by-frequency status grid (queued / running / done / flagged), current RAM use, and estimated time remaining. Because Stage E checkpoints, the user can **pause/quit and resume**; if a step is flagged non-converged, the monitor shows it amber and notes it will be interpolated.

**Presenting results.** The standard acoustics views, drawn from the `H` tensor: **on-axis frequency response**; **horizontal and vertical polar** plots (selectable frequencies or animated across frequency); a **3-D balloon**; and a **directivity map** (frequency on one axis, angle on the other, level as color) — the view that best shows beamwidth-vs-frequency, which is the whole point of a directivity-controlled speaker. All read-only views of the same dataset that gets exported.

---

## 7. Validation plan (wired in as automated self-tests)

The simulator must be **trustworthy without hardware.** Trust comes from agreement with closed-form solutions and from internal consistency. Every check below is both a `pytest` test and a callable function in `validation/`, run against *any* backend through the abstraction layer.

**V-1 — Analytic piston-in-a-baffle cross-check.** Simulate a flat circular piston and compare the BEM directivity to the closed form `D(θ) = 2·J₁(ka·sinθ)/(ka·sinθ)`. **VERIFIED** reference (standard; First Research report; COMSOL Lumped Loudspeaker Driver tutorial compares to this). Because true *infinite*-baffle BEM is awkward (and impossible in COMSOL's BEM — the infinite sound-hard boundary "cannot have a hole"), validate via a **large finite baffle** and compare in the forward hemisphere where edge effects are small, across several `ka` values, with mesh convergence. *Pass:* mean magnitude error ≤ **1 dB** and small phase error in the valid angular region (matching the First Research report's ≲1 dB acceptance).

**V-2 — Sphere benchmark (exact closed form).** Radiation from a pulsating sphere (monopole) and an oscillating sphere (dipole) have exact solutions; this is how NumCalc itself was validated (cube- and icosahedron-based sphere meshes). **VERIFIED** (Kreuzer et al. 2024). Compare BEM pressure magnitude and phase to the analytic sphere solution over frequency. *Pass:* magnitude error within the method's known accuracy (NumCalc's constant-collocation elements achieve relative error ~10⁻³–10⁻²; **VERIFIED** Kreuzer et al. 2024) → a dB threshold of ≤ ~0.5 dB in the converged regime, plus a phase threshold.

**V-3 — Mesh-convergence check.** Re-solve a fixed case at N_epw = 6 → 8 → 10 and confirm results stop changing. *Pass:* change from 8→10 below a small threshold (e.g., ≤ 0.25 dB spatially-averaged). Respect the hard floor near 2 elements/wavelength below which results are nonsense. **VERIFIED** (6–8 rule, Kreuzer et al. 2024; floor, First Research report).

**V-4 — Radiated-power / directivity-index integration.** Integrate `|p|² · weights` over the sphere to get radiated power and the directivity index, using the grid's quadrature weights. *Pass on textbook anchors:* DI → **0 dB** for a monopole (omnidirectional); DI → **3 dB** for an omni source on a large baffle (radiating into half-space) at low `ka`; plus an energy-conservation/reciprocity sanity check. **VERIFIED** anchors (standard; First Research report).

**V-5 — Phase-origin / superposition correctness (the critical one).** Build a synthetic two-driver case; assemble `H` via per-driver superposition (Stage F) and compare to a **direct two-driver BEM solve**. *Pass:* the summed field matches the direct solve to within solver tolerance. This is the test that catches any accidental per-driver re-zeroing (§3.4). It runs in CI as a guardrail on the highest-risk rule.

**V-6 — BEM-vs-analytic diffraction (planar regime).** In the planar-baffle regime, BEM should match the DED/Vanderkooy diffraction prediction; divergence flags either a mesh problem or a genuine 3-D effect. **VERIFIED** (Urban et al. 2004; First Research report). Used as a diagnostic, not a hard gate.

**Error metrics, defined once and reused:** per-frequency magnitude error (dB, mean and max over directions); phase error (degrees); spatially-averaged dB error (power-weighted with the quadrature weights); DI error (dB); and the convergence/reciprocity residuals. Thresholds as stated per test.

---

## 8. Phasing and milestones for Phase 1

Mapped to the First Research report's Stage 0–4, updated for the **~1–2 day** budget and your **M4 Max / 48 GB**. Each stage has a concrete acceptance gate; nothing proceeds until its gate is green.

**Stage 0 — Toolchain prototype on a trivial case.**
*Do:* build NumCalc native arm64 on the Mac (the gate); stand up the abstraction layer + NumCalc adapter minimally; wire V-2 (sphere) and V-1 (piston). *Gate:* NumCalc builds and runs arm64; V-1 within ≤1 dB and V-4 anchors (DI→0/3 dB) pass; the abstraction layer round-trips a solve. *Why first:* this de-risks the entire backend decision before any app investment. **If the build fails, this is where we pivot** (bempp-Numba or COMSOL-manual) — cheaply.

**Stage 1 — Single driver in the real enclosure, mid-band.**
*Do:* mesh your CAD (driver-as-primitive + boolean union + health checks), impose the rigid-piston velocity BC from a placeholder cone velocity, solve BEM up to ~2–5 kHz over a Lebedev sphere; confirm the **baffle step, diffraction ripple, and edge-rounding** effects appear (the NextGenAudio model showed rounding "significantly affects the polar and on-axis response above ~1 kHz" — **VERIFIED**, First Research report). *Gate:* qualitative diffraction features present and stable under mesh refinement; **and the timing is measured** — record peak RAM and wall-clock per frequency step at the top of the band. *Decision point:* set the production `bem_cap_hz` from this timing (full-band vs splice, DR-05). If a single mid-band solve already strains 48 GB or wall-clock, the splice is confirmed as the default.

**Stage 2 — Add the driver electrical chain (and the splice if Stage-1 says so).**
*Do:* implement the T/S LF alignment + Thorborg–Futtrup/LR-2 semi-inductance terminal response (DR-05), applied as the per-driver complex multiplier; if splicing, add the analytic piston+DED tail above `bem_cap_hz` with the magnitude+phase-matched blend. *Gate:* terminal response matches a reference RLC/semi-inductance fit; **splice seam discontinuity ≤ 0.5 dB** with no group-delay step (if splicing). **VERIFIED** thresholds basis (First Research report).

**Stage 3 — Multi-driver, full sphere, all drivers, with the data contract.**
*Do:* solve each driver independently at unit velocity; assemble `H` referenced to one global origin; pass **V-5** (the two-driver superposition test); write the native HDF5 and the `.frd`/CLF exports. *Gate:* V-5 passes (phase-origin discipline proven); a multi-driver dataset exports and re-imports losslessly (`test_hdf5_roundtrip`); full-run wall-clock within ~1–2 days for a representative enclosure (using the Stage-1 cap decision).

**Stage 4 — Hand off to Phase 2 and close the loop.**
*Do:* feed the `H` tensor into a beamforming check — build the accept/reject covariance matrices (Luo) or run the GLL complex-summation forward model — and **reproduce a known constant-directivity result** (a CBT Legendre-shaded pattern, or the NextGenAudio cardioid). *Gate:* a known target pattern is reproduced from BeamSimII data, demonstrating the contract is correct end-to-end. **VERIFIED** targets basis (Keele CBT; Luo 2024; First Research report).

The GUI is developed **in parallel from Stage 2 onward but is never on the critical path** — the headless pipeline and its tests are what gate each stage.

---

## 9. Risk register

Each risk: what it is, likelihood/impact, and a concrete mitigation already reflected in the plan.

**R-01 — NumCalc fails to build/run native arm64 on this Mac (or FMM misbehaves).** *Likelihood:* low (Mac is a preferred platform, FMM is in-core; **VERIFIED**). *Impact:* high if it happened. *Mitigation:* it is the **Stage-0 gate**, hit first and cheaply; the **abstraction layer** (DR-02) lets us pivot to bempp-cl (Numba, lower frequency reach) or COMSOL-manual without rearchitecting; document the exact build steps so the result is reproducible.

**R-02 — Phase-reference discipline is silently violated → mis-steered beams.** *Likelihood:* medium (it's a subtle, easy-to-introduce bug). *Impact:* very high (Phase 2 fails invisibly). *Mitigation:* the schema mandates one origin (§3.4); a dedicated **assertion** in `assembly/phase_origin.py`; the **V-5 two-driver test** runs in CI as a permanent guardrail; an explicit rule in `CODING_STANDARDS.md` forbidding per-driver re-zeroing.

**R-03 — Cone-breakup limitation (no true breakup from T/S + CAD).** *Likelihood:* certain (it's a scope boundary, not a bug). *Impact:* medium (top-octave directivity detail is approximate). *Mitigation:* documented scope boundary (Overview); rigid-piston/cap BC; the GUI/metadata **mark** the spliced/approximate HF region so results aren't over-trusted; this matches the accepted scope.

**R-04 — Compute/memory pressure on 48 GB; top-octave dominates.** *Likelihood:* high. *Impact:* medium (long runs, or a cap needed). *Mitigation:* **RAM-aware, highest-frequency-first scheduler** built on NumCalc's `-estimate_ram`/`max_ram_load` (**VERIFIED** mechanism); **frequency-band remeshing** (cheap low-frequency steps); **analytic HF splice** as a first-class lever (DR-05); **per-driver-once superposition** so filters never trigger re-solves; **checkpointing/resume** so long runs survive interruption; an **Estimate** step that predicts peak RAM/time before committing.

**R-05 — Analytic-splice artifacts (magnitude/phase steps at the seam).** *Likelihood:* medium (only if splicing). *Impact:* medium. *Mitigation:* magnitude **and** phase matched at the lower seam, octave-wide raised-cosine blend, continuity validated at Stage-2 gate (≤0.5 dB, no group-delay step); the planar-only nature of the analytic diffraction is confined to the HF region where each driver's own beaming dominates.

**R-06 — Mesh quality from rough/imported user geometry.** *Likelihood:* high (you flagged it). *Impact:* medium-high (bad mesh → NumCalc non-convergence at HF). *Mitigation:* **driver-as-app-primitive** removes the worst guessing (DR-03); an explicit **geometry health-check stage** with auto-repair and located, plain-English errors; the **parametric builder** as an always-clean path; the **non-convergence detector** (R-07) downstream.

**R-07 — High-frequency non-convergence (critical/irregular frequencies).** *Likelihood:* medium (a known NumCalc behavior). *Impact:* medium. *Mitigation:* Burton–Miller is already used (stabilizes it; **VERIFIED**); the solver stage **detects** non-converged steps, retries with more iterations, and **flags + interpolates** rather than emitting garbage; flagged bins are marked in metadata. **VERIFIED** behavior basis (Mesh2HRTF Wiki).

**R-08 — Long-run reliability (crash/power loss mid-solve).** *Likelihood:* medium over multi-day runs. *Impact:* medium. *Mitigation:* per-step **checkpointing** and **resumable** jobs (Stage E); disk-space checks for intermediate results.

**R-09 — Format/version drift between Phase 1 and Phase 2.** *Likelihood:* medium over time. *Impact:* medium. *Mitigation:* `schema_version` attribute; a standalone `DATA_CONTRACT.md`; `test_hdf5_roundtrip` guards the format; semantic versioning + CHANGELOG (§11) record any contract change deliberately.

**R-10 — COMSOL automation on Apple Silicon (if ever used as a backend).** *Likelihood:* low (it's a manual fallback). *Impact:* low-medium. *Mitigation:* use COMSOL **manually** for special cases; if scripted later, use the **Java API** route, not LiveLink-for-MATLAB (which needs Rosetta); pin **performance cores** in COMSOL settings on the M4 Max. **VERIFIED** caveats (COMSOL KB).

---

## 10. Recommended Phase-1 build order

A prioritized sequence for when coding begins. **No implementation in this document** — this is the order, with the rationale that the physics core and the data contract come first and the GUI comes last (a thin shell over a proven, tested engine). Each item lands with its tests.

1. **Repository, tooling, and the data contract on paper.** Set up the repo, `pyproject.toml`/`uv.lock`, `.gitignore`, CI skeleton, and write `DATA_CONTRACT.md` (extract §3). *Build the contract before the thing that produces it.*
2. **`core/types.py` + `core/sphere.py` + `core/units.py`.** The normalized data types and the sphere grids/weights — everything else depends on these.
3. **`backends/base.py` (the interface) + the NumCalc adapter, minimally.** Get a solve to run through the abstraction (the Stage-0 gate).
4. **`validation/` (sphere + piston + power/DI) wired as tests.** Trust from day one; these run against the adapter from #3.
5. **`geometry/` (primitives → health → assemble → mesh).** The parametric path first (always clean), then import + health checks.
6. **`backends/numcalc/ncinp_writer.py` + `scheduler.py` + `reader.py`.** Full automatic BEM setup and the RAM-aware, resumable, highest-freq-first job manager.
7. **`assembly/` (superpose → phase_origin → tensor) + `io/hdf5_store.py`.** Produce the contract; pass the V-5 two-driver test.
8. **`driver/` (thiele_small → inductance → terminal) and `splice/` (gated on Stage-1 timing).** The electrical chain and, if needed, the HF splice with the blend.
9. **`io/frd_export.py` + `clf_export.py` (+ optional SOFA).** Interoperability exports.
10. **`gui/` (DR-04), last.** Wrap the working headless pipeline; geometry/placement view, parameters, run monitor, results views.
11. **`bempp/` validation backend (when time allows).** An independent cross-check on NumCalc through the same interface.

---

## 11. Version control, GitHub, and project setup (written for no prior software-dev experience)

You said you have a GitHub account you've never used, and that in a past project you lost track of changes and versions. This section fixes that. It is deliberately plain. The mechanics will mostly be performed by Claude Code on your behalf, but you should understand the concepts so you can direct it and review what it does.

### 11.1 The mental model (in plain terms)

**Git** is a time machine for your project folder. Every time you reach a sensible stopping point, you take a **snapshot** (called a *commit*) with a short note describing what changed. Git keeps every snapshot forever, so you can always go back to any past state, compare two states, or recover something you deleted. This is the cure for "I lost track of my versions": with Git, *every* version is saved and labeled, and you never again keep `script_final_v2_REALfinal.py` files.

**GitHub** is a website that stores a copy of your Git history in the cloud. It is your **backup** (if your laptop dies, your whole project and its history are safe), the **home** of the project, and the place where milestone versions are published as **Releases**. It also runs automatic checks (Actions) when you push new snapshots.

Three everyday verbs: **commit** (take a labeled snapshot, locally), **push** (upload your snapshots to GitHub), **pull** (download snapshots from GitHub). Two structural ideas: a **branch** (a parallel line of work — you do risky changes on a branch so `main` always works), and a **tag/release** (a permanent, named marker on a particular snapshot, e.g. `v0.1.0`, used for milestones).

### 11.2 Versioning strategy (so versions stay legible forever)

- **Use Semantic Versioning** for releases: **MAJOR.MINOR.PATCH** (e.g., `0.3.1`). MAJOR changes break compatibility (especially the data contract); MINOR adds features compatibly; PATCH is fixes. During Phase 1 you stay in **0.x** (pre-1.0 = "still stabilizing"); reaching a fully working Phase 1 that you'd hand to Phase 2 is a natural **1.0.0**.
- **Tag each milestone** from §8 as a release: Stage 0 toolchain green → `v0.1.0`; single driver in enclosure → `v0.2.0`; electrical chain/splice → `v0.3.0`; multi-driver + contract → `v0.4.0`; Phase-1 complete → `v1.0.0`. A tag is a permanent bookmark you can always return to or share.
- **Keep a `CHANGELOG.md`** — a human-readable list, newest first, of what changed in each version. This plus the tags means "what state was the project in at milestone X" is answerable in seconds.
- **Bump the data contract's `schema_version`** (separate from the app version) only when the on-disk format changes, and note it in the CHANGELOG. This protects Phase 2 from silent format drift.
- **One commit = one coherent change**, with a clear message ("Add Lebedev sphere grids and weights", not "stuff"). Commit at logical stopping points, not once a week. Frequent, well-labeled commits are exactly what was missing in your previous project.

### 11.3 Branching strategy (kept deliberately light for a solo project)

- `main` is **always working** (its tests pass).
- For anything non-trivial or risky, make a short-lived **feature branch** (e.g., `feature/numcalc-adapter`), do the work and its tests there, then merge it into `main` when green. For tiny fixes, committing straight to `main` is fine.
- This is enough structure to protect you and not so much that it gets in the way. (No need for the heavier multi-branch workflows teams use.)

### 11.4 What must NOT go into Git (the `.gitignore`)

Git is for **source code and documents**, not for large generated data or installed tools. Exclude:
- the virtual environment and caches (`.venv/`, `__pycache__/`, `*.pyc`),
- the **compiled NumCalc binary** and the Mesh2HRTF checkout (these are built/installed, not your source),
- **large solve outputs and meshes** — the HDF5 result files, `be.out`/`fe.out`, generated meshes, balloons. Keep these in a `runs/` or `data/` directory that is git-ignored.
- OS cruft (`.DS_Store`).

*Why:* committing multi-GB result files bloats the repo and is exactly the kind of thing that makes a repo unwieldy. If you ever need to version a specific large artifact, that's what a Release attachment or **Git LFS** is for — but for this personal project, simply keep big outputs out of the repo. **HEURISTIC** (standard practice).

### 11.5 How this meshes with Claude Code

Claude Code runs in your terminal and can perform all the Git operations for you — initialize the repo, make commits with good messages, create branches, tag releases, and push to GitHub — when you ask it to. The healthy rhythm: you direct *when* to commit (at logical stopping points) and Claude Code does it and proposes a message; at each milestone you ask it to tag a release and update the CHANGELOG; you review the changes (Claude Code can show you what changed before committing). You stay in control of the history; Claude Code handles the mechanics. (For installing Claude Code itself and its current requirements, follow the official docs at docs.claude.com rather than any remembered specifics.)

### 11.6 Pre-flight setup checklist — do this *before* any code is written

Run top to bottom. Items marked *(Claude Code can do this)* you can delegate once the repo exists; the first few you'll do yourself or alongside it.

1. **Install Apple's command-line tools** (gives you `git`, `clang`, `make`): in Terminal, `xcode-select --install`.
2. **Confirm Git and set your identity:** `git --version`; then set your name and email (`git config --global user.name "…"` and `…user.email "…"`) so commits are attributed.
3. **Decide the repo name** (e.g., `beamsim2`) and **create an empty repository on GitHub** (Private is fine). Don't add files yet.
4. **Install Python 3.12** (python.org arm64 installer or `pyenv`) and **install `uv`**.
5. **Create the project skeleton** matching §4 (folders, `README.md`, `CHANGELOG.md`, `LICENSE`, `pyproject.toml`, `.gitignore`, `.python-version`). *(Claude Code can do this.)*
6. **Make the first commit and connect to GitHub:** initialize Git in the folder, make the initial commit, add the GitHub repo as the remote, and push. *(Claude Code can do this.)* After this, your project and its history exist safely in the cloud.
7. **Put the three authoritative docs in `docs/`** (the Overview, the First Research report, and this Gameplan) and commit them, so the repo carries its own context.
8. **Create the virtual environment from `pyproject.toml` with `uv`** and confirm it activates. *(Claude Code can do this.)*
9. **Stage-0 gate, run now because everything depends on it:** clone Mesh2HRTF, `make` NumCalc, confirm the binary is `arm64` (`file NumCalc`), and run it once on the bundled sphere example. *(Claude Code can do this.)* If this succeeds, DR-01 is confirmed and the project is properly organized to begin.
10. **Tag `v0.0.0`** ("project scaffolding") so even the empty-but-organized starting point is a permanent bookmark. *(Claude Code can do this.)*

When these ten are done, the project is set up the way a professional would expect — versioned, backed up, reproducible, and with the one load-bearing technical assumption (NumCalc on this Mac) already proven — and Stage-0 coding from §8/§10 can begin.

---

## Appendix A — Glossary (DSP/numerical terms bridged to acoustics)

- **BEM (Boundary Element Method):** solves the sound field using only the *surface* of the geometry (no air-volume mesh). Like predicting the whole radiated field from what the speaker's skin is doing.
- **Helmholtz equation:** the frequency-domain wave equation BEM solves — "what steady tone field results from this vibrating surface."
- **Collocation / constant elements:** NumCalc assumes pressure and velocity are constant on each little surface tile; accuracy comes from using enough small tiles (the 6–8-per-wavelength rule).
- **Burton–Miller formulation:** a fix that keeps BEM stable at certain frequencies where the basic method would otherwise give a wrong, non-unique answer.
- **FMM / ML-FMM (Fast Multipole Method):** a math shortcut that lets far-apart surface tiles influence each other in bulk instead of one-by-one — turning an impossibly expensive calculation into a feasible one. It's why high frequencies are tractable.
- **`ka` product:** circumference-to-wavelength ratio of a radiator (`k = 2πf/c`, `a` = radius). Small `ka` → omnidirectional; large `ka` → beamy. This is the dial behind "cones beam at high frequency."
- **Spherical harmonics (SH):** the "Fourier series on a sphere" — a compact way to describe and smoothly interpolate a 3-D radiation pattern.
- **Lebedev / Fliege grid:** clever ways to place sample points on a sphere so that summing over them (with weights) gives accurate integrals — used for the observation directions and for power/DI math.
- **Steering matrix / covariance matrix (Phase 2):** the beamformer stacks each driver's directional response into columns (the steering matrix) and integrates them over "where you want sound" and "where you don't" to build the matrices it optimizes. This is exactly what the `H` tensor feeds.
- **Phase origin / time-of-flight:** the single reference point all driver responses are measured from; keeping each driver's true travel-time delay intact is what lets the beamformer aim the beam. Removing it (re-zeroing) is the cardinal sin (§3.4).
- **Thiele/Small (T/S) parameters:** the small-signal numbers from a driver datasheet that set its low-frequency behavior and box alignment.
- **Semi-inductance (voice-coil):** the realistic model of a voice coil as a *lossy* inductor (because of eddy/skin effects), which shapes the driver's top-octave on-axis response — more accurate than a plain inductor `Le`.

---

*End of gameplan. Once reviewed and accepted, treat this as the project's architecture/handoff reference and add it to the Claude Project knowledge.*
