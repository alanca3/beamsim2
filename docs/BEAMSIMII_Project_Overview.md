# BeamSimII — Project Overview

> **Save this file into the Claude Project's knowledge.** It is the standing context for every future session. Suggested filename: `BEAMSIMII_Project_Overview.md`. It is paired with `BEAMSIMII_First_Research` (the planning-phase research report), which holds the supporting technical detail and citations.

---

## 1. What BeamSimII Is

BeamSimII is a macOS desktop application (with a graphical interface) for designing **active, directivity-controlled loudspeakers** — speakers like the Bang & Olufsen Beolab 90 or the Kii Three, where per-driver filters shape the beam width, directivity pattern, and in-room response.

The core idea is **simulate first, build later**. Rather than physically building and measuring drivers in an enclosure before designing the control filters, BeamSimII simulates the loudspeaker's full three-dimensional acoustic radiation from a geometric model plus a handful of driver parameters, producing a dataset that *mimics real anechoic measurements*. Those simulated "measurements" then drive a filter-design algorithm.

The whole point is to remove the build-and-measure loop from the front of the design process.

---

## 2. The Two Phases

The project splits cleanly into two halves. **This project's near-term focus is Phase 1.** Phase 2 is kept in view at all times because the output of Phase 1 *is the input contract* for Phase 2 — designing Phase 1's data wrong would cripple Phase 2.

**Phase 1 — The Radiation / "Measurement" Simulator (current focus).**
Given a 3D model of an enclosure and its driver(s), plus lumped-element (Thiele/Small) parameters, simulate the complex (magnitude **and** phase) acoustic response of each driver over the full sphere of directions, into an anechoic free field ("driver in box in an infinite room"). The result is a per-driver, frequency-dependent, full-sphere directional dataset — the simulated equivalent of taking a loudspeaker into an anechoic chamber and measuring its full balloon.

**Phase 2 — The Beamforming Filter Designer (later).**
Take the Phase 1 dataset (or, in principle, real measurements in the same format) and compute a set of per-driver filters that achieve an arbitrary target: a chosen beam width, a chosen directivity pattern, and a desired in-room response. Room/in-room behavior enters here, not in Phase 1.

---

## 3. Decided Technical Direction

These decisions are settled. Treat them as the foundation; flag (do not silently change) any proposal that departs from them.

- **Physics engine: the Boundary Element Method (BEM).** BEM solves the acoustic problem on the *surface* of the geometry and natively captures baffle diffraction, the full 3D enclosure shape, driver placement (path-length, timing, phase), and mutual coupling between drivers — directly from the CAD surface the user provides. Analytic methods cannot represent an arbitrary 3D enclosure or side/rear drivers, so they are not the engine. (See `BEAMSIMII_First_Research` for the full method comparison.)

- **The app sets up BEM for the user.** The user is comfortable modeling a box and a rough driver but does **not** know how to configure a BEM solver. BeamSimII must accept imported/parametric geometry plus parameters and **automatically generate the entire BEM setup** — meshing, boundary conditions, frequency list, solver configuration, and job execution. The user never hand-configures a solve.

- **Recommended default BEM backend: NumCalc / Mesh2HRTF** (open-source, free, proven for loudspeaker radiation, fully scriptable input so the app can generate the setup, buildable on macOS, and parallelizable across frequencies). The architecture should keep the solver behind an abstraction layer so an alternative backend (e.g., bempp-cl, or COMSOL driven via its API) can be swapped in. **Final selection is a Phase-1 planning decision** (see §7).

- **Driver model.** The cone is modeled as a simple vibrating surface (a rigid piston, or a single parameterized cap shape) **inside** the BEM mesh. This lets BEM compute the driver's aperture directivity (from its rough shape and size) together with its box/baffle interaction — correctly and without double-counting. The Thiele/Small parameters and voice-coil inductance supply the driver's **on-axis terminal frequency response** (low-frequency rolloff and box alignment; high-frequency rolloff from lossy voice-coil inductance), applied as a per-driver complex multiplier across all directions. **Precise cone-breakup patterns are explicitly out of scope** — approximate off-axis behavior is acceptable.

- **Platform & packaging.** macOS, GUI-driven, open-source and self-contained dependencies preferred. Licenses are not a concern for this personal project, but **AKABAK/ABEC are off the table** (the user cannot use them).

- **Compute budget.** Solve times of **~1–2 days are acceptable**. This relaxes the earlier 6-hour ceiling and makes fuller-band BEM feasible, but efficiency tactics (solving each driver once and reusing it, frequency-band remeshing, optional analytic high-frequency splicing) remain valuable.

---

## 4. The Data Contract (the linchpin)

The single most important design artifact in the whole project is the **format of the Phase 1 output**, because it is the interface between the two phases. It should be designed *backward from what the filter designer needs*, even now.

**Primary representation:** a per-driver complex transfer-function tensor

```
H : [ driver  x  frequency  x  direction ]   (complex: magnitude + phase)
```

where `direction` indexes a **near-uniform sphere sampling** (Lebedev / Fliege / spherical t-design / icosphere — not a naive latitude/longitude grid, which oversamples the poles and integrates poorly). Each driver's slab of this tensor is, literally, the set of columns the Phase 2 beamformer assembles into its steering matrix. The covariance matrices the filter designer builds over "target" and "reject" angular regions are integrals over this same sphere.

**Non-negotiable correctness rule — single phase origin.** Every driver's response must be referenced to **one common spatial phase origin** (the global coordinate origin), preserving the true time-of-flight and excess phase implied by each driver's real 3D position. If per-driver responses are independently minimum-phase-ified or re-zeroed, the multi-driver sum will **silently mis-steer the beam**. This is the highest-risk software detail in the project.

**Carried metadata (must travel with the dataset):** driver positions and orientations in 3D; the coordinate system and phase-origin definition; the sphere-sampling scheme and its integration weights; the observation radius / far-field assumption; the frequency grid; reference conditions (speed of sound, temperature/humidity/pressure); diaphragm-area normalization; and units / reference level.

**Export targets (for interoperability and validation):** a native working format (HDF5 or NumPy, optionally SOFA); `.frd` per-driver/per-angle for **VituixCAD**; and a balloon format (CLF) for cross-checking against industry tools. **REW** and **MATLAB** are also part of the user's existing workflow.

---

## 5. The User — Who Future Sessions Are Talking To

- **Strong** in hands-on acoustics and measurement: room modes, T60/EDT, REW, VituixCAD, polar/directivity analysis, anechoic measurement practice.
- **Not a programmer.** The user can read Python but does not write it from scratch. Claude writes all code; the user supplies acoustics judgment, geometry, and parameters.
- **Limited fluency in high-level DSP / numerical mathematics.** Explanations should bridge DSP and numerical concepts to acoustics analogies wherever possible.
- Has access to Python, MATLAB, and COMSOL, and is willing to work across tools — but does **not** know COMSOL's BEM tooling or hyper-technical numerical methods, which is exactly why the app must abstract them.

**Working agreement** (applies to all sessions):
- Code must be complete, runnable, and heavily annotated — never pseudocode, never "fill in the rest." Dimensional comments on arrays (e.g. `# H: [F x M x N] complex`), plain-English block explanations, parameter/return descriptions. If code is too long for one response, say so and split it into explicit, labeled parts.
- Label technical/physics claims **VERIFIED / INFERRED / HEURISTIC** with author/year citations.
- Conversational answers in prose with minimal formatting; technical documents, code, and reference material structured and thorough.
- Engage pushback with reasoning; own mistakes plainly without excessive apology; veracity over reassurance.

---

## 6. Phase 1 — What "Done" Looks Like

Phase 1 is successful when the user can, **without touching a BEM solver directly**:

1. Provide an enclosure shape and one or more rough driver geometries (imported or built parametrically in the GUI), place them in 3D, and enter T/S parameters and an observation sphere.
2. Launch a single run; the app meshes, sets up BEM, solves (within ~1–2 days), and assembles the per-driver complex `H` tensor.
3. Inspect results in the GUI (directivity maps, balloons, horizontal/vertical polars, on-axis response).
4. Export a dataset, with full metadata and a single consistent phase origin, in a format Phase 2 can consume directly.

And the simulator is *trustworthy*: it passes automated validation against the analytic piston-in-a-baffle solution and a sphere benchmark, and its radiated-power / directivity-index integrals behave correctly.

---

## 7. Open Architecture Questions (for the Phase 1 Planning Session)

These are deliberately **not** pre-decided; the planning session should resolve them with reasoning:

1. **Final BEM backend** — NumCalc/Mesh2HRTF (recommended default) vs. bempp-cl vs. COMSOL-via-API — including an explicit check of **macOS / Apple-Silicon build feasibility** and how the solver is bundled with the app.
2. **GUI framework** for a macOS Python app (e.g., PySide6/Qt is a strong candidate).
3. **Driver-model efficiency fork** — run BEM full-band (now feasible within the 1–2 day budget), or splice an analytic piston-directivity model above a BEM frequency cap to save compute. Decide from real Stage-1 timing; if splicing, design the blend to avoid magnitude/phase discontinuities.
4. **Geometry input** — supported import formats (STEP/STL/OBJ) and/or a parametric box-and-driver builder in the GUI.

---

## 8. Relationship to Prior Work

This is a **clean restart**. Earlier iterations of a similarly named effort (a prior "beamsim" v1–v5) exist but are **not** authoritative here and should be ignored except as informal background. The two authoritative documents are **this overview** and **`BEAMSIMII_First_Research`**. Everything else starts fresh in this project to avoid drift.
