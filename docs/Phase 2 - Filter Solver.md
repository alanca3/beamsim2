# BeamSimII — Phase 2 Gameplan: The Beamforming Filter Designer

> The authoritative architecture/spec for **Phase 2** — the automatic filter designer — as
> `BEAMSIMII_Gameplan.md` is for Phase 1. Decision records (DR-P2-01…06), the pipeline, the
> filter/data contract (§3), the module layout (§4), the verified core math (§5), the GUI
> (§6), the validation plan (§7), milestones (§8), risk register (§9), and build order (§10).
> Flag departures; do not change silently. Paired with `Research Phase 2.md` (the cited,
> adversarially-verified research report this plan distills) and `DATA_CONTRACT.md` (the
> Phase-1 → Phase-2 data interface).

---

## 0. How to read this; what Phase 2 is

BeamSimII Phase 1 (the BEM radiation simulator, functionally complete) outputs the per-driver
complex transfer tensor **H[M drivers × F frequencies × N sphere-directions]** (complex128,
lossless HDF5) on a near-uniform sphere grid with quadrature weights, every driver referenced
to **one common spatial phase origin** (true time-of-flight preserved — the cardinal rule,
`DATA_CONTRACT.md §3.4`).

**Phase 2 — the automatic filter designer — is the project's reason for existing.** Given H, the
user specifies a **target beam** (a *shape* — omni / cardioid / wide / narrow / constant-directivity,
optionally a continuous cardioid-order, or an arbitrary custom pattern — and a *steering direction*).
The app solves per-driver complex weights `w_m(f)`, shows the **achieved** directivity against the
target, and exports the design. This converts a simulated anechoic balloon into a directivity-
controlled loudspeaker design (the Beolab-90 / Kii-Three class) **without building hardware first**.

The forward model is the AES GLL complex summation:
`P(f, dir) = Σ_m w_m(f) · H[m, f, dir]` — already coded as `validation/closed_loop.steer_response`.

**Cardinal rule, restated for Phase 2 (sacred).** The beamformer consumes H with its **native
inter-driver phase** and **never re-zeroes, minimum-phase-ifies, time-aligns, or per-driver-
normalizes** any driver. The inter-driver phase *is* the steering information. `tests/test_phase_origin.py`
must stay green; reproducing a cardioid null from H is the end-to-end proof the contract steers a beam.

---

## 1. Decision records

### DR-P2-01 — One shared engine, three solver modes (not four beamformers)
Beamforming for "variable beam shape / beam steering" is one quadratic program over a weighted
complex covariance assembled on the sphere. Expose three modes (plus the trivial delay-and-sum
corner), **both LS and constant-DI first-class** (DR confirmed by the user 2026-06-20):
- **(i) Delay-and-sum** — `w = c/M` (exists in `closed_loop.py`). The robustness corner / baseline.
- **(ii) Regularized least-squares / pressure-matching (engine #1, foundation)** — matches an
  arbitrary complex target field `b_f(dir)` on the sphere. Steered lobes, cardioid, nulls, and
  approximate constant-DI all encode as a target field. MVDR/LCMV are constrained special cases.
- **(iii) Luo MECD/MSCD constant-directivity (engine #2, co-equal)** — holds a directivity index
  constant vs frequency via a generalized Rayleigh quotient on accept/reject covariance + a QCQP.
- **ACC / sound-zones is NOT a separate mode** — its `(R_bright, R_dark)` generalized eigenproblem
  is the same machinery as Luo's accept/reject; it is subsumed as region selection of the eigen
  max-ratio sub-mode.

### DR-P2-02 — House sign convention (the #1 silent-failure guard)
The coded forward model is `P = Σ_m w_m H_m`. Pin ONE convention so every solver's `w` drops in
with **no extra conjugation**:
- Look vector **`c = conj(H[:, f, look])`**.
- Covariance **`R = conj(H_f) · diag(a) · H_fᵀ`** (`[M×M]` Hermitian PSD; `a_n` = Lebedev weights).

The microphone-array literature uses the **conjugate** convention (`R = Σ a_n H_n H_nᴴ`, `d = H_look`,
`wᴴd = 1`); copied verbatim it yields weights that **mirror-steer the beam** — exactly the silent
mis-steer the cardinal rule forbids. The **round-trip steering test** (V-RT, §7) is the empirical
arbiter and is implemented and green **before** any solver mode is locked.

### DR-P2-03 — Audit-first export; filter realization deferred (user decision 2026-06-20)
v1 exports **for audit in the user's own tools** (VituixCAD / REW): filtered per-driver `.frd`
(the weights `w_m(f)` baked into each driver's directional response on regular H/V polar arcs),
the combined steered response, and the raw complex weights. **FIR/biquad coefficient export and the
FIR-realization step are DEFERRED** to Stage P2-5, gated on the user picking a deployment DSP.
Consequence stated honestly: **v1 produces a verifiable design + audit artifacts, not a directly
deployable filter file.** The realization math is specified (§5.4) so the deferral is architected-for.

### DR-P2-04 — Anechoic free-field only for v1 (user decision 2026-06-20)
v1 designs against the simulated full-sphere anechoic directivity (the current H-on-sphere contract).
**In-room / sound-zone control (off-sphere control points, room transfer functions) is deferred** to a
later sub-stage; it needs inputs outside the current contract.

### DR-P2-05 — Robustness is one user-facing knob: a White-Noise-Gain (WNG) floor
Regularization is mandatory, not optional (a handful of heterogeneous drivers → superdirective
blow-up at low f). Tikhonov diagonal loading is exposed as a single **WNG floor (dB)** slider; the
loading ε is solved per frequency by monotone bisection to hit the floor (§5.3). This same knob is
the feasibility guard rail for "arbitrary custom" targets.

### DR-P2-06 — Dense sphere grid + SH resampling is a Phase-1 prerequisite Phase-2 inherits
`core/sphere.py` ships only Lebedev {6, 14, 26}; `core/sh_transform.py` is a stub. Beam design and
the audit deliverable need **hundreds–thousands of directions** and **SH resampling to a regular grid**
(VituixCAD/REW want matched H/V polar arcs; Lebedev points don't lie on them). Therefore **Stage P2-0
expands the simulator's grid from max 26 → thousands of points** (dense Lebedev tables with exact
quadrature weights) and implements the SH transform/resampling. Bonus: this unlocks CLF export and the
deferred sparse-frequency SH interpolation. The cheap **V-CARDIOID convention guard runs at N=26**
(degree-2 intensity, exact under Lebedev-26), so the convention is pinned before the dense grid lands.

---

## 2. End-to-end Phase-2 pipeline

```
RadiationDataset (H[M,F,N] + sphere grid a_n + driver positions + c)   ← Phase-1 output / loaded HDF5
        │
        ▼  TargetSpec  (preset | cardioid-order | steering | arbitrary | nulls; band; robustness)
  beamform.targets   → target field b_f(dir) and/or accept/reject masks on the (dense) sphere grid
        │
        ▼
  beamform.covariance → c = conj(H_look),  R = conj(H) diag(a) H^T   [house convention, DR-P2-02]
        │
        ▼  mode = delay-sum | LS/PM | MVDR/LCMV | Luo-MECD/MSCD
  beamform.weights  + beamform.regularize (WNG-floor loading)  → w_m(f)  [M×F]
        │
        ▼
  beamform.forward (= closed_loop.steer_response) → P(f,dir) = Σ_m w_m H_m ; achieved DI/beamwidth/metrics
        │
        ├──► GUI: achieved-vs-target plots (polar / balloon / DI-vs-f / on-axis)
        └──► io.filter_export: filtered per-driver .frd (SH-resampled H/V arcs) + combined .frd + raw weights
                 [DEFERRED P2-5]  beamform.realize → FIR/biquad → io: CamillaDSP/miniDSP coefficient export
```

Key correctness steps: (1) per-driver superposition means weights recombine H slices with no
re-solve (the Phase-1 "solve once, reuse" payoff); (2) every step preserves the single phase origin;
(3) regularization is applied inside the solve, never as a post-hoc fudge; (4) infeasible targets are
flagged (achievable-WNG), never silently mis-realized.

---

## 3. The filter / data contract (designed backward from what the user audits)

### 3.1 Input — consumed from `RadiationDataset` (already in the Phase-1 contract)
`stacked_h_full(ds)` → `H[M,F,N]` complex128; `ds.frequencies` [F]; `ds.directions` (unit_vectors
[N×3], weights [N] sum=4π, theta_phi [N×2]); per driver `attrs["position"]` [3] and `attrs["radius"]`;
root `attrs["speed_of_sound"]` (the exact `c` the BEM used — present since `run.py:473`).
**Prerequisite fills (Stage P2-0a):** write `diaphragm_area` (= π·radius²) to the contract; add a
`schema_version` read guard in `read_dataset`.

### 3.2 `TargetSpec` (new) — what the user asks for
```
TargetSpec:
  mode:        "preset" | "cardioid_order" | "steering_only" | "custom"
  preset:      "omni" | "cardioid" | "supercardioid" | "hypercardioid" | "wide" | "narrow"  (mode=preset)
  order_a:     float in [0,1]   T(θ)=a+(1−a)cosθ            (mode=cardioid_order)
  steer_dir:   unit [3] (az,el)                              (all modes)
  nulls:       list[unit [3]]  (soft target zeros or LCMV hard constraints)
  band_hz:     (f_lo, f_hi)  — design band; out-of-band weights gated by GRPQ / pass-through
  robustness:  wng_floor_db  (the single DR-P2-05 knob; slider s∈[0,1] → dB)
  engine:      "ls" | "mvdr" | "lcmv" | "luo_mscd" | "luo_mecd" | "delay_sum"
```
`targets.build_target(spec, ds)` → `b_f[F,N]` complex target field and/or `(accept_mask, reject_mask)`.

### 3.3 `DesignResult` (new) — what the solver returns
```
DesignResult:
  weights:        w[M,F] complex128         — the per-driver complex weights
  steered_field:  P[F,N] complex128         — achieved Σ_m w_m H_m
  metrics:        di_db[F], beamwidth_deg[F], wng_db[F], target_error_db[F], feasible_mask[F] bool
  spec:           TargetSpec                 — echoed back
  attrs:          provenance (engine, convention="house: P=Σ w_m H_m", c used, etc.)
```

### 3.4 Export schema (v1, audit-first — DR-P2-03)
- **Filtered per-driver `.frd`** — for each driver, `w_m(f)·H_full[m]` resampled (SH, §5.5) to a
  regular H/V polar arc set VituixCAD/REW ingest (freq / mag dB / phase deg). Header records the
  weights are **baked in** (audit-only) and restates the single-phase-origin note.
- **Combined steered `.frd`** — `P(f)` on the same arcs (the achieved system response).
- **Raw weights** — `w[M,F]` complex as `.npz` + an HDF5 `/design` group (re-loadable; lossless).
- **[DEFERRED P2-5]** FIR taps (float WAV + text) + CamillaDSP YAML; biquad cascade (tonal only).

---

## 4. Module layout (`src/beamsim2/beamform/`, Qt-free; mirrors the Phase-1 split)

```
beamform/
  __init__.py
  targets.py     ← TargetSpec → b_f(dir) + accept/reject masks (presets / cardioid-order / steer / custom / nulls)
  covariance.py  ← c = conj(H_look) ; R = conj(H) diag(a) H^T            [house convention, DR-P2-02]
  weights.py     ← delay_sum | ls_pressure_match | mvdr | lcmv | luo_mscd | luo_mecd  → w[M,F]
  regularize.py  ← WNG-floor diagonal loading (log-ε bisection); ε_min floor; feasibility flagging
  forward.py     ← thin reuse of closed_loop.steer_response; achieved DI / beamwidth / target-error metrics
  design.py      ← orchestrator: (RadiationDataset, TargetSpec) → DesignResult
  realize.py     ← [DEFERRED P2-5] w_m(f) → causal per-driver FIR (shared τ) ; optional IIR/biquad
io/filter_export.py            ← filtered/-combined .frd (via core.sh_transform resample) + raw weights
gui/filter_designer_view.py    ← target-spec UI + WNG slider + achieved-vs-target plots + audit export (tab 4)
core/sphere.py                 ← [Stage P2-0b] dense Lebedev tables + Balloon preset (max 26 → thousands)
core/sh_transform.py           ← [Stage P2-0c] forward SH (quadrature) + inverse to regular grid
tests/test_beamform_*.py       ← V-RT, V-CARDIOID, V-CBT, WNG-invariant, export round-trip
```
Dependency direction unchanged: `gui` imports `beamform`/`core`; `core`/`beamform` never import `gui`.

---

## 5. Core math (verified; engineering exp(−jωt)/exp(+jkr) convention)

All covariance/DI/target integrals over the sphere are quadrature sums `Σ_n a_n · f(dir_n)` with the
dataset's Lebedev weights `a_n` (sum = 4π). `H_f := H[:, f, :]` is `[M×N]`; `W := diag(a_n)` is `[N×N]`.

### 5.1 Pressure-matching (engine #1, the primary workhorse)
Cost `J = (H_fᵀ w − b_f)ᴴ W (H_fᵀ w − b_f) + λ wᴴ w`. Minimizer (verified for **this** model):
```
w_f = ( conj(H_f) · W · H_fᵀ  +  λ I_M )⁻¹ · conj(H_f) · W · b_f
```
`conj(H_f) W H_fᵀ` is `[M×M]` Hermitian PSD. Solve via `scipy.linalg.solve(A, rhs, assume_a='pos')`
(Cholesky) for λ>0; for tiny λ prefer the stacked real LS `[√W H_fᵀ; √λ I] w = [√W b; 0]` via
`numpy.linalg.lstsq`. **Do NOT** use the microphone-array `(H W Hᴴ + λI)⁻¹ H W b` form — it solves the
conjugated model `P = Hᴴ w` and **mirror-steers** (DR-P2-02).

### 5.2 MVDR / LCMV (constrained special cases)
`c = conj(H[:, f, look])`, `R = conj(H_f) W H_fᵀ`:
- MVDR (distortionless toward look, min output power): `w = (R+εI)⁻¹ c / (cᴴ (R+εI)⁻¹ c)`.
- LCMV (hard nulls): `C = [c_look, c_null1, …]` `[M×K]`, `g = [1,0,…]ᵀ`,
  `w = R⁻¹ C (Cᴴ R⁻¹ C)⁻¹ g`. At most `K ≤ M−1` independent nulls.

### 5.3 Regularization & robustness (WNG floor — DR-P2-05)
Under `cᴴ w = 1`, **`WNG(w) = 1/‖w‖²`**, so a WNG floor ≡ a weight-norm cap. Loaded form
`w(ε) = (R+εI)⁻¹ c / (cᴴ (R+εI)⁻¹ c)`; `WNG(ε)` is **monotone increasing** in ε (ε=0 → max-directivity,
fragile; ε→∞ → delay-and-sum `w=c/M`, WNG=M). Solve `WNG(ε) = W_target` per frequency by **plain
1-D bisection on log ε**. Always add `ε_min ≈ 1e-10·trace(R)/M` before Cholesky; normalize R by
trace/M. GUI slider `s∈[0,1]` maps to `W_floor_dB` from ≈ −20 dB ("sharpest; matched low-noise
drivers") to `10·log10(M)` ("most forgiving"); default ≈ −6 dB ("balanced"). Flag low-f bins where the
floor is unreachable (`feasible_mask=False`; DI rolls off gracefully — never garbage).
Low-f scaling: `WNG ~ (kd)^{+2(N−1)}` ≈ **6·(order) dB/octave**.

### 5.4 Luo MECD/MSCD constant-directivity (engine #2)
Per frequency, accept covariance `A` and reject covariance `R` (region-weighted outer-product sums,
normalized): `A = Σ_n a_n f_A(r_n) conj(H_n) H_nᵀ / Σ a_n f_A`, likewise `R` with `f_R`. Generalized
Rayleigh quotient `G(w) = wᴴ A w / wᴴ R w`; `GDI(dB) = 10 log10 G`.
- **Pass 1 (ceiling):** `τ_max(f) = scipy.linalg.eigh(A, R, eigvals_only=True).max()` — the per-f
  directivity **ceiling**, NOT the shipped answer. Choose one constant `τ* = min(10^{target_dB/10},
  min_f τ_max(f))` so the target DI is feasible at every frequency.
- **Pass 2 (QCQP at fixed τ*):** `D = A − τ* R` (indefinite Hermitian). Solve `wᴴ D w = 0` with:
  - **MSCD** (max sensitivity / distortionless min-norm): `min wᴴw s.t. wᴴDw=0, cᴴw=1`;
    analytic `w(λ)=μ(I−λD)⁻¹c`, scalar secular root via `scipy.optimize.brentq`.
  - **MECD** (max efficiency, unit-norm): `max wᴴCw s.t. wᴴDw=0, wᴴw=1` by projected ascent (~5 iters).
- **GRPQ generalized crossovers:** per-driver per-frequency band gates `Λ=diag(λ_n(f))∈[0,1]`,
  `Γ=diag(diag(R))`; use `R̂ = ΛRΛ + Γ(I−Λ²)` in the QCQP so out-of-band drivers auto-vanish.
- **Implementation-time verification (do at code time, not before):** confirm the GRPQ `R̂` sign and
  the MSCD secular form against the local Luo PDF (`arXiv:2407.01860`) — these two blocks were
  reconstructed from a summary in research.

### 5.5 Filter realization (DEFERRED to P2-5; specified now because the math is subtle)
Two-step, default **linear-phase FIR via IFFT + window**, with **ONE shared modeling delay τ applied
identically to all drivers** (a common `exp(−j2πfτ)` factors out of P = pure latency; a *per-driver*
delay or per-driver min-phase **re-steers the beam** → forbidden; assert equal τ). Per driver:
interpolate `(log|w_m|, unwrapped phase)` onto a dense uniform FFT grid (`Nfft~2¹⁶`) → **conjugate**
(NumCalc engineering → numpy DSP convention; a straight `ifft` peaks anti-causally) → Hermitian-extend
→ `·exp(−j2πfτ)` → `np.fft.ifft` → `fftshift` → truncate to `Ntaps` → `scipy.signal.windows.kaiser(β=8)`
→ verify magnitude + *relative* phase via `freqz`. `scipy.signal.firls/firwin2/remez` cannot fit
arbitrary phase; there is no `invfreqz`. Optional low-latency IIR: complex Levy equation-error
(`np.linalg.lstsq`) → output-error refine (`scipy.optimize.least_squares`) → `scipy.signal.zpk2sos`,
stabilize poles, verify relative phase preserved. **NEVER `scipy.signal.minimum_phase` per driver.**
Convention guard: feed `w = exp(−j2πfT)` and assert the filter delays by **+T**.

---

## 6. GUI design (the Filter-Designer tab — never on the critical path)

A fifth top-level tab **[5] Filter Designer** (after Results), a thin shell over `beamform.design`,
following the Phase-1 GUI conventions (`AppState`, matplotlib `_MplCanvas`, background `QThread`
worker, one-way core↔gui dependency).

```
[1] Geometry  [2] Drivers  [3] Simulation  [4] Results  [5] Filter Designer
  ┌─ Target ─────────────────┐   ┌─ Achieved vs Target ───────────────────┐
  │ Source: (•) current solve │   │  Polar (H/V) · Balloon · DI-vs-freq ·  │
  │         ( ) open dataset… │   │  on-axis — target overlaid on achieved │
  │ Pattern: [Cardioid ▼]     │   └────────────────────────────────────────┘
  │ Order a: [===|---] 0.5    │   Metrics: DI(f), beamwidth(f), WNG(f),
  │ Steer:  az[0°] el[0°]     │            target error(f), feasible band
  │ Nulls:  [+ add direction] │
  │ Band:   [__]–[__] Hz      │   [Design]  (background worker)
  │ Engine: [LS / Luo ▼]      │   [Export…] → filtered .frd / combined .frd / raw weights
  │ Robustness: [===|----]    │              (FIR/biquad greyed until P2-5)
  └───────────────────────────┘
```
Reads `AppState.result.dataset` (in-memory after a solve) or a loaded HDF5. "Design" runs
`beamform.design` on a worker; plots overlay the **achieved** P on the **target** so the user sees the
match. Infeasible bands are shown amber (mirrors the Phase-1 non-converged styling).

---

## 7. Validation plan (wired as automated self-tests; gameplan style)

Every check is a `pytest` test and a callable in `beamform/` or `validation/`. CI-safe synthetic-H
variants run without the NumCalc binary; `@local_only` real-BEM variants reuse the 2-driver box solve.

- **V-RT — round-trip steering (the convention + cardinal-rule gate; runs at N=26).** Design `w` for a
  commanded steer direction from synthetic monopole H; reconstruct `P = Σ_m w_m H_m`; assert the main
  lobe points the **commanded** way (not its mirror) and the delay-and-sum null reproduces. Guards
  DR-P2-02 and the single phase origin. **Built first, before any solver mode is locked.**
- **V-CARDIOID — first-order DI anchors (CI-safe, exact at N=26).** Two-element endfire monopole array;
  DI via Lebedev quadrature: cardioid **4.771 dB** (null at 180°), supercardioid **5.719 dB**,
  hypercardioid **6.021 dB**; assert null angle `acos(−a/(1−a))` within ±0.5° (the sign-flip guard — a
  symmetric-beamwidth test alone cannot catch a global sign flip) and DI within ±0.1 dB.
- **V-CBT — constant beamwidth (engine #2; needs the dense grid).** Curved monopole arc, **real**
  Legendre amplitude weights (Keele polynomial `U(x)=1+0.066x−1.8x²+0.743x³`, `x=θ/θ₀`; curvature
  time-of-flight lives in H → honors the cardinal rule); assert −6 dB beamwidth ≈ `0.64·(2θ₀)` flat vs
  frequency above an empirical cutoff. Unmeasurable at N=26 — gated on Stage P2-0.
- **V-WNG — robustness invariants.** WNG monotone ↑ and DI monotone ↓ in the loading; `w(s=1)=c/M`;
  loaded R Hermitian PD (Cholesky succeeds); `cᴴw=1` exactly (distortionless ⇒ phase origin intact).
- **V-SH — SH resample round-trip (Stage P2-0).** A known band-limited field projected to SH and
  evaluated on a regular grid returns to the Lebedev samples within tolerance.
- **V-EXPORT — audit round-trip.** Export filtered/combined `.frd` + raw weights, re-import the weights,
  re-evaluate `P`, and confirm it reproduces the designed beam within tolerance.

`tests/test_phase_origin.py` must stay green throughout. `black` + `ruff` clean; full suite before any
session close.

---

## 8. Milestones (each gate is the finish line; nothing proceeds until green)

- **Stage P2-0 — foundation: grid+SH + contract hardening + convention pin.**
  (a) Cheap first: `beamform/` skeleton, `diaphragm_area` write, `schema_version` read guard, **V-RT**.
  (b) The #1 to-do: **dense Lebedev grids (max 26 → thousands)** + "Balloon" GUI preset in
  `core/sphere.py`; `core/sh_transform.py` forward+inverse SH. *Gate:* V-RT green; dense grid solves;
  V-SH round-trips. Bonus: unblocks CLF + sparse-freq interpolation.
- **Stage P2-1 — LS/pressure-matching engine (#1) + targets + robustness.** *Gate:* V-CARDIOID DI
  anchors + null-angle sign; V-WNG invariants; arbitrary-target feasibility flagging.
- **Stage P2-2 — Luo constant-DI engine (#2) + GRPQ.** *Gate:* V-CBT constant beamwidth; constant-τ
  GDI holds across the band on a synthetic array.
- **Stage P2-3 — GUI Filter-Designer tab + audit export (v1 usable end-to-end).** *Gate:*
  design→view→export round-trip through the GUI worker; smoke/offscreen test; opens in VituixCAD/REW.
- **Stage P2-5 — [DEFERRED, gated on the user's deployment DSP] filter realization + deploy export.**
  `realize.py` FIR (+ optional biquad) + coefficient export (CamillaDSP/miniDSP). *Gate:*
  realize→export→re-import→re-evaluate reproduces the designed beam; equal-τ + delay-sign guards.

The GUI rides along from P2-3 but never gates a stage; the headless core + its tests do.

---

## 9. Risk register

- **R-P2-1 — Conjugation/mirror-steer (the H convention isn't `P=Σ w_m H_m`).** *Impact:* very high
  (silent). *Mitigation:* DR-P2-02 house convention; **V-RT** built first and run on real H; reuse the
  proven `closed_loop` forward model.
- **R-P2-2 — Low-frequency ill-conditioning / superdirective blow-up.** *Impact:* high. *Mitigation:*
  mandatory WNG-floor loading (DR-P2-05); `ε_min`; feasibility flagging; delay-and-sum as the ε→∞ anchor.
- **R-P2-3 — Infeasible "arbitrary" targets.** *Impact:* medium. *Mitigation:* achievable-WNG guard
  rail; report `target_error_db` + `feasible_mask`; never silently mis-realize.
- **R-P2-4 — Grid too coarse for HF directivity / audit (N≤26).** *Impact:* high (blocks the deliverable).
  *Mitigation:* Stage P2-0 dense grid + SH; V-SH; V-CARDIOID confirms the cheap path at 26 meanwhile.
- **R-P2-5 — Luo GRPQ/MSCD sign or form error.** *Impact:* medium. *Mitigation:* code-time check vs the
  arXiv PDF; constant-τ GDI + V-CBT gates catch a wrong sign.
- **R-P2-6 — Filter realization re-steers (per-driver delay/min-phase).** *Impact:* very high (deferred,
  but latent). *Mitigation:* §5.5 shared-τ rule; equal-τ assertion; +T delay guard test (Stage P2-5).
- **R-P2-7 — Cardinal-rule regression.** *Impact:* very high. *Mitigation:* `tests/test_phase_origin.py`
  stays green; no per-driver re-zero anywhere in `beamform/`.

---

## 10. Build order

1. **P2-0a** — `beamform/` skeleton; `diaphragm_area` write + `schema_version` read guard; **V-RT**.
2. **P2-0b** — dense Lebedev tables + Balloon preset (`core/sphere.py`); weight/SH-orthonormality tests.
3. **P2-0c** — `core/sh_transform.py` forward+inverse SH; **V-SH**.
4. **P2-1** — `targets` + `covariance` + `weights(ls/mvdr/lcmv)` + `regularize` + `forward` + `design`;
   **V-CARDIOID**, **V-WNG**.
5. **P2-2** — `weights(luo_mscd/luo_mecd)` + GRPQ; **V-CBT**.
6. **P2-3** — `io/filter_export` (filtered/combined `.frd` + raw weights); `gui/filter_designer_view`;
   **V-EXPORT** + GUI smoke.
7. **P2-5 (deferred)** — `beamform/realize` FIR/biquad + deployment export; realization round-trip.

Each numbered item lands as a coherent commit on `feature/phase2-filter-designer`; merge to `main`
when its gate is green. Tag a Phase-2 milestone semver only when the user calls it (no tag at kickoff).
```
