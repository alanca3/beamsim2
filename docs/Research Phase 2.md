# BeamSimII — Phase 2 Research Report (Filter Designer / Beamformer)

> Standing technical reference for Phase 2 (the automatic beamforming filter designer).
> Produced by a deep, adversarially-verified research workflow (run `wf_17949dc9-468`,
> 17 agents: 8 topic finders -> 8 adversarial verifiers -> 1 synthesis) on 2026-06-20.
> Paired with `Phase 2 - Filter Solver.md` (the gameplan). The **Synthesis** below is the
> decision-oriented summary; the **Appendix** is the full per-topic dossier (raw findings +
> adversarial-verification verdicts) it was distilled from. Topics covered: 8.
>
> Reading order: skim the Synthesis (sections 1-9) for decisions; consult the Appendix for
> exact formulas, sources, and where verification corrected a finding.

---

# Part I — Synthesis (decision-oriented)

## BeamSimII Phase-2 Beamforming Filter Designer — Synthesis & Decision Report

**Global convention pin (read first; load-bearing for every section).** The dossier's eight topics use *three* different conjugation conventions. They are unified here to ONE house convention that matches BeamSimII's coded forward model `P(f,dir) = Σ_m w_m(f) · H[m,f,dir]`:

- **Steering/look vector:** `c = conj(H[:, f, look])`.
- **Covariance assembly:** `R[m,m'] = Σ_n a_n · conj(H[m,f,n]) · H[m',f,n]` = `conj(H_f) · diag(a) · H_f^T`, where `a_n` are the Lebedev quadrature weights shipped with the dataset. `R` is `[M×M]` Hermitian PSD.
- Built this way, every solver's output `w` drops **directly** into `P = Σ_m w_m H_m` with **no extra conjugation**.

This is the single most error-prone decision. Topic 5 (robustness) is stated in the *receive* convention (`Γ=Σ a_n H_n H_n^H`, `d=H[:,look]` un-conjugated, `w^H d=1`); its weights are the **complex conjugate** of the transmit-consistent answer and, plugged into `P=Σ w_m H_m`, mirror-steer the beam. Its formulas are restated below in the house convention (`d→c=conj(H_look)`). The two covariances are complex conjugates → identical real eigenvalues, conjugate eigenvectors → exactly the silent mis-steer the cardinal rule and V-5 forbid. **The round-trip steering test is the empirical arbiter** of whether the stored H convention truly matches `G=H^T`; this cannot be verified from the dossier and is the #1 §9 open question.

---

## 1. ALGORITHM FAMILY & STAGING

**Recommendation: ONE shared engine — a weighted complex covariance assembly over the Lebedev sphere (`R = conj(H) diag(a) H^T`) — exposing three solver modes, plus the existing delay-and-sum as the trivial corner.** Do not build four separate beamformers; they are variants of one quadratic program.

- **(i) Delay-and-sum** [exists in `closed_loop.py`]: `w = c/M`, `c = conj(H[:,look])`. This is the `λ→∞` / max-WNG corner of every mode below. Keep as the robustness anchor and sanity baseline.
- **(ii) Regularized least-squares / pressure-matching — THE PRIMARY ENGINE.** Workhorse for "arbitrary user-specified beam shape + steering." Any target (steered lobe, cardioid, nulls, constant-DI template) encodes as one complex `[N]` target field `b_f` on the sphere; weights fall out of one regularized `[M×M]` solve per frequency. MVDR, LCMV, and max-directivity are constrained special cases of it. **Chosen because:** it is the only mode that natively accepts an arbitrary phase-controlled target, it is the minimal-surface-area core, and it directly mirrors how the closest commercial analog (B&O Beolab 90) designs its filters (measured per-driver response + LS optimization to a target).
- **(iii) Luo MECD/MSCD constant-directivity — THE SPECIALIST (advanced mode).** Layer on when the user explicitly demands a *directivity number held constant vs frequency* (GDI=τ at every bin). Reuses the same A/R covariance assembly; adds the QCQP constraint `w^H D w = 0` and the GRPQ per-driver band gating that generalizes crossovers. Ship after (ii) is green.
- **(iv) ACC / sound-zones — NOT a separate mode.** Verified across Topics 1, 3-ACC, 7: ACC's `(R_b, R_d)` generalized eigenproblem is *mathematically identical* to the Luo accept/reject eigenproblem. It is subsumed as a **region-selection feature** of the eigen max-ratio sub-mode (`eigh(A, R+εI)`): "max front/back ratio," "null this rear sector," classic ACC all fall out by choosing accept/reject angular masks. Planarity control's machinery is moot in the far field; only its diagonal direction-weighting `Γ` survives, as per-direction angular masks. **No standalone ACC or planarity mode is warranted.**

**Staging:** Stage A = LS pressure-matching (ii) + DS baseline + presets. Stage B = eigen max-ratio sub-mode (covers ACC/max-DI by region choice). Stage C = Luo constant-DI QCQP (iii). Filter realization (§3) runs after every mode.

---

## 2. CORE MATH TO IMPLEMENT

**(A) Pressure-matching closed form (PRIMARY).** Define `G := H_f^T` (literally the GLL sum `P = G w`). Weighted Tikhonov cost `J = (G w − b)^H W (G w − b) + λ w^H w`, `W = diag(a_n)`. Minimizer (verified for *this* model):

```
w_f = ( conj(H_f) · W · H_f^T  +  λ I_M )^{-1} · conj(H_f) · W · b_f
```

`conj(H_f) W H_f^T` is `[M×M]` Hermitian PSD. **DO NOT** copy the microphone-literature `(H W H^H + λI)^{-1} H W b` form — that solves the conjugated model `P=H^H w` and mirror-steers. Implement with `scipy.linalg.solve(A, rhs, assume_a='pos')` (Cholesky) when `λ>0`; for tiny `λ` prefer the stacked real LS `[√W H^T; √λ I] w = [√W b; 0]` via `numpy.linalg.lstsq`.

**(B) MVDR / LCMV (constrained special cases, house convention).** `c = conj(H[:,look])`, synthetic covariance `R = conj(H_f) W H_f^T`:
- MVDR: `w = (R+εI)^{-1} c / (c^H (R+εI)^{-1} c)`.
- LCMV (hard nulls): `C = [c_look, c_null1, …]` `[M×K]`, `g=[1,0,…]^T`, `w = R^{-1}C (C^H R^{-1}C)^{-1} g`. Max `K≤M` (≤ M−1 independent nulls).

**(C) Luo A/R covariance + generalized eigenproblem (SPECIALIST).** Per frequency, `[N×N]→` here `[M×M]` (M=drivers):
```
A = Σ_n a_n f_A(r_n) conj(H_n) H_n^T / Σ_n a_n f_A(r_n)        # accept covariance, Hermitian PSD
R = Σ_n a_n f_R(r_n) conj(H_n) H_n^T / Σ_n a_n f_R(r_n)        # reject covariance
```
Generalized Rayleigh quotient `G(w) = w^H A w / w^H R w`; GDI(dB) = `10 log10 G`. **Pass 1:** `τ_max(f) = scipy.linalg.eigh(A, R, eigvals_only=True).max()` — the per-f directivity *ceiling* (NOT the answer; never ship `eigh` as constant-DI). Pick one constant `τ* = min(10^{target_dB/10}, floor over f of τ_max(f))` — use the **min over f** of the ceilings for a single τ feasible at every frequency (the paper's `min{6dB, 10log10 max τ}` reading is an OPEN flag, §9). **Pass 2:** at fixed τ*, `D = A − τ*R` (indefinite Hermitian), solve the QCQP `w^H D w = 0`:
- **MSCD** (max sensitivity / distortionless min-norm): `min w^H w s.t. w^H D w=0, c^H w=1`. Analytic: `w(λ)=μ(I−λD)^{-1}c`, `μ=1/(c^H(I−λD)^{-1}c)`, scalar secular root `c^H(I−λD)^{-H}D(I−λD)^{-1}c=0` via `scipy.optimize.brentq` (root nearest 0, bracketed between eigenvalue-reciprocal poles straddling zero).
- **MECD** (max efficiency, unit-norm): `max w^H C w s.t. w^H D w=0, w^H w=1` (C=A in the paper's experiments). Projected-ascent loop (~5 iters): gradient step `w+αCw`, project onto `w^H D w=0`, renormalize.

**(D) GRPQ generalized crossovers (folds into R).** `Γ=diag(diag(R))`, `Λ=diag(λ_n(f))∈[0,1]` per-driver per-frequency band gates (→1 in band, →0 out): `R̂ = ΛRΛ + Γ(I−Λ²)`. Use `R̂` (not raw R) in the QCQP so out-of-band drivers auto-vanish — no explicit crossover filters.

**Convention flags.** (1) The exp(−jωt)/exp(+jkr) engineering time convention does **not** separately matter for directivity — it's a real power ratio invariant under global conjugation; only the *relative* conjugation pinned in the global note matters. (2) Hermitian (`^H`, conjugate) transpose everywhere, never plain `^T` (papers using `^T` are real time-domain FIR formulations). (3) **Down-weighted as UNVERIFIED:** the "MECD less-lobing / MSCD more-lobing" characterization — empirical, not confirmed in the fetched text. (4) **Implementation-time check (do not re-research now):** the exact GRPQ R̂ sign and the MSCD secular form were reconstructed from a summarizer; verify against the local PDF when coding those two blocks.

---

## 3. FILTER REALIZATION RECIPE (the crux)

**Verdict: TWO-STEP (optimizer → `w_m(f)` → fit), default LINEAR-PHASE FIR via IFFT+window, with ONE shared modeling delay τ applied identically to all drivers.** This is unambiguous and is the only path that natively realizes arbitrary magnitude AND phase.

**Why FIR, not biquads:** `w_m(f)` is complex with non-trivial phase that *is* the inter-driver time-of-flight / steering. A minimum-phase biquad cascade Hilbert-locks phase to magnitude and **cannot** carry steering phase. (Correction carried: "FIR is the *only* faithful realization" is overstated — IIR+allpass+delay also works mathematically — but FIR is the practical universal choice.) `scipy.signal.firls/firwin2/remez` all produce real linear-phase filters from a magnitude spec and **cannot** fit arbitrary phase; there is no `scipy.signal.invfreqz`.

**The recipe (per driver):**
1. Interpolate `w_m(f)` from the (log/sparse) BEM grid onto a dense uniform linear FFT grid (`Nfft ~ 2^16`), interpolating **(log|w|, unwrapped phase)** or (Re, Im) — never wrapped phase (the #1 error source).
2. **Conjugate `w_m(f)` before IFFT** (or flip the whole H tensor to DSP convention up front). *[Correction carried — Topic-3 claim 6 was REFUTED: numpy's `fft` uses the exp(−j) analysis kernel, so a NumCalc engineering-convention weight (transfer ~exp(+jωD)) must be conjugated to land causally; a straight ifft peaks anti-causally.]*
3. Hermitian-extend (`W(−f)=conj(W(f))`, real DC/Nyquist) so taps come out real.
4. Multiply by `exp(−j 2π f τ)` (shared modeling delay), `np.fft.ifft`, `np.fft.fftshift`, truncate to `Ntaps` about center, window with `scipy.signal.windows.kaiser(Ntaps, beta=8)`.
5. Verify: `scipy.signal.freqz`, remove the `(Ntaps−1)/2` linear phase, assert magnitude AND inter-driver relative phase match within tolerance and pre-truncation IR decayed below ~−120 dB.

**Cardinal-rule constraint:** τ must be **identical for all drivers** — a common `exp(−j2πfτ)` factors out of `P` (pure latency, no beam change); a per-driver delay or per-driver min-phase conversion re-steers the beam. Assert equal-τ.

**Tap/latency budget:** τ ≈ `(Ntaps−1)/2` samples. Lowest steered frequency / steepest phase slope sets the tap count — full-band woofer beams down to ~50–100 Hz at 48 kHz may need `2^12–2^16` taps; mid/high-only beams need hundreds–low thousands. Expose `Ntaps` and τ as user latency/fidelity knobs.

**Fittability:** add a frequency-smoothness regularizer on `w_m(f)` *before* fitting (GRPQ Λ gating and/or explicit cross-bin smoothing) so short filters are realizable. Keep one-step joint filter-and-sum as a documented fallback only if latency is critical or weights won't smooth.

**Optional low-latency IIR export:** fit the **complex** `w_m(f)` via a rolled Levy equation-error solve (`np.linalg.lstsq`) refined by `scipy.optimize.least_squares` (output-error), factor to biquads with `scipy.signal.zpk2sos`, stabilize poles, and verify relative phase preserved. **NEVER** `scipy.signal.minimum_phase` per driver. Fall back to FIR if IIR can't hold the steering phase ramp.

**Convention guard test:** feed `w=exp(−j2πfT)` through the realizer and assert the filter delays by `+T` (not `−T`) — catches the conjugation/delay-sign error that flips steering.

---

## 4. REGULARIZATION & ROBUSTNESS

**Mandatory:** Tikhonov diagonal loading on the inverted matrix, exposed as a **single user-facing robustness knob = a White-Noise-Gain (WNG) floor in dB.**

- **Master identity (under `c^H w=1`):** `WNG(w) = 1/||w||²`, so a WNG floor ≡ a weight-norm bound `||w||² ≤ 1/W_floor`. WNG is the robustness proxy against driver self-noise and independent per-channel gain/phase/position error.
- **Loaded form:** `w(ε) = (R+εI)^{-1} c / (c^H (R+εI)^{-1} c)`. ε=0 → max-directivity (fragile); ε→∞ → delay-and-sum (`w=c/M`, max WNG=M). `WNG(ε)` is continuous and strictly **monotone increasing** in ε → solve `WNG(ε)=W_target` per frequency by **plain 1-D bisection on log ε** (do NOT copy Atkins' two-phase Algorithm 1 — that targets a non-monotone SNR-gain; for WNG a single bisection suffices). Use the bounded `α∈[0,1]`, `Γ_α=(1−α)R+αI`, `ε=α/(1−α)` parameterization internally for stability.
- **Numerical floor:** always add `ε_min ≈ 1e-10·trace(R)/M` before Cholesky (R is rank-deficient at low f where drivers radiate alike). Detect and *flag* low-f bins where the WNG floor is unreachable (ε clamps, DI rolls off gracefully) — never emit garbage.
- **Low-frequency scaling (CORRECTED):** `κ(R) ~ (kd)^{-2(N-1)}`, `WNG ~ (kd)^{+2(N-1)}` — i.e. **6·(order) dB/octave**, NOT the finder's 4(N−1). Don't over-budget headroom. A constant WNG floor vs f auto-grows ε at low f, yielding a constant-directivity beam until physics clamps.
- **Condition-number-targeting loading (CORRECTED):** `ε = (λmax − κmax·λmin)/(κmax − 1)`, clipped ≥0 — NOT `λmax/κmax`.
- **Normalize R** (by `trace/M`) so the WNG floor means the same across frequencies/drivers.
- **The one knob, in user language:** slider `s∈[0,1]` → `W_floor_dB = W_min + s(W_max − W_min)`, `W_min≈−20 dB` ("sharpest beam, needs matched low-noise drivers") to `W_max=10log10(M) dB` ("most forgiving / delay-and-sum"). Ship −6 dB as "balanced." Optional alt framing = Boyd's per-driver matching tolerance ρ(%) with the honest floor `rejection ≥ rejection_nominal + ρ/(1−ρ)` (ρ=5% ⇒ no better than ~−25.6 dB).
- In the eigen/Luo modes the GRPQ `Γ(I−Λ²)` term doubles as the conditioning floor (anisotropic per-driver shrinkage toward diag(R), not isotropic εI — do not describe it as plain WNG/Tikhonov).

**Self-test invariants:** WNG monotone↑ and DF monotone↓ in α; `w(α=1)=c/M`; loaded matrix Hermitian PD (Cholesky succeeds); `c^H w=1` exactly (distortionless ⇒ phase-origin intact).

---

## 5. TARGET-PATTERN SPECIFICATION & GUI MODES

Grounded in commercial CD speakers (B&O Beolab 90, Kii Three, Dutch&Dutch 8c): the market consensus is **named PRESET patterns + DISCRETE/steered direction**, never a free continuous "set beamwidth = 34°" number (arbitrary continuous beamwidth is physically unrealizable for a sparse heterogeneous array and invites impossible requests).

**Recommended v1 modes:**
- **Pattern preset picker:** Omni / Cardioid / Wide / Narrow, optionally a single **"cardioid-order" slider** `a∈[0,1]` in `T(θ)=a+(1−a)cosθ` (omni a=1 → cardioid a=0.5 → hypercardioid a=0.25 → figure-8 a=0). Beolab 90 ships purely discrete (3 widths × 5 directions); a slider is the flexibility-vs-simplicity §9 question.
- **Steering direction picker:** continuous (az, el) on the sphere is cheap to expose (it's just a phase progression / re-centering of the look vector `c` and the accept window `f_A`), unlike continuous beamwidth.
- **Constant-directivity toggle:** "hold pattern constant over X–Y Hz" → use the SAME angular target `g(Θ)` at every f (LS route), or switch to the Luo τ-constant QCQP (advanced).
- **Reject-region choice (surface explicitly):** whether `f_R` = full sphere (→ classical-DI-like) or a rear/side window (→ true cardioid/null steering). GDI ≠ classical DI unless `f_R` is the uniform sphere.

Each preset maps to an analytic real target `b_f(n) = A0(f)·g(Θ_n)` on the Lebedev grid (steering phase emerges from H's preserved time-of-flight; b stays real). Nulls = `b=0` at chosen directions (soft) or LCMV hard constraints (toggle). The B&O lineage confirms the LS-pressure-matching-to-target structure (its exact solver is undisclosed; "regularized LS pressure-matching" is the faithful formalization, not a stated B&O fact).

---

## 6. VALIDATION TARGETS

Two Phase-2 V-tests, mirroring the gameplan V-test style, both using the dataset's Lebedev quadrature for DI. All formulas in engineering convention; the cardioid null test is the CI-safe convention guard (a symmetric beamwidth test CANNOT catch a global sign flip).

**V-CARDIOID (primary, cheap, exact, convention-sensitive).** Two-element endfire monopole array, weights `w_f=1, w_r=−exp(+jω·(a/(1−a))·d/c)`. Anchors (convention-independent, verified by closed form AND sphere integral):

| Pattern | a | Q=3/(4a²−2a+1) | DI (dB) | null angle |
|---|---|---|---|---|
| Cardioid | 0.5 | 3 | **4.771** | **180.0°** |
| Supercardioid (max FBR) | (√3−1)/2=0.36603 | 3.732 | **5.719** | 125.26° |
| Hypercardioid (max DI) | 0.25 | 4 | **6.021** | 109.47° |
| Dipole | 0 | 3 | 4.771 | 90° |

Assert: null at `acos(−a/(1−a))` within ±0.5° (the sign-flip guard — must land at 180°, not 0°); DI via Lebedev quadrature within ±0.1 dB. Element factor `exp(−jk u·r_m)`, delay `exp(+jωτ)` — encode once in a tested helper.

**V-CBT (constant-beamwidth).** Curved monopole arc (R≈0.5 m, total cap 2θ₀≈40°, ~41 elements), **REAL** Legendre amplitude weights (curvature time-of-flight lives in H — honors the cardinal rule, decouples the test from delay-sign). Use Keele's polynomial `U(x)=1+0.066x−1.8x²+0.743x³` (x=θ/θ₀, U=0 for x>1; **provenance corrected: this is from the Keele CBT *patent* lineage, NOT AES Paper 1**). Assert −6 dB beamwidth converges to **≈0.64·(2θ₀)** above an empirically-measured cutoff and is flat vs f (std < ~2° over the constant band). **Two regimes:** untruncated ≈0.64; −12 dB-truncated ≈0.78 — pick the threshold regime deliberately. Do NOT hardcode a closed-form cutoff (scaling only: `f_low ∝ c/L_arc`, `f_high ~ c/(2·spacing)`).

**V-5 extension (the round-trip gate).** Extend the existing two-driver superposition / phase-origin test to a full beamforming round-trip: design w from a known steered target, reconstruct `P(f,dir)=Σ_m w_m H[m,f,dir]`, assert the lobe points the *commanded* way (not its mirror) and reproduces the delay-and-sum null. This single test guards both the conjugation convention and the cardinal rule.

---

## 7. LIBRARIES & DEPENDENCIES

**Numerical stack (CORE, all already present, BSD):** `scipy.linalg.solve(assume_a='pos')` (ridge LS), `scipy.linalg.eigh(A, R+εI)` (generalized eigenproblem — returns ascending eigenvalues, take largest; R must be PD), `scipy.optimize.brentq` (MSCD secular root), `scipy.linalg.cho_solve` (loaded MVDR), `scipy.signal` (freqz/zpk2sos/windows; NOT firls/firwin2/remez for the complex-phase fit — they can't), `scipy.special.sph_harm_y` (reuse, already in project; per memory, NOT removed `sph_harm`).

**Reuse verdicts:**
- **Luo MECD/MSCD math** — implement in-house from arXiv:2407.01860 (no code released). PRIMARY methodological core.
- **pyroomacoustics** (MIT) — REFERENCE-ONLY. Transcribe the ~10-line DS/LCMV/MVDR algebra structure; do NOT depend (2D-only, receive-convention, sign-inconsistent internally).
- **Acoular** (BSD), **sfs-python** (MIT), **spaudiopy / sound_field_analysis-py** (MIT) — REFERENCE-ONLY / SKIP. sfs has no LS solver (analytic driving functions only); borrow `sfs.tapering` window ideas for sidelobe control. Promote SH libs to deps only if an SH-domain design route is later adopted (not needed for a handful of drivers).
- **cvxpy** (Apache-2.0) + **Clarabel** (Apache-2.0) or **SCS** (MIT) — OPTIONAL, only for the Luo SDP relaxation (Eq.19) or a multi-constraint SOCP (per-driver power caps + WNG floor). The single-quadratic-constraint MECD/MSCD need no conic solver. **License trap: never bundle ECOS (GPL)** — it historically came in via cvxpy's default.
- **Grassin demo / MathWorks article** — SKIP (viz/concept only).

All chosen deps are MIT/BSD/Apache-2.0 — compatible with the self-contained open-source app.

---

## 8. EXPORT FORMATS

**PRIMARY v1 = per-driver FIR.** The beam is non-minimum-phase; FIR is the faithful realization.
- Export per-driver **mono 32-bit-float impulse responses** PLUS an auto-generated **CamillaDSP YAML** wiring them via `Conv`. *(Correction carried: the universal common denominator is 32-bit-**float coefficients**, not the WAV **container** — miniDSP OpenDRC/Flex want raw IEEE-754 float binary or plain text, not WAV; CamillaDSP accepts WAV. Offer both: WAV + plain-text taps (CamillaDSP `Raw`/`TEXT`, one tap/line).)*
- Export controls: design sample rate (44.1/48/96 kHz — taps are fs-specific), tap count, IR phase type (linear vs min) with correct peak-position/latency convention (REW centers linear-phase FIR mid-file; min-phase at sample 0).

**SECONDARY/optional v1 = per-driver biquad cascade for the TONAL/EQ stage ONLY** (T/S terminal-response correction, level/shelf — NOT the beam), with an explicit caveat that biquads do not reproduce the beam. Offer **both a1/a2 sign conventions**: (1) miniDSP / REW-"Generic"-export (a1,a2 pre-negated, `y=b0x+b1x1+b2x2+a1y1+a2y2`); (2) RBJ/CamillaDSP standard (`−a1y1−a2y2`). miniDSP→CamillaDSP requires flipping a1,a2 back. Note: REW *internal* is RBJ-standard; only its Generic export pre-negates.

**Interop bonus = per-driver filtered .frd** (apply `w_m` to the on-axis H slice → freq/mag/phase) so the VituixCAD/REW-expert user audits the designed directivity/power response in his own tools.

**Plug-in to existing `io/`:** add an exporters submodule alongside the Phase-1 HDF5 writer; the export acceptance gate is a round-trip test (design → export → re-import taps → re-evaluate `P(f,dir)` vs the intended beam).

---

## 9. OPEN QUESTIONS FOR THE USER

1. **Filter output / deployment target (highest priority):** FIR vs IIR; which target DSP (CamillaDSP / miniDSP Flex/OpenDRC / Hypex/Powersoft plate amp / VituixCAD audit)? This fixes the export internal representation and whether IIR fitting is a required feature.
2. **Tap / latency budget:** what is the lowest frequency the beamformer must steer (sets the dominant tap count), and the maximum acceptable bulk latency τ? Cap LF directivity for realizability if needed.
3. **Robustness policy:** what WNG floor (dB) / max driver-excursion tolerance bounds λ for the real woofer/mid/tweeter array? Sets the default robustness slider mapping.
4. **Anechoic-only vs in-room v1:** confirm Phase-2 v1 is free-field far-field directivity only (the current H-on-sphere contract), or whether a near-field/in-room sound-zone path (measured/simulated RIRs, control points off the sphere) is wanted — that is a separate, larger feature.
5. **Target spec UX:** discrete named presets (Omni/Cardioid/Wide/Narrow) only, or a continuous cardioid-order slider + continuous steering? Market precedent is purely discrete; flexibility vs honesty tradeoff.
6. **Reject-region semantics / constant-DI intent:** for the constant-DI mode, what GDI target τ and what accept/reject windows (`f_A`, `f_R` — CTA-2034 listening window vs full sphere) match the user's "constant directivity" intent? And the exact τ* clamp semantics (min-over-f feasibility floor vs allowing out-of-range bins to fall back to single-driver omni).
7. **Nulls — soft vs hard:** `b=0` soft target vs LCMV hard constraints (max M−1 nulls)? Likely both with a UI toggle.

**Implementation-time verification flags (carry into the gameplan, do NOT re-research now — the adversarial verification is already done):** (a) confirm H's on-disk convention via the round-trip steering test before locking the solver; (b) open the local Luo PDF to verify the GRPQ `R̂=ΛRΛ+Γ(I−Λ²)` sign and the MSCD secular form before coding those two blocks; (c) empirically verify the IFFT conjugation/delay sign with the pure-delay round-trip test; (d) confirm CamillaDSP `Biquad Free` and VituixCAD a1/a2 signs against current docs at export time.

---

# Part II — Full Research Dossier (per-topic findings + adversarial verification)

> Each topic below carries the finder's report, its cited sources, the design decisions it
> implied, open questions, and the adversarial verifier's verdicts (confirmed / partially-correct
> / refuted / uncertain) with corrections. Where the Synthesis and a finding disagree, the
> Synthesis (which weighs the verdicts) wins.

## TOPIC: Yuancheng Luo, "Constant Directivity Loudspeaker Beamforming" (EUSIPCO 2024, arXiv:2407.01860) — exact MECD/MSCD/GDI/GRPQ formulation and its mapping onto BeamSimII Phase 2 (per-driver complex weights w_m(f) from H[m,f,direction]).

### Report
# Constant Directivity Loudspeaker Beamforming (Luo, EUSIPCO 2024) — Formulation & BeamSimII Mapping

**Source note on confidence.** All equations below were extracted from the arXiv HTML (`/html/2407.01860v3`) via an LLM-summarizer that hedged and compressed in places. I have reconstructed the math into self-consistent form and labelled each block CONFIRMED (appeared verbatim/near-verbatim and is internally consistent) or INFERRED (reconstructed from paraphrase). The full PDF is saved locally at `/Users/andy/.claude/projects/-Users-andy-beamsim2/9d574e36-7a70-4541-a590-9f302912f029/tool-results/webfetch-1781992460544-a81tvy.pdf` and should be opened for adversarial verification of the GRPQ (Eqs 3-6) and MSCD (Eqs 20-22) blocks before they are coded.

---

## 0. Notation and the central object: the steering vector d(r)

- `N` = number of loudspeakers/drivers in the array (paper's experiment: N=3, a mid/full-range/tweeter array; solver test N=8).
- `r` = a Cartesian unit direction on the sphere (the paper parametrizes directions as unit vectors r, not (theta,phi)).
- `d(r) ∈ C^N` = the **steering vector = the stacked per-loudspeaker anechoic complex pressure responses in direction r** at one frequency. This is EXACTLY BeamSimII's `H[:, f, direction]` (length-N column over drivers, at fixed f and direction). The paper calls these "steering vectors / anechoic frequency responses." (CONFIRMED that d(r) is the per-transducer response vector; the paper never writes the symbol H but the role is identical.)
- `w ∈ C^N` = per-driver complex weights (the deliverable, one vector per frequency).
- Beam/steered pressure in the paper's convention: `P(r) = d^H(r) w` (Hermitian, conjugate on d). This is INFERRED from the MSCD distortionless constraint `c^H w = 1` with `c = d(r_look)`, and from `A = E[d d^H]` giving `w^H A w = E[|d^H w|^2]`.

### *** CRITICAL CONVENTION — weight conjugation (load-bearing for the cardinal phase rule) ***
BeamSimII's forward model is `P(f,dir) = Σ_m w_m H[m,f,dir] = w^T H` — **no conjugate on H**. The paper's quadratic forms use `d^H w`, i.e. they conjugate the response. Worked algebra:

```
paper objective  w^H A w = E_r[ |d^H(r) w|^2 ] = E_r[ |Σ_m conj(H_m) w_m|^2 ]
BeamSimII output P       = Σ_m H_m w_m
Let u_m = conj(w_m):  Σ_m conj(H_m) conj(u_m) = conj( Σ_m H_m u_m ) = conj(P)
=> |d^H w| = |P|  iff  w_paper = conj(w_beamsim)
```

**Therefore: the eigenvector/weights produced by the paper's quadratic forms must be CONJUGATED before being inserted into BeamSimII's `P = Σ_m w_m H[m,f,dir]`.** Equivalently (and cleaner to code): **build A, R, C, c directly from `conj(H)`** — i.e. set `d(r) := conj(H[:,f,dir])` — and then use the resulting w directly with no conjugation. Either path works; pick one and assert it in a test. Getting this backwards inverts inter-driver phase and silently mis-steers the beam — exactly the failure the cardinal rule and `tests/test_phase_origin.py` (V-5) forbid. (Time-convention exp(-jωt) vs exp(+jωt) does NOT separately matter here: directivity is a real power ratio `w^H A w / w^H R w`, invariant under global conjugation; only the *relative* conjugation between the paper's `d^H w` and BeamSimII's `w^T H` matters, and that is the conjugation above.)

---

## 1. Accept covariance A and reject covariance R  (Eq. 1)

```
A = E_{r ~ f_A}[ d(r) d^H(r) ]          # [N x N] Hermitian PSD, per frequency
R = E_{r ~ f_R}[ d(r) d^H(r) ]          # [N x N] Hermitian PSD, per frequency
```
(CONFIRMED, Eq. 1.) `(.)^H` = conjugate (Hermitian) transpose. Both are N×N (drivers × drivers), one pair **per frequency**.

- `f_A(r)`, `f_R(r)` are **probability density functions** over direction (the paper treats A,R as *expectations/averages*, normalized to integrate to 1 — INFERRED but strongly implied). `f_A` = "forward-facing listening window" (the **accept**/target region — this is where BeamSimII's target beamwidth/cardioid shape is encoded). `f_R` = "side-facing reflection window" (the **reject** region). A separate evaluation density `f_C` defines C for heterogeneous arrays (see §2).
- **Quadrature realization for BeamSimII (the GLL summation):** the paper writes expectations; on BeamSimII's Lebedev grid with weights `q_n` (Σ q_n = 4π) and a direction-selecting/weighting density `f(r_n) ≥ 0`:
  ```
  A = Σ_n q_n f_A(r_n) d(r_n) d^H(r_n)   /  (Σ_n q_n f_A(r_n))     # normalized expectation
  R = Σ_n q_n f_R(r_n) d(r_n) d^H(r_n)   /  (Σ_n q_n f_R(r_n))
  ```
  In matrix form, with `D = [d(r_1) ... d(r_Ndir)]` (N×Ndir) and diagonal weight `W_A = diag(q_n f_A(r_n))`: `A = D W_A D^H / trace-normalizer`. (This is the concrete bridge from H to A/R; the normalizer is a positive scalar that cancels in the Rayleigh quotient, so it can be dropped if only the ratio matters — but keep it if you want GDI to read in absolute dB.)

**Important:** GDI (generalized DI) ≠ classical DI unless `f_R` = uniform over the whole sphere. The classical DI is on-axis power / sphere-average power. Here the denominator is over the *reject* density, which may be a side/back window. The user's "target beam (beamwidth/cardioid)" sets `f_A`; whether `f_R` is the full sphere (→ classical-DI-like) or a reject window (→ true cardioid/null steering) is a **design choice to surface in the UI**, not bury.

---

## 2. Generalized Rayleigh quotient (GRQ) and MECD vs MSCD  (Eqs. 2, 9, 20)

**GRQ (Eq. 2, CONFIRMED):**
```
G(A,R,w) = (w^H A w) / (w^H R w)
         = (x^H Q x)/(x^H x),   with  x = L^H w,  Q = L^{-1} A L^{-H},  R = L L^H (Cholesky)
```
The unconstrained maximizer of GRQ is the **top generalized eigenvector of (A,R)**: `w* = L^{-H} v`, where `v` = eigenvector of `Q` for its largest eigenvalue. In SciPy this is exactly `scipy.linalg.eigh(A, R)` → take the eigenvector for the largest eigenvalue. **This max-GRQ value is the maximum achievable directivity at that frequency — it is the UPPER BOUND, not the constant-directivity answer (see §3).**

**Diagonalization used throughout (Eq. 7, CONFIRMED):** `Q = V E V^H`, `z = V^H x`, `E = diag(eigenvalues)`. So `τ = (z^H E z)/(z^H z)` ranges between the min and max eigenvalues of Q.

**MECD — Maximum Efficiency Constant Directivity (Eq. 9, CONFIRMED):**
```
argmax_w  w^H C w   s.t.   w^H D w = 0,   w^H w = 1
```
- `C` is a Hermitian "efficiency"/evaluation covariance. **In the paper's experiment C = A** (accept covariance), but in general C is defined over a *separate* evaluation density `f_C` so heterogeneous drivers' efficiency can be scored differently from the accept region. (CONFIRMED that C is generally separate, set = A in experiments.)
- `D = A − τ R` (the constant-GDI constraint matrix; see §3). `w^H D w = 0` ⟺ `G(A,R,w) = τ`.
- Interpretation: among all weight vectors that hit *exactly* the target directivity τ, pick the one that maximizes acoustic-electrical efficiency `w^H C w` at unit weight norm.

**MSCD — Maximum Sensitivity Constant Directivity (Eq. 20, CONFIRMED):**
```
argmin_w  w^H w   s.t.   w^H D w = 0,   c^H w = 1
```
- `c = d(r_look)` = steering vector in the look/on-axis direction. `c^H w = 1` is a **distortionless** constraint (fix on-axis response to unity).
- Minimizing `w^H w` minimizes electrical input power for fixed acoustic on-axis output → **maximizes sensitivity** (acoustic-out per electrical-in), subject to constant directivity.

**MECD vs MSCD difference (the practical distinction):**
- MECD: unit-norm weights, maximize an efficiency quadratic → favors "efficient" use of the most capable drivers; experimentally MECD beampatterns show **less lobing**.
- MSCD: distortionless on-axis, minimize weight energy → flat on-axis response guaranteed, but experimentally **more lobing** than MECD.
- Both enforce the *same* constant-GDI constraint `w^H D w = 0`.

---

## 3. The "constant" in constant directivity + the GDI constraint  (Eq. 8)

This is the crux and the easiest thing to get wrong. `eigh(A,R)` gives *maximum* directivity per frequency, which **varies with f** — that is NOT constant directivity. Constant directivity = enforce the **same** target ratio `τ*` at **every** frequency via the equality constraint:

```
G(A,R,w) = τ   ⟺   w^H A w = τ · w^H R w   ⟺   w^H (A − τR) w = 0   ⟺   w^H D w = 0,   D = A − τR
```
(CONFIRMED.) **GDI in dB = 10·log10(τ)** = `10 log10( w^H A w / w^H R w )` (CONFIRMED: paper's experiment uses `G(A,R,w)=6 dB`; τ is the linear-scale ratio).

**Choosing the single constant τ* (the paper's recipe, CONFIRMED):**
```
τ* (in dB) = min{ target_dB , 10 log10( max_f  τ_max(f) ) }   # paper: min{6, 10log10 max τ}
```
where `τ_max(f)` = largest generalized eigenvalue of (A(f), R(f)) (the per-frequency directivity ceiling). You cannot demand more directivity than the array can deliver at its worst frequency, so τ* is clamped to the floor of the per-frequency maxima.

**Eigendecomposition of D (Eq. 8, CONFIRMED):**
```
D = A − τR = L V (E − τ I) V^H L^H
```
where `R = L L^H` (Cholesky), and `V, E` diagonalize `Q = L^{-1} A L^{-H} = V E V^H`. Because τ < some eigenvalues and > others, `E − τI` is **indefinite** → D is indefinite → `w^H D w = 0` is a (non-convex) quadratic equality constraint. This is what makes both problems QCQPs rather than plain eigenproblems.

---

## 4. How it is solved (numerical recipe)

The paper frames both designs as **Quadratic Equality-Constrained Quadratic Programs (QCQP)** and gives **fast analytic / few-iteration** solutions (NOT a black-box SDP — they show their projected-ascent beats a Dinkelbach-style "DM" baseline, 5 vs 50 iterations, and beats SDPT3 SDP which needs O(N²) variables).

**MECD — Algorithm 1, "Projected Ascent" (CONFIRMED structure):**
```
Require: Hermitian C; D = A − τ*R; step size α>0; K iterations; init w_0 (e.g. top gen-eigvec of (A,R), then projected onto w^H D w=0)
for k = 1..K:
    w_tmp = w_{k-1} + α C w_{k-1}            # (2) gradient-ascent step on w^H C w
    v*    = Proj(w_tmp, D)                    # (3) project onto feasible surface w^H D w = 0 (Eq. 12)
    w_k   = (w_tmp + v*) / || w_tmp + v* ||   # (4) re-normalize to unit norm
return w_K
```
- **Projection subproblem (Eq. 12, CONFIRMED):** `argmin_v v^H v s.t. (w_k+v)^H D (w_k+v) = 0`. Closed-form (Eq. 14): `w_k + v* = V (I − λ E)^{-1} V^H w_k`, where here V,E are eigenpairs in the rotated coordinates and `λ` is the Lagrange multiplier found from a **scalar secular equation** (§4a). (Reported as ~5 iterations to converge.)

**MSCD — analytic (Eqs. 21-22, CONFIRMED structure, INFERRED exact signs):**
```
w(λ) = μ (I − λ D)^{-1} c                              # (21) from Lagrangian, λ real Lagrange multiplier
Enforce constraints:
   c^H w = 1   =>   μ = 1 / ( c^H (I − λ D)^{-1} c )
   w^H D w = 0 =>   c^H (I − λ D)^{-H} D (I − λ D)^{-1} c = 0   # (22) secular equation in scalar λ
```
Solve the scalar secular equation (22) for λ (root nearest 0), back-substitute → closed-form `w`. No iteration loop beyond 1-D root finding.

### 4a. The quadratic secular-equation root finding (Sec. III)
Both designs reduce the Lagrange multiplier to a **scalar** root of a secular function (CONFIRMED):
```
S(λ) = u^H (I − λ E)^{-H} E (I − λ E)^{-1} u = 0      # MECD form (Eq. 15); MSCD analogous (Eqs 23-24)
```
- `S(λ)` has poles at `λ = 1/E_ii`. The relevant root `λ*` is the one **nearest 0** (smallest-norm projection / multiplier), bracketed between the two eigenvalue-reciprocal poles straddling zero (brackets `b_-`, `b_+` = the largest negative and smallest positive `1/E_ii`). Within that pole-free bracket S is monotone → a 1-D root finder (Newton/bisection, `scipy.optimize.brentq`) converges fast. (Bracket-selection details INFERRED from "λ* nearest 0" + secular structure.)

**Bottom line on what gives w_m(f):** per frequency, after assembling A(f),R(f),C(f),c(f) from `conj(H)`: form `D = A − τ*R`, then either run Algorithm-1 projected ascent (MECD) or solve the 1-D secular equation and back-substitute (MSCD). Then **conjugate the result** (or, if you already built from conj(H), don't) to get the BeamSimII `w_m(f)`.

---

## 5. GRPQ — Generalized Rayleigh Penalty Quotient (generalized crossovers)  (Eqs. 3-6)

This is the loudspeaker-specific novelty (mic arrays don't need it): a per-driver, per-frequency penalty that drives `|w_n(f)| → 0` outside driver n's operating band, producing **generalized crossovers** without explicit crossover filters.

```
Γ = diag( diag(R) )                       # (3) diagonal matrix = the diagonal of R (per-driver self-power), [N x N]
Σ = diag[σ_1, ..., σ_N],   0 ≤ σ_n ≤ ∞    # (3) per-driver penalty strengths (unbounded)
Λ = (Σ + I)^{-1/2} = diag[λ_1,...,λ_N],  0 ≤ λ_n ≤ 1   # (4) bounded reparametrization; small λ_n = heavy penalty
```
**The penalty enters the GRQ denominator** (the reject/power term):
```
GRPQ = G(A, R + Γ Σ, w) = (w^H A w) / (w^H (R + ΓΣ) w)         # penalized quotient
```
With the bounded change of variables `y = Λ^{-1} w` (Eqs. 5-6), this becomes a well-conditioned standard GRQ with a **modified reject matrix** (INFERRED exact form — VERIFY against PDF):
```
R̂ = Λ R Λ + Γ (I − Λ²)                    # (6) attenuates OFF-diagonal coupling of R by Λ², adds diagonal floor
```
**Per-frequency / per-driver setting (CONFIRMED mechanism):** `λ_n` is indexed by **both** driver n and frequency f. Set `λ_n(f) → 1` (no penalty) inside driver n's operating band and `λ_n(f) → 0` (full penalty) outside it, following a smooth "operating-range curve" (paper's Fig. 2). Lowering `λ_n(f)→0` outside the band forces `|w_n(f)| → 0`. Because the curve is smooth in f, the crossover transition is **smooth, not abrupt** — and continuity across frequency is enforced *implicitly* through these Λ(f) curves, not via an explicit cross-frequency coupling term. (CONFIRMED: there is NO explicit frequency-loop coupling / no cross-frequency smoothness penalty; regularity comes entirely from the per-driver Λ(f) bands.)

**This replaces explicit crossovers in BeamSimII:** the woofer/mid/tweeter band assignment becomes a set of Λ(f) gating curves the user (or an auto-heuristic from T/S params / driver passband) specifies.

---

## 6. Regularization / robustness

- **Primary regularizer = the GRPQ penalty `ΓΣ` / Λ(f)** itself (§5). It both conditions the problem and zeroes out-of-band drivers.
- **Diagonal loading of R:** R is singular/ill-conditioned at low frequency (drivers nearly in-phase → rank-deficient covariance; Cholesky `R = L L^H` fails). The `Γ(I − Λ²)` term in R̂ adds a positive diagonal floor that **regularizes R** (acts like diagonal loading / white-noise-gain control). At frequencies where only one driver is in-band, A and R collapse toward rank-1 and the design correctly becomes "that one driver only" (paper: "at low frequency directivity grows omni-directional, only mid-range active"). (The paper does not separately name a WNG constraint; the diagonal penalty is the robustness mechanism — INFERRED.)
- Practical: add explicit `R += ε·I` (small ε, e.g. 1e-6·trace(R)/N) as a numerical safety net before Cholesky regardless.

---

## 7. Concrete Python mapping (per-frequency loop)

```python
# H: [M x F x Ndir] complex128, M=drivers ; q: [Ndir] Lebedev weights ; r_hat: [Ndir x 3] unit dirs
# fA, fR : [Ndir] nonneg accept/reject densities ; look_idx : index of on-axis direction
import numpy as np
from scipy.linalg import eigh, cholesky, solve
from scipy.optimize import brentq

w = np.zeros((M, F), complex)           # output weights, BeamSimII convention (P = sum_m w_m H_m)

# ---- pass 1: per-f directivity ceiling to pick the single constant tau* ----
tau_max = np.empty(F)
for f in range(F):
    Dr = np.conj(H[:, f, :])            # <-- build steering vectors from conj(H)  (handles the conjugation)
    A  = (Dr * (q*fA)) @ Dr.conj().T / np.sum(q*fA)     # [M x M], = D Wa D^H normalized
    R  = (Dr * (q*fR)) @ Dr.conj().T / np.sum(q*fR)
    R += 1e-9*np.trace(R)/M*np.eye(M)   # diagonal load (R singular at low f)
    evals = eigh(A, R, eigvals_only=True)
    tau_max[f] = evals.max()            # max GRQ at this f (UPPER BOUND, not the answer)

tau_star = min(10**(target_GDI_dB/10), tau_max.max())   # paper: min{target, max_f tau_max}; note clamp uses max over f of the per-f ceilings? -> see open_q

# ---- pass 2: solve constant-directivity QCQP at fixed tau* every f ----
for f in range(F):
    Dr = np.conj(H[:, f, :])
    Lam = np.diag(lambda_band[:, f])    # GRPQ per-driver operating-band gates in [0,1]
    A  = (Dr*(q*fA)) @ Dr.conj().T / np.sum(q*fA)
    R  = (Dr*(q*fR)) @ Dr.conj().T / np.sum(q*fR)
    Gam= np.diag(np.diag(R).real)
    Rhat = Lam@R@Lam + Gam@(np.eye(M) - Lam@Lam)        # GRPQ modified reject (VERIFY signs vs PDF)
    Rhat += 1e-9*np.trace(Rhat)/M*np.eye(M)
    D  = A - tau_star*Rhat
    # --- MECD: projected ascent (Alg.1) OR  MSCD: secular root-find (Eqs 21-22) ---
    w_f = solve_mecd(A=A, R=Rhat, C=A, D=D)             # returns paper-convention weights
    w[:, f] = w_f                                       # NO extra conjugate: we already used conj(H)
```

**Pitfalls (each must be guarded with a test):**
1. **Conjugation** (§0): build from `conj(H)` xor conjugate the output — never both, never neither. Guard with the two-driver superposition test (V-5).
2. `eigh(A,R)` gives the **ceiling, not constant directivity** — never ship it as the answer.
3. **R singular at low f** → Cholesky fails → diagonal-load before factoring.
4. **Eigenvector phase/gain ambiguity:** generalized eigenvectors are defined up to a complex scale. MSCD's `c^H w = 1` pins it; MECD's `w^H w = 1` pins magnitude but NOT global phase. Fix global phase by rotating so the look-direction response `d^H(r_look) w` is real-positive — otherwise frame-to-frame phase jitter corrupts the per-driver FIR/IIR realization. (This is on top of the cardinal rule: never re-zero an *individual* driver; only a single *common* global phase rotation of the whole w vector is allowed.)
5. **τ* clamp** must use the worst (min over f) of the per-frequency ceilings if you require the SAME τ at every f, OR the paper's `10log10(max τ)` reading — see open question.
6. **GDI ≠ classical DI** unless f_R = uniform sphere — make the reject density explicit.

---

## 8. Related Luo work (context, not new math)
- Luo, "Spherical harmonic covariance and magnitude function encodings for beamformer design," EURASIP J. Audio Speech Music Process. 2021 (open access, Springer). This is reference [9] of the 2024 paper and the origin of the covariance/GRQ machinery (it does mic-array beampattern *fitting* via SH-encoded covariances). The 2024 paper adapts that covariance formalism to *loudspeakers* by adding (a) the constant-GDI equality constraint, (b) MECD/MSCD efficiency/sensitivity objectives, (c) the GRPQ generalized-crossover penalty. For BeamSimII, the 2021 paper is the place to mine the SH-covariance assembly if a smoother (basis-projected) A/R is wanted instead of raw Lebedev quadrature.

### Sources
- (primary) Yuancheng Luo, Constant Directivity Loudspeaker Beamforming (arXiv abstract page, v3) — https://arxiv.org/abs/2407.01860
- (primary) Constant Directivity Loudspeaker Beamforming — full HTML (arXiv:2407.01860v3) — https://arxiv.org/html/2407.01860v3
- (primary) Constant Directivity Loudspeaker Beamforming — PDF (arXiv:2407.01860) — https://arxiv.org/pdf/2407.01860
- (primary) Constant Directivity Loudspeaker Beamforming — EUSIPCO 2024 proceedings PDF (EURASIP) — https://eurasip.org/Proceedings/Eusipco/Eusipco2024/pdfs/0000246.pdf
- (primary) Y. Luo, Spherical harmonic covariance and magnitude function encodings for beamformer design, EURASIP J. Audio Speech Music Process. 2021 (ref [9]) — https://asmp-eurasipjournals.springeropen.com/articles/10.1186/s13636-021-00230-7

### Decisions implied
- BeamSimII must assemble A/R/C from conj(H[:,f,:]) (or conjugate the final weights), and the existing two-driver superposition / phase-origin test (V-5) should be extended to a beamforming round-trip that asserts the steered P(f,dir) = Σ_m w_m H[m,f,dir] matches the intended beam — this is the guard for both the conjugation and the cardinal phase rule.
- Phase-2 implements a per-frequency loop: pass 1 computes τ_max(f) via scipy.linalg.eigh(A,R) to set one constant τ*; pass 2 solves the QCQP at fixed τ* for w(f). Offer two solver modes: MECD (projected ascent, Algorithm 1, ~5 iters) and MSCD (1-D secular root via scipy.optimize.brentq on S(λ), closed-form back-substitution).
- The user's 'target beam (beamwidth / cardioid-order / steering direction)' maps to the accept density f_A(r) (a window around the look direction) and the reject density f_R(r) (whole sphere → classical-DI-like, OR a side/back window → true cardioid/null). Expose f_R choice explicitly; GDI ≠ classical DI when f_R ≠ uniform sphere. Steering = re-centering f_A and the look-direction c = d(r_look).
- Woofer/mid/tweeter crossovers are NOT separate filters: implement them as per-driver, per-frequency Λ(f) gating curves λ_n(f) in [0,1] (1 in-band, →0 out-of-band, smooth transition), auto-initialized from each driver's T/S passband, then folded into the GRPQ modified reject matrix R̂ = ΛRΛ + Γ(I−Λ²).
- Numerical hardening: diagonal-load R (R += ε·trace(R)/N·I, ε~1e-9..1e-6) before Cholesky because R is rank-deficient/singular at low frequency; this doubles as the robustness/WNG regularizer. Resolve the eigenvector phase/gain ambiguity by a SINGLE global phase rotation of w (e.g. make d^H(r_look) w real-positive) — never an independent per-driver re-zeroing.
- Before coding the GRPQ (Eqs 3-6) and MSCD (Eqs 20-22) blocks, open the locally saved PDF and verify the exact sign/structure of R̂ = ΛRΛ + Γ(I−Λ²) and the MSCD solution w = μ(I − λD)^{-1}c with secular equation c^H(I−λD)^{-H} D (I−λD)^{-1} c = 0 — these were reconstructed from a summarizer and are the lowest-confidence equations in the report.

### Open questions
- Exact τ* clamp semantics: the paper writes the experiment's constant GDI as min{6 dB, 10 log10 max(τ)}. 'max(τ)' likely means the max over the per-frequency directivity ceilings, but for a SINGLE constant τ feasible at EVERY frequency you must clamp to the MIN over f of τ_max(f). Confirm against the PDF whether the constant target is feasibility-floored (min_f) or whether out-of-range frequencies are simply allowed to fall back to single-driver omni behavior.
- Exact GRPQ R̂ form and whether the penalty is added as R+ΓΣ (Eq stating G(A, R+ΓΣ, w)) versus the reparametrized R̂ = ΛRΛ + Γ(I−Λ²) — and the precise definition of Γ (diag of R, or diag of R's diagonal magnitudes). Sign of the (I−Λ²) term must be verified in the PDF.
- MSCD exact closed form: confirm w = μ(I − λD)^{-1}c (real part of λ?) and the secular equation c^H(I−λD)^{-H} D (I−λD)^{-1} c = 0, plus how the root λ* nearest 0 is bracketed between eigenvalue-reciprocal poles.
- How d(r) (the anechoic per-driver response) is normalized in the paper — by SPL, by efficiency, or raw — since C = A vs a separate efficiency density f_C changes what 'maximum efficiency' means for heterogeneous drivers. For BeamSimII, decide whether C uses the same Lebedev accept density or a driver-efficiency-weighted density.
- Whether the paper applies any explicit cross-frequency smoothness on w(f) beyond the implicit Λ(f) bands; if frame-to-frame phase/gain jitter appears in practice, BeamSimII may need an added continuity regularizer before the FIR/IIR realization step (not in the paper).

### Adversarial verification verdicts
  - [confirmed] 1. Exactly one conjugation: w_beamsim = conj(w_paper), or equivalently A/R/C/c must be built from conj(H[:,f,dir]). Because the paper uses A=E[d d^H] and d^H w (conjugate on d) while BeamSimII's forward model is P=Σ_m w_m H[m,f,dir] with no conjugate. -> CORRECTION: Confirmed by derivation, but flag one premise as INFERRED rather than verbatim: the paper never writes a standalone beam-output expression P = d^H w. The conjugate-on-d convention is reconstructed from c^H w = 1 (c=d) and A = E[d d^H], which is unambiguous. Note also the conclusion is robust to whether the paper's output is d^H w or w^H d — both give the same |.| and the same single-conjugation result.
  - [confirmed] 2. A = E_{r~f_A}[d d^H], R = E_{r~f_R}[d d^H], each [N x N] Hermitian PSD per frequency, (.)^H conjugate transpose; on Lebedev A = Σ_n q_n f_A(r_n) d(r_n) d^H(r_n) (optionally normalized by Σ q_n f_A); d(r)=H[:,f,direction]. -> CORRECTION: Confirmed. Minor: paper calls f_A/f_R probability density functions (implying normalization) but does not state an explicit ∫=1 condition in the shown text; the quadrature normalizer is therefore a correct-but-implied reconstruction, not verbatim.
  - [confirmed] 3. Constant directivity via quadratic equality constraint w^H D w = 0, D = A − τR, equivalent to G(A,R,w)=τ; GDI(dB)=10 log10(τ)=10 log10(w^H A w / w^H R w). τ clamped to floor of per-frequency ceilings τ_max(f) = top generalized eigenvalue of (A(f),R(f)). -> CORRECTION: None. The apparent gap ('top eigenvalue' vs 'between smallest and largest eigenvalues') is not a conflict: the feasible interval's upper endpoint equals the top generalized eigenvalue = the max-directivity ceiling at that frequency.
  - [confirmed] 4. scipy.linalg.eigh(A,R) top generalized eigenvector gives per-frequency MAXIMUM directivity (upper bound), NOT the constant-directivity answer; deliverable requires solving the equality-constrained QCQP (MECD projected-ascent Alg.1, or MSCD secular-equation Eqs 21-22) at fixed τ* common to all frequencies. -> CORRECTION: None. (Implementation note: scipy.linalg.eigh(A,R) requires R positive-definite; with a reject-only f_R, R can be singular/ill-conditioned — this is exactly what the GRPQ R-hat regularization addresses, see claim 6.)
  - [partially-correct] 5. MECD (Eq.9)=argmax w^H C w s.t. w^H D w=0, w^H w=1 (C=A in experiments, generally separate eval covariance); MSCD (Eq.20)=argmin w^H w s.t. w^H D w=0, c^H w=1 with c=d(r_look). MECD maximizes efficiency at unit norm (less lobing); MSCD distortionless min-energy (more lobing). -> CORRECTION: The math (Eqs 9, 20, C=A in experiments, c=d(r_look)) is fully confirmed. But the qualitative 'MECD less lobing / MSCD more lobing' characterization is an EMPIRICAL claim about the experiments and was NOT verified in the fetched text — mark that sub-clause uncertain. Also state explicitly that MSCD = 'maximum sensitivity' (paper's name) = distortionless minimum-norm (finder's name) to avoid the appearance of a naming discrepancy.
  - [confirmed] 6. GRPQ: Γ=diag(diag(R)), Σ=diag[σ_n] (0≤σ_n≤∞), Λ=(Σ+I)^{-1/2}=diag[λ_n] (0≤λ_n≤1); penalized denominator R̂=ΛRΛ+Γ(I−Λ²); λ_n(f)→0 outside driver n's band forces |w_n(f)|→0, replacing crossovers. λ_n indexed by driver AND frequency; cross-frequency continuity implicit via smooth Λ(f) (no explicit cross-frequency coupling term). -> CORRECTION: Confirmed. The 'no explicit cross-frequency coupling term' part is correct as stated: the regularization is per-frequency (R̂(f) built from Λ(f)); cross-frequency smoothness is the user's responsibility via choosing smooth band curves Λ(f). The paper does not introduce a separate cross-frequency penalty in the shown formulation.
  EXTRA: PRIMARY SOURCE: arXiv:2407.01860v3 HTML (Yuancheng Luo, 'Constant Directivity Loudspeaker Beamforming', accepted EUSIPCO 2024). All equation forms (Eq.1 covariances, Eq.2 GRQ, Eqs 3-6 GRPQ Γ/Σ/Λ/R̂, w^H D w=0 with D=A−τR, Eq.9 MECD, Eq.20 MSCD, Eqs 21-22 MSCD secular solution) were confirmed at equation level from the HTML; the local PDF could not be re-rendered (poppler not installed) but the HTML is the authoritative source and was sufficient.

Two precise caveats the finder should carry into the plan: (1) The beam-output expression P = d^H w is INFERRED, never written verbatim in the paper — only A=E[d d^H] and the MSCD constraint c^H w=1 (c=d) appear. The single-conjugation result (w_beamsim=conj(w_paper)) follows rigorously from these and is robust to d^H-w vs w^H-d ambiguity, but the reader should know the premise is reconstructed, not quoted. (2) MSCD's paper name is 'maximum SENSITIVITY constant directivity'; the finder's 'distortionless minimum-energy' is the same optimization (min ‖w‖² s.t. c^H w=1) — flag this so it doesn't look like a mismatch.

IMPLEMENTATION GOTCHA connecting claims 4 and 6: scipy.linalg.eigh(A, R) solves the generalized eigenproblem only if R is positive-definite (it Cholesky-factorizes R=LL^H, exactly Eq.2's diagonalization). A reject-only / narrow f_R yields a low-rank or ill-conditioned R and eigh will fail or return garbage. The GRPQ penalized denominator R̂ = Γ + Λ(R−Γ)Λ (claim 6) regularizes precisely this — its diagonal-loading-like structure (Γ=diag(diag(R))) keeps R̂ PD. Use R̂ (not raw R) inside eigh/Cholesky, and validate PD-ness before solving. Also: D = A − τR is indefinite (it is by construction a difference straddling the GRQ value), so w^H D w=0 is a genuine equality-constrained QCQP on an indefinite form — eigh(A,R) alone cannot produce it; the MECD projected-ascent or MSCD secular root is required.

EMPIRICAL sub-claim NOT verified: the qualitative 'MECD = less lobing, MSCD = more lobing' ranking. The math distinction is solid (MECD: unit-norm efficiency-max; MSCD: distortionless min-norm), but the lobing characterization should be treated as a hypothesis to confirm against the paper's Fig. results or in BeamSimII simulation, not as an established fact.

ABSTRACT (verbatim fragments, arxiv.org/abs/2407.01860): heterogeneous loudspeaker constraints 'due to arrayed transducers with varying operating ranges in frequency, acoustic-electrical sensitivity, efficiency, and directivity'; proposes 'two novel beamformer designs that optimize for maximum efficiency constant directivity (MECD) and maximum sensitivity constant directivity (MSCD)' with 'fast-converging analytic solutions from quadratic equality constrained quadratic program formulations.'


---

## TOPIC: Regularized least-squares / pressure-matching / mode-matching beamforming for loudspeaker arrays (BeamSimII Phase 2 filter designer)

### Report
# Regularized LS Pressure-Matching Beamforming for Loudspeaker Arrays

This is the general workhorse for "arbitrary user-specified beam shape." Every target — steered main lobe, nulls, cardioid, approximate constant-DI — encodes as a complex target field `b(f, Ω)` sampled on the same sphere grid as `H`, and the weights fall out of one regularized linear solve per frequency. All of the harder named methods (MVDR, LCMV, max-directivity, constant-directivity) are special cases or constrained variants of the same quadratic program.

Notation used throughout:
- `M` = number of drivers, `N` = number of sphere directions (Lebedev nodes), `F` = number of frequencies.
- `H[m, f, n]` = stored complex transfer tensor, shape `[M x F x N]` complex128. For a fixed `f`, write `H_f` as an `[M x N]` matrix.
- `w_f` = driver weight vector, shape `[M]` complex128 (one filter tap per driver at this frequency).
- `b_f` = target/desired pressure field, shape `[N]` complex128.
- `W` = real diagonal `[N x N]` matrix of Lebedev quadrature weights `a_n` (so `sum_n a_n = 4*pi` or `=1` depending on normalization; keep it consistent).
- `(.)^H` = conjugate transpose, `(.)^T` = transpose, `conj(.)` = elementwise conjugate.

---

## 1. Pressure-matching problem and the EXACT closed form (CONVENTION-CRITICAL)

**Forward (GLL) model as BeamSimII computes it:** `P_f(n) = sum_m w_m(f) * H[m,f,n]`. In matrix form this is exactly
`P_f = H_f^T w_f`  (`H_f^T` is `[N x M]`, `w_f` is `[M]`, `P_f` is `[N]`).

**Weighted Tikhonov cost:**
`J(w) = (H_f^T w - b_f)^H W (H_f^T w - b_f) + lambda * w^H w`.

**Normal equations / closed form — derived for THIS model (verified numerically):**
Define the design matrix `G := H_f^T` (so `P = G w`). The minimizer is the standard weighted ridge solution
`w_f = (G^H W G + lambda I_M)^-1 G^H W b_f`.
Substituting `G = H_f^T`, and using `(H_f^T)^H = conj(H_f)`:
**`w_f = (conj(H_f) W H_f^T + lambda I_M)^-1 conj(H_f) W b_f`.**

This is the form to implement. `conj(H_f) W H_f^T` is `[M x M]` Hermitian PSD; the solve is tiny (`M` = handful).

**CRITICAL CONVENTION FLAG (load-bearing, ties to the project's cardinal phase-origin rule).** The "tidy" textbook form often quoted, `w = (H W H^H + lambda I)^-1 H W b`, is **NOT** the solution to the model BeamSimII actually sums. I verified numerically that `(H_f W H_f^H + lambda I)^-1 H_f W b_f` is exactly the minimizer of a DIFFERENT model, `P = H_f^H w` (i.e. it presumes `P_f(n) = sum_m w_m conj(H[m,f,n])`, the conjugated transfer). The two solutions are NOT simple conjugates of each other in general (they only relate by conjugation if you ALSO conjugate `b`). So:
- Picking the wrong one applies conjugated phase to every driver weight, which mis-steers the beam to its phase mirror — precisely the silent mis-steer the project's V-5 / phase-origin rule guards against, and it interacts with the NumCalc engineering time convention (exp(-jwt), outgoing exp(+jkr)) noted in CLAUDE.md.
- **Rule: define `G` by literally writing out `P = G w` for the summation as coded (`G = H_f^T`), then use `w = (G^H W G + lambda I)^-1 G^H W b`. Do not copy the `H W H^H` form from the microphone literature.** This conjugation IS the transmit-vs-receive reciprocity difference (see section 4): the loudspeaker transfer plays the role of the conjugated receive array manifold.

**numpy mapping (one frequency):**
```
Hc = H_f.conj()                       # conj(H_f), [M x N]
A  = Hc @ W @ H_f.T + lam*np.eye(M)    # [M x M] Hermitian PSD
rhs = Hc @ W @ b_f                     # [M]
w_f = np.linalg.solve(A, rhs)          # use solve, not inv
```
Loop over `f` (independent solves; trivially parallelizable). For numerical robustness prefer `scipy.linalg.solve(A, rhs, assume_a='pos')` (Cholesky) when `lambda>0` makes `A` SPD, or build the stacked real LS and call `numpy.linalg.lstsq` on `[sqrt(W) H^T ; sqrt(lambda) I] w = [sqrt(W) b ; 0]` for better conditioning at tiny `lambda`.

**Validation hook:** synthesize a known steered main lobe via this solve, reconstruct `P = H_f^T w`, and confirm the lobe points the commanded way (not its mirror). Reuse the existing two-driver superposition / `tests/test_phase_origin.py` machinery.

---

## 2. Specifying the target field `b(f, Ω)`

`b_f` is just a complex `[N]` vector on the Lebedev grid. Recipes:

**(a) Steered narrow beam.** Two practical choices:
- *Distortionless + shaped:* set `b_f(n) = A0 * g(angle(Ω_n, Ω_steer))` where `g` is a desired angular taper (e.g. `g=cos^p(Θ)` for `Θ<=90deg` else 0, or a raised-cosine of target beamwidth), `A0` an on-axis reference level. Phase of `b` is typically flat (0) in the steer-aligned far field, OR set to the phase of the look-direction column of `H` to keep the on-axis response physical. Increasing `p` narrows the lobe (request only what the array can physically do at that `f`; below the array's natural directivity the solve just returns near-uniform weights).
- *Pure delay-and-sum baseline / steering vector:* the look direction "steering vector" is `d_f = conj(H_f[:, n_steer])` (the conjugated look-direction column), because for transmit, max on-axis pressure uses weights matched to the conjugate of the transfer (phase-conjugation / time-reversal focusing). This is the loudspeaker analogue of the array manifold.

**(b) Cardioid / hypercardioid.** First-order pattern in the steer-relative angle `Θ`: `g(Θ) = a + (1-a) cos(Θ)`, with `a=0.5` cardioid (null at 180deg), `a=0.25` hypercardioid (max DI first-order), `a=0.37` supercardioid (max front-to-back), `a=0` figure-8. Order-`q` patterns: `g(Θ) = (a + (1-a) cos Θ)^q` or Legendre/Chebyshev expansions `sum_q c_q P_q(cos Θ)` (the spherical-loudspeaker far-field beam is exactly `B(Θ)=sum_n d_n (2n+1)/(4pi) P_n(cosΘ)`, so choosing `d_n` = the cardioid's Legendre coefficients gives the target directly). Set `b_f(n) = A0 * g(Θ_n)`; complex (with a 180deg back-null) is naturally represented because `g` can go negative. Nulls: simply set `b_f(n)=0` at the chosen rejection directions (and optionally up-weight those `W` entries, or add hard LCMV constraints — section 4).

**(c) Approximate frequency-invariant / constant-beamwidth.** Use the SAME angular target shape `g(Θ)` (same beamwidth, same `a`/`p`/order) at EVERY frequency: `b_f(n) = A0(f) * g(Θ_n)` with only a per-`f` scalar gain `A0(f)`. Because `b_f`'s angular shape does not change with `f`, the LS solver chases a constant pattern and frequency-invariance falls out wherever the array is physically capable (i.e. above the diffraction/aperture limit). Below that limit the regularizer caps effort and the beam naturally broadens — expected and acoustically correct. See section 5 for the directivity-locked variant.

---

## 3. Regularization parameter `lambda` (effort vs accuracy)

- **Meaning:** `lambda` trades pattern-match error against array effort `||w||^2` (sum of squared driving signals). Large effort = strongly self-cancelling near-superdirective weights, huge driver excursion, and extreme sensitivity to transfer/position errors. `lambda` is exactly the white-noise-gain (WNG) / robustness knob: `WNG = |w^H d|^2 / (w^H w)`; raising `lambda` raises WNG (robustness) at the cost of directivity.
- **Diagonal-loading equivalence:** `+ lambda I` here is identical in spirit to diagonal loading `R -> R + epsilon I` in MVDR (section 4). `lambda` has units that make it comparable to the eigenvalues of `G^H W G`; normalize by `trace(G^H W G)/M` so a dimensionless `beta` is portable across frequencies and arrays: `lambda(f) = beta * trace(G^H W G)/M`.
- **Frequency dependence (Kirkeby-Nelson):** use a frequency-DEPENDENT `lambda(f)`, small in the band where the array is well-conditioned, ramped up at the low-frequency superdirective end and at any out-of-band/critical-frequency region. This is the standard regularized-inversion practice from Kirkeby & Nelson (JAES 1999) "Digital Filter Design for Inversion Problems in Sound Reproduction."
- **Selection — L-curve:** sweep `lambda` over a log grid (e.g. `1e-6` to `1e2` times the trace-normalized scale), plot residual `||H^T w - b||_W` vs effort `||w||` on log-log; pick the corner of maximum curvature. Programmatic corner: maximize the Menger curvature of the three-point L-curve (Hansen). Alternatives: fix a WNG floor (e.g. `WNG >= -10 dB`) and pick the smallest `lambda` meeting it per frequency; or GCV (generalized cross-validation) minimizing `||(I - G G_lambda^+)b||^2 / (trace(I - G G_lambda^+))^2`. The WNG-floor approach is most defensible for a loudspeaker product because it directly bounds excursion/robustness.

---

## 4. Relationship to MVDR / LCMV and diagonal loading

**MVDR (minimum-variance distortionless response), loudspeaker/transmit form.** From `min_w w^H R w  s.t.  d^H w = 1`:
`w = R^-1 d / (d^H R^-1 d)`.
- For loudspeakers, `R` is a SYNTHETIC `[M x M]` covariance, NOT a measured noise covariance: build it by quadrature-integrating the (conjugated) array manifold over a "reject" or total region, `R = sum_n a_n d_f(n) d_f(n)^H` with Lebedev weights `a_n` (this is `conj(H_f) W_reject H_f^T`). `d = conj(H_f[:, n_steer])` is the look-direction (conjugated) steering vector. `R^-1 d` is computed with a solve, then normalized by `d^H R^-1 d`.
- **Diagonal loading regularizes it:** replace `R -> R + epsilon I`, giving `w = (R+epsilon I)^-1 d / (d^H (R+epsilon I)^-1 d)`. `epsilon` bounds WNG / effort exactly like `lambda` in section 1–3. As `epsilon -> infinity`, MVDR -> delay-and-sum (max WNG, `w propto d`); as `epsilon -> 0`, MVDR -> max-directivity (superdirective, fragile). The constant-directivity paper's MSCD (`min w^H w s.t. c^H w = 1`, `c = d`) is exactly the `epsilon -> infinity` / max-WNG corner.
- **Equivalence to pressure matching:** MVDR is the constrained-quadratic dual of the section-1 unconstrained LS; both live on the same effort-vs-accuracy curve, just parameterized by a hard constraint (MVDR) vs a soft penalty (`lambda`).

**LCMV (linearly constrained minimum variance) — for nulls + distortionless.** `min_w w^H R w  s.t.  C^H w = g`, where `C = [d_look, d_null1, d_null2, ...]` (`[M x K]`, each column a conjugated steering vector) and `g = [1, 0, 0, ...]^T`. Closed form:
`w = R^-1 C (C^H R^-1 C)^-1 g`.
This is the clean way to place hard nulls (rejection directions) while keeping unit response on-axis — better-conditioned than encoding nulls as zeros in a soft LS target when exact rejection matters. Needs `K <= M` (can only place ~`M-1` independent nulls). Diagonal-load `R` for robustness.

**Max-directivity (spherical-loudspeaker analytic, Pasqual/Rafaely).** Directivity factor is a generalized Rayleigh quotient `Q = (d^H A d)/(d^H R d)` with Hermitian `[M x M]` `A` (numerator/look) and `R` (total-power) matrices assembled by sphere quadrature; the maximizer is the dominant generalized eigenvector of `(A, R)` (`scipy.linalg.eigh(A, R)`, take largest eigenpair). Independent steering without recomputing: in the spherical-harmonic domain `w_nm = (d_n / b_n(k r0)) * conj(Y_n^m(theta0,phi0))`. This is the analytic alternative to LS when the target is "max DI" rather than a user-drawn shape.

**Reciprocity / convention summary (the single real difference vs the mic-array literature):** receive beamforming uses the array manifold `a(Ω)` directly; transmit (radiation) uses its conjugate. Concretely, the look-direction steering vector for loudspeakers is `conj(H_f[:,n_steer])`, and the entire `H W H^H` vs `conj(H) W H^T` flip in section 1 is this same conjugation. Get it right once at the `G := H^T` definition and everything downstream (MVDR `d`, LCMV `C`, covariance `R`) inherits it.

---

## 5. Frequency-invariant / constant-directivity within the LS framework

Two primary-source-backed routes, both implementable on top of section 1:

**(a) Constant-shape target (simplest, recommended default).** As in 2(c): hold the angular target shape `g(Θ)` fixed across all `f`, only scaling by `A0(f)`. The per-frequency LS solve then naturally yields a frequency-invariant beam wherever physically achievable. Measure success with Spatial Response Variation (SRV): `SRV = (1/F) sum_f || normalize(P_f) - mean_f normalize(P_f) ||^2_W`; minimize/report it. Add post-summation gain normalization so on-axis level is flat vs `f` (the standard FIR-constant-beamwidth fix for low-frequency roll-off).

**(b) Directivity-locked per-bin constraint (Constant Directivity Loudspeaker Beamforming, arXiv:2407.01860).** Enforce a constant target generalized directivity index `tau` at every frequency bin via the quadratic equality `w^H D w = 0` with `D = A - tau R` (Hermitian indefinite; `A`,`R` are accept/reject covariances assembled by sphere quadrature over forward/side density windows). Feasibility requires `tau` between the smallest and largest generalized eigenvalues of `(A,R)`. Solve methods from the paper: MECD = `max w^H C w  s.t. w^H D w = 0, w^H w = 1` (projected-ascent, Algorithm 1, or convex SDP relaxation on `W=ww^H`); MSCD = `min w^H w  s.t. c^H w = 1` (max-WNG/effort-min corner, analytic). Numerical core: Cholesky `R = L L^H`, transform to a standard Rayleigh quotient on `x = L^H w` and take eigenpairs of `L^-1 A L^-H` (`scipy.linalg.eigh`). Per-driver band-limiting (crossover generalization) via bounded penalty weights `Lambda = diag(lambda_n)`, `0<=lambda_n<=1`, driving `|w_n| -> 0` outside driver `n`'s passband — directly useful for the woofer/mid/tweeter heterogeneous array.

For BeamSimII's "handful of heterogeneous drivers," route (a) with a WNG floor and trace-normalized `lambda(f)` is the pragmatic default; route (b) is the upgrade when the user explicitly demands a directivity number held constant vs frequency.

---

## Python / scipy implementation summary
- Core solve per `f`: `scipy.linalg.solve(conj(H_f)@W@H_f.T + lam*I, conj(H_f)@W@b_f, assume_a='pos')`. Vectorize across `f` with a Python loop (F small) or `np.linalg.solve` on a batched `[F,M,M]` stack.
- `W = np.diag(lebedev_weights)` (already shipped with the dataset).
- MVDR: `wd = scipy.linalg.solve(R+eps*I, d); w = wd/(d.conj()@wd)`.
- LCMV: `Rinv_C = scipy.linalg.solve(R+eps*I, C); w = Rinv_C @ scipy.linalg.solve(C.conj().T@Rinv_C, g)`.
- Max-directivity / constant-DI eig: `vals,vecs = scipy.linalg.eigh(A, R)`; take `vecs[:,-1]`.
- L-curve: log-sweep `lambda`, compute residual+effort, pick max-curvature corner (or WNG-floor pick).
- After weights: feed `w_m(f)` to the filter-realization stage (FIR via IFFT of the complex frequency response, or IIR/biquad fit). Keep the common phase origin intact — do not minimum-phase-ify per driver (cardinal rule).

### Sources
- (primary) Constant Directivity Loudspeaker Beamforming (Bilbao et al.), arXiv:2407.01860 — https://arxiv.org/html/2407.01860
- (primary) Optimal model-based beamforming and independent steering for spherical loudspeaker arrays (Pasqual/Rafaely line), arXiv:2310.04202 — https://arxiv.org/abs/2310.04202
- (primary) Perceptual Quality Enhancement of Sound Field Synthesis Based on Combination of Pressure and Amplitude Matching (gives exact PM closed form d=(G^H G + beta I)^-1 G^H u_des), arXiv:2307.13941 — https://arxiv.org/pdf/2307.13941
- (primary) Weighted Pressure Mode Matching for Sound Field Reproduction: Theoretical and Experimental Comparisons (Koyama group), arXiv:2303.13027 — https://arxiv.org/abs/2303.13027
- (primary) Sound Field Reproduction with Weighted Mode Matching / Infinite-Dimensional Harmonic Analysis (Ueno, Koyama, Saruwatari), arXiv:2111.11045 — https://arxiv.org/abs/2111.11045
- (primary) Digital Filter Design for Inversion Problems in Sound Reproduction (Kirkeby, Nelson; frequency-dependent regularization), AES E-Library id=12098 — https://aes.org/publications/elibrary-page/?id=12098
- (secondary) Pressure-Matching beamforming method for loudspeaker arrays with frequency-dependent selection of control points (ResearchGate) — https://www.researchgate.net/publication/282246353_Pressure-Matching_beamforming_method_for_loudspeaker_arrays_with_frequency_dependent_selection_of_control_points
- (secondary) A least squares approach to the design of frequency invariant beamformers (ResearchGate) — https://www.researchgate.net/publication/228954450_A_least_squares_approach_to_the_design_of_frequency_invariant_beamformers
- (primary) A superdirective array of phase shift sources (JASA 2012; superdirective loudspeaker effort/contrast tradeoff) — https://pubs.aip.org/asa/jasa/article-abstract/132/2/746/640322/A-superdirective-array-of-phase-shift-sources

### Decisions implied
- Define the design matrix as G = H_f^T (literally matching the GLL summation P = sum_m w_m H[m,f,n]) and implement w_f = solve(conj(H_f) @ W @ H_f.T + lam*I, conj(H_f) @ W @ b_f). Do NOT copy the H W H^H form from the microphone-array literature. Add a code comment and a unit test asserting the steered lobe points the commanded way, not its phase mirror (reuse tests/test_phase_origin.py / V-5 machinery).
- Build a single regularized-LS solver core (one [M x M] solve per frequency) and express MVDR, LCMV, max-directivity, and constant-DI as constrained variants/special cases of it, rather than separate engines. Use scipy.linalg.solve(assume_a='pos') for the SPD ridge solve and scipy.linalg.eigh(A,R) for the directivity-eigenvalue variants.
- Represent the user's target as a complex [N] field b_f on the existing Lebedev grid; provide library shape generators: steered taper cos^p(Theta), first-order pattern a+(1-a)cos(Theta) with presets (cardioid a=0.5, hypercardioid 0.25, supercardioid ~0.37), Legendre/Chebyshev order-q, and explicit null directions (b=0 or LCMV hard constraint).
- Make lambda frequency-dependent and trace-normalized: lambda(f) = beta * trace(G^H W G)/M, with selection by a WNG floor (e.g. WNG >= -10 dB, directly bounding driver excursion/robustness) as the product-default, and an L-curve (max-curvature corner) / GCV option for analysis. Expose beta and the WNG floor to the user, not raw lambda.
- Use the Lebedev quadrature weights (already shipped with the dataset) as the diagonal W in every sphere integral: the pressure-match weighting, the MVDR/LCMV covariance assembly R = sum_n a_n d d^H, the directivity Rayleigh-quotient matrices A,R, and the directivity-index / SRV metrics. Keep weight normalization consistent across all of them.
- For constant-directivity-vs-frequency, ship route (a) constant-shape target as default and route (b) the GDI=tau per-bin constraint (arXiv:2407.01860, MECD/MSCD) as an advanced mode; include per-driver passband penalty weights Lambda=diag(lambda_n) so woofer/mid/tweeter weights auto-vanish out of band (crossover generalization).
- Preserve the common phase origin end-to-end: the LS weights are complex per driver per frequency; hand them to FIR (IFFT of the complex response) or IIR/biquad fitting WITHOUT per-driver minimum-phase conversion or re-zeroing (cardinal rule), so true time-of-flight phase survives into the realized filters.

### Open questions
- Which on-disk convention does H actually store — is P = sum_m w_m H[m,f,n] computed with H as-is (so G=H^T) under the NumCalc engineering convention exp(-jwt)? Confirm by a round-trip steering test before finalizing the solver, since this fixes the conjugation in section 1.
- What WNG floor (dB) and max driver-excursion limit should bound lambda for real woofer/mid/tweeter arrays? This sets the default regularization policy and needs the user's robustness/excursion tolerance.
- Should nulls be soft (b=0 + up-weighted W) or hard (LCMV constraints)? Hard LCMV gives exact rejection but consumes degrees of freedom (max M-1 nulls); soft is more robust for a 3-5 driver array — likely want both with a UI toggle.
- For the constant-DI advanced mode, what GDI target tau and what accept/reject angular windows (f_A, f_R) match the user's 'constant directivity' intent (e.g. CTA-2034 style listening window vs full sphere)?
- Filter realization: target FIR length / latency budget vs IIR-biquad order — does the Phase-2 export need linear-phase FIR (preserves the common-origin phase trivially) or low-latency IIR (needs careful phase preservation)?

### Adversarial verification verdicts
  - [confirmed] Claim 1: For the model P_f = H_f^T w_f (H_f is [M x N]), the weighted-Tikhonov minimizer of (H^T w - b)^H W (H^T w - b) + lambda ||w||^2 is w_f = (conj(H_f) W H_f^T + lambda I_M)^-1 conj(H_f) W b_f, and this is a true local minimum.
  - [confirmed] Claim 2: The 'tidy' form w = (H W H^H + lambda I)^-1 H W b is NOT the solution to the BeamSimII-summed model; it is the exact solution to the conjugated model P = H^H w. The two are not simple conjugates unless b is also conjugated (user-form == conj(correct) is FALSE; user-form == model-B solution is TRUE; user-form == conj(correct-with-conj-b) is TRUE). Wrong choice applies conjugated per-driver phase and mirror-steers the beam.
  - [confirmed] Claim 3: MVDR transmit form w = R^-1 d / (d^H R^-1 d) from min w^H R w s.t. d^H w = 1, where R is a synthetic quadrature-assembled [M x M] covariance R = sum_n a_n d(n) d(n)^H (not measured noise), d = conj(H_f[:,n_steer]); diagonal loading R -> R + eps I bounds WNG and interpolates between max-directivity (eps->0) and delay-and-sum (eps->inf). -> CORRECTION: Minor nuance for the plan: the eps->WNG mapping is not closed-form. The relationship between the diagonal-loading factor and a target white-noise-gain level is not simple; in the array literature it generally requires an iterative search or recent analytic-bound methods. So 'diagonal loading bounds WNG' is qualitatively right but should not be implemented as a direct one-line eps<->WNG formula.
  - [confirmed] Claim 4: LCMV closed form w = R^-1 C (C^H R^-1 C)^-1 g with C = [d_look, d_null...] ([M x K]), g = [1,0,...]^T, requiring K <= M (at most M-1 independent nulls). -> CORRECTION: Dimension nuance, not an error: in the standard array literature C is [N_elements x K]; in the BeamSimII transmit mapping the weight vector has length M (drivers), so C is [M x K] and the bound is K <= M, i.e. one distortionless look-direction plus at most M-1 independent nulls. The finder's [M x K] and 'at most M-1 nulls' are the correct transmit-side adaptation.
  - [confirmed] Claim 5: Constant directivity per frequency bin (arXiv:2407.01860) is enforced by the quadratic equality w^H D w = 0 with D = A - tau R (A,R Hermitian accept/reject covariances), feasible only for tau between the smallest and largest generalized eigenvalues of (A,R); solved via Cholesky R=LL^H and eigendecomposition of L^-1 A L^-H. -> CORRECTION: Two precision points: (1) R (the REJECT matrix), not A, is the one Cholesky-factored as R=LL^H — the finder's text is consistent with this but the plan should label it explicitly to avoid swapping A<->R. (2) The covariances are assembled with accept/reject probability-density weighting functions f_A, f_R over directions, which on a Lebedev grid becomes a quadrature-weighted sum a_n d(n)d(n)^H; treat the Lebedev weights times the region indicator as the density. The paper names two designs (MECD = max-efficiency, MSCD = max-sensitivity) sharing this constraint; tau interval is open (e1 < tau < eN) except at the extrema.
  - [partially-correct] Claim 6: Within the LS framework, frequency-invariance is obtained by holding the angular target shape g(Theta) fixed across all f (only a per-f scalar gain varies), b_f(n) = A0(f) g(Theta_n); success measured by Spatial Response Variation (SRV). -> CORRECTION: Imprecision: the literature explicitly distinguishes the conventional 'pre-specified fixed desired beampattern' approach (which is exactly b_f(n)=A0(f)g(Theta_n)) from SRV-BASED design, whose whole selling point is NOT pre-specifying a fixed target — SRV-based methods gain extra degrees of freedom precisely by leaving the template free (it becomes a function of w). So pairing 'fixed target b_f' with 'SRV' is fine if SRV is used only as a post-hoc diagnostic metric on the fixed-target design; it is incorrect to describe that fixed-target LS as 'SRV-based design.' For the plan: use a fixed g(Theta) target as the simple frequency-invariance recipe, and use SRV (or weighted-SRV) only to score/report the result — do not conflate the two as the same method.


---

## TOPIC: Acoustic Contrast Control (ACC), Pressure Matching (PM), and Planarity Control for sound zones — as alternative/complementary beamformer families for BeamSimII Phase 2, and whether a separate ACC mode is needed alongside the Luo directivity beamformer.

### Report
# ACC / PM / Planarity Control vs the Luo Directivity Beamformer — for BeamSimII Phase 2

## 0. Setup and notation (mapped onto BeamSimII's H tensor)

Sound-zone control places **control points** in space (not directions on a sphere). Let there be a "bright" zone B (where you want sound) and a "dark" zone D (where you want silence), with control points x_b (b=1..N_B) and x_d (d=1..N_D). The plant/transfer matrices at one frequency f are:

- G_b ∈ C^[N_B × M]  — pressure at each bright control point from each of the M loudspeakers (column m = response of source m). G_b[b,m] = transfer function source m → point x_b.
- G_d ∈ C^[N_D × M]  — same for dark control points.
- q ∈ C^[M]  — complex loudspeaker weights (the thing you solve for; identical role to BeamSimII's w_m(f)).

BeamSimII's H[m, f, direction] IS exactly this plant, but evaluated at **far-field directions on a sphere** rather than near-field control points. So for BeamSimII a "control point" = a sphere direction n, and G[n,m] = H[m, f, n]. Everything below transfers by replacing "zone of control points" with "angular region of sphere directions" and "sum over control points" with "Lebedev-quadrature-weighted sum over directions." This is the single most important mapping for the implementation plan.

CONVENTION NOTE (load-bearing): All matrices use the **Hermitian** (conjugate-transpose) form ^H, because pressures are complex. The "spatial correlation matrix" R = G^H G is **M × M Hermitian positive-semidefinite** (it lives in *loudspeaker* space, not control-point space). BeamSimII must build R by accumulating outer products of the **conjugated** H columns with the quadrature weight: R[m,m'] = Σ_n a_n · conj(H[m,f,n]) · H[m',f,n]. If you accidentally use the un-conjugated transpose you silently solve a different (wrong) problem. This is the same Hermitian-transpose subtlety flagged for Luo's covariance build, and it interacts with the engineering exp(−jωt) time convention NumCalc uses: as long as H is stored in NumCalc's native convention and you conjugate consistently in R, the eigenproblem is convention-agnostic (the Rayleigh quotient q^H R q is real and the optimal q's absolute phase is arbitrary). The *only* place convention bites is if you later impose a phase-matched target (PM, see §2) — then the target plane wave must be written in the SAME engineering convention exp(+jk·n·x) as H.

---

## 1. Acoustic Contrast Control (ACC) — Choi & Kim 2002

### Objective
Maximize the ratio of **acoustic potential energy** (∝ |p|²) spatially averaged over the bright zone to that over the dark zone. Spatially-averaged potential energy in a zone for weights q is q^H R q with the zone's spatial correlation matrix R:

- R_b = G_b^H G_b   (M × M, bright-zone spatial correlation matrix; with quadrature/spatial weighting W_b: R_b = G_b^H W_b G_b, W_b diagonal of zone-averaging weights — for BeamSimII W = diag(Lebedev weights a_n over the bright angular cap))
- R_d = G_d^H G_d   (dark-zone spatial correlation matrix; analogously G_d^H W_d G_d)

The **acoustic contrast** is the Rayleigh quotient:

    C(q) = (q^H R_b q) / (q^H R_d q)

### Solution (generalized eigenproblem)
Maximize q^H R_b q subject to q^H R_d q = const. Lagrangian L(q,λ) = q^H R_b q − λ(q^H R_d q − const); ∂L/∂q^H = 0 gives:

    R_b q = λ R_d q         (generalized eigenproblem, matrix pencil (R_b, R_d))

equivalently, when R_d is invertible:

    R_d^{-1} R_b q = λ q

The optimal q is the **principal (largest-λ) eigenvector**, and λ_max = the achievable contrast. This is VERIFIED against the standard derivation (Choi & Kim, JASA 111(4):1695–1700, 2002, "Generation of an acoustically bright zone with an illuminated region using multiple sources"; and the ICSV18 Francombe/Jackson comparison restating the Lagrangian → largest-eigenvector result). The eigenvector is determined only up to scale and absolute phase — ACC controls **energy only, not phase/waveform** in the bright zone (this is its defining limitation).

### Relation to Luo's A/R covariance ratio — THEY ARE THE SAME MACHINERY
BeamSimII's docs already specify Luo (EUSIPCO 2024, arXiv:2407.01860): build covariance **A** over the "accept"/target angular region and **R** over the "reject"/whole-sphere region by integrating H over the sphere, then maximize the ratio → generalized eigenproblem. That is *literally* ACC with the relabeling:

    A ↔ R_b  (accept = bright),    R ↔ R_d  (reject = dark),
    A q = λ R q   ⇔   R_b q = λ R_d q.

Both are the **identical Rayleigh-quotient / generalized-eigenvalue maximization of an "energy here / energy there" ratio**, solved by `scipy.linalg.eigh(A, R)` (Hermitian generalized eigensolver) and taking the eigenvector of the largest eigenvalue. The only differences are cosmetic: (a) ACC literature uses near-field *control points*, Luo uses far-field *sphere directions*; (b) Luo's "reject" region is often the whole sphere (giving a directivity-index-like denominator), whereas classic ACC's dark zone is a disjoint silent region. Mathematically nothing changes. CONCLUSION for Q1: ACC ≡ Luo's MECD/MaxDI eigenproblem viewed through the sound-zone lens.

---

## 2. Pressure Matching (PM) — and how it differs from ACC

PM does NOT maximize an energy ratio; it minimizes the **complex error to a fully specified target field** (magnitude AND phase) at the control points, in a least-squares sense. Define a target pressure vector p_des stacked over bright (= a desired plane wave from the steering direction, complex-valued) and dark (= 0) control points; stack G = [G_b; G_d] accordingly. The Tikhonov-regularized cost and closed-form solution:

    J(q) = || G q − p_des ||²  +  β ||q||²
    q_opt = (G^H G + β I)^{-1} G^H p_des

(Often weighted: J = ||W^{1/2}(Gq − p_des)||² + β||q||², with W up-weighting bright vs dark, → q = (G^H W G + βI)^{-1} G^H W p_des.) This is VERIFIED as the canonical regularized normal-equation / Tikhonov solution (Coleman/Jackson/Olik JASA 135(4):1929–1940, 2014; and the standard PSZ formulation).

### Key differences ACC vs PM
- **What's controlled:** ACC controls only the energy ratio → bright-zone field can be any shape (often a messy standing wave), only its level vs the dark zone is optimized. PM controls the **full complex field**, so the bright zone reproduces a chosen wavefront (e.g. a clean plane wave with correct phase) — but at the cost of lower contrast.
- **Solution type:** ACC = eigenproblem (homogeneous, scale/phase-free). PM = linear solve (inhomogeneous, target-driven, gives an absolute amplitude/phase).
- **Performance trade:** ACC achieves the *highest* dark-zone cancellation/contrast; PM gives a *better-behaved bright zone* but smaller contrast. Hybrid "ACC-PM" methods exist that blend them.
- **Robustness:** for PM, regularization β keeps helping robustness even after conditioning is fixed; for ACC, once the matrix inversion (of R_d) is adequately conditioned, extra regularization does little for robustness (Coleman et al. 2014). This is a real, citable asymmetry.

---

## 3. Planarity Control (PC) — the genuinely-different third method

PC (Coleman, Jackson, Olik, Pedersen — AES 52nd Conf. 2013 "Optimizing the planarity of sound zones"; JASA 136:1725–1735, 2014 "Personal audio with a planar bright zone"; metric from Jackson et al., POMA 19:055056, 2013, "Sound field planarity characterized by superdirective beamforming") fixes ACC's bright-zone ugliness *without* fully constraining phase like PM. Recipe:

1. Keep ACC's **dark-zone cancellation term unchanged** (R_d, identical to ACC → same near-perfect cancellation).
2. Replace the bright-zone energy q^H R_b q with energy measured **in a plane-wave-decomposition domain**, spatially filtered to favor the target direction. Introduce a plane-wave decomposition / superdirective-beamforming matrix that maps bright control-point pressures to plane-wave components arriving from a discrete set of directions, and a **diagonal direction-weighting matrix Γ (a.k.a. C)** whose nonzero diagonal entries select/allow the desired arrival direction(s) (e.g. a raised-cosine window around the steering azimuth; in their experiments Γ allowed 120°–240°).
3. The bright-zone term becomes q^H (G_b^H Y^H Γ Y G_b) q where Y is the plane-wave decomposition (steering) matrix and Γ the diagonal direction weighting. The optimization is again a **generalized eigenproblem** of the same form, principal eigenvector:

    (G_b^H Y^H Γ Y G_b) q = λ R_d q   →   q = principal eigenvector.

So PC is **structurally ACC with a direction-selective spatial filter inserted into the bright-zone correlation matrix.** Result: an ACC-like dark zone PLUS a bright zone that is a near-single plane wave steerable to a chosen direction (high "planarity"), at contrast between ACC (best) and PM.

### Planarity metric (the figure of merit)
Decompose the reproduced bright-zone field into plane-wave components; planarity =

    P = energy of the single largest plane-wave component / total energy summed over all plane-wave components

(per Jackson et al. POMA 2013, using superdirective beamforming on the bright-zone microphones; with û_i the unit direction of component i and u the resultant vector, the metric is the proportion of total beamformed energy concentrated in the dominant direction). P → 1 means a perfect single plane wave (ideal "beam"); P low means diffuse/standing-wave field. **This is essentially a far-field directivity concentration metric** — which is exactly what BeamSimII's directivity/beam-shape frame already optimizes directly on the sphere.

---

## 4. When ACC/sound-zones is the right frame vs the directivity/beam-shape frame — and whether BeamSimII needs a separate ACC mode

### The frames
- **Sound-zone frame (ACC/PM/PC):** control points in a bounded near-field/room region; you care about absolute pressure at *places* (a listener's head vs a silent seat). Inherently a near-field, room-aware, multi-point problem; transfer functions G come from measured/simulated room impulse responses.
- **Directivity/beam-shape frame (BeamSimII / Luo / CBT):** you care about the **far-field angular pattern** P(f, n) = Σ_m w_m(f) H[m,f,n] over the sphere — beamwidth, cardioid order, constant-directivity-vs-frequency, steering. Free-field, single observation radius, angular regions.

### Verdict for BeamSimII (Q3)
BeamSimII operates in the **free-field directivity frame** with H on a Lebedev sphere referenced to one phase origin. In this frame:

- **ACC is NOT a separate algorithm you need to add — it is mathematically identical to the Luo "accept/reject" generalized eigenproblem you already plan to build** (§1). "Maximize bright-region energy / dark-region energy on the sphere" is the same `eigh(A, R)` solve. If a user asks for "maximum front-to-back ratio" or "null this rear angular sector," that IS ACC, and it falls out of the existing covariance-ratio code by choosing the accept region = front cap and the reject region = rear sector. So you get ACC "for free" as a *region-selection* feature of the directivity beamformer, not a new mode.
- **PM is genuinely different and worth offering** as a complementary *forward-design* mode: it lets the user specify a complete complex far-field target (e.g. an ideal cardioid pattern with a specific on-axis phase/level, or a constant-directivity template) and solve the regularized linear system q = (H^H W H + βI)^{-1} H^H p_des per frequency. PM is the natural way to hit a **prescribed pattern shape** (cardioid-order, constant-DI template) rather than just "max energy ratio." It also preserves the absolute amplitude/phase, which matters for BeamSimII's cardinal rule (common phase origin) — PM never re-zeros a driver; it solves for w_m honoring H's built-in time-of-flight phase.
- **Planarity control's *machinery* is not needed** (its whole point — converting a diffuse near-field standing-wave bright zone into a plane wave — is moot in the far field, where every observation already sees a single propagating direction). BUT planarity's *spirit* — the diagonal direction-weighting matrix Γ selecting allowed arrival directions — maps onto a useful BeamSimII feature: **angular weighting masks** on the sphere (raised-cosine windows around the steer direction, sidelobe-region penalties). That is, BeamSimII should support a per-direction weight vector w_dir(n) in the covariance integrals, which is the far-field analog of Γ.

### Recommended BeamSimII Phase-2 design (the practical answer)
Implement ONE shared engine — weighted complex covariance matrices over the Lebedev sphere — exposing two solver modes:
1. **Max-ratio (eigen) mode** = Luo/ACC unified: build A (accept region), R (reject region) with per-direction angular masks (the Γ analog) and Lebedev weights; solve generalized eigenproblem `scipy.linalg.eigh(A, R + εI)`, take principal eigenvector. Covers "max DI," "max front/back ratio," "null this sector," classic ACC — all by region/mask choice.
2. **Pattern-match (linear) mode** = PM: user specifies a complex far-field target template p_des(f,n) (cardioid, constant-DI, prescribed beamwidth) and BeamSimII solves the Tikhonov LS `(H^H W H + βI)^{-1} H^H W p_des`. This is the right tool for "shape the beam to *this* pattern with controlled phase."

No standalone "ACC mode" and no "planarity mode" are warranted; both reduce to mode 1 with appropriate region/mask choices, and mode 2 (PM) supplies what the eigen-mode cannot (explicit shaped, phase-controlled targets).

---

## 5. Regularization in ACC/PM (effort + robustness) — directly reusable

Both modes are ill-conditioned inverse problems (heterogeneous woofer/mid/tweeter arrays → near-singular R/H^H H, especially at low ka where drivers are nearly omnidirectional and highly correlated). Regularization does two jobs (Coleman et al. 2014; Elliott/Cheer/Choi/Kim, IEEE TASLP 20(7):2123–2133, 2012, "Robustness and regularization of personal audio systems"):

- **Effort constraint / array effort:** array effort ≡ Σ_m |q_m|² = ||q||². Adding β||q||² (equivalently the +βI in the inverse) bounds the driving signals → prevents huge, fragile loudspeaker weights ("contrast oversaturation," runaway gains). The constraint is frequently *active* at several frequencies; modern work (e.g. the 2025 array-effort-constrained ACC) auto-adapts β to bound ||q|| rather than fixing β.
- **Robustness (white-noise-gain / conditioning):** transfer-function errors, driver gain/phase mismatch, and position errors act like spatially-white perturbations; their effect scales with ||q||². Larger β = smaller ||q|| = higher robustness (better white-noise gain), trading reproduction accuracy. β also conditions the matrix inversion (Tikhonov: replaces singular values σ_i by σ_i/(σ_i²+β), capping amplification of small σ_i).

For the **eigen-mode (ACC/Luo)**, regularize the *denominator* matrix: solve `eigh(A, R + βI)` (β·I or β·trace(R)/M·I). NOTE the asymmetry: for ACC, once R is conditioned, extra β does little for robustness; for PM, β keeps improving robustness. Practical recipe: pick β by an effort/robustness target (e.g. constrain white-noise gain or ||q|| ≤ q_max, or L-curve), per frequency, β scaled to the matrix trace so it's frequency-adaptive. Expose β (or an effort cap) as the single user-facing robustness knob in both modes.

### Numerical recipe summary (per frequency f)
1. Build H_f = H[:, f, :] (M × N). Apply angular masks/weights: define diagonal W_accept (Lebedev a_n inside accept region, else 0), W_reject similarly.
2. A = H_f conj() · W_accept · H_f^T arranged Hermitian → A[m,m'] = Σ_n a_n^accept conj(H[m,f,n]) H[m',f,n]; R likewise. (M×M Hermitian PSD.)
3. Eigen-mode: `λ, V = scipy.linalg.eigh(A, R + βI)`; w = V[:, -1] (largest λ). Normalize w (e.g. on-axis gain = target).
4. PM-mode: w = `scipy.linalg.solve(H_f.conj() @ W @ H_f.T + βI, H_f.conj() @ W @ p_des)` (M×M solve). Keep H in NumCalc engineering convention; write p_des plane-wave target as exp(+jk n·x) accordingly.
5. Both: w is per-frequency complex; realize as FIR (frequency-sampling/IFFT of w_m(f) across the frequency grid) or fit IIR/biquads. The phase of w_m(f) carries true inter-driver delay — DO NOT min-phase-ify per driver (cardinal rule).

### Sources
- (primary) Choi & Kim, Generation of an acoustically bright zone with an illuminated region using multiple sources, JASA 111(4):1695-1700 (2002) [original ACC paper] — https://pubs.aip.org/asa/jasa/article-abstract/111/4/1695/546514
- (primary) Coleman, Jackson, Olik, Møller, Olsen, Pedersen — Acoustic contrast, planarity and robustness of sound zone methods using a circular loudspeaker array, JASA 135(4):1929-1940 (2014) — https://pubs.aip.org/asa/jasa/article/135/4/1929/968046/Acoustic-contrast-planarity-and-robustness-of
- (primary) Coleman, Jackson, Olik, Pedersen — Personal audio with a planar bright zone, JASA 136(4):1725-1735 (2014) — https://pubs.aip.org/asa/jasa/article-abstract/136/4/1725
- (primary) Jackson, Jacobsen, Coleman, Pedersen — Sound field planarity characterized by superdirective beamforming, Proc. Mtgs. Acoust. 19:055056 (2013) [planarity metric] — https://pubs.aip.org/asa/poma/article/19/1/055056/988437/Sound-field-planarity-characterized-by
- (primary) Coleman, Jackson, Olik, Pedersen — Optimizing the Planarity of Sound Zones, AES 52nd Intl. Conf. (2013) preprint — https://personalpages.surrey.ac.uk/p.jackson/pub/aes13b/ColemanEtAl_AES13_preprint.pdf
- (primary) Coleman, Jackson et al. — Stereophonic Personal Audio Reproduction Using Planarity Control Optimization, ICSV14 preprint (plane-wave decomposition + diagonal weighting matrix Gamma) — https://personalpages.surrey.ac.uk/p.jackson/pub/icsv14/ColemanJacksonEtAl_ICSV14a_preprint.pdf
- (primary) Elliott, Cheer, Choi, Kim — Robustness and regularization of personal audio systems, IEEE/ACM TASLP 20(7):2123-2133 (2012) — https://ieeexplore.ieee.org/document/6privately
- (primary) Luo (Amazon) — Constant Directivity Loudspeaker Beamforming, EUSIPCO 2024 pp.246-250; arXiv:2407.01860 (accept/reject covariance ratio = same eigenproblem as ACC) — https://arxiv.org/abs/2407.01860
- (secondary) Patent WO2014108365A1 — A sound-field control method using a planarity measure (Coleman/Jackson, plane-wave weighting matrix) — https://patents.google.com/patent/WO2014108365A1/en
- (primary) Constrained optimization of acoustic contrast for personal sound zones based on array effort control, Applied Acoustics (2025) — adaptive effort/regularization — https://www.sciencedirect.com/science/article/abs/pii/S0003682X25005304
- (secondary) A Review of Sound Field Control, Applied Sciences 12(14):7319 (2022) — survey relating ACC/PM/PC/effort regularization — https://www.mdpi.com/2076-3417/12/14/7319

### Decisions implied
- Build ONE shared Phase-2 engine: weighted complex covariance matrices over the Lebedev sphere (R[m,m']=sum_n a_n conj(H[m,f,n]) H[m',f,n]) with per-direction angular masks; expose two solver modes rather than separate ACC/Luo/planarity modes.
- Mode 1 (max-ratio/eigen) unifies ACC and Luo: choose an 'accept' angular region A and a 'reject' region R (incl. nulling specific rear sectors = front/back-ratio / classic ACC), then solve scipy.linalg.eigh(A_mat, R_mat + beta*I) and take the principal eigenvector as w_m(f).
- Mode 2 (pattern-match/PM) is a genuinely additive forward-design tool: let the user specify a complete complex far-field target template p_des(f,n) (cardioid order, constant-DI, prescribed beamwidth) and solve w=(H^H W H + beta I)^{-1} H^H W p_des per frequency; write p_des in the NumCalc engineering convention exp(+jk n.x).
- Do NOT implement a standalone ACC mode or a planarity-control mode: ACC collapses into Mode 1 by region choice, and planarity's bright-zone-to-plane-wave conversion is moot in the far field. Instead port only planarity's diagonal direction-weighting matrix Gamma into Mode 1 as per-direction angular weighting masks (e.g. raised-cosine windows around the steer direction, sidelobe-region penalties).
- Expose a single robustness knob (regularization beta or an equivalent array-effort/white-noise-gain cap) shared by both modes; regularize the denominator in the eigen-mode (eigh(A, R + beta I)) and the normal matrix in PM ((... + beta I)^{-1}); make beta frequency-adaptive by scaling to the matrix trace, and consider an effort-constrained auto-beta to avoid contrast oversaturation / runaway weights at low ka.
- Enforce the Hermitian-conjugate convention everywhere covariance/PM matrices are assembled, with a unit test asserting R is Hermitian PSD; never min-phase-ify or re-zero per-driver weights w_m(f) since their phase carries true inter-driver time-of-flight (cardinal rule), which both eigen-mode and PM honor automatically because H carries the common phase origin.

### Open questions
- Exact form of the planarity-control plane-wave decomposition matrix Y (superdirective beamformer weights and the discrete arrival-direction grid) and how Gamma's raised-cosine window is parameterized — the precise equations were in the Coleman 2014 JASA / ICSV14 PDFs which did not text-extract here; confirm before porting Gamma as an angular mask if exact reproduction is wanted (vs the far-field-mask approximation proposed).
- Whether BeamSimII users will ever pose a true near-field/in-room sound-zone problem (e.g. silent-seat in a car/room) rather than free-field directivity — if so, a genuine control-point (non-sphere) PM/ACC path with measured/simulated RIRs would be a separate, larger feature outside the current H-on-sphere contract.
- Optimal automatic regularization strategy (fixed beta vs L-curve vs white-noise-gain target vs 2025 array-effort-constrained adaptive beta) for the heterogeneous few-driver woofer/mid/tweeter case at low ka where R is most ill-conditioned.
- How to best realize the per-frequency complex w_m(f) from either mode as stable FIR or IIR/biquad filters while preserving inter-driver phase (linear-phase FIR via frequency sampling vs min-phase magnitude + separate bulk delay) — interacts with the cardinal common-phase-origin rule and warrants its own DSP research pass.

### Adversarial verification verdicts
  - [confirmed] The ACC objective is the Rayleigh quotient C(q)=(q^H R_b q)/(q^H R_d q) with R_b=G_b^H G_b and R_d=G_d^H G_d (M x M Hermitian PSD in loudspeaker space); its maximizer is the principal eigenvector of R_b q = lambda R_d q, equivalently the top eigenvector of R_d^{-1} R_b.
  - [confirmed] ACC's (R_b,R_d) generalized eigenproblem and Luo's accept/reject covariance-ratio eigenproblem (A q = lambda R q) are the identical Rayleigh-quotient machinery, differing only in labeling and near-field control points vs far-field sphere directions, so BeamSimII gets ACC 'for free' as a region-selection feature of the existing covariance-ratio beamformer. -> CORRECTION: The MATH is identical and confirmed; the engineering claim 'for free' is overstated. Two non-cosmetic differences carry real implementation surface: (1) classic ACC's dark zone is a disjoint spatial region whereas BeamSimII's is an angular sector of sphere directions; (2) on far-field directions R_d (the reject/whole-sphere matrix) is frequently rank-deficient / ill-conditioned, which is precisely why regularization (claim 6) is needed before R_d^{-1} R_b is formed. So ACC is a region-selection + R_d-conditioning feature of the existing covariance-ratio beamformer, not literally free.
  - [confirmed] Pressure matching minimizes J=||Gq - p_des||^2 + beta||q||^2 with closed form q=(G^H G + beta I)^{-1} G^H p_des; unlike ACC it controls the FULL complex (magnitude AND phase) field, yielding lower contrast but a clean prescribed wavefront, and its target plane wave must use the same engineering exp(+jk r) convention as NumCalc's H.
  - [confirmed] Planarity control keeps ACC's dark-zone matrix R_d unchanged and replaces the bright-zone energy term with q^H (G_b^H Y^H Gamma Y G_b) q, where Y is a plane-wave-decomposition/superdirective steering matrix and Gamma is a DIAGONAL direction-weighting matrix; it remains a generalized eigenproblem solved by the principal eigenvector. -> CORRECTION: Precise per the paper but one caveat: the dark-zone matrix in the eigenproblem is (G_B^H G_B + lambda I), i.e. R_d plus the effort-regularization +lambda I, not the bare R_d. So 'R_d unchanged' holds only modulo the +lambda I regularization that is already present in the ACC eigenproblem too. Gamma's diagonal entries are weights in [0,1] (e.g. raised-cosine around the target direction), not strictly a 0/1 selector.
  - [confirmed] Spatial correlation matrices are built with the Hermitian (conjugate) transpose; for BeamSimII R[m,m']=sum_n a_n conj(H[m,f,n]) H[m',f,n] using Lebedev weights a_n, and using the plain (non-conjugated) transpose silently solves a different, wrong problem.
  - [partially-correct] Regularization adds beta||q||^2 to bound array effort and improve robustness; for ACC, once R_d's inversion is conditioned extra beta does little for robustness, whereas for PM beta keeps improving robustness (asymmetry from Coleman et al. 2014). -> CORRECTION: Split the claim. (a) 'beta bounds effort and improves robustness/WNG' is CONFIRMED (Elliott et al. 2012; Tikhonov). (b) The specific asymmetry -- ACC robustness saturating after conditioning while PM keeps benefiting from beta, attributed to Coleman 2014 -- is UNVERIFIED and likely overstated/wrong: Elliott et al. 2012 shows ACC robustness is itself governed by effort regularization, so increasing beta continues to trade contrast for robustness in ACC too. Treat the asymmetry as not established; if needed, verify against the full Coleman et al. JASA 135(4):1929-1940 (2014) text (paywalled; I could only confirm its abstract-level focus on contrast/planarity/robustness, not a stated ACC-vs-PM regularization asymmetry).
  EXTRA: 1) Transpose-vs-Hermitian convention split (load-bearing, easy to trip on): the open VAST primary source (Lee/Nielsen/Christensen, arXiv:1911.10016) and the Coleman ICSV21 PM/PC equations sometimes appear with plain transpose (^T) because those are REAL time-domain FIR formulations (R_C = sum y_m y_m^T). BeamSimII operates per-frequency on COMPLEX H, so every such ^T must become Hermitian ^H. The finder got this right for BeamSimII but an implementer copying equations verbatim from a time-domain paper would introduce the exact silent bug claim 5 warns about. 2) Missing canonical robustness reference: Elliott, Cheer, Choi & Kim, IEEE TASLP 20(7):2123-2133 (2012), 'Robustness and Regularization of Personal Audio Systems,' is THE primary source for ACC effort/regularization/WNG robustness and should anchor claim 6 in the plan (the finder cited only Coleman 2014). 3) Planarity-control eigenproblem dark matrix: it is (G_B^H G_B + lambda I), i.e. dark Gram PLUS effort regularization -- the +lambda I is not optional cosmetic; it is what makes the inverse well-posed. Same +lambda I already sits in the plain ACC eigenproblem on the far-field sphere where R_d is often rank-deficient. 4) VAST/variable-span framework (joint diagonalization U^T R_b U = Lambda, U^T R_d U = I) is the clean way to expose an ACC<->PM trade-off knob (rank-1 top eigenvector = pure ACC; full span = PM); worth flagging to the user as the unifying generalization of claims 1-3, citing arXiv:1911.10016. 5) Coleman et al. JASA 135(4):1929-1940 (2014) full text is paywalled (AIP) and the Surrey S3 open print-version link uses short-lived signed URLs that expired during verification; the abstract/secondary sources confirm its contrast/planarity/robustness comparison but I could not read its exact regularization-asymmetry wording from primary text.


---

## TOPIC: Realizing optimal per-frequency complex beamforming weights w_m(f) as causal per-driver FIR/IIR filters for the BeamSimII Phase-2 filter designer

### Report
# Turning per-bin complex weights w_m(f) into realizable causal per-driver filters

## 0. The forward model and what "the filter" must reproduce

The Phase-2 forward (GLL complex-summation) model is
  P(f, dir) = sum_m w_m(f) * H[m, f, dir]
This is exactly the filter-and-sum beamformer response that appears throughout the array
literature, e.g. B(w, phi, theta) = sum_n W_n(w) g_n(w, phi, theta) (Mabande/Kellermann
filter-and-sum; identical structure). The optimizer (your MECD/MSCD-style per-bin solve, or
a pressure-matching / LS solve) produces one complex number w_m(f) per driver m per frequency
bin f. "Realizing a filter" means: build a causal LTI filter for driver m whose discrete-time
frequency response W_m(e^{jw}) equals (a delayed copy of) the optimizer's w_m(f), so that when
the real audio is run through it the array produces P(f,dir).

KEY FRAMING (decides everything below): w_m(f) is a *complex* target with arbitrary magnitude
AND arbitrary phase. The inter-driver *relative* phase at each frequency IS the steering. So
filter realization is an arbitrary-magnitude-AND-phase design problem, not a magnitude/EQ
problem, and the relative phase between drivers must be preserved to within a tight tolerance.

---

## 1. Causality: optimal per-bin weights are non-causal / two-sided. CONFIRMED.

An arbitrary complex frequency response w_m(f) is in general neither minimum-phase nor causal.
Reason: its inverse DTFT h_m[n] = IDFT{w_m(f)} is in general a *two-sided* sequence with energy
at negative time indices (n<0). A causal FIR filter must have h[n]=0 for n<0, which a raw IDFT
of an arbitrary complex spectrum does not satisfy. The canonical illustration (Boschen,
dsprelated #1760): a brick-wall response IDFTs to a sinc that extends from n = -inf to +inf;
any non-trivial phase target similarly spreads energy both sides of n=0.

Fix = BULK MODELING DELAY. Add a common linear-phase term exp(-j*2*pi*f*tau) to the target
(equivalently fftshift the impulse response and start the time axis at the first tap). This
slides the two-sided response forward in time so essentially all its energy sits at n>=0, making
the truncated/windowed filter causal. The cost is tau seconds of latency, identical for the
realization regardless of which driver. The center tap then represents "zero relative delay,"
taps before it are negative relative delays, taps after are positive — all referenced to the
shared +tau bulk delay. (This is the standard digital-beamformer trick: the (M-1)/2 group delay
of a linear-phase FIR makes the otherwise non-causal two-sided response realizable.)

### 1a. CARDINAL-RULE CONSTRAINT (this is the load-bearing realization rule)
The bulk modeling delay tau MUST be the SAME for every driver. A single shared delay multiplies
every w_m(f) by the same exp(-j*2*pi*f*tau); since it is common to all m it factors straight out
of the sum P(f,dir) = exp(-j2pi f tau) * sum_m w_m(f) H[m,f,dir], so it only adds global latency
and does NOT change the beam. By contrast, applying a *per-driver* delay, or per-driver
minimum-phase conversion, or per-driver group-delay re-zeroing, changes the inter-driver relative
phase and silently mis-steers the beam. This is precisely the project's cardinal rule
("never minimum-phase-ify or re-zero a driver independently") expressed in the filter-realization
domain. => Phase-faithful default = linear-phase FIR with ONE shared modeling delay.

### 1b. Typical delay / tap budgets
- Modeling delay tau ~ (Ntaps-1)/2 samples for a symmetric truncation window (the impulse-response
  center). Boschen's worked example: 165 taps -> 82-sample bulk delay; an audio crossover example
  cited 1024 samples ~ 21.3 ms at 48 kHz.
- Tap count vs accuracy is a continuous trade: errors "can be 100s of dB lower if desired" with more
  taps; more taps and more delay buy a closer match. Boschen verified decay to -120 dB before
  truncating to bound time-aliasing.
- Practical loudspeaker-FIR budgets: low frequencies (long wavelength, steep phase slopes vs f) drive
  the tap count. For full-band woofer-to-tweeter beamforming at 48 kHz expect O(2^12 .. 2^16) taps if
  you insist on linear phase down to ~50-100 Hz; mid/high-only beams need far fewer (hundreds to low
  thousands). Budget per-driver Ntaps to cover the *slowest-varying-vs-frequency* (lowest-f) phase you
  must reproduce. The GLL format itself allows up to 64k-sample impulse responses per source, a useful
  upper-bound sanity check on what production tools consider reasonable.

---

## 2. FIR design from a complex frequency response

### 2a. Frequency-sampling / IFFT+window method (RECOMMENDED CORE for arbitrary complex phase)
This is the only route that natively handles arbitrary magnitude AND phase, and it is pure
numpy/scipy. Recipe (Boschen #1760, validated; mirrors the Eclipse Audio loudspeaker-FIR guide):

  1. Build the dense target spectrum. Interpolate w_m(f) from the (possibly log-spaced, sparse)
     simulation frequency grid onto a *dense uniform linear* grid of size Nfft (oversample heavily,
     e.g. Nfft = 2^15..2^17). Interpolate magnitude and *unwrapped* phase separately (or real/imag),
     NOT wrapped phase. PITFALL: the H tensor / weights live on a log or sparse frequency grid;
     interpolating across log frequency onto the linear FFT grid is the #1 error source — unwrap
     phase first, interpolate in (log|w|, unwrapped angle) or (Re, Im), then re-exponentiate.
  2. Enforce Hermitian symmetry so taps come out real (see Section 6): set the negative-frequency
     half to conj-mirror of the positive half, W(-f)=conj(W(f)), real DC and Nyquist.
  3. Add the shared modeling delay: multiply by exp(-j*2*pi*f*tau) (sign per Section 6), OR apply
     fftshift after the IFFT.
  4. h = ifft(W_dense)  (np.fft.ifft); then np.fft.fftshift to center the two-sided response.
  5. Truncate to Ntaps around the center: coeff = h[c-Ntaps//2 : c+Ntaps//2+1].
  6. Window: coeff *= scipy.signal.windows.kaiser(Ntaps, beta=8) (Boschen used Kaiser beta=8;
     Blackman-Harris / Hann also standard). Windowing trades passband/stopband ripple (Gibbs) for
     transition-band width.
  7. VERIFY: w,h = scipy.signal.freqz(coeff, worN=Nfft, whole=True); remove the known linear phase
     ((Ntaps-1)/2 samples) and overlay against the target on both magnitude and phase. Confirm the
     pre-truncation impulse decayed below ~-120 dB so time-aliasing is negligible.
  PITFALLS summary: (i) log->linear interpolation; (ii) wrapped-phase interpolation; (iii) too-small
  Nfft -> time-domain aliasing (wrap-around of the two-sided tails); (iv) forgetting Hermitian
  symmetry -> complex taps; (v) per-driver (not shared) delay -> mis-steer.

### 2b. Least-squares / minimax (scipy) — and their HARD limitation
VERIFIED against the scipy API and MATLAB-equivalence tables: scipy.signal.firls, firwin2, and
remez ALL design *real, symmetric, linear-phase* FIR filters (firls builds a Type-I linear-phase
filter with odd length; remez = Parks-McClellan minimax; firwin2 = windowed frequency-sampling of a
magnitude spec). They take a *magnitude/gain* spec vs frequency and impose linear phase themselves —
they CANNOT directly fit an arbitrary target phase. There is NO single scipy call for
"arbitrary-complex-phase complex-tap FIR." Consequences for the plan:
  - For arbitrary complex w_m(f): use the IFFT+window route (2a) or roll a complex least-squares solve
    (set up the over-determined system min_h || E w_m - exp(-j w tau) target ||^2 over a dense
    frequency set, where E is the DFT matrix restricted to Ntaps columns; solve with np.linalg.lstsq;
    optionally add a frequency-domain weighting diag to emphasize the steered/critical band). The
    complex-LS solve is the principled cousin of frequency-sampling and lets you weight bands.
  - firls/remez remain useful for the *magnitude-only* sub-problems (e.g. a per-driver shading/EQ that
    you deliberately want linear-phase), and for building band-limiting / anti-alias stages.

### 2c. ONE-STEP (joint, bake the spatial objective into the taps) vs TWO-STEP
- TWO-STEP (standard, and what GLL/EASE-class tools do): (i) per-frequency optimizer -> w_m(f);
  (ii) fit a filter to w_m(f). Modular, inspectable (the user/non-programmer can see w_m(f) and the
  realized response side by side), matches your data contract where w_m(f) is the natural
  intermediate.
- ONE-STEP / joint filter-and-sum design (Mabande & Kellermann; Kajala & Hamalainen filter-and-sum
  with adjustable characteristics; FIR constant-beamwidth designs): solve directly for the FIR tap
  vectors {h_m} of length L that minimize a spatial cost integrated over a frequency band — e.g.
  min_{h} sum_f sum_dir |sum_m H_m(f,dir) (DFT_f h_m) - P_target(f,dir)|^2 (+ robustness/white-noise-
  gain regularization). The spatial objective and the finite filter length are coupled in one solve.
- WHY one-step exists (the real discriminator, not "power"): per-bin optimal weights w_m(f) are often
  *erratic vs frequency* because the per-frequency problem is ill-conditioned / non-unique for small
  heterogeneous arrays. Fitting a short filter to an erratic w_m(f) needs huge tap counts or fails.
  One-step bakes finite-length (=> cross-frequency smoothness) into the design so the result is
  short-filter-realizable by construction.
- RECOMMENDATION for BeamSimII: TWO-STEP, but regularize w_m(f) for frequency-smoothness *before*
  fitting. Levers: (a) the GRPQ/penalty term in the per-bin optimizer (the arXiv 2407.01860
  bounded-weighting-matrix Lambda, Eqs. 3-6) to tame erratic weights; (b) an explicit cross-frequency
  smoothness penalty on w_m(f); (c) magnitude/phase smoothing on the log grid before IFFT. Use
  one-step as the FALLBACK when latency is critical (need short filters) or when weights refuse to
  smooth. Deciding constraints to expose in the plan: "can w_m(f) be regularized enough to fit short
  filters?" and "is latency critical?".

---

## 3. IIR / biquad realization

### 3a. The methods and the scipy reality
VERIFIED: there is NO scipy.signal.invfreqz. invfreqz/invfreqs are MATLAB routines that fit a
rational H(z)=B(z)/A(z) of specified order to a complex frequency response. Default = Levi
equation-error method (linear LS: build A(w)b - ... = desired and solve with backslash); the
improved mode refines with damped Gauss-Newton output-error minimization. Both fit real coefficients
by matching the response at +f and -f simultaneously. Family members: Prony, Yule-Walker,
Steiglitz-McBride, vector fitting.
To get this in Python:
  - Roll the Levi equation-error linear solve in numpy: assemble the over-determined complex linear
    system in (b,a) and solve with np.linalg.lstsq, fitting against complex w_m(f) samples.
  - Refine with scipy.optimize.least_squares minimizing output-error (||B/A - target||) for the
    Gauss-Newton step. Optionally project poles inside the unit circle for stability
    (invfreqz with weighting is NOT guaranteed stable).
  - Or a third-party port / vector-fitting package (several exist on PyPI/GitHub; maturity varies).
  - scipy.signal.tf2sos / zpk2sos to factor the fitted H(z) into a cascade of biquads (sos form) for
    numerically robust realization; scipy.signal.sosfilt to apply.

### 3b. Biquad-cascade greedy approach
Iterative: find the frequency region of max deviation between current and target response, add one
biquad to correct it, re-optimize, repeat until sections exhausted. Good for a small parametric-EQ
+ shelving + allpass cascade per driver.

### 3c. The GLL "gain + delay + IIR-crossover" form (Section 5 detail)
Production directivity tools (EASE / GLL, Feistel-Ahnert-Hughes-Olson, AES 7254, 2007) represent each
driver/passband as: a measured complex directivity balloon H[m,f,dir] (magnitude AND phase, up to 64k
IRs) PLUS a per-driver processing chain of GAIN + DELAY + IIR/FIR crossover-EQ filter. Total array
directivity is the COMPLEX (interference) summation P(f,dir)=sum_m Filter_m(f) H[m,f,dir] — exactly
your forward model, with the "Interference Sum" switch toggling complex-vs-power summation. So GLL is
a two-step pipeline and its realized filter object is (scalar gain) x (pure delay) x (low-order IIR
crossover). This is the natural EXPORT TARGET form for hardware DSP / DSP-amp presets.

### 3d. When IIR is preferable, and its hard limit
- Preferable when: low latency mandatory (IIR has no bulk modeling delay), tiny coefficient footprint,
  target maps onto a few standard biquads (shelves, peaks, crossover slopes), running on a fixed DSP.
- HARD LIMIT for beamforming: the standard "convert to minimum phase then fit" IIR trick
  (scipy.signal.minimum_phase + invfreqz) is FORBIDDEN here — it discards the absolute phase and
  re-zeros each driver independently, destroying inter-driver relative phase and mis-steering the
  beam. IIR is admissible ONLY if you fit the *complex* w_m(f) (complex h in the Levi/output-error
  solve, matching phase) AND then verify the realized inter-driver relative phase is preserved within
  tolerance (e.g. < a few degrees across the steered band). A low-order rational filter often CANNOT
  reproduce the steep, near-linear phase ramps that steering delays require; in that regime IIR fails
  and you must fall back to FIR (or add explicit per-driver *shared-referenced* allpass/delay). State
  this limit explicitly in the plan.

---

## 4. Latency vs fidelity; long-FIR convolution; per-driver alignment

- Latency/fidelity: linear-phase FIR latency = tau ~ (Ntaps-1)/2 samples, reported to the user as the
  beamformer's bulk delay. More taps -> closer match to w_m(f) and lower error floor -> more latency.
  IIR -> ~zero added latency but limited phase fidelity (Section 3d).
- Per-driver delay alignment: each driver's *true* time-of-flight is already encoded in H (the cardinal
  rule). Do NOT re-zero it. The only delay you add at realization time is the SHARED tau. If a driver's
  filter needs an integer bulk delay for causality, give every driver the same one. Any per-driver
  *relative* delay must come from the optimizer's w_m(f) phase, not from a hand-applied per-driver delay.
- Long FIR application: this deliverable is almost certainly OFFLINE filter export (coefficients +
  visualization), so "latency" = the reported modeling-delay number and partitioned convolution is a
  footnote. If you ever apply the filters to audio in-app, use scipy.signal.oaconvolve (overlap-add)
  or fftconvolve; true real-time low-latency playback would need uniformly/non-uniformly partitioned
  convolution (Gardner), but that is out of scope for a filter *designer/exporter*.

---

## 5. How real tools apply per-driver filters to directional data
- AES GLL / EASE (Feistel, Ahnert, Hughes, Olson, AES 7254, 2007, "Simulating Directivity Behavior of
  Loudspeakers with Crossover Filters"): per-driver complex balloon x (gain+delay+IIR/FIR crossover),
  combined by COMPLEX/interference summation = your forward model. Two-step. Export form = gain+delay+IIR.
- VituixCAD: per-driver measured complex frequency response + per-driver filter blocks (IIR biquads /
  delay / gain / optional FIR), summed complex with the directivity data to predict off-axis/power/DI —
  same two-step complex-summation pattern, widely used by DIY/pro crossover designers.
- NextGenAudio "least-squares cardioid filters in MATLAB": the LS approach — set up the desired
  front/back pressure (cardioid null) as a target and solve least-squares for the per-driver complex
  filters that realize it; canonical small-array, one-step-flavored LS realization (solve for the filter
  responses directly against a spatial target), then export FIR/IIR.

---

## 6. Convention foot-guns (call these out explicitly)

(a) Real taps require Hermitian symmetry. To get real audio-filter coefficients, the full-band spectrum
    fed to the IFFT must satisfy W(-f)=conj(W(f)) with real DC and (if present) real Nyquist. You only
    have the positive-frequency w_m(f); conjugate-mirror it onto the negative half before np.fft.ifft.
    Skip this and the taps come out complex (a sign you mis-built the spectrum).
(b) Time convention. H (and thus w_m(f)) is in NumCalc ENGINEERING convention exp(-jwt) (outgoing
    ~ exp(+jkr)). numpy.fft FORWARD transform also uses exp(-j2pi kn/N), so numpy's *inverse* (ifft,
    exp(+j2pi kn/N)) maps an engineering-convention spectrum to a physically sensible engineering-
    convention impulse response WITHOUT an extra conjugation — i.e. engineering convention and numpy.fft
    are consistent, so a straight ifft of the Hermitian-extended w_m(f) is correct. If any quantity were
    instead in Kinsler PHYSICS convention exp(+jwt), you would conjugate it first. Verify on a known
    pure-delay weight (w = exp(-j2pi f T)) that the realized filter delays by +T, not -T; a sign error
    here flips the steering direction (beam points to the mirror angle).
(c) Modeling-delay sign. The bulk delay must ADD positive group delay: multiply the (engineering-
    convention) spectrum by exp(-j*2pi*f*tau) before IFFT, or equivalently fftshift the impulse response
    by +(Ntaps-1)/2 samples. Getting the sign wrong makes the filter non-causal again (energy at n<0).
(d) Covariance / Hermitian transpose. In the optimizer (MECD/MSCD, pressure-matching), accept/reject
    covariance matrices are formed as outer products of steering vectors d(dir) with CONJUGATE transpose
    (Hermitian, ^H), e.g. C = integral f(dir) d(dir) d(dir)^H dOmega using Lebedev weights. Using plain
    transpose (not conjugate) silently solves a different problem. This does not change realization but
    is the matching convention note for the weight-solve stage.

---

## 7. Concrete recommended recipe (defensible default for the plan)

1. Optimizer outputs w_m(f) on the simulation frequency grid, with a frequency-smoothness regularizer
   on (GRPQ Lambda penalty and/or explicit cross-bin smoothing) so the weights are fittable.
2. TWO-STEP realization. Default filter type = LINEAR-PHASE FIR via IFFT+window (Section 2a), ONE shared
   modeling delay tau for all drivers.
3. Interpolate (log|w|, unwrapped phase) onto a dense linear FFT grid (Nfft ~ 2^16), Hermitian-extend,
   apply exp(-j2pi f tau), ifft, fftshift, truncate to per-driver Ntaps, Kaiser(beta~8) window.
4. Choose Ntaps from the lowest frequency / steepest phase slope you must reproduce; report tau as
   latency; verify (freqz, remove (Ntaps-1)/2 linear phase) that magnitude AND relative inter-driver
   phase match within tolerance and the pre-truncation IR decayed below ~-120 dB.
5. OPTIONAL low-latency export: fit COMPLEX w_m(f) with a Levi equation-error + output-error IIR
   (numpy lstsq + scipy.optimize.least_squares), factor to biquads (zpk2sos), stabilize poles, and
   verify relative phase preserved. NEVER use minimum_phase per driver. If IIR cannot hold the steering
   phase, fall back to FIR.
6. ONE-STEP joint filter-and-sum (min spatial error over band for fixed tap length) only if latency is
   critical or weights won't smooth.

### Sources
- (primary) Constant Directivity Loudspeaker Beamforming (Yuancheng Luo, Amazon) — arXiv:2407.01860 — https://arxiv.org/html/2407.01860
- (secondary) FIR Filter to Match Any Magnitude and Phase Response (Dan Boschen, DSPRelated #1760) — https://www.dsprelated.com/showarticle/1760.php
- (secondary) Frequency Sampling Method for FIR Filter Design (Julius O. Smith, Spectral Audio Signal Processing) — https://www.dsprelated.com/freebooks/sasp/Frequency_Sampling_Method_FIR.html
- (secondary) The Complete FIR Filter Guide for Loudspeaker Audio Optimization (Eclipse Audio) — https://eclipseaudio.com/fir-filter-guide/
- (primary) Generic Loudspeaker Library (GLL) White Paper, Oct 2007 (AFMG / Feistel & Ahnert) — https://www.afmg.eu/sites/default/files/2021-07/GLL_White_Paper_October07.pdf
- (primary) Simulating Directivity Behavior of Loudspeakers with Crossover Filters (Feistel, Ahnert, Hughes, Olson, AES Convention Paper 7254, 2007) — AFMG info page — https://www.afmg.eu/en/simulating-directivity-behavior-loudspeakers-crossover-filters
- (primary) GLL Loudspeaker File Format (AFMG) — per-driver IIR/FIR EQ + crossover filters, complex phase data — https://www.afmg.eu/en/gll-loudspeaker-file-format
- (primary) FIR-Based Symmetrical Acoustic Beamformer With a Constant Beamwidth (Signal Processing / ScienceDirect) — https://www.sciencedirect.com/science/article/abs/pii/S0165168416301700
- (primary) Filter-and-sum beamformer with adjustable filter characteristics (Kajala & Hamalainen, IEEE) — https://ieeexplore.ieee.org/document/940257/
- (primary) Window-Based Constant Beamwidth Beamformer (PMC) — https://pmc.ncbi.nlm.nih.gov/articles/PMC6539959/
- (primary) MATLAB invfreqz / invfreqs documentation (Levi equation-error + Gauss-Newton output-error IIR fit) — https://www.mathworks.com/help/signal/ref/invfreqz.html
- (secondary) scipy.signal FIR design (firls/firwin2/remez) vs MATLAB equivalence (MNE filter docs) — https://mne.tools/stable/auto_tutorials/preprocessing/25_background_filtering.html

### Decisions implied
- Default Phase-2 filter realization = LINEAR-PHASE FIR via IFFT+window (frequency-sampling), with ONE shared modeling delay tau applied identically to all drivers; report tau as the beamformer latency.
- Implement the FIR designer in pure numpy/scipy: interpolate (log|w|, unwrapped phase) onto a dense linear FFT grid, Hermitian-extend, multiply by exp(-j2pi f tau), np.fft.ifft, np.fft.fftshift, truncate to per-driver Ntaps, apply scipy.signal.windows.kaiser(beta~8); do NOT rely on firls/firwin2/remez for the complex-phase fit.
- Adopt a TWO-STEP pipeline (optimizer -> w_m(f) -> filter fit) and add a frequency-smoothness regularizer on w_m(f) (GRPQ-style Lambda penalty and/or explicit cross-bin smoothing) BEFORE fitting, so short filters are realizable; keep one-step joint filter-and-sum as a documented fallback for latency-critical cases.
- Provide an OPTIONAL low-latency IIR export that fits the COMPLEX w_m(f) via a rolled Levi equation-error solve (np.linalg.lstsq) refined by scipy.optimize.least_squares, factored to biquads with scipy.signal.zpk2sos, with explicit pole-stabilization and a relative-phase-preservation check; NEVER call scipy.signal.minimum_phase per driver.
- Add a verification/self-test stage: reconstruct each realized filter with scipy.signal.freqz, remove the known (Ntaps-1)/2 linear phase, and assert magnitude AND inter-driver relative phase match w_m(f) within tolerance (e.g. a few degrees over the steered band) and that the pre-truncation impulse response decayed below ~-120 dB (bounds time-aliasing).
- Choose per-driver Ntaps from the lowest frequency / steepest phase slope that must be reproduced (low-f woofer beams dominate the tap budget; mid/high-only beams need far fewer taps); expose Ntaps and tau as user-visible latency/fidelity knobs.
- Default export object form = gain + shared-delay + (FIR taps | IIR biquad sos), matching the GLL/VituixCAD per-driver filter convention for downstream DSP, with w_m(f) kept as an inspectable intermediate for the non-programmer user.
- Add a convention guard/unit test: feed a pure-delay weight w=exp(-j2pi f T) through the realizer and assert the resulting filter delays by +T (not -T), catching engineering-vs-physics conjugation and modeling-delay sign errors that would flip steering direction.

### Open questions
- Is the Phase-2 deliverable strictly offline filter EXPORT (coefficients + plots), or will the app also apply filters to audio in real time? This decides whether partitioned convolution (Gardner) matters at all or is a footnote.
- What relative-phase preservation tolerance (degrees, over which frequency band and which directions) constitutes a passing realization — needed to set the FIR tap budget and to gate IIR admissibility.
- What is the lowest frequency the beamformer must steer (sets the dominant tap-count / latency budget) and the maximum acceptable bulk latency tau for the target use case.
- Will the per-bin optimizer expose a frequency-smoothness regularizer (GRPQ Lambda or explicit cross-bin penalty), or must smoothing be a separate post-process before fitting — affects whether short filters are achievable and whether one-step is ever needed.
- Exact internal filter representation of the chosen Phase-2 hardware/export target (raw FIR taps vs biquad sos vs gain+delay+IIR-crossover GLL-style block) — determines whether IIR fitting is a required feature or an optional convenience.

### Adversarial verification verdicts
  - [confirmed] Claim 1: A single bulk modeling delay tau applied IDENTICALLY to every driver factors out of P(f,dir)=exp(-j2pi f tau) sum_m w_m(f) H[m,f,dir], preserving inter-driver relative phase and beam steering; a per-driver delay or per-driver minimum-phase conversion changes relative phase and silently mis-steers the beam.
  - [partially-correct] Claim 2: An arbitrary complex target w_m(f) has a two-sided non-causal inverse-DTFT impulse response; it is made causal only by adding a linear-phase modeling delay (fftshift / exp(-j2pi f tau)) so energy moves to n>=0, at cost of ~(Ntaps-1)/2 samples latency. -> CORRECTION: An arbitrary complex w_m(f) has a generally two-sided non-causal impulse response. The PHASE-PRESERVING way to make it causal is a shared linear-phase modeling delay (fftshift / exp(-j2pi f tau)), costing ~(Ntaps-1)/2 samples latency. Minimum-phase reconstruction is the other mathematically causal route but is forbidden here because it alters phase (see Claim 4).
  - [confirmed] Claim 3: scipy.signal has NO invfreqz equivalent, and firls / firwin2 / remez all produce REAL, symmetric, LINEAR-PHASE FIR filters driven by a magnitude/gain spec -- none can directly fit an arbitrary target phase; arbitrary-complex-phase FIR must use the IFFT+window (frequency-sampling) route or a custom complex least-squares solve.
  - [confirmed] Claim 4: 'convert target to minimum phase then fit with invfreqz' is FORBIDDEN for beamforming weights (discards absolute phase, mis-steers); IIR is admissible only by fitting the COMPLEX w_m(f) (Levi equation-error + Gauss-Newton output-error) and verifying inter-driver relative phase is preserved.
  - [partially-correct] Claim 5: Production directivity tools (GLL/EASE, Feistel et al. AES 7254 2007; VituixCAD) are TWO-STEP: per-driver complex balloon times a gain+delay+IIR/FIR-crossover filter, combined by COMPLEX (interference) summation P=sum_m Filter_m H[m,f,dir] -- exactly the BeamSimII forward model. -> CORRECTION: Production tools (AFMG GLL/EASE, AES 7254; VituixCAD) do use per-source complex directivity times per-source gain/delay/crossover-EQ filters combined by coherent complex pressure summation -- structurally identical to BeamSimII's P=sum_m w_m H_m. Confirmed DESCRIPTIVELY (complex/phase-aware data required; per-pass-band responses added together) but the literal summation equation was not obtainable from a primary non-paywalled source; treat the exact-equation correspondence as well-supported-by-structure, not quoted-from-primary.
  - [refuted] Claim 6: For real-valued taps the dense FFT spectrum must be Hermitian-extended W(-f)=conj(W(f)); because NumCalc engineering convention exp(-jwt) and numpy.fft forward both use exp(-j2pi kn/N), a straight numpy.fft.ifft of engineering-convention weights needs NO extra conjugation (a physics-convention exp(+jwt) quantity would need conjugation first). -> CORRECTION: Real taps require Hermitian extension W(-f)=conj(W(f)) (correct). But numpy.fft.fft's exp(-j) is an ANALYSIS kernel matching the PHYSICS/DSP convention, so the conjugation direction is REVERSED: a NumCalc ENGINEERING-convention (exp-jwt) weight (physical-delay transfer exp(+jwD)) must be CONJUGATED before np.fft.ifft to land causally at n=D (empirically, straight ifft peaks anti-causally at n=N-D). A physics/DSP exp(+jwt) quantity needs NO conjugation. Practical rule: conjugate ONCE and consistently -- flip the whole H tensor to DSP convention up front, OR conjugate at the ifft step. Doing neither yields anti-causal taps AND a conjugated phase per driver; since conjugation is not a common scalar it does not factor out and mirror-steers the beam.


---

## TOPIC: Robustness and regularization for beamformer weight design with a small number of heterogeneous loudspeaker drivers (superdirective robustness via white-noise-gain constraint / diagonal loading), as applied to BeamSimII Phase 2.

### Report
## Scope and how the literature maps to BeamSimII

The microphone-array superdirectivity literature transfers to BeamSimII's loudspeaker beamformer almost line-for-line by RECIPROCITY: a receive beamformer that combines mic signals `w^H y` is the same math object as a transmit array that excites drivers with weights `w` and radiates `P(direction) = sum_m w_m H[m, direction]`. The "array gain" / "array sensitivity" in direction θ is `|w^H d(θ)|` (Boyd/Mutapcic 2007). Crucial substitutions for BeamSimII:

- The analytic plane-wave steering vector `d(ω,θ)` in every paper is REPLACED by the BEM column `d_f = H[:, f, target_dir]` (length M = #drivers), complex128, already carrying each driver's true time-of-flight phase relative to the common origin. **Do not re-normalize or min-phase it — that is exactly the cardinal-rule violation.**
- The diffuse/isotropic coherence matrix `Γ_d(ω)` (the `sinc` matrix in array papers) is REPLACED by the measured cross-driver radiation coherence from H over the sphere with Lebedev weights: `Γ_f = sum_n a_n H[:,f,n] H[:,f,n]^H / (4π)`, `a_n` = Lebedev quadrature weights (sum a_n = 4π). This is the Bitzer & Simmer (2001) "modified coherence function" framework; it is the matrix inverted in the supergain solution and the one that goes ill-conditioned at low frequency.

Notation: M = #drivers; `w ∈ C^M`; `d = H[:,f,steer]`; `(·)^H` conjugate transpose; engineering convention exp(−jωt) — **H from NumCalc is already exp(−jωt), so do NOT conjugate it; with `w^H d = 1` you reproduce the on-axis target with correct phase. Using d^* steers to the mirror direction.** Quadratic forms `w^H A w` are real ≥ 0 since Γ_f, I are Hermitian PSD.

---

## 1. White-noise gain (WNG) and the WNG constraint

**Definition (Atkins/Cohen/Benesty 2016, Eq. 5; Cox-Zeskind-Owen 1987):**

    WNG(w) = |w^H d|^2 / (w^H w)        ≤ M

WNG is the array SNR-gain when the noise/error field is spatially WHITE (Γ = I); the standard ROBUSTNESS proxy — gain against (a) uncorrelated driver self-noise and (b) small INDEPENDENT per-channel gain/phase/position errors. Its reciprocal **1/WNG = the sensitivity T_se** (Cox-Zeskind-Owen): amplification of independent per-channel perturbations.

Under the distortionless constraint `w^H d = 1` (BeamSimII's natural unit-on-axis normalization):

    WNG(w) = 1 / (w^H w) = 1 / ||w||^2

So the **WNG floor `WNG ≥ W_floor` is identically the weight-norm bound:**

    ||w||^2 ≤ 1 / W_floor        (with w^H d = 1)         (★ master robustness inequality)

This is the load-bearing equivalence: bounding WNG from below = bounding squared weight norm from above. A superdirective solution at low ka has WNG ≪ 1 (e.g. −50 dB for a 4-element 3rd-order design at 1 kHz), so ||w||^2 ≈ 1e5 and a 1% driver error yields ~30 dB of output error. W_floor = 0 dB forces ||w|| ≤ 1; W_floor = −10 dB forces ||w||^2 ≤ 10.

**Max-WNG = delay-and-sum (Atkins Eq. 6):** `w_DS = d/(d^H d) = d/M`, WNG = M. Most robust, least directive — the natural "fully damped" slider end.

**Tradeoff:** directivity factor `DF(w) = |w^H d|^2/(w^H Γ_f w) ≤ M^2` (Atkins Eq. 7). Max-DF (superdirective/MVDR) and max-WNG (DS) are the extremes; every robust design lives on the curve between them, set by one scalar.

---

## 2. Diagonal loading / Tikhonov regularization ↔ WNG constraint

**Regularized superdirective (= diagonally-loaded MVDR), Atkins Eq. 9:**

    w(ε) = [Γ_f + ε I]^{-1} d  /  ( d^H [Γ_f + ε I]^{-1} d )           (R1)

ε=0 → pure superdirective (max DF, min WNG); ε→∞ → delay-and-sum (max WNG, min DF). ε ≥ 0 is the Lagrange multiplier dual to "max DF subject to WNG ≥ W_floor" (Cox-Zeskind-Owen): loading the diagonal IS imposing the WNG constraint.

**Bounded one-knob form (Atkins Eq. 11) — RECOMMENDED (finite [0,1] knob):**

    Γ_{d,α} = (1−α) Γ_f + α I,   α ∈ [0,1]
    w(α) = Γ_{d,α}^{-1} d / ( d^H Γ_{d,α}^{-1} d )                     (R2)

Same family: ε = α/(1−α). α=0 → superdirective; α=1 → delay-and-sum.

**Eigenvalue picture (why loading raises WNG).** Γ_f = U Λ U^H, λ_1≥…≥λ_M>0; loading shifts λ_i→λ_i+ε. With g_i = (U^H d)_i:

    w^H w  ∝ sum_i |g_i|^2/(λ_i+ε)^2
    w^H d  ∝ sum_i |g_i|^2/(λ_i+ε)
    WNG(ε) = ( sum_i |g_i|^2/(λ_i+ε) )^2 / ( sum_i |g_i|^2/(λ_i+ε)^2 )   (R3)

Tiny λ_i (which blow up 1/λ_i^2 in the norm) get floored at ε, so ||w||^2 collapses and WNG rises **monotonically** in ε (and α); DF falls monotonically. Monotonicity makes "ε achieving target WNG" a well-posed 1-D root-find (Atkins Sec. 4).

**Loading that achieves a target WNG:** since WNG(ε) is continuous and strictly monotone from WNG(0) up to M, solve `WNG(ε) = W_target` by bisection/Newton on log ε per frequency, O(log(1/tol)) iters. No clean closed form for general Γ_f (Atkins states this explicitly); iterative solve is correct. Atkins Algorithm 1 ("Minimize-and-Search") is the reference recipe, complexity O(|F|·log2((M^2−M)/σ)).

---

## 3. Choosing the regularization parameter

**(a) Target-WNG (primary).** Pick WNG FLOOR W_floor(f) in dB; solve R3 per frequency for ε(f). Direct physical control. Floors: 0 dB (very robust), −6…−10 dB (practical sweet spot), −20 dB (aggressive supergain, needs well-matched drivers).

**(b) Fixed effort norm ||w||^2 ≤ β.** Identical to (a) via β = 1/W_floor. Most intuitive "amplifier headroom / cone-excursion budget" knob for a loudspeaker designer. Smallest ε with ||w(ε)||^2 ≤ β (monotone → bisection).

**(c) L-curve.** Per frequency, plot log(beam-shape error / rejection level G(w)) vs log(||w||^2); pick ε at MAX CURVATURE corner (Hansen parametric curvature). Good automatic default; less transparent to a user.

**(d) Worst-case per-channel error ρ (Boyd/Mutapcic 2007).** State per-driver relative implementation error ρ (ρ=0.05 ≈ 5% magnitude / ±2.86° phase). Worst-case design = weighted complex L1-regularization (their Eq. 5); L2 diagonal loading (Eq. 3, σ_i^2 = per-channel error power) is within ~0.5 dB of it. Quotable UI honesty bound:

    rejection_floor ≥ rejection_nominal + ρ/(1−ρ)            (Boyd Eq. 9)

ρ=0.05 ⇒ cannot beat ~−25.6 dB rejection regardless of geometry/driver count.

**Frequency dependence (critical).** Regularization MUST grow toward low f because Γ_f conditioning worsens (item 4). Hold a CONSTANT WNG FLOOR vs frequency and let ε(f) auto-grow at low f (the solve does this). A constant WNG floor yields a CONSTANT-DIRECTIVITY beam over frequency until physics forbids it, then ε clamps and DI rolls off gracefully — exactly BeamSimII's constant-directivity target. Always add a tiny FLOOR loading ε_min ≈ 1e-10·trace(Γ_f)/M (Atkins use 1e-14) to avoid singular inversion; note they warn that at the very lowest bins this floor dominates and the target WNG cannot be met — flag to user, never emit garbage.

---

## 4. Condition number at low frequency

**Why it explodes.** For closely-spaced drivers Γ_f becomes near-rank-deficient as f→0 (all drivers radiate near-identically, kr≪1 ⇒ near-collinear columns / "cluster of colliding nodes"). Smallest eigenvalue vanishes, condition number diverges. Established N-element scaling:

    λ_min(Γ_f) ∝ (k d)^{2(N−1)},    κ(Γ_f) = λ_max/λ_min ∝ (k d)^{-2(N−1)}   (C1)

N=2: κ ~ (kd)^{-2}; N=3: ~(kd)^{-4}. Doubling f improves conditioning by 6(N−1) dB; halving f wrecks it equally. Origin of WNG collapse: `w_SD = Γ_f^{-1}d/(d^H Γ_f^{-1}d)` divides by λ_min, so ||w||^2 ~ (kd)^{-4(N−1)} and WNG ~ (kd)^{+4(N−1)}→0.

**Fix.** Loading floors the spectrum: κ(Γ_f+εI) ≤ (λ_max+ε)/ε ≈ λ_max/ε once ε≫λ_min. To bound condition number to κ_max:

    ε ≈ λ_max / κ_max                  (condition-number-targeting loading)   (C2)

Loading "may be formulated as constraining condition number of R, norm of w, OR white-noise gain" — all the same knob. For BeamSimII, computing κ(Γ_f) per frequency is also the diagnostic/trigger: κ > ~1e8 (double precision losing ~8 of 16 digits) ⇒ force regularization, surface "supergain-limited below X Hz."

---

## 5. Numerical guidance + the single user-facing robustness knob

**Per-frequency weight solve (core of Phase 2):**
1. d = H[:, f, steer_idx] (do NOT conjugate/renormalize — preserves time-of-flight).
2. Γ_f = sum_n a_n H[:,f,n] H[:,f,n]^H / (4π) (Lebedev-weighted, Hermitian PSD). For pure max-DF, Γ may be built from the desired-noise/rejection region only.
3. ε_min = 1e-10·trace(Γ_f)/M (always added).
4. Solve WNG(ε)=W_floor(f) for ε≥ε_min by bisection on log ε (monotone, ~30–40 iters to 1e-6); equivalently in α∈[0,1] (R2) — bounded interval is more stable.
5. w = (Γ_f+εI)^{-1} d / (d^H (Γ_f+εI)^{-1} d) via Cholesky / scipy.linalg.cho_solve (Hermitian PD after loading; M tiny so cost negligible).
6. Report achieved WNG(w), DF/DI(w), κ(Γ_f) for UI plots.

**Typical ranges (M = 2–8 drivers):**
- Relative loading ε/λ_max: 1e-6 (barely) … 1e-1 (heavy); 1e-2 a sane moderate default; DMA literature uses 1e-4 (aggressive) … 1e-2.
- WNG floor: −20 dB (aggressive) … −10 dB (moderate, good default) … 0 dB (DS-like, bulletproof). Ship −6 dB as "balanced".
- **Normalize Γ_f before loading** (divide by trace/M, or scale d to ||d||=√M) so ε and the WNG floor mean the same thing across drivers/frequencies; otherwise a fixed ε behaves wildly differently at 30 Hz vs 3 kHz.

**ONE non-expert knob — "Robustness / Effort" slider s∈[0,1] → WNG FLOOR (dB):**
W_floor_dB(s) = W_min + s·(W_max − W_min), W_min=−20 dB (s=0 "max directivity, fragile"), W_max capped at 10·log10(M) dB (s=1 "delay-and-sum, robust"). Each frequency solves its own ε(f) to hold this floor; low f auto-gets more loading. Slider ends in user language: "Sharpest beam (needs matched, low-noise drivers)" ↔ "Most forgiving (tolerates driver mismatch & noise)." Live readouts: WNG vs f, DI vs f, max|w_m| ("drive effort"). Optionally expose Boyd ρ ("assumed driver matching tolerance, %") as alternative framing with the honest rejection-floor bound (Eq. 9).

**Self-test invariants:** WNG(α) monotone↑; DF(α) monotone↓; w(α=1)==d/M (DS) to machine precision; w(α=0)==normalized Γ_f^{-1}d; loaded matrix Hermitian PD (Cholesky succeeds); w^H d == 1 exactly after normalization (distortionless preserved → cardinal phase-origin rule intact).

### Sources
- (primary) H. Cox, R. M. Zeskind, M. M. Owen, "Robust Adaptive Beamforming," IEEE Trans. ASSP 35(10):1365-1376, Oct. 1987. Origin of WNG-constraint, sensitivity=1/WNG=||w||^2, diagonal-loading-as-Lagrange-multiplier. NOTE: authors are Cox, ZESKIND, Owen — the task's 'Zucker' is a misattribution (Cox-Zucker is unrelated arithmetic geometry). — https://ieeexplore.ieee.org/document/1165054/
- (primary) A. Atkins, Y. Ben-Hur, I. Cohen, J. Benesty, "Robust Superdirective Beamformer with Optimal Regularization," IWAENC 2016. Verbatim WNG (5), DS (6), DF (7), superdirective (8), regularized SD (9), bounded alpha-form (11), Minimize-and-Search Algorithm 1. — https://israelcohen.com/wp-content/uploads/2018/05/IWAENC2016_Atkins1.pdf
- (primary) A. Mutapcic, S.-J. Kim, S. Boyd, "Beamforming With Uncertain Weights," IEEE Signal Processing Letters 14(5):348-351, May 2007. Rejection-band SOCP (1)-(2), L2 diagonal-loading (3), worst-case multiplicative-uncertainty L1 form (5), rho->rejection-floor bound (9) p_rob >= p_nom + rho/(1-rho). — https://web.stanford.edu/~boyd/papers/pdf/beamform_reg.pdf
- (secondary) J. Bitzer, K. U. Simmer, "Superdirective Microphone Arrays," in Brandstein & Ward (eds.), Microphone Arrays, ch. 2, pp. 19-38, Springer 2001. Canonical modified-coherence-matrix superdirective framework Gamma->(1-mu)Gamma+mu*I and WNG-constrained MVDR. — https://link.springer.com/chapter/10.1007/978-3-662-04619-7_2
- (primary) S. Doclo, M. Moonen, "Superdirective Beamforming Robust Against Microphone Mismatch," IEEE Trans. ASLP 15(2):617-631, 2007 (and IEEE Trans. SP 51(10):2511-2526, 2003). Statistical mic-characteristic robustness belongs to the WNG-constraint/diagonal-loading regularization class. — https://ieeexplore.ieee.org/document/4100694/
- (primary) R. Berkun, I. Cohen, J. Benesty, "Combined Beamformers for Robust Broadband Regularized Superdirective Beamforming," IEEE/ACM Trans. ASLP 23(5):877-886, 2015 (and IWAENC 2016 tunable variant). Tunable DS<->SD blend with one robustness parameter. — https://dl.acm.org/doi/abs/10.1109/TASLP.2015.2410139
- (primary) "Design of a Differential Loudspeaker Line Array for Steerable Frequency-Invariant Beamforming," Sensors (MDPI) 24(19):6277, 2024. Loudspeaker (transmit) robust differential beamforming via Tikhonov-regularized modal matching; WNG floors and frequency-dependent regularization — closest analog to BeamSimII. — https://www.mdpi.com/1424-8220/24/19/6277
- (primary) J. Li, P. Stoica, Z. Wang, "On Robust Capon Beamforming and Diagonal Loading," IEEE Trans. SP 51(7):1702-1715, 2003. Equivalence of diagonal loading, norm constraint, and condition-number constraint; computing the loading level for a given norm/WNG bound. — https://ieeexplore.ieee.org/document/1206680/
- (primary) "Adaptive Diagonal Loading for Norm Constrained Beamforming," arXiv:2605.04342, 2026. WNG W=1/|w^H w| under distortionless constraint; W=(d^H R^-1 d)^2/(d^H R^-2 d); loading set to meet norm/WNG/condition-number target. — https://arxiv.org/html/2605.04342
- (primary) "Twenty-Five Years of Advances in Beamforming," arXiv:2211.02165, 2022. Survey unifying MVDR, diagonal loading epsilon, WNG/norm constraints, and worst-case SOCP robust formulations under one regularization framework. — https://arxiv.org/pdf/2211.02165

### Decisions implied
- Phase 2 weight solve is per-frequency diagonally-loaded MVDR: w = (Gamma_f + eps*I)^-1 d / (d^H (Gamma_f + eps*I)^-1 d), solved by Cholesky (scipy.linalg.cho_solve) since M is tiny; use the bounded alpha in [0,1] parameterization internally for numerical stability.
- Build Gamma_f as the Lebedev-quadrature-weighted cross-driver coherence Gamma_f = sum_n a_n H[:,f,n] H[:,f,n]^H / (4*pi) directly from the Phase-1 H tensor and the stored quadrature weights; the steering vector is d = H[:,f,steer_idx] used WITHOUT conjugation or renormalization to preserve the common phase origin (cardinal rule).
- Expose exactly ONE non-expert robustness knob, a slider s in [0,1] mapped to a WNG FLOOR in dB (W_min ~ -20 dB to W_max ~ 10*log10(M) dB); per frequency solve WNG(eps)=W_floor by bisection on log(eps), giving automatic extra regularization at low frequency and a constant-directivity-vs-frequency beam until physics clamps it.
- Always add a numerical floor loading eps_min ~ 1e-10 * trace(Gamma_f)/M before inversion, and detect/flag the low-frequency bins where the requested WNG floor is unreachable (eps clamps, DI rolls off) rather than emitting garbage; compute and surface kappa(Gamma_f) per frequency as the supergain-limit diagnostic (trigger regularization when kappa > ~1e8).
- Normalize Gamma_f (by trace/M) and/or scale so ||d||=sqrt(M) before loading so the loading level and WNG floor have geometry/frequency-independent meaning across heterogeneous woofer/mid/tweeter drivers.
- Ship self-tests asserting the robustness invariants: WNG monotone-increasing and DF monotone-decreasing in alpha; w(alpha=1)==d/M (delay-and-sum); loaded matrix Hermitian positive-definite (Cholesky succeeds); and w^H d == 1 exactly (distortionless preserved => phase-origin/time-of-flight intact, guarding the V-5 superposition test).
- Offer an optional alternative framing of the same knob as Boyd's per-driver matching tolerance rho (%), and display its honest worst-case rejection floor rejection >= rejection_nominal + rho/(1-rho) so the UI never promises a beam the hardware cannot physically realize.

### Open questions


### Adversarial verification verdicts
  - [confirmed] Under distortionless constraint w^H d = 1, WNG = 1/||w||^2; so a WNG floor WNG >= W_floor is exactly the convex norm bound ||w||^2 <= 1/W_floor (Cox-Zeskind-Owen 1987; Atkins 2016 Eq.5; arXiv:2605.04342).
  - [confirmed] Regularized superdirective weight w(eps) = (Gamma_f + eps I)^-1 d / (d^H (Gamma_f + eps I)^-1 d) (Atkins Eq.9); eps=0 = pure MVDR/superdirective, eps->inf = delay-and-sum d/M; bounded form Gamma_{d,alpha}=(1-alpha)Gamma_f+alpha I, alpha in [0,1], with eps=alpha/(1-alpha) (Atkins Eq.11).
  - [confirmed] WNG(eps) is continuous and strictly MONOTONICALLY INCREASING in eps (and alpha) while DF is strictly monotonically decreasing; this makes WNG(eps)=W_target a well-posed 1-D bisection/Newton root-find per frequency (Atkins Sec.4 + Algorithm 1). No general closed form for eps. -> CORRECTION: One precision fix on the algorithm attribution: Atkins Algorithm 1 ('MAS - Minimize and Search') is written to target the combined-noise SNR GAIN G0, which the paper proves is NOT monotonic in alpha (it has a single minimum at alpha_min, decreasing on [0,alpha_min] then increasing on [alpha_min,1]) -- hence the two-section minimize-then-search. For BeamSimII's WNG (or DF) target, the quantity IS strictly monotone, so the 'Minimize' phase is unnecessary: a plain 1-D bisection/Newton on log(eps) or on alpha suffices. The claim's characterization of WNG/DF monotonicity is exactly right; just do not copy the two-phase Algorithm 1 verbatim when the target is WNG -- a single monotone bisection is correct and simpler.
  - [partially-correct] Condition number of the diffuse/cross-driver coherence matrix scales as kappa ~ (k*d)^{-2(N-1)} as f->0 (smallest eigenvalue ~ (k*d)^{2(N-1)}); this drives superdirective WNG collapse: ||w||^2 ~ (k*d)^{-4(N-1)}, WNG ~ (k*d)^{+4(N-1)}. -> CORRECTION: The correct scaling is ||w||^2 ~ (k*d)^{-2(N-1)} and WNG ~ (k*d)^{+2(N-1)} -- SAME exponent as the condition number, not double it. The claim's error is naively squaring the kappa exponent because ||w||^2 involves Gamma^-2 (so '(N-1) condition-number factors squared = 4(N-1)'). That double-counts: the steering vector's projection onto the smallest-eigenvalue eigenmode also vanishes as (kd)^{N-1}, cancelling one factor of (kd)^{2(N-1)} in d^H Gamma^-2 d. Physical discriminator (the well-known result that settles it): a 1st-order differential array (N=2, order 1) loses WNG at 6 dB/octave = (kd)^2, NOT 12 dB/octave. General rule: WNG falls 6*order dB/octave = (kd)^{2*order} = (kd)^{2(N-1)}. Primary references for this rule: Elko, 'Superdirectional Microphone Arrays' / Benesty & Chen, 'Study and Design of Differential Microphone Arrays' (Springer 2013); Bitzer & Simmer (2001). Practical impact for BeamSimII: the WNG collapse and the per-channel-error amplification are 2x LESS severe (in dB-exponent) than the claim states -- still catastrophic at low ka for high differential order, but the plan should not over-budget headroom by assuming the 4(N-1) law.
  - [partially-correct] Diagonal loading is three equivalent knobs: constraining the condition number (eps ~ lambda_max/kappa_max), the weight norm ||w||^2, or the white-noise gain (Li-Stoica-Wang 2003; arXiv:2211.02165); loading shifts each eigenvalue lambda_i -> lambda_i + eps. -> CORRECTION: The parenthetical loading formula 'eps ~ lambda_max/kappa_max' is an oversimplification and is not what the primary sources give. arXiv:2605.04342 Eq.(16) gives the loading that enforces a target condition number kappa_max as mu = max(0, (lambda_max - kappa_max*lambda_min)/(kappa_max - 1)), which depends on BOTH lambda_max AND lambda_min (and kappa_max), not lambda_max/kappa_max alone. Use mu = (lambda_max - kappa_max*lambda_min)/(kappa_max - 1) (clipped at >=0). Also note the three knobs are monotonically equivalent / interchangeable but are NOT the identical numerical eps in general -- mapping between a WNG target and an eps still requires the per-frequency solve of claim 3; only under the distortionless constraint is the WNG<->norm map the clean algebraic identity of claim 1.
  - [confirmed] Boyd/Mutapcic 2007: worst-case robustness against per-channel multiplicative implementation error of relative magnitude rho is a weighted complex L1-regularization; ordinary L2 diagonal loading comes within ~0.5 dB of it; achievable rejection is floored at rejection_nominal + rho/(1-rho) (e.g. rho=0.05 => no better than ~-25.6 dB), independent of geometry/driver count.


---

## TOPIC: Phase 2 validation-target patterns: Keele CBT Legendre shading + first-order cardioid family (exact analytic references for pytest tolerance checks)

### Report
# Phase-2 Validation Targets: CBT Legendre Shading & First-Order Cardioid Family

All formulas below are stated in BeamSimII's **engineering convention** (time factor exp(-jwt), outgoing wave ~ exp(+jkr)). I verified every numeric anchor in `uv run python` against the project's scipy. Convention subtleties are called out inline because the cardioid family is front/back asymmetric and a sign error is silently invisible to a symmetric beamwidth test (it is NOT invisible for the cardioid null).

---

## PART A — Keele CBT (Constant Beamwidth Transducer)

### A.0 Origin of the theory (cite this, not just Keele)
The continuous-shading CBT is from underwater acoustics: **Rogers & Van Buren, "New approach to a constant beamwidth transducer," JASA 64(1), 1978** (DOI 10.1121/1.381954). Keele adapted it to loudspeaker line arrays in five+ AES papers starting 2000. The transducer is a spherical (or circular-arc) cap of half-angle theta_0 whose surface normal velocity (or pressure) is shaded by a **Legendre function** P_nu(cos theta), theta measured from the cap axis, 0 <= theta <= theta_0.

### A.1 Exact Legendre shading (the requested exact form)
Surface velocity/pressure shading over the cap:
```
S(theta) = P_nu(cos theta),   0 <= theta <= theta_0    (theta from cap axis)
```
- `theta_0` = cap half-angle (total cap arc angle = 2*theta_0).
- `nu` = Legendre **degree, generally NON-INTEGER**, fixed by the boundary condition that the cap **rim is the first zero** of the shading:
```
P_nu(cos theta_0) = 0   (nu = smallest positive root)
```
- P_nu for non-integer nu is computed via the Gauss hypergeometric form (no special lib needed):
```
P_nu(x) = 2F1(-nu, nu+1; 1; (1-x)/2)   # scipy.special.hyp2f1(-nu, nu+1, 1, (1-x)/2)
```

**VERIFIED closed-form order relation (Mehler-Heine asymptotic, I derived & checked numerically):**
```
(nu + 0.5) * theta_0  ≈  j_{0,1} = 2.404825   (first zero of Bessel J0)
=>  nu ≈ 2.4048/theta_0 - 0.5     (theta_0 in radians)
```
Checked: cap_half=20deg -> nu=6.383, 30deg -> 4.084, 45deg -> 2.548, 50deg -> 2.240; the product (nu+0.5)*theta_0 = 2.403, 2.400, 2.394, 2.391 respectively — within 0.5% of 2.4048 across the whole range. This is the Rogers&VanBuren design relation in clean form (nu scales as 1/theta_0). [INFERRED from theory + VERIFIED numerically, 2026.]

### A.2 Keele's practical polynomial approximation (the implementable form — USE THIS for weights)
Keele's third/four-term power-series fit to P_nu vs **normalized position** x = theta/theta_0 (valid for all useful orders), from CBT Paper 1 (AES 109th, 2000):
```
U(x) = 1 + 0.066*x - 1.8*x^2 + 0.743*x^3,   0 <= x <= 1   (U = 0 for x > 1)
```
- U(0) = 1 (center driver, no attenuation); U(1) = 0.009 ≈ 0 (rim drivers ~ -40 dB).
- **Amplitude weight in dB:** `U_dB = 20*log10(U(x))`.
- VERIFIED numerically: U=0.5 (-6 dB) at **x = 0.6398** (Keele states "≈0.64"); U = -3.40 dB at x=0.5; U = -12 dB at x = 0.822.

### A.3 Why constant beamwidth, the 0.64 rule, and the cutoff
- **Frequency independence:** the Legendre shading is purely an amplitude taper, **independent of frequency**. Above a cutoff the shaded aperture produces a polar pattern whose -6 dB beamwidth is locked to the geometry, not to wavelength.
- **Beamwidth rule (VERIFIED two independent ways):** because the shading is 0.5 (-6 dB) at x≈0.64, the on-axis **-6 dB beamwidth ≈ 0.64 * (total cap arc angle) = 0.64 * 2*theta_0**. I confirmed this from the *exact* Legendre P_nu too: P_nu(cos(x*theta_0)) crosses 0.5 at x = 0.634–0.642 for cap half-angles 20–50deg. (e.g. 40deg total cap -> ~25.6deg beamwidth.)
- **Truncation:** truncating/re-expanding the shading at the **-12 dB** point (outermost drivers at -12 dB instead of -inf) widens the effective beamwidth to **≈0.78 * arc angle** (-12 dB sits at x≈0.82, and 0.64/0.82≈0.78). Keele's discretization study found a **3-dB stepped approximation maintained out to -12 dB** does not significantly degrade pattern control vs the continuous curve.
- **Cutoff frequency [HEURISTIC — state as scaling, not exact]:** constant-beamwidth holds **above** a low cutoff set by the **total arc length / array height L**: f_low ∝ c/L (longer array -> lower cutoff; doubling array height lowers cutoff one octave — confirmed by Keele's ground-plane mirror trick that doubles effective length and extends control an octave). **Driver spacing s sets the UPPER limit** (grating-lobe onset, roughly f_high ~ c/(2s)). I measured the lower-cutoff behavior empirically (see A.4): for R=0.5 m, 40deg cap, 41 drivers, the beamwidth converges to its constant value above ~4–8 kHz. Do not present f_low as an exact closed form; the inverse-to-arc-length scaling is the defensible claim.

### A.4 Minimal CBT validation case in BeamSimII (RECIPE — VERIFIED it works)
I synthesized this and it reproduces constant beamwidth:
1. **Geometry:** N≈41 monopole-ish drivers on a circular arc, radius R (≈0.5 m), spanning total arc 2*theta_0 (≈40deg, so theta_0=20deg). Drivers at true 3-D positions on the arc — **H carries the curvature time-of-flight automatically**.
2. **Weights:** **REAL** amplitude weights w_m = U(x_m), x_m = (element arc-angle from center)/theta_0. **No added phase** — the curved geometry already supplies the focusing delay. (This keeps the V-test cleanly decoupled from the delay-sign question; see CARDINAL-RULE note below.)
3. **Forward model:** P(f, dir) = sum_m w_m * H[m,f,dir] (the GLL complex sum). For a self-contained analytic sanity check without BEM, use monopole element factor in engineering convention: `EF_m(u) = exp(-j k (u . r_m))`.
4. **Acceptance:** sweep f; measure -6 dB beamwidth of |P(f, dir)| in the arc plane. Assert it **converges to ~0.64*2*theta_0** (±~2deg) and is **flat vs frequency above cutoff** (e.g. std of beamwidth over 4–12 kHz < 2deg).
   - My run (R=0.5m, 40deg cap, 41 mono): beamwidth = 180deg @500Hz, 112deg @1k, 50deg @2k, **28deg @4k, 24.5deg @8k, 25deg @12k** vs predicted 25.6deg. Constant above ~4 kHz. This is the V-test shape.

### A.5 Straight-line / flat-panel CBT via delays + Multi-CBT
- **Flat-panel emulation of the arc:** mount drivers on a straight line at y_m from center; add a per-driver **delay** equal to the sagitta of the arc divided by c. Exact: `tau_m = (R - sqrt(R^2 - y_m^2))/c`; **paraxial approximation `tau_m ≈ y_m^2/(2*R*c)`** (Keele, AES Paper 5653, 2002). In the engineering convention this is a complex weight phase `exp(+j*w*tau_m)` (delay -> +jw*tau; see PART B sign note). Keep the SAME Legendre amplitude U(x_m).
- **Power-response note:** delay-derived (frequency-independent-DSP) CBT has **half the LF power-response rolloff** of a true curved cabinet (3 dB/oct vs 6 dB/oct).
- **Multi-CBT / overlapped shading (AES, "Directivity-Customizable... Overlapped Shading"):** several CBT sub-apertures with overlapping Legendre shadings are summed to synthesize a chosen directivity-vs-frequency, decoupling beamwidth from physical array length.

---

## PART B — First-Order Cardioid Family (the clean CI-safe analytic anchors)

### B.1 The pattern family
General first-order axisymmetric pattern (theta from the look axis):
```
T(theta) = alpha + (1 - alpha) * cos(theta),   alpha in [0, 1]
```
(Some texts write a0 + a1*cos with a0+a1=1; alpha == a0.)

### B.2 Two-element delay-and-sum realization — ENGINEERING CONVENTION (sign-VERIFIED)
Endfire pair on z-axis, look direction +z, theta from +z. **Front** monopole at origin, **rear** monopole at z = -d. Electronic delay T_e applied to the rear channel; differential (subtract) structure:
```
y(theta, f) = w_f * 1  -  w_r * exp(+j*w*T_e) * exp(+j*k*d*cos(theta))
```
where, in the engineering convention:
- monopole element factor at r_m for far-field dir u: **`exp(-j k (u . r_m))`** (NOT +j; derived from |R-r_m| ≈ R - u.r_m so exp(+jk|R-r_m|)=exp(+jkR)exp(-jk u.r_m)). Rear at z=-d, u_z=cos(theta) -> rear EF = exp(+j k d cos theta).
- a pure time delay tau multiplies the spectrum by **`exp(+j*w*tau)`** (delay -> +jw*tau in exp(-jwt) convention). **This is the opposite sign from the physics convention** and is the single most error-prone line; my first attempt had it backwards (null landed at 0deg).
- The **subtract** (minus sign on the rear term) is required: I verified that `front - rear*exp(+jw*d/c)` gives a perfect null at theta=180deg and peak at 0deg; a `+` sign or wrong delay sign flips the null to 0deg.

**Mapping to alpha (VERIFIED):**
```
alpha = T_e / (T_e + d/c)     <=>     T_e = (alpha/(1-alpha)) * (d/c)
```
- alpha = 0 (T_e=0): dipole/figure-8. alpha = 0.5 (T_e=d/c): cardioid. 0<alpha<0.5: hyper/super. 0.5<alpha<1: sub-cardioid.
- I confirmed the realized delay-sum |y| matches the analytic |alpha+(1-alpha)cos theta| to **1.7e-5** at kd<<1 (d=3mm, 300Hz), and that the realized null angle equals the analytic null for every member (table below).
- **kd rolloff / EQ:** the differential output rolls off ~ as |2 sin(...)| ≈ k*d*(1+...) at low frequency, i.e. **+6 dB/oct** sensitivity slope; realizing a flat on-axis response requires a **-6 dB/oct LF boost (integrator-type EQ)**. Mark the exact EQ as out-of-scope unless realizing terminal response.

### B.3 Null angle, directivity factor Q, DI — EXACT ANALYTIC ANCHORS (all VERIFIED numerically)
Null angle (where T=0): `cos(theta_null) = -alpha/(1-alpha)` (real null only if alpha <= 0.5).

**Directivity factor closed form (VERIFIED vs full sphere integral):**
```
Q = 3 / (4*alpha^2 - 2*alpha + 1)        DI = 10*log10(Q) dB
```
Front-to-back ratio: FBR(alpha) = [integral_front |T|^2] / [integral_back |T|^2].

| Pattern        | alpha            | Q       | DI (dB)  | null angle | notes |
|----------------|------------------|---------|----------|------------|-------|
| Omni           | 1.0              | 1.000   | 0.000    | none       | RE=1 |
| Subcardioid    | ~0.7             | ~1.97   | ~2.9     | none (cos=-2.33) | between omni & cardioid |
| Cardioid       | 0.5              | 3.000   | **4.771**| **180.0deg** | null at rear |
| Supercardioid  | (sqrt(3)-1)/2 = **0.36603** | 3.732 | **5.719**| **125.26deg** | **max front-to-back ratio** (FBR=13.93 ≈ 11.4 dB) |
| Hypercardioid  | 0.25             | 4.000   | **6.021**| **109.47deg** | **max DI** (cos^-1(-1/3)) |
| Dipole (fig-8) | 0.0              | 3.000   | 4.771    | 90.0deg    | |

Key verified facts to anchor pytest tolerances:
- **Cardioid DI = 4.77 dB, Q=3, null exactly 180deg.**
- **Hypercardioid DI = 6.02 dB (the maximum over all first-order), Q=4, null = cos^-1(-1/3) = 109.47deg.** Optimum alpha=1/4 follows from minimizing the denominator 4a^2-2a+1.
- **Supercardioid alpha = (sqrt(3)-1)/2 = 0.36603 (max FBR, NOT max DI), DI = 5.72 dB, Q=3.732, null = 125.26deg, FBR = 13.93 (≈11.4 dB).**
- These DI/Q/null numbers are **convention-independent** (computed from the real pattern T(theta)); the engineering sign only enters when you *realize* the pattern from element delays or match against H.

### B.4 Minimal cardioid V-test in BeamSimII
1. Two monopole "drivers" at known 3-D positions (endfire spacing d, e.g. 3–20 mm) in H, or use the analytic monopole EF.
2. Set complex weights w_f=1, w_r = -exp(+j*w*(alpha/(1-alpha))*(d/c)) (engineering sign).
3. P(f,dir) = sum_m w_m H[m,f,dir]. Assert: (a) null at the analytic theta_null within ~0.5deg; (b) DI computed via Lebedev quadrature = 4.77/5.72/6.02 dB within ~0.1 dB for alpha=0.5/0.366/0.25; (c) on-axis (theta=0) is the peak. **The DI integral uses the Lebedev weights** (DI = |P(0)|^2 / (sum_i wq_i |P(dir_i)|^2 / sum_i wq_i)).

---

## CARDINAL-RULE / convention crib (put in the V-test docstrings)
- **Element factor (eng. conv.):** `exp(-j k (u . r_m))`. **Delay tau:** `exp(+j w tau)`. Physics convention is the conjugate of both — do not mix.
- **Curved CBT V-test:** drivers at true positions -> **real** Legendre weights (H carries curvature delay). **Flat-panel CBT:** add `exp(+j w * y_m^2/(2 R c))` curvature delay. Same +jw sign as the cardioid.
- **Never re-zero a driver's phase.** All H share the global origin; the cardioid null at 180deg and the CBT focusing both depend on the preserved inter-driver time-of-flight. A symmetric beamwidth test will NOT catch a global sign flip — the cardioid null test will.

### Sources
- (primary) Rogers & Van Buren, A new approach to a constant beamwidth transducer, JASA 64(1) 1978 — https://asa.scitation.org/doi/10.1121/1.381954
- (primary) Van Buren et al., Experimental constant beamwidth transducer, JASA 73(6) 1983 — https://pubs.aip.org/asa/jasa/article-abstract/73/6/2200/628387/Experimental-constant-beamwidth
- (primary) Keele, The Application of Broadband Constant Beamwidth Transducer (CBT) Theory to Loudspeaker Arrays (AES 109th Conv., 2000, Paper 1) — https://keele-omholt-technologies.com/papers/Keele-CBT-Paper-1-Sept.-2000-Application-of-CBT-Theory-to-Loudspeaker-Arrays.pdf
- (primary) Keele, Practical Implementation of CBT Loudspeaker Circular-Arc Line Arrays (AES, Paper 5863, 2003) — https://www.researchgate.net/publication/228902314_Practical_Implementation_of_Constant_Beamwidth_Transducer_CBT_Loudspeaker_Circular-Arc_Line_Arrays
- (primary) Keele, Implementation of Straight-Line and Flat-Panel CBT Loudspeaker Arrays Using Signal Delays (AES, Paper 5653, 2002) — https://www.aes.org/e-lib/download.cfm?ID=11236
- (primary) Keele, Full-Sphere Sound Field of CBT Loudspeaker Line Arrays (AES, Paper 3) PDF — https://audioartistry.com/Papers/CBT%20Paper%203%20Full-Sphere%20Sound%20Field%20of%20CBT%20Arrays.pdf
- (primary) Keele, Design of CBT Loudspeaker Line Arrays for Sound Reinforcement (AES, Paper 10, 2016) PDF — https://keele-omholt-technologies.com/papers/Keele-CBT-Paper-10-Sept.-2016-Design-of-CBT-Loudspeaker-Line-Arrays-for-Sound-Reinforcement.pdf
- (primary) Directivity-Customizable Loudspeaker Arrays Using CBT Overlapped Shading (AES Conv. paper, elib 18034) — https://secure.aes.org/forum/pubs/conventions/?elib=18034
- (secondary) dbkeele.com — Constant Beamwidth Transducers overview (author's site) — https://dbkeele.com/constant-beamwidth-transducers/
- (primary) Buck, First Order Differential Microphone Arrays for Automotive Applications (IWAENC 2001) — https://www.iwaenc.org/proceedings/2001/main/data/buck.pdf
- (primary) Benesty & Chen, Study and Design of Differential Microphone Arrays (Springer, ch.3 First-Order) — https://link.springer.com/chapter/10.1007/978-3-642-33753-6_3
- (primary) On the Design and Implementation of Higher Order Differential Microphones (KCL/elucidare PDF) — https://www.elucidare.co.uk/assignments/project_KCLaudio/On%20the%20Design%20and%20Implementation%20of%20Higher%20Order%20Differential%20Microphones.pdf
- (primary) Elko et al., Steerable and variable first-order differential microphone array (US6041127A) — https://patents.google.com/patent/US6041127A/en
- (secondary) doctorproaudio — Microphone pickup polar patterns (cardioid/super/hyper/sub) reference values — https://www.doctorproaudio.com/content.php?2321-microphone-pickup-patterns

### Decisions implied
- Implement TWO Phase-2 V-tests: (V-CBT) a curved monopole/piston array with REAL Legendre amplitude weights asserting frequency-independent -6dB beamwidth ≈ 0.64*(2*theta_0) above cutoff; (V-CARDIOID) a two-element endfire array asserting analytic null angle (±0.5deg) and DI (±0.1 dB) for cardioid/super/hyper. The cardioid test is the CI-safe primary anchor (cheap, exact, convention-sensitive).
- Provide a shared analytic reference module exposing: Q(alpha)=3/(4a^2-2a+1), DI_dB(alpha), null_angle(alpha)=acos(-a/(1-a)), pattern T(theta)=a+(1-a)cos(theta), and named constants cardioid/super/hyper/sub/dipole with their (alpha,Q,DI,null) tuples — used as pytest fixtures.
- Encode the engineering-convention element factor exp(-j k (u.r_m)) and delay factor exp(+j w tau) ONCE in a tested helper; the cardioid V-test must assert the null lands at 180deg (not 0deg) as the guard against a global sign flip — the existing symmetric beamwidth tests cannot catch that error.
- For the CBT weight generator, default to Keele's polynomial U(x)=1+0.066x-1.8x^2+0.743x^3 (real amplitude) with an optional exact-Legendre P_nu mode (nu from P_nu(cos theta_0)=0, nu0≈2.4048/theta_0-0.5 as initial guess; P_nu via hyp2f1). Offer a 3-dB-stepped + -12dB-truncation discretization option matching Keele's practical recommendation.
- Keep CBT weights REAL for the curved-array V-test (curvature time-of-flight lives in H, honoring the cardinal phase-origin rule). Only the flat-panel/straight-line CBT realization adds a complex curvature-delay phase exp(+j w * y_m^2/(2 R c)) (paraxial) or exp(+j w (R-sqrt(R^2-y^2))/c) (exact), using the SAME +jw engineering sign as the cardioid delay.
- Do NOT hardcode an exact CBT lower-cutoff frequency formula; expose it as a scaling (f_low ∝ c/L_arc, f_high ~ c/(2*spacing)) and let the V-test assert beamwidth-vs-frequency *flatness* (std < ~2deg) over a band empirically above cutoff rather than asserting a single closed-form cutoff value.
- Compute DI in all V-tests via the dataset's Lebedev quadrature weights: DI = |P(look)|^2 / (sum_i wq_i |P(dir_i)|^2 / sum_i wq_i) — reuse the existing V-4 power/DI machinery so the cardioid 4.77/5.72/6.02 dB anchors validate the quadrature path too.

### Open questions
- Exact verbatim text of Keele Eq.(1)/Eq.(3) could not be machine-extracted (FlateDecode PDFs, no poppler in this read-only env). The polynomial U(x) and the 0.64 relation are cross-verified numerically and via the exact Legendre derivation, but a human should confirm the precise coefficients against the AES Paper 1 PDF before locking the test reference.
- The exact paraxial-vs-exact curvature-delay formula and the precise CBT lower-cutoff expression in Keele Paper 10 / Paper 5653 were not extractable; the y^2/(2Rc) paraxial form is standard and confirmed in secondary sources but the paper's exact normalization (reference point, sign relative to array center) should be checked against the primary PDF.
- Whether Phase-2 should validate against the Multi-CBT / overlapped-shading constant-directivity-vs-frequency target (more complex) or stop at single-aperture CBT for the V-test. The single-aperture curved-array test is sufficient to prove the pipeline; overlapped shading is a richer but optional target.
- Supercardioid alpha is defined here as the max-front-to-back-ratio value (sqrt(3)-1)/2 = 0.36603; some references define 'supercardioid' loosely. The V-test should pin the definition explicitly (max FBR) to avoid an ambiguous DI anchor (5.72 dB).

### Adversarial verification verdicts
  - [confirmed] First-order directivity factor Q = 3/(4*alpha^2 - 2*alpha + 1), DI = 10log10(Q); cardioid(alpha=0.5) Q=3 DI=4.771 dB, hypercardioid(alpha=0.25) Q=4 DI=6.021 dB, supercardioid(alpha=(sqrt3-1)/2=0.36603) Q=3.732 DI=5.719 dB; verified by closed form AND direct sphere integral of (alpha+(1-alpha)cos theta)^2.
  - [confirmed] First-order null at cos(theta_null) = -alpha/(1-alpha): cardioid 180deg, hypercardioid 109.47deg (=acos(-1/3)), supercardioid 125.26deg; subcardioid (alpha~0.7) and omni have no real null.
  - [confirmed] Engineering convention (exp(-jwt), outgoing exp(+jkr)): far-field monopole element factor exp(-j k (u.r_m)); time delay tau multiplies spectrum by exp(+j w tau). Endfire cardioid nulling at theta=180deg is y = front - rear*exp(+j w d/c), front at origin, rear at z=-d. Using +j element-factor sign or a + (sum) structure puts the null at 0deg.
  - [confirmed] Keele CBT Legendre shading polynomial U(x)=1+0.066x-1.8x^2+0.743x^3 (x=theta/theta_0, U=0 for x>1) satisfies U(0)=1, U(1)=0.009, crosses 0.5 (-6 dB) at x=0.6398, reproducing 'beamwidth ~ 0.64 * total cap arc angle'. -> CORRECTION: Coefficients and endpoint values: CONFIRMED verbatim (US12,445,768B2). The x=0.6398 / '0.64*arc' relation: correct but it is an INFERRED numerical property of the polynomial, NOT a statement in the cited Keele source; the patent's own stated geometric rule is the -12 dB-truncated 0.7776 (78%) ratio. Provenance 'CBT Paper 1 / AES 109th 2000' is unconfirmed; attribute to the Keele CBT patent lineage instead. U(1)=0.009 corresponds to ~-40.9 dB, not exactly the -40 dB the finder rounds to.
  - [confirmed] Exact CBT shading is the Legendre function P_nu(cos theta) over 0<=theta<=theta_0, non-integer degree nu fixed by P_nu(cos theta_0)=0 (rim=first zero); nu satisfies Mehler-Heine (nu+0.5)*theta_0 ~ 2.4048 (first J0 zero), i.e. nu ~ 2.4048/theta_0 - 0.5; exact P_nu also crosses 0.5 at x=0.634-0.642.
  - [confirmed] Curved monopole array (R=0.5m, 40deg total cap, 41 elements), REAL weights w_m=U(x_m), P=sum w_m exp(-jk u.r_m): -6dB beamwidth converges to ~25deg (=0.64*40deg), frequency-independent above ~4 kHz, widening to 180deg at 500Hz.
  EXTRA: All six claims are confirmed; the only material correction is to claim 4's provenance and the framing of the "0.64 rule." Three points the implementer should carry forward:

1. SOURCE for the polynomial: The exact coefficients (1, 0.066, -1.8, 0.743) are confirmed VERBATIM only in US Patent 12,445,768 B2 (and the polynomial family is Keele-CBT lineage). The finder's attribution to "CBT Paper 1, AES 109th, 2000" is UNVERIFIED (could not open that PDF). Cite the patent, not the 2000 paper.

2. theta_0 DEFINITION subtlety (load-bearing): The patent defines theta_0 = HALF the arc angle, and x = theta/theta_0, so x runs 0..1 from center to rim. The "0.64 * total cap arc angle" rule means 0.64*(2*theta_0). The patent's OWN stated geometric rule is different: the -12 dB-TRUNCATED shading gives beamwidth = 0.7776 * arc angle (~78%), with worked example "39deg arc -> 30deg beam." So there are TWO regimes: untruncated (~0.64, the U crosses 0.5 at x=0.6398) and -12dB-truncated (~0.78). The finder conflated these slightly -- both are correct but apply to different truncation choices. Pick the regime deliberately for the V-test acceptance threshold.

3. The "0.64 rule" and "U crosses 0.5 at x=0.6398" are INFERRED numerical properties, NOT statements found in any primary source I could read; they are verified by computation only. Likewise the Mehler-Heine relation (nu+0.5)*theta_0 ~ 2.4048 is INFERRED-from-theory + numerically confirmed; I could not read the Rogers & Van Buren 1978 full text (paywalled), so do NOT cite a specific R&VB closed-form (e.g. "nu = 0.5(4.81/alpha - 1)") as primary-source-verified.

4. A SECOND, distinct published polynomial exists: mu(x)=1+0.0561x-1.3017x^2+0.457x^3 in US11,889,263 (Space-Shaded CBT), described as the ~30deg-truncated fit. Not a contradiction, but worth knowing two official fits circulate.

5. Supercardioid is DEFINITION-DEPENDENT. Claim 2 uses the max-front-to-back-ratio definition -> alpha=(sqrt3-1)/2~0.366, null 125.3deg, DI 5.7 dB. Microphone-vendor "front pickup angle ~115deg" numbers are the -6 dB front beamwidth, a different quantity -- not a contradiction. Document which 'supercardioid' the target uses so a future reviewer isn't ambushed.

6. The cutoff frequency in claim 6 (~4 kHz) is array-length dependent (f_low ~ c/L scaling), correctly flagged by the finder as HEURISTIC, not a closed form. The V-test should assert frequency-independence ABOVE a measured cutoff, not at a hardcoded 4 kHz.


---

## TOPIC: Reusable open-source libraries and reference implementations for the BeamSimII Phase-2 transmit beamformer + per-driver filter designer (license, maturity, reuse verdict)

### Report
# Phase-2 Beamformer + Filter Designer: Library Reuse Survey

## 0. The spine that connects everything (read first)

BeamSimII's per-driver tensor column `H[:, f, r]` (one complex value per driver m, fixed
frequency f, sphere direction r) **is exactly the "steering vector" `d(r)` of every
beamforming formulation below.** It is already (a) full-3D, (b) referenced to one common
phase origin (so relative time-of-flight is encoded in the phase), and (c) sampled on a
Lebedev grid with quadrature weights that travel with the dataset. Consequences:

- We **reuse the math** (delay-and-sum, MVDR/LCMV algebra, the Luo GRQ/QCQP), not anybody's
  steering-vector *code*. Every receive-array library computes `d(r)` from mic geometry +
  an assumed free-field Green's function; we instead read `d(r)=H[:,f,r]` straight from the
  BEM solve, which already contains diffraction, baffle, enclosure, and driver T/S coloring.
- Sphere integrals (covariances, radiated power, directivity) are **Lebedev quadrature
  sums** over the weights `w_quad(r)` already in the dataset:
  `∫_S g(r) dΩ ≈ 4π · Σ_r w_quad(r) · g(r)` (weights normalized to Σ w_quad = 1).

Forward model (your GLL complex summation), per frequency f:
`P(f, r) = Σ_m w_m(f) · H[m, f, r] = w(f)^H · conj(H[:, f, r])`  — or equivalently
`w(f)^T · H[:, f, r]`; pick ONE convention and keep it. (Below I write `d = H[:,f,r]`,
`P = w^T d`, so the matched/DS weight is `w = conj(d)`.)

---

## 1. The methodological centerpiece — Luo, "Constant Directivity Loudspeaker Beamforming" (EUSIPCO 2024 / arXiv:2407.01860)

This is a **transmit** (loudspeaker) beamformer paper, by Yuancheng Luo (Amazon), purpose-built
for exactly our problem: small heterogeneous driver arrays, per-transducer operating-band /
power limits, constant directivity vs frequency. **Reference-only (no code released), but its
math is the recommended core of Phase-2.** Verified equations:

- **Generalized directivity index (GDI)** = generalized Rayleigh quotient
  `G(w) = (w^H A w) / (w^H R w)`, with N×N Hermitian covariances (N = #drivers):
  - `A = E_{r~f_A}[ d(r) d(r)^H ]`  ("accept"/listening-window covariance)
  - `R = E_{r~f_R}[ d(r) d(r)^H ]`  ("reject"/everything-else covariance)
  - `f_A`, `f_R` are direction probability-density (weighting) functions over the sphere.
    In our implementation these expectations become Lebedev sums:
    `A = Σ_r f_A(r) w_quad(r) d(r) d(r)^H`, `R = Σ_r f_R(r) w_quad(r) d(r) d(r)^H`.
    Setting `f_R` = uniform over the full sphere makes `w^H R w` proportional to total
    radiated acoustic power, so `G` is a true directivity factor.
- **Constant-directivity constraint** (target directivity τ, real scalar): define the
  **indefinite** Hermitian matrix `D = A − τR`; "GDI = τ" becomes the single quadratic
  equality `w^H D w = 0`.
- **MSCD** (Maximum Sensitivity Constant Directivity): maximize on-axis sensitivity
  `|c^H w|` (with `c = d(r_steer)`, the steering direction's column) subject to `w^H D w = 0`
  and `c^H w = 1`. **Solved in closed form** — a quadratic-constrained least-squares solution
  for indefinite D (a secular-equation / generalized-eigenvalue solve).
- **MECD** (Maximum Efficiency Constant Directivity): maximize acoustic efficiency
  `w^H C w` (C = power-into-region matrix) s.t. `w^H D w = 0`. Solved by **Algorithm 1 =
  projected ascent**: `w_k ← w_{k-1} + α C w_{k-1}`, project onto `{w: w^H D w = 0}`,
  normalize; converges in ~5 iterations with differential multipliers (~50 naive).
- **Frequency regularization (GRPQ)** — this is how WNG / per-driver-band control enters
  the *analytic* route (NO separate SOCP needed): penalty `Γ Σ` added to the denominator,
  `Γ = diag(diag(R))`, `Σ = diag(σ_1..σ_N)`, `0 ≤ σ_n ≤ ∞`. Equivalent bounded form via
  `Λ = (Σ+I)^{-1/2} = diag(λ_n)`, `0 ≤ λ_n ≤ 1`; change of variable `y = Λ^{-1} w` gives
  `R̂ = Λ R Λ + Γ(I − Λ²)`. Driving `λ_n → 0` outside driver n's operating band forces
  `|w_n| → 0` — i.e. this regularizer **generalizes crossover filters and provides WNG /
  robustness control simultaneously (it is diagonal loading / Tikhonov on R).**
- An **SDP relaxation** (Eq. 19: solve PSD `W ⪰ 0`, recover rank-1 `W = w w^H`) is given as an
  *alternative* baseline — the only branch that would actually need a conic solver.

Solver primitives this implies: `scipy.linalg.eigh(A, R)` (generalized symmetric/Hermitian
eigenproblem — both inputs Hermitian; returns real eigenvalues, the bound of `G`), plus a
1-D secular-equation root find (Brent / Newton) for the constrained solution, plus a small
projected-ascent loop for MECD. **No external optimizer required for the core.**

---

## 2. Library-by-library verdicts

### (1) pyroomacoustics (LCAV) — REFERENCE-ONLY
- License **MIT** (mature, actively maintained, v0.10.x). pip-installable.
- `beamforming.py` gives clean, readable **weight-algebra you should mirror**:
  - `rake_delay_and_sum_weights`: `w = (1/M)/(K+1) · Σ_k W[:,k]` (sum of steering columns).
  - `rake_one_forcing_weights` (LCMV/distortionless): `R_nq = R_n + a_bad a_bad^H`,
    `w = R_nq^{-1} A_s (A_s^H R_nq^{-1} A_s)^{-1} b`. This is the generic LCMV form you'll
    reuse for "unit response toward steer, nulls toward reflections."
  - MVDR/max-SINR follow the `w = R^{-1} a / (a^H R^{-1} a)` pattern.
- **Why not a dependency:** (a) steering vector is hard-coded **2D-only** and uses the
  *receive*, opposite-outgoing-sign Green's function `exp(−jω D/c)` (negative exponent on
  mic-to-source distance) — BeamSimII uses `exp(+jkr)` outgoing (engineering conv.), so the
  geometry/steering code is **not** drop-in. (b) It's built around `MicrophoneArray`/STFT
  recording, not a transmit design. (c) We already HAVE `d` from `H`, so the only valuable
  part is the 5–10 lines of linear algebra per method — copy the structure, not the package.
- Convention note: pyroomacoustics' own `far_field_weights` uses `exp(+2πj f·proj/c)`
  (POSITIVE) while its steering vector uses negative — i.e. the DS weight is the conjugate of
  the steering vector. This conjugate-match is the universally safe pattern: `w = conj(d)`.

### (2) Acoular — REFERENCE-ONLY (concept transfer only)
- License **BSD-3-Clause** (very permissive), mature, actively maintained.
- It is **receive-side source-mapping** (CSM/beamforming over a focus grid). Transferable
  *concepts/data structures*: the `SteeringVector` abstraction (grid × mic transfer matrix),
  the cross-spectral-matrix (CSM) as the data object, and the family of steering-vector
  normalization "formulations I–IV." None of its solver code applies to transmit design.
- Verdict: skim for API-shape inspiration (how to package a `SteeringVector`/`Grid` object);
  do not depend.

### (3) sfs-python (Sound Field Synthesis Toolbox, sfstoolbox.org) — REFERENCE-ONLY (likely skip)
- License **MIT**. Modules confirmed: `sfs.fd.{source,wfs,nfchoa,sdm,esa}`,
  `sfs.td.{source,wfs,nfchoa}`, `sfs.array`, `sfs.tapering`, `sfs.util`, `sfs.plot2d/3d`.
- **Driving functions are ANALYTIC only** — WFS, NFC-HOA, SDM (spectral division), ESA. There
  is **NO numerical least-squares / pressure-matching / sound-field-control solver module**
  (no `sfs.fd.pm` or equivalent). So it does **not** give us the LS sound-field-control
  machinery the task asked about.
- What's reference-worthy: `sfs.array` (loudspeaker-array geometry containers), `sfs.tapering`
  (window/edge-taper functions — useful to taper driver weights and suppress sidelobes), and
  worked driving-function math if we ever want a WFS sanity baseline. Verdict: borrow tapering
  ideas; otherwise skip as a dependency.

### (4) SH-domain libs: spaudiopy / sound_field_analysis-py / spatial-audio-resources — REFERENCE-ONLY (pip-dependency ONLY if we commit to the SH route)
- **spaudiopy** (chris-hld): **MIT**, maintained, Python 3.9+. Has `spaudiopy.sph` (SH
  transforms incl. `sph_harm_all`, `sph_harm_large` for order >84), grids, decoders.
- **sound_field_analysis-py** (Chalmers): **MIT**. Spherical-array analysis, `sph` module,
  radial filters.
- These matter only **if** we choose to project `H` onto a spherical-harmonic basis (using the
  Lebedev weights for the SHT) and design the beam in the SH domain (covariance interpolation
  across frequency, smooth steering, order-truncation as a constant-DI knob). For a handful of
  drivers the **direct-grid** Luo formulation is simpler and avoids SHT order/aliasing issues,
  so SH is optional. Note: BeamSimII already uses `scipy.special.sph_harm_y` (per memory) —
  for plain SH evaluation we may not need a new dep at all. Verdict: reference-only;
  promote to pip-dependency only if we deliberately adopt the SH design route.

### (5) Optimization — cvxpy + a conic solver = OPTIONAL pip-dependency; scipy.linalg/optimize = CORE
- **scipy.linalg.eigh(A, R)**: the workhorse. Solves the generalized Hermitian eigenproblem
  for the GRQ bound and the constrained solutions. Part of SciPy (BSD-3), already a dependency.
- **scipy.optimize** (Brent/`brentq`, Newton): 1-D secular-equation root find for the
  quadratic-equality QCQP; `scipy.optimize.minimize` if we ever want a general fallback. Core.
- **cvxpy** (license **Apache-2.0**, very mature, de-facto standard DSL): needed ONLY for the
  optional SDP relaxation (Luo Eq. 19) or if we add *multiple* convex constraints that don't
  reduce to one quadratic equality — e.g. simultaneous per-driver power caps + WNG floor as an
  SOCP/QCQP. For the single-quadratic-constraint MECD/MSCD, cvxpy is unnecessary.
  - **Solver choice / license trap:** cvxpy historically pulled in **ECOS, which is GPL** —
    a contamination risk for a self-contained MIT/BSD-style app. Use **Clarabel**
    (**Apache-2.0**, Rust/Python conic IPM, handles QP/SOCP/SDP, no epigraph reformulation
    for quadratic objectives) or **SCS** (MIT). Avoid bundling ECOS. cvxpy ≥ recent versions
    have moved off ECOS-by-default for exactly this reason.
- WNG-constrained robust beamforming in the analytic route = **diagonal loading**
  `R ← R + ε I` (== Luo's `Γ(I−Λ²)` term). State this equivalence in code comments so a
  future maintainer doesn't reach for an SOCP when a scalar ε suffices.

### (6) scipy.signal — FILTER REALIZATION: pip-dependency (already present), but with a sharp caveat
- For **magnitude-only** sub-targets: `firwin2` (arbitrary magnitude via frequency sampling),
  `firls` (weighted least-squares, linear phase), `remez` (equiripple/minimax). `freqz` for
  verification. `iirfilter`/`bilinear`/`sosfreqz` for biquad realization; `tf2sos`/`zpk2sos`.
- **CAVEAT (load-bearing):** the designed weights `w_m(f)` are **complex with non-trivial,
  non-linear phase** (they encode each driver's true time-of-flight to preserve steering).
  `firls`/`firwin2`/`remez` produce **linear-phase / real-symmetric** filters and **cannot**
  match an arbitrary complex target — they handle only the magnitude sub-problem. SciPy has
  **no `invfreqz`** (that's MATLAB). So for the actual per-driver filter:
  - **Complex FIR via frequency sampling + IFFT**: sample `w_m(f)` on the FFT grid (interp the
    F BEM frequencies onto a dense linear grid), enforce conjugate symmetry, IFFT to taps,
    window. This realizes arbitrary magnitude AND phase but is **non-causal** → needs a
    **bulk group delay**.
  - **IIR fit (`invfreqz` analog)**: Levy / Sanathanan–Koerner least-squares rational fit, or
    `scipy.signal.invres`-style assembly; no first-class SciPy function — port or vendor a
    small routine.
- **Cardinal-rule constraint on the bulk delay:** any added latency to make the complex FIR
  causal MUST be the **same delay applied to all drivers**. A common delay shifts the absolute
  phase origin (harmless). A *per-driver* delay alters *relative* time-of-flight and silently
  re-steers the beam → violates the cardinal rule. Bake "single global FIR bulk delay" into
  the realizer's contract; assert it.

### (7) Reference demos — Grassin & MathWorks — REFERENCE/SKIP
- **CGrassin/acoustic_beamforming** (GitHub): a hobbyist **narrowband phase-offset
  beamsteering** demo (Arduino tone array). Its `beamforming_pattern_gen.py` plots
  `|Σ_n a(n) exp(jφ_n)|` for a uniform line array — pedagogically nice for the **array-factor
  visualization** and grating-lobe (`d ≤ λ/2`) intuition, but it is single-frequency,
  free-field point-source, ULA-only. **Skip for code; reference for the viz idea.** (License
  unconfirmed — treat as non-reusable regardless.)
- **MathWorks "Making All the Right Noises"**: a Philips-Research *marketing/technical
  article*, not code. Method = simulate the sound field of an arbitrary loudspeaker array and
  optimize driving filters to create different SPL zones (personal sound / contrast control).
  Conceptual confirmation that "design filters from a simulated transfer function" is the
  right shape; the actual optimization (acoustic-contrast / pressure-matching) overlaps Luo's
  GRQ. **Reference-only (no reusable artifact).**

---

## 3. Recommended Phase-2 software stack (net)

- **Core math:** implement Luo MECD/MSCD + GRQ directly on `H` (no library copies its
  formulas). Primitives from **SciPy** (`scipy.linalg.eigh(A,R)`, `scipy.optimize.brentq`) —
  already a dependency.
- **Weight-algebra reference:** pyroomacoustics `beamforming.py` (MIT) — read for the
  DS/LCMV/MVDR structure, transcribe ~10-line formulas, do **not** depend.
- **Optional convex extras:** `cvxpy` (Apache-2.0) + **Clarabel** (Apache-2.0) or **SCS**
  (MIT) ONLY for the SDP relaxation / multi-constraint SOCP variant. Never bundle ECOS (GPL).
- **Filter realization:** `scipy.signal` for magnitude/biquad parts; a small in-house
  complex-FIR (freq-sample + IFFT, common bulk delay) and optional IIR (SK/Levy) fitter.
- **Tapering / array containers / SH (optional):** borrow ideas from `sfs` (MIT) and
  `spaudiopy` (MIT); promote to deps only if we adopt SH-domain design.

All recommended deps are MIT/BSD/Apache-2.0 — compatible with a self-contained open-source
macOS app. The single license hazard is **ECOS (GPL)** sneaking in via cvxpy's default solver.

### Sources
- (primary) Luo, Constant Directivity Loudspeaker Beamforming (arXiv:2407.01860, HTML) — https://arxiv.org/html/2407.01860
- (primary) Luo, Constant Directivity Loudspeaker Beamforming (arXiv abstract, EUSIPCO 2024) — https://arxiv.org/abs/2407.01860
- (primary) Constant Directivity Loudspeaker Beamforming - Amazon Science — https://www.amazon.science/publications/constant-directivity-loudspeaker-beamforming
- (primary) pyroomacoustics beamforming.py source (LCAV, MIT) — https://raw.githubusercontent.com/LCAV/pyroomacoustics/master/pyroomacoustics/beamforming.py
- (primary) pyroomacoustics.beamforming module documentation (0.10.0) — https://pyroomacoustics.readthedocs.io/en/pypi-release/pyroomacoustics.beamforming.html
- (primary) Acoular GitHub (BSD-3-Clause) — https://github.com/acoular/acoular
- (primary) SFS Toolbox Python API/index (modules; MIT) — https://sfs-python.readthedocs.io/en/0.6.2/
- (primary) sfs-python GitHub (license: MIT) — https://github.com/sfstoolbox/sfs-python
- (primary) spaudiopy GitHub (chris-hld, MIT) — https://github.com/chris-hld/spaudiopy
- (primary) sound_field_analysis-py GitHub (Chalmers, MIT) — https://github.com/AppliedAcousticsChalmers/sound_field_analysis-py
- (primary) CVXPY GitHub (Apache-2.0) + ECOS/GPL dependency discussion (issue #2301) — https://github.com/cvxpy/cvxpy/issues/2301
- (primary) Clarabel solver (Apache-2.0, PyPI) — https://pypi.org/project/clarabel/0.6.0
- (primary) CVXPY solver features (ECOS/SCS/Clarabel/SDP/SOCP) — https://www.cvxpy.org/tutorial/solvers/index.html
- (tertiary) CGrassin/acoustic_beamforming (GitHub demo) — https://github.com/CGrassin/acoustic_beamforming
- (secondary) MathWorks, Making All the Right Noises: Shaping Sound with Audio Beamforming — https://www.mathworks.com/company/technical-articles/making-all-the-right-noises-shaping-sound-with-audio-beamforming.html

### Decisions implied
- Implement the Phase-2 beamformer core in-house from Luo's GRQ/QCQP math, treating H[:, f, r] directly as the steering vector d(r); do NOT add pyroomacoustics/acoular/sfs as runtime dependencies for the solver.
- Build covariances A = sum_r f_A(r) w_quad(r) H[:,f,r] H[:,f,r]^H and R = sum_r f_R(r) w_quad(r) H[:,f,r] H[:,f,r]^H using the dataset's Lebedev quadrature weights; expose f_A (listening window) and f_R (sphere/reject) as the user's beam-SHAPE specification, and r_steer (giving c = H[:,f,r_steer]) as the steering input.
- Use scipy.linalg.eigh(A, R) for the generalized eigenproblem and scipy.optimize.brentq for the secular-equation root find; implement MSCD as the closed-form solution and MECD as a small projected-ascent loop. Treat cvxpy+Clarabel as an OPTIONAL extra dependency, added only if/when the SDP relaxation or multi-constraint SOCP variant is implemented; never depend on ECOS.
- Implement WNG/robustness as scalar diagonal loading R <- R + epsilon*I (== Luo's regularization), and per-driver crossover/band-limiting via the diagonal Lambda weighting; document this so maintainers don't reach for an SOCP when a scalar suffices.
- Provide a filter-realization module: (a) magnitude/biquad parts via scipy.signal (firwin2/firls/freqz/zpk2sos), and (b) a custom complex-FIR realizer (interp w_m(f) to a dense FFT grid, enforce conjugate symmetry, IFFT, window) plus an optional IIR (Sanathanan-Koerner/Levy) fitter; SciPy has no invfreqz so this must be ported/vendored.
- Enforce a SINGLE global bulk group delay shared by all drivers in the FIR realizer (assert per-driver delays are equal); add a regression test mirroring the existing two-driver superposition / phase-origin guard to ensure realized filters preserve relative time-of-flight.
- Keep convention discipline: choose one of P = w^T H or w^H conj(H) and use it everywhere; the matched/DS weight is w = conj(d). DS and the d d^H covariances are sign-convention-robust (conj(H)*H = |H|^2), so the only hazard is importing a formula from a physics-convention (exp(+jwt)) source — flag any such import.
- Optionally borrow sfs.tapering-style window functions for sidelobe control and spaudiopy/sound_field_analysis-py only if an SH-domain design route is later adopted; otherwise reuse the existing scipy.special.sph_harm_y rather than adding an SH dependency.

### Open questions
- Direct-grid (Luo) vs spherical-harmonic-domain design: for a handful of drivers the direct-grid GRQ avoids SHT order/aliasing concerns, but SH may give smoother frequency interpolation and a natural constant-DI knob (order truncation). Needs a decision before committing dependencies.
- Exact role of cvxpy: confirm against the full Luo PDF whether MECD+MSCD closed-form/projected-ascent fully cover the intended feature set, or whether per-driver acoustic+electrical power caps + WNG floor require a multi-constraint SOCP (which would make cvxpy+Clarabel a genuine dependency, not optional).
- Filter target: FIR (linear bulk delay, arbitrary complex, longer latency) vs IIR/biquad (low latency, but harder arbitrary-complex fit and stability). Which does Phase-2 export prioritize, and what latency budget is acceptable for the common bulk delay?
- How to interpolate the F BEM frequency samples (coarse, possibly log-spaced) onto the dense linear FFT grid needed for IFFT-based complex FIR design without phase-unwrap artifacts at the common phase origin.
- Whether to regularize/smooth w_m(f) across frequency (the BEM grid is discrete) to avoid filters with excessive group-delay ripple; if so, does a per-frequency independent solve suffice or is a joint (frequency-coupled) optimization needed.

### Adversarial verification verdicts
  - [confirmed] BeamSimII's tensor column H[:,f,r] IS the steering vector d(r) for all beamforming formulations (Luo GRQ, DS, MVDR/LCMV); reuse the math not any library's steering-vector code; sphere integrals (A,R, power, directivity) are computed as Lebedev quadrature sums over the dataset's weights. -> CORRECTION: Imprecise on ONE convention-dependent subtlety the finder itself muddied (its §0 writes both P=w^T d with w=conj(d) AND w^H conj(H)): whether the steering vector is d=H[:,f,r] or its conjugate is fixed by the chosen forward-model convention (P=w^T d vs P=w^H d) under the engineering exp(-jωt) time convention. A and R MUST be assembled from the SAME choice of d (Hermitian outer product d d^H). 'H IS d' is correct only once that one convention is pinned; recommend P=w^T d (so d=H[:,f,r], DS weight w=conj(d)) and keep it everywhere.
  - [confirmed] Luo's MECD/MSCD are solved WITHOUT an external convex solver: MSCD has a closed-form quadratic-constrained-least-squares (secular-equation) solution for indefinite D=A-tau*R; MECD uses a ~5-iteration projected-ascent algorithm; SDP relaxation (Eq.19) is only an alternative baseline. Core primitives = scipy.linalg.eigh(A,R) plus a 1-D root find. cvxpy OPTIONAL. -> CORRECTION: Minor (does not change verdict): scipy.linalg.eigh(A,R) returns ASCENDING real eigenvalues, so the directivity BOUND is the LARGEST eigenvalue, and R must be positive-definite (holds when f_R uniform/full-rank). eigh gives the UNCONSTRAINED GDI bound; the constrained weight comes from the Eq.22/23 secular root-find, as the claim's 'plus a 1-D root find' already states. The finder's prose 'maximize |c^H w|' is the MVDR dual of Eq.20's 'minimize w^H w s.t. c^H w=1' — equivalent up to scale.
  - [partially-correct] WNG/robustness and per-driver operating-band control in the analytic route are achieved by diagonal loading / Tikhonov regularization of the reject covariance R (equivalently Luo's Gamma,Sigma,Lambda frequency-regularization), NOT by a separate SOCP white-noise-gain constraint. -> CORRECTION: The GRPQ regularizer (Γ=diag(diag(R)), Λ) provides per-driver operating-band/crossover control and conditioning with NO separate SOCP white-noise-gain constraint (this part holds). But it is NOT accurately described as 'diagonal loading / Tikhonov of R' for WNG: Γ=diag(diag(R)) shrinks toward R's diagonal (not toward εI), R̂ attenuates off-diagonals by λ_iλ_j (anisotropic, per-driver), and the paper does not characterize its purpose as white-noise-gain — it explicitly downplays WNG for heterogeneous arrays. Treat WNG/robustness as a goal the Γ/Λ shrinkage can serve, not as a paper-stated equivalence.
  - [confirmed] pyroomacoustics is MIT-licensed, mature, reference-only here: 2D-only steering vector using receive/opposite-sign Green's function exp(-jwD/c) conflicting with BeamSimII's engineering exp(+jkr); only its ~10-line DS/LCMV/MVDR weight-algebra (w=R^{-1}a/(a^H R^{-1}a)) is worth transcribing. -> CORRECTION: Add one verified nuance: within pyroomacoustics the signs are inconsistent — the steering vector uses NEGATIVE exp(-jωD/c) but far_field_weights uses POSITIVE exp(+2jπ f proj/c). So when transcribing, fix the sign convention deliberately rather than copying either verbatim; this strengthens (does not weaken) the claim's point that the geometry/steering code is not drop-in.
  - [confirmed] Linear-phase FIR designers (scipy.signal.firls, firwin2, remez) CANNOT realize complex non-linear-phase weights w_m(f); arbitrary-complex realization needs frequency-sample+IFFT FIR (or IIR/invfreqz fit), which is non-causal and needs a bulk group delay COMMON to all drivers — a per-driver delay re-steers the beam and violates the cardinal rule.
  - [confirmed] sfs-python (MIT) provides ONLY analytic driving functions (WFS, NFC-HOA, SDM, ESA) and contains NO numerical least-squares / pressure-matching / sound-field-control solver module; reference-only (tapering/array-geometry ideas), otherwise skippable.


---

## TOPIC: Phase-2 BeamSimII: directivity-controlled-speaker beam-shape UX models + filter deployment/export formats (FIR vs biquad) for the automatic beamformer/filter designer

### Report

# Part A — How directivity-controlled speakers expose "beam shape" to users

## A.1 The dominant commercial model: PRESET beam shapes + DISCRETE steering, not arbitrary continuous targets

Every shipping DSP-directivity loudspeaker found exposes a small set of named preset patterns plus a discrete steering selection, never a free numeric "set the beamwidth to 34 degrees" control.

**Bang & Olufsen Beolab 90** (18 drivers, per-driver amp+DSP — the closest analog to BeamSimII's model):
- **Beam Width Control**: exactly THREE discrete modes — **Narrow** (precise sweet spot), **Wide** (broad front, "like Beolab 5"), **Omni** (360-degree). (tonmeister.ca / audioholics / Stereophile)
- **Beam Direction Control**: the user defines ONE OF FIVE discrete directions as the acoustic "front." Not a continuous angle. (tonmeister.ca, B&O technical sound guide)
- **How the filters are made (load-bearing for our forward model)**: per Geoff Martin (B&O tonmeister), "the magnitude and phase responses of the filters are the result of measurements of the drivers in their locations, and an optimisation algorithm designed to find the best possible solution for a target given beam width." This is EXACTLY BeamSimII's structure: measured/simulated per-driver responses (our H tensor) + a least-squares optimization to a target beamwidth → per-driver complex filters. The B&O filters explicitly include BOTH a per-driver magnitude clean-up AND beam-steering phase. (tonmeister.ca "Behind the scenes"; research lineage = Møller, Olsen, Agerkvist, Dyreby, Munch, "Circular Loudspeaker Arrays with Controllable Directivity," ~2010, DTU.)
- **Active Room Compensation**: a separate post-step; B&O calibrates with only ~3 measurement positions because Narrow mode already minimizes room interaction. (Stereophile)
- **Core design tension stated explicitly**: drivers must be CLOSE for HF beam control but FAR for LF directivity control — the spacing/aliasing tradeoff. This bounds what a heterogeneous woofer/mid/tweeter stack can achieve per band.

**Kii Three**: DSP-controlled side/rear drivers ("Active Wave Focusing") produce a **cardioid** pattern down to ~54 Hz. Fixed target pattern (cardioid), not user-variable shape; the DSP achieves a single designed directivity. (Sound On Sound)

**Dutch & Dutch 8c**: cardioid midrange achieved ACOUSTICALLY (enclosure loading + rear-firing woofers), cardioid to ~100 Hz; the DSP handles crossover/EQ/room, not pattern selection. Confirms "cardioid" is the canonical named bass/mid pattern. (Sound On Sound)

**Genelec (GLM / "The Ones" coaxial)**: directivity is fixed by the waveguide; DSP = room calibration (GLM/AutoCal), not beam shaping. So "constant directivity" is realized passively, and DSP only does room/level/delay.

## A.2 Implication for BeamSimII Phase-2 v1 UI

The market consensus is **named preset patterns + a steering selection**, because (a) users think in patterns ("cardioid," "narrow," "omni"), not in mathematical beamwidth specs, and (b) arbitrary continuous targets are physically unrealizable across the full band for a sparse heterogeneous array (the spacing tradeoff), so exposing a continuous knob invites the user to request impossible beams.

Recommended v1 target-spec modes (a PRESET pattern picker + a steering picker, mirroring Beolab 90):
- **Pattern**: Omni / Cardioid / Wide / Narrow / (optional) Supercardioid/Hypercardioid (these are just cardioid family with a parameter b in `1/2(1+cos)` → general `a + (1-a)cos θ`, a in [0,1]: omni a=1, cardioid a=0.5, hypercardioid a≈0.25, figure-8 a=0). One scalar "cardioid order" slider can unify Omni→Cardioid→Figure-8.
- **Steering direction**: a discrete set of directions (e.g., on-axis, ±15°, ±30°) OR a continuous (azimuth, elevation) picker on the sphere — continuous steering is cheap to expose because it is just a phase progression, unlike continuous beamwidth.
- **Constant-Directivity-vs-frequency** as a toggle/checkbox ("hold pattern constant over X–Y Hz band"), since frequency-invariant beamwidth is a well-defined, desirable objective and maps cleanly to a per-frequency target.

This keeps v1 honest: a finite, physically-vetted preset set, each mapped to an analytic target directivity function D_target(θ,φ; f) that the least-squares solver matches.

## A.3 The target → weights math (forward model is already GLL pressure-matching)

The natural solver for "match a target directivity" given H is **regularized least-squares pressure matching** (the same family as the array literature and as B&O's optimization):

For each frequency f, let H_f be the [N directions × M drivers] complex matrix (H[m,f,n] transposed), d_f the [N×1] complex target pressure pattern sampled on the Lebedev grid, and Λ = diag(quadrature weights w_n) so the L2 norm is a true spherical integral. Solve:

  w_f = argmin_w  || Λ^{1/2} (H_f w − d_f) ||² + β || w ||²

Closed form (Tikhonov / diagonal loading):

  w_f = ( H_f^H Λ H_f + β I )^{-1} H_f^H Λ d_f

- `H_f^H` = conjugate (Hermitian) transpose — convention-critical (see hazards).
- β = regularization, tied directly to White Noise Gain / array effort `||w||²`. Choose β per-frequency via an L-curve or by constraining WNG = ||H_f w||² / (||w||² · something) to a floor (e.g. WNG ≥ −10 dB) — there is NO closed-form β↔WNG map, so iterate. This is the standard robustness knob; without it the superdirective LF solution blows up driver effort and is hyper-sensitive to model error.
- d_f construction: for a steered pattern at direction u0, d_f(n) = D_target(angle between u_n and u0) — purely real (the inter-driver steering phase emerges from H_f's true time-of-flight, which is preserved by the cardinal rule). For constant-directivity, use the SAME angular D_target at every f.

This is the exact "filter-and-sum beamforming" the array literature solves with FIR (Da−target cost), and exactly what B&O describes as "optimisation to a target beam width."

---

# Part B — Filter deployment / export formats

## B.1 FIR formats (the faithful realization — see Part C)

A per-driver FIR filter realizes w_m(f) by IFFT to taps h_m[n]. Deployment formats, in order of practitioner ubiquity:

1. **Mono impulse-response WAV, 32-bit float** — the universal convolver currency. REW/HouseCurve convention: mono WAV, 32-bit float; for a (non-causal) linear-phase FIR the impulse peak is centered (REW puts FIR peak at 250 ms in a 500 ms file to hold pre-ringing); a minimum-phase/causal IR has its peak at the first sample. Convolver must support the tap length (e.g. 24000 taps ≈ 500 ms @ 48 kHz). Consumed by: CamillaDSP (Conv/Wav), miniDSP OpenDRC/Flex-with-FIR, Hypex/Powersoft/Linea Research plate amps, Audacity, any convolution plugin.
   - CRITICAL: sample rate is baked in — the FIR is only valid at the design fs. Export must label fs and let the user pick (44.1/48/96 kHz).

2. **Plain-text coefficient file (one tap per line)** — CamillaDSP `Conv` `type: Raw` `format: TEXT`. Also accepted by many tools as CSV. Human-inspectable, fs-independent file but fs-dependent taps.

3. **CamillaDSP Conv (raw binary or wav)** — exact YAML:
```yaml
filters:
  drv0_fir:
    type: Conv
    parameters:
      type: Raw            # or Wav, or Values
      filename: drv0.txt
      format: TEXT         # enum: TEXT | S16_LE | S24_3_LE | S24_4_RJ_LE | S24_4_LJ_LE | S32_LE | F32_LE | F64_LE
      skip_bytes_lines: 0
      read_bytes_lines: 0
```
   For WAV: `type: Wav`, `filename: drv0.wav`, `channel: 0`. For inline: `type: Values`, `values: [...]`.

## B.2 IIR / biquad formats (per-driver tonal EQ ONLY — cannot carry the beam, see Part C)

Standard biquad: `H(z) = (b0 + b1 z^-1 + b2 z^-2)/(a0 + a1 z^-1 + a2 z^-2)`, a0 normalized to 1.

THE export hazard — two incompatible sign conventions for a1,a2 (verified, HouseCurve / REW / VituixCAD / miniDSP):
- **miniDSP / REW-"Generic"**: a0 normalized to 1, **a1 and a2 SIGN-FLIPPED**. Difference equation: `y = b0·x + b1·x1 + b2·x2 + a1·y1 + a2·y2`. Real REW example line:
  `biquad1, b0=1.000744..., b1=-1.998486..., b2=0.997760..., a1=1.998486..., a2=-0.998504,` — note a1 is POSITIVE (already negated). File format: text, blocks `biquadN, b0=.., b1=.., b2=.., a1=.., a2=..,` ; LAST line must have NO trailing comma; if <10 biquads only that many PEQ slots are set; pad unused with `b0=1,b1=0,b2=0,a1=0,a2=0`.
- **RBJ cookbook / VituixCAD raw / CamillaDSP**: standard sign, difference equation `y = b0·x + b1·x1 + b2·x2 − a1·y1 − a2·y2`. To convert a miniDSP file to CamillaDSP you must FLIP a1,a2 back. CamillaDSP `Biquad`/`Free` YAML:
```yaml
filters:
  drv0_peq:
    type: Biquad
    parameters:
      type: Free          # "normalized coefficients a1, a2, b0, b1, b2"
      a1: -1.998486        # standard sign (NOT miniDSP-flipped)
      a2:  0.998504
      b0:  1.000744
      b1: -1.998486
      b2:  0.997760
```
- VituixCAD: its IIR blocks compute b0,b1,b2,a1,a2; export window can emit miniDSP-style (sign-flipped a1/a2, compatible with miniDSP Advanced/SigmaStudio) or RBJ-style; v2 supports a configurable template (.vxt) for arbitrary text export. It also has a dedicated "Digital Biquad (BiQ)" block for raw-coefficient targets. ALL biquad exports are sample-rate-specific.

## B.3 VituixCAD interop (the user is a VituixCAD/REW expert)

- VituixCAD ingests per-driver responses as **.frd files** (frequency / magnitude / phase) and per-driver impulse responses. BeamSimII CAN export per-driver **filtered .frd** (apply w_m(f) to the on-axis H slice and write freq/mag/phase) so the user can audit the design inside VituixCAD's directivity/power tools.
- VituixCAD FIR export = impulse responses in WAV or TXT (NOT biquads) for the active-filter chain — same FIR currency as B.1. Active (non-minimum-phase) filter blocks are shown in blue/"FIR" text precisely because they cannot be reduced to biquads. This is independent corroboration of Part C.

---

# Part C — v1 export recommendation (the spine)

## C.1 Decisive technical fact: the beamformer is NON-MINIMUM-PHASE → FIR, not biquads

The per-driver beamforming weights w_m(f) carry the inter-driver RELATIVE PHASE that steers and shapes the beam. That relative phase IS the inter-driver time-of-flight encoded by BeamSimII's cardinal rule (shared spatial phase origin, never re-zeroed). A minimum-phase IIR/biquad filter has its phase rigidly locked to its magnitude by the Hilbert/Bode relation — so a biquad cascade fitted to |w_m(f)| will produce the WRONG phase and CANNOT represent steering delays. Therefore:

- **FIR is the only faithful realization of the beamformer.** It represents arbitrary magnitude AND phase per driver and preserves inter-driver delay. This is precisely why the entire array/personal-audio literature (`J(a)=||Da − target||²` filter-and-sum) and VituixCAD's active-filter export are FIR. Make FIR the PRIMARY v1 export.
- **Biquads are NOT co-equal.** They can only realize a per-driver minimum-phase tonal/EQ stage (e.g. the T/S terminal-response correction, woofer/tweeter level/shelf), NOT the beam. If offered at all in v1, biquad export must carry an explicit caveat: "biquads realize per-driver EQ only and do NOT reproduce the beam; the array directivity requires the FIR export." Presenting FIR-vs-biquad as a free choice would be physically wrong for this application.

## C.2 Concrete v1 export recommendation

**Primary (ship in v1): per-driver FIR, exported as mono 32-bit-float WAV (impulse response), one file per driver channel, plus a ready-to-load CamillaDSP YAML referencing them.**
- Rationale: 32-bit-float mono WAV is the universal convolver format (CamillaDSP, miniDSP OpenDRC/Flex-FIR, plate amps, plugins). CamillaDSP is the pragmatic hobbyist/pro multichannel target (free, cross-platform, M-channel, native FIR `Conv`), and the user's tools (REW/VituixCAD) round-trip WAV/TXT FIR natively.
- Also offer plain-text taps (CamillaDSP `Raw`/`TEXT`, one tap per line) for inspectability and OpenDRC.
- Export controls: design sample rate (44.1/48/96 kHz), tap count / FIR length, and linear-phase-vs-min-phase framing of the IR (with the correct peak-position convention so the convolver latency is handled).

**Secondary (optional v1, clearly scoped): per-driver biquad cascade for the tonal/EQ stage only**, in BOTH sign conventions selectable (miniDSP/REW-Generic with flipped a1/a2; and standard/CamillaDSP), with the non-beam caveat. This serves users on biquad-only DSPs (miniDSP 2x4HD PEQ) who accept that they get the per-driver EQ but a degraded/none beam.

**Interop bonus: per-driver filtered .frd export** so the VituixCAD-expert user can load the designed result back into VituixCAD for directivity/power-response auditing.


### Sources
- (primary) B&O Tech: BeoLab 90 – Behind the scenes (Geoff Martin, B&O tonmeister) — http://www.tonmeister.ca/wordpress/2015/10/06/beolab-90-behind-the-scenes/
- (secondary) Bang & Olufsen BeoLab 90 loudspeaker review (Stereophile) — https://www.stereophile.com/content/bang-olufsen-beolab-90-loudspeaker
- (secondary) Bang & Olufsen Beolab 90 Titan Edition (Audioholics) — Beam Width/Direction Control modes — https://www.audioholics.com/tower-speaker-reviews/bang-olufsen-beolab-90-titan
- (primary) Beolab 90 Technical Sound Guide (Bang & Olufsen A/S, official PDF) — https://bangolufsenassistentgohe.blob.core.windows.net/manuals/SPEAKERS/BEOLAB_90/beolab_90_technical_sound_guide_v11.pdf
- (secondary) Kii Audio Three / BXT review (Sound On Sound) — https://www.soundonsound.com/reviews/kii-audio-three-bxt
- (secondary) Dutch & Dutch 8C review (Sound On Sound) — cardioid midrange method — https://www.soundonsound.com/reviews/dutch-dutch-8c
- (primary) CamillaDSP README (HEnquist, official) — Conv format enum, Biquad Free, pipeline — https://raw.githubusercontent.com/HEnquist/camilladsp/master/README.md
- (primary) CamillaDSP docs (official) — https://github.com/HEnquist/camilladsp
- (secondary) HouseCurve — File Formats (miniDSP biquad text format, sign convention, REW WAV IR conventions) — https://housecurve.com/docs/manual/file_formats.html
- (primary) miniDSP Flex PEQ / file import docs (official) — https://docs.minidsp.com/product-manuals/flex/dsp-reference/peq.html
- (primary) miniDSP — Advanced Biquad Filter Programming (official) — https://www.minidsp.com/applications/advanced-tools/advanced-biquad-programming
- (primary) VituixCAD Online Manual (Kimmo Saunisto, official) — IIR/FIR export, sign convention, .frd — https://kimmosaunisto.net/Software/VituixCAD/VituixCAD_help_11.pdf
- (primary) VituixCAD v2 changelog (export window / .vxt template, BiQ block) — https://kimmosaunisto.net/Software/VituixCAD/changelog2.html
- (primary) Møller, Olsen, Agerkvist, Dyreby, Munch — Circular Loudspeaker Arrays with Controllable Directivity (DTU/AES, ~2010), as cited by B&O — http://www.tonmeister.ca/wordpress/2015/10/06/beolab-90-behind-the-scenes/
- (primary) Robust superdirective / constant-beamwidth beamforming & Tikhonov-regularized pressure matching (survey of array literature, ResearchGate/PMC/Frontiers) — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8587748/

### Decisions implied
- v1 GUI target-spec = a PRESET PATTERN picker (Omni / Cardioid / Wide / Narrow, optionally a single 'cardioid order' slider a∈[0,1] mapping omni→cardioid→figure-8) + a STEERING direction picker (continuous az/el on the sphere is fine; beamwidth must stay discrete/preset) + an optional 'constant directivity over band X–Y Hz' toggle. Do NOT expose a free continuous-beamwidth number.
- Each preset maps to an analytic target directivity D_target(θ,φ;f) sampled on the Lebedev grid as the real target vector d_f; the solver is per-frequency Tikhonov-regularized pressure matching w_f = (H_f^H Λ H_f + βI)^{-1} H_f^H Λ d_f with Λ = quadrature weights.
- Expose β (or equivalently a White-Noise-Gain / array-effort floor) as the robustness control; choose it per-frequency by L-curve or by constraining WNG to a floor — there is no closed-form β↔WNG mapping, so iterate. This caps superdirective LF blow-up and model-error sensitivity.
- PRIMARY v1 export = per-driver FIR as mono 32-bit-float WAV impulse responses (one per driver channel) PLUS an auto-generated CamillaDSP YAML wiring them via Conv/Wav into the per-channel pipeline; also offer plain-text taps (CamillaDSP Raw/TEXT). FIR realizes the beam faithfully.
- SECONDARY/optional v1 export = per-driver biquad cascade for the TONAL/EQ stage only, selectable in BOTH sign conventions (miniDSP-flipped and standard/CamillaDSP), with an explicit caveat that biquads do NOT realize the beam. Never present FIR and biquad as equivalent options.
- Add a per-driver filtered .frd export (apply w_m to the on-axis H slice → freq/mag/phase) so the VituixCAD/REW-expert user can audit the designed directivity/power response in their own tools.
- Export module must take design sample rate (44.1/48/96 kHz), tap count, and IR phase type (linear vs minimum) as explicit user parameters, and must record/handle the IR peak-position/latency convention (REW puts linear-phase FIR peak mid-file, min-phase at sample 0).
- Implement and unit-test the exp(−jωt)→FIR IFFT sign bridge AND the biquad a1/a2 sign per target before shipping; a round-trip test (design → export → re-import → re-evaluate P(f,dir) vs intended beam) is the acceptance gate for the export module.

### Open questions
- Exact IFFT sign/conjugation to convert complex W_m(f) (NumCalc exp(−jωt) engineering convention) into real causal FIR taps without time-reversal — must be empirically verified at implementation (round-trip the synthesized beam through the exported FIR).
- Per-driver FIR tap length / latency budget vs LF beam control: superdirective low-frequency targets demand long FIRs (and large delay); need to pick a max tap count (e.g. 4096–24000 @ 48 kHz) and decide whether to cap LF directivity for realizability.
- Whether v1 ships a continuous 'cardioid-order' slider or only discrete named presets (Omni/Cardioid/Wide/Narrow) — UX simplicity vs flexibility; market precedent (Beolab 90) is purely discrete.
- Whether to bundle a Hypex FusionAmp / Powersoft / Linea Research plate-amp export profile in v1, or defer to CamillaDSP + WAV/biquad only (these plate amps have proprietary or app-specific FIR/biquad ingestion that needs per-vendor verification).
- Confirm CamillaDSP Biquad 'Free' uses the standard (non-flipped) a1/a2 sign in the running engine (HouseCurve cross-reference says CamillaDSP needs the miniDSP signs flipped back; verify against a current CamillaDSP version, since the official README excerpt did not state the transfer-function sign explicitly).
- How to present/limit the steering direction set so requested beams stay physically realizable for a given sparse heterogeneous driver layout (tie steering options to the actual array's controllable angular range vs frequency).

### Adversarial verification verdicts
  - [partially-correct] Beamforming weights w_m(f) are inherently NON-minimum-phase (inter-driver relative phase encodes time-of-flight and steers/shapes the beam), so a minimum-phase biquad cascade (phase Hilbert-locked to magnitude) CANNOT realize them; FIR is the only faithful realization; biquads can carry only the per-driver minimum-phase tonal/EQ stage. -> CORRECTION: Overstated to say 'FIR is the ONLY faithful realization.' Arbitrary excess phase can also be realized by a minimum-phase IIR cascade PLUS an all-pass section and/or a pure (fractional) delay. FIR (arbitrary/linear phase) is the most PRACTICAL and universal realization, and is what you should use — but it is not the unique mathematical option. Restate as: 'a pure minimum-phase biquad cascade cannot carry the beam-steering excess phase; FIR is the practical universal realization (or IIR + all-pass/delay).'
  - [confirmed] Two mutually incompatible biquad sign conventions exist and must be handled at export: (1) miniDSP/REW-Generic: a0 normalized to 1, a1/a2 sign-FLIPPED, y = b0x+b1x1+b2x2 + a1y1 + a2y2 (a1 stored positive); (2) RBJ cookbook / CamillaDSP: standard sign, y = b0x+b1x1+b2x2 − a1y1 − a2y2. miniDSP->CamillaDSP requires flipping a1,a2 sign. -> CORRECTION: Caveat on naming: REW's INTERNAL/native and academic representation is the RBJ standard ('1 +' denominator, subtract feedback); it is specifically REW's 'Generic'/miniDSP EXPORT mode that pre-negates a1,a2. The claim's '(1) miniDSP / REW-Generic' label is correct as long as 'REW-Generic' means the Generic/miniDSP export profile, not REW internally.
  - [confirmed] Commercial DSP-directivity speakers expose PRESET beam shapes + DISCRETE steering, not arbitrary continuous beamwidth: Beolab 90 = 3 width presets (Narrow/Wide/Omni) + 5 discrete beam directions; Kii Three and D&D 8c ship a single fixed cardioid pattern. So Phase-2 v1 should offer a finite preset pattern set + steering. -> CORRECTION: The Kii 'down to ~54 Hz' figure that appears in the finder's report Part A is NOT well-supported (see extra_findings); but claim 3 as worded only says Kii ships a fixed cardioid, which IS confirmed.
  - [partially-correct] B&O Beolab 90 derives its per-driver filters by measuring each driver in situ (magnitude AND phase) then optimizing to a target beam width — structurally identical to BeamSimII solving regularized least-squares pressure-matching: w_f = (H_f^H Λ H_f + βI)^{-1} H_f^H Λ d_f, Λ = diag(Lebedev weights), H_f^H = conjugate (Hermitian) transpose. -> CORRECTION: Soften 'structurally IDENTICAL.' B&O publicly states only that it uses measurements + 'a custom optimisation algorithm' to a target beam width; it does NOT disclose that the solver is specifically regularized (Tikhonov) least-squares pressure-matching, nor that B&O uses Lebedev quadrature weighting. The verbatim Geoff Martin sentence quoted in the claim could not be confirmed word-for-word from the primary page. So: 'B&O uses measured per-driver magnitude+phase plus an optimization to a target beamwidth (confirmed); regularized LS pressure-matching is a faithful and standard FORMALIZATION of that approach for BeamSimII (the finder's inference), not a stated B&O fact.'
  - [partially-correct] CamillaDSP Conv 'format' enum is exactly {TEXT, S16_LE, S24_3_LE, S24_4_RJ_LE, S24_4_LJ_LE, S32_LE, F32_LE, F64_LE}; type in {Raw, Wav, Values}; Biquad 'Free' takes normalized a1,a2,b0,b1,b2 in standard (non-flipped) sign. Mono 32-bit-float WAV impulse response is the universal convolver format across CamillaDSP, miniDSP OpenDRC/Flex-FIR, plate amps, and REW/VituixCAD. -> CORRECTION: The 'universal mono-32-bit-float-WAV' claim is REFUTED for miniDSP. miniDSP OpenDRC/Flex FIR import expects a RAW IEEE-754 single-precision 32-bit-float BINARY coefficient file (or plain-text coefficients), per the miniDSP Flex/OpenDRC manuals — NOT a WAV container; a WAV IR must be converted to raw float (e.g. via SoX) before import. CamillaDSP DOES accept WAV (Conv type Wav). So the universal common denominator is 32-bit-FLOAT COEFFICIENTS, not the WAV CONTAINER. Also: VituixCAD's exported biquad sign convention is NOT independently verified here and should not ride on CamillaDSP's confirmation — treat VituixCAD-raw sign as uncertain until checked against VituixCAD docs.
  - [confirmed] Implementation bridge: H is in NumCalc engineering convention exp(-jωt). When IFFT-ing complex W_m(f) to FIR taps, the IFFT sign/conjugation must match exp(-jωt) or the impulse comes out time-reversed/conjugated. This DSP-side z-domain a1/a2 + IR-peak sign work is SEPARATE from the acoustic time-convention bridge; both must be reconciled. -> CORRECTION: No correction to the principle. Practical note for code time: the cleanest guard is a round-trip unit test — synthesize a single complex exponential / known delayed impulse spectrum in the exp(-jωt) convention, transform with your chosen IFFT path, and assert the tap peak lands at the expected (positive) sample lag; this catches both the conjugation sign and the fftshift/peak-position issue at once.
  EXTRA: VERBATIM-QUOTE RISK (claim 4): The exact Geoff Martin sentence quoted in the claim ('the magnitude and phase responses of the filters are the result of measurements of the drivers in their locations, and an optimisation algorithm designed to find the best possible solution for a target given beam width') could NOT be confirmed word-for-word from the tonmeister 'Behind the scenes' page or Stereophile (Stereophile/tonmeister returned 403 or a differently-worded passage). The SUBSTANCE (measured per-driver response + optimization to a target beamwidth, with the 15 non-front drivers receiving phase-controlled beam-shaping signals) IS supported by Stereophile's reporting. Treat the quotation as a paraphrase, not a verified verbatim citation, in the plan.\n\nKii '54 Hz' (finder report Part A, not in claim 3 itself): unsupported as a single figure — sources span Kii marketing ~40 Hz, Stereophile measured <80 Hz, Recording Magazine ~50 Hz, Erin's Audio Corner measured transition 70-90 Hz. Use a RANGE ('cardioid roughly down to the 40-90 Hz region depending on source/criterion') rather than '54 Hz'.\n\nREW naming subtlety worth surfacing in the plan: REW's native/internal coefficients are RBJ-standard ('1+' denominator, subtract feedback); only its 'Generic'/miniDSP EXPORT profile pre-negates a1,a2. So 'REW' alone is ambiguous — always specify the export profile when documenting the import path.\n\nCamillaDSP biquad realization detail (from source): it uses Direct Form II Transposed (two state vars s1,s2), not literal Direct Form I, but the math is equivalent to y = b0x+b1x1+b2x2 - a1y1 - a2y2. Relevant only if you ever try to match internal state, not for coefficient export.\n\nVituixCAD biquad export sign convention is unverified in this pass and should be confirmed against VituixCAD's own documentation before grouping it with the 'standard sign' camp (claims 2 and 5 both assert this without independent support here).\n\nB&O front/rear split (useful for the UX section): the three FRONT drivers are a phase-correct 3-way aimed at the listener; the OTHER 15 drivers per speaker do the beam shaping. This maps cleanly onto BeamSimII's heterogeneous woofer/mid/tweeter + auxiliary-driver model and reinforces claim 3's 'preset patterns + steering' recommendation.
