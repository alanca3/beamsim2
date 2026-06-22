# Chunk 3b — Empirical findings & implementation decisions (constant-directivity hardening)

> Authoritative record of what the 3b diagnose → research → empirical-prototype campaign actually
> showed, including **three places where the evidence overturned the working premises** (including
> two of the kickoff's own "established findings"). Written 2026-06-22. Methodology mirrors 3a:
> a self-contained numpy prototype against the REAL repo forward model (`monopole_field` +
> `design()`), then a 4-probe parallel research+prototype workflow, cross-checked against the
> current `design(constant_di)`.

## The 3b fix, in one line
The constant-directivity engine now optimizes the **proper directivity index** (`A = c cᴴ`,
`R` = whole-sphere covariance — Luo's actual objective), held constant across frequency by a
**single band-wide τ\*** chosen to honor an **honest WNG floor** (the WNG-vs-τ curve is *unimodal*,
not monotone), made realizable by a **cardinal-safe global-phase continuity alignment + one shared
delay** (NOT a smoothing kernel — smoothing is harmful), with a relative **`eps_min·I` floor** for
well-posedness. The gate proves **constant directivity index** (Luo's objective) and a *~constant*
−6 dB beamwidth.

## Diagnose-first (confirmed against the real `design(constant_di)`)
- **Filters are violently rough.** Per-bin MSCD `phase_roughness ≈ 5.6 rad` (vs the 3a cardioid's
  0.015 rad). complex-curvature 2.2.
- **No WNG floor.** `constant_di` WNG dips to −1.7 dB; `max_directivity` to **−15.6 dB**
  (superdirective blow-up at low freq). Only `mvdr`/`lcmv` floor today.
- **Edge-bin ill-posedness.** At the top bin `D = A−τ*R` is barely indefinite (positive eig 0.045);
  `A` min-eig ~9e-5 at band edges. Justifies `eps_min·I`.
- **The deep one — constant cap-GDI ≠ constant beamwidth.** The shipped engine (A=accept-cap,
  R=whole-sphere) holds the *cap-ratio* GDI perfectly constant (ptp 0.0) but the proper DI varies
  (ptp 6.7 dB) and the −6 dB beamwidth narrows (std 17° over 1500–4200 Hz on a 50-driver cap). The
  per-bin optimizer narrows the main lobe while sidelobes fill the cap-complement.

## Three corrected premises (measured, not assumed)
1. **The shipped `A = accept-cap` GDI is a *deviation* from Luo, not the directivity index.** Luo
   (EUSIPCO 2024, arXiv:2407.01860) holds the classical generalized-Rayleigh **directivity index**
   `D = 4π·(wᴴ A w)/(wᴴ R w)` with **`A = c cᴴ`** (rank-1, `c = conj(H_look)`), `R` = whole-sphere
   covariance. Switching to this (`directivity_mode="index"`) holds the **field-measured DI flat to
   0.000 dB** by construction. The cap-ratio (`"region"`) mode is a legitimate but *different*
   objective (front-to-total power concentration); it is retained as a non-default option.
2. **Constant directivity index does NOT give constant −6 dB beamwidth.** With DI flat to 1e-11 dB,
   the beamwidth still drifts 40→31° (std 3.0°, ptp 9°) over a 2.8:1 band. Beamwidth is a Keele-CBT
   geometric/Legendre-shading property, not an MSCD output. **The gate asserts constant DI as the
   load-bearing claim and ~constant beamwidth (honest std bound) as a secondary check** — it does
   NOT claim MSCD pins the beamwidth.
3. **"Constraint-preserving frequency *smoothing*" is the wrong mechanism — and "WNG monotone in τ"
   is false.**
   - The ~5.6 rad roughness is almost entirely a **reference artifact**: an arbitrary per-bin global
     phase (secular-root / eigenvector sign) plus phase noise on near-silent drivers (61% of
     (driver,bin) have |w| < 10% of peak). Two **exact, cardinal-safe** ops fix it: (a) per-bin
     global-phase continuity alignment (rotate each `w_f` to maximize `Re⟨w_f, w_{f-1}⟩`), (b) one
     shared modeling delay. Honest (magnitude-gated) roughness drops 2.1→0.8 rad while DI stays
     flat and the QCQP constraints (`|cᴴw|=1`, `wᴴ D w=0`) hold to machine precision. An actual
     smoothing **kernel injects DI ripple (0.6–3.1 dB) and widens the beam — harmful** → no kernel
     smoother for constant_di. This *honors* kickoff decision 2 (constraint-preserving, NOT 3a's
     additive `μ·DᵀD` penalty) — the alignment **is** an exact constraint-preserving projection.
   - **WNG(τ) is unimodal, not monotone-decreasing** (kickoff "established finding #3" was wrong for
     proper-DI). It peaks near the matched-field WNG at an interior `τ_peak` and falls on both
     sides. The honest floor search bisects ONE shared `τ*` on the descending branch
     `[min_f τ_peak, ceiling]`, where min-over-bins WNG increases as τ drops. A naive "lower τ to
     raise WNG" collapses τ*→0 and destroys constant DI.

## Implementation spec (each with its own check)
1. **Proper-DI objective** — `TargetSpec.directivity_mode ∈ {"region","index"}`. `design()` builds
   `A = c cᴴ` (index) or accept-cap covariance (region), `R` = whole-sphere covariance. Default
   keeps the existing `test_mscd_holds_generalized_di_constant_across_frequency` green; the GUI and
   the V-CBT gate select `"index"`. *(Default choice vetted with advisor — see below.)*
2. **Honest WNG floor for `constant_di`** — two-pass with ONE shared `τ*`: pass-1 per-bin proper-DI
   ceiling `τ_max(f)` (unfloored `max_directivity`), `ceiling = 0.98·min_f τ_max` (capped by
   `target_gdi_db`); pass-2 bisect `τ*` on `[min_f τ_peak, ceiling]` to the largest τ* whose
   worst-bin WNG ≥ floor. If even `min_f τ_peak` fails → clamp + `band_feasible=False` (DI stays
   flat; never silent garbage).
3. **Honest WNG floor for `max_directivity`** (engine) — per-bin bisect `log(eps)` of the R-loading
   in `w = top-gen-eig(A, R+eps I)` (monotone↑); clamp+flag if floor > matched-field ceiling
   `10log10‖c‖²`. The internal pass-1 ceiling finder for `constant_di` stays **unfloored**.
4. **`eps_min·I` floor** — relative `eps_min·trace(R)/M` on both A and R before all eig/secular
   work; default `eps_min = 1e-7`. DI is eps_min-invariant at fixed τ (<0.1 dB); it rescues the
   barely-indefinite rank-1-A ceiling at band edges.
5. **Cardinal-safe realization** — after the per-bin MSCD solve at τ*, apply global-phase continuity
   alignment + one shared delay. Do NOT re-impose `cᴴw = 1∠0` (that re-injects rough TOF phase);
   leave `cᴴw = 1∠φ_f` with smooth φ_f (|cᴴw|=1 preserved; all downstream metrics are
   magnitude/power, global-phase-invariant).
6. **Magnitude-gated phase roughness** — gate realizability on a metric that ignores phase noise on
   near-silent drivers (|w| < ~10% of peak across the 2nd-diff stencil), not raw `phase_roughness`.
7. **frac_mu** (3a carry-forward) — **KEEP `frac_mu = 1e-2`**. Confirmed active + beam-safe on a
   3-driver under-determined stressor (80–500 Hz, supercardioid, wng=−3): 31% curvature / 39%
   roughness cut at DI drift 0.11 dB; beam-preserving through 1e-1, harmful at ≥3e-1. No change.
8. **MECD** — stays a documented `NotImplementedError` stub (kickoff decision 1: MSCD-only).

## Gate (3b): engine-level V-CBT, on the 50-driver cap, in the flat-CBT band
`design(constant_di, directivity_mode="index")` on `cbt_cap(Rc=0.12, θ0=45°, 6 rings, dx=0.035)`
(~50 monopoles, icosphere(4) grid). **Band = the flat-CBT regime** `geomspace(2300, 4200, 6)`
(k·a 3.6→6.5 ≥ the Keele cutoff k·a≈3 ≈ 1929 Hz; SH-resolvable: k·Rc 5.0→9.2 < the order-16 ceiling
at ~7280 Hz). The full 1500–4200 band was rejected because 1500 Hz sits *below* the CBT cutoff where
the beam is still collapsing (a physics limit, like 3a's spatial-aliasing upper edge), not a solver
defect.

Measured on the flat-CBT band with the honest WNG floor (floor = 0 dB): **field DI ptp 0.000 dB**
(constant at ~14 dB), **beamwidth ptp 7°, std 2.4°** (42→35°), WNG floored to ≥0 dB at every bin,
all bins feasible. The full-band floor-0 result was ptp 20°/std 6.2° — the band restriction is what
makes beamwidth honestly callable constant.

Assert: (a) field DI ptp < ~0.05 dB (constant directivity index, by construction — the load-bearing
claim); (b) beamwidth finite at every bin and ~constant (ptp ≤ ~10°, std ≤ ~4°, all within a sane
window — the secondary CBT check, honest about the regime); (c) magnitude-gated roughness < ~1.0 rad
(realizable, after global-phase alignment); (d) WNG floor respected, `feasible_mask` all True;
(e) cardinal-rule control: collapse the cap to the origin → DI → 0. Keep the constant-DI tests,
V-cardioid, V-RT, V-5 phase-origin, V-1/V-2 green throughout.

**Verified follow-ups (advisor):** floored `max_directivity` at the default −6 dB floor still passes
the pinned `test_max_directivity_varies_and_dominates_constant_di` (varies + dominates). No
design()-output test asserts a real/unit `cᴴw`, so leaving `cᴴw = 1∠φ_f` is safe. `directivity_mode`
default is `"region"` (existing pinned tests untouched); GUI + the V-CBT gate pass `"index"`.
