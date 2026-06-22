# Chunk 3a — Empirical findings & implementation decisions

> Authoritative record of what the 3a research+prototype+baseline experiments actually
> showed, including two places where the empirical evidence overturned the
> `docs/Bug_Fix_Proposal.md` Chunk-3 framing. Written 2026-06-22.
> Methodology: a self-contained numpy prototype against the REAL repo forward model
> (`monopole_field` + `directivity_index`, icosphere(4), two monopoles at z=±0.043 m,
> d=0.086 m, c=343.2), cross-checked against the current `design()`.

## The 3a fix, in one line
A **complex, frequency-dependent virtual-source target** (origin monopole + normalized
origin dipole, built with the same forward operator the solver inverts) plus an **honest
white-noise-gain (WNG) floor imposed inside the LS solve** make the dual-opposed cardioid
hold across a band **with realizable (smooth) filters and bounded WNG**. Frequency coupling
is implemented but is *insurance for 3b*, not the lever for the 2-driver gate.

## Gate result (validated, reproduced twice)
DI ≈ 4.75 dB, rear null −47→−28 dB, WNG ≥ −12 dB, **across 150–~670 Hz** (8/12 of a
150–1500 Hz sweep). The upper edge is **physics**, not a solver defect: above kd ≈ π
(≈670 Hz for d=0.086 m) the 2-element pair spatially aliases; the analytic delay-sum
ground-truth weights show the **identical** DI taper. Wider bands need more drivers / smaller
spacing. Filters: max |2nd-diff of unwrapped phase| = **0.015 rad** (trivially realizable).

## Cardinal-rule proof (the new V-cardioid proof)
Two independent confirmations that steering comes ENTIRELY from H's inter-driver phase:
1. A shared `exp(−j2πfτ)` ramp (common latency, all drivers) leaves |P| invariant to 6e-16.
2. **Decisive control:** collapse both drivers to the origin (identical H rows → zero
   inter-driver phase); with the SAME target the cardioid dies to **DI = 0.000 dB**.

## Two corrected research/proposal premises (measured, not assumed)
1. **"A real frequency-independent target cannot make a cardioid" is FALSE for M=2.** The LS
   solve absorbs any global complex scale/phase, so a real target and the complex
   virtual-source target give *identical per-frequency* DI/null. Old `design()` at
   `wng_floor=-60` already passes the DI/null band gate 8/8. The complex target's real,
   verified payoff is **cross-frequency filter realizability** (0.015 rad vs the old 0.47 rad
   — 30× smoother), which is the proposal's actual goal #2. Justify the target fix by
   *realizability*, not by per-bin pattern error.
2. **The recommended shared delay `τ = r_obs/c` is wrong for this target** — it *adds* a phase
   ramp and makes filters ringier (5.62 rad vs 0.018 rad). The virtual-source target already
   carries the `exp(+jk r_obs)/r_obs` radial phase, so the optimal shared `τ ≈ 0`. The solver
   auto-selects τ by minimizing look-column phase roughness (comes out ≈0), which *confirms*
   the realizability win is in the target.

## Baseline (why the green gate is a real gate, per advisor)
| Configuration | DI/null band gate | filter roughness |
|---|---|---|
| OLD `design()` ls, `wng_floor=-60` (fragile) | **8/8 pass** | 0.47 rad (rough) |
| OLD `design()` ls, `wng_floor=-6` (default) | **0/8** (DI collapses to ~2.5) | 0.47 rad |
| NEW formulation, `wng_floor=-12` | 8/8 pass | 0.015 rad |

The discriminating tests are therefore **filter realizability** (smoothness < ~0.1 rad: old
0.47 fails, new 0.015 passes) and **robust-WNG honesty** (the cardioid survives at the default
robustness setting: old collapses, new holds). The bare DI/null band test is *not* a
discriminator (old@−60 passes it too); it is kept as a regression lock, not as proof of the fix.

## Frequency-coupling ablation (honest scope note)
prototype `solve_coupled` on the 150–670 Hz band:
- `frac_mu=0` (coupling off): 8/8, smoothness 0.017 rad
- `frac_mu=1e-2`: 8/8, 0.018 rad
- `frac_mu=1.0`: 7/8, 0.041 rad (slightly *worse* — nothing to smooth, adds cross-f bias)

→ For the well-posed 2-driver cardioid, coupling is near-inert (the target already gives
smooth filters). It is implemented and **unit-tested directly** (mu=0 ⇒ per-bin solve; a
non-trivial mu reduces the *complex* weight curvature on the clean fixture while preserving the
null). It becomes load-bearing in **3b** (superdirective / under-determined regimes where the
per-bin WNG-floor search forces λ to swing across frequency). The 3a cardioid gate does NOT
claim to prove it.

⚠️ **3b action:** the shipped default `frac_mu=1e-2` makes the coupling near-inert in `design()`
itself (curv 0.019→0.018), so the *integration path* never exercises meaningful coupling — only
the isolated unit test (at `mu≈1.0`) does. 3b must **re-validate the default `frac_mu`** against a
fixture where coupling matters, rather than trusting the 3a default.

## Implementation decision DR-P2-03
The LS engine (`spec.engine == "ls"`) becomes frequency-coupled **by default but degrades to
the per-frequency solve when F < 3** (the 2nd-difference curvature operator needs F ≥ 3; with
fewer bins there is nothing to smooth, so the coupled path collapses exactly to the per-bin
`ls_pressure_match`). This keeps the existing single-bin LS tests green. The coupling is a
no-op at F<3 and an additive PSD term at F≥3, so it can never change the sign convention or the
steering. The honest WNG floor for LS uses a **grid search** (LS WNG is non-monotone in λ — the
MVDR bisection is invalid) and reports the **distortionless** WNG (renormalize by one global
scalar `1/(cᴴw)` — cardinal-rule safe).
