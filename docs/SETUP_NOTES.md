# BeamSimII — Setup Notes

## Stage-0 Toolchain Gate: NumCalc arm64 Verification

**Date:** 2026-06-17
**Machine:** MacBook Pro 16" Late 2024, Apple M4 Max (12P+4E cores, 48 GB), macOS 15.7.7

### Result: GATE PASSED

NumCalc built and ran natively on Apple Silicon (arm64) with no errors.

### Build details

**Source:** Mesh2HRTF repository, commit `e45d0436a6fbeca3db13828cbae23ca109225be3`
**Clone location:** `/Users/andy/mesh2hrtf` (outside the beamsim2 repo, never committed)
**Source directory:** `/Users/andy/mesh2hrtf/mesh2hrtf/NumCalc/src/`
**Binary path:** `/Users/andy/mesh2hrtf/mesh2hrtf/NumCalc/bin/NumCalc`

### Architecture check (`file` output)

```
/Users/andy/mesh2hrtf/mesh2hrtf/NumCalc/bin/NumCalc: Mach-O 64-bit executable arm64
```

### Live solve verification

Ran NumCalc against the bundled SHTF test project (1 frequency step, 100 Hz):

```
Step 1, Frequency = 100 Hz
Single level fast multipole BEM
Number of equations = 2412
CGS converged in 29 iterations
Total time: 1 second
Output: be.out/be.1/{pBoundary, pEvalGrid, vBoundary, vEvalGrid}
```

**DR-01 confirmed:** NumCalc builds and runs native arm64 on the M4 Max. The ML-FMM
is active ("Single level fast multipole BEM"). The binary is the production BEM backend
for all Phase 1 solves.

### Notes on build warnings

Clang emitted deprecation warnings for `sprintf` (use `snprintf`) and a VLA extension
warning in `NC_Input.cpp`. These are cosmetic — upstream code, zero effect on
correctness or performance.

### NumCalc on PATH

Binary is at `/Users/andy/mesh2hrtf/mesh2hrtf/NumCalc/bin/NumCalc`. Add to PATH or
symlink to `/usr/local/bin/NumCalc` when needed (requires sudo). The `NumCalcAdapter`
in `backends/numcalc/adapter.py` will accept a configurable binary path so this is not
required for the app to work.
