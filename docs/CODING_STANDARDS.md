# BeamSimII — Coding Standards

> Extracted from `BEAMSIMII_Gameplan.md §5.1`. This is the binding reference for all
> code written in this project. **Follow strictly; flag any deviation with reasoning.**

---

## Rules (non-negotiable)

### 1. Complete and runnable — never pseudocode, never stubs in delivered functions

Every function in a delivered module must run without error. If something is too long
for one session, say so and split explicitly. A function body that contains `pass`,
`...`, or a `# TODO` comment is not delivered code.

Exception: functions that explicitly raise `NotImplementedError` with a clear message
are fine as placeholders for future schemes (e.g., Fliege-Maier grids). The
distinction: it must *tell* you it's not implemented, not silently return wrong data.

### 2. Dimensional comments on every significant array

Use the project's notation:

```python
# H_bem: [F × N] complex128
# unit_vectors: [N × 3] float64
# weights: [N] float64
```

Shapes are part of the type. The reader should never have to trace data flow to know
what shape a variable is. Apply to: function parameters that are arrays, local
variables holding intermediate arrays, return values, dataclass fields.

### 3. NumPy-style docstrings on every public function

Minimum structure:

```python
def my_function(param: np.ndarray) -> np.ndarray:
    """
    One-line description of what this does in acoustics terms.

    Longer explanation if needed — bridge to the acoustics concept.

    Parameters
    ----------
    param : np.ndarray, shape [N, 3]
        Description including shape and units.

    Returns
    -------
    np.ndarray, shape [N]
        Description including shape and units.

    Raises
    ------
    ValueError
        If param has wrong shape or invalid values.
    """
```

### 4. Claim labels: VERIFIED / INFERRED / HEURISTIC

Every physics formula, numerical constant, or engineering judgment in a comment or
docstring must carry one of:

- **VERIFIED**: confirmed from a cited published source. Include author/year.
- **INFERRED**: reasoned from evidence but not directly confirmed for this exact case.
- **HEURISTIC**: rule of thumb / engineering judgment.

Example:

```python
# VERIFIED: c = sqrt(γ R_d T_K) for dry air
# (Kinsler, Frey, Coppens, Sanders, "Fundamentals of Acoustics" 4th ed., §4.2)
```

Never omit the label on a formula. A formula with no label is an unreviewed claim.

### 5. Bridge to acoustics in all explanations

The primary reader is an acoustics domain expert, not a software engineer. When
explaining a numerical concept, lead with the acoustics analogue:

- "quadrature weights" → "the same role as cos(θ) dθ dφ in a surface integral — they
  correct for the fact that Cartesian grid points pack more densely near the poles"
- "SH orthogonality" → "each spherical harmonic represents a distinct radiation
  pattern; orthogonality means they don't cross-talk when measured on a sphere"

### 6. Type hints on all public functions

```python
def lebedev(n_points: int = 26, *, radius: float = 1.0) -> ObservationPoints: ...
```

Use `Optional[X]` or `X | None` for values that can be `None`. Use `numpy.ndarray`
for array parameters; add shape as a comment or docstring note.

### 7. Formatting: black + ruff

- Line length: 100 characters (configured in `pyproject.toml`).
- Run `black src/ tests/` and `ruff check src/ tests/` before committing.
- Style is never a discussion — the tools decide.

### 8. No silent behavior changes to locked decisions

The architecture decisions (DR-01 through DR-06) are locked. If code must deviate,
flag it with `# DR-XX: deviation because ...` and document the reason. Never silently
change behavior.

---

## The test discipline

- **A self-test for every subsystem.** Run `pytest` before any session closes.
- Tests go in `tests/`; name them `test_<module>.py`.
- Tests must pass with the project's installed dependencies — no test-only hacks that
  mask real failures.
- Validation tests (§7 of the gameplan) are part of the test suite.

---

## Structural rules

- **GUI imports core; core never imports GUI.** The dependency direction is one-way.
- **No solver-specific objects cross the `BEMBackend` boundary** (DR-02). `core/types.py`
  defines the normalized types; adapters translate to/from them.
- **No mutable defaults in dataclasses.** Use `field(default_factory=...)`.
- **Array convention:** float64 for real arrays, complex128 for pressure/transfer
  function arrays. Never silently downcast.
