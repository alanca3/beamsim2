# BeamSimII — Coding Standards (§5 of Gameplan)

> Extracted from `BEAMSIMII_Gameplan.md` §5 as a standalone reference.

See `BEAMSIMII_Gameplan.md` §5 for the full coding-standards specification.
Key points:
- Python 3.12; type hints on public functions; black + ruff for formatting/linting.
- Label claims in comments/docs: VERIFIED / INFERRED / HEURISTIC + author/year citations.
- Bridge acoustics concepts in explanations for the acoustics-expert reader.
- Full `tests/` suite must pass before any session closes.
- GUI imports core; core **never** imports GUI.
- No silent behavior changes to locked decisions (flag and reason any override).
