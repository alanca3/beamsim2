"""Validation: internal-consistency cross-checks against closed-form solutions.

Implemented and wired as pytest tests: V-1 (piston/cap directivity,
``analytic_piston``), V-2 (pulsating sphere, ``sphere_benchmark``), V-4 (power /
directivity index, ``power_di``), V-5 (two-driver superposition / phase origin,
``assembly.phase_origin``).

NOT yet implemented (see docs/handoffs phase-1 audit): V-3 (mesh convergence
N_epw 6→8→10 — ``convergence.py`` is a stub) and V-6 (BEM-vs-analytic diffraction
diagnostic). Do not assume coverage these modules do not yet provide.
"""
