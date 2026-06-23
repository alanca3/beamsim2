"""BeamSimII Phase 2 — the automatic beamforming filter designer.

Given the Phase-1 per-driver complex tensor ``H[M drivers x F frequencies x
N sphere-directions]`` (assembled by :mod:`beamsim2.assembly`, persisted by
:mod:`beamsim2.io.hdf5_store`), this package solves per-driver complex weights
``w_m(f)`` that steer / shape the radiated beam toward a user target, evaluates
the achieved directivity, and (later) realizes the weights as deployable filters.

House convention (load-bearing — see ``docs/Phase 2 - Filter Solver.md`` DR-P2-02):
the coded forward model is ``P(f, dir) = sum_m w_m(f) * H[m, f, dir]`` (the AES GLL
complex summation, already implemented as
:func:`beamsim2.validation.closed_loop.steer_response`). Every solver in this
package is written so its weights drop into that sum with **no extra conjugation**:

* look vector       ``c = conj(H[:, f, look])``
* covariance        ``R = conj(H_f) @ diag(a) @ H_f.T``   (a = Lebedev weights)

Cardinal rule (sacred): the beamformer consumes ``H`` with its native inter-driver
phase and never re-zeroes / minimum-phase-ifies / per-driver-normalizes any driver
(``DATA_CONTRACT.md`` Section 3.4; ``tests/test_phase_origin.py`` must stay green).

Modules
-------
targets      Build a target field / accept-reject masks from a ``TargetSpec``.
covariance   Look vector and weighted complex covariance (house convention).
weights      Solver modes: delay-sum, regularized LS / pressure-matching, MVDR/LCMV,
             Luo MECD/MSCD constant-directivity.
regularize   White-noise-gain-floor diagonal loading (the single robustness knob).
forward      Achieved field + directivity metrics (reuses ``closed_loop``).
design       Engine dispatch: (RadiationDataset, TargetSpec) -> DesignResult.
orchestrator Auto-Design (``engine="auto"``): a principled escalation ladder over the
             well-posed engines that picks the one best meeting the target (Chunk 3c).
realize      [DEFERRED, Stage P2-5] complex weights -> causal per-driver FIR/IIR.

See ``docs/Phase 2 - Filter Solver.md`` (the gameplan) and ``docs/Research Phase 2.md``
(the cited research) for the full spec.
"""

from __future__ import annotations

__all__: list[str] = []
