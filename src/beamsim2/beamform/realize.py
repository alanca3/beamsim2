"""Filter realization: complex weights ``w_m(f)`` -> causal per-driver filters.

DEFERRED to Stage P2-5 (gated on the user choosing a deployment DSP; v1 exports for
audit only, DR-P2-03). Specified now so the deferral is architected-for.

Recipe (verified): two-step, default linear-phase FIR via IFFT + window, with ONE shared
modeling delay ``tau`` applied identically to ALL drivers. A common ``exp(-j 2 pi f tau)``
factors out of ``P`` (pure latency); a *per-driver* delay or per-driver minimum-phase
**re-steers the beam** and is forbidden (cardinal rule). Per driver: interpolate
``(log|w|, unwrapped phase)`` onto a dense uniform FFT grid -> conjugate (engineering ->
numpy DSP convention) -> Hermitian-extend -> ``* exp(-j 2 pi f tau)`` -> ``ifft`` ->
``fftshift`` -> truncate to ``Ntaps`` -> Kaiser window -> verify via ``freqz``. Optional
low-latency IIR via complex Levy equation-error + output-error refine + ``zpk2sos``
(never ``scipy.signal.minimum_phase`` per driver). Guard test: ``w = exp(-j 2 pi f T)``
must realize a ``+T`` delay.

References
----------
docs/Phase 2 - Filter Solver.md Section 5.5.
"""

from __future__ import annotations

import numpy as np


def realize_fir(
    weights: np.ndarray,
    frequencies: np.ndarray,
    fs: float,
    n_taps: int,
) -> np.ndarray:
    """Linear-phase FIR realization of per-driver complex weights (Stage P2-5).

    Returns ``[M, n_taps]`` real taps with one shared modeling delay across drivers.
    """
    raise NotImplementedError("Stage P2-5 (deferred): FIR realization not yet implemented.")
