"""V-WNG-SCALE (Chunk 5a): the WNG floor must be invariant to the absolute level of H.

The keystone Chunk-5 bug: ``white_noise_gain_db`` measured the *absolute* matched-field
power ``||c||^2`` (Pa^2), not the dimensionless array gain. For real BEM data
(``|H| ~ 1e-3`` Pa) the WNG ceiling sat tens of dB below any usable floor, so every bin
was flagged infeasible and every adaptive engine clamped to maximum loading (collapsing
to the omni / delay-sum corner). The synthetic monopole fixtures used elsewhere have
``|H| ~ 1`` (``||c||^2 ~ M``), which *masked* the bug — the metric read identically before
and after the fix there.

This gate locks the fixed property directly: a global scale on ``H`` (the exact thing that
varies between a unit-amplitude monopole and Pa-valued BEM output) must leave the design's
``feasible_mask``, ``di_db`` and ``wng_db`` unchanged. It also checks the ceiling is the
scale-free ``10 log10(M)`` and that a previously-unreachable floor on faint data is now met.

Engineering convention exp(-jwt), outgoing exp(+jkr) (repo ``monopole_field``).
"""

from __future__ import annotations

import numpy as np
import pytest

from beamsim2.assembly.tensor import build_dataset, stacked_h_full
from beamsim2.beamform.covariance import look_vector
from beamsim2.beamform.design import design
from beamsim2.beamform.regularize import max_white_noise_gain_db, white_noise_gain_db
from beamsim2.beamform.targets import TargetSpec
from beamsim2.core.sphere import icosphere
from beamsim2.core.types import ComplexField

_C = 343.2
# Two opposed drivers on the front/back of one box (the user's run2 geometry: d ~ 0.16 m
# along +x, the loudspeaker front). The cardioid is steered along that front axis.
_POS = [np.array([0.08, 0.0, 0.0]), np.array([-0.08, 0.0, 0.0])]
_FRONT = np.array([1.0, 0.0, 0.0])
_BAND = np.geomspace(100.0, 500.0, 6)


def _dataset(freqs, *, scale: float = 1.0):
    """Dual-opposed monopole dataset on icosphere(4), H multiplied by ``scale``."""
    from beamsim2.validation.closed_loop import monopole_field

    obs = icosphere(4)
    H = scale * monopole_field(np.asarray(_POS, float), obs, np.asarray(freqs, float), c=_C)
    inputs = [
        (
            f"d{i}",
            ComplexField(
                frequencies=np.asarray(freqs, float),
                pressure=H[i],
                convergence_flags=np.ones(len(freqs), bool),
            ),
            {"name": f"d{i}", "position": list(_POS[i])},
        )
        for i in range(len(_POS))
    ]
    ds = build_dataset(inputs, obs, root_attrs={"phase_origin": [0, 0, 0], "speed_of_sound": _C})
    return ds, obs


@pytest.mark.parametrize("engine", ["ls", "mvdr", "max_directivity", "constant_di"])
def test_design_is_invariant_to_absolute_h_level(engine):
    """Scaling H by 1e-3 (unit monopole -> Pa-valued BEM) must not change the design metrics."""
    spec_kwargs = dict(
        mode="preset",
        preset="cardioid",
        steer_dir=_FRONT,
        engine=engine,
        wng_floor_db=-6.0,
        directivity_mode="index",
    )
    ds1, _ = _dataset(_BAND, scale=1.0)
    ds2, _ = _dataset(_BAND, scale=1e-3)  # faint, BEM-like magnitudes
    r1 = design(ds1, TargetSpec(**spec_kwargs))
    r2 = design(ds2, TargetSpec(**spec_kwargs))

    assert np.array_equal(r1.metrics["feasible_mask"], r2.metrics["feasible_mask"]), (
        f"{engine}: feasibility changed with absolute H level "
        f"({int(r1.metrics['feasible_mask'].sum())} vs {int(r2.metrics['feasible_mask'].sum())})"
    )
    assert np.allclose(
        r1.metrics["di_db"], r2.metrics["di_db"], atol=1e-6
    ), "DI not scale-invariant"
    w1 = np.asarray(r1.metrics["wng_db"], float)
    w2 = np.asarray(r2.metrics["wng_db"], float)
    finite = np.isfinite(w1) & np.isfinite(w2)
    assert np.allclose(w1[finite], w2[finite], atol=1e-6), "WNG not scale-invariant"


def test_matched_field_ceiling_is_10log10_M_regardless_of_level():
    """The WNG ceiling is the scale-free 10 log10(M) (= 3.01 dB for M=2), not 10 log10(||c||^2)."""
    for scale in (1.0, 1e-3, 1e3):
        ds, _ = _dataset(_BAND, scale=scale)
        h = stacked_h_full(ds)
        uv = ds.directions.unit_vectors
        look = int(np.argmax(uv @ _FRONT))
        for f in range(len(_BAND)):
            c = look_vector(h[:, f, :], look)
            assert max_white_noise_gain_db(c) == pytest.approx(10.0 * np.log10(2.0), abs=1e-9)
            # The matched-field weights w=c/M sit exactly at that ceiling.
            assert white_noise_gain_db(c / 2.0, c) == pytest.approx(10.0 * np.log10(2.0), abs=1e-9)


def test_faint_data_is_now_feasible():
    """A -6 dB floor is reachable on faint (1e-3) data — the exact regression that failed."""
    ds, _ = _dataset(_BAND, scale=1e-3)
    r = design(
        ds,
        TargetSpec(
            mode="preset",
            preset="cardioid",
            steer_dir=_FRONT,
            engine="ls",
            wng_floor_db=-6.0,
            directivity_mode="index",
        ),
    )
    assert r.metrics["feasible_mask"].all(), "faint-data cardioid should meet the -6 dB floor"
    # And it is a real cardioid in-band, not the collapsed omni the bug produced.
    assert float(np.max(r.metrics["di_db"])) > 4.0, "DI collapsed -> WNG normalization not applied"
