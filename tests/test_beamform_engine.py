"""V-CARDIOID + engine tests for the LS/MVDR/LCMV beamformer (Stage P2-1).

Two layers, per the gameplan:

* **Metric / forward-model gate** — analytic first-order patterns integrate to their
  textbook directivity indices (cardioid 4.771, supercardioid 5.719, hypercardioid 6.021,
  dipole 4.771, omni 0 dB), confirming the quadrature DI metric.
* **Engine gate** — the designer's LS / delay-sum / MVDR / LCMV modes steer to the
  commanded direction, reproduce the cardioid where the array supports it, respect the WNG
  floor, and place LCMV nulls. (Constant-directivity-vs-frequency *shape* is gated by V-CBT
  in P2-2; this stage gates steering + achievable-shape + robustness.)
"""

from __future__ import annotations

import numpy as np
import pytest

from beamsim2.assembly.tensor import build_dataset
from beamsim2.beamform.design import design
from beamsim2.beamform.targets import TargetSpec
from beamsim2.core.sphere import icosphere
from beamsim2.core.types import ComplexField
from beamsim2.validation.closed_loop import monopole_field
from beamsim2.validation.power_di import directivity_index

_C = 343.2


def _di_of_pattern(pattern: np.ndarray, obs) -> float:
    return float(directivity_index(pattern[None, :].astype(np.complex128), obs.weights)[0])


def _make_endfire_dataset(freqs, half_spacing=0.043):
    """Two-monopole end-fire dataset on an icosphere-2562 grid."""
    obs = icosphere(4)
    pos = [np.array([0.0, 0.0, -half_spacing]), np.array([0.0, 0.0, half_spacing])]
    H = monopole_field(np.array(pos), obs, np.asarray(freqs, float), c=_C)  # [2,F,N]
    inputs = [
        (
            f"d{i}",
            ComplexField(
                frequencies=np.asarray(freqs, float),
                pressure=H[i],
                convergence_flags=np.ones(len(freqs), bool),
            ),
            {"name": f"d{i}", "position": list(pos[i])},
        )
        for i in range(2)
    ]
    ds = build_dataset(inputs, obs, root_attrs={"phase_origin": [0, 0, 0], "speed_of_sound": _C})
    return ds, obs


# ---------------------------------------------------------------------------
# Metric / forward-model gate — first-order DI anchors
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "a,expected_db",
    [
        (1.0, 0.0),  # omni
        (0.5, 4.7712),  # cardioid
        (0.366, 5.7188),  # supercardioid (closed form 10log10(1/(a^2+(1-a)^2/3)))
        (0.25, 6.0206),  # hypercardioid (max first-order DI)
        (0.0, 4.7712),  # figure-8 / dipole
    ],
)
def test_first_order_di_anchors(a, expected_db):
    """Analytic first-order pattern integrates to its textbook DI on the dense grid."""
    obs = icosphere(5)  # 10242 points for accuracy
    cos_ang = obs.unit_vectors[:, 2]
    pattern = (a + (1.0 - a) * cos_ang).astype(np.complex128)
    assert _di_of_pattern(pattern, obs) == pytest.approx(expected_db, abs=0.02)


# ---------------------------------------------------------------------------
# Engine gate
# ---------------------------------------------------------------------------
def test_ls_reproduces_cardioid_in_achievable_regime():
    """LS-to-cardioid on a compact (low-ka) array reproduces DI ~ 4.77 and peaks at +z."""
    ds, obs = _make_endfire_dataset([300.0])
    spec = TargetSpec(
        mode="preset",
        preset="cardioid",
        steer_dir=np.array([0.0, 0.0, 1.0]),
        engine="ls",
        wng_floor_db=-60.0,
    )
    r = design(ds, spec)
    assert r.metrics["di_db"][0] == pytest.approx(4.77, abs=0.2)
    peak = int(np.argmax(np.abs(r.steered_field[0])))
    assert obs.unit_vectors[peak, 2] > 0.95  # main lobe near +z


def test_ls_omni_gives_zero_di():
    ds, _ = _make_endfire_dataset([500.0])
    r = design(ds, TargetSpec(mode="preset", preset="omni", engine="ls", wng_floor_db=-60.0))
    assert r.metrics["di_db"][0] == pytest.approx(0.0, abs=0.3)


@pytest.mark.parametrize("engine", ["delay_sum", "mvdr", "ls"])
@pytest.mark.parametrize("steer", [np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, -1.0])])
def test_engines_steer_to_commanded_direction(engine, steer):
    """Every engine's main lobe lands at the commanded steering direction."""
    ds, obs = _make_endfire_dataset([1500.0])
    spec = TargetSpec(mode="steering_only", steer_dir=steer, engine=engine, wng_floor_db=-6.0)
    r = design(ds, spec)
    peak = int(np.argmax(np.abs(r.steered_field[0])))
    assert obs.unit_vectors[peak] @ steer > 0.9


def test_mvdr_respects_wng_floor():
    """MVDR achieves at least the requested WNG floor where feasible."""
    ds, _ = _make_endfire_dataset([400.0, 1500.0])
    floor = -3.0
    r = design(
        ds,
        TargetSpec(
            mode="steering_only", steer_dir=np.array([0, 0, 1.0]), engine="mvdr", wng_floor_db=floor
        ),
    )
    feas = r.metrics["feasible_mask"]
    assert np.all(r.metrics["wng_db"][feas] >= floor - 0.2)


def test_wng_floor_above_ceiling_is_flagged_infeasible():
    """A WNG floor above the M-driver ceiling (10log10 2 ~ 3 dB) is flagged, not faked."""
    ds, _ = _make_endfire_dataset([800.0])
    r = design(
        ds,
        TargetSpec(
            mode="steering_only", steer_dir=np.array([0, 0, 1.0]), engine="mvdr", wng_floor_db=8.0
        ),
    )
    assert not r.metrics["feasible_mask"][0]


def test_lcmv_places_null():
    """LCMV drives the response toward -z to ~zero while keeping unit response at +z."""
    ds, obs = _make_endfire_dataset([1500.0])
    r = design(
        ds,
        TargetSpec(
            mode="steering_only",
            steer_dir=np.array([0, 0, 1.0]),
            nulls=[np.array([0.0, 0.0, -1.0])],
            engine="lcmv",
            wng_floor_db=-6.0,
        ),
    )
    P = r.steered_field[0]
    look = int(np.argmax(obs.unit_vectors[:, 2]))
    null = int(np.argmin(obs.unit_vectors[:, 2]))
    null_depth = 20.0 * np.log10(np.abs(P[null]) / np.abs(P[look]))
    assert null_depth < -40.0


def test_wng_invariants_monotone_and_distortionless():
    """V-WNG: loaded MVDR is distortionless; WNG rises monotonically with loading and the
    most-robust limit is never more directive than the superdirective (unloaded) one."""
    from beamsim2.assembly.tensor import stacked_h_full
    from beamsim2.beamform.covariance import covariance, look_vector
    from beamsim2.beamform.regularize import loaded_mvdr_weights, white_noise_gain_db
    from beamsim2.validation.power_di import directivity_index

    ds, obs = _make_endfire_dataset([1500.0])
    h_f = stacked_h_full(ds)[:, 0, :]  # [M, N]
    look = int(np.argmax(obs.unit_vectors[:, 2]))
    c = look_vector(h_f, look)
    r = covariance(h_f, obs.weights)

    eps_sweep = [0.0, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e4]
    wngs, dis = [], []
    for eps in eps_sweep:
        w = loaded_mvdr_weights(r, c, eps)
        assert abs(np.conj(c) @ w - 1.0) < 1e-9  # distortionless
        wngs.append(white_noise_gain_db(w, c))
        dis.append(float(directivity_index((w @ h_f)[None, :], obs.weights)[0]))
    assert np.all(np.diff(wngs) >= -1e-9), "WNG must be monotone increasing in loading"
    # The least-loaded (most superdirective) beamformer has the highest pattern directivity;
    # the most-robust limit has the lowest. (Per-step DI is not strictly monotone.)
    assert dis[0] >= dis[-1] - 1e-9, "max-robustness DI must not exceed superdirective DI"

    # Max-robustness limit: weights become parallel to the look vector.
    w_inf = loaded_mvdr_weights(r, c, 1e8)
    cosine = abs(np.vdot(w_inf, c)) / (np.linalg.norm(w_inf) * np.linalg.norm(c))
    assert cosine == pytest.approx(1.0, abs=1e-6)


def test_design_result_structure():
    ds, _ = _make_endfire_dataset([300.0, 1000.0])
    r = design(ds, TargetSpec(engine="ls"))
    assert r.weights.shape == (2, 2)
    assert r.steered_field.shape[0] == 2
    for key in ("di_db", "beamwidth_deg", "target_error_db", "wng_db", "feasible_mask"):
        assert key in r.metrics
        assert len(r.metrics[key]) == 2
    assert r.attrs["convention"].startswith("house")
