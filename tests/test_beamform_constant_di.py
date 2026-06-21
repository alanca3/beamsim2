"""Luo constant-DI engine + V-CBT (Stage P2-2).

Two validations:

* **Constant-DI engine (Luo MSCD / max-directivity)** — the two-pass optimizer holds the
  generalized directivity index (the Luo objective) *constant across frequency* (exact, by
  construction: ``w^H D w = 0`` => GDI == tau*), is distortionless, and never exceeds the
  per-frequency ceiling from the generalized eigenproblem.
* **V-CBT** — a Keele Constant-Beamwidth-Transducer (a Legendre-amplitude-shaded spherical
  cap of monopoles) radiates a frequency-independent -6 dB beamwidth ~ 0.64*(2*theta0) above
  its cutoff, while the same cap unshaded does not. This shows BeamSimII can represent a
  curved 3-D layout and reproduce a known constant-directivity result (the cap's curvature
  time-of-flight lives in the field, honoring the single-phase-origin cardinal rule).
"""

from __future__ import annotations

import numpy as np
import pytest

from beamsim2.assembly.tensor import build_dataset, stacked_h_full
from beamsim2.beamform.covariance import covariance, directivity_factor, look_vector
from beamsim2.beamform.design import design
from beamsim2.beamform.targets import TargetSpec, build_target
from beamsim2.beamform.weights import luo_mscd, max_directivity
from beamsim2.core.sphere import icosphere
from beamsim2.core.types import ComplexField
from beamsim2.validation.closed_loop import monopole_field

_C = 343.2


def _make_array_dataset(positions, freqs):
    obs = icosphere(4)
    H = monopole_field(np.asarray(positions, float), obs, np.asarray(freqs, float), c=_C)
    inputs = [
        (
            f"d{i}",
            ComplexField(
                frequencies=np.asarray(freqs, float),
                pressure=H[i],
                convergence_flags=np.ones(len(freqs), bool),
            ),
            {"name": f"d{i}", "position": list(positions[i])},
        )
        for i in range(len(positions))
    ]
    ds = build_dataset(inputs, obs, root_attrs={"phase_origin": [0, 0, 0], "speed_of_sound": _C})
    return ds, obs


# ---------------------------------------------------------------------------
# Luo constant-DI engine
# ---------------------------------------------------------------------------
def test_mscd_holds_generalized_di_constant_across_frequency():
    """The constant_di engine achieves the same GDI at every frequency (the Luo objective)."""
    pos = 0.05 * np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, -1]], float)
    freqs = np.array([400.0, 700.0, 1200.0, 2000.0, 3500.0])
    ds, obs = _make_array_dataset(pos, freqs)
    spec = TargetSpec(
        steer_dir=np.array([0, 0, 1.0]), engine="constant_di", accept_halfangle_deg=45.0
    )
    r = design(ds, spec)

    target = build_target(spec, obs, freqs)
    h = stacked_h_full(ds)
    gdis = []
    for f in range(len(freqs)):
        a = covariance(h[:, f, :], obs.weights, mask=target.accept_mask)
        rr = covariance(h[:, f, :], obs.weights, mask=target.reject_mask)
        gdis.append(10.0 * np.log10(directivity_factor(r.weights[:, f], a, rr)))
    gdis = np.array(gdis)
    # All frequencies share one constant GDI, equal to the reported target.
    assert np.ptp(gdis) < 1e-3
    assert gdis.mean() == pytest.approx(r.attrs["constant_gdi_db"], abs=1e-3)
    assert np.all(r.metrics["feasible_mask"])


def test_max_directivity_varies_and_dominates_constant_di():
    """max_directivity reaches each frequency's ceiling, so its GDI >= the constant target."""
    pos = 0.05 * np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, -1]], float)
    freqs = np.array([400.0, 1200.0, 3500.0])
    ds, obs = _make_array_dataset(pos, freqs)
    spec_md = TargetSpec(
        steer_dir=np.array([0, 0, 1.0]), engine="max_directivity", accept_halfangle_deg=45.0
    )
    spec_cd = TargetSpec(
        steer_dir=np.array([0, 0, 1.0]), engine="constant_di", accept_halfangle_deg=45.0
    )
    r_md = design(ds, spec_md)
    r_cd = design(ds, spec_cd)

    target = build_target(spec_md, obs, freqs)
    h = stacked_h_full(ds)
    md_gdi, cd_gdi = [], []
    for f in range(len(freqs)):
        a = covariance(h[:, f, :], obs.weights, mask=target.accept_mask)
        rr = covariance(h[:, f, :], obs.weights, mask=target.reject_mask)
        md_gdi.append(directivity_factor(r_md.weights[:, f], a, rr))
        cd_gdi.append(directivity_factor(r_cd.weights[:, f], a, rr))
    assert np.ptp(md_gdi) > 1e-3  # max-directivity GDI is NOT constant
    assert np.all(
        np.array(md_gdi) >= np.array(cd_gdi) - 1e-9
    )  # ceiling dominates the constant target


def test_mscd_solution_is_distortionless_with_zero_quadratic():
    """luo_mscd returns w with c^H w = 1 and w^H (A - tau R) w = 0."""
    pos = 0.05 * np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
    ds, obs = _make_array_dataset(pos, [1500.0])
    h_f = stacked_h_full(ds)[:, 0, :]
    spec = TargetSpec(steer_dir=np.array([0, 0, 1.0]), accept_halfangle_deg=40.0)
    target = build_target(spec, obs, np.array([1500.0]))
    a = covariance(h_f, obs.weights, mask=target.accept_mask)
    rr = covariance(h_f, obs.weights, mask=target.reject_mask)
    c = look_vector(h_f, target.look_idx)
    _, tau_max = max_directivity(a, rr)
    tau = 0.8 * tau_max
    w = luo_mscd(a, rr, c, tau)
    assert abs(np.conj(c) @ w - 1.0) < 1e-9
    assert abs(np.real(np.conj(w) @ (a - tau * rr) @ w)) < 1e-9
    assert directivity_factor(w, a, rr) == pytest.approx(tau, rel=1e-6)


def test_mscd_raises_for_infeasible_tau():
    """A tau above the directivity ceiling has no constant-DI solution; luo_mscd refuses."""
    pos = 0.05 * np.array([[1, 0, 0], [-1, 0, 0], [0, 0, 1]], float)
    ds, obs = _make_array_dataset(pos, [1500.0])
    h_f = stacked_h_full(ds)[:, 0, :]
    spec = TargetSpec(steer_dir=np.array([0, 0, 1.0]), accept_halfangle_deg=40.0)
    target = build_target(spec, obs, np.array([1500.0]))
    a = covariance(h_f, obs.weights, mask=target.accept_mask)
    rr = covariance(h_f, obs.weights, mask=target.reject_mask)
    c = look_vector(h_f, target.look_idx)
    _, tau_max = max_directivity(a, rr)
    with pytest.raises(ValueError, match="indefinite"):
        luo_mscd(a, rr, c, tau_max * 1.5)


# ---------------------------------------------------------------------------
# V-CBT — Legendre-shaded Constant-Beamwidth Transducer
# ---------------------------------------------------------------------------
def _cbt_cap(Rc: float, theta0_deg: float, n_rings: int):
    """Monopoles on a spherical cap with Keele Legendre amplitude shading.

    Returns positions ``[M, 3]`` and real shading weights ``[M]`` (1 at the cap centre,
    tapering to ~0 at the rim via U(x) = 1 + 0.066x - 1.8x^2 + 0.743x^3, x = psi/theta0).
    """
    th0 = np.deg2rad(theta0_deg)
    pts, shade = [], []

    def u(x):
        return 1.0 + 0.066 * x - 1.8 * x**2 + 0.743 * x**3

    for ir in range(n_rings):
        x = ir / (n_rings - 1)
        psi = th0 * x
        n_phi = 1 if ir == 0 else max(6, int(round(2 * np.pi * Rc * np.sin(psi) / 0.012)))
        for k in range(n_phi):
            ph = 2 * np.pi * k / n_phi
            pts.append(
                Rc * np.array([np.sin(psi) * np.cos(ph), np.sin(psi) * np.sin(ph), np.cos(psi)])
            )
            shade.append(u(x))
    return np.array(pts), np.array(shade)


def _beamwidth_on_arc(positions, weights, freq, *, level_db=-6.0, n=2001, r=2.0):
    """-level_db beamwidth (deg) of a monopole array in the +z xz-plane, measured directly."""
    ang = np.linspace(-np.pi / 2, np.pi / 2, n)
    d = np.column_stack([np.sin(ang), np.zeros(n), np.cos(ang)])  # unit dirs in xz-plane
    obs_pts = r * d  # [n, 3]
    k = 2.0 * np.pi * freq / _C
    dist = np.linalg.norm(obs_pts[:, None, :] - positions[None, :, :], axis=2)  # [n, M]
    p = (weights[None, :] * np.exp(1j * k * dist) / dist).sum(axis=1)  # [n]
    level = 20.0 * np.log10(np.abs(p) / np.max(np.abs(p)) + 1e-300)
    mid = int(np.argmin(np.abs(ang)))
    lo, hi = mid, mid
    while lo > 0 and level[lo] >= level_db:
        lo -= 1
    while hi < n - 1 and level[hi] >= level_db:
        hi += 1
    if lo == 0 or hi == n - 1:
        return np.nan
    return np.rad2deg(ang[hi] - ang[lo])


def test_cbt_constant_beamwidth():
    """A Legendre-shaded CBT cap holds a constant -6 dB beamwidth ~ 0.64*(2*theta0) above cutoff."""
    theta0 = 40.0
    pos, shade = _cbt_cap(Rc=0.30, theta0_deg=theta0, n_rings=11)
    assert len(pos) > 300  # dense enough to avoid spatial aliasing in band
    keele = 0.64 * 2.0 * theta0  # ~51 deg

    band = [4000.0, 6000.0, 9000.0, 13000.0]
    shaded = np.array([_beamwidth_on_arc(pos, shade, f) for f in band])
    unshaded = np.array([_beamwidth_on_arc(pos, np.ones(len(pos)), f) for f in band])

    # Shaded CBT: beamwidth is close to the Keele value and nearly constant across the band.
    assert np.all(np.isfinite(shaded))
    assert np.all(np.abs(shaded - keele) < 8.0), f"shaded beamwidth off Keele: {shaded}"
    assert np.nanstd(shaded) < 2.5, f"shaded beamwidth not constant: {shaded}"
    # The same cap UNSHADED does not form the CBT beam (beamwidths nowhere near Keele) —
    # the Legendre shading is what produces the constant directivity.
    assert (
        np.nanmean(np.abs(unshaded - keele)) > 20.0
    ), f"unshaded unexpectedly CBT-like: {unshaded}"
