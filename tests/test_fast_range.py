"""The vectorized range checker gives the same bounds as ranges.check_range, up to the
double-precision rounding allowance, and rejects what it rejects."""

import dataclasses
from fractions import Fraction

import numpy as np
import pytest

from certsim import fastcheck
from certsim.checker import CertificateError, check
from certsim.fastcheck import TINY, Ball, _bilin, fast_check_range
from certsim.ranges import check_range
from certsim.solver import default_grid, solve
from certsim.spec import Box, default_spec, lesion_tiles, node_masks

MONO = default_spec(voxels=(), tiles=lesion_tiles(r_max=3e-3, depth=4e-3, step=1e-3),
                    regions=(Box(0.0, 3e-3, -8e-3, -5e-3),))
BIPOLAR = default_spec(voxels=(), regions=(), tiles=lesion_tiles(r_max=3e-3, depth=4e-3, step=1e-3),
                       L=2e-3, ring_spacing=4e-3, ring_length=2e-3)
RHOS = [2.0, 2.75, 3.5]


def at(spec, rho):
    return dataclasses.replace(spec, sigma_blood=rho * spec.sigma_tissue)


def overlaps(a, b):
    return a.lo <= b.hi and b.lo <= a.hi


def assert_agree(fast, slow):
    """Both enclose the same exact quantities over the range: every interval overlaps, eta_max
    agrees to the double-precision allowance, and the fast intervals are barely wider."""
    assert abs(fast.eta_max - slow.eta_max) <= 1e-6 * slow.eta_max
    for a, b in list(zip(fast.tile_rms_field, slow.tile_rms_field)) + list(
            zip(fast.region_rms_field, slow.region_rms_field)):
        assert overlaps(a, b)
        assert (a.hi - a.lo) <= (b.hi - b.lo) * (1 + 2e-3) + 1e-6 * abs(b.hi)


def certs_for(spec, h_fine, degree, rhos=RHOS):
    r, z = default_grid(spec, h_fine=h_fine)
    return r, z, [solve(at(spec, x), r, z, degree=degree) for x in rhos]


@pytest.mark.parametrize("spec,degree", [(MONO, 1), (MONO, 2), (BIPOLAR, 2)])
def test_agrees_with_rigorous_range_checker(spec, degree):
    _, _, certs = certs_for(spec, 0.5e-3, degree)
    assert_agree(fast_check_range(spec, certs, RHOS), check_range(spec, certs, RHOS))


@pytest.mark.parametrize("seed", range(3))
def test_agrees_on_adversarial_certificates(seed):
    """Noise of very different sizes in the free values of every member: the bounds are then
    driven by large radii and cancellations between members, not by near-exact solutions."""
    rng = np.random.default_rng(seed)
    r, z, certs = certs_for(MONO, 1e-3, 2)
    masks = node_masks(MONO, r, z, 2)
    v_free = masks["active"] & ~(masks["electrode"] | masks["ground"])
    psi_free = masks["active"] & ~(masks["axis"] | masks["shaft"])
    noisy = []
    for c in certs:
        sv, sp = 10.0 ** rng.uniform(-9, -1), 10.0 ** rng.uniform(-12, -3)
        noisy.append(dataclasses.replace(c, v=c.v + v_free * sv * rng.standard_normal(c.v.shape),
                                         psi=c.psi + psi_free * sp * rng.standard_normal(c.psi.shape)))
    fast, slow = fast_check_range(MONO, noisy, RHOS), check_range(MONO, noisy, RHOS)
    assert abs(fast.eta_max - slow.eta_max) <= 1e-6 * slow.eta_max
    for a, b in list(zip(fast.tile_rms_field, slow.tile_rms_field)) + list(
            zip(fast.region_rms_field, slow.region_rms_field)):
        assert overlaps(a, b)


def test_range_contains_point_bounds():
    """At ratios inside the range, including ones not used to build it, the fast range bounds
    contain the 256-bit point bounds."""
    r, z, certs = certs_for(MONO, 0.5e-3, 2)
    rep = fast_check_range(MONO, certs, RHOS)
    for rho in (2.0, 2.3, 3.2):
        pt = check(at(MONO, rho), solve(at(MONO, rho), r, z, degree=2))
        for R, Q in zip(rep.tile_rms_field, pt.tile_rms_field):
            assert R.lo <= Q.lo and Q.hi <= R.hi


def test_rejects_inadmissible_member():
    _, _, certs = certs_for(MONO, 1e-3, 2)
    bad_v = certs[1].v.copy()
    bad_v[0, 0] = 0.5  # the axis node at the bottom is grounded
    with pytest.raises(CertificateError):
        fast_check_range(MONO, [certs[0], dataclasses.replace(certs[1], v=bad_v), certs[2]], RHOS)


def test_rejects_members_on_different_grids():
    _, _, certs = certs_for(MONO, 1e-3, 2)
    r2, z2 = default_grid(MONO, h_fine=0.5e-3)
    other = solve(at(MONO, RHOS[1]), r2, z2, degree=2)
    with pytest.raises(CertificateError):
        fast_check_range(MONO, [certs[0], other, certs[2]], RHOS)


def test_voxels_are_not_supported():
    spec = default_spec()  # has axis voxels
    r, z = default_grid(spec, h_fine=1e-3)
    certs = [solve(at(spec, x), r, z, degree=1) for x in RHOS]
    with pytest.raises(CertificateError):
        fast_check_range(spec, certs, RHOS)


def test_degree3_single_segment_with_a_region_equal_to_a_tile_and_small_chunks():
    """A cell that lies in both a region and a tile is counted in both; one segment; degree 3;
    chunks smaller than a grid row, so the sliding window crosses many chunk boundaries."""
    tiles = lesion_tiles(r_max=2e-3, depth=2e-3, step=1e-3)
    t0 = tiles[0]
    spec = default_spec(voxels=(), tiles=tiles, regions=(Box(t0.r_lo, t0.r_hi, t0.z_lo, t0.z_hi),))
    rhos = [2.0, 3.0]
    _, _, certs = certs_for(spec, 0.5e-3, 3, rhos)
    slow = check_range(spec, certs, rhos)
    for chunk in (37, 20000):
        assert_agree(fast_check_range(spec, certs, rhos, chunk=chunk), slow)
    fast = fast_check_range(spec, certs, rhos, chunk=37)
    assert (fast.region_rms_field[0].lo, fast.region_rms_field[0].hi) == (fast.tile_rms_field[0].lo, fast.tile_rms_field[0].hi)


def test_overlapping_tiles_are_refused():
    tiles = lesion_tiles(r_max=2e-3, depth=2e-3, step=1e-3)
    spec = default_spec(voxels=(), regions=(), tiles=tiles + tiles[:1])
    _, _, certs = certs_for(spec, 1e-3, 1)
    with pytest.raises(ValueError):
        fast_check_range(spec, certs, RHOS)


def test_refuses_to_run_when_subnormals_are_flushed(monkeypatch):
    _, _, certs = certs_for(MONO, 1e-3, 1)
    monkeypatch.setattr(fastcheck, "_subnormals_preserved", lambda: False)
    with pytest.raises(RuntimeError):
        fast_check_range(MONO, certs, RHOS)


def test_bilinear_form_encloses_exact_value_including_underflow():
    """_bilin against exact rational arithmetic at the ball midpoints, for ordinary inputs, heavy
    cancellation, and mixed scales where products inside the matrix products underflow."""
    rng = np.random.default_rng(1)
    p = 2
    B, D = p + 1, 2 * p + 2
    bb = np.arange(B)
    w = Ball.of(1.0 / (bb[:, None] + bb[None, :] + 1.0))
    cases = [(1.0, 1.0, 1.0), (1e-318, 1e-6, 1e3), (1e-310, 1e-12, 1e8), (5e-324, 1e-5, 1e10)]
    for sx, sm, sy in cases:
        X = Ball(rng.standard_normal((4, B, D)) * sx)
        Y = Ball(rng.standard_normal((4, B, D)) * sy)
        M = Ball(np.abs(rng.standard_normal((4, 2 * D))) * sm)
        out = _bilin(X, Y, M, w)
        for c in range(4):
            exact = sum(Fraction(float(w.m[b1, b2])) * Fraction(float(X.m[c, b1, k1])) * Fraction(float(Y.m[c, b2, k2]))
                        * Fraction(float(M.m[c, k1 + k2]))
                        for b1 in range(B) for b2 in range(B) for k1 in range(D) for k2 in range(D))
            # w is itself a ball around 1/(b1+b2+1); its midpoints are what was multiplied
            assert abs(exact - Fraction(float(out.m[c]))) <= Fraction(float(out.r[c]))
    assert TINY == 2.0**-1074
