"""The vectorized ball-arithmetic checker gives the same bounds as checker.check,
up to a slightly larger rounding allowance, and rejects what it rejects."""

import dataclasses

import numpy as np
import pytest

from certsim.checker import CertificateError, check
from certsim.fastcheck import fast_check
from certsim.solver import default_grid, solve
from certsim.spec import Box, default_spec, lesion_tiles, node_masks

MONO = default_spec(voxels=(), tiles=lesion_tiles(r_max=4e-3, depth=6e-3, step=1e-3),
                    regions=(Box(0.0, 3e-3, -8e-3, -5e-3),))
BIPOLAR = default_spec(voxels=(), regions=(), tiles=lesion_tiles(r_max=3e-3, depth=4e-3, step=1e-3),
                       L=2e-3, ring_spacing=4e-3, ring_length=2e-3)


def overlaps(a, b):
    return a.lo <= b.hi and b.lo <= a.hi


def width(iv):
    return iv.hi - iv.lo


def assert_agree(fast, slow, slack=2e-3):
    """Both enclose the same exact quantities: they overlap, and the fast one is barely wider.

    Double-precision balls carry a radius of order 1e-7 relative on eta (the rigorous checker's is
    of order 1e-70), so the allowance (1e-4 for eta) is relative to the value, not to the rigorous width.
    """
    pairs = [(fast.eta, slow.eta, 1e-4), (fast.conductance, slow.conductance, 1e-8)]
    pairs += [(a, b, 1e-8) for a, b in zip(fast.tile_rms_field, slow.tile_rms_field)]
    pairs += [(a, b, 1e-8) for a, b in zip(fast.region_rms_field, slow.region_rms_field)]
    for a, b, rel in pairs:
        assert overlaps(a, b)
        assert width(a) <= width(b) * (1 + slack) + rel * abs(b.mid)


@pytest.mark.parametrize("spec,degree", [(MONO, 1), (MONO, 2), (MONO, 3), (BIPOLAR, 1), (BIPOLAR, 2)])
def test_agrees_with_rigorous_checker(spec, degree):
    r, z = default_grid(spec, h_fine=0.5e-3)
    cert = solve(spec, r, z, degree=degree)
    assert_agree(fast_check(spec, cert), check(spec, cert))


@pytest.mark.parametrize("seed", range(4))
def test_agrees_on_adversarial_certificates(seed):
    """Random noise of very different sizes in the free values of v and psi: the bounds are then
    driven by large radii and cancellations, not by a near-exact solution."""
    rng = np.random.default_rng(seed)
    r, z = default_grid(MONO, h_fine=1e-3)
    cert = solve(MONO, r, z, degree=2)
    masks = node_masks(MONO, r, z, 2)
    v_free = masks["active"] & ~(masks["electrode"] | masks["ground"])
    psi_free = masks["active"] & ~(masks["axis"] | masks["shaft"])
    scale_v, scale_p = 10.0 ** rng.uniform(-9, -1), 10.0 ** rng.uniform(-12, -3)
    v = cert.v + v_free * scale_v * rng.standard_normal(cert.v.shape)
    psi = cert.psi + psi_free * scale_p * rng.standard_normal(cert.psi.shape)
    noisy = dataclasses.replace(cert, v=v, psi=psi)
    fast, slow = fast_check(MONO, noisy), check(MONO, noisy)
    for a, b in [(fast.conductance, slow.conductance), (fast.eta, slow.eta)] + list(
            zip(fast.tile_rms_field, slow.tile_rms_field)):
        assert overlaps(a, b)


def test_rejects_wrong_boundary_values():
    r, z = default_grid(MONO, h_fine=1e-3)
    cert = solve(MONO, r, z, degree=2)
    masks = node_masks(MONO, r, z, 2)
    for field, mask in (("v", "electrode"), ("v", "ground"), ("psi", "axis"), ("psi", "shaft")):
        arr = getattr(cert, field).copy()
        j, i = np.argwhere(masks[mask])[len(np.argwhere(masks[mask])) // 2]
        arr[j, i] = arr[j, i] * (1 + 1e-9) + 1e-12
        with pytest.raises(CertificateError):
            fast_check(MONO, dataclasses.replace(cert, **{field: arr}))


@pytest.mark.parametrize("bug", ["planar", "swap_sigma"])
def test_wrong_physics_gets_no_support(bug):
    r, z = default_grid(MONO, h_fine=0.5e-3)
    cert = solve(MONO, r, z, degree=2, bug=bug)
    rep = fast_check(MONO, cert)
    assert not rep.conductance.supports(cert.claims["conductance"], 1e-3)


def test_voxels_are_not_supported():
    with pytest.raises(ValueError):
        spec = default_spec()  # has axis voxels
        r, z = default_grid(spec, h_fine=1e-3)
        fast_check(spec, solve(spec, r, z, degree=1))
