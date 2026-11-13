"""The checker accepts honest certificates, rejects inadmissible ones, and
never lets a wrong solver's claims pass as certified."""

import dataclasses

import numpy as np
import pytest

from certsim.checker import CertificateError, check
from certsim.solver import default_grid, solve
from certsim.spec import default_spec, node_masks

SPEC = default_spec()


@pytest.fixture(scope="module")
def coarse():
    r, z = default_grid(SPEC, h_fine=0.5e-3, h0=2e-5, kappa=0.2)
    cert = solve(SPEC, r, z, degree=1)
    return cert, check(SPEC, cert)


@pytest.fixture(scope="module")
def medium():
    r, z = default_grid(SPEC, h_fine=0.25e-3, h0=5e-6, kappa=0.2, h0_box=2e-5)
    cert = solve(SPEC, r, z, degree=1)
    return cert, check(SPEC, cert)


def overlaps(a, b):
    return a.lo <= b.hi and b.lo <= a.hi


def test_honest_certificate_bounds_are_consistent(coarse, medium):
    (c_cert, c_rep), (m_cert, m_rep) = coarse, medium
    # Dirichlet energies are upper bounds of the true conductance, so a finer
    # solver's energy can never fall below a valid lower bound.
    assert c_rep.conductance.lo <= m_cert.claims["conductance"]
    assert m_rep.conductance.lo <= c_cert.claims["conductance"]
    assert overlaps(c_rep.conductance, m_rep.conductance)
    assert m_rep.conductance.rel_width < c_rep.conductance.rel_width
    for a, b in zip(c_rep.voxel_mean_dudz, m_rep.voxel_mean_dudz):
        assert overlaps(a, b)
    assert overlaps(c_rep.region_rms_field[0], m_rep.region_rms_field[0])


def test_honest_claims_are_supported(medium):
    cert, rep = medium
    assert rep.conductance.supports(cert.claims["conductance"], rel_tol=5e-3)
    # raw voxel means are not guaranteed to be inside, but an honest solver's
    # should be within one interval width of it
    for claim, iv in zip(cert.claims["voxel_mean"], rep.voxel_mean_dudz):
        w = iv.hi - iv.lo
        assert iv.lo - w <= claim <= iv.hi + w


def _corrupt(cert, **arrays):
    return dataclasses.replace(cert, **arrays)


def _masks(cert):
    return node_masks(SPEC, cert.r, cert.z)


@pytest.mark.parametrize(
    "mutation, message",
    [
        ("electrode", "electrode voltage"),
        ("ground", "zero on the ground"),
        ("axis", "axis"),
        ("shaft", "shaft"),
        ("adjoint", "Dirichlet nodes"),
        ("grid", "breakpoints"),
    ],
)
def test_inadmissible_certificates_are_rejected(coarse, mutation, message):
    cert, _ = coarse
    m = _masks(cert)
    v, psi, zh = cert.v.copy(), cert.psi.copy(), cert.zh.copy()
    if mutation == "electrode":
        v[np.argwhere(m["electrode"])[3][0], np.argwhere(m["electrode"])[3][1]] *= 1.0001
        bad = _corrupt(cert, v=v)
    elif mutation == "ground":
        j, i = np.argwhere(m["ground"])[5]
        v[j, i] = 1e-9
        bad = _corrupt(cert, v=v)
    elif mutation == "axis":
        j, i = np.argwhere(m["axis"])[10]
        psi[j, i] = 1e-12
        bad = _corrupt(cert, psi=psi)
    elif mutation == "shaft":
        j, i = np.argwhere(m["shaft"])[2]
        psi[j, i] += 1e-12
        bad = _corrupt(cert, psi=psi)
    elif mutation == "adjoint":
        j, i = np.argwhere(m["electrode"])[1]
        zh[4, j, i] = 1e-9
        bad = _corrupt(cert, zh=zh)
    else:
        # drop the grid line through the voxel radius
        k = int(np.nonzero(cert.r == SPEC.voxels[0].r_hi)[0][0])
        keep = np.ones(len(cert.r), dtype=bool)
        keep[k] = False
        bad = _corrupt(cert, r=cert.r[keep], v=v[:, keep], psi=psi[:, keep],
                       zh=zh[:, :, keep], psi_adj=cert.psi_adj[:, :, keep])
    with pytest.raises(CertificateError, match=message):
        check(SPEC, bad)


def test_noisy_but_admissible_certificate_still_brackets(coarse):
    cert, rep = coarse
    m = _masks(cert)
    rng = np.random.default_rng(0)
    v = cert.v.copy()
    free = m["active"] & ~m["electrode"] & ~m["ground"]
    v[free] += 1e-3 * rng.standard_normal(free.sum())
    noisy = check(SPEC, _corrupt(cert, v=v))
    assert noisy.conductance.hi - noisy.conductance.lo > rep.conductance.hi - rep.conductance.lo
    assert overlaps(noisy.conductance, rep.conductance)
    for a, b in zip(noisy.voxel_mean_dudz, rep.voxel_mean_dudz):
        assert overlaps(a, b)


@pytest.mark.parametrize("bug", ["planar", "swap_sigma"])
def test_wrong_physics_is_caught(medium, bug):
    """A solver that models the wrong physics can still emit an admissible
    certificate, but the checker evaluates it with the true physics, so the
    certificate no longer supports the solver's own claim."""
    _, honest = medium
    r, z = default_grid(SPEC, h_fine=0.25e-3, h0=5e-6, kappa=0.2, h0_box=2e-5)
    cert = solve(SPEC, r, z, degree=1, bug=bug)
    rep = check(SPEC, cert)
    assert not rep.conductance.supports(cert.claims["conductance"], rel_tol=5e-3)
    # the bounds stay valid: they still bracket the honest result
    assert overlaps(rep.conductance, honest.conductance)


def test_lesion_depth_is_monotone_in_voltage(medium):
    _, rep = medium
    depths = [rep.lesion_depth(v, threshold=7e4)[0] for v in (500, 1000, 1500, 2000)]
    assert depths == sorted(depths)
    assert depths[-1] > depths[0]


def test_disk_on_half_space_matches_newman():
    """Independent physics check. With insulating blood the tip is a disk
    electrode on a tissue half-space: R = 1/(4 sigma a) (Newman 1966). The
    grounded box at 50-71 mm lowers it by 1/(2 pi sigma rho) for rho in that
    range (point-source far field), so the true value lies in about
    [800 - 12.7, 800 - 9.0] ohm."""
    spec = default_spec(sigma_blood=1e-9, voxels=(), regions=())
    r, z = default_grid(spec, h_fine=0.25e-3, h0=5e-6, kappa=0.2)
    rep = check(spec, solve(spec, r, z, degree=1))
    newman = 1 / (4 * spec.sigma_tissue * spec.a)
    assert overlaps(rep.impedance, type(rep.impedance)(newman - 12.8, newman - 8.9))
