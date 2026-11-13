"""Bounds over a conductivity-ratio range contain independent point bounds,
including at ratios that were not used to build the range certificate."""

import dataclasses

import numpy as np
import pytest

from certsim.checker import CertificateError, check
from certsim.ranges import check_range
from certsim.solver import default_grid, solve
from certsim.spec import default_spec, lesion_tiles

SPEC = default_spec(voxels=(), tiles=lesion_tiles(r_max=3e-3, depth=4e-3, step=1e-3))
RHOS = [2.0, 2.75, 3.5]


def at(rho):
    return dataclasses.replace(SPEC, sigma_blood=rho * SPEC.sigma_tissue)


@pytest.fixture(scope="module")
def setup():
    r, z = default_grid(SPEC, h_fine=0.5e-3)
    certs = [solve(at(x), r, z, degree=2) for x in RHOS]
    return r, z, certs, check_range(SPEC, certs, RHOS)


@pytest.mark.parametrize("rho", [2.0, 2.3, 3.2, 3.5])
def test_range_contains_point_bounds(setup, rho):
    r, z, _, rep = setup
    pt = check(at(rho), solve(at(rho), r, z, degree=2))
    for R, Q in zip(rep.tile_rms_field, pt.tile_rms_field):
        assert R.lo <= Q.lo and Q.hi <= R.hi


def test_range_rejects_inadmissible_member(setup):
    _, _, certs, _ = setup
    bad_v = certs[1].v.copy()
    bad_v[0, 0] = 0.5  # the axis node at the bottom is grounded
    bad = [certs[0], dataclasses.replace(certs[1], v=bad_v), certs[2]]
    with pytest.raises(CertificateError):
        check_range(SPEC, bad, RHOS)


def test_threshold_range_is_conservative(setup):
    *_, rep = setup
    single = np.array(rep.lesion_map(1000.0, 7e4))
    ranged = np.array(rep.lesion_map(1000.0, (6e4, 8e4)))
    assert np.all(ranged[ranged == 1] == single[ranged == 1])  # lethal for the range => lethal at 700
    assert np.all(single[ranged == -1] == -1)  # spared for the range => spared at 700
    assert (ranged == 0).sum() >= (single == 0).sum()


def test_range_ignores_the_specified_blood_conductivity(setup):
    """Cells are tissue or blood by geometry; spec.sigma_blood plays no part in a range,
    so a specification with equal conductivities gives the same bounds."""
    _, _, certs, rep = setup
    same = check_range(dataclasses.replace(SPEC, sigma_blood=SPEC.sigma_tissue), certs, RHOS)
    assert same.eta_max == rep.eta_max
    for a, b in zip(same.tile_rms_field, rep.tile_rms_field):
        assert (a.lo, a.hi) == (b.lo, b.hi)
