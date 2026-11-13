"""Degree-p certificates: tighter bounds, consistent with degree 1, and the
extra boundary nodes inside cell edges are checked like the corners."""

import dataclasses

import numpy as np
import pytest

from certsim.checker import CertificateError, Interval, check
from certsim.solver import default_grid, solve
from certsim.spec import default_spec, lesion_tiles, node_masks

SPEC = default_spec(voxels=(), tiles=lesion_tiles(r_max=4e-3, depth=6e-3, step=1e-3))


@pytest.fixture(scope="module")
def grid():
    return default_grid(SPEC, h_fine=0.5e-3)


@pytest.fixture(scope="module")
def reports(grid):
    r, z = grid
    out = {}
    for p in (1, 2, 3):
        cert = solve(SPEC, r, z, degree=p)
        out[p] = (cert, check(SPEC, cert))
    return out


def overlaps(a, b):
    return a.lo <= b.hi and b.lo <= a.hi


def test_degrees_agree_and_tighten(reports):
    for p in (1, 2, 3):
        for q in (1, 2, 3):
            assert overlaps(reports[p][1].conductance, reports[q][1].conductance)
            for a, b in zip(reports[p][1].tile_rms_field, reports[q][1].tile_rms_field):
                assert overlaps(a, b)
    etas = [reports[p][1].eta.hi for p in (1, 2, 3)]
    assert etas[0] > etas[1] > etas[2]
    assert reports[3][1].conductance.rel_width < 1e-3 * reports[1][1].conductance.rel_width


def test_degree3_claims_are_supported(reports):
    cert, rep = reports[3]
    assert rep.conductance.supports(cert.claims["conductance"], rel_tol=1e-6)


def _edge_interior(mask, p):
    """A node of the mask that is not a grid corner, i.e. inside a cell edge."""
    jj, ii = np.nonzero(mask)
    for j, i in zip(jj, ii):
        if j % p or i % p:
            return j, i
    raise AssertionError("no edge-interior node")


@pytest.mark.parametrize("field, label, message", [
    ("v", "electrode", "electrode voltage"),
    ("v", "ground", "zero on the ground"),
    ("psi", "axis", "axis"),
    ("psi", "shaft", "shaft"),
])
def test_edge_interior_boundary_nodes_are_checked(reports, grid, field, label, message):
    cert, _ = reports[3]
    m = node_masks(SPEC, *grid, 3)[label]
    j, i = _edge_interior(m, 3)
    arr = getattr(cert, field).copy()
    arr[j, i] = arr[j, i] * (1 + 1e-9) + 1e-12
    with pytest.raises(CertificateError, match=message):
        check(SPEC, dataclasses.replace(cert, **{field: arr}))


def test_lesion_extent_grows_with_voltage(reports):
    _, rep = reports[3]
    depths = [rep.lesion_extent(v, 7e4)["depth"][0] for v in (500, 1000, 1500, 2000)]
    assert depths == sorted(depths) and depths[-1] > depths[0]
    for v in (500, 1000, 2000):
        e = rep.lesion_extent(v, 7e4)
        inner, outer = e["depth"]
        assert outer is None or inner <= outer


def test_outer_depth_covers_every_column(reports):
    """No tile that may be lethal lies below the outer depth, on or off the axis."""
    _, rep = reports[3]
    for v in (500, 1000, 2000):
        outer = rep.lesion_extent(v, 7e4)["depth"][1]
        if outer is None:
            continue
        for b, s in zip(rep.tiles, rep.lesion_map(v, 7e4)):
            if s != -1:
                assert -b.z_lo <= outer + 1e-15


def test_disk_on_half_space_degree3():
    """Newman's disk resistance, as in test_checker, now to much higher accuracy."""
    spec = default_spec(sigma_blood=1e-9, voxels=(), regions=())
    r, z = default_grid(spec, h_fine=0.5e-3)
    rep = check(spec, solve(spec, r, z, degree=3))
    newman = 1 / (4 * spec.sigma_tissue * spec.a)
    window = Interval(newman - 12.8, newman - 8.9)
    assert window.lo <= rep.impedance.lo and rep.impedance.hi <= window.hi
    assert rep.impedance.hi - rep.impedance.lo < 0.05  # degree 1 on this grid: ~1 ohm
