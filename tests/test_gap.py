"""Catheter-tissue gap: blood under the tip lowers the impedance (blood conducts
better than myocardium) and the lesion cannot get deeper."""

import dataclasses

import numpy as np

from certsim.checker import check
from certsim.solver import default_grid, solve
from certsim.spec import default_spec, lesion_tiles, node_masks

BASE = default_spec(voxels=(), regions=(), tiles=lesion_tiles(r_max=3e-3, depth=5e-3, step=1e-3))


def run(gap):
    spec = dataclasses.replace(BASE, gap=gap)
    r, z = default_grid(spec, h_fine=0.5e-3)
    cert = solve(spec, r, z, degree=2)
    return spec, r, z, cert, check(spec, cert)


def test_gap_geometry_and_trends():
    _, _, _, _, contact = run(0.0)
    spec, r, z, cert, lifted = run(1e-3)
    m = node_masks(spec, r, z, 2)
    # heights of the degree-2 node rows: cell corners plus one mid-cell row each
    z_nodes = np.append((z[:-1, None] + np.diff(z)[:, None] * np.array([0.0, 0.5])).ravel(), z[-1])
    rows = np.nonzero(m["electrode"].any(axis=1))[0]
    assert z_nodes[rows].min() == spec.tip  # electrode nodes start exactly at the tip
    assert np.all(cert.v[m["electrode"]] == spec.voltage)
    assert lifted.impedance.hi < contact.impedance.lo
    assert lifted.lesion_extent(1000.0, 7e4)["depth"][0] <= contact.lesion_extent(1000.0, 7e4)["depth"][0]
