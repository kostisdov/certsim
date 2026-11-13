"""The display helpers agree with what the checker certifies."""

import numpy as np

from certsim.checker import check
from certsim.solver import default_grid, solve
from certsim.spec import default_spec, lesion_tiles
from certsim.viz import envelope, field_at, tile_grid


def test_sampled_field_lies_in_certified_tile_intervals():
    spec = default_spec(voxels=(), regions=(), tiles=lesion_tiles(r_max=3e-3, depth=4e-3, step=1e-3))
    r, z = default_grid(spec, h_fine=0.5e-3)
    cert = solve(spec, r, z, degree=2)
    rep = check(spec, cert)
    x, w = np.polynomial.legendre.leggauss(10)
    x, w = (x + 1) / 2, w / 2
    for b, iv in zip(spec.tiles, rep.tile_rms_field):
        R, Z = np.meshgrid(b.r_lo + (b.r_hi - b.r_lo) * x, b.z_lo + (b.z_hi - b.z_lo) * x, indexing="ij")
        Er, Ez = field_at(spec, cert, R, Z)
        W = np.outer(w, w) * R
        rms = np.sqrt(np.sum(W * (Er**2 + Ez**2)) / np.sum(W))
        assert iv.lo <= rms <= iv.hi

    # the envelope agrees with the report's lesion extent
    z_e, inner, outer, _ = envelope(rep, 1000.0, 7e4)
    e = rep.lesion_extent(1000.0, 7e4)
    assert np.isclose(inner.max(), e["radius"][0])
    r_edges, _, _ = tile_grid(rep)
    assert r_edges[-1] == 3e-3
