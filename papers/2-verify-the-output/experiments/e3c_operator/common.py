"""Shared definitions for E3c: the parameter family, its grids, and the sample sets.

The family is the monopolar lesion-map problem of paper 1 at contact, with two
parameters: the conductivity ratio rho = sigma_blood / sigma_tissue and the
electrode length L. Changing L moves only the grid line at the top of the
electrode, so every member's certificate grid is the reference grid (L = 3 mm)
with its z-lines above the tissue mapped piecewise linearly. All grids then have
the same number of lines, and a network can work on one fixed index grid.
"""

import numpy as np

from certsim.solver import default_grid
from certsim.spec import Box, default_spec, lesion_tiles, node_masks

SIGMA_T = 0.25
DEGREE = 3
L0 = 3.0e-3
RHO_RANGE = (1.7, 3.75)
L_RANGE = (2.5e-3, 3.5e-3)


def spec_for(rho, L):
    return default_spec(voxels=(), sigma_tissue=SIGMA_T, sigma_blood=rho * SIGMA_T, L=L,
                        tiles=lesion_tiles(r_max=6e-3, depth=8e-3, step=0.25e-3),
                        regions=(Box(0.0, 5e-3, -10e-3, -6e-3),))


_REF = None


def reference_grid():
    global _REF
    if _REF is None:
        _REF = default_grid(spec_for(2.8, L0))
    return _REF


def grid_for(L):
    """Reference grid with the z-lines above the tissue mapped so that the top of
    the electrode lies at L; lines at and below z = 0 do not move."""
    r, z0 = reference_grid()
    ztop = z0[-1]
    z = z0.copy()
    lower = (z0 > 0) & (z0 < L0)
    upper = (z0 > L0) & (z0 < ztop)
    z[lower] = z0[lower] * (L / L0)
    z[upper] = L + (z0[upper] - L0) * (ztop - L) / (ztop - L0)
    z[z0 == L0] = L
    assert np.all(np.diff(z) > 0)
    return r.copy(), z


def masks_for(L):
    r, z = grid_for(L)
    return node_masks(spec_for(2.8, L), r, z, DEGREE)


def sample_sets():
    """Training (Latin hypercube), calibration and in-range test (uniform random),
    and out-of-range test points, as lists of (rho, L)."""
    from scipy.stats import qmc

    lo = np.array([np.log(RHO_RANGE[0]), L_RANGE[0]])
    hi = np.array([np.log(RHO_RANGE[1]), L_RANGE[1]])

    def to_params(u):
        x = lo + u * (hi - lo)
        return [(float(np.exp(a)), float(b)) for a, b in x]

    train = to_params(qmc.LatinHypercube(d=2, seed=0).random(64))
    rng = np.random.default_rng(1)
    calib = to_params(rng.random((24, 2)))
    test_in = to_params(rng.random((12, 2)))
    test_out = [(0.7, 3.0e-3), (1.0, 3.0e-3), (5.0, 3.0e-3), (10.0, 3.0e-3),
                (2.8, 2.0e-3), (2.8, 4.0e-3), (0.7, 2.0e-3), (10.0, 4.0e-3)]
    return {"train": train, "calib": calib, "test_in": test_in, "test_out": test_out}
