"""The potential-to-certificate adapter lets any solver's potential be checked."""

import numpy as np

from certsim.checker import check
from certsim.solver import certify_potential, default_grid, solve
from certsim.spec import default_spec

SPEC = default_spec(voxels=(), regions=())


def test_adapter_reproduces_own_solver():
    r, z = default_grid(SPEC, h_fine=1e-3)
    own = solve(SPEC, r, z, degree=2)
    cert = certify_potential(SPEC, r, z, own.v, 2, claims=own.claims)
    assert np.array_equal(cert.psi, own.psi)
    a, b = check(SPEC, own).conductance, check(SPEC, cert).conductance
    assert (a.lo, a.hi) == (b.lo, b.hi)


def test_adapter_bounds_contain_truth_for_a_perturbed_potential():
    """A worse potential from some other solver gets a wider, still valid bound."""
    r, z = default_grid(SPEC, h_fine=1e-3)
    own = solve(SPEC, r, z, degree=2)
    exact = check(SPEC, own).conductance
    rng = np.random.default_rng(0)
    free = np.abs(own.v - SPEC.voltage) > 1e-12  # leave electrode values exact
    free &= own.v > 1e-12  # and ground values
    noisy = own.v + free * 1e-3 * rng.standard_normal(own.v.shape)
    rep = check(SPEC, certify_potential(SPEC, r, z, noisy, 2))
    assert rep.conductance.lo <= exact.hi and exact.lo <= rep.conductance.hi
    assert rep.conductance.hi - rep.conductance.lo > exact.hi - exact.lo
