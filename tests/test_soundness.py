"""Regression tests for the trusted base: materials are decided by geometry, the
impedance is rounded outwards, and the vectorized checker refuses flush-to-zero."""

import math
from fractions import Fraction

import pytest

from certsim import fastcheck
from certsim.checker import CertificateError, Interval, Report, check
from certsim.solver import default_grid, solve
from certsim.spec import Box, default_spec


def test_box_across_the_interface_is_rejected_with_equal_conductivities():
    tile = Box(2e-3, 3e-3, -0.5e-3, 0.5e-3)  # straddles z = 0, away from the catheter
    spec = default_spec(voxels=(), sigma_blood=0.25, tiles=(tile,))
    r, z = default_grid(spec, h_fine=1e-3)
    cert = solve(default_spec(voxels=(), sigma_blood=0.25), r, z, degree=1)
    with pytest.raises(CertificateError, match="single material"):
        check(spec, cert)


def report(g_lo, g_hi):
    return Report(conductance=Interval(g_lo, g_hi), eta=Interval(0.0, 1.0), voxel_mean_dudz=[],
                  voxel_mean_field=[], voxel_rms_field=[], region_rms_field=[])


@pytest.mark.parametrize("g", [(0.0106, 0.0107), (1 / 3, 1 / 3 + 1e-12), (0.1, 0.1)])
def test_impedance_encloses_the_exact_reciprocal(g):
    z = report(*g).impedance
    assert Fraction(z.lo) <= 1 / Fraction(g[1]) and 1 / Fraction(g[0]) <= Fraction(z.hi)


def test_impedance_is_unbounded_when_the_conductance_may_vanish():
    assert report(0.0, 0.01).impedance.hi == math.inf


def test_subnormals_are_preserved_here():
    assert fastcheck._subnormals_preserved()


def test_fast_check_refuses_flush_to_zero(monkeypatch):
    monkeypatch.setattr(fastcheck, "_subnormals_preserved", lambda: False)
    with pytest.raises(RuntimeError, match="flushed to zero"):
        fastcheck.fast_check(default_spec(voxels=()), None)
