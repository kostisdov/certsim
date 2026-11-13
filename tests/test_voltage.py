"""Bounds scale correctly with the specified voltage, and Thomson's bound holds
for the bipolar (two-terminal) catheter as well."""

import dataclasses

import pytest

from certsim.checker import check
from certsim.solver import default_grid, solve
from certsim.spec import default_spec

MONO = default_spec(voxels=(), regions=())
BI = dataclasses.replace(MONO, L=2e-3, ring_spacing=4e-3, ring_length=2e-3)


@pytest.mark.parametrize("base", [MONO, BI], ids=["monopolar", "bipolar"])
def test_conductance_independent_of_spec_voltage(base):
    r, z = default_grid(base, h_fine=0.5e-3)
    reps = {}
    for V in (1.0, 0.25, 40.0):
        spec = dataclasses.replace(base, voltage=V)
        reps[V] = check(spec, solve(spec, r, z, degree=2))
    g1 = reps[1.0].conductance
    for V, rep in reps.items():
        g = rep.conductance
        # same physics, so the enclosures must overlap and be equally tight
        assert g.lo <= g1.hi and g1.lo <= g.hi
        assert g.rel_width < 2 * g1.rel_width + 1e-12
    # with Thomson in use, the bipolar enclosure is as tight as the monopolar one
    assert g1.rel_width < 1e-5
