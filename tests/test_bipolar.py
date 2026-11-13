"""Bipolar catheter: tip at +V/2, ring at -V/2, floating (insulated far field)."""

import dataclasses

import numpy as np
import pytest

from certsim.checker import CertificateError, check
from certsim.solver import default_grid, solve
from certsim.spec import default_spec, lesion_tiles, node_masks

SPEC = default_spec(voxels=(), regions=(), tiles=lesion_tiles(r_max=3e-3, depth=4e-3, step=1e-3),
                    L=2e-3, ring_spacing=4e-3, ring_length=2e-3)


@pytest.fixture(scope="module")
def certs():
    r, z = default_grid(SPEC, h_fine=0.5e-3)
    out = {p: solve(SPEC, r, z, degree=p) for p in (1, 2)}
    return r, z, out, {p: check(SPEC, c) for p, c in out.items()}


def test_bipolar_consistency(certs):
    r, z, out, reps = certs
    assert reps[1].conductance.lo <= reps[2].conductance.hi and reps[2].conductance.lo <= reps[1].conductance.hi
    for a, b in zip(reps[1].tile_rms_field, reps[2].tile_rms_field):
        assert a.lo <= b.hi and b.lo <= a.hi
    # floating: all tip current returns through the ring, and power = current * voltage
    m = node_masks(SPEC, r, z, 2)
    tip_current = 2 * np.pi * abs(out[2].psi[m["spacer"]][0])
    assert np.all(out[2].psi[m["shaft"] | m["outer"]] == 0.0)
    G = reps[2].conductance
    assert G.lo * 0.999 <= tip_current <= G.hi * 1.001


@pytest.mark.parametrize("field, label, value, message", [
    ("v", "ring", 0.0, "ring electrode voltage"),
    ("psi", "spacer", None, "spacer"),
    ("psi", "outer", 1e-9, "axis"),
])
def test_bipolar_mutations_rejected(certs, field, label, value, message):
    r, z, out, _ = certs
    cert = out[2]
    m = node_masks(SPEC, r, z, 2)[label]
    arr = getattr(cert, field).copy()
    j, i = np.argwhere(m)[len(np.argwhere(m)) // 2]
    arr[j, i] = arr[j, i] + 1e-9 if value is None else value
    with pytest.raises(CertificateError, match=message):
        check(SPEC, dataclasses.replace(cert, **{field: arr}))
