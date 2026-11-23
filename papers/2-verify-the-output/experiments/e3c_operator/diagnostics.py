"""The certificate grid's thin cells and the operator's nodal errors (certsim env).

For the reference grid (L = 3 mm): the number of active cells thinner than 10 nm,
their largest aspect ratio, and the change of the reference potential across
them. For the operator trained with the nodal loss, the nodal errors of its
potential on the first in-range test. And the eta that a stored (single-precision)
reference field itself receives. Writes diagnostics.json.

    uv run --project ../../../.. python diagnostics.py
"""

import json
from pathlib import Path

import numpy as np

from certify_operator import admissible
from certsim.fastcheck import fast_check
from certsim.solver import Certificate
from certsim.spec import element_masks
from common import DEGREE, L0, grid_for, masks_for, spec_for

HERE = Path(__file__).resolve().parent


def main():
    r, z = grid_for(L0)
    hr, hz = np.diff(r), np.diff(z)
    HZ, HR = np.meshgrid(hz, hr, indexing="ij")
    m = masks_for(L0)
    act, _ = element_masks(spec_for(2.8, L0), r, z)
    thin = (np.minimum(HZ, HR) < 10e-9) & act
    aspect = np.maximum(HZ, HR) / np.minimum(HZ, HR)
    out = dict(active_cells=int(act.sum()), active_cells_thinner_than_10nm=int(thin.sum()),
               max_aspect_active=float(aspect[act].max()), max_aspect_thin=float(aspect[thin].max()))

    pred = np.load(HERE / "predictions.npz")
    f = str(pred["test_in_files"][0])
    d = np.load(HERE / "data" / f)
    rho, L = float(d["rho"]), float(d["L"])
    spec = spec_for(rho, L)
    v, _ = admissible(spec, pred["test_in"][0, 0], pred["test_in"][0, 1], masks_for(L))
    free = m["active"] & ~(m["electrode"] | m["ground"])
    e = np.abs(v - d["v"].astype(float))[free]
    out.update(test=f, rho=rho, L=L, nodal_error_median_V=float(np.median(e)), nodal_error_max_V=float(e.max()))

    # change of the reference potential across a thin cell, in its thin direction (cell corner nodes)
    vc = d["v"].astype(float)[::DEGREE, ::DEGREE]
    dvz = np.abs(np.diff(vc, axis=0))[:, :-1]
    dvr = np.abs(np.diff(vc, axis=1))[:-1, :]
    dv = np.where(HZ < HR, dvz, dvr)
    out["reference_change_across_thin_cells_V"] = dict(
        median=float(np.median(dv[thin])), p99=float(np.percentile(dv[thin], 99)), max=float(dv[thin].max()))
    # the stored references are single precision: what does a stored field itself certify to?
    etas = []
    for g in sorted((HERE / "data").glob("test_in_*.npz"))[:3]:
        dd = np.load(g)
        rr, zz = grid_for(float(dd["L"]))
        vv, pp = dd["v"].astype(float), dd["psi"].astype(float)
        empty = np.zeros((0, *vv.shape))
        etas.append(fast_check(spec_for(float(dd["rho"]), float(dd["L"])),
                               Certificate(rr, zz, vv, pp, empty, empty, DEGREE, {})).eta.hi)
    out["stored_reference_eta"] = [float(min(etas)), float(max(etas))]
    (HERE / "diagnostics.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
