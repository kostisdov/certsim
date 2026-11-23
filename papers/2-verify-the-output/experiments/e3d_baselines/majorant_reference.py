"""The functional majorant on the solver's own solution (certsim env).

Solves the monopolar lesion-map problem at degree 3 on the certificate grid,
certifies it, and evaluates the majorant with the current recovered from the
potential: the flux term against the certificate's eta, the Friedrichs term
against the field norm, where ||div y||^2 sits, and the Friedrichs term of the
cells more than 1 mm from both electrode corners. Writes majorant_reference.json.

    uv run --project ../../../.. python majorant_reference.py
"""

import json
from pathlib import Path

import numpy as np

import majorant
from certsim.fastcheck import fast_check
from certsim.solver import default_grid, solve
from certsim.spec import Box, default_spec, lesion_tiles

HERE = Path(__file__).resolve().parent


def main():
    spec = default_spec(voxels=(), tiles=lesion_tiles(r_max=6e-3, depth=8e-3, step=0.25e-3),
                        regions=(Box(0.0, 5e-3, -10e-3, -6e-3),))
    r, z = default_grid(spec)
    cert = solve(spec, r, z, degree=3)
    rep = fast_check(spec, cert)
    m = majorant.evaluate(spec, r, z, cert.v, per_cell=True)
    jj, ii = m["cells"]
    rc, zc = 0.5 * (r[ii] + r[ii + 1]), 0.5 * (z[jj] + z[jj + 1])
    corners = [(spec.a, spec.gap), (spec.a, spec.gap + spec.L)]  # tip edge, electrode-shaft junction
    dist = np.min([np.hypot(rc - a, zc - b) for a, b in corners], axis=0)
    share = np.sort(m["div_sq"])[::-1] / m["div_sq"].sum()
    top2 = np.argsort(m["div_sq"])[::-1][:2]
    far = dist > 1e-3
    out = dict(eta=rep.eta.hi, field_norm=float(np.sqrt(m["E_v"])), flux_term=m["flux_term"],
               friedrichs_term=m["friedrichs_term"], share_two_largest_cells=float(share[:2].sum()),
               two_largest_cells=[dict(r_mm=float(rc[k] * 1e3), z_mm=float(zc[k] * 1e3),
                                       hr_nm=float((r[ii[k] + 1] - r[ii[k]]) * 1e9),
                                       hz_nm=float((z[jj[k] + 1] - z[jj[k]]) * 1e9)) for k in top2],
               friedrichs_term_far_cells=float(m["friedrichs_factor"] * np.sqrt(m["div_sq"][far].sum())))
    (HERE / "majorant_reference.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
