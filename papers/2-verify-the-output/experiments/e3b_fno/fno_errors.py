"""Where the uniform-grid FNO errs (certsim env).

Compares the FNO's prediction at rho = 2.8 with the reference field sampled on the
same uniform grid, in the domain far from the electrode (more than 0.5 mm away)
and near it (within 0.5 mm). Writes fno_errors.json.

    uv run --project ../../../.. python fno_errors.py
"""

import json
from pathlib import Path

import numpy as np

from make_data import spec_for

HERE = Path(__file__).resolve().parent
RHO, NEAR = 2.8, 0.5e-3


def main():
    d, p = np.load(HERE / "data.npz"), np.load(HERE / "predictions.npz")
    k = int(np.argmin(abs(p["rho"] - RHO)))
    assert abs(d["test_rho"][k] - RHO) < 1e-9
    R, Z = np.meshgrid(d["r"], d["z"], indexing="ij")
    spec = spec_for(RHO)
    a, z0, z1 = spec.a, spec.gap, spec.gap + spec.L
    dist = np.hypot(np.maximum(R - a, 0.0), np.maximum(np.maximum(z0 - Z, Z - z1), 0.0))
    err = np.abs(p["v"][k] - d["test_v"][k])
    dom = d["mask"] > 0
    far, near = dom & (dist > NEAR), dom & (dist <= NEAR)
    out = dict(rho=RHO, far_mean_V=float(err[far].mean()), far_max_V=float(err[far].max()),
               near_mean_V=float(err[near].mean()), near_max_V=float(err[near].max()),
               rms_V=float(np.sqrt(np.mean(err[dom] ** 2))))
    (HERE / "fno_errors.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
