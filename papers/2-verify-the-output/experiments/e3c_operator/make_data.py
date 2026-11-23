"""Reference solutions for E3c (run in the certsim environment).

For every (rho, L) of the training, calibration and test sets, solves the
problem with our solver (degree 3) on that member's grid, checks the solution's
own certificate with the vectorized checker to get a rigorous reference
impedance, and stores the nodal potential and stream function. Each sample is
written to data/ as it finishes, so the script can be stopped and resumed.

    uv run --project ../../../.. python make_data.py [workers]
"""

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from certsim.fastcheck import fast_check
from certsim.solver import solve
from common import DEGREE, grid_for, sample_sets, spec_for

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


def one(name, k, rho, L):
    out = DATA / f"{name}_{k:03d}.npz"
    if out.exists():
        return name, k, None
    t0 = time.perf_counter()
    spec = spec_for(rho, L)
    r, z = grid_for(L)
    cert = solve(spec, r, z, degree=DEGREE)
    t_solve = time.perf_counter() - t0
    rep = fast_check(spec, cert)
    Z = rep.impedance
    np.savez(out, v=cert.v.astype(np.float32), psi=cert.psi.astype(np.float32), rho=rho, L=L,
             Z_lo=Z.lo, Z_hi=Z.hi, eta=rep.eta.hi, Z_claim=1 / cert.claims["conductance"],
             solve_s=t_solve)
    return name, k, (t_solve, Z.lo, Z.hi)


def main(workers=4):
    DATA.mkdir(exist_ok=True)
    sets = sample_sets()
    (HERE / "samples.json").write_text(json.dumps(sets, indent=1))
    jobs = [(name, k, rho, L) for name, pts in sets.items() for k, (rho, L) in enumerate(pts)]
    t0 = time.perf_counter()
    done = 0
    with ProcessPoolExecutor(workers) as pool:
        futs = [pool.submit(one, *j) for j in jobs]
        for f in as_completed(futs):
            name, k, res = f.result()
            done += 1
            msg = "cached" if res is None else f"solve {res[0]:.1f} s, Z {res[1]:.6f}-{res[2]:.6f}"
            print(f"[{done}/{len(jobs)}, {time.perf_counter() - t0:.0f} s] {name} {k}: {msg}", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
