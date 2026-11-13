"""Certified lesion extent for every plausible tissue conductivity.

Field quantities depend on sigma_blood / sigma_tissue only. Blood 0.6-0.75 S/m
over myocardium 0.2-0.35 S/m gives ratios 1.7-3.75. Certificates at 17
ratios cover the whole interval (see certsim.ranges). An optional argument sets
the number of certificates; the paper also quotes the result for 5.

    uv run python papers/1-pfa-certificates/experiments/pfa_ranges.py [K]
"""

import dataclasses
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _pfa_cache import RESULTS, load, map_spec  # noqa: E402

from certsim.ranges import check_range  # noqa: E402
from certsim.solver import default_grid, solve  # noqa: E402

RHO = (1.7, 3.75)
K = 17
VOLTAGES = (500.0, 1000.0, 1500.0, 2000.0)


def fmt(extent):
    inner, outer = extent
    return f"{inner*1e3:4.2f}-{'>edge' if outer is None else f'{outer*1e3:.2f}'}"


def main(K=K):
    spec = map_spec()
    path = RESULTS / ("pfa_range_report.pkl" if K == 17 else f"pfa_range_report_K{K}.pkl")
    if path.exists():
        rep = pickle.loads(path.read_bytes())
    else:
        rhos = list(np.linspace(*RHO, K))
        r, z = default_grid(spec)
        t0 = time.perf_counter()
        certs = [solve(dataclasses.replace(spec, sigma_blood=x * spec.sigma_tissue), r, z, degree=3) for x in rhos]
        t1 = time.perf_counter()
        rep = check_range(spec, certs, rhos)
        print(f"{K} solves {t1-t0:.0f} s, range check {time.perf_counter()-t1:.0f} s")
        path.write_bytes(pickle.dumps(rep))
    _, _, nominal = load()
    rho0 = spec.sigma_blood / spec.sigma_tissue
    w = sorted(iv.rel_width for iv in rep.tile_rms_field)
    print(f"ratio range {rep.rho[0]}-{rep.rho[1]} (nominal {rho0:.2f}); eta_max {rep.eta_max:.2e}; "
          f"tile RMS relative width median {w[len(w)//2]:.2f}, worst {w[-1]:.2f}\n")
    print("certified lesion extent [mm]: proven reached - proven not beyond")
    print("  voltage   threshold [V/cm]   depth: nominal   all ratios     radius: nominal   all ratios")
    for V in VOLTAGES:
        for thr in (7e4, (6e4, 8e4)):
            label = f"{thr/100:.0f}" if np.isscalar(thr) else f"{thr[0]/100:.0f}-{thr[1]/100:.0f}"
            n, a = nominal.lesion_extent(V, thr), rep.lesion_extent(V, thr)
            print(f"  {V:6.0f} V   {label:>16}   {fmt(n['depth']):>12} {fmt(a['depth']):>12}"
                  f"     {fmt(n['radius']):>12} {fmt(a['radius']):>12}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else K)
