"""Cost of the vectorized range check on the 17-certificate range of Section 6.2.

Solves the 17 certificates over rho in [1.7, 3.75], runs the vectorized range checker
(certsim.fastcheck.fast_check_range), and compares its bounds with the 256-bit range report
stored by pfa_ranges.py: eta_max, every tile interval, and the lesion extents of Table 2.

    uv run python papers/1-pfa-certificates/experiments/range_check_cost.py
"""

import dataclasses
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _pfa_cache import RESULTS, map_spec  # noqa: E402

from certsim.fastcheck import fast_check_range  # noqa: E402
from certsim.solver import default_grid, solve  # noqa: E402

RHO, K = (1.7, 3.75), 17
VOLTAGES = (500.0, 1000.0, 1500.0, 2000.0)


def main():
    ref_path = RESULTS / "pfa_range_report.pkl"
    if not ref_path.exists():
        raise SystemExit("run pfa_ranges.py first: it stores the 256-bit range report")
    ref = pickle.loads(ref_path.read_bytes())
    spec = map_spec()
    rhos = list(np.linspace(*RHO, K))
    r, z = default_grid(spec)
    t0 = time.perf_counter()
    certs = [solve(dataclasses.replace(spec, sigma_blood=x * spec.sigma_tissue), r, z, degree=3) for x in rhos]
    t1 = time.perf_counter()
    fast = fast_check_range(spec, certs, rhos)
    t2 = time.perf_counter()
    print(f"{K} solves {t1 - t0:.0f} s, vectorized range check {t2 - t1:.1f} s")
    print(f"eta_max: 256-bit {ref.eta_max:.6e}, vectorized {fast.eta_max:.6e}, "
          f"relative difference {abs(ref.eta_max - fast.eta_max) / ref.eta_max:.1e}")
    overlap = all(a.lo <= b.hi and b.lo <= a.hi for a, b in zip(ref.tile_rms_field, fast.tile_rms_field))
    worst = max(max(abs(a.lo - b.lo), abs(a.hi - b.hi)) / a.hi for a, b in zip(ref.tile_rms_field, fast.tile_rms_field))
    print(f"tiles {len(ref.tile_rms_field)}: every interval overlaps the 256-bit one: {overlap}; "
          f"largest relative endpoint difference {worst:.1e}")
    same = all(ref.lesion_extent(V, t) == fast.lesion_extent(V, t) for V in VOLTAGES for t in (7e4, (6e4, 8e4)))
    print(f"lesion extents over all ratios (Table 2) identical: {same}")


if __name__ == "__main__":
    main()
