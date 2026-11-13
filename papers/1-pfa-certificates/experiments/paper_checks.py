"""Reproduce the one-off checks quoted in the paper; writes results/paper_checks.log.

Needs the cached range reports of papers/1-pfa-certificates/experiments/pfa_ranges.py and
papers/1-pfa-certificates/experiments/meckes_comparison.py (run those first).

    uv run python papers/1-pfa-certificates/experiments/paper_checks.py
"""

import dataclasses
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _pfa_cache import RESULTS, map_spec  # noqa: E402
from meckes_comparison import THRESHOLD, min_voltage  # noqa: E402
from meckes_comparison import spec as bipolar_spec  # noqa: E402

from certsim.checker import check  # noqa: E402
from certsim.solver import default_grid, solve  # noqa: E402

LOG = RESULTS / "paper_checks.log"


def say(msg=""):
    print(msg)
    with LOG.open("a") as f:
        f.write(msg + "\n")


def mm(e):
    a, b = e
    return f"{a*1e3:.2f}-{'>edge' if b is None else f'{b*1e3:.2f}'} mm"


def widths(rep):
    w = sorted(iv.rel_width for iv in rep.tile_rms_field)
    return f"median {w[len(w)//2]:.3f}, worst {w[-1]:.3f}"


def main():
    LOG.write_text("")
    t0 = time.perf_counter()

    say("== 1. monopolar: point certificates against the ratio-range certificate ==")
    spec = map_spec()
    rng = pickle.loads((RESULTS / "pfa_range_report.pkl").read_bytes())
    say(f"range rho {rng.rho}: eta_max {rng.eta_max:.4e}; tile relative width {widths(rng)}")
    r, z = default_grid(spec)
    say(f"grid: {len(r)} x {len(z)} nodes, {(len(r)-1)*(len(z)-1)} cells")
    for rho in (1.7, 2.2, 3.3, 3.75):
        s = dataclasses.replace(spec, sigma_blood=rho * spec.sigma_tissue)
        pt = check(s, solve(s, r, z, degree=3))
        inside = sum(R.lo <= Q.lo and Q.hi <= R.hi for R, Q in zip(rng.tile_rms_field, pt.tile_rms_field))
        e = pt.lesion_extent(1000.0, 7e4)
        say(f"rho {rho}: {inside}/{len(spec.tiles)} tiles inside the range bounds; "
            f"1000 V, 700 V/cm: depth {mm(e['depth'])}, radius {mm(e['radius'])}")

    say("\n== 2. degree 1 with weaker grading (pfa_feasibility grid) ==")
    s = dataclasses.replace(map_spec(), tiles=())
    r1, z1 = default_grid(s, h_fine=0.05e-3, h0=5e-7, kappa=0.15)
    rep1 = check(s, solve(s, r1, z1, degree=1))
    say(f"grid {len(r1)} x {len(z1)} nodes, {(len(r1)-1)*(len(z1)-1)} cells: eta {rep1.eta.hi:.4e}")

    say("\n== 3. bipolar: conductivity ratios outside [1.7, 3.75] ==")
    b = bipolar_spec()
    rb, zb = default_grid(b)
    for rho in (0.7, 1.0, 2.8, 7.0, 10.0):
        s = dataclasses.replace(b, sigma_blood=rho * b.sigma_tissue)
        rep = check(s, solve(s, rb, zb, degree=3))
        say(f"rho {rho}: depth@500 V {mm(rep.lesion_extent(500.0, THRESHOLD)['depth'])}; "
            f"V for 1 mm {min_voltage(rep, 1e-3)}, for 3 mm {min_voltage(rep, 3e-3)} "
            f"(proven insufficient, proven sufficient); impedance {rep.impedance.lo:.4f}-{rep.impedance.hi:.4f} ohm")

    say("\n== 4. bipolar ratio-range certificate and gap factors ==")
    crng = pickle.loads((RESULTS / "meckes_contact_range.pkl").read_bytes())
    say(f"contact range rho {crng.rho}: eta_max {crng.eta_max:.4e}; tile relative width {widths(crng)}")
    v0 = min_voltage(pickle.loads((RESULTS / "meckes_gap0.pkl").read_bytes()), 3e-3)
    for g in (1, 2):
        vg = min_voltage(pickle.loads((RESULTS / f"meckes_gap{g}mm.pkl").read_bytes()), 3e-3)
        # voltage needed lies in (insufficient, sufficient]; ratio of the two intervals
        say(f"gap {g} mm: 3 mm needs ({vg[0]:.0f}, {vg[1]:.0f}] V vs ({v0[0]:.0f}, {v0[1]:.0f}] V in contact; "
            f"factor between {vg[0]/v0[1]:.2f} and {vg[1]/v0[0]:.2f}")
    say(f"\ndone in {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    np.set_printoptions(precision=4)
    main()
