"""Certified lesion map of a focal PFA electrode.

Every 0.25 mm tile under the electrode gets a guaranteed interval for its RMS
field. At a given voltage and lethal threshold a tile is then certainly lethal,
certainly spared, or undecided.

    uv run python papers/1-pfa-certificates/experiments/pfa_lesion_map.py [--voltage 1000] [--threshold 700]
"""

import argparse
import time

from certsim.checker import check
from certsim.solver import default_grid, solve
from certsim.spec import default_spec, lesion_tiles

VOLTAGES = (500.0, 1000.0, 1500.0, 2000.0)
THRESHOLDS = (4e4, 7e4, 1e5)  # V/m


def fmt(extent):
    inner, outer = extent
    return f"{inner*1e3:4.2f}-{'>edge' if outer is None else f'{outer*1e3:.2f}'}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voltage", type=float, default=1000.0)
    ap.add_argument("--threshold", type=float, default=700.0, help="lethal field [V/cm]")
    args = ap.parse_args()

    step = 0.25e-3
    spec = default_spec(voxels=(), tiles=lesion_tiles(r_max=6e-3, depth=8e-3, step=step))
    r, z = default_grid(spec)
    t0 = time.perf_counter()
    cert = solve(spec, r, z, degree=3)
    t1 = time.perf_counter()
    rep = check(spec, cert)
    t2 = time.perf_counter()

    print(f"grid {len(r)} x {len(z)}, degree 3; solve {t1-t0:.1f} s, check {t2-t1:.1f} s")
    print(f"impedance {rep.impedance.lo:.6f} .. {rep.impedance.hi:.6f} ohm (certified)")
    widths = sorted(rep.tile_rms_field, key=lambda iv: iv.rel_width)
    print(f"tile RMS field: median relative width {widths[len(widths)//2].rel_width:.1e}, "
          f"worst {widths[-1].rel_width:.1e}\n")

    V, thr = args.voltage, args.threshold * 100
    state = rep.lesion_map(V, thr)
    nr = int(round(6e-3 / step))
    print(f"lesion map at {V:.0f} V, lethal if RMS field >= {args.threshold:.0f} V/cm")
    print("  '#' certainly lethal   '.' certainly spared   '?' undecided")
    print("  depth  r = 0 " + "-" * (nr - 10) + " 6 mm")
    for k in range(len(state) // nr):
        row = state[k * nr:(k + 1) * nr]
        line = "".join("#" if s == 1 else "." if s == -1 else "?" for s in row)
        print(f"  {k*step*1e3:4.2f}  {line}")

    print("\ncertified lesion extent [mm]: proven reached - proven not beyond")
    print("  voltage   threshold   depth on axis  largest radius")
    for V in VOLTAGES:
        for thr in THRESHOLDS:
            e = rep.lesion_extent(V, thr)
            print(f"  {V:6.0f} V   {thr/100:5.0f} V/cm   {fmt(e['depth']):>13}   {fmt(e['radius']):>13}")

    reg, b = rep.region_rms_field[0], spec.regions[0]
    print(f"\nRMS field in r < {b.r_hi*1e3:g} mm, {-b.z_hi*1e3:g}-{-b.z_lo*1e3:g} mm deep, per 1000 V: "
          f"{reg.scale(1000).lo/100:.2f} .. {reg.scale(1000).hi/100:.2f} V/cm")


if __name__ == "__main__":
    main()
