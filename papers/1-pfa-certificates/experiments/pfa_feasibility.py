"""Certified feasibility of a focal PFA electrode.

Solves the forward and adjoint problems with the (untrusted) solver, checks the
certificate, and prints what is proven: the electrode impedance, the voxel-mean
field along the axis, and the lesion depth for several voltages and lethal
thresholds. This is the adjoint (voxel-mean) method at degree 1; the lesion map
example reaches tighter bounds with degree 3 and RMS tiles.

    uv run python papers/1-pfa-certificates/experiments/pfa_feasibility.py [--fast]
"""

import argparse
import time

from certsim.checker import check
from certsim.solver import default_grid, solve
from certsim.spec import default_spec

THRESHOLDS = (4e4, 7e4, 1e5)  # V/m (400, 700, 1000 V/cm)
VOLTAGES = (500.0, 1000.0, 1500.0, 2000.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="coarser grid, wider bounds")
    args = ap.parse_args()
    spec = default_spec()
    # degree 1 with adjoint voxels; papers/1-pfa-certificates/experiments/pfa_lesion_map.py uses degree 3
    grid = (dict(h_fine=0.1e-3, h0=1e-6, kappa=0.2, h0_box=1e-5) if args.fast
            else dict(h_fine=0.05e-3, h0=5e-7, kappa=0.15, h0_box=4e-6))
    r, z = default_grid(spec, **grid)

    t0 = time.perf_counter()
    cert = solve(spec, r, z, degree=1)
    t1 = time.perf_counter()
    rep = check(spec, cert)
    t2 = time.perf_counter()

    print(f"electrode a = {spec.a*1e3:g} mm, L = {spec.L*1e3:g} mm; "
          f"sigma tissue {spec.sigma_tissue} S/m, blood {spec.sigma_blood} S/m")
    print(f"grid {len(r)} x {len(z)}; solve {t1-t0:.1f} s, check {t2-t1:.1f} s\n")
    Z = rep.impedance
    print(f"impedance: {Z.lo:.3f} .. {Z.hi:.3f} ohm (certified)\n")

    print("voxel-mean field on the axis at 1000 V [V/cm], certified interval")
    for b, f in zip(rep.voxels, rep.voxel_field(1000.0)):
        print(f"  {abs(b.z_hi)*1e3:4.1f}-{abs(b.z_lo)*1e3:4.1f} mm   {f.lo/100:7.1f} .. {f.hi/100:7.1f}")

    print("\ncertified lesion depth on the axis [mm]: reached / not beyond")
    print("  voltage   " + "   ".join(f"{t/100:>6.0f} V/cm" for t in THRESHOLDS))
    for V in VOLTAGES:
        cells = []
        for t in THRESHOLDS:
            reached, beyond = rep.lesion_depth(V, t)
            cells.append(f"{reached*1e3:4.1f} / {'>8' if beyond is None else f'{beyond*1e3:3.1f}':>4}")
        print(f"  {V:6.0f} V   " + "   ".join(f"{c:>11}" for c in cells))

    reg = rep.region_rms_field[0]
    b = spec.regions[0]
    print(f"\nRMS field in r < {b.r_hi*1e3:g} mm, {-b.z_hi*1e3:g}-{-b.z_lo*1e3:g} mm deep, at 1000 V: "
          f"{reg.scale(1000).lo/100:.1f} .. {reg.scale(1000).hi/100:.1f} V/cm")


if __name__ == "__main__":
    main()
