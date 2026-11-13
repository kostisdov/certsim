"""Certified bounds against a standard mesh-convergence study (GCI).

The monopolar electrode at contact is solved on two families of systematically
refined grids (refinement ratio 2): uniform near the electrode, as a typical
user would mesh it, and graded towards the electrode corners, as in the paper. On every grid we record what a practitioner reads from the finite element
solution alone (impedance from the field energy, tile RMS field from the primal
gradient) and the certified interval for the same quantity. Richardson
extrapolation and the grid convergence index (Roache; procedure of Celik et
al. 2008, safety factor 1.25) are applied to the three finest grids of each
family and compared with a certified reference.

    uv run python papers/1-pfa-certificates/experiments/gci_comparison.py
"""

import pickle
import time
from pathlib import Path

import numpy as np
from flint import arb

from certsim.gci import gci
from certsim.checker import _Basis, _Cell, _cells_in_box, _grad, _nodal, check
from certsim.solver import default_grid, solve
from certsim.spec import Box, default_spec, element_masks

RESULTS = Path(__file__).resolve().parents[3] / "results"  # shared, at the repository root
DEGREE = 2
H_FINE = (1e-3, 0.5e-3, 0.25e-3, 0.125e-3)
TILES = {
    "corner": Box(1.25e-3, 1.5e-3, -0.25e-3, 0.0),  # touches the electrode edge
    "axis": Box(0.0, 0.25e-3, -3.75e-3, -3.5e-3),  # lesion boundary at 1000 V, 700 V/cm
}
SPEC = default_spec(voxels=(), regions=(), tiles=tuple(TILES.values()))
VOLT, THRESH = 1000.0, 7e4  # field values are reported at this voltage [V], threshold [V/m]


def grid(family, h):
    """Refinement by a common factor: spacing, far-field growth and grading slope."""
    s = h / 0.25e-3
    if family == "uniform":
        return default_grid(SPEC, h_fine=h, h0=h, kappa=0.25, growth=0.08 * s)
    return default_grid(SPEC, h_fine=h, h0=1e-9, kappa=0.25 * s, growth=0.08 * s)


def primal_tile_rms(cert, box):
    """RMS |grad v| over a tile from the primal field alone, as post-processing would give."""
    r, z = cert.r, cert.z
    active, sigma_el = element_masks(SPEC, r, z)
    basis, p = _Basis(cert.degree), cert.degree
    energy = arb(0)
    cells = _cells_in_box(box, r, z) & active
    for j, i in zip(*np.nonzero(cells)):
        cell = _Cell(arb(r[i]), arb(r[i + 1]), arb(z[j]), arb(z[j + 1]), arb(sigma_el[j, i]), p)
        sv = _grad(basis.monomials(_nodal(cert.v, j, i, p)), p, cell.hr, cell.hz, arb(sigma_el[j, i]))
        energy += cell.inner(sv, sv)
    sigma = float(sigma_el[cells][0])
    vol = np.pi * (box.r_hi**2 - box.r_lo**2) * (box.z_hi - box.z_lo)
    return float((energy / (sigma * vol)).sqrt().mid()) * VOLT


def run_grid(family, h, degree=DEGREE, bug=None):
    r, z = grid(family, h)
    t0 = time.perf_counter()
    cert = solve(SPEC, r, z, degree=degree, bug=bug)
    rep = check(SPEC, cert)
    out = {
        "family": family, "h": h, "cells": (len(r) - 1) * (len(z) - 1), "seconds": time.perf_counter() - t0,
        "Z_fe": 1.0 / cert.claims["conductance"],
        "Z_cert": (rep.impedance.lo, rep.impedance.hi),
    }
    for (name, box), iv in zip(TILES.items(), rep.tile_rms_field):
        out[f"{name}_fe"] = primal_tile_rms(cert, box)
        out[f"{name}_cert"] = (iv.lo * VOLT, iv.hi * VOLT)
    return out


def main():
    RESULTS.mkdir(exist_ok=True)
    cache = RESULTS / "gci_runs.pkl"
    runs = pickle.loads(cache.read_bytes()) if cache.exists() else {}
    for family in ("uniform", "graded"):
        for h in H_FINE:
            if (family, h) not in runs:
                runs[(family, h)] = run_grid(family, h)
                cache.write_bytes(pickle.dumps(runs))
            o = runs[(family, h)]
            print(f"{family:8s} h={h * 1e3:5.3f} mm  {o['cells']:7d} cells  {o['seconds']:6.1f} s")
    if ("reference", 0.125e-3) not in runs:
        runs[("reference", 0.125e-3)] = run_grid("graded", 0.125e-3, degree=3)
        cache.write_bytes(pickle.dumps(runs))
    ref = runs[("reference", 0.125e-3)]

    for q, unit in (("Z", "ohm"), ("corner", "V/m"), ("axis", "V/m")):
        rlo, rhi = ref[q + "_cert"]
        print(f"\n== {q} [{unit}]  certified reference {rlo:.9g} - {rhi:.9g}")
        for family in ("uniform", "graded"):
            fe = [runs[(family, h)][q + "_fe"] for h in H_FINE]
            for h, o in zip(H_FINE, (runs[(family, h)] for h in H_FINE)):
                lo, hi = o[q + "_cert"]
                print(f"  {family:8s} h={h * 1e3:5.3f}  FE {o[q + '_fe']:.9g}   "
                      f"certified {max(lo, 0.0):.9g} - {hi:.9g}")
            g = gci(fe[::-1][:3])  # three finest, fine -> coarse
            b0, b1 = g["band"]
            verdict = "misses the reference" if (b1 < rlo or b0 > rhi) else "consistent with the reference"
            print(f"  {family:8s} GCI: R {g['R']:.3f} ({g['kind']}), order {g['p']:.2f}, "
                  f"extrapolated {g['ext']:.9g}, band {b0:.9g} - {b1:.9g} (+-{(b1 - b0) / (b1 + b0) * 100:.3g}%), "
                  f"{verdict}")

    # Convergence is not correctness: solvers with deliberately wrong physics,
    # on the graded family, judged by a GCI on their own impedance and by the
    # certificate checked against the true specification.
    rlo, rhi = ref["Z_cert"]
    print(f"\n== wrong physics, graded grids [ohm]  true impedance {rlo:.9g} - {rhi:.9g}")
    for bug in ("swap_sigma", "planar"):
        for h in H_FINE[1:]:
            if (bug, h) not in runs:
                runs[(bug, h)] = run_grid("graded", h, bug=bug)
                cache.write_bytes(pickle.dumps(runs))
        fe = [runs[(bug, h)]["Z_fe"] for h in H_FINE[1:]]
        g = gci(fe[::-1])
        b0, b1 = g["band"]
        clo, chi = runs[(bug, H_FINE[-1])]["Z_cert"]
        claim = fe[-1]
        proven = claim * 0.99 <= clo and chi <= claim * 1.01  # as Interval.supports(claim, 0.01)
        print(f"  {bug:10s} solver's own Z {', '.join(f'{x:.6g}' for x in fe)}; GCI: R {g['R']:.3f} "
              f"({g['kind']}), order {g['p']:.2f}, band {b0:.6g} - {b1:.6g}")
        print(f"  {bug:10s} certified against the true spec (finest grid): {clo:.6g} - {chi:.6g}; "
              f"claim inside: {clo <= claim <= chi}; claim proven to 1%: {proven}")


if __name__ == "__main__":
    main()
