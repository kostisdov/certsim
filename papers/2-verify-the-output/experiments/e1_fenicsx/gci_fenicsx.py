"""Grid convergence study on FEniCSx's own meshes against the certificate (certsim environment).

Two families of FEniCSx solutions on gmsh meshes refined by a factor of 2
(spacing 1 to 0.125 mm in the lesion zone): quadratic triangles without corner
grading, as a typical user would mesh, and cubic triangles graded to 1 nm at
both electrode corners. For every mesh, FEniCSx's own impedance feeds the
three-grid GCI of paper 1, and the unchanged checker certifies the solution
evaluated on the certificate grid. Results are cached in gci_fenicsx.json.

    uv run python gci_fenicsx.py
"""

import json
import subprocess
from pathlib import Path

import numpy as np

from certsim.checker import check
from certsim.solver import certify_potential
from export_problem import problem_spec

HERE = Path(__file__).resolve().parent
from certsim.gci import gci  # the procedure of paper 1

H_FINE = (1.0, 0.5, 0.25, 0.125)  # mm
FAMILIES = {"uniform P2": (2, None), "graded P3": (3, 0.001)}  # degree, h0 in um (None: no grading)
REFERENCE = (93.9502742, 93.9502758)  # paper 1 certified impedance, ohm (rounded outwards)


def run(k, h, h0_um):
    h0 = h0_um if h0_um is not None else h * 1e3  # no grading: h0 equal to the spacing
    subprocess.run(["pixi", "run", "python", "fenics_unstructured.py", str(k), str(h), str(h0), "scale"],
                   cwd=HERE, check=True, capture_output=True)
    prob, sol = np.load(HERE / "problem.npz"), np.load(HERE / "fenics_solution.npz")
    spec = problem_spec()
    cert = certify_potential(spec, prob["r"], prob["z"], sol["v"], int(prob["degree"]),
                             claims={"conductance": float(sol["conductance"])})
    rep = check(spec, cert)
    widths = np.array([iv.rel_width for iv in rep.tile_rms_field])
    return {"dofs": int(sol["dofs"]), "Z_fe": 1 / float(sol["conductance"]),
            "Z_cert": [rep.impedance.lo, rep.impedance.hi], "eta": rep.eta.hi,
            "tile_median": float(np.median(widths)), "tile_worst": float(widths.max())}


def main():
    cache = HERE / "gci_fenicsx.json"
    runs = json.loads(cache.read_text()) if cache.exists() else {}
    for name, (k, h0) in FAMILIES.items():
        for h in H_FINE:
            key = f"{name} {h}"
            if key not in runs:
                runs[key] = run(k, h, h0)
                cache.write_text(json.dumps(runs, indent=1))
    rlo, rhi = REFERENCE
    print(f"reference impedance {rlo} - {rhi} ohm")
    for name in FAMILIES:
        print(f"\n== {name}")
        for h in H_FINE:
            o = runs[f"{name} {h}"]
            print(f"  h {h:5.3f} mm  {o['dofs']:7d} dofs  FE {o['Z_fe']:.7f}  certified "
                  f"{o['Z_cert'][0]:.7f} - {o['Z_cert'][1]:.7f}  eta {o['eta']:.2e}  "
                  f"tiles median {o['tile_median'] * 100:.1f}%")
        fe = [runs[f"{name} {h}"]["Z_fe"] for h in H_FINE]
        g = gci(fe[::-1][:3])
        b0, b1 = g["band"]
        verdict = "misses the reference" if (b1 < rlo or b0 > rhi) else "consistent with the reference"
        print(f"  GCI: R {g['R']:.3f} ({g['kind']}), order {g['p']:.2f}, extrapolated {g['ext']:.7f}, "
              f"band {b0:.7f} - {b1:.7f}, {verdict}")


if __name__ == "__main__":
    main()
