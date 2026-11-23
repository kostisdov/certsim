"""Run every generated solver, certify its output, and run a mesh study (certsim environment).

For each solver under samples/, at three grid resolutions: write problem.npz
(geometry and grid only, as SPEC.md describes), run the solver in its own
scratch folder with a time limit, certify its potential as is with the
unchanged checker (degree 1), and record its claimed impedance. The three
claims feed the paper 1 GCI. Results go to results.json.

Verdict per run (finest grid):
  crashed      the program failed, timed out, or wrote no valid output
  rejected     the checker found the potential inadmissible (boundary values)
  refuted      the claimed impedance lies outside the certified interval (beyond 1e-6 relative)
  unsupported  the claim lies inside, but the interval does not prove it to 0.1%
  certified    the certified interval proves the claim to within 0.1%

    uv run python run_solvers.py
"""

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from certsim.checker import CertificateError, check
from certsim.solver import certify_potential, default_grid
from certsim.spec import Box, default_spec, lesion_tiles

HERE = Path(__file__).resolve().parent
from certsim.gci import gci  # the procedure of paper 1

H_FINE = (1.0e-3, 0.5e-3, 0.25e-3)
TRUE_Z = (93.9502742, 93.9502758)  # paper 1 certified impedance, ohm
TIMEOUT = 600
ROUNDING = 1e-6  # relative slack for the solver's floating-point rounding of its claim


def problem_spec():
    return default_spec(voxels=(), tiles=lesion_tiles(r_max=6e-3, depth=8e-3, step=0.25e-3),
                        regions=(Box(0.0, 5e-3, -10e-3, -6e-3),))


def write_problem(path, spec, r, z):
    np.savez(path, r=r, z=z, a=spec.a, L=spec.L, R=spec.R, H=spec.H, Ztop=spec.Ztop, gap=spec.gap,
             sigma_tissue=spec.sigma_tissue, sigma_blood=spec.sigma_blood, voltage=spec.voltage)


def grid(spec, h):
    """Systematic refinement, as in paper 1: spacing, grading slope and far-field growth scale together."""
    s = h / 0.25e-3
    return default_grid(spec, h_fine=h, h0=1e-9, kappa=0.25 * s, growth=0.08 * s)


def run_one(solver, spec, h):
    r, z = grid(spec, h)
    out = {"h": h}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        write_problem(tmp / "problem.npz", spec, r, z)
        (tmp / "solver.py").write_text(solver.read_text())
        t0 = time.perf_counter()
        try:
            proc = subprocess.run([sys.executable, "solver.py", "problem.npz", "solution.npz"], cwd=tmp,
                                  capture_output=True, text=True, timeout=TIMEOUT)
        except subprocess.TimeoutExpired:
            return {**out, "verdict": "crashed", "why": f"timeout {TIMEOUT} s"}
        out["seconds"] = round(time.perf_counter() - t0, 1)
        if proc.returncode != 0 or not (tmp / "solution.npz").exists():
            return {**out, "verdict": "crashed", "why": (proc.stderr or "no solution.npz").strip()[-300:]}
        try:
            sol = np.load(tmp / "solution.npz")
            v, z_claim = np.asarray(sol["v"], float), float(sol["impedance"])
        except Exception as e:  # noqa: BLE001  (any malformed output)
            return {**out, "verdict": "crashed", "why": f"bad output: {e}"[:300]}
        problems = []
        if v.shape != (len(z), len(r)):
            problems.append(f"v has shape {v.shape}, expected {(len(z), len(r))}")
        elif not np.isfinite(v).all():
            problems.append(f"{int((~np.isfinite(v)).sum())} non-finite values in v")
        if not (np.isfinite(z_claim) and z_claim > 0):
            problems.append(f"impedance {z_claim}")
        if problems:
            return {**out, "verdict": "crashed", "why": "bad output: " + "; ".join(problems)}
    out["Z_claim"] = z_claim
    try:
        rep = check(spec, certify_potential(spec, r, z, v, 1, claims={"conductance": 1 / z_claim}))
    except CertificateError as e:
        return {**out, "verdict": "rejected", "why": str(e)[:300]}
    lo, hi = rep.impedance.lo, rep.impedance.hi
    out.update(Z_cert=[lo, hi], eta=rep.eta.hi)
    tol = ROUNDING * z_claim  # the solver's own floating-point rounding is not a wrong claim
    if not lo - tol <= z_claim <= hi + tol:
        out["verdict"] = "refuted"
    elif lo >= z_claim * (1 - 1e-3) and hi <= z_claim * (1 + 1e-3):
        out["verdict"] = "certified"
    else:
        out["verdict"] = "unsupported"
    return out


def main():
    spec = problem_spec()
    path = HERE / "results.json"
    results = json.loads(path.read_text()) if path.exists() else {}
    for solver in sorted(HERE.glob("samples/*/*/solver.py")):
        key = str(solver.parent.relative_to(HERE / "samples"))
        if key in results:
            continue
        runs = [run_one(solver, spec, h) for h in H_FINE]
        claims = [o.get("Z_claim") for o in runs]
        g = None
        if all(c is not None for c in claims):
            if len(set(claims)) < 3:  # a claim repeats across resolutions: no convergence to estimate
                g = {"kind": "degenerate (repeated claims)", "R": float("nan"), "ext": float("nan")}
            else:
                g = gci(claims[::-1])
        correct = claims[-1] is not None and abs(claims[-1] - sum(TRUE_Z) / 2) / (sum(TRUE_Z) / 2) < 1e-3
        results[key] = {"runs": runs, "correct_to_0.1%": correct,
                        "gci": None if g is None else {k: (list(v) if isinstance(v, tuple) else v) for k, v in g.items()}}
        path.write_text(json.dumps(results, indent=1))
        fin = runs[-1]
        gtxt = "no mesh study" if g is None else f"mesh study {g['kind']} (R {g['R']:.2f}), extrapolated {g['ext']:.4f}"
        print(f"{key}: finest {fin['verdict']}" + (f", claim {fin['Z_claim']:.4f} ohm" if "Z_claim" in fin else "")
              + (f", certified {fin['Z_cert'][0]:.4f}-{fin['Z_cert'][1]:.4f}" if "Z_cert" in fin else "")
              + f"; {gtxt}; correct to 0.1%: {correct}" + (f"; why: {fin['why'][:120]}" if "why" in fin else ""))


if __name__ == "__main__":
    main()
