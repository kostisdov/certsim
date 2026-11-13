"""Certify the FEniCSx potential with the unchanged checker (run in the certsim environment).

Reads problem.npz and fenics_solution.npz, fits the current with the untrusted
stream-function adapter, checks the certificate against the specification, and
compares with certsim's own solver on the same grid and degree.

    uv run python certify.py
"""

import time
from pathlib import Path

import numpy as np

from certsim.checker import check
from certsim.solver import Certificate, certify_potential, solve
from export_problem import problem_spec

HERE = Path(__file__).resolve().parent


def summary(name, rep, claim=None):
    widths = np.array([iv.rel_width for iv in rep.tile_rms_field])
    z = rep.impedance
    line = (f"{name:10s} impedance {z.lo:.9f} - {z.hi:.9f} ohm; eta {rep.eta.hi:.3e}; "
            f"tile width median {np.median(widths) * 100:.2f}%, worst {widths.max() * 100:.1f}%")
    if claim is not None:
        line += f"; own claim {1 / claim:.9f} ohm, proven to 1e-5: {rep.conductance.supports(claim, 1e-5)}"
    return line


def main():
    spec = problem_spec()
    prob, sol = np.load(HERE / "problem.npz"), np.load(HERE / "fenics_solution.npz")
    r, z, p = prob["r"], prob["z"], int(prob["degree"])

    t0 = time.perf_counter()
    cert = certify_potential(spec, r, z, sol["v"], p, claims={"conductance": float(sol["conductance"])})
    t_adapt = time.perf_counter() - t0
    rep = check(spec, cert)
    t_check = time.perf_counter() - t0 - t_adapt
    print(f"FEniCSx solve {float(sol['seconds']):.1f} s, adapter {t_adapt:.1f} s, check {t_check:.1f} s")
    print(summary("FEniCSx", rep, float(sol["conductance"])))

    if "psi" in sol.files:  # FEniCSx also supplied the current (fenics_dual.py): no adapter
        native = Certificate(r, z, sol["v"], sol["psi"], np.zeros((0, *sol["v"].shape)),
                             np.zeros((0, *sol["v"].shape)), p, {"conductance": float(sol["conductance"])})
        t0 = time.perf_counter()
        rep_n = check(spec, native)
        print(f"native FEniCSx certificate (its potential and its current), check {time.perf_counter() - t0:.1f} s, "
              f"dual solve {float(sol['dual_seconds']):.1f} s")
        print(summary("native", rep_n, float(sol["conductance"])))

    own = solve(spec, r, z, degree=p)
    print(summary("certsim", check(spec, own), own.claims["conductance"]))
    print(f"max |v_fenics - v_certsim| = {np.abs(cert.v - own.v).max():.3e} V")


if __name__ == "__main__":
    main()
