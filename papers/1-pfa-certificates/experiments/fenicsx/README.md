# Table 7: the same check on FEniCSx

FEniCSx 0.11 solves the monopolar lesion-map problem on the certificate grid,
with its own assembly, quadrature, and linear solver, and the unchanged checker
certifies its potential. The two sides run in separate environments and exchange
only files.

| Step | Environment | Script | Output |
|---|---|---|---|
| 1. State the problem | certsim (uv) | `export_problem.py [degree] [h_fine_mm]` | `problem.npz`: grid, degree, cell conductivities, Dirichlet node values |
| 2. Solve | FEniCSx ([pixi](https://pixi.sh)) | `fenics_solve.py` | `fenics_solution.npz`: nodal potential and FEniCSx's own conductance |
| 3. Certify | certsim (uv) | `certify.py` | Certified bounds, compared with certsim's own solver |

From this folder:

```
uv run --project ../../../.. python export_problem.py 3 0.25
pixi run solve
uv run --project ../../../.. python certify.py
```

The current is supplied by `certsim.solver.certify_potential`, the untrusted
stream-function adapter, for both solvers; the checker verifies its result like
any other. The reference output is in `results/fenicsx_certify.log`.
