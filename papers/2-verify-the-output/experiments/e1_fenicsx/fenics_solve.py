"""Solve the stated problem with FEniCSx (run in the pixi environment).

Reads problem.npz, builds a quadrilateral mesh of the active cells, solves
div(sigma grad u) = 0 in axisymmetric form with tensor Lagrange elements of the
given degree (equispaced nodes), and writes fenics_solution.npz: the potential
at every node of the degree-p node grid and FEniCSx's own conductance. Nothing
from certsim is imported.

    pixi run python fenics_solve.py
"""

import time
from pathlib import Path

import basix
import basix.ufl
import numpy as np
import ufl
from dolfinx import fem, mesh
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI

HERE = Path(__file__).resolve().parent


def main():
    d = np.load(HERE / "problem.npz")
    r, z, p = d["r"], d["z"], int(d["degree"])
    sigma_cells, dirichlet, volt = d["sigma"], d["dirichlet"], float(d["voltage"])
    nr, nz = len(r), len(z)
    t0 = time.perf_counter()

    # quadrilateral mesh of the active cells; vertex (i, j) has index j * nr + i
    jj, ii = np.nonzero(sigma_cells > 0)
    cells = np.stack([jj * nr + ii, jj * nr + ii + 1, (jj + 1) * nr + ii, (jj + 1) * nr + ii + 1], axis=1)
    used, cells = np.unique(cells, return_inverse=True)
    cells = cells.reshape(-1, 4).astype(np.int64)
    x = np.stack([r[used % nr], z[used // nr]], axis=1)
    coord = basix.ufl.element("Lagrange", "quadrilateral", 1, shape=(2,))
    msh = mesh.create_mesh(MPI.COMM_WORLD, cells, coord, x)

    # conductivity per cell, following FEniCSx's reordering of the cells
    order = msh.topology.original_cell_index
    S = fem.functionspace(msh, ("DG", 0))
    sig = fem.Function(S)
    sig.x.array[:] = sigma_cells[jj, ii][order]

    el = basix.ufl.element("Lagrange", "quadrilateral", p, lagrange_variant=basix.LagrangeVariant.equispaced)
    V = fem.functionspace(msh, el)
    u, w = ufl.TrialFunction(V), ufl.TestFunction(V)
    rr = ufl.SpatialCoordinate(msh)[0]
    a = sig * ufl.inner(ufl.grad(u), ufl.grad(w)) * rr * ufl.dx
    L = fem.Constant(msh, 0.0) * w * rr * ufl.dx

    # map every degree of freedom to its node on the degree-p node grid
    rn = np.concatenate([r[:-1, None] + np.outer(np.diff(r), np.arange(p) / p), ], axis=None)
    rn = np.append(np.sort(rn), r[-1])
    zn = np.append(np.sort(np.concatenate([z[:-1, None] + np.outer(np.diff(z), np.arange(p) / p)], axis=None)), z[-1])
    xd = V.tabulate_dof_coordinates()[:, :2]
    ci = np.clip(np.searchsorted(rn, xd[:, 0]), 1, len(rn) - 1)
    ci -= (np.abs(rn[ci - 1] - xd[:, 0]) < np.abs(rn[ci] - xd[:, 0]))
    cj = np.clip(np.searchsorted(zn, xd[:, 1]), 1, len(zn) - 1)
    cj -= (np.abs(zn[cj - 1] - xd[:, 1]) < np.abs(zn[cj] - xd[:, 1]))
    assert np.allclose(rn[ci], xd[:, 0], rtol=0, atol=1e-12) and np.allclose(zn[cj], xd[:, 1], rtol=0, atol=1e-12)

    vals = dirichlet[cj, ci]
    bdofs = np.nonzero(np.isfinite(vals))[0].astype(np.int32)
    g = fem.Function(V)
    g.x.array[bdofs] = vals[bdofs]
    bc = fem.dirichletbc(g, bdofs)
    problem = LinearProblem(a, L, bcs=[bc], petsc_options_prefix="e1_",
                            petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                                           "pc_factor_mat_solver_type": "mumps"})
    uh = problem.solve()
    if isinstance(uh, tuple):
        uh = uh[0]
    t_solve = time.perf_counter() - t0

    energy = fem.assemble_scalar(fem.form(2 * np.pi * sig * ufl.inner(ufl.grad(uh), ufl.grad(uh)) * rr * ufl.dx))
    conductance = float(energy) / volt**2

    v = np.where(np.isfinite(dirichlet), dirichlet, 0.0)
    v[cj, ci] = uh.x.array
    np.savez(HERE / "fenics_solution.npz", v=v, conductance=conductance, seconds=t_solve,
             dolfinx_version=__import__("dolfinx").__version__)
    print(f"FEniCSx {__import__('dolfinx').__version__}: {msh.topology.index_map(2).size_local} cells, "
          f"{V.dofmap.index_map.size_local} dofs, solved in {t_solve:.1f} s; "
          f"impedance {1 / conductance:.9f} ohm")


if __name__ == "__main__":
    main()
