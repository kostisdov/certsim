"""FEniCSx computes the current too: a native certificate, no adapter (pixi environment).

The current is F = (-psi_z / r, psi_r / r) for a stream function psi that is zero
on the axis and constant on the insulating shaft. In s = r^2 it minimizes the
complementary energy
    J(psi) = 1/2 int pi (psi_z^2 / s + 4 psi_s^2) / sigma ds dz - 2 pi V psi_shaft,
whose minimizer gives F = sigma grad u (Thomson's principle). FEniCSx solves for
psi on a quadrilateral mesh in (s, z) with tensor Lagrange elements of the
certificate degree (equispaced in s, as the certificate format expects), first
with psi_shaft = 1 and then rescaled to the optimal shaft value, which is linear
in it. Run after fenics_solve.py on the same problem; adds psi and the dual
conductance to fenics_solution.npz.

    pixi run python fenics_dual.py
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
    sol = dict(np.load(HERE / "fenics_solution.npz"))
    r, z, p = d["r"], d["z"], int(d["degree"])
    sigma_cells, volt, a = d["sigma"], float(d["voltage"]), float(d["a"])
    top = float(d["gap"]) + float(d["L"])
    s_lines = r**2
    nr = len(r)
    t0 = time.perf_counter()

    jj, ii = np.nonzero(sigma_cells > 0)
    cells = np.stack([jj * nr + ii, jj * nr + ii + 1, (jj + 1) * nr + ii, (jj + 1) * nr + ii + 1], axis=1)
    used, cells = np.unique(cells, return_inverse=True)
    cells = cells.reshape(-1, 4).astype(np.int64)
    x = np.stack([s_lines[used % nr], z[used // nr]], axis=1)
    coord = basix.ufl.element("Lagrange", "quadrilateral", 1, shape=(2,))
    msh = mesh.create_mesh(MPI.COMM_WORLD, cells, coord, x)

    S = fem.functionspace(msh, ("DG", 0))
    inv_sig = fem.Function(S)
    inv_sig.x.array[:] = 1.0 / sigma_cells[jj, ii][msh.topology.original_cell_index]
    el = basix.ufl.element("Lagrange", "quadrilateral", p, lagrange_variant=basix.LagrangeVariant.equispaced)
    V = fem.functionspace(msh, el)
    psi, w = ufl.TrialFunction(V), ufl.TestFunction(V)
    s, zz = ufl.SpatialCoordinate(msh)
    form = inv_sig * (psi.dx(1) * w.dx(1) / s + 4 * psi.dx(0) * w.dx(0)) * ufl.dx(metadata={"quadrature_degree": 2 * p + 6})
    rhs = fem.Constant(msh, 0.0) * w * ufl.dx

    # walls: the axis (s = 0, z < gap), where psi = 0, and the shaft (s = a^2, z >= top), where psi = 1
    xd = V.tabulate_dof_coordinates()[:, :2]
    tol = 1e-9 * a * a  # FEniCSx's dof coordinates are not bit-exact
    axis = np.nonzero(np.abs(xd[:, 0]) <= tol)[0].astype(np.int32)
    shaft = np.nonzero((np.abs(xd[:, 0] - a * a) <= tol) & (xd[:, 1] >= top - 1e-12))[0].astype(np.int32)
    assert len(axis) and len(shaft), "no axis or shaft dofs found"
    g = fem.Function(V)
    g.x.array[shaft] = 1.0
    bcs = [fem.dirichletbc(g, np.concatenate([axis, shaft]))]
    problem = LinearProblem(form, rhs, bcs=bcs, petsc_options_prefix="e1d_",
                            petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                                           "pc_factor_mat_solver_type": "mumps"})
    psi1 = problem.solve()
    if isinstance(psi1, tuple):
        psi1 = psi1[0]
    # J(c psi1) = c^2 A / 2 - 2 pi V c, with A = pi * form(psi1, psi1): optimal c = 2 V / form(psi1, psi1)
    A1 = fem.assemble_scalar(fem.form(ufl.replace(form, {psi: psi1, w: psi1})))
    c = 2 * volt / float(A1)
    t_dual = time.perf_counter() - t0

    # node grid of the certificate: equispaced in s within each cell, matching FEniCSx's nodes
    sn = np.append((s_lines[:-1, None] + np.outer(np.diff(s_lines), np.arange(p) / p)).ravel(), s_lines[-1])
    zn = np.append((z[:-1, None] + np.outer(np.diff(z), np.arange(p) / p)).ravel(), z[-1])
    ci = np.clip(np.searchsorted(sn, xd[:, 0]), 1, len(sn) - 1)
    ci -= (np.abs(sn[ci - 1] - xd[:, 0]) < np.abs(sn[ci] - xd[:, 0]))
    cj = np.clip(np.searchsorted(zn, xd[:, 1]), 1, len(zn) - 1)
    cj -= (np.abs(zn[cj - 1] - xd[:, 1]) < np.abs(zn[cj] - xd[:, 1]))
    # each dof must sit on a node, to within a tiny fraction of the local node spacing
    gap_s = np.diff(sn).min() if len(sn) > 1 else 1.0
    local = np.minimum(np.abs(np.diff(sn, prepend=-np.inf))[ci], np.abs(np.diff(sn, append=np.inf))[ci])
    assert np.all(np.abs(sn[ci] - xd[:, 0]) <= 1e-6 * np.maximum(local, gap_s))
    assert np.allclose(zn[cj], xd[:, 1], rtol=0, atol=1e-12)
    psi_grid = np.zeros((len(zn), len(sn)))
    psi_grid[cj, ci] = c * psi1.x.array
    psi_grid[cj[axis], ci[axis]] = 0.0
    psi_grid[cj[shaft], ci[shaft]] = c  # one exact value on the whole shaft wall
    sol.update(psi=psi_grid, dual_conductance=2 * np.pi * c / volt, dual_seconds=t_dual)
    np.savez(HERE / "fenics_solution.npz", **sol)
    print(f"FEniCSx dual: {V.dofmap.index_map.size_local} dofs, {t_dual:.1f} s; "
          f"impedance from its current {volt / (2 * np.pi * c):.9f} ohm")


if __name__ == "__main__":
    main()
