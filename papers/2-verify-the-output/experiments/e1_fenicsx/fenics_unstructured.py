"""Solve the stated problem with FEniCSx on its own triangular mesh (pixi environment).

Stage 2 of E1: the discretization is independent of the certificate grid.
gmsh meshes the axisymmetric domain (tissue below z = 0, blood above, minus the
electrode and its insulating shaft), graded towards the two electrode corners;
FEniCSx solves with Lagrange triangles of degree k. The solution is then
evaluated at every node of the certificate grid and written to
fenics_solution.npz, with FEniCSx's own conductance. Nothing from certsim is
imported.

    pixi run python fenics_unstructured.py [k=3] [h_fine_mm=0.25] [h0_um=0.001] [scale]
"""

import sys
import time
from pathlib import Path

import gmsh
import numpy as np
import ufl
from dolfinx import fem, geometry
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io.gmsh import model_to_mesh
from mpi4py import MPI

HERE = Path(__file__).resolve().parent
TISSUE, BLOOD, ELECTRODE, GROUND = 1, 2, 10, 11


def build_mesh(d, h_fine, h0, kappa=0.25, growth=0.08):
    a, L, R, H, Ztop, gap = (float(d[k]) for k in ("a", "L", "R", "H", "Ztop", "gap"))
    assert gap == 0.0, "stage 2 meshes the contact case"
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("pfa")
    geo = gmsh.model.geo
    pts = {name: geo.addPoint(x, y, 0) for name, (x, y) in {
        "p1": (0, -H), "p2": (R, -H), "p3": (R, 0), "p4": (a, 0), "p5": (0, 0),
        "p6": (R, Ztop), "p7": (a, Ztop), "p8": (a, L)}.items()}
    ln = lambda u, v: geo.addLine(pts[u], pts[v])  # noqa: E731
    bottom, right_t, iface, tip, axis = ln("p1", "p2"), ln("p2", "p3"), ln("p3", "p4"), ln("p4", "p5"), ln("p5", "p1")
    right_b, top, shaft, side = ln("p3", "p6"), ln("p6", "p7"), ln("p7", "p8"), ln("p8", "p4")
    tissue = geo.addPlaneSurface([geo.addCurveLoop([bottom, right_t, iface, tip, axis])])
    blood = geo.addPlaneSurface([geo.addCurveLoop([-iface, right_b, top, shaft, side])])
    geo.synchronize()
    gmsh.model.addPhysicalGroup(2, [tissue], TISSUE)
    gmsh.model.addPhysicalGroup(2, [blood], BLOOD)
    gmsh.model.addPhysicalGroup(1, [tip, side], ELECTRODE)
    gmsh.model.addPhysicalGroup(1, [bottom, right_t, right_b, top], GROUND)

    # spacing h_fine near the electrode and the lesion zone, growing outside it,
    # and graded geometrically to h0 at the two electrode corners
    f = gmsh.model.mesh.field
    corners = f.add("Distance")
    f.setNumbers(corners, "PointsList", [pts["p4"], pts["p8"]])
    graded = f.add("MathEval")
    f.setString(graded, "F", f"{h0} + {kappa} * F{corners}")
    zone = f.add("MathEval")
    f.setString(zone, "F", f"{h_fine} + {growth} * (Max(0, x - 8e-3) + Max(0, -10e-3 - y) + Max(0, y - {L + 2e-3}))")
    size = f.add("Min")
    f.setNumbers(size, "FieldsList", [graded, zone])
    f.setAsBackgroundMesh(size)
    for opt in ("MeshSizeExtendFromBoundary", "MeshSizeFromPoints", "MeshSizeFromCurvature"):
        gmsh.option.setNumber(f"Mesh.{opt}", 0)
    gmsh.model.mesh.generate(2)
    data = model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)
    gmsh.finalize()
    return data


def main(k=3, h_fine=0.25e-3, h0=1e-9, scale=False):
    """scale: refine grading slope and far-field growth with h_fine, for mesh families."""
    d = np.load(HERE / "problem.npz")
    t0 = time.perf_counter()
    s = h_fine / 0.25e-3 if scale else 1.0
    data = build_mesh(d, h_fine, h0, kappa=0.25 * s, growth=0.08 * s)
    msh, cell_tags, facet_tags = data.mesh, data.cell_tags, data.facet_tags
    t_mesh = time.perf_counter() - t0

    S = fem.functionspace(msh, ("DG", 0))
    sig = fem.Function(S)
    sig.x.array[cell_tags.find(TISSUE)] = float(d["sigma_tissue"])
    sig.x.array[cell_tags.find(BLOOD)] = float(d["sigma_blood"])

    V = fem.functionspace(msh, ("Lagrange", k))
    u, w = ufl.TrialFunction(V), ufl.TestFunction(V)
    rr = ufl.SpatialCoordinate(msh)[0]
    a = sig * ufl.inner(ufl.grad(u), ufl.grad(w)) * rr * ufl.dx
    L = fem.Constant(msh, 0.0) * w * rr * ufl.dx
    volt = float(d["voltage"])
    fdim = msh.topology.dim - 1
    bcs = [fem.dirichletbc(fem.Constant(msh, val), fem.locate_dofs_topological(V, fdim, facet_tags.find(tag)), V)
           for tag, val in ((ELECTRODE, volt), (GROUND, 0.0))]
    problem = LinearProblem(a, L, bcs=bcs, petsc_options_prefix="e1u_",
                            petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                                           "pc_factor_mat_solver_type": "mumps"})
    uh = problem.solve()
    if isinstance(uh, tuple):
        uh = uh[0]
    t_solve = time.perf_counter() - t0 - t_mesh
    energy = fem.assemble_scalar(fem.form(2 * np.pi * sig * ufl.inner(ufl.grad(uh), ufl.grad(uh)) * rr * ufl.dx))
    conductance = float(energy) / volt**2

    # evaluate at every node of the certificate grid inside the domain
    r, z, p, dirichlet = d["r"], d["z"], int(d["degree"]), d["dirichlet"]
    rn = np.append((r[:-1, None] + np.outer(np.diff(r), np.arange(p) / p)).ravel(), r[-1])
    zn = np.append((z[:-1, None] + np.outer(np.diff(z), np.arange(p) / p)).ravel(), z[-1])
    RR, ZZ = np.meshgrid(rn, zn)
    a_el = float(d["a"])
    inside = ~((RR < a_el) & (ZZ > float(d["gap"])))  # outside the electrode and shaft
    inside &= ~np.isfinite(dirichlet)
    pts = np.stack([RR[inside], ZZ[inside], np.zeros(inside.sum())], axis=1)
    tree = geometry.bb_tree(msh, msh.topology.dim)
    hits = geometry.compute_colliding_cells(msh, geometry.compute_collisions_points(tree, pts), pts)
    cells = np.array([hits.links(i)[0] if len(hits.links(i)) else -1 for i in range(len(pts))], dtype=np.int32)
    assert (cells >= 0).all(), f"{(cells < 0).sum()} certificate nodes not found in the mesh"
    v = np.where(np.isfinite(dirichlet), dirichlet, 0.0)
    v[inside] = uh.eval(pts, cells).ravel()
    t_eval = time.perf_counter() - t0 - t_mesh - t_solve

    np.savez(HERE / "fenics_solution.npz", v=v, conductance=conductance, seconds=t_solve,
             mesh_seconds=t_mesh, eval_seconds=t_eval, k=k, cells=msh.topology.index_map(2).size_local,
             dofs=V.dofmap.index_map.size_local)
    print(f"gmsh + FEniCSx: {msh.topology.index_map(2).size_local} triangles, P{k}, "
          f"{V.dofmap.index_map.size_local} dofs; mesh {t_mesh:.1f} s, solve {t_solve:.1f} s, "
          f"evaluation at {inside.sum()} nodes {t_eval:.1f} s; impedance {1 / conductance:.9f} ohm")


if __name__ == "__main__":
    args = sys.argv[1:]
    main(int(args[0]) if args else 3,
         float(args[1]) * 1e-3 if len(args) > 1 else 0.25e-3,
         float(args[2]) * 1e-6 if len(args) > 2 else 1e-9,
         len(args) > 3 and args[3] == "scale")
