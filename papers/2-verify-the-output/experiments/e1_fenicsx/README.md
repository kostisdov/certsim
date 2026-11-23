# E1: certifying FEniCSx

FEniCSx solves the stated problem with its own assembly and linear solver; the
unchanged certsim checker certifies its output. The two sides run in separate
environments and exchange only files.

| Step | Environment | Script | Output |
|---|---|---|---|
| 1. State the problem | certsim (uv) | `export_problem.py [degree] [h_fine_mm]` | `problem.npz`: grid, degree, cell conductivities, Dirichlet node values |
| 2. Solve | FEniCSx (pixi) | `fenics_solve.py` | `fenics_solution.npz`: nodal potential and FEniCSx's own conductance |
| 3. Certify | certsim (uv) | `certify.py` | Certified bounds, compared with certsim's own solver |

```
uv run python export_problem.py 3 0.25
pixi run solve
uv run python certify.py
```

Run the uv commands from this folder with the repository's environment
(`uv run --project ../../../..`), or with `PYTHONPATH` pointing to `src/`.

The current is supplied by `certsim.solver.certify_potential`, an untrusted
adapter that fits a stream function to sigma grad v; the checker verifies its
result like any other.

## Stage 1 result (same grid, tensor elements)

Paper 1 lesion-map problem, degree 3, 169 x 332 cells (45 416 active),
410 248 unknowns, FEniCSx 0.11.0:

| | Certified impedance (ohm) | eta | Tile width, median / worst | Own claim proven to 1e-5 |
|---|---|---|---|---|
| FEniCSx | 93.950274297 - 93.950275789 | 1.30e-5 | 1.89% / 18.0% | yes |
| certsim solver | 93.950274297 - 93.950275789 | 1.30e-5 | 1.89% / 18.0% | yes |

The two potentials agree to 7e-9 V. FEniCSx solves in 2.7 s, the adapter takes
5.2 s and the check 17 s. FEniCSx's claimed impedance equals the certified
lower bound, as the energy of a Galerkin solution must.

## Stage 1b: a native certificate (FEniCSx supplies the current too)

The adapter's global stream-function fit is itself a dual solve, so in stage 1
half of the bound came from certsim, not from FEniCSx. `fenics_dual.py` removes
it: FEniCSx minimizes the complementary energy for the stream function on a mesh
in (s = r^2, z), with tensor elements of the certificate degree, psi = 0 on the
axis and a constant on the shaft. The certificate then consists of FEniCSx's
potential and FEniCSx's current, checked with no certsim solver code involved.

```
pixi run python fenics_solve.py
pixi run python fenics_dual.py
uv run python certify.py        # reports the native and the adapter certificates
```

| Certificate | Certified impedance (ohm) | eta | Tiles, median / worst |
|---|---|---|---|
| native: FEniCSx potential and FEniCSx current | 93.950274297 - 93.950275789 | 1.30e-5 | 1.89% / 18.0% |
| FEniCSx potential, certsim adapter current | 93.950274297 - 93.950275789 | 1.30e-5 | 1.89% / 18.0% |
| certsim solver | 93.950274297 - 93.950275789 | 1.30e-5 | 1.89% / 18.0% |

The lower end is the energy of FEniCSx's potential, the upper end the energy of
FEniCSx's current (Thomson's principle); FEniCSx's primal and dual solves take
1.9 s each.

## Stage 2 result (FEniCSx on its own triangular mesh)

`fenics_unstructured.py [k] [h_fine_mm] [h0_um]`: gmsh meshes the geometry,
graded geometrically towards both electrode corners down to `h0`; FEniCSx
solves with Lagrange triangles of degree `k`; the solution is evaluated at the
nodes of the certificate grid (degree 3, paper 1 grid), and `certify.py`
checks it as before.

```
uv run python export_problem.py 3 0.25
pixi run python fenics_unstructured.py 3 0.25 0.001
uv run python certify.py
```

| FEniCSx | Unknowns | Certified impedance (ohm) | eta | Tile width, median / worst | Own claim proven to 1e-5 |
|---|---|---|---|---|---|
| P2, 0.25 mm, h0 1 um | 21 591 | 93.809654 - 93.950276 | 4.0e-3 | 200% / 200% | no |
| P2, 0.25 mm, h0 1 nm | 26 310 | 93.949857 - 93.950276 | 2.2e-4 | 32% / 200% | yes |
| P3, 0.25 mm, h0 1 nm | 58 927 | 93.950272 - 93.950276 | 2.0e-5 | 3.0% / 28% | yes |
| P2, 0.1 mm, h0 1 nm | 82 694 | 93.949975 - 93.950276 | 1.8e-4 | 27% / 200% | yes |
| P3, 0.1 mm, h0 1 nm | 185 710 | 93.950272 - 93.950276 | 2.1e-5 | 3.1% / 29% | yes |

Every interval contains the stage 1 bound. With cubic elements and grading to
1 nm at the corners, FEniCSx on its own mesh is certified within a factor of
about 1.6 of the matched-grid eta, from seven times fewer unknowns. The width
reports the quality of the external discretization: quadratic elements give a
bound ten times wider and mostly undecided tiles, refining from 0.25 to 0.1 mm
barely helps (the corner singularity and the element degree dominate), and
grading only to 1 um leaves the bound valid but loose.

The certificate is for the potential as evaluated on the certificate grid, so
it also includes the transfer error between the two discretizations.

## Grid convergence study on the FEniCSx meshes

`gci_fenicsx.py` runs two families of gmsh meshes refined by a factor of 2
(spacing, grading slope and far-field growth all scaled; 1 to 0.125 mm), feeds
FEniCSx's own impedance to the paper 1 three-grid GCI, and certifies every
solution. Results are cached in `gci_fenicsx.json`. Reference: the paper 1
certified impedance, 93.9502743 - 93.9502758 ohm.

| Family | Finest mesh: FEniCSx impedance | GCI: R, order, band (ohm) | GCI verdict | Certified, finest mesh (ohm) | Tile width, finest |
|---|---|---|---|---|---|
| P2, no corner grading | 93.7592719 | 0.47 (converging), 1.09, 93.5686759 - 93.9498680 | **excludes the true value** | 93.6313872 - 93.9502758 | 200% (undecided) |
| P3, graded to 1 nm | 93.9502744 | 0.035 (converging), 4.82, 93.9502744 - 93.9502744 | consistent | 93.9502739 - 93.9502758 | 2.1% |

On the ungraded quadratic family, the triplet passes the GCI's own validity
check (monotone convergence, R in (0, 1)), yet its band excludes the true
impedance, narrowly (by 4e-4 ohm). The extrapolated value, 93.9117 ohm, is off
by 0.04 ohm. On every mesh of both families the certified interval contains
the true value; on the ungraded family its undecided tiles show that the mesh
is inadequate near the electrode, while on the graded family the certificate
tightens with refinement (eta 1.3e-3, 9.7e-5, 2.0e-5, 1.5e-5).

## Next

The FEniCSx field on a certificate grid built from its own mesh nodes, to
remove the transfer error.
