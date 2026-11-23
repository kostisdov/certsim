# Task: write a solver for the potential around an ablation electrode

Write one self-contained Python 3 program, `solver.py`, using only numpy and
scipy. It is run as

    python solver.py problem.npz solution.npz

## Physics

Steady electric conduction, axisymmetric about the z axis. In cylindrical
coordinates (r, z), with r >= 0, the potential u satisfies

    (1/r) d/dr( r sigma du/dr ) + d/dz( sigma du/dz ) = 0

in the conducting region. The conductivity sigma is piecewise constant:

- tissue, z < 0: sigma = sigma_tissue;
- blood, z > 0: sigma = sigma_blood.

## Geometry and boundary conditions

All lengths are in metres. The computational box is 0 <= r <= R, -H <= z <= Ztop.

- The **electrode** is a metal cylinder occupying 0 <= r <= a, gap <= z <= gap + L.
  It is held at potential u = voltage on all of its surface (its bottom face
  z = gap, 0 <= r <= a, and its side r = a, gap <= z <= gap + L).
- The **shaft** is an insulating cylinder occupying 0 <= r <= a, gap + L <= z <= Ztop.
  No current crosses its side r = a, gap + L < z < Ztop.
- The **outer boundary** r = R, the **bottom** z = -H and the **top** z = Ztop
  (for r >= a) are grounded: u = 0.
- The **axis** r = 0 (below the electrode) is a symmetry axis: du/dr = 0.

The electrode and the shaft are not part of the conducting region.

## Input: problem.npz

| Key | Meaning |
|---|---|
| `r` | 1-D array, increasing, grid lines in r from 0 to R |
| `z` | 1-D array, increasing, grid lines in z from -H to Ztop |
| `a`, `L`, `R`, `H`, `Ztop`, `gap` | geometry (floats) |
| `sigma_tissue`, `sigma_blood` | conductivities in S/m (floats) |
| `voltage` | electrode potential in V (float) |

The grid lines include every geometric breakpoint (r = 0, a, R and z = -H, 0,
gap, gap + L, Ztop), so every grid cell lies entirely in tissue, in blood, or
inside the electrode or shaft. The grid is not uniform.

## Output: solution.npz

| Key | Meaning |
|---|---|
| `v` | 2-D float array of shape (len(z), len(r)); `v[j, i]` is the potential at the node (r[i], z[j]) |
| `impedance` | float, the electrode impedance in ohms: voltage divided by the total current leaving the electrode |

At nodes on the electrode surface `v` must equal `voltage` exactly; at grounded
nodes it must equal 0 exactly. Nodes strictly inside the electrode must equal
`voltage`; nodes strictly inside the shaft are ignored.

Use a discretization that is accurate on this non-uniform grid, for example
bilinear finite elements on the grid cells with the axisymmetric weight r in
the integrals. Reply with the complete program in a single ```python code
block.
