"""Export the stated problem for an external solver (run in the certsim environment).

Writes problem.npz: the geometry and conductivities, the electrode voltage,
and the certificate grid (tensor grid, polynomial degree, conductivity of every
cell, voltage of every Dirichlet node). A solver may use the certificate grid
or mesh the geometry itself; it imports nothing from certsim.

    uv run python export_problem.py [degree] [h_fine_mm]
"""

import sys
from pathlib import Path

import numpy as np

from certsim.solver import default_grid
from certsim.spec import Box, default_spec, element_masks, lesion_tiles, node_masks

HERE = Path(__file__).resolve().parent


def problem_spec():
    """The monopolar lesion-map problem of paper 1."""
    return default_spec(
        voxels=(),
        tiles=lesion_tiles(r_max=6e-3, depth=8e-3, step=0.25e-3),
        regions=(Box(0.0, 5e-3, -10e-3, -6e-3),),
    )


def main(degree=3, h_fine=0.25e-3):
    spec = problem_spec()
    r, z = default_grid(spec, h_fine=h_fine)
    active, sigma = element_masks(spec, r, z)
    masks = node_masks(spec, r, z, degree)
    dirichlet = np.full(masks["active"].shape, np.nan)
    dirichlet[masks["ground"]] = 0.0
    dirichlet[masks["electrode"]] = spec.tip_voltage
    geometry = dict(a=spec.a, L=spec.L, R=spec.R, H=spec.H, Ztop=spec.Ztop, gap=spec.gap,
                    sigma_tissue=spec.sigma_tissue, sigma_blood=spec.sigma_blood)
    np.savez(HERE / "problem.npz", r=r, z=z, degree=degree, voltage=spec.voltage,
             sigma=np.where(active, sigma, 0.0), dirichlet=dirichlet, **geometry)
    print(f"degree {degree}, {len(r) - 1} x {len(z) - 1} cells, {int(active.sum())} active; "
          f"{np.isfinite(dirichlet).sum()} Dirichlet nodes")


if __name__ == "__main__":
    args = sys.argv[1:]
    main(int(args[0]) if args else 3, float(args[1]) * 1e-3 if len(args) > 1 else 0.25e-3)
