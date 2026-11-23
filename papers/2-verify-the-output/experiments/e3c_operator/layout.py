"""Write layout.npz: node masks and node coordinates of the index grid (certsim env).

The masks are the same for every member of the family, because the grid
mapping keeps every line in place in index space. The training script, which
runs in the PyTorch environment without certsim, reads this file.

    uv run --project ../../../.. python layout.py
"""

from pathlib import Path

import numpy as np

from common import DEGREE, L0, grid_for, masks_for


def nodes(g, p):
    return np.append((g[:-1, None] + np.outer(np.diff(g), np.arange(p) / p)).ravel(), g[-1])


def main():
    m = masks_for(L0)
    r, z0 = grid_for(L0)
    np.savez(Path(__file__).resolve().parent / "layout.npz",
             active=m["active"], electrode=m["electrode"], ground=m["ground"], axis=m["axis"],
             shaft=m["shaft"], r_nodes=nodes(r, DEGREE), z_lines=z0, L0=L0, degree=DEGREE)
    print("layout:", m["active"].shape, "active nodes", int(m["active"].sum()))


if __name__ == "__main__":
    main()
