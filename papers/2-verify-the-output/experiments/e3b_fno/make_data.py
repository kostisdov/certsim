"""Training and test data for the FNO on a uniform grid (run in the certsim environment).

Reference potentials at many conductivity ratios rho are computed with the
certsim solver (degree 2) and interpolated onto a uniform grid over the whole
domain, as an FNO needs. Inputs per sample: log(rho), r, z and a domain mask
(1 in tissue and blood, 0 in the electrode and shaft); target: the potential,
equal to the electrode voltage inside the electrode. Writes data.npz.

    uv run --project ../../../.. python make_data.py
"""

from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from certsim.solver import default_grid, solve
from certsim.spec import default_spec

HERE = Path(__file__).resolve().parent
SIGMA_T, P = 0.25, 2
NR, NZ = 256, 512  # uniform grid: 0.2 mm spacing over 50 mm x 100 mm
TRAIN = np.geomspace(1.7, 3.75, 32)
TEST = np.array([1.8, 2.0, 2.4, 2.8, 3.1, 3.6, 0.7, 1.0, 1.3, 5.0, 7.0, 10.0])


def spec_for(rho):
    return default_spec(voxels=(), regions=(), sigma_tissue=SIGMA_T, sigma_blood=rho * SIGMA_T)


def node_coords(r, z, p):
    rn = np.append((r[:-1, None] + np.outer(np.diff(r), np.arange(p) / p)).ravel(), r[-1])
    zn = np.append((z[:-1, None] + np.outer(np.diff(z), np.arange(p) / p)).ravel(), z[-1])
    return rn, zn


def main():
    spec = spec_for(2.8)
    r, z = default_grid(spec, h_fine=0.5e-3)
    rn, zn = node_coords(r, z, P)
    ru = np.linspace(0.0, spec.R, NR)
    zu = np.linspace(-spec.H, spec.Ztop, NZ)
    RU, ZU = np.meshgrid(ru, zu, indexing="ij")
    in_metal = (RU <= spec.a) & (ZU >= spec.gap)
    in_electrode = in_metal & (ZU <= spec.gap + spec.L)
    mask = (~in_metal).astype(float)
    pts = np.stack([ZU.ravel(), RU.ravel()], axis=1)

    def field(rho):
        cert = solve(spec_for(rho), r, z, degree=P)
        f = RegularGridInterpolator((zn, rn), cert.v, method="linear")(pts).reshape(NR, NZ)
        f[in_electrode] = spec.voltage
        f[in_metal & ~in_electrode] = 0.0  # shaft: not part of the domain, masked in the loss
        return f

    out = {}
    for name, rhos in (("train", TRAIN), ("test", TEST)):
        out[f"{name}_rho"] = rhos
        out[f"{name}_v"] = np.array([field(rho) for rho in rhos])
        print(f"{name}: {len(rhos)} fields")
    np.savez(HERE / "data.npz", r=ru, z=zu, mask=mask, **out)


if __name__ == "__main__":
    main()
