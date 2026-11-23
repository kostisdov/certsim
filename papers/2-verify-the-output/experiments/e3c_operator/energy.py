"""Finite-difference energy weights on the certificate's node grid (numpy only).

The certificate measures errors in energy: the potential in
int sigma |grad v|^2 dmu, the stream function through its current,
int |curl psi / r|^2 / sigma dmu = 2 pi int (psi_r^2 + psi_z^2) / (sigma r) dr dz.
On a tensor grid graded to 1 nm, cells are up to 1e5 times longer than they are
thin, and a nodal error that differs between two nodes 1 nm apart costs a large
energy even if it is small in value. These weights turn differences between
neighbouring nodes into that energy (a lumped, five-point version of it), so a
training loss can measure what the certificate measures.
"""

import numpy as np


def _dual(x):
    d = np.empty_like(x)
    d[1:-1] = 0.5 * (x[2:] - x[:-2])
    d[0] = 0.5 * (x[1] - x[0])
    d[-1] = 0.5 * (x[-1] - x[-2])
    return d


def psi_r_nodes(r_lines, p):
    """r-coordinates of the stream-function nodes, equispaced in r^2 within each cell."""
    s = r_lines ** 2
    sn = np.append((s[:-1, None] + np.outer(np.diff(s), np.arange(p) / p)).ravel(), s[-1])
    return np.sqrt(sn)


def weights(rn, zn, active, sig_t, sig_b, kind):
    """Edge weights (wz, wr) for differences along z and along r.

    wz[j, i] multiplies (f[j+1, i] - f[j, i])^2 and wr[j, i] multiplies
    (f[j, i+1] - f[j, i])^2; kind is "v" for the potential's energy or "psi"
    for the stream function's complementary energy. rn, zn are node coordinates
    of the field (for psi, the r^2-equispaced ones).
    """
    dz, dr = np.diff(zn), np.diff(rn)
    zdual, rdual = _dual(zn), _dual(rn)
    zmid = 0.5 * (zn[1:] + zn[:-1])
    rmid = 0.5 * (rn[1:] + rn[:-1])
    sz = np.where(zmid <= 0.0, sig_t, sig_b)[:, None]  # material of a z-edge
    sr = np.where(zn <= 0.0, sig_t, sig_b)[:, None]  # material of an r-edge at height zn
    if kind == "v":
        wz = 2 * np.pi * sz * (rn * rdual)[None, :] / dz[:, None]
        wr = 2 * np.pi * sr * zdual[:, None] * (rmid / dr)[None, :]
    else:
        with np.errstate(divide="ignore"):
            inv_r = np.where(rn > 0, 1.0 / np.where(rn > 0, rn, 1.0), 0.0)
        wz = 2 * np.pi / sz * (rdual * inv_r)[None, :] / dz[:, None]
        wr = 2 * np.pi / sr * zdual[:, None] * (1.0 / (rmid * dr))[None, :]
    wz = wz * (active[1:, :] & active[:-1, :])
    wr = wr * (active[:, 1:] & active[:, :-1])
    return wz.astype(np.float64), wr.astype(np.float64)


def energy(f, wz, wr):
    return float(np.sum(wz * np.diff(f, axis=0) ** 2) + np.sum(wr * np.diff(f, axis=1) ** 2))
