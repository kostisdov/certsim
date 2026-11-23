"""Functional a posteriori majorant (Repin) for a potential, as a baseline to the certificate.

For an admissible potential v and any current y in H(div) with y.n = 0 on the
insulating walls (y need not be divergence-free),

    |||u - v||| <= ||sigma grad v - y||_{1/sigma} + C_F / sqrt(sigma_min) * ||div y||,

where |||w|||^2 = int sigma |grad w|^2 and C_F is a Friedrichs constant,
||w|| <= C_F ||grad w|| for w vanishing on the electrodes and the ground. This is
the bound that learned-majorant methods for neural solutions evaluate. In the
monopolar problem every vertical line through the domain ends on the electrode
or on the ground at both ends, so the one-dimensional Friedrichs inequality
along z gives C_F = (H + Ztop) / pi, rigorously.

Here y is recovered from v itself (nodal averaging of sigma grad v, with the
normal component set to zero on the axis and the shaft), the cheapest
admissible choice that needs no second network. The integrals are evaluated in
double precision with Gauss quadrature, as such baselines are; this module is
not part of the trusted checker.
"""

import numpy as np

from certsim.spec import element_masks, node_masks


def _lagrange(p):
    t = np.arange(p + 1) / p
    C = np.linalg.inv(np.vander(t, increasing=True))  # C[a, k]: coefficient of x**a in basis k

    def val(x):
        return np.vander(np.atleast_1d(x), p + 1, increasing=True) @ C

    def der(x):
        X = np.atleast_1d(x)
        P = np.zeros((len(X), p + 1))
        for a in range(1, p + 1):
            P[:, a] = a * X ** (a - 1)
        return P @ C

    return val, der


def _blocks(arr, p, jj, ii):
    """Per-cell nodal blocks [cell, b (z), a (r)]."""
    off = np.arange(p + 1)
    return arr[(p * jj)[:, None, None] + off[None, :, None], (p * ii)[:, None, None] + off[None, None, :]]


def evaluate(spec, r, z, v, p=3, nq=6, per_cell=False):
    """Energy of v and the majorant with a recovered current. Returns a dict; with per_cell, also
    each active cell's (row, column) index and its share of ||div y||^2."""
    val, der = _lagrange(p)
    active, sigma = element_masks(spec, r, z)
    masks = node_masks(spec, r, z, p)
    jj, ii = np.nonzero(active)
    hr, hz = (r[1:] - r[:-1])[ii], (z[1:] - z[:-1])[jj]
    r1, sig = r[ii], sigma[jj, ii]

    # nodal recovery of y = sigma grad v
    Dn = der(np.arange(p + 1) / p)  # Dn[node, basis]: derivative of basis at node
    vb = _blocks(v, p, jj, ii)
    gr = np.einsum("xa,cba->cbx", Dn, vb) / hr[:, None, None] * sig[:, None, None]
    gz = np.einsum("zb,cba->cza", Dn, vb) / hz[:, None, None] * sig[:, None, None]
    shape = v.shape
    acc_r, acc_z, cnt = np.zeros(shape), np.zeros(shape), np.zeros(shape)
    off = np.arange(p + 1)
    J = (p * jj)[:, None, None] + off[None, :, None]
    I = (p * ii)[:, None, None] + off[None, None, :]
    J, I = np.broadcast_to(J, gr.shape), np.broadcast_to(I, gr.shape)
    np.add.at(acc_r, (J, I), gr)
    np.add.at(acc_z, (J, I), gz)
    np.add.at(cnt, (J, I), 1.0)
    cnt[cnt == 0] = 1.0
    yr, yz = acc_r / cnt, acc_z / cnt
    yr[masks["axis"] | masks["shaft"]] = 0.0  # y.n = 0 on the insulating walls

    # quadrature
    gx, gw = np.polynomial.legendre.leggauss(nq)
    gx, gw = 0.5 * (gx + 1), 0.5 * gw
    B, dB = val(gx), der(gx)  # [q, basis]
    yrb, yzb = _blocks(yr, p, jj, ii), _blocks(yz, p, jj, ii)

    def at(f, dx=False, dz=False):
        Bx = dB if dx else B
        Bz = dB if dz else B
        return np.einsum("xa,zb,cba->czx", Bx, Bz, f)

    rq = r1[:, None, None] + hr[:, None, None] * gx[None, None, :]
    w = 2 * np.pi * rq * (hr * hz)[:, None, None] * np.outer(gw, gw)[None]
    vr = at(vb, dx=True) / hr[:, None, None]
    vz = at(vb, dz=True) / hz[:, None, None]
    Yr, Yz = at(yrb), at(yzb)
    dYr = at(yrb, dx=True) / hr[:, None, None]
    dYz = at(yzb, dz=True) / hz[:, None, None]
    s = sig[:, None, None]
    E_v = float(np.sum(w * s * (vr ** 2 + vz ** 2)))
    e1 = float(np.sum(w * ((s * vr - Yr) ** 2 + (s * vz - Yz) ** 2) / s))
    e2 = float(np.sum(w * (dYr + Yr / rq + dYz) ** 2))
    C_F = (spec.H + spec.Ztop) / np.pi
    sig_min = min(spec.sigma_tissue, spec.sigma_blood)
    M = np.sqrt(e1) + C_F / np.sqrt(sig_min) * np.sqrt(e2)
    V2 = spec.voltage ** 2
    G_hi = E_v / V2
    G_lo = max(np.sqrt(E_v) - M, 0.0) ** 2 / V2
    out = {"E_v": E_v, "flux_term": float(np.sqrt(e1)), "friedrichs_term": float(C_F / np.sqrt(sig_min) * np.sqrt(e2)),
           "M": float(M), "C_F": float(C_F), "Z_point": 1 / G_hi,
           "Z_majorant": [1 / G_hi, (1 / G_lo) if G_lo > 0 else float("inf")]}
    if per_cell:
        out["cells"] = (jj, ii)
        out["div_sq"] = np.sum(w * (dYr + Yr / rq + dYz) ** 2, axis=(1, 2))
        out["friedrichs_factor"] = float(C_F / np.sqrt(sig_min))
    return out
