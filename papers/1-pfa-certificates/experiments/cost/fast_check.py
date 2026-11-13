"""Vectorized float64 evaluation of the certificate bounds (E4 prototype, not rigorous).

Computes the same quantities as certsim.checker.check (hypercircle gap eta,
Dirichlet and Thomson energies, the centre norm over the domain and over each
lesion tile) and the same bounds, but in float64, vectorized over chunks of
cells, with Gauss quadrature of order nq on every cell. It is not rigorous:
rounding is not controlled, and the 1/r part of the current is integrated by
quadrature instead of exactly. Its purpose is to measure what a vectorized
check costs relative to a solve, and how closely it reproduces the rigorous
checker. The admissibility checks (exact equalities of boundary values) are the
same as in the rigorous checker.
"""

import numpy as np

from certsim.solver import _gauss, _lagrange
from certsim.spec import element_masks, node_masks

TWO_PI = 2 * np.pi


def _admissible(spec, cert, masks):
    v, psi = cert.v, cert.psi
    if not (np.all(v[masks["electrode"]] == spec.tip_voltage) and np.all(v[masks["ground"]] == 0.0)):
        raise ValueError("potential does not meet the electrode or ground values")
    if not np.all(psi[masks["axis"]] == 0.0):
        raise ValueError("stream function is not zero on the axis")
    shaft = psi[masks["shaft"] & ~masks["axis"]]
    if shaft.size and not np.all(shaft == shaft[0]):
        raise ValueError("stream function is not constant on the shaft")


def fast_check(spec, cert, nq=None, chunk=4096):
    """Return a dict with impedance bounds, eta and tile RMS bounds (floats)."""
    r, z, p = np.asarray(cert.r), np.asarray(cert.z), int(cert.degree)
    nq = nq or p + 6
    masks = node_masks(spec, r, z, p)
    _admissible(spec, cert, masks)
    active, sigma = element_masks(spec, r, z)
    jj, ii = np.nonzero(active)
    nr_f = p * (len(r) - 1) + 1
    a_idx, b_idx = (x.ravel() for x in np.meshgrid(np.arange(p + 1), np.arange(p + 1), indexing="xy"))
    x, w = _gauss(nq)
    val, der = _lagrange(p)
    L, dL = val(x), der(x)  # (nq, p+1)
    v_flat, psi_flat = cert.v.ravel(), cert.psi.ravel()

    # lesion tiles: index of the tile each cell lies in (-1 if none)
    tile_of = np.full(len(jj), -1)
    for t, b in enumerate(spec.tiles):
        inside = (r[ii] >= b.r_lo) & (r[ii + 1] <= b.r_hi) & (z[jj] >= b.z_lo) & (z[jj + 1] <= b.z_hi)
        tile_of[inside] = t
    sums = dict(E_v=0.0, W_F=0.0, eta2=0.0, cc=0.0)
    cc_tile = np.zeros(len(spec.tiles))

    for start in range(0, len(jj), chunk):
        j, i = jj[start:start + chunk], ii[start:start + chunk]
        r1, r2, z1, z2 = r[i], r[i + 1], z[j], z[j + 1]
        hr, hz, s1, hs = r2 - r1, z2 - z1, r1**2, r2**2 - r1**2
        sig = sigma[j, i]
        nodes = (p * j[:, None] + b_idx[None]) * nr_f + (p * i[:, None] + a_idx[None])  # (e, nb)
        vn, pn = v_flat[nodes], psi_flat[nodes]
        rq = r1[:, None] + hr[:, None] * x[None]  # (e, q)
        mu = TWO_PI * rq[:, :, None] * (w[:, None] * w[None])[None] * (hr * hz)[:, None, None]  # (e, qr, qz)
        # potential: Q_p in (xi, eta)
        gr = np.einsum("en,qn,pn->eqp", vn, dL[:, a_idx], L[:, b_idx]) / hr[:, None, None]
        gz = np.einsum("en,qn,pn->eqp", vn, L[:, a_idx], dL[:, b_idx]) / hz[:, None, None]
        svr, svz = sig[:, None, None] * gr, sig[:, None, None] * gz
        # stream function: degree p in (xs, eta), xs = (r^2 - s1) / hs; F = (-psi_z / r, 2 psi_s)
        xs = (rq**2 - s1[:, None]) / hs[:, None]
        Ls, dLs = val(xs), der(xs)  # (e, q, p+1)
        fr = -np.einsum("en,eqn,pn->eqp", pn, Ls[:, :, a_idx], dL[:, b_idx]) / hz[:, None, None] / rq[:, :, None]
        fz = 2 * np.einsum("en,eqn,pn->eqp", pn, dLs[:, :, a_idx], L[:, b_idx]) / hs[:, None, None]
        inv_sig = (mu / sig[:, None, None])
        sums["E_v"] += float(np.sum(inv_sig * (svr**2 + svz**2)))
        sums["W_F"] += float(np.sum(inv_sig * (fr**2 + fz**2)))
        sums["eta2"] += float(np.sum(inv_sig * ((svr - fr) ** 2 + (svz - fz) ** 2)))
        cc_cell = np.sum(inv_sig * (((svr + fr) / 2) ** 2 + ((svz + fz) / 2) ** 2), axis=(1, 2))
        sums["cc"] += float(cc_cell.sum())
        t = tile_of[start:start + chunk]
        np.add.at(cc_tile, t[t >= 0], cc_cell[t >= 0])

    eta, c_norm, V2 = np.sqrt(sums["eta2"]), np.sqrt(sums["cc"]), spec.voltage**2
    phi = TWO_PI * float(cert.psi[masks["shaft"] & ~masks["axis"]][0])
    g_lo = max(phi**2 / sums["W_F"], max(c_norm - eta / 2, 0.0) ** 2 / V2)
    g_hi = min(sums["E_v"] / V2, (c_norm + eta / 2) ** 2 / V2)
    tiles = []
    for t, b in enumerate(spec.tiles):
        sig_t = spec.sigma_tissue if b.z_hi <= 0 else spec.sigma_blood
        denom = np.sqrt(sig_t * np.pi * (b.r_hi**2 - b.r_lo**2) * (b.z_hi - b.z_lo))
        cn = np.sqrt(cc_tile[t])
        tiles.append((max(cn - eta / 2, 0.0) / denom, (cn + eta / 2) / denom))
    return {"impedance": (1 / g_hi, 1 / g_lo), "eta": eta, "tiles": tiles}
