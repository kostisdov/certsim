"""Helpers for displaying certified results. Nothing here is trusted.

The pointwise field comes from the solver's potential and is not certified;
only the tile states and intervals in a checker ``Report`` are.
"""

import numpy as np

from .solver import _lagrange
from .spec import element_masks


def field_at(spec, cert, rq, zq):
    """Solver field E = -grad v at points (rq, zq) [V/m per spec voltage].

    Points inside the catheter, or outside the grid, give NaN.
    """
    p, r, z = cert.degree, cert.r, cert.z
    rq, zq = np.asarray(rq, float), np.asarray(zq, float)
    i = np.clip(np.searchsorted(r, rq, side="right") - 1, 0, len(r) - 2)
    j = np.clip(np.searchsorted(z, zq, side="right") - 1, 0, len(z) - 2)
    hr, hz = r[i + 1] - r[i], z[j + 1] - z[j]
    xi, eta = (rq - r[i]) / hr, (zq - z[j]) / hz
    val, der = _lagrange(p)
    Lx, dLx, Ly, dLy = val(xi), der(xi), val(eta), der(eta)  # (..., p+1)
    k = np.arange(p + 1)
    rows = (p * j)[..., None, None] + k[:, None]
    cols = (p * i)[..., None, None] + k[None, :]
    block = cert.v[rows, cols]  # (..., b, a)
    dv_dxi = np.einsum("...ba,...a,...b->...", block, dLx, Ly)
    dv_deta = np.einsum("...ba,...a,...b->...", block, Lx, dLy)
    Er, Ez = -dv_dxi / hr, -dv_deta / hz
    active, _ = element_masks(spec, r, z)
    outside = (~active[j, i]) | (rq < r[0]) | (rq > r[-1]) | (zq < z[0]) | (zq > z[-1])
    Er = np.where(outside, np.nan, Er)
    Ez = np.where(outside, np.nan, Ez)
    return Er, Ez


def tile_grid(report):
    """Tile states arranged as a (rows, cols) grid, rows from the surface down.

    Returns (r_edges, z_edges, index) where index[k, m] is the tile number of
    row k (depth order) and column m (radius order).
    """
    tiles = report.tiles
    r_lo = sorted({t.r_lo for t in tiles})
    z_hi = sorted({t.z_hi for t in tiles}, reverse=True)
    pos = {(t.z_hi, t.r_lo): n for n, t in enumerate(tiles)}
    index = np.array([[pos[(zh, rl)] for rl in r_lo] for zh in z_hi])
    r_edges = np.array(r_lo + [max(t.r_hi for t in tiles)])
    z_edges = np.array(z_hi + [min(t.z_lo for t in tiles)])
    return r_edges, z_edges, index


def envelope(report, voltage, threshold):
    """Per depth row: inner radius (proven lethal from the axis) and outer
    radius (beyond which every tile is proven spared; NaN when the map edge
    may still be lethal)."""
    state = np.array(report.lesion_map(voltage, threshold))
    r_edges, z_edges, index = tile_grid(report)
    s = state[index]
    inner = np.zeros(len(index))
    outer = np.zeros(len(index))
    for k in range(len(index)):
        lethal_run = np.cumprod(s[k] == 1)
        inner[k] = r_edges[1:][lethal_run.astype(bool)].max(initial=0.0)
        not_spared = np.nonzero(s[k] != -1)[0]
        if len(not_spared) == 0:
            outer[k] = 0.0
        elif not_spared[-1] == s.shape[1] - 1:
            outer[k] = np.nan
        else:
            outer[k] = r_edges[not_spared[-1] + 1]
    return z_edges, inner, outer, s


def staircase(z_edges, radii):
    """Profile polyline (r, z) of a stack of rows with given outer radii,
    closed onto the axis below the last non-empty row."""
    pts = [(0.0, z_edges[0])]
    for k, rad in enumerate(radii):
        if not rad > 0:
            break
        pts += [(rad, z_edges[k]), (rad, z_edges[k + 1])]
    pts.append((0.0, pts[-1][1]))
    return np.array(pts)


def revolve(profile, theta0, theta1, n):
    """Surface of revolution of a (r, z) polyline about the z axis.

    Returns (vertices [N, 3], triangles [M, 3]) with x = r cos t, y = r sin t.
    """
    th = np.linspace(theta0, theta1, n + 1)
    r, z = profile[:, 0], profile[:, 1]
    V = np.stack([r[None, :] * np.cos(th)[:, None],
                  r[None, :] * np.sin(th)[:, None],
                  np.broadcast_to(z, (n + 1, len(z)))], axis=-1).reshape(-1, 3)
    m = len(profile)
    tris = []
    for a in range(n):
        for b in range(m - 1):
            p00, p01 = a * m + b, a * m + b + 1
            p10, p11 = (a + 1) * m + b, (a + 1) * m + b + 1
            tris += [(p00, p10, p11), (p00, p11, p01)]
    return V, np.array(tris, dtype=np.uint32)
