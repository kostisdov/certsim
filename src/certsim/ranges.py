"""Trusted bounds that hold for every blood/tissue conductivity ratio in a range.

With the electrode voltage fixed, scaling both conductivities by one factor
leaves the potential unchanged, so field quantities depend only on
rho = sigma_blood / sigma_tissue. Tissue conductivity stays at its specified
value and blood conductivity ranges over sigma_tissue * [rho_1, rho_K].

Admissibility does not involve sigma: v must match the Dirichlet data and
F = curl(psi) must be divergence-free with no flux through insulating walls.
Both conditions are preserved by convex combinations. Given certificates
(v_k, F_k) at rho_1 < ... < rho_K, the pair

    v(s) = (1-s) v_k + s v_{k+1},   F(s) = (1-s) F_k + s F_{k+1},   s in [0, 1]

is admissible for every rho. Prager-Synge then applies at each rho in
[rho_k, rho_{k+1}]; here s is tied to rho by rho = rho_k + s * D.

The radius is written in small quantities only, so that ball evaluation over
an interval of s does not suffer cancellation. With r_k = sigma_k grad v_k - F_k
(certificate k's own residual, sigma at rho_k) the residual at rho is

    sigma(rho) grad v(s) - F(s) = (1-s) r_k + s r_{k+1} + c(s) (grad v_k - grad v_{k+1}),

with c(s) = sigma_t D s (1-s) in blood and 0 in tissue. Hence eta(rho)^2 is a
quadratic form in ((1-s), s, c(s)) of inner products among r_k, r_{k+1} and
grad v_k - grad v_{k+1}, weighted by 1 / sigma(rho) per material. The tile
centres, which lie in tissue, are c_k + s (c_{k+1} - c_k) with
c_k = (sigma_t grad v_k + F_k) / 2. Bounds over rho come from ball evaluation
on sub-intervals of s.
"""

import math
from dataclasses import dataclass

import numpy as np
from flint import arb

from .checker import (
    PI, CertificateError, Interval, Report, _Basis, _Cell, _cells_in_box, _check_boxes, _check_grid,
    _check_values, _curl, _grad, _lin, _nodal, _pos, sqrtpos,
)
from .spec import FocalPFASpec, element_masks, node_masks, tissue_cells


@dataclass
class RangeReport:
    rho: tuple  # (rho_min, rho_max) covered
    eta_max: float  # largest hypercircle radius found over the range [sqrt(W)]
    tile_rms_field: list  # Interval per tile, valid for every rho in range [V/m]
    region_rms_field: list
    tiles: tuple = ()
    spec_voltage: float = 1.0

    # the lesion helpers only need tile intervals, tiles and spec_voltage
    lesion_map = Report.lesion_map
    lesion_extent = Report.lesion_extent


def check_range(spec: FocalPFASpec, certs, rhos, pieces: int = 64) -> RangeReport:
    """Bounds valid for every sigma_blood / sigma_tissue in [rhos[0], rhos[-1]].

    ``certs[k]`` must be a certificate for ``spec`` with sigma_blood =
    rhos[k] * sigma_tissue; all certificates share one grid and degree.
    """
    rhos = [float(x) for x in rhos]
    if len(certs) != len(rhos) or len(rhos) < 2 or any(b <= a for a, b in zip(rhos, rhos[1:])):
        raise CertificateError("need at least two certificates at increasing ratios")
    if spec.voxels:
        raise CertificateError("voxel adjoints are not supported over ranges")
    p = int(certs[0].degree)
    r, z = np.asarray(certs[0].r, float), np.asarray(certs[0].z, float)
    for c in certs:
        if int(c.degree) != p or not (np.array_equal(c.r, r) and np.array_equal(c.z, z)):
            raise CertificateError("certificates must share grid and degree")
    _check_grid(spec, r, z)
    shape = (p * (len(z) - 1) + 1, p * (len(r) - 1) + 1)
    masks = node_masks(spec, r, z, p)
    active, sigma_el = element_masks(spec, r, z)
    for c in certs:
        _check_values(spec, masks, c, shape)
    _check_boxes(spec, r, z, active)
    tissue = tissue_cells(r, z)  # by geometry: spec.sigma_blood plays no part in a range
    for b in spec.regions + spec.tiles:
        if not np.all(tissue[_cells_in_box(b, r, z)]):
            raise CertificateError(f"{b} must lie in tissue for range bounds")
    basis = _Basis(p)
    boxes = spec.regions + spec.tiles
    box_cells = [_cells_in_box(b, r, z) for b in boxes]
    cell_boxes = {}
    for bidx, cells in enumerate(box_cells):
        for j, i in zip(*np.nonzero(cells)):
            cell_boxes.setdefault((j, i), []).append(bidx)

    K = len(certs)
    st = arb(spec.sigma_tissue)
    sig_b = [st * arb(x) for x in rhos]  # blood conductivity at each sample
    # per segment k: Gram of (r_k, r_{k+1}, d) per material, of (c_k, dc) per box
    mat = [{"t": [arb(0)] * 6, "b": [arb(0)] * 6} for _ in range(K - 1)]
    box = [[[arb(0)] * 3 for _ in boxes] for _ in range(K - 1)]
    one, half = arb(1), arb(0.5)

    def gram3(cell, X, Y, Z):
        return [cell.inner(X, X), cell.inner(Y, Y), cell.inner(Z, Z),
                cell.inner(X, Y), cell.inner(X, Z), cell.inner(Y, Z)]

    for j, i in zip(*np.nonzero(active)):
        r1, r2, z1, z2 = arb(r[i]), arb(r[i + 1]), arb(z[j]), arb(z[j + 1])
        cell = _Cell(r1, r2, z1, z2, one, p)  # unweighted inner products
        hr, hz = cell.hr, cell.hz
        s1 = r1 * r1
        hs = r2 * r2 - s1
        g = [_grad(basis.monomials(_nodal(c.v, j, i, p)), p, hr, hz, one) for c in certs]
        F = [_curl(basis.monomials(_nodal(c.psi, j, i, p)), p, r1, hr, hz, s1, hs) for c in certs]
        in_tissue = bool(tissue[j, i])
        sig = [st] * K if in_tissue else sig_b
        res = [_lin((sig[k], g[k]), (-1, F[k])) for k in range(K)]
        in_boxes = cell_boxes.get((j, i), [])
        cen = [_lin((st * half, g[k]), (half, F[k])) for k in range(K)] if in_boxes else None
        m = "t" if in_tissue else "b"
        for k in range(K - 1):
            d = _lin((1, g[k]), (-1, g[k + 1]))
            vals = gram3(cell, res[k], res[k + 1], d)
            mat[k][m] = [a + b for a, b in zip(mat[k][m], vals)]
            if in_boxes:
                dc = _lin((1, cen[k + 1]), (-1, cen[k]))
                cv = [cell.inner(cen[k], cen[k]), cell.inner(cen[k], dc), cell.inner(dc, dc)]
                for bidx in in_boxes:
                    box[k][bidx] = [a + b for a, b in zip(box[k][bidx], cv)]

    def quad(G, a, b, c):
        g11, g22, g33, g12, g13, g23 = G
        return a * a * g11 + b * b * g22 + c * c * g33 + 2 * (a * b * g12 + a * c * g13 + b * c * g23)

    vols = [PI * (arb(b.r_hi) ** 2 - arb(b.r_lo) ** 2) * (arb(b.z_hi) - arb(b.z_lo)) for b in boxes]
    denoms = [(st * v).sqrt() for v in vols]
    lo = [None] * len(boxes)
    hi = [None] * len(boxes)
    eta_max = arb(0)
    for k in range(K - 1):
        D = arb(rhos[k + 1]) - arb(rhos[k])
        for q in range(pieces):
            s = arb(q / pieces).union(arb((q + 1) / pieces))  # ball covering the piece
            a, b = 1 - s, s
            sb = st * (arb(rhos[k]) + D * s)
            cb = st * D * s * (1 - s)
            eta2 = quad(mat[k]["t"], a, b, arb(0)) / st + quad(mat[k]["b"], a, b, cb) / sb
            eta = sqrtpos(eta2)
            if eta.upper() > eta_max:
                eta_max = eta.upper()
            for bidx in range(len(boxes)):
                g_cc, g_cd, g_dd = box[k][bidx]
                cn = sqrtpos((g_cc + 2 * s * g_cd + s * s * g_dd) / st)
                l_ = (_pos(cn - eta / 2) / denoms[bidx]).lower()
                h_ = ((cn + eta / 2) / denoms[bidx]).upper()
                lo[bidx] = l_ if lo[bidx] is None or l_ < lo[bidx] else lo[bidx]
                hi[bidx] = h_ if hi[bidx] is None or h_ > hi[bidx] else hi[bidx]

    ivs = [Interval.from_points(a, b) for a, b in zip(lo, hi)]
    n_reg = len(spec.regions)
    return RangeReport(
        rho=(rhos[0], rhos[-1]),
        eta_max=math.nextafter(float(eta_max), math.inf),
        tile_rms_field=ivs[n_reg:],
        region_rms_field=ivs[:n_reg],
        tiles=spec.tiles,
        spec_voltage=spec.voltage,
    )
