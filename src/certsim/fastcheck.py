"""Vectorized checker in double-precision ball arithmetic.

Computes the same bounds as ``checker.check``, from the same exact cell
integrals, but on whole arrays of cells at once. Every quantity is a ball
(midpoint, radius) of float64 arrays; each operation adds a bound on its own
rounding to the radius, so every ball contains the exact value. Admissibility is
checked by the same exact tests as in ``checker``. The moments of 1/r over each
grid column are computed once per column in arb, with the same recurrences as
``checker``; the final square roots and divisions are done in arb.

Rigour rests on these assumptions, which hold for numpy on IEEE hardware:
  A1 binary64 arithmetic with round-to-nearest, so a single +, -, x or / of
     floats has relative error at most U = 2**-53 (Higham, Thm 2.2);
  A2 no flush-to-zero or denormals-are-zero, so underflow loses at most 2**-1074
     absolutely (covered by TINY);
  A3 sums and matrix products are formed by some order of pairwise additions of
     correctly rounded products, possibly with fused multiply-add, but not by
     Strassen-type algorithms; then a sum of n terms has error at most
     gamma_n * sum |terms| (Higham, eq. 3.5), whatever the order;
  A4 the specification's numbers (conductivities, voltage) are the floats as
     stored, e.g. fl(0.7), and grid coordinates are exact floats.
Under A1-A4 every midpoint error is bounded as described in each operation, and
every radius, itself computed in round-to-nearest, is inflated by 1 + gamma_k
with k at least the number of roundings in its computation (Rump 2012), so the
computed radius bounds the exact one.

Not supported: voxel (adjoint) outputs. Use ``checker.check`` for those.
"""

import math

import numpy as np
from flint import arb, ctx, fmpq, fmpq_mat

from .checker import (CertificateError, Interval, Report, _check_boxes, _check_grid, _check_values,
                      _cells_in_box, _pmax, _pmin, _pos, lower, sqrtpos, upper)
from .spec import FocalPFASpec, element_masks, node_masks

U = 2.0**-53  # unit roundoff of float64
TINY = 2.0**-1074  # covers underflow of a single operation
PI = arb.pi()


def _gamma(n):
    return (n * U) / (1 - n * U)


RAD = 1 + _gamma(16)  # inflation for a radius formed by at most a dozen roundings


class Ball:
    """Arrays of balls mid +- rad; operations round the radius outwards."""

    __slots__ = ("m", "r")

    def __init__(self, m, r=None):
        self.m = np.asarray(m, dtype=float)
        self.r = np.zeros_like(self.m) if r is None else np.asarray(r, dtype=float)

    @staticmethod
    def exact(x):
        """A float that is exact (a grid coordinate, a node value)."""
        return Ball(x)

    @staticmethod
    def of(x):
        """A float approximating an exact real to within its own rounding."""
        x = np.asarray(x, dtype=float)
        return Ball(x, 2 * U * np.abs(x) + TINY)

    def _fin(self, m, r):
        # midpoint error <= U |exact| <= 2U |m|; the radius r took at most five roundings
        return Ball(m, (r + 2 * U * np.abs(m)) * RAD + TINY)

    def __add__(self, o):
        return self._fin(self.m + o.m, self.r + o.r)

    def __sub__(self, o):
        return self._fin(self.m - o.m, self.r + o.r)

    def __mul__(self, o):
        return self._fin(self.m * o.m, np.abs(self.m) * o.r + np.abs(o.m) * self.r + self.r * o.r)

    def __neg__(self):
        return Ball(-self.m, self.r)

    def __getitem__(self, k):
        return Ball(self.m[k], self.r[k])

    def sum(self, axis=None):
        n = self.m.size if axis is None else self.m.shape[axis]
        m = np.sum(self.m, axis=axis)
        r = np.sum(self.r, axis=axis) + _gamma(n + 2) * np.sum(np.abs(self.m), axis=axis)
        return Ball(m, r * (1 + _gamma(2 * n + 8)) + TINY)

    def to_arb(self):
        return arb(float(self.m), float(self.r * RAD + TINY))


def _recip(b):
    """1/b for balls b bounded away from zero."""
    m = 1.0 / b.m
    lo = (np.abs(b.m) - b.r) * (1 - _gamma(4))  # rounded down: a lower bound of |b|
    if np.any(lo <= 0):
        raise CertificateError("division by a quantity that may be zero")
    r = b.r / (lo * np.abs(b.m)) + 2 * U * np.abs(m)
    return Ball(m, r * RAD + TINY)


def _polymul(a, b):
    """Product of ball polynomials along the last axis."""
    n = a.m.shape[-1] + b.m.shape[-1] - 1
    shape = np.broadcast_shapes(a.m.shape[:-1], b.m.shape[:-1]) + (n,)
    out = Ball(np.zeros(shape))
    for i in range(a.m.shape[-1]):
        term = a[..., i:i + 1] * b
        pad = [(0, 0)] * (len(shape) - 1) + [(i, n - i - b.m.shape[-1])]
        out = out + Ball(np.pad(term.m, pad), np.pad(term.r, pad))
    return out


def _lagrange_to_monomial(p):
    V = fmpq_mat(p + 1, p + 1, [fmpq(k, p) ** a for k in range(p + 1) for a in range(p + 1)])
    Vi = V.inv()
    return Ball.of([[float(Vi[a, k].p) / float(Vi[a, k].q) for k in range(p + 1)] for a in range(p + 1)])


def _coeffs(nodal, Vi):
    """Monomial coefficients C[e, alpha(xi), beta(eta)] from nodal values f[e, b(eta), a(xi)].

    Vi entries are rationals; the products and sums are bounded as in Ball.
    """
    p1 = Vi.m.shape[0]
    tmp_m = np.einsum("ka,eba->ekb", Vi.m, nodal.m)
    tmp_r = (np.einsum("ka,eba->ekb", np.abs(Vi.m), nodal.r) + np.einsum("ka,eba->ekb", Vi.r, np.abs(nodal.m))
             + np.einsum("ka,eba->ekb", Vi.r, nodal.r) + _gamma(p1 + 2) * np.einsum("ka,eba->ekb", np.abs(Vi.m), np.abs(nodal.m)))
    tmp = Ball(tmp_m, tmp_r * (1 + _gamma(4 * p1 + 16)) + TINY)
    out_m = np.einsum("lb,ekb->ekl", Vi.m, tmp.m)
    out_r = (np.einsum("lb,ekb->ekl", np.abs(Vi.m), tmp.r) + np.einsum("lb,ekb->ekl", Vi.r, np.abs(tmp.m))
             + np.einsum("lb,ekb->ekl", Vi.r, tmp.r) + _gamma(p1 + 2) * np.einsum("lb,ekb->ekl", np.abs(Vi.m), np.abs(tmp.m)))
    return Ball(out_m, out_r * (1 + _gamma(4 * p1 + 16)) + TINY)


def _column_moments(r1, r2, J):
    """m_j = int_0^1 xi^j / (r1 + hr xi) d xi for j < J, per column, as balls (arb, as in checker)."""
    m_mid, m_rad = np.zeros((len(r1), J)), np.zeros((len(r1), J))
    for c, (a, b) in enumerate(zip(r1, r2)):
        if a == 0.0:
            continue  # axis column: F_r is a polynomial there and no 1/r moment is used
        a, b = arb(float(a)), arb(float(b))
        hr = b - a
        eps = hr / a
        if eps.upper() >= 0.5:
            m = [(b / a).log() / hr]
            for j in range(1, J):
                m.append((arb(1) / j - a * m[-1]) / hr)
        else:
            steps = int(ctx.prec / -math.log2(float(eps.upper()))) + 2
            top = J + steps
            mm = (1 / ((1 + eps) * (top + 1))).union(arb(1) / (top + 1))
            got = {}
            for j in range(top - 1, -1, -1):
                mm = arb(1) / (j + 1) - eps * mm
                if j < J:
                    got[j] = mm
            m = [got[j] / a for j in range(J)]
        for j, x in enumerate(m):
            m_mid[c, j] = float(x.mid())
            m_rad[c, j] = (float(x.rad()) + 2 * U * abs(float(x.mid()))) * RAD + TINY
    return Ball(m_mid, m_rad)


def _compose(dcoef, T1, T2, deg):
    """sum_k dcoef[..., k] * T(xi)**k with T = T1 xi + T2 xi**2, by Horner, padded to deg+1."""
    T = Ball(np.stack([np.zeros_like(T1.m), T1.m, T2.m], axis=-1), np.stack([np.zeros_like(T1.r), T1.r, T2.r], axis=-1))
    res = dcoef[..., -1:]
    for k in range(dcoef.m.shape[-1] - 2, -1, -1):
        res = _polymul(res, T[:, None, :] if res.m.ndim == 3 else T)
        res = res + _pad(dcoef[..., k:k + 1], res.m.shape[-1])
    return _pad(res, deg + 1)


def _pad(b, n):
    k = b.m.shape[-1]
    if k >= n:
        return b[..., :n]
    pad = [(0, 0)] * (b.m.ndim - 1) + [(0, n - k)]
    return Ball(np.pad(b.m, pad), np.pad(b.r, pad))


def _subnormals_preserved() -> bool:
    """Assumption A2 at run time: a library built with -ffast-math can set flush-to-zero
    (subnormal outputs) or denormals-are-zero (subnormal inputs) for the whole process.
    Doubling the smallest subnormal in numpy, at call time, detects either mode."""
    return bool((np.array([2.0**-1074]) * 2.0)[0] != 0.0)


def fast_check(spec: FocalPFASpec, cert, chunk: int = 20000) -> Report:
    """Same bounds as checker.check (without voxel outputs), vectorized in ball arithmetic."""
    if spec.voxels:
        raise ValueError("fast_check does not support voxel outputs; use checker.check")
    if not _subnormals_preserved():
        raise RuntimeError("subnormal numbers are flushed to zero in this process (assumption A2)")
    p = int(getattr(cert, "degree", 1))
    if p < 1:
        raise CertificateError("degree must be at least 1")
    r, z = np.asarray(cert.r, dtype=float), np.asarray(cert.z, dtype=float)
    _check_grid(spec, r, z)
    shape = (p * (len(z) - 1) + 1, p * (len(r) - 1) + 1)
    masks = node_masks(spec, r, z, p)
    active, sigma_el = element_masks(spec, r, z)
    _check_values(spec, masks, cert, shape)
    _check_boxes(spec, r, z, active)

    Vi = _lagrange_to_monomial(p)
    J = 4 * p + 4  # moments up to xi^(4p+2): N = P r + Q has degree 2p + 1
    mom = _column_moments(r[:-1], r[1:], J)  # (ncols, J)
    kk = np.arange(J)
    inv_k1 = Ball.of(1.0 / (kk + 1.0))  # int xi^k
    inv_k2 = Ball.of(1.0 / (kk + 2.0))  # int xi^(k+1)
    bb = np.arange(p + 1)
    w_eta = Ball.of(1.0 / (bb[:, None] + bb[None, :] + 1.0))  # int eta^(b1+b2)

    boxes = list(spec.regions) + list(spec.tiles)
    # regions are disjoint among themselves, and so are tiles, but a cell may lie in one of each
    index_maps = []
    for offset, group in ((0, spec.regions), (len(spec.regions), spec.tiles)):
        m = np.full(active.shape, -1)
        for k, b in enumerate(group):
            cells = _cells_in_box(b, r, z)
            if np.any(m[cells] >= 0):
                raise ValueError("fast_check needs disjoint regions and disjoint tiles")
            m[cells] = offset + k
        index_maps.append(m)
    totals = {"E_v": [], "W_F": [], "eta2": [], "cc": []}
    cc_box_m, cc_box_r = np.zeros(len(boxes)), np.zeros(len(boxes))

    jj, ii = np.nonzero(active)
    g_all = _gamma(len(jj) + 2)
    for start in range(0, len(jj), chunk):
        j, i = jj[start:start + chunk], ii[start:start + chunk]
        sums, ccb = _chunk(j, i, r, z, p, sigma_el, cert, Vi, mom, inv_k1, inv_k2, w_eta, J)
        for k in totals:
            totals[k].append(sums[k])
        for m in index_maps:
            bidx = m[j, i]
            sel = bidx >= 0
            np.add.at(cc_box_m, bidx[sel], ccb.m[sel])
            np.add.at(cc_box_r, bidx[sel], ccb.r[sel] + g_all * np.abs(ccb.m[sel]))

    def total(name):
        parts = totals[name]
        return Ball(np.array([b.m for b in parts]), np.array([b.r for b in parts])).sum().to_arb()

    E_v, W_F, eta2, cc = (total(k) for k in ("E_v", "W_F", "eta2", "cc"))
    eta, c_norm = sqrtpos(eta2), sqrtpos(cc)
    V2 = arb(spec.voltage) ** 2
    piece = masks["spacer"] if spec.bipolar else masks["shaft"]
    Phi = 2 * PI * arb(float(cert.psi[piece][0]))
    G_lo = _pmax(lower(Phi * Phi / W_F), lower(_pos(c_norm - eta / 2) ** 2 / V2))
    G_hi = _pmin(upper(E_v / V2), upper((c_norm + eta / 2) ** 2 / V2))

    def rms(k):
        b = boxes[k]
        b_sigma = sigma_el[_cells_in_box(b, r, z)][0]
        vol = PI * (arb(b.r_hi) ** 2 - arb(b.r_lo) ** 2) * (arb(b.z_hi) - arb(b.z_lo))
        # cc_box_r is a sum over up to all cells, formed in round-to-nearest
        cn = sqrtpos(arb(float(cc_box_m[k]), float(cc_box_r[k] * (1 + _gamma(2 * len(jj) + 8)) + TINY)))
        denom = (arb(b_sigma) * vol).sqrt()
        return Interval.from_points(lower(_pos(cn - eta / 2) / denom), upper((cn + eta / 2) / denom))

    n_reg = len(spec.regions)
    return Report(
        conductance=Interval.from_points(G_lo, G_hi),
        eta=Interval.from_points(lower(eta), upper(eta)),
        voxel_mean_dudz=[], voxel_mean_field=[], voxel_rms_field=[],
        region_rms_field=[rms(k) for k in range(n_reg)],
        tile_rms_field=[rms(n_reg + k) for k in range(len(spec.tiles))],
        voxels=(), tiles=spec.tiles, spec_voltage=spec.voltage,
    )


def _chunk(j, i, r, z, p, sigma_el, cert, Vi, mom, inv_k1, inv_k2, w_eta, J):
    """Integrals over a chunk of cells; returns per-chunk sums and per-cell cc."""
    ne = len(j)
    r1, r2, z1, z2 = Ball.exact(r[i]), Ball.exact(r[i + 1]), Ball.exact(z[j]), Ball.exact(z[j + 1])
    hr, hz = r2 - r1, z2 - z1
    rsum = r1 + r2
    hs = hr * rsum  # r2^2 - r1^2 without cancellation on thin cells
    sig = Ball.exact(sigma_el[j, i])
    inv_hr, inv_hz, inv_hs, inv_sig = _recip(hr), _recip(hz), _recip(hs), _recip(sig)
    on_axis = r[i] == 0.0

    # nodal values f[e, b(eta), a(xi)] of v and psi, exact floats
    bi = p * j[:, None, None] + np.arange(p + 1)[None, :, None]
    ai = p * i[:, None, None] + np.arange(p + 1)[None, None, :]
    psi_nodal = cert.psi[bi, ai]
    # On an axis cell the xs**0 coefficient of psi is a combination of its nodal values on the
    # axis (the Lagrange basis at node 0 is the only one that is nonzero at xs = 0). If those are
    # exactly zero, so is the coefficient, and psi_z / r is a polynomial; otherwise reject, as
    # checker.check does.
    if np.any(psi_nodal[on_axis][:, :, 0] != 0.0):
        raise CertificateError("1/r field on an axis cell")
    # Only derivatives of v and psi enter the bounds, so subtract each cell's corner value first:
    # the coefficients then carry rounding relative to the variation across the cell, not to the
    # value itself, which matters on thin cells. On axis cells the corner value of psi is 0.
    v_nodal = cert.v[bi, ai]
    Cv = _coeffs(Ball.exact(v_nodal) - Ball.exact(v_nodal[:, :1, :1]), Vi)  # Cv[e, alpha, beta]
    Dp = _coeffs(Ball.exact(psi_nodal) - Ball.exact(psi_nodal[:, :1, :1]), Vi)  # Dp[e, alpha(xs), beta]
    first = on_axis[:, None, None] & (np.arange(p + 1)[None, :, None] == 0)
    Dp = Ball(np.where(first, 0.0, Dp.m), np.where(first, 0.0, Dp.r))

    deg = 2 * p  # degree in xi of every field component
    al = Ball.exact(np.arange(p + 1, dtype=float))
    # sigma grad v: Pr[e, beta, xi-coeffs], Pz[e, beta, ...]
    Cv_t = Ball(np.transpose(Cv.m, (0, 2, 1)), np.transpose(Cv.r, (0, 2, 1)))  # [e, beta, alpha]
    dxi = Ball(Cv_t.m[..., 1:], Cv_t.r[..., 1:]) * al[1:]
    sv_r = _pad(dxi * (sig * inv_hr)[:, None, None], deg + 1)
    deta = Ball(np.zeros_like(Cv_t.m), np.zeros_like(Cv_t.r))
    deta.m[:, :-1], deta.r[:, :-1] = (Cv_t[:, 1:, :] * al[1:, None]).m, (Cv_t[:, 1:, :] * al[1:, None]).r
    sv_z = _pad(deta * (sig * inv_hz)[:, None, None], deg + 1)

    # current F from psi in (xs, eta): F_z = 2 psi_xs / hs (polynomial), F_r = -psi_eta / (hz r)
    Dt = Ball(np.transpose(Dp.m, (0, 2, 1)), np.transpose(Dp.r, (0, 2, 1)))  # [e, beta, alpha(xs)]
    dxs = Ball(Dt.m[..., 1:], Dt.r[..., 1:]) * al[1:]  # [e, beta, p]
    dz = Ball(np.zeros_like(Dt.m), np.zeros_like(Dt.r))
    dz.m[:, :-1], dz.r[:, :-1] = (Dt[:, 1:, :] * al[1:, None]).m, (Dt[:, 1:, :] * al[1:, None]).r
    # xs = T1 xi + T2 xi^2 with T1 = 2 r1 hr / hs = 2 r1 / (r1 + r2) and
    # T2 = hr^2 / hs = hr / (r1 + r2), free of cancellation (exactly 0 and 1 on the axis)
    inv_rsum = _recip(rsum)
    T1 = (Ball.of(2.0) * r1) * inv_rsum
    T2 = hr * inv_rsum
    T1 = Ball(np.where(on_axis, 0.0, T1.m), np.where(on_axis, 0.0, T1.r))
    T2 = Ball(np.where(on_axis, 1.0, T2.m), np.where(on_axis, 0.0, T2.r))
    Fz = _compose(dxs, T1, T2, deg) * (Ball.of(2.0) * inv_hs)[:, None, None]
    Fz = _pad(Fz, deg + 1)
    Qnum = _compose(dz, T1, T2, deg) * (-inv_hz)[:, None, None]  # F_r = Qnum / r
    Qnum = _pad(Qnum, deg + 1)
    # on the axis r = hr xi, and with the xs**0 coefficient exactly zero and xs = xi**2, Qnum has no
    # constant term (its computed radius there is only the underflow guard): F_r = (Qnum / xi) / hr
    shifted = _pad(Ball(Qnum.m[..., 1:], Qnum.r[..., 1:]), deg + 1) * inv_hr[:, None, None]
    Fr_poly = Ball(np.where(on_axis[:, None, None], shifted.m, 0.0), np.where(on_axis[:, None, None], shifted.r, 0.0))
    Fr_q = Ball(np.where(on_axis[:, None, None], 0.0, Qnum.m), np.where(on_axis[:, None, None], 0.0, Qnum.r))

    half = Ball.exact(0.5)
    zero_q = Ball(np.zeros_like(Fr_q.m))
    fields = {
        "sv": (sv_r, zero_q, sv_z),
        "F": (Fr_poly, Fr_q, Fz),
        "dp": (sv_r - Fr_poly, -Fr_q, sv_z - Fz),
        "c": ((sv_r + Fr_poly) * half, Fr_q * half, (sv_z + Fz) * half),
    }
    scale = (Ball.of(2.0) * Ball.of(math.pi) * hr * hz) * inv_sig  # 2 pi hr hz / sigma
    cols = mom[i]  # (e, J): int xi^k / (r1 + hr xi)
    rq = (r1[:, None] * inv_k1[None, :J] + hr[:, None] * inv_k2[None, :J])  # int xi^k (r1 + hr xi)
    rpoly = Ball(np.stack([r1.m, hr.m], axis=-1), np.stack([r1.r, hr.r], axis=-1))[:, None, :]
    out = {}
    for name, (P, Q, Z) in fields.items():
        # r-component P + Q / r: off the axis integrate N^2 / r with N = P r + Q, so that the
        # cancellation between P and Q happens once, in N's coefficients; on the axis Q = 0.
        N = _polymul(P, rpoly) + _pad(Q, P.m.shape[-1] + 1)
        I_off = _quad(N, cols, w_eta)
        I_axis = _quad(P, rq, w_eta)
        I_r = Ball(np.where(on_axis, I_axis.m, I_off.m), np.where(on_axis, I_axis.r, I_off.r))
        out[name] = (I_r + _quad(Z, rq, w_eta)) * scale
    sums = {"E_v": out["sv"].sum(), "W_F": out["F"].sum(), "eta2": out["dp"].sum(), "cc": out["c"].sum()}
    return sums, out["c"]


def _quad(X, M, w):
    """sum over b1, b2, k1, k2 of w[b1, b2] X[b1, k1] X[b2, k2] M[k1 + k2], per cell, as balls.

    X: (e, B, D) polynomial coefficients in xi for each eta power; M: (e, K) moments with
    K >= 2D - 1; w: (B, B) eta weights. The radius bounds the input radii and, through
    gamma, the rounding of every product and of the summation.
    """
    D = X.m.shape[-1]
    k = np.arange(D)
    Mk = M[:, k[:, None] + k[None, :]]  # (e, D, D)

    def form(A, W, Mx, B):
        """sum W[b1, b2] A[b1, k1] Mx[k1, k2] B[b2, k2], as two batched matrix products."""
        T = np.matmul(A, Mx)  # (e, b1, k2)
        V = np.matmul(W, B)  # (e, b1, k2)
        return np.sum(T * V, axis=(1, 2))

    aX, aw, aM = np.abs(X.m), np.abs(w.m), np.abs(Mk.m)
    big = aX + X.r
    n = (X.m.shape[1] * D) ** 2
    mid = form(X.m, w.m, Mk.m, X.m)
    rad = (2 * form(aX, aw, aM, X.r) + form(X.r, aw, aM, X.r)
           + form(big, aw, Mk.r, big) + form(big, w.r, aM + Mk.r, big)
           + _gamma(n + 6) * form(aX, aw, aM, aX))
    return Ball(mid, rad * (1 + _gamma(2 * n + 32)) + TINY)
