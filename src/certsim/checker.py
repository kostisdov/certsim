"""Trusted checker for focal-PFA certificates.

The checker never runs a solver. It takes the specification and the nodal
arrays of a certificate, verifies that they are admissible, and turns them into
guaranteed bounds with ball arithmetic (python-flint ``arb``). Every float in
the certificate is read as the exact binary number it is, so the only rounding
is inside ``arb``, which encloses it.

Mathematics (sigma-weighted norms, measure 2*pi*r dr dz):

* v is admissible when it equals the Dirichlet data on the electrode and the
  ground. F = curl(psi) is admissible when psi = 0 on the axis and psi is
  constant on the insulating shaft: then div F = 0 and F.n = 0 on every
  insulating boundary.
* Prager-Synge: the true sigma*grad(u) lies on the sphere with diameter
  [sigma*grad(v), F], so with eta = ||sigma*grad(v) - F||_{1/sigma}
  and centre c = (sigma*grad(v) + F)/2,  ||sigma*grad(u) - c|| <= eta/2.
* Conductance G = ||sigma*grad(u)||^2 / V^2 (tip to ring when bipolar) lies in
  [Phi^2 / ||F||^2, ||sigma*grad(v)||^2 / V^2] (Thomson / Dirichlet) and in
  [(||c|| - eta/2)^2, (||c|| + eta/2)^2] / V^2; Phi is the current F carries
  off the driven tip.
* RMS field over a box w in one material: restricting the hypercircle to w,
  ||sigma*grad(u)||_w lies in ||c||_w +- eta/2.
* Voxel mean l(u) = mean of du/dz over the voxel, with an equilibrated adjoint
  K = g + curl(psi_adj) and adjoint potential zh (zero on Dirichlet nodes):
  l(u) = l(v) - <sigma grad v, sigma grad zh> + <K - sigma grad zh, F - sigma grad v>/2
         +- eta_d * eta / 2,   eta_d = ||K - sigma grad zh||.

On each cell the fields are tensor polynomials of degree p. With local
coordinates r = r1 + hr*xi, z = z1 + hz*eta, a field is stored as lists over the
power of eta of polynomials in xi (``arb_poly``): its r-component is
Pr + Qr / r and its z-component is Pz. Every integral is then exact: monomials,
plus one logarithm from Qr / r, obtained by exact polynomial division.
"""

import math
from dataclasses import dataclass

import numpy as np
from flint import arb, arb_poly, ctx, fmpq, fmpq_mat

from .spec import Box, FocalPFASpec, element_masks, node_masks, tissue_cells

ctx.prec = 256
PI = arb.pi()


class CertificateError(Exception):
    """The certificate does not satisfy the specification."""


@dataclass(frozen=True)
class Interval:
    lo: float
    hi: float

    @classmethod
    def from_points(cls, lo: arb, hi: arb) -> "Interval":
        """Floats enclosing the exact end points lo <= hi (rounded outwards)."""
        return cls(math.nextafter(float(lo), -math.inf), math.nextafter(float(hi), math.inf))

    def scale(self, k: float) -> "Interval":
        a, b = sorted((self.lo * k, self.hi * k))
        return Interval(math.nextafter(a, -math.inf), math.nextafter(b, math.inf))

    def supports(self, claim: float, rel_tol: float) -> bool:
        """True when the certified interval proves claim to within +-rel_tol."""
        return claim - rel_tol * abs(claim) <= self.lo and self.hi <= claim + rel_tol * abs(claim)

    @property
    def mid(self):
        return 0.5 * (self.lo + self.hi)

    @property
    def rel_width(self):
        return (self.hi - self.lo) / abs(self.mid) if self.mid else math.inf


@dataclass
class Report:
    conductance: Interval  # [S] at the specified voltage
    eta: Interval  # Prager-Synge estimator [sqrt(W)]
    voxel_mean_dudz: list  # Interval per voxel: mean du/dz [V/m]
    voxel_mean_field: list  # Interval per voxel: mean |E| [V/m]
    voxel_rms_field: list  # Interval per voxel: RMS |E| [V/m]
    region_rms_field: list  # Interval per region: RMS |E| [V/m]
    tile_rms_field: list = ()  # Interval per lesion-map tile: RMS |E| [V/m]

    voxels: tuple = ()  # the specification's voxels, in order
    tiles: tuple = ()
    spec_voltage: float = 1.0

    @property
    def impedance(self) -> Interval:
        """1/conductance, rounded outwards; unbounded above if the conductance may be zero."""
        g = self.conductance
        lo = math.nextafter(1 / g.hi, -math.inf)
        hi = math.nextafter(1 / g.lo, math.inf) if g.lo > 0 else math.inf
        return Interval(lo, hi)

    def voxel_field(self, voltage: float) -> list:
        """|voxel-mean E| per voxel [V/m] at ``voltage`` (the problem is linear)."""
        k = abs(voltage / self.spec_voltage)
        out = []
        for d in self.voxel_mean_dudz:
            if d.lo > 0:
                mag = Interval(d.lo, d.hi)
            elif d.hi < 0:
                mag = Interval(-d.hi, -d.lo)
            else:
                mag = Interval(0.0, max(-d.lo, d.hi))
            out.append(mag.scale(k))
        return out

    def lesion_depth(self, voltage: float, threshold: float):
        """Certified depth of the axis lesion: (reached, not_beyond) in metres.

        A voxel is lethal when |voxel-mean E| >= threshold. ``reached`` is the
        bottom of the deepest voxel such that it and every voxel above it are
        certainly lethal; ``not_beyond`` is the top of the shallowest voxel below
        which every certified voxel is certainly not lethal (None when even the
        deepest voxel may be lethal).
        """
        fields = self.voxel_field(voltage)
        order = sorted(range(len(self.voxels)), key=lambda k: -self.voxels[k].z_hi)
        reached = 0.0
        for k in order:
            if fields[k].lo >= threshold:
                reached = -self.voxels[k].z_lo
            else:
                break
        not_beyond = None
        for k in reversed(order):
            if fields[k].hi < threshold:
                not_beyond = -self.voxels[k].z_hi
            else:
                break
        return reached, not_beyond

    def lesion_map(self, voltage: float, threshold) -> list:
        """Per tile: +1 certainly lethal, -1 certainly not, 0 undecided.

        A tile is lethal when its RMS field is at least the threshold. A
        threshold given as (low, high) means "lethal for every threshold in
        the range" (+1) and "spared for every threshold in the range" (-1).
        """
        t_lo, t_hi = (threshold, threshold) if np.isscalar(threshold) else threshold
        k = abs(voltage / self.spec_voltage)
        out = []
        for iv in self.tile_rms_field:
            s = iv.scale(k)
            out.append(1 if s.lo >= t_hi else (-1 if s.hi < t_lo else 0))
        return out

    def lesion_extent(self, voltage: float, threshold) -> dict:
        """Certified lesion depth and largest radius, in metres.

        For each quantity, ``inner`` is reached by certainly lethal tiles
        (contiguous from the axis, or from the surface on the axis), and
        ``outer`` is the extent beyond which every tile, in every column or
        row, is certainly not lethal. ``inner`` depth is measured on the axis,
        which bounds the deepest lethal tile from below. ``outer`` is None when
        the map's edge may still be lethal.
        """
        state = self.lesion_map(voltage, threshold)
        rows = {}
        for b, s in zip(self.tiles, state):
            rows.setdefault(b.z_hi, []).append((b, s))
        for row in rows.values():
            row.sort(key=lambda t: t[0].r_lo)

        def run(seq, edge_of):
            """(inner, outer) extent along an ordered sequence of (box, state)."""
            inner = 0.0
            for b, s in seq:
                if s != 1:
                    break
                inner = edge_of(b, True)
            outer = 0.0
            for b, s in reversed(seq):
                if s != -1:
                    outer = edge_of(b, True)
                    break
            if seq and seq[-1][1] != -1:
                outer = None
            return inner, outer

        cols = {}
        for b, s in zip(self.tiles, state):
            cols.setdefault(b.r_lo, []).append((b, s))
        depths = {r: run(sorted(col, key=lambda t: -t[0].z_hi), lambda b, _: -b.z_lo)
                  for r, col in cols.items()}
        d_outer = None if any(d[1] is None for d in depths.values()) else max(d[1] for d in depths.values())
        depth = (depths[min(cols)][0], d_outer)
        radii = [run(row, lambda b, _: b.r_hi) for row in rows.values()]
        r_inner = max((w[0] for w in radii), default=0.0)
        r_outer = None if any(w[1] is None for w in radii) else max((w[1] for w in radii), default=0.0)
        return {"depth": depth, "radius": (r_inner, r_outer)}


# --- verification of admissibility ---------------------------------------------


def _check_grid(spec, r, z):
    for name, g, lo, hi, bps in (("r", r, 0.0, spec.R, spec.breakpoints_r()),
                                 ("z", z, -spec.H, spec.Ztop, spec.breakpoints_z())):
        if g.ndim != 1 or not np.all(np.isfinite(g)) or not np.all(np.diff(g) > 0):
            raise CertificateError(f"{name} grid is not strictly increasing")
        if g[0] != lo or g[-1] != hi:
            raise CertificateError(f"{name} grid does not span the domain")
        missing = set(bps) - set(g.tolist())
        if missing:
            raise CertificateError(f"{name} grid misses breakpoints {sorted(missing)}")


def _require(cond, msg):
    if not np.all(cond):
        raise CertificateError(msg)


def _check_values(spec, masks, cert, shape):
    act = masks["active"]
    for name, arr in (("v", cert.v), ("psi", cert.psi)):
        _require(arr.shape == shape, f"{name} has wrong shape")
        _require(np.isfinite(arr[act]), f"{name} has non-finite values")
    _require(cert.v[masks["electrode"]] == spec.tip_voltage, "v differs from the electrode voltage")
    _require(cert.v[masks["ring"]] == spec.ring_voltage, "v differs from the ring electrode voltage")
    _require(cert.v[masks["ground"]] == 0.0, "v is not zero on the ground")
    _check_stream(masks, cert.psi, "psi", spec.bipolar)
    nvox = len(spec.voxels)
    _require(cert.zh.shape == (nvox, *shape) and cert.psi_adj.shape == (nvox, *shape),
             "adjoint arrays have wrong shape")
    for k in range(nvox):
        _require(np.isfinite(cert.zh[k][act]) & np.isfinite(cert.psi_adj[k][act]), "adjoint non-finite")
        _require(cert.zh[k][masks["electrode"] | masks["ring"] | masks["ground"]] == 0.0,
                 f"adjoint {k} is not zero on Dirichlet nodes")
        _check_stream(masks, cert.psi_adj[k], f"psi_adj[{k}]", spec.bipolar)


def _check_stream(masks, psi, name, floating):
    """F = curl(psi) has no flux through insulating walls: psi is constant along
    each connected insulating wall. The axis is one with psi = 0; for a bipolar
    (floating) catheter it connects through the insulating far boundary to the
    shaft, so all of that is 0 too and only the spacer keeps its own constant."""
    zero = masks["axis"] | (masks["outer"] | masks["shaft"] if floating else False)
    _require(psi[zero] == 0.0, f"{name} is not zero on the axis (current leaks through it)")
    for piece in ("spacer",) if floating else ("shaft",):
        vals = psi[masks[piece]]
        _require(vals.size == 0 or np.all(vals == vals[0]),
                 f"{name} is not constant on the insulating {piece}")


def _check_boxes(spec, r, z, active):
    """Every box lies in active cells of one material, decided by geometry, not by
    conductivity values (which may coincide)."""
    tissue = tissue_cells(r, z)
    for b in spec.boxes:
        cells = _cells_in_box(b, r, z)
        if not np.all(active[cells]) or len(set(tissue[cells].tolist())) != 1:
            raise CertificateError(f"{b} is not inside a single material")


def _cells_in_box(b: Box, r, z):
    ri = (r[:-1] >= b.r_lo) & (r[1:] <= b.r_hi)
    zj = (z[:-1] >= b.z_lo) & (z[1:] <= b.z_hi)
    return zj[:, None] & ri[None, :]


# --- element algebra ------------------------------------------------------------


class _Basis:
    """Exact change of basis from nodal values on k/p to monomial coefficients."""

    def __init__(self, p):
        V = fmpq_mat(p + 1, p + 1, [fmpq(k, p) ** a for k in range(p + 1) for a in range(p + 1)])
        Vi = V.inv()  # Vi[a, k]: coefficient of t**a in the Lagrange basis l_k
        self.p = p
        self.Vi = [[arb(Vi[a, k]) for k in range(p + 1)] for a in range(p + 1)]

    def monomials(self, f):
        """f[b][a]: nodal value at (xi, eta) = (a/p, b/p) -> C[alpha][beta]."""
        p, Vi = self.p, self.Vi
        tmp = [[sum((Vi[al][a] * f[b][a] for a in range(p + 1)), arb(0)) for b in range(p + 1)]
               for al in range(p + 1)]  # tmp[alpha][b]
        return [[sum((Vi[be][b] * tmp[al][b] for b in range(p + 1)), arb(0)) for be in range(p + 1)]
                for al in range(p + 1)]


def _poly(cs):
    return arb_poly(cs) if any(not c.is_zero() for c in cs) else None


def _scale(P, k):
    return None if P is None else P * k


def _add(P, Q):
    if P is None:
        return Q
    if Q is None:
        return P
    return P + Q


def _compose(cs, T):
    """sum_k cs[k] * T**k by Horner, for a list of arb coefficients."""
    res = arb_poly([cs[-1]])
    for c in reversed(cs[:-1]):
        res = res * T + c
    return res


def _grad(C, p, hr, hz, sigma):
    """sigma * grad of the degree-p interpolant with monomial coefficients C."""
    Pr, Pz = [None] * (p + 1), [None] * (p + 1)
    for be in range(p + 1):
        P = _poly([C[al][be] for al in range(p + 1)])
        if P is not None and P.degree() > 0:
            Pr[be] = P.derivative() * (sigma / hr)
        if be < p:
            Pz[be] = _scale(_poly([C[al][be + 1] * (be + 1) for al in range(p + 1)]), sigma / hz)
    return Pr, [None] * (p + 1), Pz


def _curl(D, p, r1, hr, hz, s1, hs):
    """F = (-psi_z / r, psi_r / r) for psi with monomial coefficients D in (xs, eta)."""
    on_axis = r1.is_zero()
    if on_axis:
        # xs = (r/hr)**2 = xi**2 exactly; psi vanishes on the axis, so psi_z / r
        # is a polynomial: D[0][*] must be exactly zero.
        for be in range(p + 1):
            if not D[0][be].is_zero():
                raise CertificateError("1/r field on an axis cell")

        def comp(cs):
            out = [arb(0)] * (2 * len(cs) - 1)
            for k, c in enumerate(cs):
                out[2 * k] = c
            return out
    else:
        T = arb_poly([arb(0), 2 * r1 * hr / hs, hr * hr / hs])

    Pr, Qr, Pz = [None] * (p + 1), [None] * (p + 1), [None] * (p + 1)
    for be in range(p + 1):
        dxs = [D[al][be] * al for al in range(1, p + 1)]
        if any(not c.is_zero() for c in dxs):
            Pz[be] = (_poly(comp(dxs)) if on_axis else _compose(dxs, T)) * (2 / hs)
        if be < p:
            dz = [D[al][be + 1] * (be + 1) for al in range(p + 1)]
            if any(not c.is_zero() for c in dz):
                if on_axis:
                    # -(psi_z) / r with r = hr*xi: drop the (zero) constant, shift by one
                    Pr[be] = _scale(_poly(comp(dz)[1:]), -1 / (hz * hr))
                else:
                    Qr[be] = _compose(dz, T) * (-1 / hz)
    return Pr, Qr, Pz


def _lin(*terms):
    """Linear combination sum(k * field) of fields."""
    out = None
    for k, f in terms:
        g = tuple([_scale(P, k) for P in comp] for comp in f)
        out = g if out is None else tuple([_add(a, b) for a, b in zip(oc, gc)] for oc, gc in zip(out, g))
    return out


def _int01(P):
    """Integral of a polynomial in xi over [0, 1]."""
    if P is None:
        return arb(0)
    return sum((c / (k + 1) for k, c in enumerate(P.coeffs())), arb(0))


class _Cell:
    def __init__(self, r1, r2, z1, z2, sigma, p):
        self.r1, self.hr, self.hz = r1, r2 - r1, z2 - z1
        self.p = p
        self.rpoly = arb_poly([r1, self.hr])
        self.scale = 2 * PI * self.hr * self.hz / sigma
        self.log_r = None if r1.is_zero() else (r2 / r1).log()

    def inner(self, X, Y):
        """<X, Y> with weight 1/sigma over the cell (measure 2 pi r dr dz)."""
        p = self.p
        A = B = Cq = None
        for b1 in range(p + 1):
            for b2 in range(p + 1):
                w = arb(1) / (b1 + b2 + 1)
                for P1, P2 in ((X[0][b1], Y[0][b2]), (X[2][b1], Y[2][b2])):
                    if P1 is not None and P2 is not None:
                        A = _add(A, P1 * P2 * w)
                for P1, P2 in ((X[0][b1], Y[1][b2]), (X[1][b1], Y[0][b2])):
                    if P1 is not None and P2 is not None:
                        B = _add(B, P1 * P2 * w)
                if X[1][b1] is not None and Y[1][b2] is not None:
                    Cq = _add(Cq, X[1][b1] * Y[1][b2] * w)
        total = _int01(A * self.rpoly if A is not None else None) + _int01(B)
        if Cq is not None:
            total += self._int_over_r(Cq)
        return self.scale * total

    def _int_over_r(self, N):
        """Integral over [0, 1] of N(xi) / (r1 + hr*xi)."""
        if self.log_r is None:
            raise CertificateError("1/r field on an axis cell")
        eps = self.hr / self.r1
        if eps.upper() >= 0.5:
            # thick cell (near the axis): exact division is well conditioned
            q, rem = divmod(N, self.rpoly)
            rc = rem.coeffs()
            return _int01(q) + (rc[0] if rc else arb(0)) * self.log_r / self.hr
        # thin cell: moments m_j = int xi^j / (1 + eps xi), which satisfy
        # m_j = 1/(j+1) - eps m_{j+1}; run it downwards (errors shrink by eps)
        # from an enclosure 1/((1+eps)(J+1)) <= m_J <= 1/(J+1).
        cs = N.coeffs()
        steps = int(ctx.prec / -math.log2(float(eps.upper()))) + 2
        J = len(cs) + steps
        m = (1 / ((1 + eps) * (J + 1))).union(arb(1) / (J + 1))
        moments = {}
        for j in range(J - 1, -1, -1):
            m = arb(1) / (j + 1) - eps * m
            if j < len(cs):
                moments[j] = m
        return sum((c * moments[j] for j, c in enumerate(cs)), arb(0)) / self.r1


def _nodal(arr, j, i, p):
    """Nodal values of cell (j, i) as f[b][a] (eta index b, xi index a)."""
    block = arr[p * j : p * j + p + 1, p * i : p * i + p + 1]
    return [[arb(float(x)) for x in row] for row in block]


# --- main entry point -------------------------------------------------------------


def check(spec: FocalPFASpec, cert) -> Report:
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
    basis = _Basis(p)

    boxes = spec.boxes
    box_cells = [_cells_in_box(b, r, z) for b in boxes]
    nvox = len(spec.voxels)
    vols = [PI * (arb(b.r_hi) ** 2 - arb(b.r_lo) ** 2) * (arb(b.z_hi) - arb(b.z_lo)) for b in boxes]
    gvals = [1 / vols[k] for k in range(nvox)]
    # boxes containing each cell, to avoid scanning all boxes per cell
    cell_boxes = {}
    for bidx, cells in enumerate(box_cells):
        for j, i in zip(*np.nonzero(cells)):
            cell_boxes.setdefault((j, i), []).append(bidx)

    E_v = W_F = eta2 = cc = arb(0)
    cc_box = [arb(0) for _ in boxes]
    ell_v = [arb(0) for _ in range(nvox)]
    R_k = [arb(0) for _ in range(nvox)]
    X_k = [arb(0) for _ in range(nvox)]
    etad2 = [arb(0) for _ in range(nvox)]
    half = arb(0.5)

    for j, i in zip(*np.nonzero(active)):
        r1, r2, z1, z2 = arb(r[i]), arb(r[i + 1]), arb(z[j]), arb(z[j + 1])
        sig = arb(sigma_el[j, i])
        cell = _Cell(r1, r2, z1, z2, sig, p)
        hr, hz = cell.hr, cell.hz
        s1 = r1 * r1
        hs = r2 * r2 - s1

        sv = _grad(basis.monomials(_nodal(cert.v, j, i, p)), p, hr, hz, sig)
        F = _curl(basis.monomials(_nodal(cert.psi, j, i, p)), p, r1, hr, hz, s1, hs)
        dp = _lin((1, sv), (-1, F))  # sigma grad v - F
        c = _lin((half, sv), (half, F))
        E_v += cell.inner(sv, sv)
        W_F += cell.inner(F, F)
        eta2 += cell.inner(dp, dp)
        q_cc = cell.inner(c, c)
        cc += q_cc
        in_boxes = cell_boxes.get((j, i), [])
        for bidx in in_boxes:
            cc_box[bidx] += q_cc

        for k in range(nvox):
            szk = _grad(basis.monomials(_nodal(cert.zh[k], j, i, p)), p, hr, hz, sig)
            Kk = _curl(basis.monomials(_nodal(cert.psi_adj[k], j, i, p)), p, r1, hr, hz, s1, hs)
            if k in in_boxes:
                g = ([None] * (p + 1), [None] * (p + 1), [arb_poly([gvals[k]])] + [None] * p)
                Kk = _lin((1, Kk), (1, g))
                ell_v[k] += cell.inner(g, sv)
            dd = _lin((1, Kk), (-1, szk))
            R_k[k] += cell.inner(sv, szk)
            X_k[k] += cell.inner(dd, dp)  # <K - s grad zh, sigma grad v - F>
            etad2[k] += cell.inner(dd, dd)

    eta = sqrtpos(eta2)
    c_norm = sqrtpos(cc)
    V2 = arb(spec.voltage) ** 2
    # Conductance G = ||sigma grad u||^2 / V^2 (bipolar: tip-to-ring). Phi is the
    # current F carries off the driven tip, 2*pi*(psi on the next insulating
    # piece - psi on the axis). Since <F, sigma grad u> = V * Phi in both
    # configurations, Cauchy-Schwarz gives Thomson's G >= Phi^2 / ||F||^2.
    piece = masks["spacer"] if spec.bipolar else masks["shaft"]
    Phi = 2 * PI * arb(float(cert.psi[piece][0]))
    G_lo = _pmax(lower(Phi * Phi / W_F), lower(_pos(c_norm - eta / 2) ** 2 / V2))
    G_hi = _pmin(upper(E_v / V2), upper((c_norm + eta / 2) ** 2 / V2))

    def rms(bidx):
        b_sigma = sigma_el[box_cells[bidx]][0]
        cn = sqrtpos(cc_box[bidx])
        denom = (arb(b_sigma) * vols[bidx]).sqrt()
        return lower(_pos(cn - eta / 2) / denom), upper((cn + eta / 2) / denom)

    voxel_dudz, voxel_mean, voxel_rms = [], [], []
    for k in range(nvox):
        centre = ell_v[k] - R_k[k] - X_k[k] / 2
        rad = sqrtpos(etad2[k]) * eta / 2
        lo, hi = lower(centre - rad), upper(centre + rad)
        voxel_dudz.append(Interval.from_points(lo, hi))
        rlo, rhi = rms(k)
        voxel_rms.append(Interval.from_points(rlo, rhi))
        # |mean E| = |mean du/dz| (radial parts cancel over a full ring) <= mean |E| <= RMS
        if lo > 0:
            mean_lo = lo
        elif hi < 0:
            mean_lo = -hi
        else:
            mean_lo = arb(0)
        voxel_mean.append(Interval.from_points(mean_lo, rhi))
    n_reg = len(spec.regions)
    region_rms = [Interval.from_points(*rms(nvox + m)) for m in range(n_reg)]
    tile_rms = [Interval.from_points(*rms(nvox + n_reg + m)) for m in range(len(spec.tiles))]

    return Report(
        conductance=Interval.from_points(G_lo, G_hi),
        eta=Interval.from_points(lower(eta), upper(eta)),
        voxel_mean_dudz=voxel_dudz,
        voxel_mean_field=voxel_mean,
        voxel_rms_field=voxel_rms,
        region_rms_field=region_rms,
        tile_rms_field=tile_rms,
        voxels=spec.voxels,
        tiles=spec.tiles,
        spec_voltage=spec.voltage,
    )


# --- exact end points of balls ----------------------------------------------------
#
# Bounds are combined on exact end points, never on ball midpoints, so a
# comparison between two end points is always decided exactly.


def lower(x: arb) -> arb:
    return x.lower()


def upper(x: arb) -> arb:
    return x.upper()


def _pos(x: arb) -> arb:
    """A ball enclosing max(x, 0)."""
    if x.upper() <= 0:
        return arb(0)
    if x.lower() >= 0:
        return x
    return arb(0).union(x.upper())


def sqrtpos(x: arb) -> arb:
    """A ball enclosing sqrt(max(t, 0)) for every t in x.

    arb's sqrt is NaN as soon as a ball touches a negative number, which a
    ball enclosing a tiny non-negative quantity easily does.
    """
    if x.upper() <= 0:
        return arb(0)
    if x.lower() >= 0:
        return x.sqrt()
    return arb(0).union(x.upper().sqrt())


def _pmax(a: arb, b: arb) -> arb:
    return a if a > b else b


def _pmin(a: arb, b: arb) -> arb:
    return a if a < b else b
