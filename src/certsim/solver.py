"""Untrusted solver for the focal PFA problem.

It produces a certificate made of nodal values of degree-p tensor polynomials
on each cell of a tensor grid (p = 1 is bilinear):

* ``v``: potential, degree p in (r, z) on each cell, equal to the Dirichlet data;
* ``psi``: Stokes stream function, degree p in (r**2, z) on each cell. The field
  F = (-psi_z / r, psi_r / r) is divergence-free by construction and is fitted
  to sigma * grad v;
* for every voxel output, an adjoint potential ``zh`` and stream function
  ``psi_adj`` built the same way.

Nodal values live on the degree-p node grid of shape (p*(nz-1)+1, p*(nr-1)+1):
inside a cell, node (b, a) sits at local coordinates (a/p, b/p), measured in r
for ``v`` and ``zh`` and in r**2 for the stream functions.

Nothing here has to be correct for the checker's bounds to hold. A better
solver only makes the certified intervals narrower.
"""

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .spec import Box, FocalPFASpec, element_masks, node_masks


@dataclass
class Certificate:
    r: np.ndarray
    z: np.ndarray
    v: np.ndarray  # degree-p node grid
    psi: np.ndarray
    zh: np.ndarray  # (n_voxels, *node grid)
    psi_adj: np.ndarray
    degree: int = 1
    claims: dict = field(default_factory=dict)

    def save(self, path):
        np.savez_compressed(
            path, r=self.r, z=self.z, v=self.v, psi=self.psi, zh=self.zh,
            psi_adj=self.psi_adj, degree=self.degree,
            claim_conductance=self.claims.get("conductance", np.nan),
            claim_voxel_mean=self.claims.get("voxel_mean", np.array([])),
        )

    @classmethod
    def load(cls, path):
        d = np.load(path)
        claims = {
            "conductance": float(d["claim_conductance"]),
            "voxel_mean": d["claim_voxel_mean"],
        }
        return cls(d["r"], d["z"], d["v"], d["psi"], d["zh"], d["psi_adj"], int(d["degree"]), claims)


# --- grid ---------------------------------------------------------------------


def graded_grid(breakpoints, singular, hmax_fn):
    """1-D grid through every breakpoint, graded towards singular points.

    ``singular`` is a list of (x, h0, kappa): near x the spacing is
    h0 + kappa * |distance|, capped by hmax_fn. Points are placed by
    equidistributing the integral of 1/h between consecutive breakpoints, so
    breakpoints appear exactly and no sliver cells are created.
    """
    bps = sorted(breakpoints)
    h_min = min(h0 for _, h0, _ in singular)
    pts = [bps[0]]
    for x0, x1 in zip(bps[:-1], bps[1:]):
        n_s = int(min(2_000_000, max(20_001, 4 * (x1 - x0) / h_min)))
        xs = np.linspace(x0, x1, n_s)
        h = hmax_fn(xs)
        for xc, h0, kappa in singular:
            h = np.minimum(h, h0 + kappa * np.abs(xs - xc))
        cum = np.concatenate([[0.0], np.cumsum(0.5 * (1 / h[1:] + 1 / h[:-1]) * np.diff(xs))])
        n = max(1, int(np.ceil(cum[-1])))
        inner = np.interp(np.linspace(0.0, cum[-1], n + 1)[1:-1], cum, xs)
        pts.extend(inner.tolist())
        pts.append(x1)
    return np.array(pts)


def default_grid(spec: FocalPFASpec, h_fine=0.25e-3, h0=1e-9, kappa=0.25, growth=0.08,
                 h0_box=None, kappa_box=0.25):
    """Grid graded towards the electrode edge (r = a), its corners (z = tip, top)
    and, when ``h0_box`` is given, towards the edges of the certified voxels.

    The potential behaves like sqrt(distance) at the electrode-shaft junction,
    so the energy error of the first cell scales with its size; geometric
    grading down to ``h0`` costs only a logarithmic number of extra lines.

    Spacing is ``h_fine`` in the lesion zone and around the electrode, and grows
    by ``growth`` per unit distance outside it.
    """
    zone_r = 8e-3
    zone_z = (-10e-3, (spec.ring_hi if spec.bipolar else spec.top) + 2e-3)

    def hmax_r(x):
        return h_fine + growth * np.maximum(0.0, x - zone_r)

    def hmax_z(x):
        dist = np.maximum(0.0, np.maximum(zone_z[0] - x, x - zone_z[1]))
        return h_fine + growth * dist

    sing_r = [(spec.a, h0, kappa)]
    sing_z = [(spec.tip, h0, kappa), (spec.top, h0, kappa)]
    if spec.bipolar:
        sing_z += [(spec.ring_lo, h0, kappa), (spec.ring_hi, h0, kappa)]
    if h0_box is not None:
        sing_r += [(x, h0_box, kappa_box) for x in {b.r_hi for b in spec.voxels} if x > 0]
        sing_z += [(x, h0_box, kappa_box) for x in {b.z_lo for b in spec.voxels} | {b.z_hi for b in spec.voxels}]
    r = graded_grid(spec.breakpoints_r(), sing_r, hmax_r)
    z = graded_grid(spec.breakpoints_z(), sing_z, hmax_z)
    return r, z


# --- element integrals ----------------------------------------------------------


def _gauss(nq):
    x, w = np.polynomial.legendre.leggauss(nq)
    return (x + 1) / 2, w / 2


def _lagrange(p):
    """Values and derivatives of the 1-D Lagrange basis on nodes k/p."""
    t = np.arange(p + 1) / p
    coef = np.linalg.inv(np.vander(t, p + 1, increasing=True))  # coef[a, k]: x^a in l_k

    def val(x):
        return np.einsum("ak,...a->...k", coef, np.power.outer(x, np.arange(p + 1)))

    def der(x):
        powers = np.power.outer(x, np.arange(p)) * np.arange(1, p + 1)
        return np.einsum("ak,...a->...k", coef[1:], powers)

    return val, der


class _Elements:
    """Quadrature data for all active cells, vectorised over cells."""

    def __init__(self, spec, r, z, degree=1, planar=False, sigma_swap=False):
        p = degree
        nq = p + 2
        active, sigma = element_masks(spec, r, z)
        if sigma_swap:
            sigma = np.where(sigma == spec.sigma_tissue, spec.sigma_blood, spec.sigma_tissue)
        jj, ii = np.nonzero(active)
        r1, r2 = r[ii], r[ii + 1]
        z1, z2 = z[jj], z[jj + 1]
        hr, hz = r2 - r1, z2 - z1
        s1, hs = r1**2, r2**2 - r1**2
        self.sigma = sigma[jj, ii]
        self.r1, self.r2, self.z1, self.z2 = r1, r2, z1, z2

        nr_f = p * (len(r) - 1) + 1
        a_idx, b_idx = np.meshgrid(np.arange(p + 1), np.arange(p + 1), indexing="xy")
        a_idx, b_idx = a_idx.ravel(), b_idx.ravel()  # local node l = a + (p+1) b
        self.nodes = (p * jj[:, None] + b_idx[None]) * nr_f + (p * ii[:, None] + a_idx[None])
        self.nb = (p + 1) ** 2

        x, w = _gauss(nq)
        val, der = _lagrange(p)
        L, dL = val(x), der(x)  # (nq, p+1)
        rq = r1[:, None] + hr[:, None] * x[None, :]  # (ne, nq)
        weight_r = np.ones_like(rq) if planar else rq
        self.mu = 2 * np.pi * weight_r[:, :, None] * (w[:, None] * w[None, :])[None] * (hr * hz)[:, None, None]

        # Q_p basis in (xi, eta): N_l(qa, qb) = L[qa, a] L[qb, b]
        Gxi = dL[:, a_idx].T[:, :, None] * L[:, b_idx].T[:, None, :]  # (nb, q, q)
        Geta = L[:, a_idx].T[:, :, None] * dL[:, b_idx].T[:, None, :]
        self.Gr = Gxi[None] / hr[:, None, None, None]  # (ne, nb, q, q)
        self.Gz = Geta[None] / hz[:, None, None, None]

        # stream basis: degree p in (xs, eta), xs = (r**2 - s1) / hs
        xs = (rq**2 - s1[:, None]) / hs[:, None]  # (ne, nq)
        Ls, dLs = val(xs), der(xs)  # (ne, nq, p+1)
        r_eval = (np.ones_like(rq) if planar else rq)[:, None, :, None]
        Ls_a = np.transpose(Ls[:, :, a_idx], (0, 2, 1))[:, :, :, None]  # (ne, nb, q, 1)
        dLs_a = np.transpose(dLs[:, :, a_idx], (0, 2, 1))[:, :, :, None]
        self.Fr = -Ls_a * dL[:, b_idx].T[None, :, None, :] / hz[:, None, None, None] / r_eval
        self.Fz = 2 * dLs_a * L[:, b_idx].T[None, :, None, :] / hs[:, None, None, None] * (rq[:, None, :, None] / r_eval)

    def local(self, A1, B1, A2, B2, coef):
        """sum_q coef * mu * (A1_a B1_b + A2_a B2_b) for all cells, shape (ne,nb,nb)."""
        m = self.mu * coef[:, None, None]
        return np.einsum("eqp,eaqp,ebqp->eab", m, A1, B1) + np.einsum("eqp,eaqp,ebqp->eab", m, A2, B2)

    def assemble(self, loc, n):
        nb = self.nb
        rows = np.repeat(self.nodes, nb, axis=1).ravel()
        cols = np.tile(self.nodes, (1, nb)).ravel()
        return sp.coo_matrix((loc.ravel(), (rows, cols)), shape=(n, n)).tocsc()

    def load(self, loc, n):
        return np.bincount(self.nodes.ravel(), loc.ravel(), minlength=n)

    def in_box(self, b: Box):
        return (self.r1 >= b.r_lo) & (self.r2 <= b.r_hi) & (self.z1 >= b.z_lo) & (self.z2 <= b.z_hi)


# --- solve ----------------------------------------------------------------------


def _stream_system(spec, el, masks, n):
    """Least-squares fit of a stream function psi to sigma grad v.

    psi is zero on the wall connected to the axis and takes one shared value on
    each other insulating piece (see checker._check_stream). Returns the
    prolongation P from free values to all nodes, the mass-like matrix M, the
    coupling C, and the factorized reduced system.
    """
    zero = masks["axis"] | ((masks["outer"] | masks["shaft"]) if spec.bipolar else False)
    piece_names = ("spacer",) if spec.bipolar else ("shaft",)
    pieces = [np.nonzero(masks[k] & ~zero)[0] for k in piece_names]
    fixed = zero.copy()
    for k in piece_names:
        fixed |= masks[k]
    own = np.nonzero(masks["active"] & ~fixed)[0]
    rows = [own] + pieces
    cols = [np.arange(len(own))] + [np.full(len(pc), len(own) + m) for m, pc in enumerate(pieces)]
    n_dof = len(own) + len(pieces)
    P = sp.coo_matrix(
        (np.ones(sum(len(x) for x in rows)), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n, n_dof),
    ).tocsc()
    M = el.assemble(el.local(el.Fr, el.Fr, el.Fz, el.Fz, 1 / el.sigma), n)
    C = el.assemble(el.local(el.Fr, el.Gr, el.Fz, el.Gz, np.ones_like(el.sigma)), n)
    return P, M, C, spla.splu((P.T @ M @ P).tocsc())


def certify_potential(spec: FocalPFASpec, r, z, v, degree: int, claims: dict | None = None) -> Certificate:
    """Turn a potential computed by any solver into a certificate.

    ``v`` holds nodal values on the degree-``degree`` node grid of the tensor
    grid (r, z), shape (p*(len(z)-1)+1, p*(len(r)-1)+1). The current is fitted
    as the curl of a stream function, so it is exactly divergence-free; this
    adapter is untrusted like any solver, and the checker verifies its output.
    ``claims`` are the other solver's own claims, e.g. its conductance.
    """
    p = degree
    shape = (p * (len(z) - 1) + 1, p * (len(r) - 1) + 1)
    v = np.asarray(v, dtype=float).reshape(shape)
    n = shape[0] * shape[1]
    masks = {k: m.ravel() for k, m in node_masks(spec, r, z, p).items()}
    el = _Elements(spec, r, z, p)
    P, _, C, lu_M = _stream_system(spec, el, masks, n)
    psi = P @ lu_M.solve(P.T @ (C @ v.ravel()))
    empty = np.zeros((0, *shape))
    return Certificate(r, z, v, psi.reshape(shape), empty, empty, p, dict(claims or {}))


def solve(spec: FocalPFASpec, r=None, z=None, degree: int = 3, bug: str | None = None) -> Certificate:
    """Solve the forward and adjoint problems and package them as a certificate.

    ``bug`` injects a deliberate modelling error, used to test that the checker
    catches wrong solvers: ``"planar"`` drops the axisymmetric weight r,
    ``"swap_sigma"`` swaps blood and tissue conductivity.
    """
    if r is None:
        r, z = default_grid(spec)
    p = degree
    shape = (p * (len(z) - 1) + 1, p * (len(r) - 1) + 1)
    n = shape[0] * shape[1]
    masks = {k: m.ravel() for k, m in node_masks(spec, r, z, p).items()}
    el = _Elements(spec, r, z, p, planar=(bug == "planar"), sigma_swap=(bug == "swap_sigma"))

    # primal potential
    K = el.assemble(el.local(el.Gr, el.Gr, el.Gz, el.Gz, el.sigma), n)
    dirichlet = masks["electrode"] | masks["ring"] | masks["ground"]
    free = np.nonzero(masks["active"] & ~dirichlet)[0]
    v = np.zeros(n)
    v[masks["electrode"]] = spec.tip_voltage
    v[masks["ring"]] = spec.ring_voltage
    lu_K = spla.splu(K[free][:, free].tocsc())
    v[free] = lu_K.solve(-(K[free] @ v))

    P, M, C, lu_M = _stream_system(spec, el, masks, n)
    psi = P @ lu_M.solve(P.T @ (C @ v))

    # adjoints: mean of d u / d z over each voxel
    zh_all, psi_adj_all, claim_mean = [], [], []
    for b in spec.voxels:
        vol = np.pi * (b.r_hi**2 - b.r_lo**2) * (b.z_hi - b.z_lo)
        gval = el.in_box(b) / vol
        rhs = el.load(np.einsum("eqp,eaqp->ea", el.mu * gval[:, None, None], el.Gz), n)
        zh = np.zeros(n)
        zh[free] = lu_K.solve(rhs[free])
        gs = el.load(np.einsum("eqp,eaqp->ea", el.mu * (gval / el.sigma)[:, None, None], el.Fz), n)
        psi_adj = P @ lu_M.solve(P.T @ (C @ zh - gs))
        zh_all.append(zh.reshape(shape))
        psi_adj_all.append(psi_adj.reshape(shape))
        claim_mean.append(rhs @ v)

    claims = {"conductance": float(v @ (K @ v)) / spec.voltage**2, "voxel_mean": np.array(claim_mean)}
    adj_shape = (len(spec.voxels), *shape)
    return Certificate(
        r, z, v.reshape(shape), psi.reshape(shape),
        np.array(zh_all).reshape(adj_shape), np.array(psi_adj_all).reshape(adj_shape), p, claims,
    )
