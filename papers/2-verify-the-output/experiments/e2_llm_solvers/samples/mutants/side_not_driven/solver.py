import sys
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def main():
    d = np.load(sys.argv[1])
    r = np.asarray(d['r'], dtype=float)
    z = np.asarray(d['z'], dtype=float)
    a = float(d['a']); L = float(d['L']); gap = float(d['gap'])
    st = float(d['sigma_tissue']); sb = float(d['sigma_blood'])
    V = float(d['voltage'])
    Nr, Nz = len(r), len(z)
    N = Nr * Nz

    ia = int(np.argmin(np.abs(r - a)))
    jg = int(np.argmin(np.abs(z - gap)))
    jt = int(np.argmin(np.abs(z - (gap + L))))

    hr = np.diff(r); hz = np.diff(z)

    # Cells: (j, i) -> [z_j, z_{j+1}] x [r_i, r_{i+1}]
    CI, CJ = np.meshgrid(np.arange(Nr - 1), np.arange(Nz - 1))
    active = ~((CI < ia) & (CJ >= jg))      # exclude electrode + shaft cells
    ci = CI[active].ravel(); cj = CJ[active].ravel()
    ne = ci.size
    zc = 0.5 * (z[cj] + z[cj + 1])
    sig = np.where(zc < 0.0, st, sb)
    h1 = hr[ci]; h2 = hz[cj]; r0 = r[ci]

    # Local node order: (0,0),(1,0),(1,1),(0,1) in (xi, eta)
    xi_n = np.array([0.0, 1.0, 1.0, 0.0])
    eta_n = np.array([0.0, 0.0, 1.0, 1.0])
    g = 0.5 / np.sqrt(3.0)
    pts = [0.5 - g, 0.5 + g]
    Ke = np.zeros((ne, 4, 4))
    inv_h1sq = (1.0 / h1 ** 2)[:, None, None]
    inv_h2sq = (1.0 / h2 ** 2)[:, None, None]
    for xi in pts:
        for eta in pts:
            dxi = (2 * xi_n - 1) * np.where(eta_n == 1, eta, 1 - eta)
            deta = (2 * eta_n - 1) * np.where(xi_n == 1, xi, 1 - xi)
            rq = r0 + xi * h1
            w = (0.25 * h1 * h2 * rq * sig)[:, None, None]
            Ke += w * (np.outer(dxi, dxi)[None] * inv_h1sq +
                       np.outer(deta, deta)[None] * inv_h2sq)

    nodes = np.stack([cj * Nr + ci,
                      cj * Nr + ci + 1,
                      (cj + 1) * Nr + ci + 1,
                      (cj + 1) * Nr + ci], axis=1)
    rows = np.broadcast_to(nodes[:, :, None], (ne, 4, 4)).ravel()
    cols = np.broadcast_to(nodes[:, None, :], (ne, 4, 4)).ravel()
    K = sp.coo_matrix((Ke.ravel(), (rows, cols)), shape=(N, N)).tocsr()

    # Node classification
    I, J = np.meshgrid(np.arange(Nr), np.arange(Nz))
    I = I.ravel(); J = J.ravel()
    elec = (I <= ia) & (J == jg)
    ground = (I == Nr - 1) | (J == 0) | ((J == Nz - 1) & (I >= ia))
    elec = elec & ~ground
    act_nodes = np.zeros(N, dtype=bool)
    act_nodes[nodes.ravel()] = True
    dirich = elec | ground
    free = act_nodes & ~dirich

    u = np.zeros(N)
    u[elec] = V
    fidx = np.nonzero(free)[0]
    Kf = K[fidx]
    rhs = -(Kf @ u)
    Kff = Kf[:, fidx].tocsc()
    u[fidx] = spla.spsolve(Kff, rhs)
    u[elec] = V
    u[ground] = 0.0

    # Current leaving electrode: reaction at electrode nodes
    Rres = K @ u
    current = 2.0 * np.pi * float(Rres[elec].sum())
    Z = V / current if current != 0.0 else float('inf')

    v = u.reshape(Nz, Nr)
    np.savez(sys.argv[2], v=v, impedance=float(Z))


if __name__ == '__main__':
    main()
