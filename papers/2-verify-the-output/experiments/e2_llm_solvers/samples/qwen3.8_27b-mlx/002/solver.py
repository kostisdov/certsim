#!/usr/bin/env python3
"""
solver.py - Solve for the potential around an ablation electrode.

Usage:
    python solver.py problem.npz solution.npz

Solves the axisymmetric Laplace equation with piecewise-constant conductivity
using bilinear finite elements on a non-uniform grid with the axisymmetric
weight r in the integrals.
"""

import sys
import numpy as np
from scipy.sparse import diags, csc_matrix, lil_matrix
from scipy.sparse.linalg import spsolve
import os


def main():
    if len(sys.argv) != 3:
        print("Usage: python solver.py problem.npz solution.npz")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    # Load problem data
    data = np.load(input_file)
    r = data['r']
    z = data['z']
    a = float(data['a'])
    L = float(data['L'])
    R = float(data['R'])
    H = float(data['H'])
    Ztop = float(data['Ztop'])
    gap = float(data['gap'])
    sigma_tissue = float(data['sigma_tissue'])
    sigma_blood = float(data['sigma_blood'])
    voltage = float(data['voltage'])

    nr = len(r)
    nz = len(z)

    # Grid spacing
    dr = np.diff(r)  # length nr-1
    dz = np.diff(z)  # length nz-1

    # Node index helper
    def node(i, j):
        return j * nr + i

    # Determine cell conductivity
    # Cell (i, j) spans r in [r[i], r[i+1]], z in [z[j], z[j+1]]
    # The cell is "in blood" if z[j] >= 0 (cell entirely above z=0)
    # "in tissue" if z[j+1] <= 0 (cell entirely below z=0)
    # Since grid includes z=0, cells are entirely in one region
    n_cells_r = nr - 1
    n_cells_z = nz - 1

    cell_sigma = np.zeros((n_cells_z, n_cells_r))
    for j in range(n_cells_z):
        if z[j] >= 0.0:
            cell_sigma[j, :] = sigma_blood
        elif z[j + 1] <= 0.0:
            cell_sigma[j, :] = sigma_tissue
        else:
            # This shouldn't happen if grid includes z=0
            # But just in case, use blood if midpoint is above 0
            if (z[j] + z[j + 1]) / 2.0 >= 0.0:
                cell_sigma[j, :] = sigma_blood
            else:
                cell_sigma[j, :] = sigma_tissue

    # Determine which nodes are on the electrode surface
    # Electrode: 0 <= r <= a, gap <= z <= gap + L
    # Bottom face: z = gap, 0 <= r <= a
    # Side: r = a, gap <= z <= gap + L
    # Top face: z = gap + L, 0 <= r <= a (but this is also shaft bottom)
    
    # Actually, the electrode surface is the boundary of the conducting region
    # that is in contact with the electrode. The electrode occupies
    # 0 <= r <= a, gap <= z <= gap + L.
    # The surface of the electrode is:
    # - Bottom: z = gap, r in [0, a]
    # - Side: r = a, z in [gap, gap + L]
    # - Top: z = gap + L, r in [0, a] -- but this is also the shaft bottom
    #   The shaft is insulating, so no current crosses. But the top of the
    #   electrode is a metal surface, so it's at voltage.
    
    # Find indices for r = a and z = gap, z = gap + L
    # Since grid includes these breakpoints, find exact indices
    idx_r_a = None
    for i in range(nr):
        if np.isclose(r[i], a, rtol=0, atol=1e-15):
            idx_r_a = i
            break
    if idx_r_a is None:
        # Find closest
        idx_r_a = np.argmin(np.abs(r - a))

    idx_z_gap = None
    for j in range(nz):
        if np.isclose(z[j], gap, rtol=0, atol=1e-15):
            idx_z_gap = j
            break
    if idx_z_gap is None:
        idx_z_gap = np.argmin(np.abs(z - gap))

    idx_z_gapL = None
    for j in range(nz):
        if np.isclose(z[j], gap + L, rtol=0, atol=1e-15):
            idx_z_gapL = j
            break
    if idx_z_gapL is None:
        idx_z_gapL = np.argmin(np.abs(z - (gap + L)))

    idx_z_0 = None
    for j in range(nz):
        if np.isclose(z[j], 0.0, rtol=0, atol=1e-15):
            idx_z_0 = j
            break
    if idx_z_0 is None:
        idx_z_0 = np.argmin(np.abs(z))

    # Electrode surface nodes
    # Bottom face: z = gap, r in [0, a]
    electrode_nodes = set()
    for i in range(nr):
        if r[i] <= a + 1e-15:
            electrode_nodes.add(node(i, idx_z_gap))

    # Side face: r = a, z in [gap, gap + L]
    for j in range(nz):
        if gap - 1e-15 <= z[j] <= gap + L + 1e-15:
            electrode_nodes.add(node(idx_r_a, j))

    # Top face: z = gap + L, r in [0, a]
    for i in range(nr):
        if r[i] <= a + 1e-15:
            electrode_nodes.add(node(i, idx_z_gapL))

    # Grounded nodes
    # Outer boundary r = R
    grounded_nodes = set()
    for j in range(nz):
        grounded_nodes.add(node(nr - 1, j))

    # Bottom z = -H
    for i in range(nr):
        grounded_nodes.add(node(i, 0))

    # Top z = Ztop for r >= a
    for i in range(nr):
        if r[i] >= a - 1e-15:
            grounded_nodes.add(node(i, nz - 1))

    # Nodes strictly inside the electrode
    inside_electrode = set()
    for i in range(nr):
        if r[i] < a - 1e-15:  # strictly inside
            for j in range(nz):
                if gap + 1e-15 < z[j] < gap + L - 1e-15:
                    inside_electrode.add(node(i, j))

    # Nodes strictly inside the shaft
    inside_shaft = set()
    for i in range(nr):
        if r[i] < a - 1e-15:
            for j in range(nz):
                if gap + L + 1e-15 < z[j] < Ztop - 1e-15:
                    inside_shaft.add(node(i, j))

    # Total number of nodes
    n_nodes = nr * nz

    # Build the stiffness matrix using bilinear finite elements
    # For each cell, we have 4 nodes:
    # (i, j), (i+1, j), (i, j+1), (i+1, j+1)
    # with local indices 0, 1, 2, 3

    # Bilinear shape functions on reference element [0,1]x[0,1]:
    # N0 = (1-x)(1-y), N1 = x(1-y), N2 = (1-x)y, N3 = xy
    # Gradients:
    # dN0/dx = -(1-y), dN0/dy = -(1-x)
    # dN1/dx = (1-y),  dN1/dy = -x
    # dN2/dx = -y,     dN2/dy = (1-x)
    # dN3/dx = y,      dN3/dy = x

    # For axisymmetric problem with weight r, the integral over a cell is:
    # integral sigma * (grad u . grad v) * r dr dz
    # = integral over reference element of sigma * sum_{alpha,beta}
    #     (dN_alpha/dxi * dN_beta/dxi * J_rr + dN_alpha/deta * dN_beta/deta * J_zz +
    #      dN_alpha/dxi * dN_beta/deta * J_rz + dN_alpha/deta * dN_beta/dxi * J_zr) * r * J dxi deta
    # where J is the Jacobian of the mapping.

    # For a rectangular cell [r_i, r_{i+1}] x [z_j, z_{j+1}]:
    # x = r_i + (r_{i+1}-r_i)*xi = r_i + dr_i * xi
    # y = z_j + (z_{j+1}-z_j)*eta = z_j + dz_j * eta
    # r = x = r_i + dr_i * xi
    # J = dr_i * dz_j
    # dr/dxi = dr_i, dr/deta = 0
    # dz/dxi = 0, dz/deta = dz_j
    # The metric: dsigma = dr^2 + dz^2, so
    # integral sigma * ( (du/dr)^2 + (du/dz)^2 ) * r dr dz
    # = integral sigma * ( (du/dxi * dr_i + du/deta * dz_j)^2 + ... ) 
    # Actually, let's use the standard formula.
    # grad u in physical space:
    # du/dr = (1/dr_i) * du/dxi
    # du/dz = (1/dz_j) * du/deta
    # The integral becomes:
    # integral_0^1 integral_0^1 sigma * [ (du/dxi/dr_i)^2 + (du/deta/dz_j)^2 ] * (r_i + dr_i*xi) * dr_i * dz_j dxi deta

    # This can be written as a matrix K_cell * sigma_cell
    # where K_cell depends on dr_i, dz_j, r_i.

    # Let's compute K_cell analytically.
    # K_cell[alpha, beta] = integral_0^1 integral_0^1
    #     [ dN_alpha/dxi * dN_beta/dxi / dr_i^2 + dN_alpha/deta * dN_beta/deta / dz_j^2 ]
    #     * (r_i + dr_i * xi) * dr_i * dz_j dxi deta

    # The shape function gradients on the reference element:
    # dN/dxi for N0..N3: [-(1-eta), (1-eta), -eta, eta]
    # dN/deta for N0..N3: [-(1-xi), -xi, (1-xi), xi]

    # Let's compute the integrals.
    # I1[alpha, beta] = integral_0^1 integral_0^1 (dN_alpha/dxi * dN_beta/dxi) * (r_i + dr_i*xi) * dr_i * dz_j dxi deta
    # I2[alpha, beta] = integral_0^1 integral_0^1 (dN_alpha/deta * dN_beta/deta) * (r_i + dr_i*xi) * dr_i * dz_j dxi deta

    # For I1:
    # dN/dxi: N0: -(1-eta) = eta-1, N1: 1-eta, N2: -eta, N3: eta
    # Product dN_alpha/dxi * dN_beta/dxi integrated over eta:
    # int_0^1 (eta-1)^2 deta = 1/3, int_0^1 (eta-1)*(1-eta) deta = -1/3, etc.
    # Let's just compute numerically or use known results.

    # Actually, let's use a cleaner approach. The bilinear FE on a rectangle
    # with axisymmetric weight can be computed by integrating exactly.

    # For the x-derivative part (du/dr):
    # dN/dxi values at the 4 nodes:
    # N0: eta - 1
    # N1: 1 - eta
    # N2: -eta
    # N3: eta
    # So dN/dxi = [eta-1, 1-eta, -eta, eta]
    # The product matrix B_x[alpha,beta] = dN_alpha/dxi * dN_beta/dxi
    # int_0^1 int_0^1 B_x * (r_i + dr_i*xi) * dr_i * dz_j dxi deta
    # = dz_j * dr_i * int_0^1 [ int_0^1 B_x * (r_i + dr_i*xi) dxi ] deta

    # Let's compute this step by step.
    # int_0^1 (eta-1) * (r_i + dr_i*xi) deta = (r_i + dr_i*xi) * int_0^1 (eta-1) deta = 0
    # int_0^1 (eta-1)^2 deta = 1/3
    # int_0^1 (1-eta)^2 deta = 1/3
    # int_0^1 (-eta)^2 deta = 1/3
    # int_0^1 eta^2 deta = 1/3
    # int_0^1 (eta-1)(1-eta) deta = -1/3
    # int_0^1 (eta-1)(-eta) deta = int_0^1 (eta^2 - eta) deta = 1/3 - 1/2 = -1/6
    # int_0^1 (eta-1)*eta deta = int_0^1 (eta - eta^2) deta = 1/2 - 1/3 = 1/6
    # int_0^1 (1-eta)(-eta) deta = -1/6
    # int_0^1 (1-eta)*eta deta = 1/6
    # int_0^1 (-eta)*eta deta = -1/3

    # So the matrix after integrating over eta (for the x-part):
    # B_x_eta[alpha, beta] = int_0^1 dN_alpha/dxi * dN_beta/dxi deta
    # = [[1/3, -1/3, -1/6, 1/6],
    #    [-1/3, 1/3, 1/6, -1/6],
    #    [-1/6, 1/6, 1/3, -1/3],
    #    [1/6, -1/6, -1/3, 1/3]]
    # Wait, let me recompute.
    # alpha=0, beta=0: int (eta-1)^2 = 1/3
    # alpha=0, beta=1: int (eta-1)(1-eta) = -1/3
    # alpha=0, beta=2: int (eta-1)(-eta) = int (-eta^2+eta) = -1/3+1/2 = 1/6
    # alpha=0, beta=3: int (eta-1)*eta = int (eta-eta^2) = 1/2-1/3 = 1/6
    # alpha=1, beta=0: -1/3
    # alpha=1, beta=1: 1/3
    # alpha=1, beta=2: int (1-eta)(-eta) = int (-eta+eta^2) = -1/2+1/3 = -1/6
    # alpha=1, beta=3: int (1-eta)*eta = 1/6
    # alpha=2, beta=0: 1/6
    # alpha=2, beta=1: -1/6
    # alpha=2, beta=2: 1/3
    # alpha=2, beta=3: int (-eta)*eta = -1/3
    # alpha=3, beta=0: 1/6
    # alpha=3, beta=1: 1/6
    # alpha=3, beta=2: -1/3
    # alpha=3, beta=3: 1/3

    # Now multiply by (r_i + dr_i*xi) and integrate over xi:
    # int_0^1 (r_i + dr_i*xi) dxi = r_i + dr_i/2
    # int_0^1 xi * (r_i + dr_i*xi) dxi = r_i/2 + dr_i/3
    # But the integrand also has the B_x_eta matrix which doesn't depend on xi,
    # so the xi integral just gives a factor.
    # Actually, the full integral is:
    # dz_j * int_0^1 int_0^1 B_x_eta[alpha,beta] * (r_i + dr_i*xi) dxi deta
    # But B_x_eta already integrated over eta, so:
    # = dz_j * B_x_eta[alpha,beta] * int_0^1 (r_i + dr_i*xi) dxi
    # = dz_j * B_x_eta[alpha,beta] * (r_i + dr_i/2)

    # Wait, that's not right. The integrand is:
    # (dN_alpha/dxi * dN_beta/dxi) * (r_i + dr_i * xi) * dr_i * dz_j
    # The dN/dxi depends on eta, and (r_i + dr_i*xi) depends on xi.
    # So we can separate:
    # int_0^1 int_0^1 (dN_alpha/dxi * dN_beta/dxi) * (r_i + dr_i*xi) * dr_i * dz_j dxi deta
    # = dr_i * dz_j * int_0^1 (dN_alpha/dxi * dN_beta/dxi) deta * int_0^1 (r_i + dr_i*xi) dxi
    # = dr_i * dz_j * B_x_eta[alpha,beta] * (r_i + dr_i/2)

    # Hmm, but that's only if dN/dxi doesn't depend on xi, which it doesn't.
    # But wait, dN/dxi for bilinear elements:
    # N0 = (1-xi)(1-eta), dN0/dxi = -(1-eta) -- doesn't depend on xi. Correct.
    # N1 = xi(1-eta), dN1/dxi = (1-eta) -- doesn't depend on xi.
    # N2 = (1-xi)*eta, dN2/dxi = -eta -- doesn't depend on xi.
    # N3 = xi*eta, dN3/dxi = eta -- doesn't depend on xi.
    # So yes, the xi and eta integrals separate.

    # Similarly for the y-part (du/dz):
    # dN/deta: N0: -(1-xi), N1: -xi, N2: (1-xi), N3: xi
    # B_y_xi[alpha, beta] = int_0^1 dN_alpha/deta * dN_beta/deta dxi
    # = [[1/3, -1/6, 1/3, -1/6],
    #    [-1/6, 1/3, -1/6, 1/3],
    #    [1/3, -1/6, 1/3, -1/6],
    #    [-1/6, 1/3, -1/6, 1/3]]
    # Wait, let me recompute.
    # alpha=0, beta=0: int (1-xi)^2 = 1/3
    # alpha=0, beta=1: int (1-xi)*xi = 1/6
    # alpha=0, beta=2: int (1-xi)^2 = 1/3
    # alpha=0, beta=3: int (1-xi)*xi = 1/6
    # alpha=1, beta=0: 1/6
    # alpha=1, beta=1: int xi^2 = 1/3
    # alpha=1, beta=2: int xi*(1-xi) = 1/6
    # alpha=1, beta=3: int xi^2 = 1/3
    # alpha=2, beta=0: 1/3
    # alpha=2, beta=1: 1/6
    # alpha=2, beta=2: 1/3
    # alpha=2, beta=3: 1/6
    # alpha=3, beta=0: 1/6
    # alpha=3, beta=1: 1/3
    # alpha=3, beta=2: 1/6
    # alpha=3, beta=3: 1/3

    # So B_y_xi = [[1/3, 1/6, 1/3, 1/6],
    #              [1/6, 1/3, 1/6, 1/3],
    #              [1/3, 1/6, 1/3, 1/6],
    #              [1/6, 1/3, 1/6, 1/3]]
    # And the eta integral gives int_0^1 (r_i + dr_i*eta) deta = r_i + dr_i/2
    # Wait, no. For the y-part, the integrand is:
    # (dN_alpha/deta * dN_beta/deta) * (r_i + dr_i*xi) * dr_i * dz_j
    # dN/deta depends on xi, and (r_i + dr_i*xi) also depends on xi.
    # So we can't separate as easily.
    # = dr_i * dz_j * int_0^1 [ (dN_alpha/deta * dN_beta/deta) * (r_i + dr_i*xi) ] dxi * int_0^1 deta
    # But dN/deta for bilinear:
    # N0: -(1-xi), N1: -xi, N2: (1-xi), N3: xi
    # These depend on xi but not on eta. So the eta integral gives 1.
    # = dr_i * dz_j * int_0^1 (dN_alpha/deta * dN_beta/deta) * (r_i + dr_i*xi) dxi
    # = dr_i * dz_j * [ r_i * int_0^1 (dN_alpha/deta * dN_beta/deta) dxi + dr_i * int_0^1 xi * (dN_alpha/deta * dN_beta/deta) dxi ]

    # Let's define:
    # A[alpha, beta] = int_0^1 (dN_alpha/deta * dN_beta/deta) dxi
    # B[alpha, beta] = int_0^1 xi * (dN_alpha/deta * dN_beta/deta) dxi
    # Then the y-part contribution is dr_i * dz_j * (r_i * A + dr_i * B)

    # dN/deta: N0: xi-1, N1: -xi, N2: xi-1, N3: xi
    # A[0,0] = int (xi-1)^2 = 1/3
    # A[0,1] = int (xi-1)*(-xi) = int (xi^2 - xi) = 1/3 - 1/2 = -1/6
    # A[0,2] = int (xi-1)^2 = 1/3
    # A[0,3] = int (xi-1)*xi = int (xi - xi^2) = 1/2 - 1/3 = 1/6
    # A[1,0] = -1/6
    # A[1,1] = int xi^2 = 1/3
    # A[1,2] = int (-xi)*(xi-1) = int (xi - xi^2) = 1/6
    # A[1,3] = int (-xi)*xi = -1/3
    # A[2,0] = 1/3
    # A[2,1] = 1/6
    # A[2,2] = 1/3
    # A[2,3] = 1/6
    # A[3,0] = 1/6
    # A[3,1] = -1/3
    # A[3,2] = 1/6
    # A[3,3] = 1/3

    # B[alpha, beta] = int xi * (dN_alpha/deta * dN_beta/deta) dxi
    # B[0,0] = int xi*(xi-1)^2 = int (xi^3 - 2xi^2 + xi) = 1/4 - 2/3 + 1/2 = 3/12 - 8/12 + 6/12 = 1/12
    # B[0,1] = int xi*(xi-1)*(-xi) = int (-xi^2*(xi-1)) = int (xi^3 - xi^2) = 1/4 - 1/3 = -1/12
    # B[0,2] = int xi*(xi-1)^2 = 1/12
    # B[0,3] = int xi*(xi-1)*xi = int (xi^2 - xi^3) = 1/3 - 1/4 = 1/12
    # B[1,0] = -1/12
    # B[1,1] = int xi*xi^2 = 1/4
    # B[1,2] = int xi*(-xi)*(xi-1) = int (xi^2*(xi-1)) = int (xi^3 - xi^2) = 1/4 - 1/3 = -1/12
    # B[1,3] = int xi*(-xi)*xi = -1/4
    # B[2,0] = 1/12
    # B[2,1] = -1/12
    # B[2,2] = 1/12
    # B[2,3] = 1/12
    # B[3,0] = 1/12
    # B[3,1] = -1/4
    # B[3,2] = 1/12
    # B[3,3] = 1/4

    # This is getting complicated. Let me use a different approach.
    # I'll use numerical integration (Gauss quadrature) on each cell.

    # Actually, let me just use the standard finite element assembly with
    # exact integration for the bilinear elements.

    # The stiffness matrix for a single cell with conductivity sigma is:
    # K_cell = sigma * integral (grad N_alpha . grad N_beta) * r dr dz
    # For the axisymmetric problem, grad in cylindrical coordinates is:
    # grad u = (du/dr) e_r + (du/dz) e_z
    # So grad N_alpha . grad N_beta = (dN_alpha/dr * dN_beta/dr + dN_alpha/dz * dN_beta/dz)
    # And the integral is over the cell.

    # For bilinear elements on a rectangular cell, we can use 2x2 Gauss quadrature
    # which is exact for the integrals we need (polynomials of degree up to 2 in each variable
    # for the products, and r is linear, so total degree 3, which 2x2 Gauss handles exactly
    # for products of degree 1 in each variable... actually 2x2 Gauss is exact for degree 3
    # in each variable, so it should be fine).

    # Let me use 2x2 Gauss quadrature for each cell.

    # Gauss points and weights for [0,1]
    gp = 0.5 * (1 - 1/np.sqrt(3))
    gp2 = 0.5 * (1 + 1/np.sqrt(3))
    gw = 0.5
    gp_arr = np.array([gp, gp2])
    gw_arr = np.array([gw, gw])

    # Build the global stiffness matrix
    K = lil_matrix((n_nodes, n_nodes))

    for j in range(n_cells_z):
        for i in range(n_cells_r):
            sigma_c = cell_sigma[j, i]
            if sigma_c == 0.0:
                continue

            r_i = r[i]
            r_ip1 = r[i + 1]
            z_j = z[j]
            z_jp1 = z[j + 1]
            dr_i = r_ip1 - r_i
            dz_j = z_jp1 - z_j

            # Cell nodes
            nodes_cell = [node(i, j), node(i + 1, j), node(i, j + 1), node(i + 1, j + 1)]

            K_cell = np.zeros((4, 4))

            for ig in range(2):
                for jg in range(2):
                    xi = gp_arr[ig]
                    eta = gp_arr[jg]
                    w = gw_arr[ig] * gw_arr[jg]

                    # Shape functions and gradients
                    # N0 = (1-xi)(1-eta), N1 = xi(1-eta), N2 = (1-xi)eta, N3 = xi*eta
                    N = np.array([
                        (1 - xi) * (1 - eta),
                        xi * (1 - eta),
                        (1 - xi) * eta,
                        xi * eta
                    ])

                    # dN/dxi
                    dN_dxi = np.array([
                        -(1 - eta),
                        (1 - eta),
                        -eta,
                        eta
                    ])

                    # dN/deta
                    dN_deta = np.array([
                        -(1 - xi),
                        -xi,
                        (1 - xi),
                        xi
                    ])

                    # Physical coordinates
                    r_phys = r_i + dr_i * xi
                    z_phys = z_j + dz_j * eta

                    # Gradients in physical coordinates
                    # dN/dr = dN/dxi / dr_i
                    # dN/dz = dN/deta / dz_j
                    dN_dr = dN_dxi / dr_i
                    dN_dz = dN_deta / dz_j

                    # The integral integrand: sigma * (dN_alpha/dr * dN_beta/dr + dN_alpha/dz * dN_beta/dz) * r * J
                    # where J = dr_i * dz_j
                    # = sigma * (dN_alpha/dxi * dN_beta/dxi / dr_i^2 + dN_alpha/deta * dN_beta/deta / dz_j^2) * r_phys * dr_i * dz_j

                    J = dr_i * dz_j
                    factor = sigma_c * r_phys * J

                    # Add to K_cell
                    K_cell += factor * np.outer(dN_dr, dN_dr) * w + factor * np.outer(dN_dz, dN_dz) * w

            # Assemble into global matrix
            for a_local in range(4):
                for b_local in range(4):
                    K[nodes_cell[a_local], nodes_cell[b_local]] += K_cell[a_local, b_local]

    K = K.tocsc()

    # Now handle boundary conditions
    # Free nodes (unknowns)
    fixed_nodes = set()
    fixed_nodes.update(electrode_nodes)
    fixed_nodes.update(grounded_nodes)
    fixed_nodes.update(inside_electrode)

    # Inside shaft nodes are ignored (we don't solve for them, but they might
    # appear in the matrix. We need to handle them.)
    # Actually, nodes inside the shaft are not part of the conducting region,
    # so they shouldn't appear in the matrix. But they might be connected to
    # cells that are in the shaft region. Since the shaft is insulating,
    # cells inside the shaft have sigma = 0, so they don't contribute.
    # But nodes on the shaft boundary (r = a, gap+L < z < Ztop) are connected
    # to cells in the blood region. These nodes have no current crossing the
    # shaft side, which is automatically handled by the Neumann BC (no flux
    # through the shaft surface). Since we're not adding any flux, the natural
    # BC is satisfied.

    # However, we need to make sure that nodes inside the shaft are not
    # included as unknowns. Let's identify them and exclude them.
    # Actually, the problem says "Nodes strictly inside the shaft are ignored."
    # So we don't need to output them, but they might still be in the matrix.
    # Let's just solve for all nodes except the fixed ones, and then set
    # the fixed ones to their prescribed values.

    # But nodes inside the shaft are not in the conducting region, so they
    # shouldn't have any equation. Let's check if any cell has a node inside
    # the shaft. A cell is inside the shaft if 0 <= r <= a and gap+L <= z <= Ztop.
    # Since the grid includes the breakpoints, cells are either entirely in
    # the shaft or not. Cells in the shaft have sigma = 0 (since they're not
    # in tissue or blood), so they don't contribute to K. But the nodes of
    # these cells might still be in K if they're shared with non-shaft cells.
    # Nodes on the boundary of the shaft (r = a or z = gap+L or z = Ztop)
    # are shared with blood cells, so they appear in K. Nodes strictly inside
    # the shaft are only in shaft cells, so they don't appear in K.

    # So we can just solve the system for all nodes that appear in K,
    # applying the fixed BCs.

    # Build the right-hand side
    b = np.zeros(n_nodes)

    # Apply fixed BCs
    # For electrode nodes, b = voltage
    for n in electrode_nodes:
        b[n] = voltage
    for n in inside_electrode:
        b[n] = voltage
    for n in grounded_nodes:
        b[n] = 0.0

    # Identify free nodes
    all_nodes = set(range(n_nodes))
    free_nodes = all_nodes - fixed_nodes

    # But we also need to exclude nodes that are not in any cell (i.e.,
    # nodes that don't appear in K). Let's find which nodes have non-zero
    # rows in K.
    K_csr = K.tocsr()
    has_equation = K_csr.getnnz(axis=1) > 0

    # Free nodes that have equations
    free_list = []
    free_idx = []
    for n in range(n_nodes):
        if n in free_nodes and has_equation[n]:
            free_list.append(n)
            free_idx.append(n)

    if len(free_list) == 0:
        # No unknowns, solution is just the fixed values
        v = np.zeros(n_nodes)
        for n in electrode_nodes:
            v[n] = voltage
        for n in inside_electrode:
            v[n] = voltage
        # grounded nodes are 0
        v_2d = v.reshape(nz, nr)
        np.savez(output_file, v=v_2d, impedance=0.0)
        return

    # Build the reduced system
    free_arr = np.array(free_list)
    K_ff = K[free_arr[:, None], free_arr[None, :]]
    b_ff = b[free_arr]

    # Solve
    u_ff = spsolve(K_ff, b_ff)

    # Assemble full solution
    v = np.zeros(n_nodes)
    for n in electrode_nodes:
        v[n] = voltage
    for n in inside_electrode:
        v[n] = voltage
    # Grounded nodes are 0 (already)
    for idx, n in enumerate(free_list):
        v[n] = u_ff[idx]

    # Compute total current leaving the electrode
    # I = integral over electrode surface of sigma * du/dn ds
    # The electrode surface consists of:
    # 1. Bottom face: z = gap, 0 <= r <= a, normal is -z direction (into tissue below)
    #    Current leaving electrode = -sigma * du/dz * (2*pi*r) dr (at z = gap)
    #    Wait, the normal to the electrode surface pointing outward from the
    #    electrode is -z for the bottom face. The current density is J = -sigma * grad u.
    #    Current leaving = integral J . n ds = integral (-sigma * grad u) . n ds
    #    For bottom face, n = -e_z, so J . n = -sigma * du/dz * (-1) = sigma * du/dz
    #    Wait, let me be careful.
    #    J = -sigma * grad u (current density)
    #    Current leaving the electrode = integral J . n_out ds
    #    where n_out is the outward normal from the electrode into the medium.
    #    For the bottom face (z = gap), the electrode is above, medium is below.
    #    n_out = -e_z. J . n_out = J_z * (-1) = -sigma * du/dz * (-1) = sigma * du/dz
    #    But du/dz at the bottom face: the potential increases as we go into the
    #    electrode (from tissue to electrode), so du/dz < 0 at the bottom face
    #    (since z increases upward and the electrode is at higher potential).
    #    Actually, the tissue is below (z < gap) and the electrode is at z >= gap.
    #    The potential in the tissue near the bottom face is lower than voltage.
    #    So du/dz at z = gap from the tissue side is negative (potential decreases
    #    as z increases toward the electrode... no, potential increases toward the
    #    electrode).
    #    Let me just compute the current numerically.

    # I'll compute the current by integrating the flux over the electrode surface
    # using the finite element solution.

    # For each cell adjacent to the electrode surface, compute the flux.
    # The electrode surface is at:
    # - z = gap, r in [0, a] (bottom face)
    # - r = a, z in [gap, gap + L] (side face)
    # - z = gap + L, r in [0, a] (top face)

    # For the bottom face (z = gap):
    # Cells below this face have z[j+1] = gap, so z[j] < gap.
    # The face is the top of these cells.
    # Current leaving through this face = integral sigma * du/dz * 2*pi*r dr
    # (with appropriate sign)

    # Actually, let me compute the total current as:
    # I = integral over electrode surface of (-sigma * grad u) . n ds
    # where n is the outward normal from the electrode.

    # For the bottom face (z = gap, n = -e_z):
    # I_bottom = integral_0^a (-sigma * du/dz) * (-1) * 2*pi*r dr = integral_0^a sigma * du/dz * 2*pi*r dr
    # But du/dz here is the derivative in the tissue (below the face).
    # Since the potential is higher in the electrode, du/dz < 0 at the bottom face
    # from the tissue side (potential increases as z increases, but the electrode
    # is at z >= gap, so at z = gap from below, du/dz is the rate of increase).
    # Actually, the potential at z = gap is voltage, and below it the potential
    # decreases, so du/dz > 0 at z = gap from below? No.
    # If u = voltage at z = gap and u < voltage below, then as z increases from
    # below to gap, u increases, so du/dz > 0 at z = gap from below.
    # Then I_bottom = integral sigma * du/dz * 2*pi*r dr > 0, which means current
    # is leaving the electrode downward. That makes sense.

    # For the side face (r = a, n = e_r):
    # I_side = integral_gap^{gap+L} (-sigma * du/dr) * 1 * 2*pi*a dz
    # du/dr at r = a from the blood side: potential is voltage at r = a and
    # decreases as r increases, so du/dr < 0. Then -sigma * du/dr > 0, so
    # current is leaving the electrode radially outward. Makes sense.

    # For the top face (z = gap + L, n = +e_z):
    # I_top = integral_0^a (-sigma * du/dz) * 1 * 2*pi*r dr
    # But above the top face is the shaft (insulating), so no current flows
    # through the top face. The top face is also the bottom of the shaft,
    # and since the shaft is insulating, the current through this face is 0.
    # Actually, the top face of the electrode is in contact with the shaft,
    # which is insulating, so no current flows. But the potential at this face
    # is voltage (it's part of the electrode surface). The cells above this face
    # are in the shaft (sigma = 0), so no current flows through them.
    # So I_top = 0.

    # Let me compute I_bottom and I_side.

    # For I_bottom:
    # Find cells with z[j+1] = gap (i.e., the cell's top is at z = gap)
    # and r[i] < a (cell is under the electrode)
    I_total = 0.0

    # Bottom face: z = gap
    # Cells below: z[j+1] = gap, so j such that z[j+1] = gap
    for j in range(n_cells_z):
        if np.isclose(z[j + 1], gap, rtol=0, atol=1e-12):
            # This cell's top face is at z = gap
            for i in range(n_cells_r):
                if r[i + 1] <= a + 1e-12:
                    # Cell is under the electrode
                    sigma_c = cell_sigma[j, i]
                    if sigma_c == 0:
                        continue

                    r_i = r[i]
                    r_ip1 = r[i + 1]
                    z_j = z[j]
                    z_jp1 = z[j + 1]
                    dr_i = r_ip1 - r_i
                    dz_j = z_jp1 - z_j

                    # Cell nodes
                    n0 = node(i, j)
                    n1 = node(i + 1, j)
                    n2 = node(i, j + 1)
                    n3 = node(i + 1, j + 1)

                    u0 = v[n0]
                    u1 = v[n1]
                    u2 = v[n2]
                    u3 = v[n3]

                    # Compute du/dz at the top face (z = gap)
                    # For bilinear elements, du/dz is linear in r along the face.
                    # At the top face (eta = 1):
                    # u(r, gap) = N0*u0 + N1*u1 + N2*u2 + N3*u3
                    # But N0 and N1 are at eta = 0, N2 and N3 at eta = 1.
                    # At eta = 1: N0 = 0, N1 = 0, N2 = 1-xi, N3 = xi
                    # So u(r, gap) = (1-xi)*u2 + xi*u3
                    # du/dz at eta = 1:
                    # dN0/deta = -(1-xi), dN1/deta = -xi, dN2/deta = (1-xi), dN3/deta = xi
                    # du/dz = (dN0/deta * u0 + dN1/deta * u1 + dN2/deta * u2 + dN3/deta * u3) / dz_j
                    # = (-(1-xi)*u0 - xi*u1 + (1-xi)*u2 + xi*u3) / dz_j
                    # = ((u2-u0)*(1-xi) + (u3-u1)*xi) / dz_j

                    # Integrate sigma * du/dz * 2*pi*r dr over the face
                    # du/dz is linear in xi (and hence in r):
                    # du/dz = (A*(1-xi) + B*xi) / dz_j
                    # where A = u2 - u0, B = u3 - u1
                    # r = r_i + dr_i * xi
                    # integral_0^1 sigma * (A*(1-xi) + B*xi) / dz_j * 2*pi * (r_i + dr_i*xi) * dr_i dxi
                    # = sigma * 2*pi * dr_i / dz_j * integral_0^1 (A*(1-xi) + B*xi) * (r_i + dr_i*xi) dxi
                    # = sigma * 2*pi * dr_i / dz_j * [ A*(r_i + dr_i/2) + B*(r_i/2 + dr_i/3) ]
                    # Wait, let me compute more carefully.
                    # integral_0^1 (A*(1-xi) + B*xi) * (r_i + dr_i*xi) dxi
                    # = A * integral_0^1 (1-xi)*(r_i + dr_i*xi) dxi + B * integral_0^1 xi*(r_i + dr_i*xi) dxi
                    # integral_0^1 (1-xi)*(r_i + dr_i*xi) dxi = r_i * integral_0^1 (1-xi) dxi + dr_i * integral_0^1 xi*(1-xi) dxi
                    # = r_i * 1/2 + dr_i * 1/6
                    # integral_0^1 xi*(r_i + dr_i*xi) dxi = r_i * 1/2 + dr_i * 1/3

                    A_val = u2 - u0
                    B_val = u3 - u1
                    integral_xi = A_val * (r_i / 2.0 + dr_i / 6.0) + B_val * (r_i / 2.0 + dr_i / 3.0)
                    I_bottom_cell = sigma_c * 2.0 * np.pi * dr_i / dz_j * integral_xi
                    I_total += I_bottom_cell

    # Side face: r = a, z in [gap, gap + L]
    # Cells to the right of r = a: r[i] = a, so i such that r[i] = a
    for i in range(n_cells_r):
        if np.isclose(r[i], a, rtol=0, atol=1e-12):
            for j in range(n_cells_z):
                if z[j] >= gap - 1e-12 and z[j + 1] <= gap + L + 1e-12:
                    # Cell is to the right of the electrode side
                    sigma_c = cell_sigma[j, i]
                    if sigma_c == 0:
                        continue

                    r_i = r[i]
                    r_ip1 = r[i + 1]
                    z_j = z[j]
                    z_jp1 = z[j + 1]
                    dr_i = r_ip1 - r_i
                    dz_j = z_jp1 - z_j

                    n0 = node(i, j)
                    n1 = node(i + 1, j)
                    n2 = node(i, j + 1)
                    n3 = node(i + 1, j + 1)

                    u0 = v[n0]
                    u1 = v[n1]
                    u2 = v[n2]
                    u3 = v[n3]

                    # du/dr at the left face (xi = 0):
                    # At xi = 0: N0 = 1-eta, N1 = 0, N2 = eta, N3 = 0
                    # u(a, z) = (1-eta)*u0 + eta*u2
                    # du/dr = (dN0/dxi * u0 + dN1/dxi * u1 + dN2/dxi * u2 + dN3/dxi * u3) / dr_i
                    # dN/dxi at xi = 0:
                    # dN0/dxi = -(1-eta), dN1/dxi = (1-eta), dN2/dxi = -eta, dN3/dxi = eta
                    # du/dr = (-(1-eta)*u0 + (1-eta)*u1 - eta*u2 + eta*u3) / dr_i
                    # = ((u1-u0)*(1-eta) + (u3-u2)*eta) / dr_i

                    # Current leaving through side face:
                    # I_side = integral (-sigma * du/dr) * 2*pi*a dz
                    # du/dr is linear in eta:
                    # du/dr = (C*(1-eta) + D*eta) / dr_i
                    # where C = u1 - u0, D = u3 - u2
                    # integral_0^1 (-sigma) * (C*(1-eta) + D*eta) / dr_i * 2*pi*a dz_j deta
                    # = -sigma * 2*pi*a * dz_j / dr_i * integral_0^1 (C*(1-eta) + D*eta) deta
                    # = -sigma * 2*pi*a * dz_j / dr_i * (C/2 + D/2)
                    # = -sigma * 2*pi*a * dz_j / dr_i * (C + D) / 2

                    C_val = u1 - u0
                    D_val = u3 - u2
                    I_side_cell = -sigma_c * 2.0 * np.pi * a * dz_j / dr_i * (C_val + D_val) / 2.0
                    I_total += I_side_cell

    # Compute impedance
    if abs(I_total) < 1e-30:
        impedance = 0.0
    else:
        impedance = voltage / I_total

    # Reshape v to 2D
    v_2d = v.reshape(nz, nr)

    # Save solution
    np.savez(output_file, v=v_2d, impedance=impedance)


if __name__ == '__main__':
    main()
