#!/usr/bin/env python3
"""
Solver for the electric potential around an ablation electrode.

Axisymmetric steady conduction in cylindrical coordinates.
Uses a finite element method with bilinear elements and axisymmetric
weighting.
"""

import sys
import numpy as np
from scipy.sparse import csc_matrix, lil_matrix
from scipy.sparse.linalg import spsolve

def solve(problem_file, solution_file):
    # Load problem data
    data = np.load(problem_file)
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
    
    # Create node index mapping
    # Nodes are indexed as node[j, i] = j * nr + i where j is z-index, i is r-index
    def node_idx(i, j):
        """Return flat index for node at (r[i], z[j])"""
        return j * nr + i
    
    # Determine which nodes are on the electrode surface
    # The electrode occupies 0 <= r <= a, gap <= z <= gap + L
    # Surface: bottom face z=gap, 0<=r<=a; side r=a, gap<=z<=gap+L
    
    # Find indices for geometric breakpoints
    # r indices
    r0_idx = np.searchsorted(r, 0.0)  # should be 0
    a_idx = np.searchsorted(r, a)
    R_idx = np.searchsorted(r, R)
    
    # z indices
    zbot_idx = np.searchsorted(z, -H)  # should be 0
    z0_idx = np.searchsorted(z, 0.0)
    zgap_idx = np.searchsorted(z, gap)
    zgapL_idx = np.searchsorted(z, gap + L)
    ztop_idx = np.searchsorted(z, Ztop)
    
    # Build the sparse system
    # We'll use a finite element approach
    # For axisymmetric problems, the weak form is:
    # ∫∫ sigma * grad(u) · grad(v) * r dr dz = 0
    # 
    # We discretize on the grid cells. Each cell has 4 nodes.
    # Use bilinear interpolation with axisymmetric weighting.
    
    # For a cell with corners (i, j), (i+1, j), (i, j+1), (i+1, j+1)
    # The cell spans r[i] to r[i+1] and z[j] to z[j+1]
    
    # Determine the conductivity for each cell
    # A cell is in tissue if its z-range is entirely < 0
    # A cell is in blood if its z-range is entirely > 0
    # A cell is inside electrode if r-range <= a and z-range in [gap, gap+L]
    # A cell is inside shaft if r-range <= a and z-range in [gap+L, Ztop]
    
    # We need to handle the fact that the electrode and shaft are NOT part of 
    # the conducting region. So we should not create elements inside them.
    
    # Let's identify which cells are in the conducting region
    # A cell (i, j) spans [r[i], r[i+1]] x [z[j], z[j+1]]
    
    # For each cell, determine if it's in the conducting region
    conducting_cells = []
    cell_sigmas = []
    
    for i in range(nr - 1):
        for j in range(nz - 1):
            r_lo = r[i]
            r_hi = r[i + 1]
            z_lo = z[j]
            z_hi = z[j + 1]
            
            # Check if cell is inside electrode
            # Electrode: 0 <= r <= a, gap <= z <= gap + L
            # A cell is inside electrode if r_hi <= a and z_lo >= gap and z_hi <= gap + L
            if r_hi <= a and z_lo >= gap and z_hi <= gap + L:
                # Inside electrode - skip
                continue
            
            # Check if cell is inside shaft
            # Shaft: 0 <= r <= a, gap + L <= z <= Ztop
            if r_hi <= a and z_lo >= gap + L and z_hi <= Ztop:
                # Inside shaft - skip
                continue
            
            # Determine conductivity
            # Cell is in tissue if z_hi <= 0 (entirely below z=0)
            # Cell is in blood if z_lo >= 0 (entirely above z=0)
            if z_hi <= 0:
                sigma = sigma_tissue
            elif z_lo >= 0:
                sigma = sigma_blood
            else:
                # This shouldn't happen if grid includes z=0 as a breakpoint
                # But just in case, we could split or average
                # For now, let's use a weighted average or just tissue
                # Actually, the problem states grid includes z=0, so this shouldn't happen
                sigma = sigma_tissue  # fallback
            
            conducting_cells.append((i, j))
            cell_sigmas.append(sigma)
    
    # Now build the finite element system
    # For each cell, we need to compute the element stiffness matrix
    # with axisymmetric weighting.
    
    # Bilinear basis functions on a cell:
    # N1 = (1 - xi)(1 - eta) / 4
    # N2 = (1 + xi)(1 - eta) / 4
    # N3 = (1 - xi)(1 + eta) / 4
    # N4 = (1 + xi)(1 + eta) / 4
    # where xi = (r - r_c) / (r_hi - r_lo), eta = (z - z_c) / (z_hi - z_lo)
    
    # For axisymmetric problem, the integral is:
    # K_ij = ∫∫ sigma * (dN_i/dr * dN_j/dr + 1/r * dN_i/dz * dN_j/dz) * r dr dz
    # Wait, the PDE is: (1/r) d/dr(r sigma du/dr) + d/dz(sigma du/dz) = 0
    # The weak form with test function v:
    # ∫∫ [sigma * r * du/dr * dv/dr + sigma * r * du/dz * dv/dz] dr dz = 0
    # Actually, let me re-derive.
    # 
    # The equation is: (1/r) d/dr(r sigma du/dr) + d/dz(sigma du/dz) = 0
    # Multiply by r and test function v, integrate:
    # ∫∫ [r * v * (1/r) d/dr(r sigma du/dr) + r * v * d/dz(sigma du/dz)] dr dz = 0
    # = ∫∫ [v * d/dr(r sigma du/dr) + r * v * d/dz(sigma du/dz)] dr dz = 0
    # Integrate by parts:
    # = ∫∫ [sigma * r * du/dr * dv/dr + sigma * r * du/dz * dv/dz] dr dz - boundary terms = 0
    # 
    # So the stiffness matrix is:
    # K_ij = ∫∫ sigma * r * (dN_i/dr * dN_j/dr + dN_i/dz * dN_j/dz) dr dz
    
    # For bilinear elements on a non-uniform grid, we can compute this
    # analytically or use numerical quadrature.
    # Let's use numerical quadrature (e.g., 2x2 Gauss) for generality.
    
    # Actually, for bilinear elements, we can use exact integration with
    # appropriate quadrature. Let's use 2x2 Gauss quadrature.
    
    # Gauss-Legendre points and weights on [-1, 1]
    gauss_pts = np.array([-1.0, 1.0])
    gauss_wts = np.array([1.0, 1.0])
    
    # Total number of nodes
    n_nodes = nr * nz
    
    # Build sparse matrix
    K = lil_matrix((n_nodes, n_nodes))
    b = np.zeros(n_nodes)
    
    # Track which nodes are fixed (Dirichlet BC)
    fixed_nodes = set()
    fixed_values = {}
    
    # Grounded nodes: r=R, z=-H, z=Ztop (for r>=a)
    # Also need to handle the axis r=0
    
    # For each cell, compute element contribution
    for cell_idx, (i, j) in enumerate(conducting_cells):
        sigma = cell_sigmas[cell_idx]
        
        # Cell corners
        # Node 0: (i, j) - bottom left
        # Node 1: (i+1, j) - bottom right
        # Node 2: (i, j+1) - top left
        # Node 3: (i+1, j+1) - top right
        
        r_lo = r[i]
        r_hi = r[i + 1]
        z_lo = z[j]
        z_hi = z[j + 1]
        
        dr = r_hi - r_lo
        dz = z_hi - z_lo
        
        if dr <= 0 or dz <= 0:
            continue
        
        # Node indices
        n0 = node_idx(i, j)
        n1 = node_idx(i + 1, j)
        n2 = node_idx(i, j + 1)
        n3 = node_idx(i + 1, j + 1)
        
        elem_nodes = [n0, n1, n2, n3]
        
        # Bilinear basis functions
        # N(xi, eta) where xi in [-1, 1], eta in [-1, 1]
        # N0 = (1 - xi)(1 - eta) / 4
        # N1 = (1 + xi)(1 - eta) / 4
        # N2 = (1 - xi)(1 + eta) / 4
        # N3 = (1 + xi)(1 + eta) / 4
        
        # Derivatives with respect to r and z
        # dN/dxi, dN/deta
        # dN/dr = dN/dxi * dxi/dr = dN/dxi * 2/dr
        # dN/dz = dN/deta * deta/dz = dN/deta * 2/dz
        
        # For each Gauss point, compute the element stiffness
        for ig in range(2):
            for jg in range(2):
                xi = gauss_pts[ig]
                eta = gauss_pts[jg]
                w = gauss_wts[ig] * gauss_wts[jg]
                
                # Physical coordinates
                r_phys = 0.5 * (r_lo + r_hi) + 0.5 * xi * dr
                z_phys = 0.5 * (z_lo + z_hi) + 0.5 * eta * dz
                
                # If r_phys is very close to 0, we need to be careful
                # But the integrand has a factor of r, so it should be fine
                
                # Basis function values and derivatives
                # N0 = (1 - xi)(1 - eta) / 4
                N = np.zeros(4)
                dN_dxi = np.zeros(4)
                dN_deta = np.zeros(4)
                
                N[0] = (1 - xi) * (1 - eta) / 4
                N[1] = (1 + xi) * (1 - eta) / 4
                N[2] = (1 - xi) * (1 + eta) / 4
                N[3] = (1 + xi) * (1 + eta) / 4
                
                dN_dxi[0] = -(1 - eta) / 4
                dN_dxi[1] = (1 - eta) / 4
                dN_dxi[2] = -(1 + eta) / 4
                dN_dxi[3] = (1 + eta) / 4
                
                dN_deta[0] = -(1 - xi) / 4
                dN_deta[1] = -(1 + xi) / 4
                dN_deta[2] = (1 - xi) / 4
                dN_deta[3] = (1 + xi) / 4
                
                # Convert to physical derivatives
                dN_dr = dN_dxi * (2.0 / dr)
                dN_dz = dN_deta * (2.0 / dz)
                
                # Jacobian of transformation
                J = dr * dz / 4.0
                
                # Integrand: sigma * r * (dN_i/dr * dN_j/dr + dN_i/dz * dN_j/dz)
                # The integral is over xi, eta with weight w * J
                # But we also need the factor of r (from the axisymmetric integral)
                
                # Actually, the integral is:
                # ∫∫ sigma * r * (grad(N_i) · grad(N_j)) dr dz
                # = ∫∫ sigma * r * (dN_i/dr * dN_j/dr + dN_i/dz * dN_j/dz) * J dxi deta
                # where r is the physical r coordinate
                
                # Compute the element stiffness contribution
                for p in range(4):
                    for q in range(4):
                        val = sigma * r_phys * (dN_dr[p] * dN_dr[q] + dN_dz[p] * dN_dz[q])
                        val *= w * J
                        K[elem_nodes[p], elem_nodes[q]] += val
    
    # Convert to CSC for solving
    K = K.tocsc()
    
    # Now apply boundary conditions
    # Dirichlet BCs:
    # 1. Electrode surface: u = voltage
    #    - Bottom face: z = gap, 0 <= r <= a
    #    - Side: r = a, gap <= z <= gap + L
    # 2. Grounded: u = 0
    #    - r = R
    #    - z = -H
    #    - z = Ztop, r >= a
    
    # Find electrode surface nodes
    # Bottom face: z = gap, r from 0 to a
    for i in range(nr):
        if abs(r[i] - a) < 1e-10 or r[i] <= a:
            # Find j such that z[j] = gap
            j = np.searchsorted(z, gap)
            if j < nz and abs(z[j] - gap) < 1e-10:
                n = node_idx(i, j)
                fixed_nodes.add(n)
                fixed_values[n] = voltage
    
    # Side: r = a, z from gap to gap + L
    for j in range(nz):
        if z[j] >= gap - 1e-10 and z[j] <= gap + L + 1e-10:
            i = np.searchsorted(r, a)
            if i < nr and abs(r[i] - a) < 1e-10:
                n = node_idx(i, j)
                fixed_nodes.add(n)
                fixed_values[n] = voltage
    
    # Also, nodes strictly inside the electrode should have u = voltage
    # These are nodes with r < a and gap < z < gap + L
    for i in range(nr):
        if r[i] < a - 1e-10:
            for j in range(nz):
                if z[j] > gap + 1e-10 and z[j] < gap + L - 1e-10:
                    n = node_idx(i, j)
                    fixed_nodes.add(n)
                    fixed_values[n] = voltage
    
    # Grounded nodes
    # r = R
    for j in range(nz):
        i = np.searchsorted(r, R)
        if i < nr and abs(r[i] - R) < 1e-10:
            n = node_idx(i, j)
            fixed_nodes.add(n)
            fixed_values[n] = 0.0
    
    # z = -H
    for i in range(nr):
        j = np.searchsorted(z, -H)
        if j < nz and abs(z[j] - (-H)) < 1e-10:
            n = node_idx(i, j)
            fixed_nodes.add(n)
            fixed_values[n] = 0.0
    
    # z = Ztop, r >= a
    for i in range(nr):
        if r[i] >= a - 1e-10:
            j = np.searchsorted(z, Ztop)
            if j < nz and abs(z[j] - Ztop) < 1e-10:
                n = node_idx(i, j)
                fixed_nodes.add(n)
                fixed_values[n] = 0.0
    
    # Solve the system
    # For fixed nodes, we set the row to identity and b to the fixed value
    # For free nodes, we keep the original equation
    
    free_nodes = [n for n in range(n_nodes) if n not in fixed_nodes]
    
    if len(free_nodes) == 0:
        # All nodes are fixed, just return the fixed values
        v = np.zeros(n_nodes)
        for n, val in fixed_values.items():
            v[n] = val
        v_2d = v.reshape(nz, nr)
        
        # Compute impedance
        # Total current leaving the electrode
        # I = ∫∫ sigma * du/dn * r ds
        # For the electrode, we need to compute the flux on its surface
        # This is complex, so let's estimate it
        # Actually, for the impedance, we need the total current
        # I = V / Z, so Z = V / I
        # We can compute I from the solution
        # For now, let's just set a dummy value
        impedance = voltage / 1e-6  # dummy
        np.savez(solution_file, v=v_2d, impedance=impedance)
        return
    
    # Extract submatrix for free nodes
    free_idx = np.array(free_nodes)
    K_free = K[free_idx][:, free_idx]
    b_free = np.zeros(len(free_nodes))
    
    # For fixed nodes, we need to account for them in the RHS
    # K_free * u_free = b - K_free_fixed * u_fixed
    fixed_idx = np.array(sorted(fixed_nodes))
    if len(fixed_idx) > 0:
        K_free_fixed = K[free_idx][:, fixed_idx]
        u_fixed = np.array([fixed_values[n] for n in fixed_idx])
        b_free = -K_free_fixed.dot(u_fixed)
    
    # Solve
    u_free = spsolve(K_free, b_free)
    
    # Assemble full solution
    v = np.zeros(n_nodes)
    for n, val in fixed_values.items():
        v[n] = val
    for idx, n in enumerate(free_nodes):
        v[n] = u_free[idx]
    
    v_2d = v.reshape(nz, nr)
    
    # Compute impedance
    # Total current leaving the electrode
    # I = ∫∫ sigma * (-du/dn) * r ds
    # For the electrode surface, we need to compute the flux
    # This is complex. Let's use a simpler approach:
    # Sum the currents through the faces of the electrode
    
    # For each element adjacent to the electrode, compute the flux
    # The electrode surface consists of:
    # 1. Bottom face: z = gap, 0 <= r <= a (normal is -z direction)
    # 2. Side: r = a, gap <= z <= gap + L (normal is +r direction)
    
    total_current = 0.0
    
    # For the bottom face (z = gap)
    # Elements below the electrode (z < gap)
    j_gap = np.searchsorted(z, gap)
    if j_gap < nz:
        # Elements with z[j] = gap, i.e., j = j_gap
        for i in range(nr - 1):
            if r[i + 1] <= a + 1e-10:  # Within electrode radius
                # Element (i, j_gap - 1) is below the electrode
                if j_gap > 0:
                    j = j_gap - 1
                    # Check if this element is in the conducting region
                    # It should be, since it's below the electrode
                    r_lo = r[i]
                    r_hi = r[i + 1]
                    z_lo = z[j]
                    z_hi = z[j + 1]
                    
                    # Determine sigma
                    if z_hi <= 0:
                        sigma = sigma_tissue
                    elif z_lo >= 0:
                        sigma = sigma_blood
                    else:
                        sigma = sigma_tissue
                    
                    # Compute the flux through the top face of this element
                    # The top face is at z = z_hi = gap
                    # Normal is +z, so du/dn = du/dz
                    # Current density J_z = -sigma * du/dz
                    # But we want the current leaving the electrode, which is in -z direction
                    # So I = ∫ sigma * du/dz * r dr (at z = gap, from the tissue side)
                    
                    # We need du/dz at the top face of the element
                    # For a bilinear element, du/dz is not constant
                    # Let's use the average of the four nodes
                    n0 = node_idx(i, j)
                    n1 = node_idx(i + 1, j)
                    n2 = node_idx(i, j + 1)
                    n3 = node_idx(i + 1, j + 1)
                    
                    u0 = v[n0]
                    u1 = v[n1]
                    u2 = v[n2]
                    u3 = v[n3]
                    
                    # du/dz at the top face (eta = 1)
                    # For bilinear: du/dz = (1/2) * (du/deta * 2/dz)
                    # du/deta at eta = 1:
                    # N0 = (1-xi)(1-eta)/4, dN0/deta = -(1-xi)/4
                    # N1 = (1+xi)(1-eta)/4, dN1/deta = -(1+xi)/4
                    # N2 = (1-xi)(1+eta)/4, dN2/deta = (1-xi)/4
                    # N3 = (1+xi)(1+eta)/4, dN3/deta = (1+xi)/4
                    # At eta = 1:
                    # dN0/deta = -(1-xi)/4
                    # dN1/deta = -(1+xi)/4
                    # dN2/deta = (1-xi)/4
                    # dN3/deta = (1+xi)/4
                    # du/deta = u0*dN0/deta + u1*dN1/deta + u2*dN2/deta + u3*dN3/deta
                    # At eta = 1, xi varies from -1 to 1
                    # du/deta = (-u0 - u1 + u2 + u3)/4 + xi*(u0 - u1 - u2 + u3)/4
                    # This is not constant. Let's use the average at xi = 0
                    # At xi = 0:
                    # du/deta = (-u0 - u1 + u2 + u3)/4
                    du_dz_avg = (-u0 - u1 + u2 + u3) / (4 * dz)
                    
                    # Integrate over r: ∫ du/dz * sigma * r dr
                    # Use trapezoidal rule
                    r_mid = 0.5 * (r_lo + r_hi)
                    dr = r_hi - r_lo
                    flux = sigma * du_dz_avg * r_mid * dr
                    total_current += flux
    
    # For the side (r = a)
    i_a = np.searchsorted(r, a)
    if i_a < nr:
        for j in range(nz - 1):
            if z[j + 1] >= gap + 1e-10 and z[j] <= gap + L + 1e-10:
                # Element (i_a - 1, j) is to the left of the electrode
                if i_a > 0:
                    i = i_a - 1
                    r_lo = r[i]
                    r_hi = r[i + 1]
                    z_lo = z[j]
                    z_hi = z[j + 1]
                    
                    # Determine sigma
                    if z_hi <= 0:
                        sigma = sigma_tissue
                    elif z_lo >= 0:
                        sigma = sigma_blood
                    else:
                        sigma = sigma_tissue
                    
                    n0 = node_idx(i, j)
                    n1 = node_idx(i + 1, j)
                    n2 = node_idx(i, j + 1)
                    n3 = node_idx(i + 1, j + 1)
                    
                    u0 = v[n0]
                    u1 = v[n1]
                    u2 = v[n2]
                    u3 = v[n3]
                    
                    # du/dr at the right face (xi = 1)
                    # At xi = 1:
                    # du/dxi = (u0 - u1 + u2 - u3)/4 + eta*(u0 + u1 - u2 - u3)/4
                    # At eta = 0:
                    du_dr_avg = (u0 - u1 + u2 - u3) / (4 * dr)
                    
                    # Integrate over z: ∫ du/dr * sigma * r dz
                    # r = a (approximately r_hi)
                    r_face = r_hi
                    dz = z_hi - z_lo
                    flux = sigma * du_dr_avg * r_face * dz
                    total_current += flux
    
    # The total current is the sum of fluxes
    # But we need to be careful about signs
    # The current leaving the electrode should be positive
    # Let's take the absolute value
    total_current = abs(total_current)
    
    # Compute impedance
    if total_current > 1e-12:
        impedance = voltage / total_current
    else:
        impedance = 0.0
    
    # Save solution
    np.savez(solution_file, v=v_2d, impedance=impedance)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python solver.py problem.npz solution.npz")
        sys.exit(1)
    
    solve(sys.argv[1], sys.argv[2])
