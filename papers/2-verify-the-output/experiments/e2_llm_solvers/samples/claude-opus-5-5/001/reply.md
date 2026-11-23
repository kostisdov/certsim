The program uses bilinear (Q1) finite elements on the grid cells and leaves out the cells inside the electrode and shaft. Each element integral is weighted by sigma*r and computed with 2x2 Gauss quadrature, which is exact for these integrands on any rectangular cell. It fixes the electrode and ground nodes, solves the sparse system for the remaining nodes, and gets the current as 2*pi times the sum of the residual (K*u) over the electrode nodes, from which impedance = voltage / current.

(Code block saved verbatim as solver.py.)
