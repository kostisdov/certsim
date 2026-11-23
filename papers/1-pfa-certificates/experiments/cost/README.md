# E4: the cost of checking

Three checkers compute the same bounds from the same exact cell integrals:

| Checker | Where | Arithmetic | Rigorous |
|---|---|---|---|
| arb | `certsim.checker.check` | 256-bit ball arithmetic, cell by cell | yes |
| rigorous vectorized | `certsim.fastcheck.fast_check` | float64 ball arithmetic over arrays of cells; 1/r moments per column and final roots in arb | yes |
| float prototype | `fast_check.py` (here) | float64, Gauss quadrature for the 1/r part, no rounding control | no |

In the rigorous vectorized checker every quantity is a pair of float64 arrays
(midpoint, radius), and every operation adds a bound on its own rounding to the
radius (relative error at most the unit roundoff for +, -, x; the standard
gamma_n bound for sums). Two choices keep double precision tight: the radial
part of each field is integrated as N^2 / r with N = P r + Q, so that the
cancellation between its polynomial and 1/r parts happens once, in N's
coefficients; and each cell's corner value is subtracted before the change to
monomials, so that rounding is relative to the variation across the cell, not
to the value. Voxel (adjoint) outputs are not supported.

`bench.py` solves the paper 1 lesion-map problem at degree 3 on grids refined
systematically and times each checker. MacBook Pro M5 Pro, 48 GB.

```
uv run python bench.py      # writes bench.json
```

| Grid | Unknowns | Solve | Rigorous vectorized | Share of solve | arb | Vectorized eta / arb eta | Float prototype |
|---|---|---|---|---|---|---|---|
| 0.5 mm | 157 360 | 2.2 s | 0.33 s | 15% | 5.6 s | 1.0000045 | 0.10 s |
| 0.25 mm | 506 476 | 9.5 s | 1.09 s | 11% | 17.8 s | 1.0000026 | 0.30 s |
| 0.125 mm | 1 749 265 | 52.8 s | 3.58 s | 6.8% | 59.8 s | 1.0000004 | 1.03 s |
| 0.0625 mm | 6 270 646 | 333 s | 13.2 s | **4.0%** | - | - | 3.70 s |

Every tile interval of the vectorized checker overlaps the arb one. The
vectorized check grows linearly with the number of cells while the solve grows
faster, so its share falls as the problem grows: guaranteed bounds at 4%
of a solve at 6.3 million unknowns, and about 17 times faster than the arb checker.
The rigour costs about a factor of 3.6 over the non-rigorous float prototype.
Timings rerun on 2026-11-23 with the final fastcheck.py (underflow bounds included).
