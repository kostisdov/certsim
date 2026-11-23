# E3c: surrogates that output both fields on the certificate grid, two inputs

The lesion-map problem of paper 1 at contact, with two parameters: the
conductivity ratio rho in [1.7, 3.75] and the electrode length L in
[2.5, 3.5] mm. Changing L moves only the grid lines above the tissue, so every
member's certificate grid is the reference grid (L = 3 mm) with those lines
mapped piecewise linearly (`common.grid_for`); all members share one index grid
of 997 x 508 nodes (degree 3). Our solver's certificate on a mapped grid has the
same quality as on the reference grid (eta 1.3e-5 at L = 2 mm).

Samples (`common.sample_sets`): 64 training members (Latin hypercube), 24
calibration and 12 in-range test members (uniform), and 8 out-of-range test
members (rho = 0.7, 1, 5, 10; L = 2, 4 mm). References are solved with our solver
at degree 3 and checked rigorously (`make_data.py`, 108 solves in 6 min with 4
workers). Fields are stored in float32: a stored reference field itself receives
eta of about 1.6e-4, and the projection onto the POD modes removes part of this
rounding.

Two surrogates, each made admissible by imposing the boundary values exactly
(electrode and ground values of v; psi zero on the axis and one constant on the
shaft), so every prediction is a certificate:

1. **A reduced-order surrogate** (`reduced_basis.py`): POD with 12 modes per
   field and a network with two hidden layers of 32 tanh units mapping
   (log rho, L) to the coefficients; numpy, full-batch Adam, 20 000 steps, 1 s.
2. **A Fourier neural operator on the index grid** (`train_operator.py`,
   `train_energy.py`): neuraloperator FNO, 48 x 24 Fourier modes, 32 channels,
   4 layers, 2.6 million parameters; inputs log rho and L as constant fields, the
   node coordinates (asinh-scaled) and the active mask; outputs v and psi. Trained
   on the GPU (MPS) first with a nodal mean-squared error (150 epochs on every
   second node, then 30 on the full grid; batch 2, Adam, cosine schedule from
   2e-3; 46 min), then fine-tuned with an energy loss (`energy.py`: the
   finite-difference energy of the potential's error and the complementary energy
   of the stream function's error, with the boundary values imposed inside the
   network's output).

```
uv run --project ../../../.. python make_data.py 4          # reference solutions (certsim env)
uv run --project ../../../.. python layout.py
uv run --project ../../../.. python reduced_basis.py        # writes results_reduced.json
uv run --project ../e3b_fno python train_operator.py 150 30 # PyTorch env
uv run --project ../e3b_fno python train_energy.py 60 40
uv run --project ../../../.. python certify_operator.py nodal    # writes results_nodal.json
uv run --project ../../../.. python certify_operator.py energy   # writes results_energy.json
```

## Results

Reduced-order surrogate, 12 in-range and 8 out-of-range tests:

| | In range | Out of range |
|---|---|---|
| Certificate contains the truth | 12 of 12 | 8 of 8 |
| Certified impedance width (relative) | 6.4e-7 - 1.9e-5 | 0.27 - 25% |
| eta | 8.8e-5 - 1.1e-3 | 6.0e-3 - 0.13 |
| Lesion depth (1000 V, 700 V/cm) bracketed on both sides | 6 of 12 | 0 of 8 |
| Depth bracket consistent with the reference's | 12 of 12 | 8 of 8 |
| Conformal interval (90%, half-width 1.6e-5) covers | 12 of 12 | 0 of 8 |

Neural operator trained with the nodal loss: every interval contains the truth,
but none is informative (widths of 10^3 relative and more). Its nodal errors are
small (median 3.6e-4 V, at most 8.9e-3 V on the first in-range test), yet the energy of its error is about
100 times that of the true field. The certificate grid is a tensor grid graded to
1 nm at the electrode corners, so graded lines cross the whole domain and more
than 9 000 active cells are thinner than 10 nm, with aspect ratios up to 3e6
(`diagnostics.py`). A
nodal error that differs between two nodes 1 nm apart puts a large spurious field
into such a cell. The solver's fields, and any combination of POD modes, are
consistent across these cells; a network with free nodal outputs is not.

Fine-tuning in the energy norm (`train_energy.py`, 60 coarse and 40 fine
epochs, 38 min) lowers the relative energy error on the training fields from
78 (potential) and 1 700 (stream function) to 45 and 27 (`eval_energy.py`), and eta on the
in-range tests from 3.0 - 6.8 to 0.78 - 1.1. The intervals still contain the
truth in all 20 tests but remain uninformative (widths of 100 times the
impedance and more). The limit is structural: a network whose nodal outputs are
free must make neighbouring values agree to about 1e-6 V across 1 nm cells,
about a hundred times below its typical nodal error, and no loss alone imposes
that. Fields built from solver-consistent modes satisfy it by construction.

| Operator | eta, in range | Certified width, in range | Impedance from its potential, error | Contains the truth |
|---|---|---|---|---|
| nodal loss | 3.0 - 6.8 | 1e3 - 3.5e3 | 98 - 99% | 20 of 20 |
| energy loss | 0.78 - 1.1 | 1.1e2 - 2.2e2 | 98% | 20 of 20 |
| reduced-order surrogate, for comparison | 8.8e-5 - 1.1e-3 | 6.4e-7 - 1.9e-5 | 1e-7 - 2.4e-6 | 20 of 20 |

Conformal intervals (90%, score relative to the prediction) around the operators'
impedance are 1.8 - 3.6 times the impedance wide inside the range, narrower than
the certified ones, and cover 12 (nodal loss) and 10 (energy loss) of the 12
in-range tests; outside the range they cover none of the 8. The majorant gives no bound for any surrogate here.

## Projection onto the POD basis

`project_operator.py` projects each operator prediction (boundary values imposed)
onto the first K POD modes of the 64 training solutions, the basis of the
reduced-order surrogate, by least squares on the free nodal values, and
certifies the result. The reference fields projected the same way show the best
a K-mode basis allows. Writes `results_projected.json`.

```
uv run --project ../../../.. python project_operator.py
```

| | eta, in range | Certified width, in range | Contains the truth |
|---|---|---|---|
| operator, nodal loss, K = 12 | 0.037 - 0.18 | 0.16 - 4.0 | 20 of 20 |
| operator, nodal loss, K = 24 / 48 | 0.34 - 0.77 | 5.7 - 68 | 20 of 20 |
| operator, energy loss, K = 12 | 15 - 20 | 1.6e4 - 9.6e4 | 20 of 20 |
| reference fields, K = 12 | | 3.2e-7 - 2.8e-6 | 20 of 20 |

Projection makes the fields consistent across the thin cells and narrows the
nodal-loss operator's intervals by two to four orders of magnitude, but every
in-range interval stays at least 16% of the impedance wide. On an in-range test
the reference coefficients fall below 1e-3 after the fourth mode, while the
operator's errors put about 1e-2 into every mode, in directions the solution
barely uses but that carry energy; more modes make it worse. The reduced-order
surrogate's network predicts the coefficients directly, each standardized by its
own spread.
