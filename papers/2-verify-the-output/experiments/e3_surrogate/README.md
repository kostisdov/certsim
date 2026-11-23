# E3: certifying a learned surrogate

A reduced-order surrogate of the paper 1 lesion-map problem as a function of the
conductivity ratio rho = sigma_blood / sigma_tissue: a POD basis (6 modes) of 17
training solutions with rho in [1.7, 3.75], and a small neural network (numpy,
Adam, two tanh layers of 16) mapping log(rho) to the POD coefficients of the
potential and of the stream function and to log(impedance).

The surrogate predicts only free nodal values; electrode and ground values, and
the stream function on the axis and the shaft, are exact. Every prediction is
therefore an admissible certificate, checked with the unchanged checker at the
cost of a forward pass and a check. A reference solve at each test rho gives the
true values.

```
uv run python surrogate.py     # 17 training solves, then 12 certified predictions
```

## Results

| rho | Range | Surrogate impedance (ohm) | Certified from its fields (ohm) | True (ohm) | True error of its impedance | Certified width (rel.) | eta (reference solve) | Tiles, median width |
|---|---|---|---|---|---|---|---|---|
| 1.8 | in | 134.629 | 134.81218 - 134.81226 | 134.81218 | 1.4e-3 | 6.0e-7 | 1.4e-4 (1.1e-5) | 21.6% |
| 2.0 | in | 122.856 | 123.98362 - 123.98622 | 123.98377 | 9.1e-3 | 2.1e-5 | 9.3e-4 (1.1e-5) | 139% |
| 2.4 | in | 106.856 | 106.87367 - 106.87368 | 106.87368 | 1.6e-4 | 2.1e-8 | 2.1e-5 (1.2e-5) | 3.1% |
| 2.8 | in | 93.861 | 93.95027 - 93.95029 | 93.95028 | 9.5e-4 | 1.7e-7 | 1.0e-4 (1.3e-5) | 14.9% |
| 3.1 | in | 86.437 | 86.15263 - 86.15277 | 86.15264 | 3.3e-3 | 1.7e-6 | 3.6e-4 (1.4e-5) | 52.6% |
| 3.6 | in | 75.567 | 75.69859 - 75.69864 | 75.69860 | 1.7e-3 | 5.8e-7 | 2.4e-4 (1.5e-5) | 34.7% |
| 0.7 | out | 140.108 | 260.13 - 299.03 | 263.518 | 0.47 | 0.15 | 7.0e-2 (7.0e-6) | 200% |
| 1.0 | out | 140.038 | 207.52 - 218.47 | 208.375 | 0.33 | 0.053 | 4.2e-2 (8.1e-6) | 200% |
| 1.3 | out | 140.108 | 172.61 - 175.04 | 172.771 | 0.19 | 0.014 | 2.2e-2 (9.1e-6) | 200% |
| 5.0 | out | 72.217 | 56.521 - 56.991 | 56.536 | 0.28 | 8.3e-3 | 3.1e-2 (1.7e-5) | 200% |
| 7.0 | out | 72.212 | 41.50 - 43.23 | 41.544 | 0.74 | 0.042 | 6.9e-2 (2.0e-5) | 200% |
| 10.0 | out | 72.224 | 29.67 - 32.91 | 29.732 | 1.4 | 0.11 | 1.1e-1 (2.4e-5) | 200% |

- Every certified interval contains the true impedance, inside and far outside
  the training range.
- The width reports the quality of each prediction: inside the range the fields
  are certified to 1e-8 - 1e-5; outside, eta grows by about three orders of
  magnitude and the tiles become undecided.
- The surrogate's own impedance output is refuted at every test point: 0.02-0.9%
  off inside the range, 19-140% outside. Its fields are far more accurate than
  its scalar output, so the quantity is best read from the certified fields.
  The scalar output's in-range error partly reflects this small network trained
  on 17 points.

## Note on an earlier run

The first version predicted the stream function's shaft value separately from
the neighbouring free values. The small mismatch produced spurious currents next
to the shaft, eta of order 1 to 100, and useless upper bounds, while the
potential (and so the lower bounds) were already accurate. Predicting the shaft
value inside the same POD vector fixed it.
