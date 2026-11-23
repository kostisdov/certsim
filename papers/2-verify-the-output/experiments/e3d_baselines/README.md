# E3d: baselines for certified surrogate outputs

Two kinds of evidence a surrogate user could have instead of a certificate, on
the same predictions:

1. **Split conformal prediction** (90% nominal coverage) around the surrogate's
   impedance, calibrated on held-out members of the training range, with the
   score |Z_pred - Z_true| / Z_pred, so that Z_pred (1 +- q) is the interval. It holds on
   average over draws from the calibration distribution, not for a given input.
2. **A functional majorant** (Repin), the per-output bound that learned-majorant
   methods for neural solutions evaluate (`majorant.py`). For an admissible
   potential v and any current y with y.n = 0 on the insulating walls,
   |||u - v||| <= ||sigma grad v - y||_{1/sigma} + C_F/sqrt(sigma_min) ||div y||.
   Every vertical line through the monopolar domain ends on the electrode or the
   ground at both ends, so C_F = (H + Ztop)/pi = 31.8 mm rigorously. Here y is
   recovered from v by nodal averaging of sigma grad v (normal component zero on
   the axis and the shaft), the cheapest admissible choice; the integrals are
   evaluated in double precision with Gauss quadrature, as such baselines are.

```
uv run --project ../../../.. python baselines_e3a.py    # E3a surrogate; writes results_e3a.json (~3 min)
```

The E3c baselines run inside `../e3c_operator/certify_operator.py`.

## E3a surrogate (12 test ratios, 20 calibration ratios)

The surrogate is rebuilt exactly as in `../e3_surrogate` and reproduces its
published predictions.

| | In range (6) | Out of range (6) |
|---|---|---|
| Certificate contains the truth | 6 | 6 |
| Certified width (relative) | 2e-8 - 2e-5 | 0.8 - 15% |
| Conformal, scalar output (half-width 4.9e-3): covers | 5 | 0 |
| Conformal, impedance from the field energy (half-width 3.3e-7): covers | 5 | 0 |
| Majorant M (field norm about 0.1) | 2.0e3 - 4.0e3 | 1.1e3 - 1.1e4 |

- In range, the conformal interval around the field-energy estimate is narrower
  than some certified intervals; out of range it covers none of the six, where
  that estimate is off by 2.6e-4 to 1.3e-2.
- The majorant gives no bound: its flux term is about the certificate's eta
  (1e-5), but the Friedrichs term is 1e3 - 1e4. On the reference solution, 99% of
  ||div y||^2 sits in the two 1 nm cells at the electrode-shaft junction, where the
  potential behaves like sqrt(distance); cells more than 1 mm from both corners
  alone give a Friedrichs term of 1.5e-3, still a hundred times eta. An exactly
  divergence-free current, as the stream function provides, removes this term,
  and the majorant then reduces to the hypercircle bound.

`majorant_reference.py` evaluates the majorant on the solver's own solution and
writes the numbers quoted above (flux term, Friedrichs term, the share of
||div y||^2 in the two junction cells, the term of the cells far from the corners)
to `majorant_reference.json`.
