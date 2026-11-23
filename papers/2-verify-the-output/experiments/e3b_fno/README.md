# E3b: certifying a standard Fourier neural operator after the fact

A Fourier neural operator (FNO, `neuraloperator` 2.0, PyTorch 2.14 on the GPU via
MPS) is trained to map the conductivity ratio rho to the potential on a uniform
256 x 512 grid over the whole domain (0.2 mm spacing). Unlike the E3a surrogate,
it is not admissible by construction, so each prediction needs post-processing
before it can be checked:

1. Interpolate the predicted field onto the nodes of the certificate grid
   (degree 3, paper 1 grid).
2. Make it admissible: give the electrode and ground nodes their exact values,
   either by overwriting them (naive lift) or by first spreading the electrode
   mismatch linearly over one FNO grid spacing (smooth lift).
3. Supply the current by the global stream-function fit
   (`certsim.solver.certify_potential`).
4. Check with the unchanged checker (`certsim.fastcheck`).

```
uv run --project ../../../.. python make_data.py      # 32 training and 12 test fields (certsim env)
uv run python train_fno.py                            # this folder's env: torch + neuraloperator
uv run --project ../../../.. python certify_fno.py    # writes results.json
```

Training: 32 ratios in [1.7, 3.75], 400 epochs, 506 s on MPS; final training
loss 4e-7 (about 0.6 mV RMS on a 1 V field). Prediction: 16 ms. Post-processing:
the current fit takes 5.2 s (about as much as a degree-3 primal solve), the check 1.0 s.

## Results (12 test ratios, six outside the training range)

| Source, lift | Truth inside | eta | Certified lower bound | Certified upper bound | Tiles |
|---|---|---|---|---|---|
| FNO, naive | 12 of 12 | 8.8 - 42 | 0.0006 - 0.013 ohm | equals the true value to 4 decimals | undecided |
| FNO, smooth | 12 of 12 | 2.7 - 11 | 0.009 - 0.14 ohm | same | undecided |
| control: reference sampled on the same uniform grid, smooth | 12 of 12 | 10 - 38 | 0.0007 - 0.010 ohm | same | undecided |
| control: zero field (the lift alone), smooth | 12 of 12 | 17 - 65 | 0.0002 - 0.0034 ohm | same | undecided |

(A reference solve on the certificate grid gives eta of about 1e-5.)

- Every certificate is valid, but none is informative on the side the input
  determines: the intervals are [about 0.1 ohm or less, truth].
- The upper bound is the adapter's, not the input's: even a zero field, with no
  solver at all, receives an upper bound equal to the true value to 4 decimals.
  The global current fit is a dual solve and costs about as much as a primal solve.
- The bottleneck is the route, not the FNO: the exact reference solution,
  sampled on the same 0.2 mm uniform grid and passed through the same
  interpolation and lift, receives an eta of 10 - 38, no smaller than the FNO's.
  (Its shaft region is sampled as 0 V, which makes this control harsh, but no
  sampling on a uniform grid of this spacing resolves the singular field at the
  electrode edge.) The certificate therefore says that this route cannot certify
  the near field, whatever the model; it does not by itself localize the FNO's
  errors. `fno_errors.py` compares them with the reference field on the same
  grid: at rho = 2.8, 0.15 mV mean error more than 0.5 mm from the electrode, but
  16 mV mean and 64 mV maximum within 0.5 mm of it (fno_errors.json).
- The smooth lift reduces eta about fourfold over the naive one.

Conclusion: outputs on a uniform grid can be certified validly but not usefully
for this singular problem. Certifiable surrogates should emit admissible fields
on the certificate grid, as in E3a.
