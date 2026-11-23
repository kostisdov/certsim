# certsim

Certified bounds for physics simulations. An untrusted solver returns its
answer together with a certificate. A small, independent checker verifies the
certificate against a written specification of the physics and turns it into
guaranteed error bounds for that run, rounding included. The checker never runs
or trusts the solver; it checks only what it receives.

This repository contains the code and the scripts that reproduce every number
in two papers:

> K. Dovelos, "Certificate-Based Verification of Pulsed Field Ablation Models,"
> [arXiv:2611.18148](https://arxiv.org/abs/2611.18148), 2026. Source and PDF in
> [`papers/1-pfa-certificates/`](papers/1-pfa-certificates/); tag `v1.0-preprint`.

> K. Dovelos, "Verifying the Outputs of PDE Solvers: From Finite Elements to Neural
> Surrogates," 2026. Source and PDF in
> [`papers/2-verify-the-output/`](papers/2-verify-the-output/); tag `v2.0-preprint`.

Each tag is the version described in its paper.

## Install

Python 3.13 and [uv](https://docs.astral.sh/uv/):

```
uv sync
uv run pytest            # 59 tests, about 8 minutes
```

## Example

```python
from certsim.fastcheck import fast_check
from certsim.solver import default_grid, solve
from certsim.spec import default_spec, lesion_tiles

spec = default_spec(voxels=(), tiles=lesion_tiles(r_max=6e-3, depth=8e-3, step=0.25e-3))
r, z = default_grid(spec)
cert = solve(spec, r, z, degree=3)        # untrusted: potential and stream function
rep = fast_check(spec, cert)              # or certsim.checker.check: 256-bit checker
print(rep.impedance)                      # certified impedance interval (ohm)
print(rep.lesion_extent(1000.0, 7e4))     # proven lesion depth and radius (m) at 1000 V, 700 V/cm
```

## Model

Axisymmetric quasi-static current flow, `div(sigma grad u) = 0`: myocardium in
`z < 0`, blood in `z > 0`, a flat-tipped metal electrode (radius 1.25 mm,
length 3 mm) followed by an insulating shaft, and a grounded box 50 mm away. A
bipolar variant adds a ring electrode and an insulating outer boundary, and a
blood gap can lift the tip off the tissue.

## What is checked

A certificate holds nodal values of a potential `v` and a stream function
`psi`, each a tensor polynomial of degree p on every grid cell (in `r, z` and in
`r^2, z`). The checker makes two tests:

1. **Specification check.** `v` takes the electrode and ground values at every
   boundary node, and `psi` is constant on every insulating wall, so its curl is
   an exactly charge-conserving current. A certificate that fails is rejected.
2. **Numerical verification.** Every cell integral is evaluated exactly in ball
   arithmetic. The Prager-Synge hypercircle and the Dirichlet and Thomson
   principles then bound the impedance and the RMS field of every lesion-map
   tile, which is proven lethal, proven spared, or undecided.

The bounds hold for the specified model, whatever solver produced the
certificate. Solvers with deliberately wrong physics (`solve(..., bug=...)`)
have their claims refuted or left unsupported.

## Layout

| Path | Contents |
|---|---|
| `src/certsim/spec.py` | The specification of the physics (trusted) |
| `src/certsim/checker.py` | The 256-bit checker (trusted) |
| `src/certsim/fastcheck.py` | The vectorized checker, double-precision ball arithmetic (trusted, under the assumptions stated in Section 3.4 of the paper) |
| `src/certsim/ranges.py` | Bounds over a range of conductivity ratios (trusted) |
| `src/certsim/solver.py` | The finite-element solver and the stream-function adapter (untrusted) |
| `src/certsim/gci.py`, `viz.py` | Grid convergence index; plotting helpers |
| `tests/` | Test suite, including soundness tests with perturbed and wrong-physics certificates |
| `papers/1-pfa-certificates/` | Paper 1, its figures, and `experiments/` |
| `papers/2-verify-the-output/` | Paper 2, its figures, and `experiments/` (FEniCSx, LLM-written solvers, surrogates) |
| `results/` | Logs of the reference run; the scripts write their caches and figures here |

The trusted base is `spec.py`, `checker.py` or `fastcheck.py`, `ranges.py`,
python-flint and NumPy.

## Reproducing paper 1

From the repository root (`E = papers/1-pfa-certificates/experiments`). Times
are for a MacBook Pro (M5 Pro); the logs of a full run from scratch are in
`results/`. Solves differ from run to run at rounding level (about 1e-12
relative in the bounds), which every digit reported in the paper absorbs.

| Paper | Command | Time |
|---|---|---|
| Section 6.1, Fig. 3 | `uv run python $E/pfa_lesion_map.py` | 30 s |
| Figs. 3 and 4 | `uv run --group viz python $E/pfa_figures.py` | after the above |
| Section 6.2, Table 2 | `uv run python $E/pfa_ranges.py` | 10 min |
| Section 6.2, five certificates | `uv run python $E/pfa_ranges.py 5` | 3 min |
| Section 6.3, Table 3 | `uv run python $E/gci_comparison.py` | 5 min |
| Section 6.4, Tables 4-6, Fig. 5 | `uv run --group viz python $E/meckes_comparison.py` | 20 min |
| One-off numbers in Sections 5 and 6 | `uv run python $E/paper_checks.py` (after the range scripts) | 6 min |
| Table 1 (cost of checking) | `cd $E/cost && uv run --project ../../../.. python bench.py` | 8 min |
| Table 7 (FEniCSx) | see [`$E/fenicsx/README.md`](papers/1-pfa-certificates/experiments/fenicsx/README.md) | 1 min |

Figs. 1 and 2 are drawn in the LaTeX source. `pfa_feasibility.py` runs the
earlier voxel-mean (adjoint) method, and `pfa_view.py` writes a 3D view for the
Rerun viewer (`--group viz`).

## Reproducing paper 2

See [`papers/2-verify-the-output/README.md`](papers/2-verify-the-output/README.md):
one folder per experiment, each with its commands. Besides uv, the FEniCSx
experiment needs [pixi](https://pixi.sh), and the neural operators use the
PyTorch environment in `experiments/e3b_fno` (about 1.5 hours of training on an
Apple M5 Pro GPU). A full run from scratch reproduces the paper's results files,
up to rounding in the last digits.

## Licence

Apache License 2.0, see [LICENSE](LICENSE). Copyright 2026 Konstantinos Dovelos.
