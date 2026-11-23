# E2: solvers written by language models

Each model receives `SPEC.md` (physics, geometry, boundary conditions and a file
interface, but not the boundary-node lists) and returns a program. Every program
is run at three systematically refined resolutions (1, 0.5 and 0.25 mm; spacing,
corner grading and far-field growth scaled together), its potential is certified
as is with the unchanged checker (degree 1), and its claimed impedances feed the
paper 1 GCI.

```
uv run python generate.py qwen3.8:27b-mlx 1 2 0.7   # local model, seeds 1-2
uv run python run_solvers.py                         # writes results.json
```

| Sample | Model, route | Program |
|---|---|---|
| `qwen3.8_27b-mlx/001` | Qwen 3.8 27B (MLX, local via Ollama), seed 1, T = 0.7, thinking off | 542 lines |
| `qwen3.8_27b-mlx/002` | same, seed 2 | 814 lines |
| `claude-opus-5-5/001` | Claude Opus 5.5, fresh agent with no prior context and no tools, 2026-10-28 | 88 lines |

## Results

True impedance (paper 1 certificate): 93.9502743 - 93.9502758 ohm.

| Sample | Mesh study on its own claims | Certificate, finest grid | Correct to 0.1% |
|---|---|---|---|
| Claude Opus 5.5 | converging (R 0.40, order 1.33); band 93.798 - 94.002 ohm, contains the truth | **certified**: claim 93.8997 ohm proven to 0.1% (93.8997 - 93.9782), eta 3.0e-3 | yes (0.05% low) |
| Qwen sample 2 | **converging** (R 0.44, order 1.20); band 5.5e-5 - 7.1e-5 ohm | **unsupported**: 6.3e-5 - 93.98 ohm proves nothing, eta 126 (40 000 times Claude's) | **no: off by a factor of 1.5 million** |
| Qwen sample 1 | none (no valid output) | crashed: singular system, NaN potential, impedance 0, exit code 0 | no |

Qwen sample 2 never solves the equation: every conducting node holds exactly
0 V or 1 V, so the field drops from the electrode value to zero within one cell.
Its boundary values are exactly right, so the checker accepts the potential as
admissible, and its claims change smoothly with the grid, so a mesh study
accepts the convergence. Only the certificate shows that the field is wrong: its
hypercircle gap is four orders of magnitude above that of a correct solver on
the same grids.

Claude's solver is a correct bilinear finite-element code. Its claim sits about
1e-9 relative below the rigorous lower bound because of floating-point rounding
in its current sum; the runner allows 1e-6 relative for such rounding.

## Mutation study: plausible bugs in the correct solver

`make_mutants.py` applies one realistic one-line change each to the Claude solver
(the change is recorded in each mutant's `request.json`); `run_solvers.py` runs
them like any other sample.

| Mutant | Its impedance (finest) | Mesh study on its claims | Certificate (finest) |
|---|---|---|---|
| top boundary left insulating instead of grounded | 94.16 ohm (0.2% off) | converging (R 0.40) | **rejected**: v not 0 on the ground |
| only the electrode's bottom face driven | 390.9 ohm | converging (R 0.31) | **rejected**: v differs from the electrode voltage |
| tissue and blood conductivities swapped | 136.7 ohm | converging (R 0.45) | **refuted**: 88.62 - 93.98 ohm |
| axisymmetric weight r missing | 0.393 ohm | converging (R 0.29) | **refuted**: 40.34 - 93.98 ohm |
| factor 2 pi missing in the current (field right, claim wrong) | 590.0 ohm | converging (R 0.40) | **refuted**: 93.90 - 93.98 ohm |

All five converge under refinement, so a mesh study accepts all five; the
certificate catches all five. The first is the hardest case for any check that
looks only at numbers: its impedance is within 0.2% of the truth. It breaks a
stated boundary condition, which the checker tests exactly.

For these potential-only solvers, a claim above the truth is refuted by the
upper bound, which comes from the adapter's dual solve (swapped conductivities,
missing 2 pi); a claim below it is refuted by the lower bound, the energy of the
mutant's own potential evaluated with the true physics (missing r).

## Verdicts

- crashed: failed, timed out, or wrote no valid output
- rejected: inadmissible potential (boundary values)
- refuted: claim outside the certified interval by more than 1e-6 relative
- unsupported: claim inside, but the interval does not prove it to 0.1%
- certified: the interval proves the claim to within 0.1%
