# Paper 2: verifying the outputs of PDE solvers

`certsim_paper2.tex` / `.pdf`. The unchanged checker of paper 1 applied to a
third-party finite-element package, to plausible bugs and solvers written by
language models, and to learned surrogates.

| Folder | Paper | Experiment | Environment |
|---|---|---|---|
| `experiments/e1_fenicsx/` | Section 4, Table 1, Fig. 2 | FEniCSx on the certificate grid (native certificate: its potential and its current), on its own gmsh meshes, and against a mesh study | pixi (FEniCSx) + certsim |
| `experiments/e2_llm_solvers/` | Section 5, Tables 2 and 3 | Five plausible bugs in a correct solver, and solvers written by language models from `SPEC.md` (the programs are stored) | certsim (+ Ollama to generate new local-model samples) |
| `experiments/e3_surrogate/` | Section 6.1, Table 5, Fig. 3 | Reduced-order surrogate with one input | certsim |
| `experiments/e3b_fno/` | Section 6.2, Table 6 | Fourier neural operator on a uniform grid, certified after the fact, with two controls | uv (PyTorch, neuraloperator) + certsim |
| `experiments/e3c_operator/` | Sections 6.1 and 6.3, Tables 5 and 7, Fig. 4 | Two inputs (conductivity ratio, electrode length): a reduced-order surrogate, and a Fourier neural operator on the certificate grid, trained with a nodal and then an energy loss, and projected onto the POD basis | certsim + uv (PyTorch, from e3b_fno) |
| `experiments/e3d_baselines/` | Section 6.4 | Conformal intervals and a functional majorant next to the certificate | certsim |

`make_figures.py` writes Figs. 2-4 from the result files (`uv run --project ../..
--group viz python make_figures.py`); Fig. 1 is drawn in the LaTeX source. Each
experiment folder has a README with its commands and results. Generated data
(`*.npz`, network weights) is not tracked; the scripts regenerate it. Retrained
networks give close but not bit-identical numbers.
