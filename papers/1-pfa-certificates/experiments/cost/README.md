# Table 1: the cost of checking

`bench.py` solves the monopolar lesion-map problem at degree 3 on grids refined
systematically (0.5 to 0.0625 mm in the lesion zone) and times the solve, the
vectorized checker (`certsim.fastcheck`), and, up to 0.125 mm, the 256-bit
checker (`certsim.checker`). It also times `fast_check.py`, a non-rigorous
float64 prototype, to measure what the rounding control costs. Results go to
`bench.json`.

```
uv run --project ../../../.. python bench.py
```
