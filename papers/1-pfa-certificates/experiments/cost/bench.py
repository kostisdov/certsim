"""E4: cost of checking against cost of solving, as the grid grows (certsim environment).

For the paper 1 lesion-map problem at degree 3, on grids refined systematically
from 0.5 to 0.0625 mm, time the solve, the rigorous vectorized checker
(certsim.fastcheck, double-precision ball arithmetic), the non-rigorous float64
prototype (fast_check.py), and, up to 0.125 mm, the arb checker
(certsim.checker.check). Report how closely the vectorized checkers reproduce
the arb bounds.

    uv run python bench.py
"""

import json
import time
from pathlib import Path

import numpy as np

from certsim.checker import check
from certsim.fastcheck import fast_check as ball_check
from certsim.solver import default_grid, solve
from certsim.spec import Box, default_spec, lesion_tiles
from fast_check import fast_check as float_check

HERE = Path(__file__).resolve().parent
H_FINE = (0.5e-3, 0.25e-3, 0.125e-3, 0.0625e-3)
ARB_UP_TO = 0.125e-3


def timed(f, *a):
    t0 = time.perf_counter()
    out = f(*a)
    return out, time.perf_counter() - t0


def main():
    spec = default_spec(voxels=(), tiles=lesion_tiles(r_max=6e-3, depth=8e-3, step=0.25e-3),
                        regions=(Box(0.0, 5e-3, -10e-3, -6e-3),))
    out = []
    for h in H_FINE:
        s = h / 0.25e-3
        r, z = default_grid(spec, h_fine=h, h0=1e-9, kappa=0.25 * s, growth=0.08 * s)
        cert, t_solve = timed(solve, spec, r, z, 3)
        ball, t_ball = timed(ball_check, spec, cert)
        flt, t_float = timed(float_check, spec, cert)
        row = dict(h_mm=h * 1e3, cells=(len(r) - 1) * (len(z) - 1), unknowns=int(cert.v.size),
                   solve_s=t_solve, ball_s=t_ball, float_s=t_float,
                   ball_Z=[ball.impedance.lo, ball.impedance.hi], ball_eta=[ball.eta.lo, ball.eta.hi])
        if h >= ARB_UP_TO:
            rep, row["arb_s"] = timed(check, spec, cert)
            row["arb_Z"] = [rep.impedance.lo, rep.impedance.hi]
            row["arb_eta"] = rep.eta.hi
            row["ball_eta_ratio"] = ball.eta.hi / rep.eta.hi
            row["ball_tiles_overlap"] = all(a.lo <= b.hi and b.lo <= a.hi
                                            for a, b in zip(ball.tile_rms_field, rep.tile_rms_field))
            ends = np.array([[iv.lo, iv.hi] for iv in rep.tile_rms_field])
            row["float_tile_max_rel_diff"] = float(np.max(np.abs(np.array(flt["tiles"]) - ends)
                                                          / np.maximum(np.abs(ends), 1e-300)))
        out.append(row)
        arb = (f"arb {row['arb_s']:.1f} s; ball eta / arb eta {row['ball_eta_ratio']:.7f}, "
               f"tiles overlap {row['ball_tiles_overlap']}") if "arb_s" in row else "arb skipped"
        print(f"h {h * 1e3:6.4f} mm  {row['unknowns']:8d} unknowns  solve {t_solve:6.1f} s  "
              f"rigorous vectorized {t_ball:5.2f} s ({t_ball / t_solve:.3f} of solve)  "
              f"float {t_float:5.2f} s  {arb}")
        (HERE / "bench.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
