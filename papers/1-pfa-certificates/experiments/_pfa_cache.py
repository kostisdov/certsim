"""Solve and check the lesion-map problem once, and cache the result."""

import pickle
import time
from pathlib import Path

from certsim.checker import check
from certsim.solver import Certificate, default_grid, solve
from certsim.spec import Box, default_spec, lesion_tiles

RESULTS = Path(__file__).resolve().parents[3] / "results"  # shared, at the repository root
STEP = 0.25e-3


def map_spec():
    return default_spec(
        voxels=(),
        tiles=lesion_tiles(r_max=6e-3, depth=8e-3, step=STEP),
        regions=(Box(0.0, 5e-3, -10e-3, -6e-3),),
    )


def load():
    """(spec, certificate, report), solving and checking on first use."""
    spec = map_spec()
    RESULTS.mkdir(exist_ok=True)
    cert_path, rep_path = RESULTS / "pfa_map_cert.npz", RESULTS / "pfa_map_report.pkl"
    if cert_path.exists() and rep_path.exists():
        return spec, Certificate.load(cert_path), pickle.loads(rep_path.read_bytes())
    t0 = time.perf_counter()
    r, z = default_grid(spec)
    cert = solve(spec, r, z, degree=3)
    rep = check(spec, cert)
    print(f"solved and checked in {time.perf_counter() - t0:.0f} s; cached in {RESULTS}")
    cert.save(cert_path)
    rep_path.write_bytes(pickle.dumps(rep))
    return spec, cert, rep
