"""Certify the FNO's predictions after the fact (run in the certsim environment).

Each predicted field is interpolated from the uniform grid onto the nodes of
the certificate grid (degree 3, paper 1 grid), the electrode and ground nodes
are given their exact values, which makes the potential admissible: either by
overwriting them (naive lift) or by spreading the electrode mismatch smoothly over
one surrogate grid spacing first (smooth lift),
and the current is supplied by the global stream-function fit
(certsim.solver.certify_potential), the post-hoc correction a surrogate needs.
The unchanged checker, in its vectorized form, then gives the bounds, which are
compared with the reference values of E3a. Two controls run through the same
pipeline: the reference solution sampled on the FNO's uniform grid, and a zero
field (the lift alone). Writes results.json.

    uv run --project ../../../.. python certify_fno.py
"""

import json
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from certsim.fastcheck import fast_check
from certsim.solver import certify_potential, default_grid
from certsim.spec import Box, default_spec, lesion_tiles, node_masks

HERE = Path(__file__).resolve().parent
SIGMA_T, P = 0.25, 3


def spec_for(rho):
    return default_spec(voxels=(), sigma_tissue=SIGMA_T, sigma_blood=rho * SIGMA_T,
                        tiles=lesion_tiles(r_max=6e-3, depth=8e-3, step=0.25e-3),
                        regions=(Box(0.0, 5e-3, -10e-3, -6e-3),))


def electrode_mismatch(spec, interp, RN, ZN, width):
    """Smooth correction that makes v equal the electrode voltage on the electrode.

    The mismatch g - v at the closest point of the electrode surface is spread
    linearly over a layer of the given width (about one surrogate grid spacing),
    instead of being imposed as a jump at the boundary nodes.
    """
    a, z0, z1 = spec.a, spec.gap, spec.gap + spec.L
    # closest point on the electrode surface: bottom face (r <= a, z = z0) or side (r = a, z0 <= z <= z1)
    pr = np.clip(RN, 0.0, a)
    pz = np.clip(ZN, z0, z1)
    on_side = RN > a
    pz = np.where(on_side, pz, z0)
    pr = np.where(on_side, a, pr)
    d = np.hypot(RN - pr, ZN - pz)
    mismatch = spec.tip_voltage - interp(np.stack([pr.ravel(), pz.ravel()], axis=1)).reshape(RN.shape)
    return mismatch * np.clip(1.0 - d / width, 0.0, 1.0)


def main(lift="smooth", source="fno"):
    """source: "fno" (the predictions), "reference" (the reference solution sampled on the same
    uniform grid: what any model on that grid could at best achieve), or "zero" (a zero field, so
    that the potential is the lift alone: what the certificate gives with no solver at all)."""
    pred = dict(np.load(HERE / "predictions.npz"))
    if source != "fno":
        data = np.load(HERE / "data.npz")
        assert np.allclose(data["test_rho"], pred["rho"])
        pred["v"] = data["test_v"] if source == "reference" else np.zeros_like(pred["v"])
    truth = json.loads((HERE.parent / "e3_surrogate" / "results.json").read_text())["tests"]
    spec0 = spec_for(2.8)
    r, z = default_grid(spec0)
    rn = np.append((r[:-1, None] + np.outer(np.diff(r), np.arange(P) / P)).ravel(), r[-1])
    zn = np.append((z[:-1, None] + np.outer(np.diff(z), np.arange(P) / P)).ravel(), z[-1])
    ZN, RN = np.meshgrid(zn, rn, indexing="ij")
    m = node_masks(spec0, r, z, P)
    results = {}
    for rho, field in zip(pred["rho"], pred["v"]):
        spec = spec_for(float(rho))
        t0 = time.perf_counter()
        interp = RegularGridInterpolator((pred["r"], pred["z"]), field)
        v = interp(np.stack([RN.ravel(), ZN.ravel()], axis=1)).reshape(ZN.shape)
        if lift == "smooth":
            v = v + electrode_mismatch(spec, interp, RN, ZN, width=pred["r"][1] - pred["r"][0])
        v[m["electrode"]] = spec.tip_voltage  # admissible: exact Dirichlet values
        v[m["ground"]] = 0.0
        v[~m["active"]] = 0.0
        t_lift = time.perf_counter() - t0
        cert = certify_potential(spec, r, z, v, P)
        t_fit = time.perf_counter() - t0 - t_lift
        rep = fast_check(spec, cert)
        t_check = time.perf_counter() - t0 - t_lift - t_fit
        true = truth[f"{float(rho):g}"]["Z_true"]
        lo, hi = rep.impedance.lo, rep.impedance.hi
        widths = np.array([iv.rel_width for iv in rep.tile_rms_field])
        o = dict(source=source, lift=lift, range=truth[f"{float(rho):g}"]["range"], Z_cert=[lo, hi], Z_true=true,
                 contains_truth=lo <= true <= hi, bound_rel_width=(hi - lo) / true, eta=rep.eta.hi,
                 tile_median=float(np.median(widths)), lift_s=t_lift, fit_s=t_fit, check_s=t_check)
        results[f"{float(rho):g}"] = o
        print(f"rho {rho:5.2f} ({o['range']:3s}) certified {lo:.4f} - {hi:.4f}  true {true:.4f}  "
              f"contains truth {o['contains_truth']}  width {o['bound_rel_width']:.1e}  eta {o['eta']:.1e}  "
              f"tiles {o['tile_median'] * 100:.0f}%  fit {t_fit:.1f} s, check {t_check:.1f} s")
    meta = {k: float(pred[k]) for k in ("train_seconds", "predict_seconds")}
    meta["device"] = str(pred["device"])
    return meta, results


if __name__ == "__main__":
    out = {}
    for source, lift in (("fno", "naive"), ("fno", "smooth"), ("reference", "smooth"), ("zero", "smooth")):
        print(f"== source: {source}, lift: {lift}")
        meta, out[f"{source} {lift}"] = main(lift, source)
    (HERE / "results.json").write_text(json.dumps({"meta": meta, "tests": out}, indent=1))
