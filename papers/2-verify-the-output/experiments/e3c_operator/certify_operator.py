"""Certify the neural operator's predictions and compare with two baselines (certsim env).

Each prediction becomes a certificate by imposing its boundary values exactly:
the electrode and ground values of v, zero stream function on the axis, and one
constant on the shaft, the mean of the predicted shaft values. No current fit
and no solve are involved. The vectorized checker then bounds the impedance and
the lesion map.

Baselines, on the same predictions:
- split conformal prediction: intervals around the impedance computed from the
  predicted potential (its Dirichlet energy), calibrated on 24 held-out members
  of the training range, at 90% nominal coverage;
- the functional majorant of ../e3d_baselines/majorant.py, with the current
  recovered from the predicted potential.

    uv run --project ../../../.. python certify_operator.py nodal    # operator trained with a nodal loss
    uv run --project ../../../.. python certify_operator.py energy   # after fine-tuning in the energy norm
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

from certsim.fastcheck import fast_check
from certsim.solver import Certificate
from common import DEGREE, grid_for, masks_for, spec_for

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "e3d_baselines"))
import majorant  # noqa: E402

ALPHA = 0.1
VOLT, THRESH = 1000.0, 7e4  # lesion map at 1000 V and 700 V/cm


def admissible(spec, v_pred, psi_pred, m):
    v = v_pred.astype(float).copy()
    psi = psi_pred.astype(float).copy()
    v[~m["active"]] = 0.0
    v[m["electrode"]] = spec.tip_voltage
    v[m["ground"]] = 0.0
    psi[m["axis"] | ~m["active"]] = 0.0
    shaft = m["shaft"] & ~m["axis"]
    psi[shaft] = float(np.mean(psi[shaft]))
    return v, psi


def certify(spec, r, z, v, psi):
    empty = np.zeros((0, *v.shape))
    cert = Certificate(r, z, v, psi, empty, empty, DEGREE, {})
    t0 = time.perf_counter()
    rep = fast_check(spec, cert)
    return rep, time.perf_counter() - t0


RUNS = {"nodal": ("predictions.npz", "train_meta.json"),
        "energy": ("predictions_energy.npz", "train_meta_energy.json")}


def main(tag="energy"):
    pred_file, meta_file = RUNS[tag]
    pred = np.load(HERE / pred_file)
    m = masks_for(3e-3)
    out = {}
    for name in ("calib", "test_in", "test_out"):
        rows = []
        for k, fname in enumerate(pred[name + "_files"]):
            d = np.load(HERE / "data" / str(fname))
            rho, L = float(d["rho"]), float(d["L"])
            spec = spec_for(rho, L)
            r, z = grid_for(L)
            z_true = 0.5 * (float(d["Z_lo"]) + float(d["Z_hi"]))
            v, psi = admissible(spec, pred[name][k, 0], pred[name][k, 1], m)
            rep, t_check = certify(spec, r, z, v, psi)
            maj = majorant.evaluate(spec, r, z, v)
            ref_rep, _ = certify(spec, r, z, d["v"].astype(float), d["psi"].astype(float))
            ext = rep.lesion_extent(VOLT, THRESH)
            ext_ref = ref_rep.lesion_extent(VOLT, THRESH)
            widths = np.array([iv.rel_width for iv in rep.tile_rms_field])
            lo, hi = rep.impedance.lo, rep.impedance.hi
            rows.append(dict(rho=rho, L=L, Z_true=z_true, Z_cert=[lo, hi], contains_truth=bool(lo <= z_true <= hi),
                             cert_rel_width=(hi - lo) / z_true, eta=rep.eta.hi, tile_median=float(np.median(widths)),
                             depth=ext["depth"], depth_ref=ext_ref["depth"], check_s=t_check,
                             Z_point=maj["Z_point"], point_rel_error=abs(maj["Z_point"] - z_true) / z_true,
                             majorant=maj))
            print(f"{name:8s} rho {rho:5.2f} L {L * 1e3:4.2f} mm  certified {lo:.4f}-{hi:.4f}  true {z_true:.4f}  "
                  f"width {(hi - lo) / z_true:.1e}  eta {rep.eta.hi:.1e}  point err {rows[-1]['point_rel_error']:.1e}  "
                  f"depth {ext['depth']} (ref {ext_ref['depth']})  majorant M {maj['M']:.1e}", flush=True)
        out[name] = rows

    # split conformal on the point estimate, calibrated in range
    # nonconformity score relative to the prediction, so that Z_point * (1 +- q) is the conformal interval
    scores = np.sort([abs(row["Z_point"] - row["Z_true"]) / row["Z_point"] for row in out["calib"]])
    n = len(scores)
    q = float(scores[min(n - 1, int(np.ceil((n + 1) * (1 - ALPHA))) - 1)])
    for name in ("test_in", "test_out"):
        for row in out[name]:
            lo, hi = row["Z_point"] * (1 - q), row["Z_point"] * (1 + q)
            row["conformal"] = [lo, hi]
            row["conformal_covers"] = bool(lo <= row["Z_true"] <= hi)
    summary = {}
    for name in ("test_in", "test_out"):
        rows = out[name]
        summary[name] = dict(
            n=len(rows),
            certified_contains=sum(r_["contains_truth"] for r_ in rows),
            certified_width=[min(r_["cert_rel_width"] for r_ in rows), max(r_["cert_rel_width"] for r_ in rows)],
            conformal_covers=sum(r_["conformal_covers"] for r_ in rows),
            conformal_width=2 * q,
            point_error=[min(r_["point_rel_error"] for r_ in rows), max(r_["point_rel_error"] for r_ in rows)],
            eta=[min(r_["eta"] for r_ in rows), max(r_["eta"] for r_ in rows)],
            majorant_M=[min(r_["majorant"]["M"] for r_ in rows), max(r_["majorant"]["M"] for r_ in rows)])
    meta = json.loads((HERE / meta_file).read_text())
    res = {"meta": meta, "conformal_q": q, "alpha": ALPHA, "summary": summary, **out}
    (HERE / f"results_{tag}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({"conformal_q": q, "summary": summary}, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:2])
