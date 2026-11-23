"""Certify the neural operator's predictions after projecting them onto a POD basis (certsim env).

The operator's nodal outputs are small in error but not consistent across the
certificate grid's thin cells, so its certificates are valid but uninformative
(certify_operator.py). Here each prediction, made admissible as before, is
projected (least squares on the free nodal values) onto the span of the first K
POD modes of the 64 training solutions, the basis of the reduced-order
surrogate, and then certified. The reference fields projected the same way show
the best a K-mode basis allows.

    uv run --project ../../../.. python project_operator.py      # writes results_projected.json
"""

import json
import sys
from pathlib import Path

import numpy as np

from certify_operator import admissible, certify
from common import L0, grid_for, masks_for, spec_for

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "e3_surrogate"))
from surrogate import Layout  # noqa: E402

MODES = (12, 24, 48)
VOLT, THRESH = 1000.0, 7e4
RUNS = {"nodal": "predictions.npz", "energy": "predictions_energy.npz"}


def basis(lay):
    feats = []
    for f in sorted((HERE / "data").glob("train_*.npz")):
        d = np.load(f)
        feats.append(lay.features(type("C", (), {"v": d["v"].astype(float), "psi": d["psi"].astype(float)})))
    out = {}
    for i, key in enumerate(("v", "p")):
        X = np.array([fv[i] for fv in feats])
        mean = X.mean(0)
        _, _, Vt = np.linalg.svd(X - mean, full_matrices=False)
        out[key] = (mean, Vt)
    return out


def project(B, fv, fp, k):
    (mv, Bv), (mp, Bp) = B["v"], B["p"]
    return mv + ((fv - mv) @ Bv[:k].T) @ Bv[:k], mp + ((fp - mp) @ Bp[:k].T) @ Bp[:k]


def row(spec, r, z, lay, fv, fp, z_true):
    cert = lay.certificate(r, z, fv, fp)
    rep, _ = certify(spec, r, z, cert.v, cert.psi)
    lo, hi = rep.impedance.lo, rep.impedance.hi
    return dict(contains_truth=bool(lo <= z_true <= hi), cert_rel_width=(hi - lo) / z_true, eta=rep.eta.hi,
                depth=rep.lesion_extent(VOLT, THRESH)["depth"])


def main():
    r0, z0 = grid_for(L0)
    lay = Layout(spec_for(2.8, L0), r0, z0)
    B = basis(lay)
    m = masks_for(3e-3)
    out = {}
    for tag, pred_file in RUNS.items():
        pred = np.load(HERE / pred_file)
        for name in ("test_in", "test_out"):
            for k_idx, fname in enumerate(pred[name + "_files"]):
                d = np.load(HERE / "data" / str(fname))
                rho, L = float(d["rho"]), float(d["L"])
                spec, (r, z) = spec_for(rho, L), grid_for(L)
                z_true = 0.5 * (float(d["Z_lo"]) + float(d["Z_hi"]))
                v, psi = admissible(spec, pred[name][k_idx, 0], pred[name][k_idx, 1], m)
                fv, fp = lay.features(type("C", (), {"v": v, "psi": psi}))
                ref = lay.features(type("C", (), {"v": d["v"].astype(float), "psi": d["psi"].astype(float)}))
                for k in MODES:
                    key = (tag, name, k)
                    rec = out.setdefault(key, [])
                    res = row(spec, r, z, lay, *project(B, fv, fp, k), z_true)
                    res.update(rho=rho, L=L)
                    if tag == "nodal":  # the reference projection does not depend on the operator
                        res["reference"] = row(spec, r, z, lay, *project(B, *ref, k), z_true)
                    rec.append(res)
                    print(f"{tag:6s} {name:8s} rho {rho:5.2f} L {L*1e3:4.2f}  K {k:2d}  width {res['cert_rel_width']:.1e}  "
                          f"eta {res['eta']:.1e}  truth inside {res['contains_truth']}"
                          + (f"  | reference projected: width {res['reference']['cert_rel_width']:.1e}"
                             if "reference" in res else ""), flush=True)
    summary = {}
    for (tag, name, k), rows in out.items():
        w = [x["cert_rel_width"] for x in rows]
        e = [x["eta"] for x in rows]
        s = dict(n=len(rows), contains=sum(x["contains_truth"] for x in rows), width=[min(w), max(w)], eta=[min(e), max(e)],
                 depth_bracketed=sum(x["depth"][1] is not None for x in rows))
        if "reference" in rows[0]:
            wr = [x["reference"]["cert_rel_width"] for x in rows]
            s["reference_width"] = [min(wr), max(wr)]
        summary[f"{tag}/{name}/K{k}"] = s
    (HERE / "results_projected.json").write_text(json.dumps(
        dict(summary=summary, rows={f"{t}/{n}/K{k}": v for (t, n, k), v in out.items()}), indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
