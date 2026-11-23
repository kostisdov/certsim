"""Two-input reduced-order surrogate on the certificate grid (certsim env).

The E3a construction, extended from one input (rho) to two (rho and the
electrode length L), on the same 64 training members as the neural operator:
a proper orthogonal decomposition of the free nodal values of the potential and
of the stream function (with the shaft value inside the stream-function vector),
and a small network that maps (log rho, L) to the POD coefficients. Boundary
values are exact, so every prediction is a certificate. Tested on the same 12
in-range and 8 out-of-range members, with the same conformal and majorant
baselines as the operator.

    uv run --project ../../../.. python reduced_basis.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

from certsim.fastcheck import fast_check
from common import DEGREE, L0, grid_for, spec_for

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "e3_surrogate"))
sys.path.insert(0, str(HERE.parent / "e3d_baselines"))
import majorant  # noqa: E402
from surrogate import MLP, Layout  # noqa: E402

MODES = 12
ALPHA = 0.1
VOLT, THRESH = 1000.0, 7e4


def _overlap(a, b):
    """Two certified depth brackets (reached, not_beyond; None = open) are consistent if they intersect."""
    hi_a = np.inf if a[1] is None else a[1]
    hi_b = np.inf if b[1] is None else b[1]
    return bool(a[0] <= hi_b and b[0] <= hi_a)


def load(name, lay):
    files = sorted((HERE / "data").glob(f"{name}_*.npz"))
    rows = []
    for f in files:
        d = np.load(f)
        v, psi = d["v"].astype(float), d["psi"].astype(float)
        cert = type("C", (), {"v": v, "psi": psi})
        fv, fp = lay.features(cert)
        rows.append(dict(file=f.name, rho=float(d["rho"]), L=float(d["L"]), v=fv, p=fp,
                         Z_true=0.5 * (float(d["Z_lo"]) + float(d["Z_hi"])), ref_v=v, ref_psi=psi))
    return rows


def main():
    r0, z0 = grid_for(L0)
    lay = Layout(spec_for(2.8, L0), r0, z0)
    train = load("train", lay)
    V = np.array([t["v"] for t in train])
    P = np.array([t["p"] for t in train])

    def pod(X):
        mean = X.mean(0)
        _, sv, Vt = np.linalg.svd(X - mean, full_matrices=False)
        captured = float(np.sum(sv[:MODES] ** 2) / np.sum(sv ** 2))
        return mean, Vt[:MODES], (X - mean) @ Vt[:MODES].T, captured

    v_mean, v_basis, v_coef, v_cap = pod(V)
    p_mean, p_basis, p_coef, p_cap = pod(P)
    targets = np.hstack([v_coef, p_coef])
    mu, sd = targets.mean(0), targets.std(0) + 1e-30
    x = np.array([[np.log(t["rho"]), t["L"]] for t in train])
    xm, xs = x.mean(0), x.std(0)
    net = MLP(2, targets.shape[1], width=32)
    t0 = time.perf_counter()
    loss = net.fit((x - xm) / xs, (targets - mu) / sd)
    t_train = time.perf_counter() - t0
    print(f"POD: {MODES} modes capture {v_cap:.10f} (v) and {p_cap:.10f} (psi) of the snapshot energy; "
          f"network trained in {t_train:.0f} s, loss {loss:.2e}", flush=True)

    def predict(rho, L):
        y = net.forward(((np.array([[np.log(rho), L]]) - xm) / xs))[0][0] * sd + mu
        r, z = grid_for(L)
        return lay.certificate(r, z, v_mean + y[:MODES] @ v_basis, p_mean + y[MODES:] @ p_basis), r, z

    out = {}
    for name in ("calib", "test_in", "test_out"):
        rows = []
        for t in load(name, lay):
            spec = spec_for(t["rho"], t["L"])
            t1 = time.perf_counter()
            cert, r, z = predict(t["rho"], t["L"])
            t_pred = time.perf_counter() - t1
            rep = fast_check(spec, cert)
            maj = majorant.evaluate(spec, r, z, cert.v)
            lo, hi = rep.impedance.lo, rep.impedance.hi
            zt = t["Z_true"]
            widths = np.array([iv.rel_width for iv in rep.tile_rms_field])
            empty = np.zeros((0, *cert.v.shape))
            ref = fast_check(spec, type(cert)(r, z, t["ref_v"], t["ref_psi"], empty, empty, DEGREE, {}))
            rows.append(dict(rho=t["rho"], L=t["L"], Z_true=zt, Z_cert=[lo, hi], contains_truth=bool(lo <= zt <= hi),
                             cert_rel_width=(hi - lo) / zt, eta=rep.eta.hi, tile_median=float(np.median(widths)),
                             depth=rep.lesion_extent(VOLT, THRESH)["depth"], depth_ref=ref.lesion_extent(VOLT, THRESH)["depth"],
                             Z_point=maj["Z_point"], point_rel_error=abs(maj["Z_point"] - zt) / zt, predict_s=t_pred,
                             majorant=maj))
            print(f"{name:8s} rho {t['rho']:5.2f} L {t['L'] * 1e3:4.2f} mm  certified {lo:.5f}-{hi:.5f}  true {zt:.5f}  "
                  f"width {(hi - lo) / zt:.1e}  eta {rep.eta.hi:.1e}  tiles {np.median(widths) * 100:.1f}%  "
                  f"depth {rows[-1]['depth']} (ref {rows[-1]['depth_ref']})", flush=True)
        out[name] = rows

    # nonconformity score relative to the prediction, so that Z_point * (1 +- q) is the conformal interval
    scores = np.sort([abs(row["Z_point"] - row["Z_true"]) / row["Z_point"] for row in out["calib"]])
    n = len(scores)
    q = float(scores[min(n - 1, int(np.ceil((n + 1) * (1 - ALPHA))) - 1)])
    summary = {}
    for name in ("test_in", "test_out"):
        rows = out[name]
        for row in rows:
            row["conformal_covers"] = bool(abs(row["Z_point"] - row["Z_true"]) <= q * row["Z_point"])
        summary[name] = dict(n=len(rows), certified_contains=sum(x["contains_truth"] for x in rows),
                             certified_width=[min(x["cert_rel_width"] for x in rows), max(x["cert_rel_width"] for x in rows)],
                             eta=[min(x["eta"] for x in rows), max(x["eta"] for x in rows)],
                             tile_median=[min(x["tile_median"] for x in rows), max(x["tile_median"] for x in rows)],
                             depth_bracketed=sum(x["depth"][1] is not None for x in rows),
                             depth_consistent=sum(_overlap(x["depth"], x["depth_ref"]) for x in rows),
                             conformal_covers=sum(x["conformal_covers"] for x in rows),
                             point_error=[min(x["point_rel_error"] for x in rows), max(x["point_rel_error"] for x in rows)],
                             majorant_M=[min(x["majorant"]["M"] for x in rows), max(x["majorant"]["M"] for x in rows)])
    res = dict(modes=MODES, captured=[v_cap, p_cap], train_seconds=t_train, train_loss=loss, conformal_q=q,
               alpha=ALPHA, summary=summary, **out)
    (HERE / "results_reduced.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({"conformal_q": q, "summary": summary}, indent=1))


if __name__ == "__main__":
    main()
