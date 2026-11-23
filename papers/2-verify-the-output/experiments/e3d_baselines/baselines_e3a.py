"""Baselines for the E3a surrogate: split conformal intervals and the functional
majorant, next to the certificate (certsim env).

The surrogate is rebuilt exactly as in ../e3_surrogate/surrogate.py (same
training data, seed, and optimizer), and its test predictions are checked
against that experiment's results before use. Calibration uses 20 new members
of the training range with reference solves.

    uv run --project ../../../.. python baselines_e3a.py
"""

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
E3A = HERE.parent / "e3_surrogate"
sys.path.insert(0, str(E3A))
import majorant  # noqa: E402
from certsim.fastcheck import fast_check  # noqa: E402
from certsim.solver import default_grid, solve  # noqa: E402
from surrogate import DEGREE, MLP, MODES, TEST_IN, TEST_OUT, TRAIN, Layout, spec_for  # noqa: E402

ALPHA = 0.1


def reference(rho):
    spec = spec_for(rho)
    r, z = default_grid(spec_for(2.8))
    rep = fast_check(spec, solve(spec, r, z, degree=DEGREE))
    return 0.5 * (rep.impedance.lo + rep.impedance.hi)


def build():
    spec0 = spec_for(2.8)
    r, z = default_grid(spec0)
    lay = Layout(spec0, r, z)
    d = np.load(E3A / "snapshots.npz")
    V, P, Z = d["V"], d["P"], d["Z"]

    def pod(X):
        mean = X.mean(0)
        _, _, Vt = np.linalg.svd(X - mean, full_matrices=False)
        return mean, Vt[:MODES], (X - mean) @ Vt[:MODES].T

    v_mean, v_basis, v_coef = pod(V)
    p_mean, p_basis, p_coef = pod(P)
    targets = np.hstack([v_coef, p_coef, np.log(Z)[:, None]])
    mu, sd = targets.mean(0), targets.std(0) + 1e-30
    x_train = np.log(TRAIN)[:, None]
    net = MLP(1, targets.shape[1], width=16)
    net.fit((x_train - x_train.mean()) / x_train.std(), (targets - mu) / sd)

    def predict(rho):
        x = (np.log([[rho]]) - x_train.mean()) / x_train.std()
        y = net.forward(x)[0][0] * sd + mu
        cert = lay.certificate(r, z, v_mean + y[:MODES] @ v_basis, p_mean + y[MODES:2 * MODES] @ p_basis)
        return cert, float(np.exp(y[-1]))

    return r, z, predict


def conformal_q(scores):
    s = np.sort(scores)
    n = len(s)
    return float(s[min(n - 1, int(np.ceil((n + 1) * (1 - ALPHA))) - 1)])


def main():
    r, z, predict = build()
    published = json.loads((E3A / "results.json").read_text())["tests"]
    for rho in (1.8, 0.7, 10.0):
        _, zp = predict(rho)
        ref = published[f"{rho:g}"]["Z_pred"]
        assert abs(zp - ref) <= 1e-9 * ref, (rho, zp, ref)
    print("rebuilt surrogate reproduces the published predictions")

    rng = np.random.default_rng(3)
    calib = np.exp(rng.uniform(np.log(1.7), np.log(3.75), 20))
    with ProcessPoolExecutor(4) as pool:
        truth = list(pool.map(reference, calib))
    s_head, s_field = [], []
    for rho, zt in zip(calib, truth):
        cert, zp = predict(rho)
        zf = majorant.evaluate(spec_for(rho), r, z, cert.v)["Z_point"]
        # nonconformity scores relative to the prediction, so that z * (1 +- q) is the conformal interval
        s_head.append(abs(zp - zt) / zp)
        s_field.append(abs(zf - zt) / zf)
    q_head, q_field = conformal_q(s_head), conformal_q(s_field)
    print(f"conformal half-widths at {1 - ALPHA:.0%}: scalar output {q_head:.2e}, field energy {q_field:.2e}")

    rows = []
    for rho in [*TEST_IN, *TEST_OUT]:
        rho = float(rho)
        pub = published[f"{rho:g}"]
        cert, zp = predict(rho)
        maj = majorant.evaluate(spec_for(rho), r, z, cert.v)
        zt = pub["Z_true"]
        zf = maj["Z_point"]
        rows.append(dict(rho=rho, range=pub["range"], Z_true=zt, Z_cert=pub["Z_cert"],
                         cert_rel_width=pub["bound_rel_width"],
                         head=dict(Z=zp, covers=bool(abs(zp - zt) <= q_head * zp)),
                         field=dict(Z=zf, covers=bool(abs(zf - zt) <= q_field * zf)),
                         majorant=maj))
        print(f"rho {rho:5.2f} ({pub['range']:3s})  certified width {pub['bound_rel_width']:.1e}  "
              f"conformal(head) covers {rows[-1]['head']['covers']}  conformal(field) covers {rows[-1]['field']['covers']}  "
              f"majorant M {maj['M']:.2e} (flux {maj['flux_term']:.1e}, Friedrichs {maj['friedrichs_term']:.1e})", flush=True)

    summary = {}
    for rg in ("in", "out"):
        sel = [x for x in rows if x["range"] == rg]
        summary[rg] = dict(n=len(sel), certified_contains=len(sel),
                           certified_width=[min(x["cert_rel_width"] for x in sel), max(x["cert_rel_width"] for x in sel)],
                           conformal_head_covers=sum(x["head"]["covers"] for x in sel),
                           conformal_field_covers=sum(x["field"]["covers"] for x in sel),
                           majorant_M=[min(x["majorant"]["M"] for x in sel), max(x["majorant"]["M"] for x in sel)])
    res = dict(alpha=ALPHA, calibration_rho=calib.tolist(), calibration_truth=truth,
               conformal_half_width={"head": q_head, "field": q_field}, summary=summary, tests=rows)
    (HERE / "results_e3a.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
