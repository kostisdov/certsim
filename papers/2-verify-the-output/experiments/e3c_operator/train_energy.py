"""Fine-tune the operator in the norm the certificate measures (PyTorch env of e3b_fno).

Starts from the operator trained with a nodal mean-squared error
(train_operator.py) and continues with an energy loss: the relative energy of the
potential's error plus the relative complementary energy of the stream
function's error, both with the boundary values imposed exactly inside the
network's output, as certify_operator.py imposes them. A small nodal term is
kept. The energies use the finite-difference weights of energy.py on each
member's own grid.

    uv run --project ../e3b_fno python train_energy.py [epochs_coarse epochs_fine]
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from neuralop.models import FNO

import energy
from train_operator import BATCH, DEVICE, HERE, inputs, load, z_nodes

torch.manual_seed(0)
SIG_T, DEGREE = 0.25, 3


def fields(out, stats, mk):
    """Network output -> admissible (v, psi) in physical units, differentiably."""
    v = out[:, 0] * stats["v_sd"]
    psi = out[:, 1] * stats["psi_sd"]
    v = torch.where(mk["v_fixed"], mk["v_value"], v)
    psi = torch.where(mk["psi_zero"], torch.zeros_like(psi), psi)
    sh = mk["shaft"]
    mean = (psi * sh).sum((-2, -1)) / sh.sum()
    psi = torch.where(sh, mean[:, None, None], psi)
    return v, psi


def edge_energy(f, wz, wr):
    return (wz * (f[:, 1:, :] - f[:, :-1, :]) ** 2).sum((-2, -1)) + (wr * (f[:, :, 1:] - f[:, :, :-1]) ** 2).sum((-2, -1))


def setup(lay, rho, L, Y, stride):
    act = lay["active"][::stride, ::stride]
    rn = lay["r_nodes"][::stride]
    rpsi = energy.psi_r_nodes(lay["r_nodes"][::DEGREE], DEGREE)[::stride]
    W = {k: [] for k in ("vz", "vr", "pz", "pr")}
    ref_v, ref_p = [], []
    for k, (a, b) in enumerate(zip(rho, L)):
        zn = z_nodes(lay, b)[::stride]
        vz, vr = energy.weights(rn, zn, act, SIG_T, SIG_T * a, "v")
        pz, pr = energy.weights(rpsi, zn, act, SIG_T, SIG_T * a, "psi")
        for key, w in zip(("vz", "vr", "pz", "pr"), (vz, vr, pz, pr)):
            W[key].append(w)
        ref_v.append(energy.energy(Y[k, 0, ::stride, ::stride].astype(float), vz, vr))
        ref_p.append(energy.energy(Y[k, 1, ::stride, ::stride].astype(float), pz, pr))
    W = {k: torch.tensor(np.stack(w), dtype=torch.float32).to(DEVICE) for k, w in W.items()}
    sub = {k: lay[k][::stride, ::stride] for k in ("active", "electrode", "ground", "axis", "shaft")}
    mk = {"v_fixed": torch.tensor(sub["electrode"] | sub["ground"] | ~sub["active"]).to(DEVICE),
          "v_value": torch.tensor(sub["electrode"].astype(np.float32)).to(DEVICE),
          "psi_zero": torch.tensor(sub["axis"] | ~sub["active"]).to(DEVICE),
          "shaft": torch.tensor(sub["shaft"] & ~sub["axis"]).to(DEVICE),
          "w": torch.tensor(sub["active"], dtype=torch.float32).to(DEVICE)}
    return W, torch.tensor(ref_v, dtype=torch.float32).to(DEVICE), torch.tensor(ref_p, dtype=torch.float32).to(DEVICE), mk


def run(model, X, Yt, Ys, W, ref_v, ref_p, mk, stats, epochs, lr, t0, label):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for epoch in range(epochs):
        perm = torch.randperm(len(X))
        tot = tv = tp = 0.0
        for k in range(0, len(X), BATCH):
            idx = perm[k:k + BATCH]
            out = model(X[idx])
            v, psi = fields(out, stats, mk)
            ev = edge_energy(v - Yt[idx, 0], W["vz"][idx], W["vr"][idx]) / ref_v[idx]
            ep = edge_energy(psi - Yt[idx, 1], W["pz"][idx], W["pr"][idx]) / ref_p[idx]
            l2 = (((out - Ys[idx]) ** 2) * mk["w"]).sum() / (mk["w"].sum() * 2 * len(idx))
            loss = (ev + ep).mean() + l2
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
            tv += ev.sum().item()
            tp += ep.sum().item()
        sched.step()
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"{label} epoch {epoch:4d}  loss {tot / len(X):.3e}  rel. energy v {tv / len(X):.3e}  "
                  f"psi {tp / len(X):.3e}  {time.perf_counter() - t0:.0f} s", flush=True)
    return tv / len(X), tp / len(X)


def main(epochs_coarse=60, epochs_fine=40):
    lay = dict(np.load(HERE / "layout.npz"))
    meta0 = json.loads((HERE / "train_meta.json").read_text())
    stats = {k: meta0[k] for k in ("lr_mu", "lr_sd", "L_mu", "L_sd", "v_sd", "psi_sd")}
    scale = np.array([stats["v_sd"], stats["psi_sd"]], dtype=np.float32)[None, :, None, None]
    _, rho, L, Y = load("train", lay)
    X = inputs(rho, L, lay, stats).to(DEVICE)
    Yt = torch.tensor(Y).to(DEVICE)
    Ys = torch.tensor(Y / scale).to(DEVICE)
    model = FNO(n_modes=(48, 24), hidden_channels=32, in_channels=X.shape[1], out_channels=2,
                n_layers=4, domain_padding=0.1).to(DEVICE)
    # operator.pt is written by train_operator.py in this folder, so a full load is safe
    model.load_state_dict(torch.load(HERE / "operator.pt", map_location=DEVICE, weights_only=False))
    t0 = time.perf_counter()
    s = slice(None, None, 2)
    W, rv, rp, mk = setup(lay, rho, L, Y, 2)
    run(model, X[:, :, s, s], Yt[:, :, s, s], Ys[:, :, s, s], W, rv, rp, mk, stats, epochs_coarse, 1e-3, t0, "coarse")
    del W
    W, rv, rp, mk = setup(lay, rho, L, Y, 1)
    ev, ep = run(model, X, Yt, Ys, W, rv, rp, mk, stats, epochs_fine, 1e-4, t0, "fine")
    t_train = time.perf_counter() - t0
    torch.save(model.state_dict(), HERE / "operator_energy.pt")

    model.eval()
    preds, t_pred = {}, []
    with torch.no_grad():
        for name in ("calib", "test_in", "test_out"):
            files, r_, L_, _ = load(name, lay)
            Xs = inputs(r_, L_, lay, stats).to(DEVICE)
            out = []
            for k in range(len(Xs)):
                t1 = time.perf_counter()
                out.append(model(Xs[k:k + 1]).cpu().numpy()[0] * scale[0])
                t_pred.append(time.perf_counter() - t1)
            preds[name] = np.stack(out)
            preds[name + "_files"] = np.array([f.name for f in files])
    np.savez(HERE / "predictions_energy.npz", **preds)
    meta = dict(stats, start="operator.pt (nodal loss)", epochs=[epochs_coarse, epochs_fine], train_seconds=t_train,
                predict_seconds=float(np.median(t_pred)), final_rel_energy_v=ev, final_rel_energy_psi=ep,
                device=DEVICE, n_train=len(X))
    (HERE / "train_meta_energy.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main(*[int(a) for a in sys.argv[1:3]])
