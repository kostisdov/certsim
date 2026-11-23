"""Train a Fourier neural operator that outputs both fields on the certificate grid
(PyTorch environment of e3b_fno).

The operator works on the index grid of the certificate: node (j, i) of the
degree-3 node array, whatever the physical spacing. Input channels: log(rho)
and L as constant fields, the node coordinates r and z(L) (asinh-scaled, since
the grid is graded down to 1 nm), and the active-node mask. Output channels:
the potential v and the stream function psi, standardized. The loss is the mean
squared error over active nodes. Boundary values are imposed afterwards, in
certify_operator.py, so every prediction becomes an admissible certificate.

    uv run --project ../e3b_fno python train_operator.py [epochs_coarse epochs_fine]
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from neuralop.models import FNO

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
torch.manual_seed(0)
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
BATCH, LR = 2, 2e-3
SCALE = 1e-3  # asinh scale for node coordinates [m]


def z_nodes(lay, L):
    """Node z-coordinates of the member with electrode length L (same map as common.grid_for)."""
    z0, L0, p = lay["z_lines"], float(lay["L0"]), int(lay["degree"])
    ztop = z0[-1]
    z = z0.copy()
    lower = (z0 > 0) & (z0 < L0)
    upper = (z0 > L0) & (z0 < ztop)
    z[lower] = z0[lower] * (L / L0)
    z[upper] = L + (z0[upper] - L0) * (ztop - L) / (ztop - L0)
    z[z0 == L0] = L
    return np.append((z[:-1, None] + np.outer(np.diff(z), np.arange(p) / p)).ravel(), z[-1])


def load(name, lay):
    files = sorted(DATA.glob(f"{name}_*.npz"))
    rho = np.array([float(np.load(f)["rho"]) for f in files])
    L = np.array([float(np.load(f)["L"]) for f in files])
    Y = np.stack([np.stack([np.load(f)["v"], np.load(f)["psi"]]) for f in files]).astype(np.float32)
    return files, rho, L, Y


def inputs(rho, L, lay, stats):
    act = lay["active"].astype(np.float32)
    R = np.arcsinh(lay["r_nodes"] / SCALE)
    out = []
    for a, b in zip(rho, L):
        Z = np.arcsinh(z_nodes(lay, b) / SCALE)
        RR, ZZ = np.meshgrid(R, Z)
        x = np.stack([np.full_like(act, (np.log(a) - stats["lr_mu"]) / stats["lr_sd"]),
                      np.full_like(act, (b - stats["L_mu"]) / stats["L_sd"]),
                      (RR / R.max()).astype(np.float32), (ZZ / np.abs(Z).max()).astype(np.float32), act])
        out.append(x)
    return torch.tensor(np.stack(out), dtype=torch.float32)


def train(model, X, T, w, epochs, lr, t0, label):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    total = 0.0
    for epoch in range(epochs):
        perm = torch.randperm(len(X))
        total = 0.0
        for k in range(0, len(X), BATCH):
            idx = perm[k:k + BATCH]
            loss = (((model(X[idx]) - T[idx]) ** 2) * w).sum() / (w.sum() * 2 * len(idx))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        sched.step()
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"{label} epoch {epoch:4d}  loss {total / len(X):.3e}  {time.perf_counter() - t0:.0f} s", flush=True)
    return total / len(X)


def main(epochs_coarse=150, epochs_fine=30):
    """Train on every second node first (four times cheaper), then fine-tune on the
    full certificate grid; a Fourier neural operator is defined on either."""
    lay = dict(np.load(HERE / "layout.npz"))
    _, rho, L, Y = load("train", lay)
    stats = {"lr_mu": float(np.log(rho).mean()), "lr_sd": float(np.log(rho).std()),
             "L_mu": float(L.mean()), "L_sd": float(L.std()),
             "v_sd": float(Y[:, 0].std()), "psi_sd": float(Y[:, 1].std())}
    scale = np.array([stats["v_sd"], stats["psi_sd"]], dtype=np.float32)[None, :, None, None]
    X = inputs(rho, L, lay, stats).to(DEVICE)
    T = torch.tensor(Y / scale).to(DEVICE)
    w = torch.tensor(lay["active"], dtype=torch.float32).to(DEVICE)

    model = FNO(n_modes=(48, 24), hidden_channels=32, in_channels=X.shape[1], out_channels=2,
                n_layers=4, domain_padding=0.1).to(DEVICE)
    print(f"{len(X)} training fields, grid {tuple(X.shape[2:])}, "
          f"{sum(p.numel() for p in model.parameters())} parameters, device {DEVICE}", flush=True)
    t0 = time.perf_counter()
    train(model, X[:, :, ::2, ::2], T[:, :, ::2, ::2], w[::2, ::2], epochs_coarse, LR, t0, "coarse")
    final = train(model, X, T, w, epochs_fine, LR / 10, t0, "fine")
    total = final * len(X)
    epochs = (epochs_coarse, epochs_fine)
    t_train = time.perf_counter() - t0
    torch.save(model.state_dict(), HERE / "operator.pt")

    model.eval()
    preds, t_pred = {}, []
    with torch.no_grad():
        for name in ("calib", "test_in", "test_out"):
            files, r_, L_, _ = load(name, lay)
            Xs = inputs(r_, L_, lay, stats).to(DEVICE)
            out = []
            for k in range(len(Xs)):
                t1 = time.perf_counter()
                yk = model(Xs[k:k + 1]).cpu().numpy()[0] * scale[0]
                t_pred.append(time.perf_counter() - t1)
                out.append(yk)
            preds[name] = np.stack(out)
            preds[name + "_files"] = np.array([f.name for f in files])
    np.savez(HERE / "predictions.npz", **preds)
    meta = dict(stats, epochs=epochs, train_seconds=t_train, predict_seconds=float(np.median(t_pred)),
                final_loss=total / len(X), device=DEVICE, n_train=len(X),
                parameters=sum(p.numel() for p in model.parameters()))
    (HERE / "train_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main(*[int(a) for a in sys.argv[1:3]])
