"""Train a standard Fourier Neural Operator and predict the test fields (PyTorch environment).

Uses neuraloperator's FNO on the GPU (MPS) when available. Input channels:
log(rho) as a constant field, normalized r and z, and the domain mask; output:
the potential on the uniform grid. The loss is the mean squared error over the
domain (the mask), plus the electrode, where the target is the voltage.
Writes predictions.npz with the FNO's fields at the test ratios.

    uv run python train_fno.py
"""

import time
from pathlib import Path

import numpy as np
import torch
from neuralop.models import FNO

HERE = Path(__file__).resolve().parent
torch.manual_seed(0)
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
EPOCHS, BATCH, LR = 400, 4, 2e-3


def inputs(rho, r, z, mask):
    R, Z = np.meshgrid(r / r[-1], (z - z[0]) / (z[-1] - z[0]), indexing="ij")
    x = np.stack([np.full_like(R, np.log(rho)), R, Z, mask])
    return torch.tensor(x, dtype=torch.float32)


def main():
    d = np.load(HERE / "data.npz")
    r, z, mask = d["r"], d["z"], d["mask"]
    X = torch.stack([inputs(rho, r, z, mask) for rho in d["train_rho"]]).to(DEVICE)
    Y = torch.tensor(d["train_v"], dtype=torch.float32)[:, None].to(DEVICE)
    weight = torch.tensor(mask + (d["train_v"][0] == 1.0), dtype=torch.float32).to(DEVICE)  # domain + electrode

    model = FNO(n_modes=(32, 32), hidden_channels=32, in_channels=4, out_channels=1).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    t0 = time.perf_counter()
    for epoch in range(EPOCHS):
        perm = torch.randperm(len(X))
        for k in range(0, len(X), BATCH):
            idx = perm[k:k + BATCH]
            loss = (((model(X[idx]) - Y[idx]) ** 2) * weight).sum() / (weight.sum() * len(idx))
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        if epoch % 50 == 0 or epoch == EPOCHS - 1:
            print(f"epoch {epoch:4d}  loss {loss.item():.3e}  {time.perf_counter() - t0:.0f} s")
    t_train = time.perf_counter() - t0

    model.eval()
    with torch.no_grad():
        Xt = torch.stack([inputs(rho, r, z, mask) for rho in d["test_rho"]]).to(DEVICE)
        t1 = time.perf_counter()
        pred = model(Xt)[:, 0].cpu().numpy()
        t_pred = (time.perf_counter() - t1) / len(Xt)
    np.savez(HERE / "predictions.npz", rho=d["test_rho"], v=pred.astype(float), r=r, z=z,
             train_seconds=t_train, predict_seconds=t_pred, device=DEVICE)
    print(f"trained in {t_train:.0f} s on {DEVICE}; {t_pred * 1e3:.1f} ms per prediction")


if __name__ == "__main__":
    main()
