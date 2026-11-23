"""Relative energy error of both trained operators on the training fields (PyTorch env of e3b_fno).

The measure of train_energy.py on the full grid: the energy of the potential's
error and the complementary energy of the stream function's error, each relative
to the energy of the reference field, with the boundary values imposed exactly.
Writes energy_error.json.

    uv run --project ../e3b_fno python eval_energy.py
"""

import json

import numpy as np
import torch
from neuralop.models import FNO

from train_energy import edge_energy, fields, setup
from train_operator import DEVICE, HERE, inputs, load


def main():
    lay = dict(np.load(HERE / "layout.npz"))
    meta0 = json.loads((HERE / "train_meta.json").read_text())
    stats = {k: meta0[k] for k in ("lr_mu", "lr_sd", "L_mu", "L_sd", "v_sd", "psi_sd")}
    _, rho, L, Y = load("train", lay)
    X = inputs(rho, L, lay, stats).to(DEVICE)
    Yt = torch.tensor(Y).to(DEVICE)
    W, ref_v, ref_p, mk = setup(lay, rho, L, Y, 1)
    out = {}
    for name in ("operator.pt", "operator_energy.pt"):
        model = FNO(n_modes=(48, 24), hidden_channels=32, in_channels=X.shape[1], out_channels=2,
                    n_layers=4, domain_padding=0.1).to(DEVICE)
        # both checkpoints are written by the training scripts in this folder, so a full load is safe
        model.load_state_dict(torch.load(HERE / name, map_location=DEVICE, weights_only=False))
        model.eval()
        ev, ep = [], []
        with torch.no_grad():
            for k in range(len(X)):
                v, psi = fields(model(X[k:k + 1]), stats, {key: val for key, val in mk.items()})
                ev.append((edge_energy(v - Yt[k:k + 1, 0], W["vz"][k:k + 1], W["vr"][k:k + 1]) / ref_v[k]).item())
                ep.append((edge_energy(psi - Yt[k:k + 1, 1], W["pz"][k:k + 1], W["pr"][k:k + 1]) / ref_p[k]).item())
        out[name] = dict(rel_energy_v_mean=float(np.mean(ev)), rel_energy_psi_mean=float(np.mean(ep)),
                         rel_energy_v_median=float(np.median(ev)), rel_energy_psi_median=float(np.median(ep)))
        print(name, out[name], flush=True)
    (HERE / "energy_error.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
