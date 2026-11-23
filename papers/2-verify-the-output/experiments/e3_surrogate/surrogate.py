"""E3: certify the outputs of a learned surrogate that is admissible by construction.

The surrogate is a reduced-order model of the paper 1 lesion-map problem as a
function of the conductivity ratio rho = sigma_blood / sigma_tissue: a POD basis
of training solutions, and a small neural network (numpy, Adam) that maps
log(rho) to the POD coefficients of the potential and of the stream function
(whose shaft value is part of the same POD vector), and to log(impedance).

The surrogate predicts only free nodal values. Electrode and ground values of
the potential, and the stream function on the axis and the shaft, are set
exactly, so every prediction is an admissible certificate and costs a forward
pass and a check. Each prediction is certified with the unchanged checker and
compared with a reference solve at the same rho.

    uv run python surrogate.py
"""

import json
import time
from pathlib import Path

import numpy as np

from certsim.checker import check
from certsim.solver import Certificate, default_grid, solve
from certsim.spec import Box, default_spec, lesion_tiles, node_masks

HERE = Path(__file__).resolve().parent
SIGMA_T = 0.25
DEGREE = 3
TRAIN = np.geomspace(1.7, 3.75, 17)
TEST_IN = np.array([1.8, 2.0, 2.4, 2.8, 3.1, 3.6])
TEST_OUT = np.array([0.7, 1.0, 1.3, 5.0, 7.0, 10.0])
MODES = 6


def spec_for(rho):
    return default_spec(voxels=(), sigma_tissue=SIGMA_T, sigma_blood=rho * SIGMA_T,
                        tiles=lesion_tiles(r_max=6e-3, depth=8e-3, step=0.25e-3),
                        regions=(Box(0.0, 5e-3, -10e-3, -6e-3),))


class Layout:
    """Which nodal values the surrogate predicts, and which are fixed exactly."""

    def __init__(self, spec, r, z):
        m = node_masks(spec, r, z, DEGREE)
        self.shape = m["active"].shape
        self.v_fixed = m["electrode"] | m["ground"] | ~m["active"]
        self.v_value = np.where(m["electrode"], spec.tip_voltage, 0.0)
        self.psi_zero = m["axis"] | ~m["active"]
        self.psi_shaft = m["shaft"] & ~m["axis"]
        self.psi_free = ~(self.psi_zero | self.psi_shaft)

    def features(self, cert):
        """Free values of v, and free values of psi with the shaft value appended (one vector,
        so the shaft value stays consistent with its neighbours)."""
        return cert.v[~self.v_fixed], np.append(cert.psi[self.psi_free], cert.psi[self.psi_shaft][0])

    def certificate(self, r, z, v_free, psi_aug):
        psi_free, psi_shaft = psi_aug[:-1], psi_aug[-1]
        v = self.v_value.copy()
        v[~self.v_fixed] = v_free
        psi = np.zeros(self.shape)
        psi[self.psi_free] = psi_free
        psi[self.psi_shaft] = psi_shaft  # one exact value on the whole shaft wall
        empty = np.zeros((0, *self.shape))
        return Certificate(r, z, v, psi, empty, empty, DEGREE, {})


class MLP:
    """tanh network, trained full-batch with Adam on standardized targets."""

    def __init__(self, n_in, n_out, width=32, seed=0):
        rng = np.random.default_rng(seed)
        sizes = [n_in, width, width, n_out]
        self.W = [rng.standard_normal((a, b)) / np.sqrt(a) for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b) for b in sizes[1:]]

    def forward(self, x):
        hs = [x]
        for k, (W, b) in enumerate(zip(self.W, self.b)):
            x = x @ W + b
            if k < len(self.W) - 1:
                x = np.tanh(x)
            hs.append(x)
        return x, hs

    def fit(self, x, y, steps=20000, lr=3e-3):
        params = self.W + self.b
        m = [np.zeros_like(p) for p in params]
        s = [np.zeros_like(p) for p in params]
        for t in range(1, steps + 1):
            out, hs = self.forward(x)
            g = 2 * (out - y) / len(x)
            gW, gb = [None] * len(self.W), [None] * len(self.b)
            for k in reversed(range(len(self.W))):
                gW[k], gb[k] = hs[k].T @ g, g.sum(0)
                if k:
                    g = (g @ self.W[k].T) * (1 - hs[k] ** 2)
            for i, (p, gp) in enumerate(zip(params, gW + gb)):
                m[i] = 0.9 * m[i] + 0.1 * gp
                s[i] = 0.999 * s[i] + 0.001 * gp**2
                p -= lr * (m[i] / (1 - 0.9**t)) / (np.sqrt(s[i] / (1 - 0.999**t)) + 1e-12)
        return float(np.mean((self.forward(x)[0] - y) ** 2))


def main():
    spec0 = spec_for(2.8)
    r, z = default_grid(spec0)
    lay = Layout(spec0, r, z)
    cache = HERE / "snapshots.npz"

    # training data: reference solves at the training ratios
    t0 = time.perf_counter()
    if cache.exists():
        d = np.load(cache)
        V, P, Z = d["V"], d["P"], d["Z"]
    else:
        rows = []
        for rho in TRAIN:
            cert = solve(spec_for(rho), r, z, degree=DEGREE)
            rows.append((*lay.features(cert), 1 / cert.claims["conductance"]))
        V, P = np.array([x[0] for x in rows]), np.array([x[1] for x in rows])
        Z = np.array([x[2] for x in rows])
        np.savez(cache, V=V, P=P, Z=Z)
    t_data = time.perf_counter() - t0

    # POD bases and coefficients
    def pod(X):
        mean = X.mean(0)
        U, sv, Vt = np.linalg.svd(X - mean, full_matrices=False)
        return mean, Vt[:MODES], (X - mean) @ Vt[:MODES].T

    v_mean, v_basis, v_coef = pod(V)
    p_mean, p_basis, p_coef = pod(P)
    targets = np.hstack([v_coef, p_coef, np.log(Z)[:, None]])
    mu, sd = targets.mean(0), targets.std(0) + 1e-30
    x_train = np.log(TRAIN)[:, None]
    net = MLP(1, targets.shape[1], width=16)
    loss = net.fit((x_train - x_train.mean()) / x_train.std(), (targets - mu) / sd)

    def predict(rho):
        x = (np.log([[rho]]) - x_train.mean()) / x_train.std()
        y = net.forward(x)[0][0] * sd + mu
        v_free = v_mean + y[:MODES] @ v_basis
        psi_aug = p_mean + y[MODES:2 * MODES] @ p_basis
        return lay.certificate(r, z, v_free, psi_aug), float(np.exp(y[-1]))

    results = {}
    for label, rhos in (("in", TEST_IN), ("out", TEST_OUT)):
        for rho in rhos:
            spec = spec_for(rho)
            t1 = time.perf_counter()
            cert, z_pred = predict(rho)
            t_pred = time.perf_counter() - t1
            rep = check(spec, cert)
            t_check = time.perf_counter() - t1 - t_pred
            ref = check(spec, solve(spec, r, z, degree=DEGREE))
            lo, hi = rep.impedance.lo, rep.impedance.hi
            true = 0.5 * (ref.impedance.lo + ref.impedance.hi)
            widths = np.array([iv.rel_width for iv in rep.tile_rms_field])
            if not lo - 1e-6 * z_pred <= z_pred <= hi + 1e-6 * z_pred:
                verdict = "refuted"
            elif lo >= z_pred * (1 - 1e-3) and hi <= z_pred * (1 + 1e-3):
                verdict = "certified"
            else:
                verdict = "unsupported"
            results[f"{rho:g}"] = dict(range=label, Z_pred=z_pred, Z_cert=[lo, hi], Z_true=true,
                                       true_rel_error=abs(z_pred - true) / true, bound_rel_width=(hi - lo) / true,
                                       eta=rep.eta.hi, eta_ref=ref.eta.hi, tile_median=float(np.median(widths)),
                                       verdict=verdict, predict_s=t_pred, check_s=t_check)
            o = results[f"{rho:g}"]
            print(f"rho {rho:5.2f} ({label:3s}) {verdict:11s} Z_pred {z_pred:.5f}  certified {lo:.5f}-{hi:.5f}  "
                  f"true {true:.5f}  true err {o['true_rel_error']:.1e}  bound width {o['bound_rel_width']:.1e}  "
                  f"eta {o['eta']:.1e} (ref {o['eta_ref']:.1e})  tiles {o['tile_median'] * 100:.1f}%")
    meta = dict(train=TRAIN.tolist(), modes=MODES, train_loss=loss, data_seconds=t_data)
    (HERE / "results.json").write_text(json.dumps({"meta": meta, "tests": results}, indent=1))


if __name__ == "__main__":
    main()
