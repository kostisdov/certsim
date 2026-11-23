"""Figures of paper 2 from the experiments' result files (certsim environment).

    uv run --project ../.. --group viz python make_figures.py      # writes figs/*.pdf
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402

from certsim.gci import gci  # noqa: E402

HERE = Path(__file__).resolve().parent
EXP = HERE / "experiments"
FIGS = HERE / "figs"
TRUE_Z = 0.5 * (93.9502742 + 93.9502758)
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3, "savefig.bbox": "tight"})


def fig_gci():
    runs = json.loads((EXP / "e1_fenicsx" / "gci_fenicsx.json").read_text())
    hs = [1.0, 0.5, 0.25, 0.125]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.45))
    for ax, family, title in zip(axes, ("uniform P2", "graded P3"),
                                 ("Quadratic, no corner grading", "Cubic, graded to 1 nm")):
        fe = np.array([runs[f"{family} {h}"]["Z_fe"] for h in hs]) - TRUE_Z
        cert = np.array([runs[f"{family} {h}"]["Z_cert"] for h in hs]) - TRUE_Z
        g = gci(list((fe + TRUE_Z)[::-1][:3]))
        lo, hi = np.array(g["band"]) - TRUE_Z
        x = np.arange(len(hs))
        ax.vlines(x, cert[:, 0], cert[:, 1], color="C0", lw=6, alpha=0.35, label="certified interval")
        ax.plot(x, fe, "o", color="C1", label="FEniCSx impedance")
        ax.fill_between([x[-1] - 0.25, x[-1] + 0.25], lo, hi, color="C3", alpha=0.25, label="GCI band (3 finest)")
        ax.axhline(0.0, color="k", lw=0.8, label="true value")
        ax.set_xticks(x, [f"{h:g}" for h in hs])
        ax.set_xlabel("mesh spacing (mm)")
        ax.set_title(title)
        if family == "graded P3":
            ax.set_yscale("symlog", linthresh=1e-6)
        else:
            ax.annotate(f"GCI band ends {abs(hi) * 1e4:.0f}e-4 ohm\nbelow the true value", (x[-1], hi), (1.1, -0.9),
                        fontsize=7.5, arrowprops=dict(arrowstyle="->", lw=0.6))
    axes[0].set_ylabel("impedance - true value (ohm)")
    axes[0].legend(fontsize=7, loc="lower right")
    fig.savefig(FIGS / "fig_gci.pdf")


def fig_surrogate():
    tests = json.loads((EXP / "e3_surrogate" / "results.json").read_text())["tests"]
    rho = np.array([float(k) for k in tests])
    order = np.argsort(rho)
    rho = rho[order]
    vals = [tests[k] for k in np.array(list(tests))[order]]
    err = np.array([v["true_rel_error"] for v in vals])
    width = np.array([v["bound_rel_width"] for v in vals])
    base = json.loads((EXP / "e3d_baselines" / "results_e3a.json").read_text())
    by_rho = {round(t["rho"], 6): t for t in base["tests"]}
    err_field = np.array([abs(by_rho[round(x, 6)]["field"]["Z"] - by_rho[round(x, 6)]["Z_true"])
                          / by_rho[round(x, 6)]["Z_true"] for x in rho])
    q = base["conformal_half_width"]
    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    ax.axvspan(1.7, 3.75, color="0.5", alpha=0.12, label="training range")
    ax.loglog(rho, width, "s-", color="C0", label="certified width of the impedance")
    ax.loglog(rho, err, "o--", color="C3", label="error of the surrogate's impedance output")
    ax.loglog(rho, err_field, "^:", color="C2", label="error of the impedance from its potential")
    ax.axhline(q["head"], color="C3", lw=0.8, ls=(0, (4, 3)), label="conformal half-width (90%), output")
    ax.axhline(q["field"], color="C2", lw=0.8, ls=(0, (4, 3)), label="conformal half-width (90%), potential")
    ax.set_xlabel(r"conductivity ratio $\rho=\sigma_b/\sigma_t$")
    ax.set_ylabel("relative size")
    ax.set_xticks([0.7, 1, 2, 3, 5, 10], ["0.7", "1", "2", "3", "5", "10"])
    ax.legend(fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=1, frameon=False)
    fig.savefig(FIGS / "fig_surrogate.pdf")


def fig_operator():
    """Certified width per test for the two-input surrogates on the certificate grid (E3c)."""
    runs = [("results_reduced.json", "reduced-order surrogate", "s", "C0"),
            ("projected", "neural operator, nodal loss, projected onto 12 POD modes", "D", "C2"),
            ("results_energy.json", "neural operator, energy loss", "^", "C1"),
            ("results_nodal.json", "neural operator, nodal loss", "o", "C3")]
    res = {f: json.loads((EXP / "e3c_operator" / f).read_text()) for f, *_ in runs if f != "projected"}
    proj = json.loads((EXP / "e3c_operator" / "results_projected.json").read_text())["rows"]
    res["projected"] = {"test_in": proj["nodal/test_in/K12"], "test_out": proj["nodal/test_out/K12"]}
    ref = res["results_reduced.json"]
    order_in = np.argsort([t["rho"] for t in ref["test_in"]])
    n_in, n_out = len(order_in), len(ref["test_out"])
    x_in, x_out = np.arange(n_in), n_in + 0.8 + 1.5 * np.arange(n_out)
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    ax.axvspan(-0.6, n_in - 0.4, color="0.5", alpha=0.12, lw=0)
    ax.axhline(1.0, color="k", lw=0.8, ls=(0, (4, 3)))
    ax.annotate("width equal to the impedance", (x_out[-1] + 0.6, 1.0), (0, 3), textcoords="offset points",
                ha="right", va="bottom", fontsize=7)
    for f, label, marker, color in runs:
        rows_in = [res[f]["test_in"][i] for i in order_in]
        assert all(a["rho"] == b["rho"] and a["L"] == b["L"] for a, b in zip(rows_in, [ref["test_in"][i] for i in order_in]))
        assert all(r["contains_truth"] for r in rows_in + res[f]["test_out"])
        w = [r["cert_rel_width"] for r in rows_in + res[f]["test_out"]]
        ax.semilogy(np.r_[x_in, x_out], w, marker, color=color, ms=5, label=label)
    ax.set_xticks(x_out, [f"{t['rho']:g}\n{t['L'] * 1e3:g}" for t in ref["test_out"]], fontsize=7)
    ax.text(x_out[0] - 0.7, -0.025, "$\\rho$\n$L$ (mm)", transform=ax.get_xaxis_transform(), ha="right", va="top", fontsize=7)
    ax.text((n_in - 1) / 2, 3e-3, "inside the training range\n(12 tests, by increasing $\\rho$)", ha="center", va="center", fontsize=7)
    ax.text(x_out.mean(), 3e-5, "outside the training range", ha="center", va="center", fontsize=7)
    ax.set_xlim(-0.6, x_out[-1] + 0.6)
    ax.set_ylabel("relative width of the\ncertified impedance interval")
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False)
    fig.savefig(FIGS / "fig_operator.pdf")


if __name__ == "__main__":
    FIGS.mkdir(exist_ok=True)
    for f in (fig_gci, fig_surrogate, fig_operator):
        f()
        print("wrote", f.__name__)
