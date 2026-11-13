"""Figures of the certified PFA lesion (results/fig_*.png and .pdf; the paper uses the PDFs).

    uv run --group viz python papers/1-pfa-certificates/experiments/pfa_figures.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from _pfa_cache import RESULTS, load  # noqa: E402

from certsim.viz import envelope, field_at, tile_grid  # noqa: E402

INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SURFACE = "#ffffff"  # white, so the figures sit on the page without a visible box
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3
ORANGE = "#eb6834"
METAL, SHAFT = "#898781", "#c3c2b7"

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": INK, "axes.titlesize": 10,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})


def save(fig, name):
    """PNG for quick viewing, vector PDF (raster parts at 300 dpi) for the paper."""
    fig.savefig(RESULTS / f"{name}.png", dpi=200)
    fig.savefig(RESULTS / f"{name}.pdf", dpi=300)
    plt.close(fig)
    return RESULTS / f"{name}.pdf"


def fig_map(spec, cert, rep, V=1000.0, thr_vcm=700.0):
    r = np.linspace(-6e-3, 6e-3, 1200)
    z = np.linspace(3e-3, -8e-3, 1100)
    R, Z = np.meshgrid(r, z)
    Er, Ez = field_at(spec, cert, np.abs(R), Z)
    E = np.hypot(Er, Ez) * V / spec.voltage / 100
    # drawn at the printed size (0.68 of the text width), so the fonts print at their nominal size
    fig, ax = plt.subplots(figsize=(4.35, 3.6))
    # background, tiles and electrode are rasterized together at the PDF's 300 dpi (no seams
    # between tiles in any viewer); axes, labels and legend stay vector
    ax.set_rasterization_zorder(1.5)
    cmap = LinearSegmentedColormap.from_list("blues", BLUES)
    im = ax.imshow(np.clip(E, 50, 5e3), extent=(-6, 6, -8, 3), origin="upper", cmap=cmap,
                   norm=LogNorm(50, 5e3), interpolation="bilinear")
    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label(f"Solver field |E| at {V:.0f} V (V/cm),\nnot certified", color=INK2)
    cb.outline.set_edgecolor(AXIS)

    r_edges, z_edges, index = tile_grid(rep)
    _, _, _, s = envelope(rep, V, thr_vcm * 100)
    for k in range(s.shape[0]):
        for m in range(s.shape[1]):
            if s[k, m] == -1:
                continue
            for sign in (-1, 1):
                x0 = sign * r_edges[m] * 1e3 if sign > 0 else -r_edges[m + 1] * 1e3
                w = (r_edges[m + 1] - r_edges[m]) * 1e3
                y0, h = z_edges[k + 1] * 1e3, (z_edges[k] - z_edges[k + 1]) * 1e3
                if s[k, m] == 1:
                    ax.add_patch(Rectangle((x0, y0), w, h, facecolor=ORANGE, edgecolor=ORANGE,
                                           lw=0.4, zorder=1))
                else:
                    ax.add_patch(Rectangle((x0, y0), w, h, facecolor="none", edgecolor=ORANGE,
                                           hatch="////", lw=0.0, zorder=1))
    a, g, L = spec.a * 1e3, spec.tip * 1e3, spec.L * 1e3
    ax.add_patch(Rectangle((-a, g), 2 * a, L, facecolor=METAL, lw=0, zorder=1))
    ax.add_patch(Rectangle((-a, g + L), 2 * a, max(3 - g - L, 0), facecolor=SHAFT, lw=0, zorder=1))
    ax.text(0, g + L / 2, "electrode", ha="center", va="center", color="white", fontsize=8)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.text(5.8, 0.25, "blood", ha="right", va="bottom", color="white")
    ax.text(5.8, -5.0, "myocardium", ha="right", va="top", color="white")
    ax.set_xlim(-6, 6)
    ax.set_ylim(-8, 3)
    ax.set_xlabel("Radial position (mm)")
    ax.set_ylabel("Height z (mm); tissue surface at 0")
    ax.legend(handles=[
        Patch(facecolor=ORANGE, label="Proven lethal"),
        Patch(facecolor="none", edgecolor=ORANGE, hatch="////", label="Undecided"),
        Patch(facecolor="none", edgecolor="none", label="No overlay: proven spared"),
    ], loc="lower left", fontsize=8)
    fig.tight_layout()
    return save(fig, "fig_lesion_map")


def fig_extent(rep, thresholds=(400.0, 700.0, 1000.0)):
    V = np.arange(250.0, 2501.0, 50.0)
    edge = {"depth": 8.0, "radius": 6.0}
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.85), sharex=True)
    for ax, key, label in zip(axes, ("depth", "radius"), ("Depth on axis (mm)", "Largest radius (mm)")):
        for color, t in zip(SERIES, thresholds):
            ext = [rep.lesion_extent(v, t * 100)[key] for v in V]
            inner = np.array([e[0] for e in ext]) * 1e3
            outer = np.array([e[1] * 1e3 if e[1] is not None else np.nan for e in ext])
            ok = ~np.isnan(outer)
            ax.fill_between(V[ok], inner[ok], outer[ok], step="mid", color=color, alpha=0.25, lw=0)
            ax.step(V, inner, where="mid", color=color, lw=2, label=f"{t:.0f} V/cm")
            ax.step(V[ok], outer[ok], where="mid", color=color, lw=1, alpha=0.8)
        ax.axhline(edge[key], color=AXIS, lw=0.8, ls="--")
        ax.text(V[-1], edge[key] + 0.08, "map edge", color=MUTED, fontsize=7, ha="right", va="bottom")
        ax.set_ylim(0, edge[key] + 0.5)
        ax.set_xlabel("Electrode voltage (V)")
        ax.set_ylabel(label)
        ax.grid(axis="y", color=GRID, lw=0.6)
    axes[0].legend(title="Lethal threshold", loc="upper left", fontsize=8, title_fontsize=8)
    fig.tight_layout()
    return save(fig, "fig_lesion_extent")


def main():
    spec, cert, rep = load()
    for p in (fig_map(spec, cert, rep), fig_extent(rep)):
        print("wrote", p)


if __name__ == "__main__":
    main()
