"""Compare certified bipolar lesion depths with Meckes et al. (Heart Rhythm O2, 2022).

Meckes et al. modelled a 2-electrode catheter (2 mm electrodes, 4 mm spacing)
with an irreversible-electroporation threshold of 268 V/cm and validated it
against in-vivo lesion depths. Their conductivities are in a supplement we
could not retrieve, so the contact case is certified for every blood/tissue
conductivity ratio in 1.7-3.75 instead. Assumptions of ours, not theirs: the
catheter stands perpendicular to the tissue (axisymmetric), electrode radius
1.25 mm, no temperature dependence, floating generator.

    uv run python papers/1-pfa-certificates/experiments/meckes_comparison.py
"""

import dataclasses
import pickle
import time
from pathlib import Path

import numpy as np

from certsim.checker import check
from certsim.ranges import check_range
from certsim.solver import default_grid, solve
from certsim.spec import default_spec, lesion_tiles

RESULTS = Path(__file__).resolve().parents[3] / "results"  # shared, at the repository root
THRESHOLD = 2.68e4  # V/m
RHO = (1.7, 3.75)
K = 17
GAPS = (1e-3, 2e-3, 3e-3, 4e-3)
VOLTS = np.arange(100.0, 2501.0, 25.0)

# Meckes et al. 2022: model predictions and the in-vivo data they compare with
MECKES = [
    (500.0, 2.28, "in vivo 2.25 +- 0.85 (Stewart)"),
    (700.0, 3.0, "Meckes: minimum voltage for 3 mm (a voltage, not a depth)"),
    (2200.0, 6.16, "in vivo 6.5 +- 1.7 (Koruth)"),
    (2500.0, 6.5, "upper end of 2.5-6.5 mm range"),
]


def spec(gap=0.0):
    return default_spec(
        voxels=(), regions=(), tiles=lesion_tiles(r_max=6e-3, depth=8e-3, step=0.25e-3),
        L=2e-3, ring_spacing=4e-3, ring_length=2e-3, gap=gap,
    )


def cached(name, build):
    path = RESULTS / name
    if path.exists():
        return pickle.loads(path.read_bytes())
    t0 = time.perf_counter()
    rep = build()
    print(f"  {name}: {time.perf_counter() - t0:.0f} s")
    path.write_bytes(pickle.dumps(rep))
    return rep


def contact_range():
    s = spec()
    r, z = default_grid(s)
    rhos = list(np.linspace(*RHO, K))
    certs = [solve(dataclasses.replace(s, sigma_blood=x * s.sigma_tissue), r, z, degree=3) for x in rhos]
    return check_range(s, certs, rhos)


def at_gap(g):
    s = spec(g)
    r, z = default_grid(s)
    return check(s, solve(s, r, z, degree=3))


def min_voltage(rep, depth):
    """(largest voltage proven insufficient, smallest voltage proven sufficient)."""
    insufficient = sufficient = None
    for V in VOLTS:
        reached, beyond = rep.lesion_extent(V, THRESHOLD)["depth"]
        if beyond is not None and beyond < depth:
            insufficient = V
        if sufficient is None and reached >= depth:
            sufficient = V
    return insufficient, sufficient


def fmt(extent):
    inner, outer = extent
    return f"{inner*1e3:4.2f}-{'>8' if outer is None else f'{outer*1e3:.2f}'}"


def main():
    RESULTS.mkdir(exist_ok=True)
    print("certifying (cached in results/) ...")
    contact = cached("meckes_contact_range.pkl", contact_range)
    nominal = cached("meckes_gap0.pkl", lambda: at_gap(0.0))
    gaps = {g: cached(f"meckes_gap{g*1e3:.0f}mm.pkl", lambda g=g: at_gap(g)) for g in GAPS}

    print(f"\nbipolar catheter, contact, threshold {THRESHOLD/100:.0f} V/cm, every conductivity ratio "
          f"{RHO[0]}-{RHO[1]} (eta_max {contact.eta_max:.1e})")
    print("  voltage   certified depth [mm]   Meckes 2022 [mm]")
    for V, d, note in MECKES:
        print(f"  {V:6.0f} V   {fmt(contact.lesion_extent(V, THRESHOLD)['depth']):>18}   {d:5.2f}  ({note})")

    print("\nminimum voltage for a given depth, nominal conductivities: proven insufficient / proven sufficient")
    print("  gap      1 mm deep          3 mm deep")
    for g, rep in [(0.0, nominal)] + list(gaps.items()):
        cells = []
        for depth in (1e-3, 3e-3):
            lo, hi = min_voltage(rep, depth)
            cells.append(f"{'-' if lo is None else f'{lo:.0f}'} / {'>2500' if hi is None else f'{hi:.0f}'} V")
        print(f"  {g*1e3:.0f} mm   {cells[0]:>16}   {cells[1]:>16}")
    print("  Meckes: >1 mm needs 300 V; 3 mm needs 700 V in contact and about 1500 V at 1-2 mm")
    print("  Meckes: at 1200 V the depth is negligible from 4 mm away")
    reached, beyond = gaps[4e-3].lesion_extent(1200.0, THRESHOLD)["depth"]
    print(f"  certified at 1200 V, 4 mm gap: depth {reached*1e3:.2f}-"
          f"{'>8' if beyond is None else f'{beyond*1e3:.2f}'} mm")
    print("wrote", figure(contact, gaps))


def figure(contact, gaps):
    """Certified depth band against voltage, with Meckes' model and in-vivo data."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink2, muted, grid, axis = "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
    blue, orange, aqua = "#2a78d6", "#eb6834", "#1baf7a"
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": axis, "axes.labelcolor": ink2,
                         "xtick.color": muted, "ytick.color": muted, "axes.spines.top": False,
                         "axes.spines.right": False, "legend.frameon": False,
                         "figure.facecolor": "white", "axes.facecolor": "white",
                         "savefig.facecolor": "white"})
    V = VOLTS[VOLTS >= 200]
    fig, ax = plt.subplots(figsize=(5.25, 3.15))
    for rep, color, label in ((contact, blue, "This work, contact, all ratios 1.7–3.75"),
                              (gaps[2e-3], "#86b6ef", "This work, 2 mm gap, nominal ratio")):
        ext = [rep.lesion_extent(v, THRESHOLD)["depth"] for v in V]
        inner = np.array([e[0] for e in ext]) * 1e3
        outer = np.array([e[1] * 1e3 if e[1] is not None else 8.0 for e in ext])
        ax.fill_between(V, inner, outer, step="mid", color=color, alpha=0.3, lw=0)
        ax.step(V, inner, where="mid", color=color, lw=2, label=label)
    # depth predictions only: 700 V is Meckes' minimum voltage for 3 mm, not a depth
    mv = [m for m in MECKES if m[0] in (500.0, 2200.0)]
    ax.plot([m[0] for m in mv], [m[1] for m in mv], "s", color=orange, ms=6, label="Meckes et al. model")
    ax.errorbar([500, 2200], [2.25, 6.5], yerr=[0.85, 1.7], fmt="o", color=aqua, ms=6, capsize=3,
                lw=1.5, label="In vivo, as cited by Meckes et al.")
    ax.axhline(8.0, color=axis, lw=0.8, ls="--")
    ax.text(V[0], 8.1, "Map edge: bands reaching it are open-ended", color=muted, fontsize=7, va="bottom")
    ax.set_xlabel("Applied voltage, tip to ring (V)")
    ax.set_ylabel("Lesion depth (mm)")
    ax.set_ylim(0, 8.8)
    ax.grid(axis="y", color=grid, lw=0.6)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_meckes.png", dpi=200)
    fig.savefig(RESULTS / "fig_meckes.pdf")
    plt.close(fig)
    return RESULTS / "fig_meckes.pdf"


if __name__ == "__main__":
    main()
