"""Interactive view of the certified PFA lesion in Rerun.

Writes results/pfa_lesion.rrd. Open it with

    uv run --group viz rerun results/pfa_lesion.rrd

and drag the ``voltage`` timeline. The 3D scene is a cut-away of the tissue
under the catheter: the opaque orange core is proven lethal, the translucent
shell around it is undecided (and may be lethal), and everything outside is
proven spared. Tile states and envelopes are certified; the field image is the
solver's field and is not.

    uv run --group viz python papers/1-pfa-certificates/experiments/pfa_view.py [--threshold 700]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from matplotlib.colors import LinearSegmentedColormap, LogNorm

sys.path.insert(0, str(Path(__file__).parent))
from _pfa_cache import RESULTS, load  # noqa: E402

from certsim.viz import envelope, field_at, revolve, staircase, tile_grid  # noqa: E402

MM = 1e3
ORANGE = (0xEB, 0x68, 0x34)
ORANGE_LIGHT = (0xF3, 0xA0, 0x7A)
BLUE_LIGHT = (0xCD, 0xE2, 0xFB)
METAL, SHAFT = (0x89, 0x87, 0x81), (0xC3, 0xC2, 0xB7)
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CUT = (0.0, 1.5 * np.pi)  # 270 degrees, leaving a quarter open
REF_DEPTH = 5e-3  # reference plane for collateral field


def mesh(V, T, rgba):
    return rr.Mesh3D(vertex_positions=V, triangle_indices=T, albedo_factor=rgba)


def catheter(spec):
    a, g, L = spec.a * MM, spec.tip * MM, spec.L * MM
    tip = np.array([(0.0, g), (a, g), (a, g + L)])
    shaft = np.array([(a, g + L), (a, g + L + 9.0), (0.0, g + L + 9.0)])
    rr.log("world/catheter/electrode", mesh(*revolve(tip, 0, 2 * np.pi, 48), METAL + (255,)), static=True)
    rr.log("world/catheter/shaft", mesh(*revolve(shaft, 0, 2 * np.pi, 48), SHAFT + (255,)), static=True)


def cut_faces(r_edges, z_edges, state):
    """Tile quads on the two cut planes, coloured by certified state."""
    colors = {1: ORANGE + (255,), 0: ORANGE_LIGHT + (200,), -1: BLUE_LIGHT + (60,)}
    V, T, C = [], [], []
    for th in CUT:
        c, s = np.cos(th), np.sin(th)
        for k in range(state.shape[0]):
            for m in range(state.shape[1]):
                r0, r1 = r_edges[m] * MM, r_edges[m + 1] * MM
                z0, z1 = z_edges[k + 1] * MM, z_edges[k] * MM
                base = len(V)
                V += [(r0 * c, r0 * s, z0), (r1 * c, r1 * s, z0), (r1 * c, r1 * s, z1), (r0 * c, r0 * s, z1)]
                T += [(base, base + 1, base + 2), (base, base + 2, base + 3)]
                C += [colors[int(state[k, m])]] * 4
    return rr.Mesh3D(vertex_positions=V, triangle_indices=T, vertex_colors=C)


def field_image(spec, cert, voltage, n_r=240, n_z=220):
    """RGB image of the solver's |E| [V/cm], mirrored about the axis."""
    r = np.linspace(-6e-3, 6e-3, n_r)
    z = np.linspace(3e-3, -8e-3, n_z)
    R, Z = np.meshgrid(r, z)
    Er, Ez = field_at(spec, cert, np.abs(R), Z)
    E = np.hypot(Er, Ez) * voltage / spec.voltage / 100
    cmap = LinearSegmentedColormap.from_list("blues", BLUES)
    rgb = cmap(LogNorm(10, 1e4)(np.clip(E, 10, 1e4)))[..., :3]
    rgb[np.isnan(E)] = np.array(METAL) / 255
    return (rgb * 255).astype(np.uint8), (r, z)


def state_image(r_edges, z_edges, state, px=0.05e-3):
    """Certified tile states as a mirrored segmentation image (class ids 1-3)."""
    cls = {1: 1, 0: 2, -1: 3}
    n_r, n_z = int(round(r_edges[-1] / px)), int(round((z_edges[0] - z_edges[-1]) / px))
    rc = (np.arange(n_r) + 0.5) * px
    zc = z_edges[0] - (np.arange(n_z) + 0.5) * px
    m = np.searchsorted(r_edges, rc, side="right") - 1
    k = np.searchsorted(-z_edges, -zc, side="right") - 1
    half = np.vectorize(cls.get)(state[k][:, m]).astype(np.uint8)
    return np.concatenate([half[:, ::-1], half], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=700.0, help="lethal field [V/cm]")
    ap.add_argument("--out", type=Path, default=RESULTS / "pfa_lesion.rrd")
    args = ap.parse_args()
    thr = args.threshold * 100
    spec, cert, rep = load()
    r_edges, z_edges, index = tile_grid(rep)
    ref_row = int(np.searchsorted(-z_edges, REF_DEPTH, side="right") - 1)

    rr.init("certsim_pfa_lesion")
    rr.save(args.out)
    rr.send_blueprint(rrb.Blueprint(rrb.Horizontal(
        rrb.Spatial3DView(origin="world", name="certified lesion (cut-away)"),
        rrb.Vertical(
            rrb.Spatial2DView(origin="state", name="certified tile states"),
            rrb.Spatial2DView(origin="field", name="solver field at 1000 V (not certified)"),
            rrb.TimeSeriesView(origin="metrics", name="certified extent vs voltage"),
        ),
    )))
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log("state", rr.AnnotationContext([
        (1, "proven lethal", ORANGE), (2, "undecided", ORANGE_LIGHT), (3, "proven spared", BLUE_LIGHT),
    ]), static=True)
    rr.log("readme", rr.TextDocument(__doc__.strip(), media_type=rr.MediaType.TEXT), static=True)
    catheter(spec)
    img, _ = field_image(spec, cert, 1000.0)
    rr.log("field/image", rr.Image(img), static=True)
    for name, color in (("depth_inner_mm", ORANGE), ("depth_outer_mm", ORANGE_LIGHT),
                        ("radius_inner_mm", (0x2A, 0x78, 0xD6)), ("radius_outer_mm", (0x86, 0xB6, 0xEF))):
        rr.log(f"metrics/{name}", rr.SeriesLines(colors=[color], names=[name]), static=True)

    for V in np.arange(250.0, 2501.0, 50.0):
        rr.set_time("voltage", sequence=int(V))
        z_e, inner, outer, s = envelope(rep, V, thr)
        outer_draw = np.where(np.isnan(outer), r_edges[-1], outer)
        for name, radii, rgba in (("proven", inner, ORANGE + (255,)), ("possible", outer_draw, ORANGE_LIGHT + (90,))):
            prof = staircase(z_e, radii) * MM
            if len(prof) > 2:
                rr.log(f"world/lesion/{name}", mesh(*revolve(prof, *CUT, 54), rgba))
            else:
                rr.log(f"world/lesion/{name}", rr.Clear(recursive=False))
        rr.log("world/cut_faces", cut_faces(r_edges, z_edges, s))
        rr.log("state/tiles", rr.SegmentationImage(state_image(r_edges, z_edges, s)))
        e = rep.lesion_extent(V, thr)
        (d_in, d_out), (r_in, r_out) = e["depth"], e["radius"]
        rr.log("metrics/depth_inner_mm", rr.Scalars(d_in * MM))
        rr.log("metrics/radius_inner_mm", rr.Scalars(r_in * MM))
        if d_out is not None:
            rr.log("metrics/depth_outer_mm", rr.Scalars(d_out * MM))
        if r_out is not None:
            rr.log("metrics/radius_outer_mm", rr.Scalars(r_out * MM))
        ref = max(rep.tile_rms_field[n].scale(V).hi for n in index[ref_row]) / 100
        rr.log("metrics/max_field_at_5mm_V_per_cm", rr.Scalars(ref))
    print(f"wrote {args.out}\nopen with: uv run --group viz rerun {args.out}")


if __name__ == "__main__":
    main()
