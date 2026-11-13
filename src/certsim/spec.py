"""Problem specification for an axisymmetric focal pulsed-field-ablation electrode.

The specification is the trusted statement of the physics problem. The solver
and the checker both read it; nothing the solver returns is believed unless it
matches the specification.

Model (cylindrical coordinates r, z; everything is rotationally symmetric):

* myocardium fills z < 0, blood fills z > 0;
* a metal electrode of radius ``a`` occupies 0 <= r <= a, g <= z <= g + L and
  is held at ``voltage``; its flat tip sits a gap g >= 0 above the tissue
  surface z = 0, with blood in between (g = 0: the tip touches the tissue);
* an insulating catheter shaft of the same radius continues from z = g + L to
  the top of the domain;
* the outer boundary r = R, z = -H, z = Ztop is grounded (a distant return).

Bipolar catheter (``ring_spacing`` set): an insulating spacer of that length
follows the tip electrode, then a ring electrode of length ``ring_length`` on
r = a, then the insulating shaft. The tip is held at +voltage/2 and the ring
at -voltage/2, so ``voltage`` is the applied potential difference. A bipolar
generator is floating, so the far boundary is insulating instead of grounded:
the axis, the far boundary and the shaft above the ring then form one
connected insulating wall, and no net current leaves through the far field.

The potential u solves div(sigma grad u) = 0. All boundaries of the domain are
axis-aligned, so every quantity lives on a rectangular "box" grid.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Box:
    """Axis-aligned region in the (r, z) half-plane, swept around the axis."""

    r_lo: float
    r_hi: float
    z_lo: float
    z_hi: float


@dataclass(frozen=True)
class FocalPFASpec:
    a: float = 1.25e-3  # electrode radius [m]
    L: float = 3.0e-3  # exposed electrode length [m]
    R: float = 50e-3  # radius of the grounded outer boundary [m]
    H: float = 50e-3  # depth of the grounded bottom boundary [m]
    Ztop: float = 50e-3  # height of the grounded top boundary [m]
    sigma_tissue: float = 0.25  # myocardium conductivity [S/m]
    sigma_blood: float = 0.7  # blood conductivity [S/m]
    voltage: float = 1.0  # electrode potential [V]; the problem is linear in it
    gap: float = 0.0  # blood layer between electrode tip and tissue [m]
    ring_spacing: float | None = None  # bipolar: insulated length between tip and ring [m]
    ring_length: float = 2e-3  # bipolar: length of the ring electrode [m]
    voxels: tuple[Box, ...] = ()  # regions whose mean field is certified
    regions: tuple[Box, ...] = ()  # regions whose RMS field is certified
    tiles: tuple[Box, ...] = ()  # lesion-map tiles whose RMS field is certified

    @property
    def tip(self) -> float:
        """Height of the electrode tip above the tissue surface."""
        return self.gap

    @property
    def top(self) -> float:
        """Height where the electrode ends and the insulating shaft begins."""
        return self.gap + self.L

    @property
    def bipolar(self) -> bool:
        return self.ring_spacing is not None

    @property
    def ring_lo(self) -> float:
        return self.top + self.ring_spacing

    @property
    def ring_hi(self) -> float:
        return self.ring_lo + self.ring_length

    @property
    def tip_voltage(self) -> float:
        return self.voltage / 2 if self.bipolar else self.voltage

    @property
    def ring_voltage(self) -> float:
        return -self.voltage / 2

    @property
    def boxes(self) -> tuple[Box, ...]:
        return self.voxels + self.regions + self.tiles

    def breakpoints_r(self) -> set[float]:
        pts = {0.0, self.a, self.R}
        for b in self.boxes:
            pts |= {b.r_lo, b.r_hi}
        return pts

    def breakpoints_z(self) -> set[float]:
        pts = {-self.H, 0.0, self.tip, self.top, self.Ztop}
        if self.bipolar:
            pts |= {self.ring_lo, self.ring_hi}
        for b in self.boxes:
            pts |= {b.z_lo, b.z_hi}
        return pts


def axis_voxels(radius: float, step: float, depth: float) -> tuple[Box, ...]:
    """Stack of small cylinders on the axis below the electrode tip."""
    n = int(round(depth / step))
    return tuple(Box(0.0, radius, -(k + 1) * step, -k * step) for k in range(n))


def lesion_tiles(r_max: float, depth: float, step: float) -> tuple[Box, ...]:
    """Square tiles covering 0 <= r <= r_max, -depth <= z <= 0."""
    nr, nz = int(round(r_max / step)), int(round(depth / step))
    return tuple(
        Box(i * step, (i + 1) * step, -(k + 1) * step, -k * step) for k in range(nz) for i in range(nr)
    )


def default_spec(**overrides) -> FocalPFASpec:
    params = dict(
        voxels=axis_voxels(radius=0.25e-3, step=0.5e-3, depth=8e-3),
        regions=(Box(0.0, 5e-3, -10e-3, -6e-3),),
    )
    params.update(overrides)
    return FocalPFASpec(**params)


# --- geometry rules shared by solver and checker ------------------------------


def tissue_cells(r: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Cells in the tissue half-space z <= 0, by geometry, indexed [j, i] as in element_masks."""
    return np.broadcast_to((np.asarray(z)[1:] <= 0.0)[:, None], (len(z) - 1, len(r) - 1))


def element_masks(spec: FocalPFASpec, r: np.ndarray, z: np.ndarray):
    """Active-element mask and per-element conductivity on a tensor grid.

    Elements are indexed [j, i] for the cell [r[i], r[i+1]] x [z[j], z[j+1]].
    A cell is removed when it lies inside the electrode or the shaft.
    """
    r_hi = r[1:][None, :]
    z_lo = z[:-1][:, None]
    z_hi = z[1:][:, None]
    inside_catheter = (r_hi <= spec.a) & (z_lo >= spec.tip)
    active = ~inside_catheter
    sigma = np.where(tissue_cells(r, z), spec.sigma_tissue, spec.sigma_blood)
    return active, sigma


def node_masks(spec: FocalPFASpec, r: np.ndarray, z: np.ndarray, degree: int = 1):
    """Boundary classification of nodes, as arrays over the degree-p node grid.

    For degree p every cell carries (p+1) x (p+1) nodes, so the node grid has
    shape (p*(nz-1)+1, p*(nr-1)+1) and grid node [j, i] is node [p*j, p*i].
    A node inside a cell edge gets a boundary label when both end points of
    the edge carry it; for this geometry that is exactly the boundary edges.
    """
    coarse = _coarse_node_masks(spec, r, z)
    if degree == 1:
        return coarse
    p = degree
    active_el, _ = element_masks(spec, r, z)
    nz, nr = len(z), len(r)
    shape = (p * (nz - 1) + 1, p * (nr - 1) + 1)
    out = {}
    active = np.zeros(shape, dtype=bool)
    for j, i in zip(*np.nonzero(active_el)):
        active[p * j : p * j + p + 1, p * i : p * i + p + 1] = True
    out["active"] = active
    inner = np.arange(1, p)
    for name in ("electrode", "ring", "ground", "outer", "axis", "spacer", "shaft"):
        m = coarse[name]
        f = np.zeros(shape, dtype=bool)
        f[::p, ::p] = m
        jj, ii = np.nonzero(m[:, :-1] & m[:, 1:])  # horizontal edges
        f[(p * jj)[:, None], (p * ii)[:, None] + inner[None, :]] = True
        jj, ii = np.nonzero(m[:-1, :] & m[1:, :])  # vertical edges
        f[(p * jj)[:, None] + inner[None, :], (p * ii)[:, None]] = True
        out[name] = f & active
    return out


def _coarse_node_masks(spec: FocalPFASpec, r: np.ndarray, z: np.ndarray):
    """Boundary classification of grid nodes, as arrays of shape (nz, nr)."""
    active_el, _ = element_masks(spec, r, z)
    nz, nr = len(z), len(r)
    active = np.zeros((nz, nr), dtype=bool)
    for dj in (0, 1):
        for di in (0, 1):
            active[dj : nz - 1 + dj, di : nr - 1 + di] |= active_el
    R, Z = np.meshgrid(r, z)
    electrode = active & (
        ((Z == spec.tip) & (R <= spec.a)) | ((R == spec.a) & (Z >= spec.tip) & (Z <= spec.top))
    )
    outer = active & ((R == spec.R) | (Z == -spec.H) | (Z == spec.Ztop))
    ground = np.zeros_like(active) if spec.bipolar else outer
    axis = active & (R == 0.0)
    side = active & (R == spec.a)
    if spec.bipolar:
        ring = side & (Z >= spec.ring_lo) & (Z <= spec.ring_hi)
        spacer = side & (Z >= spec.top) & (Z <= spec.ring_lo)
        shaft = side & (Z >= spec.ring_hi)
    else:
        ring = spacer = np.zeros_like(active)
        shaft = side & (Z >= spec.top)
    return {
        "active": active,
        "electrode": electrode,
        "ring": ring,
        "ground": ground,  # grounded outer boundary (monopolar only)
        "outer": outer,
        "axis": axis,
        "spacer": spacer,  # insulator between tip and ring (bipolar only)
        "shaft": shaft,
    }
