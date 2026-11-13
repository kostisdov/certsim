"""Grid convergence index: the standard mesh-convergence estimate certificates are compared with.

Not part of the trusted checker; used by the experiments of both papers.
"""

import numpy as np


def gci(phi, r=2.0, fs=1.25):
    """Celik et al. (2008) three-grid procedure; phi ordered fine -> coarse.

    The convergence ratio R = e21 / e32 classifies the triplet (Stern et al. 2001):
    0 < R < 1 is monotone convergence, the only case in which the GCI is valid;
    R > 1 is divergence and R < 0 oscillation. The band is also returned for the
    other cases, as a practitioner applying the formula regardless would obtain it.
    """
    f1, f2, f3 = phi
    e21, e32 = f2 - f1, f3 - f2
    R = e21 / e32
    kind = "converging" if 0 < R < 1 else ("diverging" if R >= 1 else "oscillating")
    p = abs(np.log(abs(e32 / e21))) / np.log(r)
    ext = (r**p * f1 - f2) / (r**p - 1)
    g = fs * abs((f1 - f2) / f1) / (r**p - 1)
    return {"R": R, "kind": kind, "p": p, "ext": ext, "band": (f1 * (1 - g), f1 * (1 + g))}
