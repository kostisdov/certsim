"""Plausible one-line bugs in the correct solver of E2 (a mutation study).

Each mutant is the Claude solver with one realistic change. Mutants are written
to samples/mutants/<name>/ with the change recorded, and run_solvers.py then
treats them like any other sample.

    uv run python make_mutants.py
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE / "samples" / "claude-opus-5-5" / "001" / "solver.py"

MUTANTS = {
    "no_r_weight": ("the axisymmetric weight r is missing from the element integrals",
                    "w = (0.25 * h1 * h2 * rq * sig)[:, None, None]",
                    "w = (0.25 * h1 * h2 * sig)[:, None, None]"),
    "swapped_sigma": ("tissue and blood conductivities are swapped",
                      "sig = np.where(zc < 0.0, st, sb)",
                      "sig = np.where(zc < 0.0, sb, st)"),
    "top_not_grounded": ("the top boundary is left insulating instead of grounded",
                         "ground = (I == Nr - 1) | (J == 0) | ((J == Nz - 1) & (I >= ia))",
                         "ground = (I == Nr - 1) | (J == 0)"),
    "side_not_driven": ("only the bottom face of the electrode is held at the voltage",
                        "elec = (I <= ia) & (J >= jg) & (J <= jt)",
                        "elec = (I <= ia) & (J == jg)"),
    "current_without_2pi": ("the total current omits the factor 2 pi (the field is right, the claim is not)",
                            "current = 2.0 * np.pi * float(Rres[elec].sum())",
                            "current = float(Rres[elec].sum())"),
}


def main():
    base = BASE.read_text()
    for name, (what, old, new) in MUTANTS.items():
        assert base.count(old) == 1, name
        out = HERE / "samples" / "mutants" / name
        out.mkdir(parents=True, exist_ok=True)
        (out / "solver.py").write_text(base.replace(old, new))
        (out / "request.json").write_text(json.dumps(
            {"model": "mutant of claude-opus-5-5/001", "mutation": what, "old": old, "new": new}, indent=1))
        print(f"{name}: {what}")


if __name__ == "__main__":
    main()
