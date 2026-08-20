#!/usr/bin/env python
"""Fetch PRyMordial and precompute its cached tables. Run once after install.

PRyMordial (Burns, Tait & Valli, EPJC 84 (2024) 86, arXiv:2307.07061) is not
on PyPI — it is a clone-and-import package, GPL-3.0 licensed, so we keep it
out of this repository and fetch it here instead:

    python scripts/setup_prymordial.py

This clones a pinned commit into extern/PRyMordial (override the location
with the PRYM_DIR environment variable) and then runs one Standard Model
BBN computation with the save flags on, so the thermodynamic background and
the n<->p weak rates are stored on disk. The MCP tools reuse those tables,
which makes parameter scans ~6x faster and lets the server run read-only.
"""

import os
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/vallima/PRyMordial"
PINNED_COMMIT = "725d8a8db3ad5ea2630580d825c9d0d69ed76533"  # Aug 2026 head
DEFAULT_DIR = Path(__file__).resolve().parents[1] / "extern" / "PRyMordial"


def main() -> None:
    target = Path(os.environ.get("PRYM_DIR", DEFAULT_DIR))

    if (target / "PRyM" / "PRyM_main.py").exists():
        print(f"PRyMordial already present at {target}")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        print(f"Cloning {REPO_URL} -> {target}")
        subprocess.run(["git", "clone", REPO_URL, str(target)], check=True)
        subprocess.run(["git", "-C", str(target), "checkout", PINNED_COMMIT],
                       check=True)

    thermo = target / "PRyMrates" / "thermo" / "Tgamma_Tnu.txt"
    ntop = target / "PRyMrates" / "nTOp" / "nTOp_frwrd_HT.txt"
    if thermo.exists() and ntop.exists():
        print("Cached tables already present — nothing to do.")
        return

    print("Precomputing background thermodynamics and weak rates "
          "(one SM run, takes ~1 minute)...")
    os.chdir(target)
    sys.path.insert(0, str(target))
    import PRyM.PRyM_init as PRyMini

    PRyMini.verbose_flag = False
    PRyMini.save_bckg_flag = True
    PRyMini.save_nTOp_flag = True
    import PRyM.PRyM_main as PRyMmain

    res = PRyMmain.PRyMclass().PRyMresults()
    if not (thermo.exists() and ntop.exists()):
        raise RuntimeError("PRyMordial did not write its cache tables.")
    print(f"Done. SM check: Neff = {res[0]:.3f} (expect 3.044), "
          f"Yp = {res[4]:.4f} (expect ~0.247), "
          f"D/H = {res[5]:.3f}e-5 (expect ~2.5).")


if __name__ == "__main__":
    main()
