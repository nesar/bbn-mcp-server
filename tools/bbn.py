"""Science layer: adapter around PRyMordial.

PRyMordial (Burns, Tait & Valli, EPJC 84 (2024) 86, arXiv:2307.07061) is the
community Python code for Big Bang Nucleosynthesis within and beyond the
Standard Model. It is a clone-and-import package configured through module
globals in PRyM.PRyM_init, and it resolves its nuclear rate tables relative
to the working directory captured at import time. This adapter contains all
of that handling so the tool layer stays plain:

- locates the checkout (scripts/setup_prymordial.py puts it in extern/,
  override with the PRYM_DIR environment variable),
- imports it exactly once, with the working directory temporarily switched
  and stdout diverted to stderr (the stdio MCP transport owns stdout),
- serializes runs behind a lock (PRyM configuration is module-global state),
- reuses the precomputed background/weak-rate tables whenever the run leaves
  the Standard Model thermal history unchanged (delta_neff == 0), which makes
  baryon-density and neutron-lifetime scans ~6x faster.
"""

import contextlib
import os
import sys
import threading
from pathlib import Path

DEFAULT_DIR = Path(__file__).resolve().parents[1] / "extern" / "PRyMordial"

RESULT_KEYS = [
    "Neff",             # effective number of neutrino species after BBN
    "Omega_nu_rel_h2_x1e6",
    "inv_Omega_nu_nr_h2_per_eV",
    "Yp_CMB",           # helium-4 baryonic mass fraction (CMB convention)
    "Yp_BBN",           # helium-4 as 4*Y_He4 (BBN convention, what PDG quotes)
    "D_H_x1e5",         # deuterium over hydrogen x 1e5
    "He3_H_x1e5",       # helium-3 over hydrogen x 1e5
    "Li7_H_x1e10",      # lithium-7 over hydrogen x 1e10
]

_lock = threading.Lock()
_prym = None


def prym_dir() -> Path:
    return Path(os.environ.get("PRYM_DIR", str(DEFAULT_DIR)))


def _load():
    """Import PRyMordial once; returns (PRyM_init, PRyM_main) modules."""
    global _prym
    if _prym is None:
        directory = prym_dir()
        if not (directory / "PRyM" / "PRyM_main.py").exists():
            raise RuntimeError(
                f"PRyMordial not found at {directory}. Run "
                "'python scripts/setup_prymordial.py' once (see README), or "
                "point the PRYM_DIR environment variable at a checkout."
            )
        cwd = os.getcwd()
        sys.path.insert(0, str(directory))
        os.chdir(directory)  # PRyM_init captures its data root from getcwd()
        try:
            with contextlib.redirect_stdout(sys.stderr):
                import PRyM.PRyM_init as PRyMini

                PRyMini.verbose_flag = False
                PRyMini.julia_flag = False
                import PRyM.PRyM_main as PRyMmain
        finally:
            os.chdir(cwd)
        _prym = (PRyMini, PRyMmain)
    return _prym


def cached_tables_exist() -> bool:
    directory = prym_dir()
    return (
        (directory / "PRyMrates" / "thermo" / "Tgamma_Tnu.txt").exists()
        and (directory / "PRyMrates" / "nTOp" / "nTOp_frwrd_HT.txt").exists()
    )


def eta10(omega_b_h2: float) -> float:
    """Baryon-to-photon ratio in units of 1e-10 for a given omega_b h^2."""
    PRyMini, _ = _load()
    return 1.0e10 * PRyMini.Omegabh2_to_eta0b * omega_b_h2


def run_bbn(
    omega_b_h2: float,
    delta_neff: float = 0.0,
    tau_n_s: float = 878.4,
    small_network: bool = False,
) -> dict[str, float]:
    """Run one BBN computation and return the abundance dictionary.

    small_network=True uses the 12-reaction network (D/H shifts by ~0.2%),
    appropriate for scans; single evaluations use all 63 reactions.
    """
    PRyMini, PRyMmain = _load()
    with _lock:
        reuse = (delta_neff == 0.0) and cached_tables_exist()
        PRyMini.smallnet_flag = small_network
        PRyMini.compute_bckg_flag = not reuse
        PRyMini.compute_nTOp_flag = not reuse
        PRyMini.save_bckg_flag = False
        PRyMini.save_nTOp_flag = False
        PRyMini.tau_n_flag = True
        PRyMini.Omegabh2 = omega_b_h2
        PRyMini.eta0b = PRyMini.Omegabh2_to_eta0b * omega_b_h2
        PRyMini.DeltaNeff = delta_neff
        PRyMini.tau_n = tau_n_s
        with contextlib.redirect_stdout(sys.stderr):
            res = PRyMmain.PRyMclass().PRyMresults()
    return {key: float(value) for key, value in zip(RESULT_KEYS, res)}
