"""MCP tool functions for Big Bang Nucleosynthesis.

Plain Python functions — no MCP imports. The type hints, Field constraints,
and docstrings below become the MCP tool schema that agents see.

Data flows between tools as CSV file paths: the scan_* tools write one CSV
each, the plot_* and fit_* tools read them back. Only small numbers and file
paths ever pass through the LLM context.
"""

import csv
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
from pydantic import BaseModel, Field, validate_call

from .bbn import eta10, run_bbn

# Observational anchors, quoted everywhere predictions are compared.
# theory_error is the nuclear-rate systematic on the PREDICTION (PRyMordial
# runs its default PRIMAT rate compilation; the d+p radiative capture rate
# alone moves D/H by ~2-3%, see the PRyMordial paper, arXiv:2307.07061).
# Pulls and fits use the measurement and theory errors in quadrature.
OBSERVATIONS = {
    "Yp_BBN": {
        "value": 0.2453,
        "error": 0.0034,
        "theory_error": 0.0003,
        "label": "primordial helium-4 mass fraction",
        "reference": "Aver et al., JCAP 03 (2021) 027 (metal-poor H II regions)",
    },
    "D_H_x1e5": {
        "value": 2.547,
        "error": 0.025,
        "theory_error": 0.06,
        "label": "primordial deuterium D/H x 1e5",
        "reference": "Cooke, Pettini & Steidel, ApJ 855 (2018) 102 (quasar absorbers)",
    },
    "Li7_H_x1e10": {
        "value": 1.6,
        "error": 0.3,
        "theory_error": 0.5,
        "label": "lithium-7 Li/H x 1e10 (Spite plateau)",
        "reference": "Sbordone et al., A&A 522 (2010) A26",
    },
}


def _total_error(obs: dict) -> float:
    """Measurement and nuclear-rate theory errors in quadrature."""
    return float(np.hypot(obs["error"], obs.get("theory_error", 0.0)))

CMB_OMEGA_B = {
    "value": 0.02237,
    "error": 0.00015,
    "reference": "Planck 2018 (TT,TE,EE+lowE+lensing), A&A 641 (2020) A6",
}

NEUTRON_LIFETIME_S = {
    "pdg_average": {"value": 878.4, "error": 0.5,
                    "reference": "PDG 2024 average (bottle-dominated)"},
    "bottle_ucn_tau": {"value": 877.75, "error": 0.36,
                       "reference": "UCNtau, PRL 127 (2021) 162501"},
    "beam_bl1": {"value": 887.7, "error": 2.2,
                 "reference": "Yue et al., PRL 111 (2013) 222501"},
}

SCAN_BARYON_COLUMNS = ["omega_b_h2", "eta10", "Yp_BBN", "D_H_x1e5",
                       "He3_H_x1e5", "Li7_H_x1e10"]
SCAN_NEFF_COLUMNS = ["delta_neff", "Neff", "Yp_BBN", "D_H_x1e5", "Li7_H_x1e10"]
SCAN_TAU_COLUMNS = ["tau_n_s", "Yp_BBN", "D_H_x1e5"]


class ArtifactResult(BaseModel):
    """Uniform result contract returned by every tool."""

    status: Literal["success"]
    files: list[str]
    message: str
    metadata: dict[str, Any]


def _write_csv(path: Path, source: str, columns: list[str], rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        f.write(f"# source: {source}\n")
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(rows)


def _read_csv(path_str: str, expected_columns: list[str]) -> tuple[str, dict[str, np.ndarray]]:
    """Read a CSV written by _write_csv, checking it has the right columns."""
    path = Path(path_str).expanduser().resolve()
    lines = path.read_text(encoding="utf-8").splitlines()
    source = lines[0].removeprefix("# source:").strip() if lines[0].startswith("#") else path.stem
    header = lines[1].split(",") if lines[0].startswith("#") else lines[0].split(",")
    missing = [c for c in expected_columns if c not in header]
    if missing:
        raise ValueError(
            f"{path.name} is missing columns {missing} (found {header}). "
            "Pass a CSV produced by the matching scan_* tool."
        )
    data = np.loadtxt(path, delimiter=",", skiprows=2 if lines[0].startswith("#") else 1).T
    return source, {name: col for name, col in zip(header, np.atleast_2d(data))}


def _outdir(output_dir: str) -> Path:
    outdir = Path(output_dir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def _pulls(result: dict[str, float]) -> dict[str, dict[str, float | str]]:
    """Compare predictions with the observational anchors, in sigma."""
    comparison = {}
    for key, obs in OBSERVATIONS.items():
        if key in result:
            pull = (result[key] - obs["value"]) / _total_error(obs)
            comparison[key] = {
                "predicted": round(result[key], 4),
                "observed": obs["value"],
                "obs_error": obs["error"],
                "theory_error": obs.get("theory_error", 0.0),
                "pull_sigma": round(pull, 2),
                "reference": obs["reference"],
            }
    return comparison


@validate_call
def describe_bbn_inputs() -> ArtifactResult:
    """Explain the physics inputs, observables, and stories of the BBN tools.

    Use this tool first. It lists the three tunable parameters (baryon
    density, extra relativistic species, neutron lifetime), the measured
    primordial abundances the predictions are compared against, and the
    classic questions these tools can answer — the CMB/BBN concordance test,
    counting light particle species via N_eff, the neutron-lifetime puzzle,
    and the lithium problem.
    """
    return ArtifactResult(
        status="success",
        files=[],
        message=(
            "BBN with PRyMordial: predict light-element abundances from the "
            "first three minutes and confront them with astronomical data."
        ),
        metadata={
            "parameters": {
                "omega_b_h2": {
                    "meaning": "physical baryon density Omega_b h^2 (sets the "
                               "baryon-to-photon ratio eta)",
                    "cmb_value": CMB_OMEGA_B,
                },
                "delta_neff": {
                    "meaning": "extra relativistic energy density beyond the "
                               "three SM neutrinos (dark radiation, e.g. a "
                               "light sterile state, axions, hidden photons); "
                               "SM prediction is Neff = 3.044",
                },
                "tau_n_s": {
                    "meaning": "neutron lifetime in seconds; normalizes the "
                               "weak rates that set the n/p ratio at freeze-out",
                    "measurements": NEUTRON_LIFETIME_S,
                    "puzzle": "bottle and beam experiments disagree by ~10 s "
                              "(~4 sigma) — BBN helium is an independent probe",
                },
            },
            "observed_abundances": OBSERVATIONS,
            "classic_questions": [
                "Does the baryon density measured in the CMB predict the "
                "deuterium abundance seen in quasar spectra? (concordance)",
                "How many light particle species were relativistic during "
                "the first minutes? (Neff)",
                "Which neutron lifetime do the primordial abundances prefer?",
                "Why is predicted Li7 ~3x the Spite plateau? (the unsolved "
                "lithium problem)",
            ],
            "software": "PRyMordial, Burns, Tait & Valli, EPJC 84 (2024) 86, "
                        "arXiv:2307.07061 (fetched by scripts/setup_prymordial.py)",
            "runtime_note": "compute_abundances takes seconds; scans take up "
                            "to ~1 minute — do not retry while one is running.",
        },
    )


@validate_call
def compute_abundances(
    omega_b_h2: Annotated[float, Field(ge=0.005, le=0.05)] = 0.02237,
    delta_neff: Annotated[float, Field(ge=-2.0, le=3.0)] = 0.0,
    tau_n_s: Annotated[float, Field(ge=800.0, le=1000.0)] = 878.4,
) -> ArtifactResult:
    """Compute the primordial light-element abundances for one set of inputs.

    Runs the full 63-reaction PRyMordial network and returns Neff, the
    helium-4 mass fraction Yp, D/H, He3/H and Li7/H, each compared with the
    measured primordial values (pull in sigma). Use scan_* tools to explore a
    parameter range instead of calling this repeatedly.

    Args:
        omega_b_h2: Physical baryon density Omega_b h^2 (CMB: 0.02237).
        delta_neff: Extra relativistic species beyond the SM (0 = none).
        tau_n_s: Neutron lifetime in seconds (PDG average: 878.4).
    """
    result = run_bbn(omega_b_h2, delta_neff, tau_n_s, small_network=False)
    return ArtifactResult(
        status="success",
        files=[],
        message=(
            f"BBN with omega_b h^2={omega_b_h2:g}, dNeff={delta_neff:g}, "
            f"tau_n={tau_n_s:g}s: Yp={result['Yp_BBN']:.4f}, "
            f"D/H={result['D_H_x1e5']:.3f}e-5, Neff={result['Neff']:.3f}."
        ),
        metadata={
            "inputs": {"omega_b_h2": omega_b_h2, "eta10": round(eta10(omega_b_h2), 3),
                       "delta_neff": delta_neff, "tau_n_s": tau_n_s},
            "predictions": {k: round(v, 4) for k, v in result.items()},
            "comparison_with_observations": _pulls(result),
        },
    )


@validate_call
def scan_baryon_density(
    output_dir: Annotated[str, Field(min_length=1)],
    omega_b_h2_min: Annotated[float, Field(ge=0.005)] = 0.010,
    omega_b_h2_max: Annotated[float, Field(le=0.05)] = 0.032,
    n_points: Annotated[int, Field(ge=5, le=30)] = 12,
    delta_neff: Annotated[float, Field(ge=-2.0, le=3.0)] = 0.0,
    tau_n_s: Annotated[float, Field(ge=800.0, le=1000.0)] = 878.4,
) -> ArtifactResult:
    """Scan the baryon density and tabulate the light-element abundances.

    Writes a CSV (columns omega_b_h2, eta10, Yp_BBN, D_H_x1e5, He3_H_x1e5,
    Li7_H_x1e10) for plot_abundance_curves and fit_baryon_density. This is
    the calculation behind the classic 'Schramm plot'. Uses the 12-reaction
    network (abundance shifts ~0.2% vs the full one). Takes ~10-60 seconds.

    Args:
        output_dir: Directory where the CSV is written.
        omega_b_h2_min: Lower end of the Omega_b h^2 range.
        omega_b_h2_max: Upper end of the Omega_b h^2 range.
        n_points: Number of scan points.
        delta_neff: Extra relativistic species held fixed during the scan.
        tau_n_s: Neutron lifetime in seconds held fixed during the scan.
    """
    if omega_b_h2_min >= omega_b_h2_max:
        raise ValueError("omega_b_h2_min must be smaller than omega_b_h2_max.")
    grid = np.linspace(omega_b_h2_min, omega_b_h2_max, n_points)
    rows = []
    for ob in grid:
        r = run_bbn(float(ob), delta_neff, tau_n_s, small_network=True)
        rows.append((ob, eta10(float(ob)), r["Yp_BBN"], r["D_H_x1e5"],
                     r["He3_H_x1e5"], r["Li7_H_x1e10"]))
    csv_path = _outdir(output_dir) / "bbn_baryon_scan.csv"
    _write_csv(csv_path,
               f"PRyMordial baryon-density scan, dNeff={delta_neff:g}, tau_n={tau_n_s:g}s",
               SCAN_BARYON_COLUMNS, rows)
    return ArtifactResult(
        status="success",
        files=[str(csv_path)],
        message=(
            f"Scanned omega_b h^2 over [{omega_b_h2_min:g}, {omega_b_h2_max:g}] "
            f"({n_points} points)."
        ),
        metadata={
            "columns": SCAN_BARYON_COLUMNS,
            "delta_neff": delta_neff,
            "tau_n_s": tau_n_s,
            "cmb_omega_b_h2": CMB_OMEGA_B,
        },
    )


@validate_call
def plot_abundance_curves(
    scan_file: Annotated[str, Field(min_length=1)],
    output_dir: Annotated[str, Field(min_length=1)],
) -> ArtifactResult:
    """Draw the Schramm plot: abundances vs baryon density, data over theory.

    Use this tool after scan_baryon_density. Three stacked panels — Yp, D/H,
    Li7/H — as functions of Omega_b h^2, each with its measured primordial
    band, plus the CMB baryon density as a vertical band. Where the curves
    cross the bands is the BBN determination of the baryon density; the Li7
    panel shows the lithium problem.

    Args:
        scan_file: CSV written by scan_baryon_density.
        output_dir: Directory where the PNG is written.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _, d = _read_csv(scan_file, SCAN_BARYON_COLUMNS)
    fig, axes = plt.subplots(3, 1, figsize=(7.5, 10), sharex=True,
                             gridspec_kw={"hspace": 0.08})
    panels = [
        ("Yp_BBN", r"$Y_p$", "linear"),
        ("D_H_x1e5", r"$10^5\,\mathrm{D/H}$", "linear"),
        ("Li7_H_x1e10", r"$10^{10}\,{}^7\mathrm{Li/H}$", "linear"),
    ]
    for ax, (col, label, yscale) in zip(axes, panels):
        ax.plot(d["omega_b_h2"], d[col], color="C0", linewidth=2,
                label="BBN prediction (PRyMordial)")
        obs = OBSERVATIONS[col]
        err = _total_error(obs)
        ax.axhspan(obs["value"] - err, obs["value"] + err,
                   color="C2", alpha=0.35,
                   label="observed (incl. rate syst.)")
        ax.axvspan(CMB_OMEGA_B["value"] - CMB_OMEGA_B["error"],
                   CMB_OMEGA_B["value"] + CMB_OMEGA_B["error"],
                   color="C3", alpha=0.5,
                   label=r"CMB $\Omega_b h^2$ (Planck)")
        ax.set_ylabel(label)
        ax.set_yscale(yscale)
    axes[0].set_title("Primordial abundances vs baryon density")
    axes[0].legend(fontsize="small", loc="lower right")
    axes[-1].set_xlabel(r"$\Omega_b h^2$")

    secax = axes[0].secondary_xaxis(
        "top",
        functions=(lambda ob: eta10(1.0) * ob, lambda e: e / eta10(1.0)),
    )
    secax.set_xlabel(r"$\eta_{10}$ (baryon-to-photon ratio $\times 10^{10}$)")

    plot_path = _outdir(output_dir) / "bbn_abundance_curves.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return ArtifactResult(
        status="success",
        files=[str(plot_path)],
        message="Plotted Yp, D/H and Li7/H vs baryon density with observed bands.",
        metadata={
            "observed_bands": {k: OBSERVATIONS[k] for k, _, _ in panels},
            "cmb_band": CMB_OMEGA_B,
        },
    )


@validate_call
def scan_neff(
    output_dir: Annotated[str, Field(min_length=1)],
    delta_neff_min: Annotated[float, Field(ge=-2.0)] = -1.0,
    delta_neff_max: Annotated[float, Field(le=3.0)] = 2.0,
    n_points: Annotated[int, Field(ge=3, le=9)] = 5,
    omega_b_h2: Annotated[float, Field(ge=0.005, le=0.05)] = 0.02237,
    tau_n_s: Annotated[float, Field(ge=800.0, le=1000.0)] = 878.4,
) -> ArtifactResult:
    """Scan extra relativistic species (dark radiation) and tabulate abundances.

    Writes a CSV (columns delta_neff, Neff, Yp_BBN, D_H_x1e5, Li7_H_x1e10)
    for plot_neff_impact. Each point re-solves the full thermal history —
    neutrino decoupling and expansion rate change with delta_neff — so this
    takes ~10 seconds per point. Keep n_points small.

    Args:
        output_dir: Directory where the CSV is written.
        delta_neff_min: Lower end of the extra-species range (must be > -3).
        delta_neff_max: Upper end of the range.
        n_points: Number of scan points (each is a full recomputation).
        omega_b_h2: Baryon density held fixed during the scan.
        tau_n_s: Neutron lifetime in seconds held fixed during the scan.
    """
    if delta_neff_min >= delta_neff_max:
        raise ValueError("delta_neff_min must be smaller than delta_neff_max.")
    grid = np.linspace(delta_neff_min, delta_neff_max, n_points)
    rows = []
    for dn in grid:
        r = run_bbn(omega_b_h2, float(dn), tau_n_s, small_network=True)
        rows.append((dn, r["Neff"], r["Yp_BBN"], r["D_H_x1e5"], r["Li7_H_x1e10"]))
    csv_path = _outdir(output_dir) / "bbn_neff_scan.csv"
    _write_csv(csv_path,
               f"PRyMordial dNeff scan, omega_b_h2={omega_b_h2:g}, tau_n={tau_n_s:g}s",
               SCAN_NEFF_COLUMNS, rows)
    return ArtifactResult(
        status="success",
        files=[str(csv_path)],
        message=(
            f"Scanned delta_neff over [{delta_neff_min:g}, {delta_neff_max:g}] "
            f"({n_points} points)."
        ),
        metadata={"columns": SCAN_NEFF_COLUMNS, "omega_b_h2": omega_b_h2,
                  "tau_n_s": tau_n_s, "sm_neff": 3.044},
    )


@validate_call
def plot_neff_impact(
    scan_file: Annotated[str, Field(min_length=1)],
    output_dir: Annotated[str, Field(min_length=1)],
) -> ArtifactResult:
    """Plot how extra relativistic species shift helium and deuterium.

    Use this tool after scan_neff. Two stacked panels: Yp and D/H vs
    delta_neff, each with the measured primordial band — where the prediction
    exits a band, that amount of dark radiation is excluded. This is how BBN
    counts the light degrees of freedom of the universe at t ~ 1 s.

    Args:
        scan_file: CSV written by scan_neff.
        output_dir: Directory where the PNG is written.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _, d = _read_csv(scan_file, SCAN_NEFF_COLUMNS)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 8), sharex=True,
                                   gridspec_kw={"hspace": 0.08})
    for ax, col, label in ((ax1, "Yp_BBN", r"$Y_p$"),
                           (ax2, "D_H_x1e5", r"$10^5\,\mathrm{D/H}$")):
        ax.plot(d["delta_neff"], d[col], color="C0", linewidth=2, marker="o",
                label="BBN prediction")
        obs = OBSERVATIONS[col if col in OBSERVATIONS else "Yp_BBN"]
        err = _total_error(obs)
        ax.axhspan(obs["value"] - err, obs["value"] + err,
                   color="C2", alpha=0.35,
                   label="observed (incl. rate syst.)")
        ax.axvline(0.0, color="black", linewidth=1, linestyle=":")
        ax.set_ylabel(label)
        ax.legend(fontsize="small")
    ax1.set_title("Dark radiation and the first-minutes expansion rate")
    ax2.set_xlabel(r"$\Delta N_{\rm eff}$ (extra relativistic species)")

    plot_path = _outdir(output_dir) / "bbn_neff_impact.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return ArtifactResult(
        status="success",
        files=[str(plot_path)],
        message="Plotted Yp and D/H vs delta_neff with observed bands.",
        metadata={"sm_point": "delta_neff = 0 (Neff = 3.044)"},
    )


@validate_call
def scan_neutron_lifetime(
    output_dir: Annotated[str, Field(min_length=1)],
    tau_n_min_s: Annotated[float, Field(ge=800.0)] = 866.0,
    tau_n_max_s: Annotated[float, Field(le=1000.0)] = 896.0,
    n_points: Annotated[int, Field(ge=5, le=25)] = 11,
    omega_b_h2: Annotated[float, Field(ge=0.005, le=0.05)] = 0.02237,
) -> ArtifactResult:
    """Scan the neutron lifetime and tabulate helium-4 and deuterium.

    Writes a CSV (columns tau_n_s, Yp_BBN, D_H_x1e5) for
    plot_neutron_lifetime_impact. The neutron lifetime sets the n/p ratio
    at weak freeze-out and is the dominant nuclear-physics uncertainty on
    Yp — and its laboratory measurements currently disagree (bottle vs beam).

    Args:
        output_dir: Directory where the CSV is written.
        tau_n_min_s: Lower end of the lifetime range in seconds.
        tau_n_max_s: Upper end of the lifetime range in seconds.
        n_points: Number of scan points.
        omega_b_h2: Baryon density held fixed during the scan.
    """
    if tau_n_min_s >= tau_n_max_s:
        raise ValueError("tau_n_min_s must be smaller than tau_n_max_s.")
    grid = np.linspace(tau_n_min_s, tau_n_max_s, n_points)
    rows = []
    for tau in grid:
        r = run_bbn(omega_b_h2, 0.0, float(tau), small_network=True)
        rows.append((tau, r["Yp_BBN"], r["D_H_x1e5"]))
    csv_path = _outdir(output_dir) / "bbn_tau_n_scan.csv"
    _write_csv(csv_path,
               f"PRyMordial neutron-lifetime scan, omega_b_h2={omega_b_h2:g}",
               SCAN_TAU_COLUMNS, rows)
    return ArtifactResult(
        status="success",
        files=[str(csv_path)],
        message=(
            f"Scanned tau_n over [{tau_n_min_s:g}, {tau_n_max_s:g}] s "
            f"({n_points} points)."
        ),
        metadata={"columns": SCAN_TAU_COLUMNS,
                  "laboratory_measurements": NEUTRON_LIFETIME_S},
    )


@validate_call
def plot_neutron_lifetime_impact(
    scan_file: Annotated[str, Field(min_length=1)],
    output_dir: Annotated[str, Field(min_length=1)],
) -> ArtifactResult:
    """Plot primordial helium vs neutron lifetime, with bottle and beam bands.

    Use this tool after scan_neutron_lifetime. Shows Yp(tau_n) crossing the
    observed helium band, with the discrepant bottle and beam laboratory
    measurements marked — a cosmological angle on a live particle-physics
    puzzle.

    Args:
        scan_file: CSV written by scan_neutron_lifetime.
        output_dir: Directory where the PNG is written.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _, d = _read_csv(scan_file, SCAN_TAU_COLUMNS)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(d["tau_n_s"], d["Yp_BBN"], color="C0", linewidth=2,
            label=r"BBN prediction $Y_p(\tau_n)$")
    obs = OBSERVATIONS["Yp_BBN"]
    ax.axhspan(obs["value"] - obs["error"], obs["value"] + obs["error"],
               color="C2", alpha=0.35, label="observed primordial helium")
    for key, style in (("bottle_ucn_tau", "C3"), ("beam_bl1", "C1")):
        m = NEUTRON_LIFETIME_S[key]
        ax.axvspan(m["value"] - m["error"], m["value"] + m["error"],
                   color=style, alpha=0.4,
                   label=f"{key.split('_')[0]} experiment: "
                         rf"$\tau_n = {m['value']:g} \pm {m['error']:g}$ s")
    ax.set_xlabel(r"$\tau_n$ [s]")
    ax.set_ylabel(r"$Y_p$")
    ax.set_title("The neutron-lifetime puzzle seen from the early universe")
    ax.legend(fontsize="small")

    plot_path = _outdir(output_dir) / "bbn_tau_n_impact.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return ArtifactResult(
        status="success",
        files=[str(plot_path)],
        message="Plotted Yp vs tau_n with observed helium and lab measurement bands.",
        metadata={"laboratory_measurements": NEUTRON_LIFETIME_S},
    )


@validate_call
def fit_baryon_density(
    scan_file: Annotated[str, Field(min_length=1)],
    observables: Annotated[list[Literal["D_H_x1e5", "Yp_BBN"]],
                           Field(min_length=1)] = ["D_H_x1e5", "Yp_BBN"],
) -> ArtifactResult:
    """Determine the baryon density preferred by the primordial abundances.

    Use this tool after scan_baryon_density. Builds chi^2(Omega_b h^2)
    against the chosen measured abundances (deuterium is the precision
    baryometer; helium adds a weak cross-check), reports the best-fit value
    with its 1-sigma interval, and compares it with the CMB measurement —
    the BBN/CMB concordance test in one number.

    Args:
        scan_file: CSV written by scan_baryon_density (make sure the scan
            range brackets the minimum, e.g. 0.010-0.032).
        observables: Which measured abundances to include in the fit.
    """
    _, d = _read_csv(scan_file, SCAN_BARYON_COLUMNS)
    ob_grid = d["omega_b_h2"]
    fine = np.linspace(ob_grid.min(), ob_grid.max(), 2000)
    chi2 = np.zeros_like(fine)
    for name in observables:
        obs = OBSERVATIONS[name]
        pred = np.interp(fine, ob_grid, d[name])
        chi2 += ((pred - obs["value"]) / _total_error(obs)) ** 2

    i_best = int(np.argmin(chi2))
    if i_best in (0, len(fine) - 1):
        raise ValueError(
            "The chi^2 minimum sits at the edge of the scanned range — "
            "re-run scan_baryon_density with a wider omega_b_h2 range."
        )
    best = float(fine[i_best])
    within = fine[chi2 <= chi2[i_best] + 1.0]
    err_low, err_high = best - float(within.min()), float(within.max()) - best

    cmb = CMB_OMEGA_B
    sigma_combined = float(np.hypot(max(err_low, err_high), cmb["error"]))
    tension = (best - cmb["value"]) / sigma_combined
    return ArtifactResult(
        status="success",
        files=[],
        message=(
            f"BBN prefers omega_b h^2 = {best:.5f} +{err_high:.5f} "
            f"-{err_low:.5f} from {'+'.join(observables)}; CMB gives "
            f"{cmb['value']:.5f} +/- {cmb['error']:.5f} "
            f"({tension:+.1f} sigma apart)."
        ),
        metadata={
            "best_fit_omega_b_h2": round(best, 5),
            "err_plus": round(err_high, 5),
            "err_minus": round(err_low, 5),
            "eta10_best": round(eta10(best), 3),
            "chi2_min": round(float(chi2[i_best]), 3),
            "observables_used": list(observables),
            "cmb_measurement": cmb,
            "tension_sigma": round(float(tension), 2),
            "note": (
                "Errors include the nuclear-rate systematic on the "
                "predictions (PRIMAT compilation, PRyMordial default); the "
                "d(p,gamma)He3 rate is the dominant D/H uncertainty."
            ),
        },
    )
