"""MCP tool functions for Big Bang Nucleosynthesis.

Plain Python functions — no MCP imports. The type hints, Field constraints,
and docstrings below become the MCP tool schema that agents see.

Data flows between tools as CSV file paths: scan_bbn_parameter writes one
CSV per scan, plot_bbn_scan and fit_bbn_baryon_density read them back. Only
small numbers and file paths ever pass through the LLM context.

Scope (shared with the sterile-neutrino server): this server takes Delta
N_eff, Omega_b h^2 and the neutron lifetime as INPUTS and predicts the
light-element abundances. How much Delta N_eff a particular particle model
(e.g. a sterile neutrino) PRODUCES is computed by that model's own server.
"""

import csv
import hashlib
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
from pydantic import BaseModel, Field, validate_call

from .bbn import eta10, run_bbn
from .plotting import (CMB_BAND, LAB_BANDS, LINESTYLES, MUTED, OBS_BAND,
                       PALETTE, rc_params, wrap)

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


# Per-parameter scan settings. delta_neff points re-solve the thermal
# history (~10 s each), so they get a tighter cap.
SCAN_PARAMETERS = {
    "omega_b_h2": {"default_range": (0.010, 0.032), "bounds": (0.005, 0.05),
                   "max_points": 30,
                   "label": r"$\Omega_b h^2$"},
    "delta_neff": {"default_range": (-1.0, 2.0), "bounds": (-2.0, 3.0),
                   "max_points": 9,
                   "label": r"$\Delta N_{\rm eff}$"},
    "tau_n_s": {"default_range": (866.0, 896.0), "bounds": (800.0, 1000.0),
                "max_points": 25,
                "label": r"$\tau_n$ [s]"},
}
SCAN_COLUMNS = ["omega_b_h2", "eta10", "delta_neff", "Neff", "tau_n_s",
                "Yp_BBN", "D_H_x1e5", "He3_H_x1e5", "Li7_H_x1e10"]

# observable column -> (axis label, short name)
OBSERVABLE_AXES = {
    "Yp_BBN": (r"$Y_p$", "Yp"),
    "D_H_x1e5": (r"$\mathrm{D/H}\ [10^{-5}]$", "D/H"),
    "He3_H_x1e5": (r"${}^{3}\mathrm{He/H}\ [10^{-5}]$", "He3/H"),
    "Li7_H_x1e10": (r"${}^{7}\mathrm{Li/H}\ [10^{-10}]$", "Li7/H"),
}
DEFAULT_PANELS = {
    "omega_b_h2": ["Yp_BBN", "D_H_x1e5", "He3_H_x1e5", "Li7_H_x1e10"],
    "delta_neff": ["Yp_BBN", "D_H_x1e5"],
    "tau_n_s": ["Yp_BBN", "D_H_x1e5"],
}
FIXED_SYMBOLS = {"omega_b_h2": r"$\Omega_b h^2$", "delta_neff": r"$\Delta N_{\rm eff}$",
                 "tau_n_s": r"$\tau_n$"}

ScanParameter = Literal["omega_b_h2", "delta_neff", "tau_n_s"]
Observable = Literal["Yp_BBN", "D_H_x1e5", "He3_H_x1e5", "Li7_H_x1e10"]


class ArtifactResult(BaseModel):
    """Uniform result contract returned by every tool."""

    status: Literal["success"]
    files: list[str]
    message: str
    metadata: dict[str, Any]


def _slug(**params) -> str:
    """Short stable hash so scans at different inputs never overwrite."""
    blob = ",".join(f"{k}={params[k]}" for k in sorted(params))
    return hashlib.sha1(blob.encode()).hexdigest()[:6]


def _write_csv(path: Path, header: dict[str, str], columns: list[str], rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        for key, value in header.items():
            f.write(f"# {key}: {value}\n")
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(rows)


def _read_scan(path_str: str) -> tuple[dict[str, str], dict[str, np.ndarray]]:
    """Read a CSV written by scan_bbn_parameter: (header dict, columns)."""
    path = Path(path_str).expanduser().resolve()
    lines = path.read_text(encoding="utf-8").splitlines()
    header, i = {}, 0
    while i < len(lines) and lines[i].startswith("#"):
        key, _, value = lines[i].lstrip("# ").partition(":")
        header[key.strip()] = value.strip()
        i += 1
    names = lines[i].split(",")
    missing = [c for c in SCAN_COLUMNS if c not in names]
    if missing or "scanned_parameter" not in header:
        raise ValueError(
            f"{path.name} is not a scan_bbn_parameter CSV (missing columns "
            f"{missing or 'scanned_parameter header'}). Pass a file written "
            "by scan_bbn_parameter.")
    data = np.loadtxt(lines[i + 1:], delimiter=",", ndmin=2).T
    return header, {name: col for name, col in zip(names, data)}


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
    """Explain the physics inputs, observables, and workflow of the BBN tools.

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
            "workflow": [
                "compute_bbn_abundances: one parameter point, with pulls",
                "scan_bbn_parameter: vary omega_b_h2, delta_neff or tau_n_s "
                "-> CSV",
                "plot_bbn_scan: figure from one or several scan CSVs",
                "fit_bbn_baryon_density: BBN omega_b_h2 vs the CMB value "
                "(needs an omega_b_h2 scan)",
            ],
            "scope": "Delta N_eff is an INPUT here. To get the Delta N_eff "
                     "a specific particle model produces (e.g. a sterile "
                     "neutrino of given mass and mixing), use that model's "
                     "server, then feed the value to this one.",
            "runtime_note": "compute_bbn_abundances takes seconds; scans "
                            "take up to ~1 minute (delta_neff scans ~10 s "
                            "per point) — do not retry while one is running.",
        },
    )



@validate_call
def compute_bbn_abundances(
    omega_b_h2: Annotated[float, Field(ge=0.005, le=0.05, description="Physical baryon density Omega_b h^2 (CMB: 0.02237).")] = 0.02237,
    delta_neff: Annotated[float, Field(ge=-2.0, le=3.0, description="Extra relativistic species beyond the SM (0 = none; SM N_eff = 3.044). An INPUT here — get model-specific values from the model's own server.")] = 0.0,
    tau_n_s: Annotated[float, Field(ge=800.0, le=1000.0, description="Neutron lifetime in seconds (PDG average: 878.4).")] = 878.4,
) -> ArtifactResult:
    """Predict the primordial light-element abundances (BBN) at one input point.

    Runs the full 63-reaction PRyMordial network and returns N_eff, the
    helium-4 mass fraction Yp, D/H, He3/H and Li7/H, each compared with the
    measured primordial values (pull in sigma, measurement and nuclear-rate
    errors in quadrature). To explore a range, call scan_bbn_parameter once
    instead of calling this repeatedly.
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
def scan_bbn_parameter(
    output_dir: Annotated[str, Field(min_length=1)],
    parameter: Annotated[ScanParameter, Field(description="Which input to vary: 'omega_b_h2' (baryon density — the Schramm plot; needed by fit_bbn_baryon_density), 'delta_neff' (dark radiation; each point re-solves the thermal history, ~10 s, max 9 points), or 'tau_n_s' (neutron lifetime).")],
    range_min: Annotated[float | None, Field(description="Lower end of the scan. Default: 0.010 (omega_b_h2), -1 (delta_neff), 866 (tau_n_s).")] = None,
    range_max: Annotated[float | None, Field(description="Upper end. Default: 0.032 (omega_b_h2), 2 (delta_neff), 896 (tau_n_s).")] = None,
    n_points: Annotated[int, Field(ge=3, le=30, description="Scan points (caps: 30 omega_b_h2, 9 delta_neff, 25 tau_n_s).")] = 9,
    omega_b_h2: Annotated[float, Field(ge=0.005, le=0.05, description="Baryon density held fixed when another parameter is scanned.")] = 0.02237,
    delta_neff: Annotated[float, Field(ge=-2.0, le=3.0, description="Delta N_eff held fixed when another parameter is scanned.")] = 0.0,
    tau_n_s: Annotated[float, Field(ge=800.0, le=1000.0, description="Neutron lifetime held fixed when another parameter is scanned.")] = 878.4,
) -> ArtifactResult:
    """Scan one BBN input and tabulate all light-element abundances to CSV.

    One tool for the three classic scans: baryon density (the Schramm plot
    and the BBN baryometer), Delta N_eff (BBN as a counter of light species),
    neutron lifetime (the bottle-vs-beam puzzle). The other two inputs are
    held at the given fixed values, recorded in the CSV header; every scan
    gets its own file name, so scans at different fixed values coexist and
    overlay in plot_bbn_scan. Columns: omega_b_h2, eta10, delta_neff, Neff,
    tau_n_s, Yp_BBN, D_H_x1e5, He3_H_x1e5, Li7_H_x1e10. Uses the 12-reaction
    network (abundances within ~0.2% of the full one). Takes ~10-90 s.
    """
    spec = SCAN_PARAMETERS[parameter]
    lo = spec["default_range"][0] if range_min is None else range_min
    hi = spec["default_range"][1] if range_max is None else range_max
    b_lo, b_hi = spec["bounds"]
    if not (b_lo <= lo < hi <= b_hi):
        raise ValueError(
            f"{parameter} range must satisfy {b_lo:g} <= range_min < range_max "
            f"<= {b_hi:g} (got {lo:g}..{hi:g}); range_min must be smaller "
            "than range_max.")
    if n_points > spec["max_points"]:
        raise ValueError(f"{parameter} scans allow at most {spec['max_points']} "
                         "points (each point is a full BBN run).")

    fixed = {"omega_b_h2": omega_b_h2, "delta_neff": delta_neff, "tau_n_s": tau_n_s}
    rows = []
    for value in np.linspace(lo, hi, n_points):
        point = dict(fixed, **{parameter: float(value)})
        r = run_bbn(point["omega_b_h2"], point["delta_neff"], point["tau_n_s"],
                    small_network=True)
        rows.append((point["omega_b_h2"], eta10(point["omega_b_h2"]),
                     point["delta_neff"], r["Neff"], point["tau_n_s"],
                     r["Yp_BBN"], r["D_H_x1e5"], r["He3_H_x1e5"],
                     r["Li7_H_x1e10"]))

    held = {k: v for k, v in fixed.items() if k != parameter}
    slug = _slug(parameter=parameter, lo=lo, hi=hi, n=n_points, **held)
    csv_path = _outdir(output_dir) / f"bbn_scan_{parameter}_{slug}.csv"
    _write_csv(csv_path, {
        "source": "PRyMordial (12-reaction network)",
        "scanned_parameter": parameter,
        "fixed": ", ".join(f"{k}={v:g}" for k, v in held.items()),
    }, SCAN_COLUMNS, rows)
    table = np.array(rows)
    col = {name: table[:, i] for i, name in enumerate(SCAN_COLUMNS)}
    return ArtifactResult(
        status="success",
        files=[str(csv_path)],
        message=(f"Scanned {parameter} over [{lo:g}, {hi:g}] ({n_points} points) "
                 f"with {', '.join(f'{k}={v:g}' for k, v in held.items())}."),
        metadata={
            "scanned_parameter": parameter, "range": [lo, hi],
            "fixed_inputs": held, "columns": SCAN_COLUMNS,
            "endpoints": {k: [round(float(col[k][0]), 4), round(float(col[k][-1]), 4)]
                          for k in ("Yp_BBN", "D_H_x1e5", "Li7_H_x1e10", "Neff")},
            "next": ("plot_bbn_scan for the figure"
                     + ("; fit_bbn_baryon_density for the BBN baryon density"
                        if parameter == "omega_b_h2" else "")),
        },
    )


def _scan_label(header: dict, varying: set[str]) -> str:
    """Legend entry: the fixed inputs that differ between overlaid scans."""
    fixed = dict(item.split("=") for item in header.get("fixed", "").split(", ") if "=" in item)
    parts = [f"{FIXED_SYMBOLS[k]} = {float(fixed[k]):g}" for k in sorted(varying) if k in fixed]
    return ", ".join(parts) or "BBN prediction"


@validate_call
def plot_bbn_scan(
    scan_files: Annotated[list[str], Field(min_length=1, max_length=6, description="CSV(s) from scan_bbn_parameter, all scanning the SAME parameter (e.g. omega_b_h2 scans at delta_neff = 0 and 1 overlay as separate curves).")],
    output_dir: Annotated[str, Field(min_length=1)],
    observables: Annotated[list[Observable] | None, Field(description="Panels to draw, top to bottom. Default: Yp, D/H, He3/H, Li7/H for omega_b_h2 scans; Yp and D/H otherwise.")] = None,
    labels: Annotated[list[str] | None, Field(description="Short legend entry per scan file. Default: the fixed inputs that differ between files.")] = None,
    title: Annotated[str | None, Field(description="Optional title (wrapped). Leave unset for a paper-ready figure.")] = None,
    save_pdf: Annotated[bool, Field(description="Also write a vector PDF next to the PNG (for manuscripts).")] = False,
    output_name: Annotated[str | None, Field(description="Optional output file stem.")] = None,
) -> ArtifactResult:
    """Plot BBN abundance predictions from scan CSVs against the measured values.

    Decorations follow the scanned parameter:
    - omega_b_h2: the Schramm plot — stacked panels vs Omega_b h^2 with the
      measured primordial bands and the Planck CMB baryon density; the top
      axis shows eta_10. The Li7/H panel shows the lithium problem.
    - delta_neff: Yp and D/H vs Delta N_eff with measured bands and the SM
      point marked; the top axis shows N_eff.
    - tau_n_s: Yp and D/H vs the neutron lifetime with the bottle and beam
      laboratory measurements as vertical bands.
    Measured bands include the nuclear-rate theory error in quadrature (as
    in the pulls). Journal-style mathtext, no LaTeX install needed.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    scans = [_read_scan(f) for f in scan_files]
    params = {h["scanned_parameter"] for h, _ in scans}
    if len(params) > 1:
        raise ValueError(f"All scan_files must scan the same parameter (got "
                         f"{sorted(params)}); plot them in separate calls.")
    parameter = params.pop()
    panels = list(observables or DEFAULT_PANELS[parameter])
    if labels is not None and len(labels) != len(scans):
        raise ValueError(f"labels has {len(labels)} entries for {len(scans)} files.")
    fixed_sets = [h.get("fixed", "") for h, _ in scans]
    varying = set()
    if len(scans) > 1:
        parsed = [dict(i.split("=") for i in f.split(", ") if "=" in i) for f in fixed_sets]
        varying = {k for k in parsed[0] if len({p.get(k) for p in parsed}) > 1}
    legend_labels = labels or [_scan_label(h, varying) for h, _ in scans]
    if len(scans) == 1 and labels is None:
        legend_labels = ["BBN prediction (PRyMordial)"]

    with plt.rc_context(rc_params()):
        height = 1.9 + 2.0 * len(panels)
        fig, axes = plt.subplots(len(panels), 1, figsize=(6.4, height),
                                 sharex=True, layout="constrained",
                                 squeeze=False)
        axes = axes[:, 0]
        for ax, obs_key in zip(axes, panels):
            for i, (_, d) in enumerate(scans):
                ax.plot(d[parameter], d[obs_key], color=PALETTE[i],
                        linestyle=LINESTYLES[i], label=legend_labels[i],
                        marker="o" if parameter == "delta_neff" else None,
                        markersize=4, zorder=3)
            if obs_key in OBSERVATIONS:
                obs = OBSERVATIONS[obs_key]
                err = _total_error(obs)
                ax.axhspan(obs["value"] - err, obs["value"] + err,
                           color=OBS_BAND, alpha=0.35, linewidth=0, zorder=1,
                           label="measured primordial value")
            if parameter == "omega_b_h2":
                ax.axvspan(CMB_OMEGA_B["value"] - CMB_OMEGA_B["error"],
                           CMB_OMEGA_B["value"] + CMB_OMEGA_B["error"],
                           color=CMB_BAND, alpha=0.3, linewidth=0, zorder=2,
                           label=r"Planck CMB $\Omega_b h^2$")
            elif parameter == "delta_neff":
                ax.axvline(0.0, color=MUTED, linewidth=0.9, linestyle=":",
                           zorder=2, label=r"Standard Model ($N_{\rm eff}=3.044$)")
            else:
                for j, key in enumerate(("bottle_ucn_tau", "beam_bl1")):
                    m = NEUTRON_LIFETIME_S[key]
                    ax.axvspan(m["value"] - m["error"], m["value"] + m["error"],
                               color=LAB_BANDS[j], alpha=0.3, linewidth=0, zorder=2,
                               label=rf"{key.split('_')[0]}: $\tau_n = {m['value']:g} \pm {m['error']:g}$ s")
            values = np.concatenate([d[obs_key] for _, d in scans])
            if values.min() > 0 and values.max() / values.min() > 3:
                ax.set_yscale("log")   # Schramm-plot convention for wide spans
                ax.yaxis.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0)))
                ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
                ax.yaxis.set_minor_formatter(NullFormatter())
            ax.set_ylabel(OBSERVABLE_AXES[obs_key][0])
        axes[-1].set_xlabel(SCAN_PARAMETERS[parameter]["label"])

        d0 = scans[0][1]
        if parameter == "omega_b_h2":
            k = eta10(1.0)
            top = axes[0].secondary_xaxis("top", functions=(lambda ob: k * ob,
                                                            lambda e: e / k))
            top.set_xlabel(r"$\eta_{10}$")
        elif parameter == "delta_neff":
            offset = float(np.mean(d0["Neff"] - d0["delta_neff"]))
            top = axes[0].secondary_xaxis("top", functions=(lambda x: x + offset,
                                                            lambda n: n - offset))
            top.set_xlabel(r"$N_{\rm eff}$")
        if parameter in ("omega_b_h2", "delta_neff"):
            top.tick_params(which="both", direction="in")
        if title:
            fig.suptitle(wrap(title, 60), x=0.02, ha="left")

        handles, names = [], []
        for ax in axes:
            for h, n in zip(*ax.get_legend_handles_labels()):
                if n not in names:
                    handles.append(h)
                    names.append(n)
        fig.legend(handles, [wrap(n, 44) for n in names],
                   loc="outside lower center", ncol=2, borderaxespad=0.3)

        stem = output_name or f"bbn_{parameter}_{_slug(f=tuple(scan_files), o=tuple(panels))}"
        out = _outdir(output_dir)
        plot_path = out / f"{stem}.png"
        fig.savefig(plot_path, bbox_inches="tight", pad_inches=0.08, facecolor="white")
        files = [str(plot_path)]
        if save_pdf:
            fig.savefig(plot_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.08)
            files.append(str(plot_path.with_suffix(".pdf")))
        plt.close(fig)

    meta = {"scanned_parameter": parameter, "panels": panels,
            "legend_labels": legend_labels,
            "observed_bands": {k: OBSERVATIONS[k] for k in panels if k in OBSERVATIONS}}
    if parameter == "omega_b_h2":
        meta["cmb_band"] = CMB_OMEGA_B
    if parameter == "tau_n_s":
        meta["laboratory_measurements"] = NEUTRON_LIFETIME_S
    return ArtifactResult(
        status="success", files=files,
        message=f"Plotted {', '.join(OBSERVABLE_AXES[p][1] for p in panels)} vs "
                f"{parameter} for {len(scans)} scan(s) with measured bands.",
        metadata=meta)


@validate_call
def fit_bbn_baryon_density(
    scan_file: Annotated[str, Field(min_length=1, description="CSV from scan_bbn_parameter with parameter='omega_b_h2'; the range must bracket the minimum (e.g. 0.010-0.032).")],
    observables: Annotated[list[Literal["D_H_x1e5", "Yp_BBN"]],
                           Field(min_length=1, description="Measured abundances in the chi^2: deuterium is the precision baryometer; helium is a weak cross-check.")] = ["D_H_x1e5", "Yp_BBN"],
) -> ArtifactResult:
    """Fit the baryon density preferred by BBN abundances and compare with the CMB.

    Builds chi^2(Omega_b h^2) from an omega_b_h2 scan against the chosen
    measured abundances (measurement and nuclear-rate errors in quadrature),
    reports the best fit with its 1-sigma interval, and its tension with the
    Planck value — the BBN/CMB concordance test in one number. The fit holds
    delta_neff and tau_n_s at the values the scan was run with (in metadata).
    """
    header, d = _read_scan(scan_file)
    if header["scanned_parameter"] != "omega_b_h2":
        raise ValueError(
            f"{Path(scan_file).name} scans {header['scanned_parameter']}; the "
            "fit needs scan_bbn_parameter(parameter='omega_b_h2').")
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
            "re-run scan_bbn_parameter(parameter='omega_b_h2') over a wider range.")
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
            "held_fixed": header.get("fixed", ""),
            "cmb_measurement": cmb,
            "tension_sigma": round(float(tension), 2),
            "note": (
                "Errors include the nuclear-rate systematic on the "
                "predictions (PRIMAT compilation, PRyMordial default); the "
                "d(p,gamma)He3 rate is the dominant D/H uncertainty."
            ),
        },
    )
