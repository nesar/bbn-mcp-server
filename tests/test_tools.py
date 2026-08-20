import asyncio

import numpy as np
import pytest

import tools
from mcp_server.server import create_server
from tools.bbn import cached_tables_exist, prym_dir

needs_prym = pytest.mark.skipif(
    not (prym_dir() / "PRyM" / "PRyM_main.py").exists(),
    reason="PRyMordial not fetched — run scripts/setup_prymordial.py",
)


def test_server_exposes_all_tools():
    mcp = create_server()
    assert mcp.name == "BBN MCP Server"
    listed = {t.name for t in asyncio.run(mcp.list_tools())}
    assert listed == set(tools.__all__)
    assert len(listed) == 9


def test_describe_lists_the_physics():
    meta = tools.describe_bbn_inputs().metadata
    assert set(meta["parameters"]) == {"omega_b_h2", "delta_neff", "tau_n_s"}
    assert "D_H_x1e5" in meta["observed_abundances"]


@needs_prym
def test_sm_abundances_match_textbook_values():
    res = tools.compute_abundances()
    pred = res.metadata["predictions"]
    assert pred["Neff"] == pytest.approx(3.044, abs=0.01)
    assert pred["Yp_BBN"] == pytest.approx(0.247, abs=0.003)
    assert pred["D_H_x1e5"] == pytest.approx(2.5, abs=0.15)
    pulls = res.metadata["comparison_with_observations"]
    assert abs(pulls["Yp_BBN"]["pull_sigma"]) < 2.0


@needs_prym
def test_baryon_scan_fit_and_plot(tmp_path):
    assert cached_tables_exist(), "setup_prymordial.py should have precomputed tables"
    out = str(tmp_path)
    scan = tools.scan_baryon_density(output_dir=out, omega_b_h2_min=0.016,
                                     omega_b_h2_max=0.028, n_points=7)
    data = np.loadtxt(scan.files[0], delimiter=",", skiprows=2)
    assert np.all(np.diff(data[:, 3]) < 0)  # D/H falls with omega_b
    assert np.all(np.diff(data[:, 2]) > 0)  # Yp rises with omega_b

    plot = tools.plot_abundance_curves(scan_file=scan.files[0], output_dir=out)
    assert (tmp_path / "bbn_abundance_curves.png").exists()
    assert plot.status == "success"

    fit = tools.fit_baryon_density(scan_file=scan.files[0])
    # concordance: BBN and CMB agree on omega_b within ~2 sigma
    assert fit.metadata["best_fit_omega_b_h2"] == pytest.approx(0.0222, abs=0.001)
    assert abs(fit.metadata["tension_sigma"]) < 2.5


@needs_prym
def test_neutron_lifetime_scan_and_plot(tmp_path):
    out = str(tmp_path)
    scan = tools.scan_neutron_lifetime(output_dir=out, n_points=5)
    data = np.loadtxt(scan.files[0], delimiter=",", skiprows=2)
    assert np.all(np.diff(data[:, 1]) > 0)  # longer-lived neutrons -> more He4
    plot = tools.plot_neutron_lifetime_impact(scan_file=scan.files[0],
                                              output_dir=out)
    assert plot.status == "success"


def test_scan_rejects_inverted_range(tmp_path):
    with pytest.raises(ValueError, match="smaller than"):
        tools.scan_baryon_density(output_dir=str(tmp_path),
                                  omega_b_h2_min=0.03, omega_b_h2_max=0.01)


def test_plot_rejects_wrong_csv(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("# source: junk\na,b\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing columns"):
        tools.plot_abundance_curves(scan_file=str(bad), output_dir=str(tmp_path))
