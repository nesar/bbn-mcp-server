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


def _col(path, name):
    _, cols = tools.bbn_tools._read_scan(path)
    return cols[name]


def test_server_exposes_all_tools():
    mcp = create_server()
    assert mcp.name == "BBN MCP Server"
    listed = {t.name for t in asyncio.run(mcp.list_tools())}
    assert listed == set(tools.__all__) == {
        "describe_bbn_inputs", "compute_bbn_abundances", "scan_bbn_parameter",
        "plot_bbn_scan", "fit_bbn_baryon_density"}
    # every tool name carries the bbn token (no generic names across servers)
    assert all("bbn" in name for name in listed)


def test_describe_lists_the_physics():
    meta = tools.describe_bbn_inputs().metadata
    assert set(meta["parameters"]) == {"omega_b_h2", "delta_neff", "tau_n_s"}
    assert "D_H_x1e5" in meta["observed_abundances"]
    assert "INPUT" in meta["scope"]


@needs_prym
def test_sm_abundances_match_textbook_values():
    res = tools.compute_bbn_abundances()
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
    scan = tools.scan_bbn_parameter(output_dir=out, parameter="omega_b_h2",
                                    range_min=0.016, range_max=0.028, n_points=7)
    f = scan.files[0]
    assert np.all(np.diff(_col(f, "D_H_x1e5")) < 0)  # D/H falls with omega_b
    assert np.all(np.diff(_col(f, "Yp_BBN")) > 0)    # Yp rises with omega_b

    plot = tools.plot_bbn_scan(scan_files=[f], output_dir=out, save_pdf=True)
    assert [p.rsplit(".", 1)[1] for p in plot.files] == ["png", "pdf"]
    assert plot.metadata["panels"][-1] == "Li7_H_x1e10"

    fit = tools.fit_bbn_baryon_density(scan_file=f)
    # concordance: BBN and CMB agree on omega_b within ~2 sigma
    assert fit.metadata["best_fit_omega_b_h2"] == pytest.approx(0.0222, abs=0.001)
    assert abs(fit.metadata["tension_sigma"]) < 2.5


@needs_prym
def test_scans_at_different_fixed_inputs_do_not_overwrite(tmp_path):
    out = str(tmp_path)
    a = tools.scan_bbn_parameter(output_dir=out, parameter="omega_b_h2", n_points=4)
    b = tools.scan_bbn_parameter(output_dir=out, parameter="omega_b_h2", n_points=4,
                                 tau_n_s=887.7)
    assert a.files[0] != b.files[0]
    plot = tools.plot_bbn_scan(scan_files=a.files + b.files, output_dir=out)
    assert plot.metadata["legend_labels"][1].endswith("887.7")


@needs_prym
def test_neutron_lifetime_scan_and_plot(tmp_path):
    out = str(tmp_path)
    scan = tools.scan_bbn_parameter(output_dir=out, parameter="tau_n_s", n_points=5)
    assert np.all(np.diff(_col(scan.files[0], "Yp_BBN")) > 0)  # more He4
    plot = tools.plot_bbn_scan(scan_files=scan.files, output_dir=out)
    assert "laboratory_measurements" in plot.metadata
    with pytest.raises(ValueError, match="needs scan_bbn_parameter"):
        tools.fit_bbn_baryon_density(scan_file=scan.files[0])


def test_scan_rejects_bad_ranges(tmp_path):
    with pytest.raises(ValueError, match="smaller than"):
        tools.scan_bbn_parameter(output_dir=str(tmp_path), parameter="omega_b_h2",
                                 range_min=0.03, range_max=0.01)
    with pytest.raises(ValueError, match="at most 9"):
        tools.scan_bbn_parameter(output_dir=str(tmp_path), parameter="delta_neff",
                                 n_points=20)


def test_plot_rejects_wrong_csv(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("# source: junk\na,b\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not a scan_bbn_parameter CSV"):
        tools.plot_bbn_scan(scan_files=[str(bad)], output_dir=str(tmp_path))
