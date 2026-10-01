# bbn-mcp-server

An MCP server that exposes science tools — Big Bang Nucleosynthesis, the
particle physics of the universe's first three minutes — to any LLM agent.
Built on the same pattern as
[`spectra-mcp-server`](https://github.com/HEP-KE/spectra-mcp-server) and
[`gaia-mcp-server`](https://github.com/HEP-KE/gaia-mcp-server); the
multi-agent client in
[`multiagent-client-demo`](https://github.com/HEP-KE/multiagent-client-demo)
drives any of these servers unchanged.

## The science

Between one second and a few minutes after the Big Bang, the expanding
universe ran a nuclear reactor. Its output — the primordial abundances of
helium-4, deuterium, helium-3 and lithium-7 — is still measurable today, and
it depends on exactly three things an agent can turn:

- the **baryon density** Ω_b h² (how much ordinary matter exists),
- **extra relativistic species** ΔN_eff (any light particle beyond the three
  Standard Model neutrinos — dark radiation, sterile states, axions —
  speeds up the expansion and changes the yields),
- the **neutron lifetime** τ_n (which sets the neutron/proton ratio at weak
  freeze-out — and whose laboratory measurements currently disagree by ~10 s,
  the "bottle vs beam" puzzle).

![Primordial abundances vs baryon density](output/bbn_abundance_curves.png)

The classic stories are all here: the abundances and the CMB agree on the
baryon density of the universe (**concordance**), helium counts the light
degrees of freedom at t ~ 1 s, and the lithium-7 panel shows a famous
**unsolved problem** — the prediction is ~3× the observed Spite plateau.

All computations run on
[**PRyMordial**](https://github.com/vallima/PRyMordial) (Burns, Tait &
Valli, [EPJC 84 (2024) 86](https://arxiv.org/abs/2307.07061)), the
community Python code for precision BBN within and beyond the Standard
Model. PRyMordial is GPL-3.0 and not on PyPI, so it is **not** vendored
here — a setup script fetches a pinned copy into `extern/` (gitignored).

## Tools

| tool | what it does |
|---|---|
| `describe_bbn_inputs()` | parameters, measured abundances, references, the classic questions — call it first |
| `compute_bbn_abundances(omega_b_h2, delta_neff, tau_n_s)` | one full 63-reaction run: N_eff, Yp, D/H, ³He/H, ⁷Li/H, each with its pull (σ) against observation |
| `scan_bbn_parameter(output_dir, parameter, ...)` | vary one input — `omega_b_h2` (the Schramm-plot calculation), `delta_neff` (each point re-solves the thermal history), or `tau_n_s` — with the other two held fixed → CSV (unique name per scan, so scans at different fixed values coexist) |
| `plot_bbn_scan(scan_files, output_dir, ...)` | publication figure from one or several scans of the same parameter: Yp, D/H, ³He/H, ⁷Li/H panels with measured bands, plus the Planck Ω_b h² band, the SM N_eff line, or the bottle/beam τ_n bands as appropriate; optional PDF |
| `fit_bbn_baryon_density(scan_file, observables)` | χ² fit on an Ω_b h² scan: the BBN-preferred Ω_b h² ± 1σ, compared with Planck — the concordance test in one number |

ΔN_eff is an **input** here. The ΔN_eff a specific particle model produces
(e.g. a sterile neutrino of given mass and mixing) comes from that model's
own server; feed the value into these tools. Figures use matplotlib's
built-in mathtext (STIX fonts) — no LaTeX installation is needed.

Predictions carry the nuclear-rate systematic (PRIMAT compilation,
PRyMordial's default) in quadrature with the measurement errors — the
d(p,γ)³He rate alone moves D/H by a few percent, which is a genuine, live
discussion in the field, not a bug.

Two conventions worth copying into any science MCP server:

1. Every tool returns `{status, files, message, metadata}` (`ArtifactResult`).
2. Arrays move between tools **as file paths**, never through the agent's
   context window.

## Install

```bash
conda create -n bbn-tutorial python=3.12 -y
conda activate bbn-tutorial
pip install -e ".[dev]"
python scripts/setup_prymordial.py   # fetches PRyMordial + precomputes tables (~1 min)
pytest
```

The setup script clones a pinned PRyMordial commit into `extern/PRyMordial`
(override with the `PRYM_DIR` env var) and runs one Standard Model
computation with the save flags on, so the thermal background and weak rates
are cached — that makes parameter scans ~6× faster and lets the server run
without write access to the checkout.

Already have the `spectra-tutorial` env from the other servers? You can
reuse it: `pip install numba numdifftools`, run the setup script, then
always launch this server (and pytest) **from this repo's root directory** —
all tutorial servers export packages named `tools` and `mcp_server`, so
don't `pip install -e` several of them into one env; running from the repo
root makes the local packages win.

## Run the server

```bash
python -m mcp_server --transport streamable-http --port 8003
```

Clients connect to `http://127.0.0.1:8003/mcp`. Port **8003** by convention,
so this server can run alongside spectra (8000), gaia (8001) and lattice
(8002). Stop with **Ctrl+C**.

To use this server from Claude Code, the Claude desktop app, Codex, Cursor,
or any other MCP client, see [`docs/mcp-clients.md`](docs/mcp-clients.md) —
a checked-in `.mcp.json` already wires it into Claude Code.

## Questions to ask an agent

> "Does the baryon density measured by Planck in the CMB correctly predict
> the deuterium abundance we observe in quasar spectra? Scan the baryon
> density, fit it from the abundances, and give me the verdict in sigma."

> "If there were one extra neutrino species in the early universe, what
> would have happened to primordial helium, and is that allowed by the
> data?"

The first one becomes a genuine **two-server** question when this server is
hosted next to a CMB-capable one (compute the damping tail there, the
deuterium here).
