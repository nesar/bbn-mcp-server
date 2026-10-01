"""Publication-style matplotlib settings shared by every plotting tool.

- NO system LaTeX: hosted VMs have no TeX. Math is rendered by matplotlib's
  built-in mathtext with the STIX fonts (Times-like, journal look), which
  ship with matplotlib itself.
- Nothing may overflow the canvas: constrained layout, short axis labels,
  wrapped titles, legends collected below the axes when they grow.
- Colors: a fixed-order, colorblind-validated palette, always paired with
  distinct linestyles (identity is never color-alone). Observational bands
  use neutral/translucent fills so theory curves stay the visual focus.
"""

import re

# Okabe-Ito-derived, reordered so adjacent slots stay separable under
# deutan/protan CVD (validated: adjacent-pair CVD dE >= 11, normal >= 18).
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#882255",
           "#B8860B", "#CC79A7", "#6B4C9A", "#E07B39"]
LINESTYLES = ["-", "--", "-.", ":", (0, (6, 1.5, 1, 1.5, 1, 1.5)),
              (0, (1, 1)), (0, (8, 2)), (0, (3, 1, 1, 1))]
INK = "#222222"
MUTED = "#6b6b6b"
OBS_BAND = "#9a9a9a"      # measured abundances
CMB_BAND = "#D55E00"      # Planck baryon density
LAB_BANDS = ["#0072B2", "#B8860B", "#882255"]  # laboratory measurements


def rc_params(base_size: float = 12.0) -> dict:
    return {
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": base_size,
        "axes.labelsize": base_size + 1,
        "axes.titlesize": base_size + 1,
        "legend.fontsize": base_size - 1.5,
        "xtick.labelsize": base_size - 0.5,
        "ytick.labelsize": base_size - 0.5,
        "axes.linewidth": 0.9,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "xtick.minor.size": 2.5,
        "ytick.minor.size": 2.5,
        "lines.linewidth": 1.8,
        "legend.frameon": False,
        "legend.handlelength": 2.6,
        "savefig.dpi": 200,
        "figure.dpi": 100,
        "axes.unicode_minus": True,
    }


def wrap(text: str, width: int) -> str:
    """Wrap text without breaking inside $...$ math spans."""
    if len(text) <= width:
        return text
    tokens = re.findall(r"\$[^$]*\$|\S+", text)
    lines, current = [], ""
    for tok in tokens:
        candidate = f"{current} {tok}".strip()
        visible = len(re.sub(r"\\[a-zA-Z]+|[{}$^_\\]", "", candidate))
        if visible > width and current:
            lines.append(current)
            current = tok
        else:
            current = candidate
    lines.append(current)
    return "\n".join(lines)
