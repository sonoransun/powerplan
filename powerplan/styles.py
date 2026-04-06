"""
Unified visual design language for PowerPlan visualizations.

Colorblind-safe palette (Wong 2011 adapted), consistent typography,
and reusable drawing helpers for system diagrams.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter


# ──────────────────────────────────────────────────────────────────────
# Color Palette — colorblind-safe (distinguishable in deuteranopia)
# ──────────────────────────────────────────────────────────────────────

PALETTE = {
    # Energy flow roles
    "demand":      "#CC6677",
    "generation":  "#DDCC77",
    "discharge":   "#88CCEE",
    "charge":      "#44AA99",
    "grid_import": "#AA4499",
    "grid_export": "#332288",
    "curtail":     "#999933",
    "loss":        "#BBBBBB",

    # Source types
    "solar":       "#DDCC77",
    "wind":        "#88CCEE",
    "hydro":       "#44AA99",
    "geothermal":  "#CC6677",
    "gas":         "#882255",
    "fusion":      "#AA4499",
    "antimatter":  "#6699CC",

    # Storage types
    "lithium":     "#88CCEE",
    "sodium":      "#44AA99",
    "flow":        "#CC6677",
    "flywheel":    "#DDCC77",
    "hydrogen":    "#999933",
    "supercap":    "#332288",
    "smes":        "#AA4499",

    # UI chrome
    "bg_panel":    "#FAFAFA",
    "bg_night":    "#E8E8F0",
    "grid_line":   "#E0E0E0",
    "text_dark":   "#2D3436",
    "text_mid":    "#636E72",
    "text_light":  "#B2BEC3",
    "accent":      "#0984E3",
}

# Ordered category cycle for bar charts
_CATEGORY_CYCLE = [
    "#332288", "#88CCEE", "#44AA99", "#999933",
    "#DDCC77", "#CC6677", "#882255", "#AA4499",
    "#6699CC", "#117733", "#661100", "#BBBBBB",
]


# ──────────────────────────────────────────────────────────────────────
# Typography
# ──────────────────────────────────────────────────────────────────────

FONTS = {
    "suptitle":   {"fontsize": 15, "fontweight": "bold", "color": PALETTE["text_dark"]},
    "title":      {"fontsize": 11, "fontweight": "semibold", "color": PALETTE["text_dark"]},
    "subtitle":   {"fontsize": 10, "color": PALETTE["text_mid"]},
    "label":      {"fontsize": 9, "color": PALETTE["text_dark"]},
    "tick_size":  8,
    "annotation": {"fontsize": 7.5, "color": PALETTE["text_mid"]},
    "legend":     {"fontsize": 8, "framealpha": 0.92, "edgecolor": "#CCC", "fancybox": True},
}


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def apply_style(ax):
    """Apply consistent professional styling to an axes."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CCCCCC")
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_color("#CCCCCC")
    ax.spines["bottom"].set_linewidth(0.5)
    ax.grid(True, color=PALETTE["grid_line"], linewidth=0.3, alpha=0.7,
            linestyle="--")
    ax.tick_params(labelsize=FONTS["tick_size"], colors=PALETTE["text_mid"])
    ax.set_facecolor(PALETTE["bg_panel"])


def source_color(name: str) -> str:
    """Map a source name to its palette color."""
    n = name.lower()
    for key in ["solar", "wind", "hydro", "geothermal", "gas",
                "fusion", "antimatter"]:
        if key in n:
            return PALETTE[key]
    return PALETTE["generation"]


def storage_color(name: str) -> str:
    """Map a storage name to its palette color."""
    n = name.lower()
    for key in ["lithium", "li-ion", "sodium", "na ", "flow", "vanadium",
                "flywheel", "hydrogen", "h2", "supercap", "graphene",
                "smes"]:
        if key in n:
            mapped = {"li-ion": "lithium", "na ": "sodium",
                      "vanadium": "flow", "h2": "hydrogen",
                      "graphene": "supercap"}.get(key, key)
            return PALETTE.get(mapped, PALETTE["loss"])
    return PALETTE["loss"]


def categorical_colors(n: int) -> list[str]:
    """Return n distinct categorical colors."""
    return [_CATEGORY_CYCLE[i % len(_CATEGORY_CYCLE)] for i in range(n)]


def format_kw(x, _=None) -> str:
    """Format power values for axis labels."""
    if abs(x) >= 1_000_000:
        return f"{x/1e6:.1f} GW"
    if abs(x) >= 1000:
        return f"{x/1000:.1f} MW"
    return f"{x:.0f} kW"


def format_kwh(x, _=None) -> str:
    """Format energy values for axis labels."""
    if abs(x) >= 1_000_000:
        return f"{x/1e6:.1f} GWh"
    if abs(x) >= 1000:
        return f"{x/1000:.1f} MWh"
    return f"{x:.0f} kWh"


def kw_formatter():
    return FuncFormatter(lambda x, _: format_kw(x))


def kwh_formatter():
    return FuncFormatter(lambda x, _: format_kwh(x))


def add_nighttime_shading(ax, days_axis):
    """Add subtle day/night shading to a time-series axes."""
    for d in range(int(np.max(days_axis)) + 1):
        # Night: 18:00 to next-day 06:00 (0.75 to 1.25 in day units)
        ax.axvspan(d + 0.75, d + 1.25, color=PALETTE["bg_night"], alpha=0.04,
                   zorder=0, linewidth=0)


def styled_legend(ax, **kwargs):
    """Add a consistently styled legend."""
    defaults = dict(FONTS["legend"])
    defaults.update(kwargs)
    return ax.legend(**defaults)


def draw_rounded_box(ax, x, y, w, h, color, label="", fontsize=8,
                     alpha=0.85):
    """Draw a labeled rounded-rectangle box on an axes."""
    box = mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.01",
        facecolor=color, edgecolor="white", alpha=alpha, linewidth=1.5,
        transform=ax.transAxes, zorder=5,
    )
    ax.add_patch(box)
    if label:
        ax.text(x + w / 2, y + h / 2, label,
                transform=ax.transAxes, ha="center", va="center",
                fontsize=fontsize, color="white", fontweight="bold",
                zorder=6)


def draw_flow_arrow(ax, start, end, width=0.01, color="#888888",
                    alpha=0.6):
    """Draw a curved flow arrow between two points (axes coordinates)."""
    arrow = mpatches.FancyArrowPatch(
        start, end, connectionstyle="arc3,rad=0.15",
        arrowstyle=f"->,head_length=8,head_width=5",
        color=color, linewidth=max(1, width * 40), alpha=alpha,
        transform=ax.transAxes, zorder=4,
    )
    ax.add_patch(arrow)
