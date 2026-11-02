"""Shared figure styling for the workshop-paper plots: one canonical per-model
colour map and a spine/​border cleaner, so every figure is visually consistent.

Colours are the first six slots of the data-viz reference palette (blue, green,
magenta, yellow, aqua, orange), a colourblind-safe order validated on the
adjacent pairlist. Assigned to models in a FIXED order (never cycled) so a model
keeps its colour across every figure.
"""
from __future__ import annotations

# model slug -> hex. Fixed assignment; do not reorder (colour follows the entity).
MODEL_COLORS = {
    "llama3.1-8b":  "#2a78d6",  # blue
    "llama3.1-70b": "#008300",  # green
    "qwen3-8b":     "#e87ba4",  # magenta
    "qwen3-32b":    "#eda100",  # yellow
    "gemma-2-9b":   "#1baf7a",  # aqua
    "gemma-2-27b":  "#eb6834",  # orange
}

# display name -> hex (same mapping, keyed by the label used in legends)
MODEL_COLORS_BY_NAME = {
    "Llama 3.1-8B":  "#2a78d6",
    "Llama 3.1-70B": "#008300",
    "Qwen 3-8B":     "#e87ba4",
    "Qwen 3-32B":    "#eda100",
    "Gemma 2-9B":    "#1baf7a",
    "Gemma 2-27B":   "#eb6834",
}

# canonical display order (family-grouped, small before large)
MODEL_ORDER = ["llama3.1-8b", "llama3.1-70b", "qwen3-8b", "qwen3-32b",
               "gemma-2-9b", "gemma-2-27b"]

# recessive chrome
GRID = "#e1e0d9"
AXIS = "#c3c2b7"


def clean_axes(ax, keep=("left", "bottom"), grid_axis="y"):
    """Remove hard borders: drop the top/right (and any not in `keep`) spines and
    soften the remaining ones to a hairline; light y-grid only. Call per-axis."""
    for side, sp in ax.spines.items():
        if side in keep:
            sp.set_color(AXIS)
            sp.set_linewidth(0.8)
        else:
            sp.set_visible(False)
    ax.tick_params(colors="#52514e", length=3, width=0.8)
    if grid_axis:
        ax.grid(True, axis=grid_axis, color=GRID, lw=0.6, alpha=0.9, zorder=0)
        ax.set_axisbelow(True)


# ---- print-size style for the camera-ready (2026-10-27) -------------------------
# Follows the figure-style skill (figure_style_kernel.apply_figure_style): three font
# sizes mapped to role (8 titles/axis labels/series names, 7 legends/annotations,
# 6 ticks), regular-weight left-aligned titles, outward ticks, frameless legends,
# 300 dpi. Every figure is drawn at its printed width, so these are the sizes on the page.
TEXTWIDTH = 5.5   # inches, NeurIPS text width
FS = {"title": 8, "label": 8, "legend": 7, "annot": 7, "tick": 6}
LW = {"data": 1.2, "trend": 1.8, "ref": 0.8, "err": 0.7, "axis": 0.6}
# condition colours follow Fig. 1 (parametric / decision-time / post-decision evidence)
COND_COLORS = {"A1": "#7A6E96", "A2": "#2E6E8E", "A3": "#B5544B"}
NEUTRAL = {"dark": "#44525C", "light": "#9AA5AD"}


def paper_style():
    """Apply the figure-style skill's rcParams with the paper's size ladder."""
    from src.evaluation.figure_style_kernel import apply_figure_style
    apply_figure_style(frame="open", sizes=(FS["title"], FS["legend"], FS["tick"]), grid=False)
