"""Color palettes and styling utilities for plotly figures."""

from typing import Any

import plotly.graph_objects as go

# fmt:off
# Color palettes
colors_light = [
    '#ff4d4d', '#ff7f50', '#ffff00', '#00ff7f', '#00ffff',
    '#1e90ff', '#9370db', '#ff69b4', '#cd5c5c', '#8fbc8f',
    '#ffd700', '#32cd32', '#00bfff', '#ff00ff', '#ff8c00'
]

colors_dark = [
    '#cc0000', '#cc5500', '#cccc00', '#00cc66', '#00cccc',
    '#0066cc', '#6a5acd', '#ff1493', '#8b0000', '#2e8b57',
    '#daa520', '#228b22', '#0099cc', '#cc00cc', '#d2691e'
] 

colors_names: dict[str, str] = {
    "yellow": "#ffdd00", "red": "#ff006d", "cyan": "#00ffff", "purple": "#8f00ff", "orange": "#ff7d00",
    "lime": "#adff02", "green": "#04e762", "pink": "#ff00cc", "white": "#ffffff", "blue": "#0d41e1", 
    "sky": "#0080ff", "spring": "#00F59B"
} 

publication_colors: dict[str, str] = {
    "primary_blue": "#6A7BC8",
    "dark_blue": "#4C61BD", 
    "light_blue": "#8AA1E9",
    "purple": "#A064B9",
    "dark_purple": "#763F8D"
}

colors_gray : list[str] = ["#333333", "#666666", "#999999", "#CCCCCC"]
colors_blue: list[str] = ["#3a0ca3", "#3f37c9", "#4361ee", "#4895ef"]
colors_purple: list[str] = ["#6411ad", "#822faf", "#973aa8", "#c05299"]
colors_red: list[str] = ["#800f2f", "#a4133c", "#c9184a"]
# fmt:on

# Universal font settings
FONT_FAMILY = "Helvetica"
FONT_COLOR = "#333333"

# Font sizes for different elements
FONT_SIZES: dict[str, int] = {
    "title": 20,
    "axis_title": 16,
    "tick_label": 16,
    "subtitle": 11,
    "legend": 12,
    "subplot_title": 16,
}

# Universal axis style settings
AXIS_STYLE: dict[str, Any] = {
    "showgrid": True,
    "gridwidth": 1,
    "gridcolor": "#E7E7E7",
    "zeroline": False,
    "linewidth": 2,
    "linecolor": "#333333",
}

# Universal layout settings
LAYOUT_STYLE: dict[str, Any] = {
    "plot_bgcolor": "#FBFCFF",
    "paper_bgcolor": "#FBFCFF",
    "margin": dict(t=40, b=40, r=40),
}

# Development style settings
DEVELOPMENT_STYLE: dict[str, Any] = {
    "template": "plotly_dark",
    "plot_bgcolor": "black",
    "paper_bgcolor": "black",
    "font": dict(color="white"),
}


def get_font_dict(size: int) -> dict[str, Any]:
    """Helper function to create consistent font dictionaries.

    Args:
        size: Font size to use

    Returns:
        Dictionary with font settings
    """
    return dict(family=FONT_FAMILY, size=size, color=FONT_COLOR)


def apply_publication_fonts(fig: go.Figure) -> None:
    """Apply publication-quality font settings to a figure.

    Args:
        fig: A plotly figure
    """
    # Update global font
    fig.update_layout(font=get_font_dict(FONT_SIZES["tick_label"]))

    # Update title font if title exists
    if fig.layout.title is not None:
        fig.layout.title.update(font=get_font_dict(FONT_SIZES["title"]))


def update_axis(axis: go.layout.XAxis | go.layout.YAxis, axis_style: dict[str, Any]) -> None:
    """Helper function to update a single axis with publication styling.

    Args:
        axis: Axis to update
        axis_style: Style parameters to apply
    """
    axis.update(
        axis_style, title_font=get_font_dict(FONT_SIZES["axis_title"]), tickfont=get_font_dict(FONT_SIZES["tick_label"])
    )


def apply_axis_style(fig: go.Figure, row: int | None = None, col: int | None = None, **kwargs: Any) -> None:
    """Apply publication-quality axis styling to a figure.

    Args:
        fig: A plotly figure
        row: Optional row index for subplots
        col: Optional column index for subplots
        **kwargs: Additional axis style parameters to override defaults
    """
    axis_style = AXIS_STYLE.copy()
    axis_style.update(kwargs)

    if row is not None and col is not None:
        update_axis(fig.get_xaxes()[row - 1], axis_style)
        update_axis(fig.get_yaxes()[col - 1], axis_style)
    else:
        update_axis(fig.layout.xaxis, axis_style)
        update_axis(fig.layout.yaxis, axis_style)


def apply_publication_style(fig: go.Figure, **kwargs: Any) -> None:
    """Apply all publication-quality styling to a figure.

    Args:
        fig: A plotly figure
        show_legend: Whether to show and style the legend
        **kwargs: Additional layout parameters to override defaults
    """
    # Apply fonts
    apply_publication_fonts(fig)

    # Apply axis style to all axes
    # Handle both single plot and subplot cases by looking for axis objects in layout
    for key in fig.layout:
        if key.startswith("xaxis") or key.startswith("yaxis"):
            update_axis(getattr(fig.layout, key), AXIS_STYLE)

    layout_style: dict[str, Any] = LAYOUT_STYLE.copy()
    layout_style.update(kwargs)
    fig.update_layout(layout_style)


def apply_development_style(fig: go.Figure) -> None:
    """Apply dark theme development styling to a figure.

    This applies a dark theme with black background, suitable for development
    and debugging visualizations.

    Args:
        fig: A plotly figure
    """
    fig.update_layout(**DEVELOPMENT_STYLE)
