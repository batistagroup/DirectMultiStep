import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm

from directmultistep.analysis import style
from directmultistep.analysis.style import (
    FONT_COLOR,
    apply_publication_style,
    publication_colors,
)
from directmultistep.utils.pre_process import is_convergent


def create_split_bar_trace(
    route_lengths: list[int], label: str, sep_threshold: int, color: str
) -> tuple[go.Bar, go.Bar]:
    """Create two bar traces split by a threshold value.

    Args:
        route_lengths: List of route lengths to plot
        label: Label for the traces
        sep_threshold: Threshold value to split traces
        color: Color for both traces

    Returns:
        Tuple of two bar traces - one for values <= threshold, one for values > threshold
    """
    unique_lengths, counts = np.unique(route_lengths, return_counts=True)
    relative_abundance = counts / len(route_lengths)

    trace_settings = dict(
        name=label,
        marker=dict(color=color),
        hovertemplate="Route Length: %{x}<br>Relative Abundance: %{y:.2%}<extra></extra>",
        textposition="auto",
    )

    # Split data by threshold
    mask_short = unique_lengths <= sep_threshold
    mask_long = unique_lengths > sep_threshold

    trace1 = go.Bar(x=unique_lengths[mask_short], y=relative_abundance[mask_short], **trace_settings)

    trace2 = go.Bar(x=unique_lengths[mask_long], y=relative_abundance[mask_long], showlegend=False, **trace_settings)

    return trace1, trace2


def plot_route_length_distribution(
    train_steps: list[int],
    n1_steps: list[int],
    n5_steps: list[int],
) -> go.Figure:
    """Create a split plot showing the distribution of route lengths for different datasets.

    Args:
        train_steps: List of route lengths from training set
        n1_steps: List of route lengths from n1 dataset
        n5_steps: List of route lengths from n5 dataset
        save_path: Optional path to save the figure. If None, figure is not saved.

    Returns:
        Plotly figure object containing the visualization
    """
    # Plot settings
    colors = [FONT_COLOR, publication_colors["dark_blue"], publication_colors["dark_purple"]]
    sep_threshold = 6

    fig = make_subplots(rows=1, cols=2)

    datasets = [(train_steps, "Training Routes", colors[0]), (n1_steps, "n1", colors[1]), (n5_steps, "n5", colors[2])]

    for steps, label, color in datasets:
        trace1, trace2 = create_split_bar_trace(steps, label, sep_threshold, color)
        fig.add_trace(trace1, row=1, col=1)
        fig.add_trace(trace2, row=1, col=2)

    style.AXIS_STYLE["linecolor"] = None
    apply_publication_style(fig)

    # fmt:off
    fig.update_layout(width=1000, height=300, margin=dict(l=100, r=50, t=20, b=50))

    for col in [1, 2]:
        fig.update_xaxes(title_text="<b>Route Length</b>", showgrid=False, row=1, col=col, dtick=1)

    fig.update_yaxes(title_text="<b>Relative Abundance</b>", tickformat=",.0%", dtick=0.1, row=1, col=1)
    fig.update_yaxes(tickformat=",.2%", dtick=0.003, row=1, col=2)
    fig.update_layout(legend=dict( orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=0.99))

    return fig


def create_leaf_bar_trace(path_strings: list[str], label: str, color: str) -> go.Bar:
    """Create a bar trace showing distribution of number of leaves at root node.

    Args:
        path_strings: List of path strings to analyze
        label: Label for the trace
        color: Color for the trace

    Returns:
        Bar trace showing leaf distribution
    """
    n_leaves = []
    for path in tqdm(path_strings):
        path_dict = eval(path)
        root_leaves = sum(
            1 for child in path_dict["children"] if "children" not in child or len(child["children"]) == 0
        )
        n_leaves.append(root_leaves)

    unique_lengths, counts = np.unique(n_leaves, return_counts=True)

    unique_lengths = unique_lengths[:4]
    counts = counts[:4]
    relative_abundance = counts / len(path_strings)

    return go.Bar(
        x=unique_lengths,
        y=relative_abundance,
        name=label,
        marker=dict(color=color),
        hovertemplate="Number of Leaves: %{x}<br>Relative Frequency: %{y:.2%}<extra></extra>",
        text=[f"{v:.1%}" for v in relative_abundance],
        textposition="auto",
        textfont=dict(size=12),
    )


def plot_leaf_distribution(
    train_paths: list[str],
    n1_paths: list[str],
    n5_paths: list[str],
) -> go.Figure:
    """Create a plot showing the distribution of number of leaves for different datasets.

    Args:
        train_paths: List of path strings from training set
        n1_paths: List of path strings from n1 dataset
        n5_paths: List of path strings from n5 dataset

    Returns:
        Plotly figure object containing the visualization
    """
    colors = [FONT_COLOR, publication_colors["dark_blue"], publication_colors["dark_purple"]]
    datasets = [(train_paths, "Training Routes", colors[0]), (n1_paths, "n1", colors[1]), (n5_paths, "n5", colors[2])]

    fig = go.Figure()

    for paths, label, color in datasets:
        fig.add_trace(create_leaf_bar_trace(paths, label, color))

    style.AXIS_STYLE["linecolor"] = None
    apply_publication_style(fig)
    # fmt:off
    fig.update_layout(width=700, height=250, bargap=0.08, yaxis_range=[0, 0.81],
        margin=dict(l=100, r=50, t=20, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=0.99))

    fig.update_xaxes(title_text="<b>Number of Leaves at Root Node</b>", dtick=1, showgrid=False)
    fig.update_yaxes(title_text="<b>Relative Frequency</b>", tickformat=",.0%")

    return fig


def create_convergent_fraction_trace(
    path_strings: list[str], route_lengths: list[int], label: str, color: str
) -> go.Bar:
    """Create a bar trace showing fraction of convergent routes by length.

    Args:
        path_strings: List of path strings to analyze
        route_lengths: List of corresponding route lengths
        label: Label for the trace
        color: Color for the trace

    Returns:
        Bar trace showing convergent fraction by length
    """
    # Group paths by length and compute convergent fraction
    max_length = 10
    fractions = []
    lengths = []

    for length in tqdm(range(1, max_length + 1)):
        # Get paths of this length
        mask = np.array(route_lengths) == length
        paths_at_length = np.array(path_strings)[mask]

        if len(paths_at_length) == 0:
            continue

        # Compute fraction of convergent paths
        n_convergent = sum(1 for path in paths_at_length if is_convergent(eval(path)))
        fraction = n_convergent / len(paths_at_length)

        fractions.append(fraction)
        lengths.append(length)

    return go.Bar(
        x=lengths,
        y=fractions,
        name=label,
        marker=dict(color=color),
        hovertemplate="Route Length: %{x}<br>Convergent Fraction: %{y:.2%}<extra></extra>",
        # text=[f"{v:.1%}" for v in fractions],
        # textposition="auto",
        # textfont=dict(size=12),
    )


def plot_convergent_fraction_by_length(
    train_paths: list[str],
    train_lengths: list[int],
    n1_paths: list[str],
    n1_lengths: list[int],
    n5_paths: list[str],
    n5_lengths: list[int],
) -> go.Figure:
    """Create a plot showing fraction of convergent routes by length for different datasets.

    Args:
        train_paths: List of path strings from training set
        train_lengths: List of route lengths from training set
        n1_paths: List of path strings from n1 dataset
        n1_lengths: List of route lengths from n1 dataset
        n5_paths: List of path strings from n5 dataset
        n5_lengths: List of route lengths from n5 dataset

    Returns:
        Plotly figure object containing the visualization
    """
    colors = [FONT_COLOR, publication_colors["dark_blue"], publication_colors["dark_purple"]]
    datasets = [
        (train_paths, train_lengths, "Training Routes", colors[0]),
        (n1_paths, n1_lengths, "n1", colors[1]),
        (n5_paths, n5_lengths, "n5", colors[2]),
    ]

    fig = go.Figure()

    for paths, lengths, label, color in datasets:
        fig.add_trace(create_convergent_fraction_trace(paths, lengths, label, color))

    style.AXIS_STYLE["linecolor"] = None
    apply_publication_style(fig)
    # fmt:off
    fig.update_layout(width=1000, height=250, bargap=0.15,
        margin=dict(l=100, r=50, t=20, b=50),
        legend=dict( orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=0.99),
        yaxis=dict(range=[0, 0.31], dtick=0.05),
    )
    # fmt:on
    fig.update_xaxes(title_text="<b>Route Length</b>", dtick=1, showgrid=False)
    fig.update_yaxes(title_text="<b>Fraction Convergent</b>", tickformat=",.0%")

    return fig


def plot_convergent_fraction_overall(
    train_paths: list[str],
    n1_paths: list[str],
    n5_paths: list[str],
) -> go.Figure:
    """Create a plot showing overall fraction of convergent routes for different datasets.

    Args:
        train_paths: List of path strings from training set
        n1_paths: List of path strings from n1 dataset
        n5_paths: List of path strings from n5 dataset

    Returns:
        Plotly figure object containing the visualization
    """
    colors = [FONT_COLOR, publication_colors["dark_blue"], publication_colors["dark_purple"]]
    datasets = [
        (train_paths, "Training Routes", colors[0]),
        (n1_paths, "n1", colors[1]),
        (n5_paths, "n5", colors[2]),
    ]

    fractions = []
    labels = []
    colors_used = []

    for paths, label, color in tqdm(datasets):
        n_convergent = sum(1 for path in paths if is_convergent(eval(path)))
        fraction = n_convergent / len(paths)
        fractions.append(fraction)
        labels.append(label)
        colors_used.append(color)

    fig = go.Figure()
    # fmt:off
    fig.add_trace(
        go.Bar(x=labels, y=fractions,
            marker=dict(color=colors_used), text=[f"{v:.1%}" for v in fractions], textposition="auto"))

    style.AXIS_STYLE["linecolor"] = None
    apply_publication_style(fig)
    fig.update_layout(width=600, height=250, margin=dict(l=100, r=50, t=20, b=50), showlegend=False)
    # fmt:on
    fig.update_xaxes(title_text="<b>Dataset</b>", showgrid=False)
    fig.update_yaxes(title_text="<b>Fraction Convergent</b>", tickformat=",.0%", dtick=0.05, range=[0, 0.31])

    return fig
