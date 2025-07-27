from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from directmultistep.analysis import style
from directmultistep.utils.logging_config import logger


def load_training_df(train_path: Path, run_name: str, ignore_ids: list[int] | None = None) -> pd.DataFrame:
    logger.debug(f"Loading {run_name=}")
    log_path = train_path / run_name / "lightning_logs"
    dfs = []
    versions = [log.name for log in log_path.glob("version_*")]
    logger.debug(f"Found versions: {versions} for {run_name}")
    ignored_folders = {f"version_{i}" for i in ignore_ids} if ignore_ids is not None else set()
    for version in sorted(versions, key=lambda x: int(x.split("_")[1])):
        if version in ignored_folders:
            continue
        temp_df = pd.read_csv(log_path / version / "metrics.csv")
        logger.debug(f"Loaded df with shape {temp_df.shape}")
        dfs.append(temp_df)
    df = pd.concat(dfs)
    df = df.reset_index(drop=True)
    return df


def create_train_trace(df: pd.DataFrame, run_name: str, color: str, x_axis: str) -> go.Scatter:
    return go.Scatter(
        x=df[x_axis],
        y=df["train_loss"],
        mode="lines",
        name=f"train_loss {run_name}",
        line_color=color,
        showlegend=True,
        legendgroup=run_name,
    )


def create_val_trace(df: pd.DataFrame, run_name: str, color: str, x_axis: str) -> go.Scatter:
    val_df = df.dropna(subset=["val_loss"])
    return go.Scatter(
        x=val_df[x_axis],
        y=val_df["val_loss"],
        mode="lines+markers",
        name=f"val_loss {run_name}",
        line_color=color,
        showlegend=True,
        hovertemplate="%{fullData.name}<br>"
        + "epoch=%{customdata}<br>"
        + f"{x_axis}=%{{x}}<br>"
        + "val_loss=%{y}<extra></extra>",
        customdata=val_df["epoch"],
    )


@dataclass
class RunConfig:
    """Configuration for a training run visualization."""

    run_name: str  # Folder name of the run
    trace_name: str  # Display name for the traces
    include_val: bool = True  # Whether to include validation curve
    ignore_ids: list[int] | None = None  # Version IDs to ignore when loading data


def plot_training_curves(
    train_path: Path,
    runs: list[RunConfig],
    x_axis: str = "processed_tokens",
) -> go.Figure:
    """Create a figure showing training and validation curves for multiple runs.

    Args:
        train_path: Path to training data directory
        runs: List of run configurations specifying what and how to plot
        x_axis: Column to use for x-axis values ("processed_tokens", "epoch", or "step")

    Returns:
        Plotly figure with training and validation curves
    """
    traces = []
    for i, run in enumerate(runs):
        df = load_training_df(train_path, run.run_name, run.ignore_ids)
        color_idx = i % len(style.colors_light)
        traces.append(
            create_train_trace(df, run.trace_name, style.colors_light[color_idx % len(style.colors_light)], x_axis)
        )
        if run.include_val:
            traces.append(
                create_val_trace(df, run.trace_name, style.colors_dark[color_idx % len(style.colors_dark)], x_axis)
            )

    fig = go.Figure(data=traces)

    fig.update_layout(
        title="Training Loss",
        xaxis_title=x_axis,
        yaxis_title="Loss",
        width=1000,
    )
    style.apply_development_style(fig)

    return fig


def get_lr_trace(df: pd.DataFrame, run_name: str) -> go.Scatter:
    return go.Scatter(
        x=df["step"],
        y=df["train_lr"],
        mode="lines",
        name=f"learning rate {run_name}",
        showlegend=True,
        legendgroup=run_name,
    )


def plot_learning_rates(
    train_path: Path,
    runs: list[RunConfig],
) -> go.Figure:
    """Create a figure showing learning rate curves for multiple runs.

    Args:
        train_path: Path to training data directory
        runs: List of run configurations specifying what and how to plot

    Returns:
        Plotly figure with learning rate curves
    """
    traces = []
    for run in runs:
        df = load_training_df(train_path, run.run_name, run.ignore_ids)
        traces.append(get_lr_trace(df, run.trace_name))

    fig = go.Figure(data=traces)

    fig.update_layout(
        title="Learning Rate",
        xaxis_title="Step",
        yaxis_title="Learning Rate",
        width=800,
    )
    style.apply_development_style(fig)

    return fig


if __name__ == "__main__":
    train_path = Path("data/training")

    runs = [
        RunConfig(run_name="baseline_run", trace_name="Baseline Model"),
        RunConfig(run_name="improved_run", trace_name="Improved Model", include_val=True),
        RunConfig(
            run_name="experimental_run",
            trace_name="Experimental Model",
            include_val=False,  # Only show training curve
        ),
    ]

    fig = plot_training_curves(train_path, runs)
    fig.show()
