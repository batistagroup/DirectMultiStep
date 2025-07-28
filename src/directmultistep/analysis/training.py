import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import plotly.graph_objects as go

from directmultistep.analysis import style
from directmultistep.utils.logging_config import logger


def _cast_numeric_values(data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """attempts to cognize string values into floats. fails silently."""
    # these are the columns we expect to be numeric afaict from the original code
    numeric_cols = {"step", "train_loss", "val_loss", "epoch", "processed_tokens", "train_lr"}
    for row in data:
        for key, value in row.items():
            if key in numeric_cols and value:
                try:
                    row[key] = float(value)
                except (ValueError, TypeError):
                    row[key] = None  # nullify if conversion fails
            elif not value:
                row[key] = None  # treat empty strings as null
    return data


def load_training_data(
    train_path: Path, run_name: str, ignore_ids: list[int] | None = None
) -> list[dict[str, float | None]]:
    """loads data, but returns a list of dicts, not a... DataFrame."""
    logger.debug(f"loading {run_name=}")
    log_path = train_path / run_name / "lightning_logs"
    all_rows = []
    versions = sorted([p for p in log_path.glob("version_*") if p.is_dir()], key=lambda x: int(x.name.split("_")[1]))
    logger.debug(f"found versions: {[v.name for v in versions]} for {run_name}")
    ignored_folders = {f"version_{i}" for i in ignore_ids or []}

    for version_path in versions:
        if version_path.name in ignored_folders:
            continue
        metrics_file = version_path / "metrics.csv"
        if not metrics_file.exists():
            continue
        with open(metrics_file, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            all_rows.extend(list(reader))

    return _cast_numeric_values(all_rows)


def create_train_trace(data: list[dict[str, float | None]], run_name: str, color: str, x_axis: str) -> go.Scatter:
    return go.Scatter(
        x=[r[x_axis] for r in data if r.get(x_axis) is not None],
        y=[r["train_loss"] for r in data if r.get("train_loss") is not None],
        mode="lines",
        name=f"train_loss {run_name}",
        line_color=color,
        showlegend=True,
        legendgroup=run_name,
    )


def create_val_trace(data: list[dict[str, float | None]], run_name: str, color: str, x_axis: str) -> go.Scatter:
    val_data = [r for r in data if r.get("val_loss") is not None]
    return go.Scatter(
        x=[r[x_axis] for r in val_data],
        y=[r["val_loss"] for r in val_data],
        mode="lines+markers",
        name=f"val_loss {run_name}",
        line_color=color,
        showlegend=True,
        hovertemplate="%{fullData.name}<br>"
        + "epoch=%{customdata}<br>"
        + f"{x_axis}=%{{x}}<br>"
        + "val_loss=%{y}<extra></extra>",
        customdata=[r.get("epoch") for r in val_data],
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
    log_x: bool = False,
    log_y: bool = False,
) -> go.Figure:
    """makes a graph. you get it."""
    traces = []
    for i, run in enumerate(runs):
        data = load_training_data(train_path, run.run_name, run.ignore_ids)
        if not data:
            logger.debug(f"no data found for {run.run_name}, skipping.")
            continue

        color_idx = i % len(style.colors_light)
        traces.append(create_train_trace(data, run.trace_name, style.colors_light[color_idx], x_axis))
        if run.include_val:
            traces.append(create_val_trace(data, run.trace_name, style.colors_dark[color_idx], x_axis))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title="Training Loss",
        xaxis_title=x_axis,
        yaxis_title="Loss",
        xaxis_type="log" if log_x else "linear",
        yaxis_type="log" if log_y else "linear",
    )
    style.apply_development_style(fig)
    return fig


def get_lr_trace(data: list[dict[str, float | None]], run_name: str) -> go.Scatter:
    lr_data = [r for r in data if r.get("train_lr") is not None and r.get("step") is not None]
    return go.Scatter(
        x=[r["step"] for r in lr_data],
        y=[r["train_lr"] for r in lr_data],
        mode="lines",
        name=f"learning rate {run_name}",
        showlegend=True,
        legendgroup=run_name,
    )


def plot_learning_rates(train_path: Path, runs: list[RunConfig]) -> go.Figure:
    """makes another graph. also obvious."""
    traces = []
    for run in runs:
        data = load_training_data(train_path, run.run_name, run.ignore_ids)
        if data:
            traces.append(get_lr_trace(data, run.trace_name))

    fig = go.Figure(data=traces)
    fig.update_layout(title="Learning Rate", xaxis_title="Step", yaxis_title="Learning Rate", width=800)
    style.apply_development_style(fig)
    return fig


if __name__ == "__main__":
    train_path = Path("data/training")

    runs = [RunConfig(run_name="sm_6x3_6x3_256_noboth_unique", trace_name="sm 6x3 6x3 256 noboth unique")]

    fig = plot_training_curves(train_path, runs, log_x=False, log_y=True)
    fig.show()
