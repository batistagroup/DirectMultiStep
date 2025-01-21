import pickle
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import yaml
from plotly.subplots import make_subplots
from tqdm import tqdm

from directmultistep.analysis import style
from directmultistep.analysis.style import (
    FONT_SIZES,
    apply_publication_style,
    get_font_dict,
)
from directmultistep.utils.io import DatasetDict
from directmultistep.utils.logging_config import logger
from directmultistep.utils.post_process import (
    PathsProcessedType,
    calculate_top_k_counts_by_step_length,
    find_matching_paths,
    find_top_n_accuracy,
)
from directmultistep.utils.pre_process import is_convergent


@dataclass
class ModelPlotConfig:
    """Configuration for model plotting.

    Attributes:
        model_name: Name of the model (e.g. 'flex_20M', 'flash_10M').
        epoch: Epoch number as string (e.g. 'epoch=20').
        variant_base: Base variant string (e.g. 'b50_sm_st_ea=1_da=1').
        true_reacs: Whether to use true reactions.
        stock: Whether to use stock compounds.
        ds_name: Dataset name (e.g. 'n1', 'n5').
    """

    model_name: str
    epoch: str
    variant_base: str
    true_reacs: bool = True
    stock: bool = True
    ds_name: str = "n1"

    def __post_init__(self) -> None:
        if "nosm" in self.variant_base:
            self.true_reacs = False

    @property
    def display_name(self) -> str:
        """Generate display name from model name.

        Returns:
            str: Display name of the model.
        """
        base = self.model_name.replace("_", " ").title()

        if "nosm" in self.variant_base:
            base += " (no SM)"
        elif "sm" in self.variant_base:
            base += " (SM)"

        if "ea=1" in self.variant_base and "da=1" in self.variant_base:
            base = base.replace("(", "(Mono, ")
            base += ")"
        elif "ea=2" in self.variant_base and "da=2" in self.variant_base:
            base = base.replace("(", "(Duo, ")
            base += ")"

        return base

    @property
    def variant(self) -> str:
        """Get the full variant string.

        Returns:
            str: Full variant string.
        """
        return f"{self.ds_name}_{self.variant_base}"

    @property
    def save_suffix(self) -> str:
        """Get the name of the save file.

        Returns:
            str: Save file suffix.
        """
        return f"{self.model_name}_{self.variant}"

    @property
    def processed_paths_name(self) -> str:
        """Get the name of the processed paths file.

        Returns:
            str: Processed paths file name.
        """
        return f"processed_paths_NS2n_true_reacs={self.true_reacs}_stock={self.stock}.pkl"

    @property
    def correct_paths_name(self) -> str:
        """Get the name of the correct paths file.

        Returns:
            str: Correct paths file name.
        """
        return "correct_paths_NS2n.pkl"

    def with_dataset(self, ds_name: str) -> "ModelPlotConfig":
        """Create a new config with dataset information.

        Args:
            dataset: Dataset dictionary.

        Returns:
            ModelPlotConfig: New config with dataset information.
        """
        return replace(self, ds_name=ds_name)

    def get_result_path(self, eval_path: Path) -> Path:
        """Get the path to the results directory for this config.

        Args:
            eval_path: Path to the evaluation directory.

        Returns:
            Path: Path to the results directory.
        """
        return eval_path / self.model_name / self.epoch / self.variant


def load_predicted_routes(path: Path) -> PathsProcessedType:
    """Load predicted routes from a pickle file.

    Args:
        path: Path to the pickle file.

    Returns:
        PathsProcessedType: Loaded predicted routes.
    """
    with open(path, "rb") as f:
        routes: PathsProcessedType = pickle.load(f)
    logger.info(f"Loaded {len(routes)} predicted routes")
    return routes


def get_convergent_indices(path_strings: list[str]) -> set[int]:
    """Identify indices of convergent routes in dataset.

    Args:
        path_strings: List of path strings.

    Returns:
        set[int]: Set of indices of convergent routes.
    """
    convergent_idxs = set()
    logger.info("Finding convergent routes")
    for i, path_str in enumerate(tqdm(path_strings)):
        path_dict = eval(path_str)
        if is_convergent(path_dict):
            convergent_idxs.add(i)
    return convergent_idxs


def calculate_prediction_stats(predictions: list[int]) -> tuple[float, float, float, float]:
    """Calculate mean and median statistics for a list of predictions.

    Args:
        predictions: List of prediction counts.

    Returns:
        Tuple of (mean, median, filtered_mean, filtered_median) where filtered
        versions only consider predictions with count > 0.
    """
    mean = np.float64(np.mean(predictions)).item()
    median = np.float64(np.median(predictions)).item()

    filtered = [x for x in predictions if x > 0]
    filtered_mean = np.float64(np.mean(filtered)).item() if filtered else 0.0
    filtered_median = np.float64(np.median(filtered)).item() if filtered else 0.0

    return mean, median, filtered_mean, filtered_median


class RouteAnalyzer:
    """Analyzes predicted routes and calculates various statistics."""

    def __init__(self, predicted_routes: PathsProcessedType, true_routes: list[str], k_vals: list[int] | None = None):
        """Initializes the RouteAnalyzer.

        Args:
            predicted_routes: Predicted routes.
            true_routes: True routes.
            k_vals: List of k values for top-k accuracy calculation.
        """
        self.predicted_routes = predicted_routes
        self.true_routes = true_routes
        self.k_vals = k_vals if k_vals is not None else [1, 2, 3, 4, 5, 10, 20, 50]
        self.convergent_idxs = get_convergent_indices(true_routes)
        self.non_convergent_idxs = set(range(len(true_routes))) - self.convergent_idxs

    def analyze_convergence_stats(self) -> None:
        """Analyze and log basic convergence statistics."""
        n_convergent = len(self.convergent_idxs)
        total = len(self.true_routes)
        logger.info(f"Found {n_convergent} convergent routes out of {total} total routes")
        logger.info(f"Percentage convergent: {100 * n_convergent / total:.1f}%")

    def calculate_top_k_accuracies(self, save_path: Path | None = None) -> dict[str, dict[str, str]]:
        """Calculate top-k accuracies for different route subsets and optionally save results.

        Args:
            save_path: Optional path to save detailed accuracies to YAML file.

        Returns:
            dict[str, dict[str, str]]: Dictionary of top-k accuracies.
        """
        results = {}
        route_types = {"all": None, "convergent": self.non_convergent_idxs, "non_convergent": self.convergent_idxs}

        with tqdm(total=len(route_types), desc="Analyzing top-k accuracy") as pbar:
            for route_type, ignore_ids in route_types.items():
                pbar.set_description(f"{route_type} routes")
                _, perm_matches = find_matching_paths(self.predicted_routes, self.true_routes, ignore_ids=ignore_ids)
                results[route_type] = find_top_n_accuracy(perm_matches, self.k_vals)
                pbar.update(1)

        if save_path is not None:
            save_path = save_path / "top_k_accuracy_detailed.yaml"
            with open(save_path, "w") as f:
                yaml.dump(results, f, default_flow_style=False)
            logger.info(f"Saved detailed accuracies to {save_path}")

        return results

    def analyze_and_log_results(self) -> dict[str, dict[str, str]]:
        """Run full analysis and log results.

        Returns:
            dict[str, dict[str, str]]: Dictionary of top-k accuracies.
        """
        self.analyze_convergence_stats()
        results = self.calculate_top_k_accuracies()

        for route_type, accuracies in results.items():
            logger.info(f"\nTop-k accuracy for {route_type} routes:")
            logger.info(accuracies)

        return results

    def visualize_route_distributions(self, dataset_name: str = "") -> go.Figure:
        """Create a publication-quality figure showing the distribution of predicted routes.

        Args:
            dataset_name: Name of the dataset being analyzed, used in plot title.

        Returns:
            go.Figure: Plotly figure object.
        """
        n_predictions = [len(routes) for routes in self.predicted_routes]

        conv_predictions = [n_predictions[i] for i in self.convergent_idxs]
        nonconv_predictions = [n_predictions[i] for i in self.non_convergent_idxs]

        mean_all, median_all, mean_all_filtered, median_all_filtered = calculate_prediction_stats(n_predictions)
        mean_conv, median_conv, mean_conv_filtered, median_conv_filtered = calculate_prediction_stats(conv_predictions)
        mean_nonconv, median_nonconv, mean_nonconv_filtered, median_nonconv_filtered = calculate_prediction_stats(
            nonconv_predictions
        )

        # fmt: off
        fig = make_subplots(rows=1, cols=3,
            subplot_titles=(
                f'All Routes<br><span style="font-size:{FONT_SIZES["subplot_title"]}px">mean: {mean_all:.1f}, median: {median_all:.1f} (mean*: {mean_all_filtered:.1f}, median*: {median_all_filtered:.1f})</span>',
                f'Convergent Routes<br><span style="font-size:{FONT_SIZES["subplot_title"]}px">mean: {mean_conv:.1f}, median: {median_conv:.1f} (mean*: {mean_conv_filtered:.1f}, median*: {median_conv_filtered:.1f})</span>',
                f'Non-convergent Routes<br><span style="font-size:{FONT_SIZES["subplot_title"]}px">mean: {mean_nonconv:.1f}, median: {median_nonconv:.1f} (mean*: {mean_nonconv_filtered:.1f}, median*: {median_nonconv_filtered:.1f})</span>'
            ), horizontal_spacing=0.1)

        histogram_style = dict(opacity =0.75, nbinsx=30,histnorm='percent',marker_color=style.publication_colors["dark_blue"])

        data = [(n_predictions, "All"), (conv_predictions, "Convergent"), (nonconv_predictions, "Non-convergent")]
        for i, (predictions, name) in enumerate(data, start=1):
            fig.add_trace(go.Histogram(x=predictions, name=name, **histogram_style), row=1, col=i)

        title = "Distribution of Predicted Routes per Target"
        if dataset_name:
            title = f"{title} - {dataset_name}"
            
        fig.update_layout(title=dict(text=title, x=0.5, xanchor='center'), showlegend=False, height=400, width=1200,)

        apply_publication_style(fig)

        for i in range(1, 4):
            fig.update_xaxes(title=dict(text="Number of Predicted Routes", font=get_font_dict(FONT_SIZES["axis_title"]), standoff=15), row=1, col=i)
            fig.update_yaxes(title=dict(text="Percentage (%)", font=get_font_dict(FONT_SIZES["axis_title"]), standoff=15), row=1, col=i)
        # fmt: on
        return fig

    @staticmethod
    def create_comparative_bar_plots(
        result_paths: list[Path], trace_names: list[str], k_vals: list[int] | None = None, title: str = ""
    ) -> go.Figure:
        """Create comparative bar plots showing top-k accuracy for different configurations.

        Args:
            result_paths: List of paths to top_k_accuracy_detailed.yaml files.
            trace_names: List of names for each trace (must match length of result_paths).
            k_vals: Optional list of k values to show. If None, shows all k values.
            title: Title for the plot.

        Returns:
            go.Figure: Plotly figure object.
        """
        if len(result_paths) != len(trace_names):
            raise ValueError("Number of result paths must match number of trace names")

        results = []
        for path in result_paths:
            with open(path / "top_k_accuracy_detailed.yaml") as f:
                results.append(yaml.safe_load(f))

        # fmt: off
        fig = make_subplots(rows=3, cols=1, horizontal_spacing=0.07, vertical_spacing=0.12,
            subplot_titles=[f"<b>{t}</b>" for t in ('(a) all routes', '(b) convergent routes', '(c) non-convergent routes')])

        categories = ['all', 'convergent', 'non_convergent']
        positions = [1, 2, 3]

        colors = style.colors_blue + style.colors_purple + style.colors_red

        for cat, pos in zip(categories, positions):
            x = list(results[0][cat].keys())
            x.sort(key=lambda k: int(k.split()[-1]))
            
            if k_vals is not None:
                k_vals_str = [f"Top {k}" for k in k_vals]
                x = [k for k in x if k in k_vals_str]
            
            for i, (result, name) in enumerate(zip(results, trace_names)):
                y = [float(result[cat][k].strip('%')) for k in x]
                
                fig.add_trace(
                    go.Bar(name=name, x=x, y=y, showlegend=pos == 1, marker_color=colors[i % len(colors)],
                        legendgroup=name,), row=pos, col=1)

        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor='center'),
            barmode='group', height=600, width=1000,
            legend=dict(font=get_font_dict(FONT_SIZES["legend"]), orientation="h", yanchor="bottom",
                y=-0.20, xanchor="center", x=0.5, entrywidth=140, tracegroupgap=0))

        style.AXIS_STYLE["linecolor"] = None
        apply_publication_style(fig)

        for i in range(1, 4):
            fig.update_yaxes(dtick=10, title=dict(text="Accuracy (%)", font=get_font_dict(FONT_SIZES["axis_title"])), row=i, col=1)
            fig.update_xaxes(showgrid=False, row=i, col=1)
        # fmt: on
        return fig

    @staticmethod
    def _calculate_accuracy_by_length_data(
        predicted_routes: PathsProcessedType,
        dataset: DatasetDict,
        k_vals: list[int],
        ignore_ids: set[int] | None = None,
    ) -> tuple[list[int], dict[int, dict[str, int]]]:
        """Helper function to calculate accuracy by length data.

        Args:
            predicted_routes: List of predicted routes.
            dataset: Dataset dictionary.
            k_vals: List of k values to calculate accuracy for.
            ignore_ids: Optional set of indices to ignore.

        Returns:
            Tuple of (lengths, step_stats) where step_stats maps length to accuracy stats.
        """
        _, perm_matches = find_matching_paths(predicted_routes, dataset["path_strings"], ignore_ids=ignore_ids)
        step_stats = calculate_top_k_counts_by_step_length(perm_matches, dataset["n_steps_list"], k_vals)
        lengths = list(step_stats.keys())
        return lengths, step_stats

    @staticmethod
    def create_accuracy_by_length_plot(
        result_paths: list[Path],
        datasets: list[DatasetDict],
        configs: list[ModelPlotConfig],
        k_vals: list[int],
        title: str = "",
    ) -> go.Figure:
        """Create plot showing accuracy by route length.

        Args:
            result_paths: List of paths to result directories.
            datasets: List of datasets to analyze.
            configs: List of model configurations.
            k_vals: List of k values to calculate accuracy for.
            title: Title for the plot.

        Returns:
            go.Figure: Plotly figure object.
        """
        fig = go.Figure()

        cset = style.publication_colors
        colors = [cset["primary_blue"], cset["dark_blue"], cset["purple"], cset["dark_purple"]]

        for i, (path, dataset, config) in enumerate(zip(result_paths, datasets, configs)):
            paths_name = config.processed_paths_name

            with open(path / paths_name, "rb") as f:
                predicted_routes = pickle.load(f)

            lengths, step_stats = RouteAnalyzer._calculate_accuracy_by_length_data(predicted_routes, dataset, k_vals)

            for k_idx, k in enumerate(k_vals):
                accuracies = [
                    step_stats[length].get(f"Top {k}", 0) / step_stats[length]["Total"] * 100 for length in lengths
                ]
                # fmt:off
                fig.add_trace(go.Bar(name=f"{dataset['ds_name']} (Top-{k})", x=lengths, y=accuracies, marker_color=colors[i * len(k_vals) + k_idx]))

        fig.update_layout(
            barmode="group",
            height=300,
            width=1000,
            xaxis=dict(title="<b>Route Length</b>", dtick=1),
            yaxis=dict(title="<b>Accuracy (%)</b>", dtick=10, range=[0, 82]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        )
        # fmt: on
        style.AXIS_STYLE["linecolor"] = None
        apply_publication_style(fig)
        fig.update_xaxes(showgrid=False)
        return fig

    @staticmethod
    def create_accuracy_by_length_subplots(
        result_paths: list[Path],
        datasets: list[DatasetDict],
        configs: list[ModelPlotConfig],
        k_vals: list[int],
        title: str = "",
    ) -> go.Figure:
        """Create plot showing accuracy by route length with subplots for all/convergent/non-convergent routes.

        Args:
            result_paths: List of paths to result directories.
            datasets: List of datasets to analyze.
            configs: List of model configurations.
            k_vals: List of k values to calculate accuracy for.
            title: Title for the plot.

        Returns:
            go.Figure: Plotly figure object.
        """
        fig = make_subplots(
            rows=3,
            cols=1,
            subplot_titles=[
                f"<b>{t}</b>" for t in ("(a) all routes", "(b) convergent routes", "(c) non-convergent routes")
            ],
            vertical_spacing=0.12,
        )

        cset = style.publication_colors
        colors = [cset["primary_blue"], cset["dark_blue"], cset["purple"], cset["dark_purple"]]

        for i, (path, dataset, config) in enumerate(zip(result_paths, datasets, configs)):
            paths_name = config.processed_paths_name

            with open(path / paths_name, "rb") as f:
                predicted_routes = pickle.load(f)

            analyzer = RouteAnalyzer(predicted_routes, dataset["path_strings"])
            route_types = {
                "all": (None, 1),
                "convergent": (analyzer.non_convergent_idxs, 2),
                "non_convergent": (analyzer.convergent_idxs, 3),
            }

            for route_type, (ignore_ids, row) in route_types.items():
                lengths, step_stats = RouteAnalyzer._calculate_accuracy_by_length_data(
                    predicted_routes, dataset, k_vals, ignore_ids=ignore_ids
                )

                for k_idx, k in enumerate(k_vals):
                    accuracies = [
                        step_stats[length].get(f"Top {k}", 0) / step_stats[length]["Total"] * 100 for length in lengths
                    ]
                    # fmt:off
                    fig.add_trace(go.Bar(name=f"{dataset['ds_name']} (Top-{k})",
                            x=lengths,y=accuracies, marker_color=colors[i * len(k_vals) + k_idx],
                            showlegend=row == 1, legendgroup=f"{dataset['ds_name']} (Top-{k})"
                        ), row=row, col=1)

        fig.update_layout(
            barmode="group",
            height=900,
            width=1000,
            legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
        )

        for i in range(1, 4):
            fig.update_xaxes(title="<b>Route Length</b>", dtick=1, row=i, col=1, showgrid=False)
            fig.update_yaxes(title="<b>Accuracy (%)</b>", dtick=10, range=[0, 82], row=i, col=1)
        # fmt: on
        style.AXIS_STYLE["linecolor"] = None
        apply_publication_style(fig)
        fig.update_xaxes(showgrid=False)
        return fig

    @staticmethod
    def visualize_route_processing_stages(
        valid_routes: PathsProcessedType,
        processed_routes_no_stock: PathsProcessedType,
        processed_routes_with_stock: PathsProcessedType,
        true_routes: list[str],
        dataset_name: str = "",
        show_filtered_stats: bool = False,
    ) -> go.Figure:
        """Create a publication-quality figure showing the distribution of routes at different processing stages.

        Args:
            valid_routes: Valid routes from beam search.
            processed_routes_no_stock: Routes after canonicalization/removing repetitions.
            processed_routes_with_stock: Routes after applying stock filter.
            true_routes: True routes for convergence analysis.
            dataset_name: Name of the dataset being analyzed.
            show_filtered_stats: Whether to show filtered statistics (mean* and median*).

        Returns:
            go.Figure: Plotly figure object.
        """
        # Get convergent indices
        convergent_idxs = get_convergent_indices(true_routes)
        non_convergent_idxs = set(range(len(true_routes))) - convergent_idxs

        def get_predictions_by_type(routes: PathsProcessedType) -> tuple[list[int], list[int], list[int]]:
            all_predictions = [len(routes) for routes in routes]
            conv_predictions = [all_predictions[i] for i in convergent_idxs]
            nonconv_predictions = [all_predictions[i] for i in non_convergent_idxs]
            return all_predictions, conv_predictions, nonconv_predictions

        valid_all, valid_conv, valid_nonconv = get_predictions_by_type(valid_routes)
        no_stock_all, no_stock_conv, no_stock_nonconv = get_predictions_by_type(processed_routes_no_stock)
        with_stock_all, with_stock_conv, with_stock_nonconv = get_predictions_by_type(processed_routes_with_stock)

        # Create subplot titles
        def create_subtitle(stage: str, predictions: list[int]) -> str:
            mean, median, mean_f, median_f = calculate_prediction_stats(predictions)
            base = f"{stage}<br><span style=\"font-size:{FONT_SIZES['subplot_title']-4}px\">"
            stats = f"mean={mean:.1f}, median={median:.1f}"
            if show_filtered_stats:
                stats += f" (μ*={mean_f:.1f}, m*={median_f:.1f})"
            return base + stats + "</span>"

        # fmt:off
        fig = make_subplots(rows=3, cols=3,
            subplot_titles=[
                create_subtitle("<b>(a) valid routes (all)</b>", valid_all),
                create_subtitle("<b>(b) valid routes (convergent)</b>", valid_conv),
                create_subtitle("<b>(c) valid routes (non-convergent)</b>", valid_nonconv),
                create_subtitle("<b>(d) after canonicalization (all)</b>", no_stock_all),
                create_subtitle("<b>(e) after canonicalization (convergent)</b>", no_stock_conv),
                create_subtitle("<b>(f) after canonicalization (non-convergent)</b>", no_stock_nonconv),
                create_subtitle("<b>(g) after stock filter (all)</b>", with_stock_all),
                create_subtitle("<b>(h) after stock filter (convergent)</b>", with_stock_conv),
                create_subtitle("<b>(i) after stock filter (non-convergent)</b>", with_stock_nonconv),
            ], vertical_spacing=0.10, horizontal_spacing=0.05)

        histogram_style = dict(histnorm='percent', marker_color=style.publication_colors["dark_blue"], marker_line_width=0)

        data = [
            (valid_all, valid_conv, valid_nonconv),
            (no_stock_all, no_stock_conv, no_stock_nonconv),
            (with_stock_all, with_stock_conv, with_stock_nonconv)
        ]

        for row, (all_pred, conv_pred, nonconv_pred) in enumerate(data, start=1):
            for col, predictions in enumerate([all_pred, conv_pred, nonconv_pred], start=1):
                fig.add_trace(go.Histogram(x=predictions, xbins=dict(start=0, end=50, size=2), **histogram_style), row=row, col=col)

        apply_publication_style(fig)
        fig.update_layout(showlegend=False, height=900, width=1200, margin_t=60, bargap=0.03)

        for row in range(1, 4):
            for col in range(1, 4):
                fig.update_xaxes(title=None, dtick=5, range=[0, 50], row=row, col=col)
                if row == 3:
                    fig.update_xaxes(title=dict(text="<b>Number of Routes</b>", font=get_font_dict(FONT_SIZES["axis_title"]), standoff=15), row=row, col=col)
                
                if col == 1:
                    fig.update_yaxes(title=dict(text="<b>Percentage (%)</b>", font=get_font_dict(FONT_SIZES["axis_title"]), standoff=15), row=row, col=col)
                else:
                    fig.update_yaxes(title=None, row=row, col=col)

        return fig


def process_model_configs(
    eval_path: Path, configs: list[ModelPlotConfig], dataset: DatasetDict
) -> tuple[list[Path], list[str]]:
    """Process model configurations and ensure top-k accuracies are calculated.

    Args:
        eval_path: Path to evaluation directory.
        configs: List of model configurations.
        dataset: Dataset to process.

    Returns:
        Tuple of (result_paths, trace_names) for plotting.
    """
    result_paths = []
    trace_names = []

    for config in configs:
        res_path = config.get_result_path(eval_path)
        accuracy_file = res_path / "top_k_accuracy_detailed.yaml"

        if not accuracy_file.exists():
            logger.info(f"Calculating accuracies for {config.display_name}...")
            predicted_routes = load_predicted_routes(res_path / config.processed_paths_name)
            analyzer = RouteAnalyzer(predicted_routes, dataset["path_strings"])
            analyzer.calculate_top_k_accuracies(save_path=res_path)

        result_paths.append(res_path)
        trace_names.append(config.display_name)

    return result_paths, trace_names
