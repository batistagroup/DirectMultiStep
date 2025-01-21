import pickle
from pathlib import Path

import plotly.io as pio

from directmultistep.analysis.paper.dataset_analysis import (
    plot_convergent_fraction_by_length,
    plot_convergent_fraction_overall,
    plot_leaf_distribution,
    plot_route_length_distribution,
)
from directmultistep.analysis.paper.linear_vs_convergent import (
    ModelPlotConfig,
    RouteAnalyzer,
    process_model_configs,
)
from directmultistep.utils.io import load_dataset_sm
from directmultistep.utils.logging_config import logger

pio.kaleido.scope.mathjax = None

base_path = Path(__name__).resolve().parent
save_path = base_path / "data" / "figures" / "paper"
base_path = Path("/Users/morgunov/batista/RetroChallenge")
prcsd_path = base_path / "data" / "processed"
eval_path = base_path / "data" / "evaluation"
save_path.mkdir(parents=True, exist_ok=True)
folders = [f.name for f in eval_path.glob("*/")]


if __name__ == "__main__":
    # Load datasets
    train_dataset = load_dataset_sm(prcsd_path / "unique_dataset_nperms=3_nsms=all_noboth.pkl")
    n1_dataset = load_dataset_sm(prcsd_path / "n1_dataset_nperms=1_nsms=1.pkl")
    n5_dataset = load_dataset_sm(prcsd_path / "n5_dataset_nperms=1_nsms=1.pkl")

    rerun = {
        "route-distribution": False,
        "leaf-distribution": False,
        "convergent-fraction": False,
        "topk-accuracy": False,
        "extraction-distribution": False,
        "accuracy-by-length": True,
    }

    # ------------ Route Length Distribution in Datasets ------------
    if rerun["route-distribution"]:
        fig = plot_route_length_distribution(
            train_dataset["n_steps_list"],
            n1_dataset["n_steps_list"],
            n5_dataset["n_steps_list"],
        )
        fig.write_image(save_path / "route_length_distribution.pdf")
        # fig.write_html(save_path / "route_length_distribution.html", include_plotlyjs="cdn")

    # ------------ Leaf Distribution in Datasets ------------
    if rerun["leaf-distribution"]:
        fig = plot_leaf_distribution(
            train_dataset["path_strings"],
            n1_dataset["path_strings"],
            n5_dataset["path_strings"],
        )
        fig.write_image(save_path / "leaf_distribution.pdf")
        # fig.write_html(save_path / "leaf_distribution.html", include_plotlyjs="cdn")

    # ------------ Convergent Route Fraction by Length ------------
    if rerun["convergent-fraction"]:
        fig = plot_convergent_fraction_by_length(
            train_dataset["path_strings"],
            train_dataset["n_steps_list"],
            n1_dataset["path_strings"],
            n1_dataset["n_steps_list"],
            n5_dataset["path_strings"],
            n5_dataset["n_steps_list"],
        )
        # fig.show()
        fig.write_image(save_path / "convergent_fraction_by_length.pdf")
        # fig.write_html(save_path / "convergent_fraction_by_length.html", include_plotlyjs="cdn")

        fig = plot_convergent_fraction_overall(
            train_dataset["path_strings"],
            n1_dataset["path_strings"],
            n5_dataset["path_strings"],
        )
        fig.write_image(save_path / "convergent_fraction_overall.pdf")
        # fig.write_html(save_path / "convergent_fraction_overall.html", include_plotlyjs="cdn")

    # ----------------------------------------------------------------
    # fmt:off
    model_configs = [
        ModelPlotConfig(model_name="flex_20M", epoch="epoch=20", variant_base="b50_sm_st_ea=1_da=1"),
        ModelPlotConfig(model_name="flash_10M", epoch="epoch=46", variant_base="b50_sm_st"),
        ModelPlotConfig(model_name="flash_20M", epoch="epoch=31", variant_base="b50_sm_st"),
        ModelPlotConfig(model_name="flex_20M", epoch="epoch=20", variant_base="b50_sm_st_ea=2_da=2"),
        ModelPlotConfig(model_name="flash_10M", epoch="epoch=46", variant_base="b50_nosm_st"),
        ModelPlotConfig(model_name="flex_20M", epoch="epoch=20", variant_base="b50_nosm_st_ea=2_da=2"),
        ModelPlotConfig(model_name="deep_40M", epoch="epoch=47", variant_base="b50_nosm_st"),
        ModelPlotConfig(model_name="wide_40M", epoch="epoch=31", variant_base="b50_nosm_st_ea=2_da=2"),
        ModelPlotConfig(model_name="explorer_19M", epoch="epoch=18", variant_base="b50_sm_nost_ea=2_da=2"),
        ModelPlotConfig(model_name="explorer_19M", epoch="epoch=18", variant_base="b50_nosm_nost_ea=2_da=2"),
        ModelPlotConfig(model_name="explorer_50M", epoch="epoch=16", variant_base="b50_nosm_nost_ea=2_da=2"),
    ]
    # fmt:on

    # ------------ Top-K Accuracy for Convergent/Non-Convergent Routes ------------
    if rerun["topk-accuracy"]:
        for dataset in [n1_dataset, n5_dataset]:
            configs = [config.with_dataset(dataset["ds_name"]) for config in model_configs]
            result_paths, trace_names = process_model_configs(eval_path, configs, dataset)
            fig = RouteAnalyzer.create_comparative_bar_plots(
                result_paths,
                trace_names,
                k_vals=[1, 2, 3, 4, 5, 10],  # Only show these k values
                # title="Top-k Accuracy Comparison - n5 Dataset",
            )
            fig.write_image(
                save_path / f"{dataset['ds_name']}_topk_accuracy_subplots.pdf",
            )
            # fig.write_html(save_path / f"{dataset['ds_name']}_topk_accuracy_subplots.html", include_plotlyjs="cdn")

    # ------------ Route Distribution Plots ------------
    if rerun["extraction-distribution"]:
        for dataset in [n1_dataset, n5_dataset]:
            for config in model_configs:
                config = config.with_dataset(dataset["ds_name"])
                logger.info(f"Processing {config.model_name} evaluation {config.variant}")
                res_path = config.get_result_path(eval_path)

                with open(res_path / "valid_paths_NS2n.pkl", "rb") as f:
                    valid_routes = pickle.load(f)
                with open(res_path / "processed_paths_NS2n_true_reacs=False_stock=False.pkl", "rb") as f:
                    processed_no_stock = pickle.load(f)
                with open(res_path / "processed_paths_NS2n_true_reacs=False_stock=True.pkl", "rb") as f:
                    processed_with_stock = pickle.load(f)

                fig = RouteAnalyzer.visualize_route_processing_stages(
                    valid_routes=valid_routes,
                    processed_routes_no_stock=processed_no_stock,
                    processed_routes_with_stock=processed_with_stock,
                    true_routes=dataset["path_strings"],
                    dataset_name=f"{dataset['ds_name']} Dataset ({config.model_name}, {config.display_name})",
                    show_filtered_stats=False,
                )
                fig.write_image(save_path / f"{dataset['ds_name']}_route_processing_stages_{config.save_suffix}.pdf")
                # fig.write_html(
                #     save_path / f"{dataset['ds_name']}_route_processing_stages_{config.save_suffix}.html",
                #     include_plotlyjs="cdn",
                # )

    # ------------ Top-K Accuracy by Route Length ------------
    if rerun["accuracy-by-length"]:
        for base_config in model_configs:
            datasets = [n1_dataset, n5_dataset]
            configs = [base_config.with_dataset(ds["ds_name"]) for ds in datasets]

            result_paths = []
            trace_names = []
            for config, dataset in zip(configs, datasets):
                paths, names = process_model_configs(eval_path, [config], dataset)
                result_paths.extend(paths)
                trace_names.extend(names)

            # Create single plot
            fig = RouteAnalyzer.create_accuracy_by_length_plot(
                result_paths=result_paths,
                datasets=datasets,
                configs=configs,
                k_vals=[1, 10],
                title="Top-k Accuracy by Route Length",
            )
            fig.write_image(save_path / f"accuracy_by_length_{configs[0].save_suffix}.pdf")
            # fig.write_html(save_path / f"accuracy_by_length_{configs[0].save_suffix}.html", include_plotlyjs="cdn")

            # Create subplot figure
            fig = RouteAnalyzer.create_accuracy_by_length_subplots(
                result_paths=result_paths,
                datasets=datasets,
                configs=configs,
                k_vals=[1, 10],
                title="Top-k Accuracy by Route Length - Route Type Comparison",
            )
            fig.write_image(save_path / f"accuracy_by_length_subplots_{configs[0].save_suffix}.pdf")
            # fig.write_html(
            #     save_path / f"accuracy_by_length_subplots_{configs[0].save_suffix}.html", include_plotlyjs="cdn"
            # )
