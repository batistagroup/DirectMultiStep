from pathlib import Path

from directmultistep.analysis.training import (
    RunConfig,
    plot_learning_rates,
    plot_training_curves,
)

data_path = Path(__name__).resolve().parent / "data"
train_path = data_path / "training"
eval_path = data_path / "evaluation"


runs = [
    RunConfig("sm_6x3_6x3_256_noboth_unique", "Flash"),
    RunConfig("nosm_6x3_6x3_256_noboth_unique", "Flash (no SM)"),
    RunConfig("nosm_12x3_36x3_256_noboth", "Deep"),
    RunConfig("moe_sm_2x3_6x3_6x3_256_cap_3.5e-4", "Flex"),
    RunConfig("moe_nosm_2x3_12x3_12x3_256_cap_3.5e-4", "Wide"),
    RunConfig("moe_nosm_2x3_6x3_24x3_256_cap_3e-4_nosteps_v2", "Explorer XL"),
]
if __name__ == "__main__":
    train_fig_tokens = plot_training_curves(train_path, runs, x_axis="processed_tokens")
    train_fig_tokens.show()

    train_fig_epoch = plot_training_curves(train_path, runs, x_axis="epoch")
    train_fig_epoch.show()

    train_fig_step = plot_training_curves(train_path, runs, x_axis="step")
    train_fig_step.show()

    lr_fig = plot_learning_rates(train_path, runs)
    lr_fig.show()
