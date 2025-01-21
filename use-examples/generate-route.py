from pathlib import Path

from directmultistep.generate import generate_routes
from directmultistep.utils.web_visualize import draw_tree_from_path_string

data_path = Path(__file__).resolve().parents[1] / "data"
ckpt_path = data_path / "checkpoints"
fig_path = data_path / "figures"
config_path = data_path / "configs" / "dms_dictionary.yaml"


def visualize_routes(path_strings: list[str], theme: str = "light") -> list[str]:
    """Visualize synthesis routes and return SVG strings."""
    import random

    request_id = "rnd_" + str(random.randint(0, 1000000))
    save_folder = fig_path / request_id
    save_folder.mkdir(parents=True, exist_ok=True)

    svg_results = []
    for i, path_string in enumerate(path_strings):
        svg_tree = draw_tree_from_path_string(
            path_string=path_string,
            save_path=save_folder / f"result_{i}",
            width=600,
            height=600,
            x_margin=40,
            y_margin=120,
            theme=theme,
        )
        svg_results.append(svg_tree)

    return svg_results


if __name__ == "__main__":
    # Example usage
    target = "CNCc1cc(-c2ccccc2F)n(S(=O)(=O)c2cccnc2)c1"
    sm = "CN"

    # Find routes with starting material using flash model
    paths = generate_routes(
        target, n_steps=2, starting_material=sm, model="flash", beam_size=5, config_path=config_path, ckpt_dir=ckpt_path
    )
    # paths = generate_routes(target, n_steps=2, starting_material=sm, model="flash-20M", beam_size=5)
    # paths = generate_routes(target, n_steps=2, starting_material=sm, model="flex-20M", beam_size=5)

    # # Find routes without starting material using deep model
    # paths = generate_routes(target, n_steps=2, model="deep")
    # paths = generate_routes(target, n_steps=2, model="wide", beam_size=20)

    # # Find routes using explorer model (automatically determines steps)
    # paths = generate_routes(target, starting_material=sm, model="explorer", beam_size=5)
    # paths = generate_routes(target, model="explorer XL", beam_size=5)

    svg_contents = visualize_routes(paths)
