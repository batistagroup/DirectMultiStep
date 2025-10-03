from pathlib import Path
from directmultistep import generate_routes, generate_routes_batched

def run_beam1():
    target = "CNCc1ccccc1"
    starting_material = "CN"
    n_steps = 1

    routes = generate_routes(
        target=target,
        n_steps=n_steps,
        starting_material=starting_material,
        beam_size=5,
        model="flash",
        config_path=Path("data/configs/dms_dictionary.yaml"),
        ckpt_dir=Path("data/checkpoints"),
    )

    with open('b-1-routes.txt', 'w') as f:
        for route in routes[:3]:
            f.write(route + '\n')

def run_beam2():
    targets = ["CNCc1ccccc1"]*2
    starting_materials = ["CN"]*2
    n_steps_list = [1]*2

    routes = generate_routes_batched(
        targets=targets,
        n_steps_list=n_steps_list,
        starting_materials=starting_materials,
        beam_size=5,
        model="flash",
        config_path=Path("data/configs/dms_dictionary.yaml"),
        ckpt_dir=Path("data/checkpoints"),
    )

    with open('b-2-routes.txt', 'w') as f:
        for i, (target, routes_for_target) in enumerate(zip(targets, routes)):
            f.write(f"Target {i+1}: {target}\n")
            f.write(f"Routes: {len(routes_for_target)}\n")
            for route in routes_for_target[:3]:
                f.write(route + '\n')

if __name__ == "__main__":
    run_beam1()
    run_beam2()
