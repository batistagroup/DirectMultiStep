from pathlib import Path

from directmultistep import generate_routes, generate_routes_batched


def run_beam1() -> None:
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

    with open("b-1-routes.txt", "w") as f:
        for route in routes[:3]:
            f.write(route + "\n")


def run_beam2() -> None:
    targets = ["CNCc1ccccc1"] * 2
    starting_materials = ["CN"] * 2
    n_steps_list = [1] * 2

    routes = generate_routes_batched(
        targets=targets,
        n_steps_list=n_steps_list,
        starting_materials=starting_materials,
        beam_size=5,
        model="flash",
        config_path=Path("data/configs/dms_dictionary.yaml"),
        ckpt_dir=Path("data/checkpoints"),
    )

    with open("b-2-routes.txt", "w") as f:
        for i, (target, routes_for_target) in enumerate(zip(targets, routes, strict=False)):
            f.write(f"Target {i + 1}: {target}\n")
            f.write(f"Routes: {len(routes_for_target)}\n")
            for route in routes_for_target[:3]:
                f.write(route + "\n")


def run_beam_hard() -> None:
    targets_list = [
        "CC(=O)OC1=CC=CC=C1C(=O)O",
        "CNCc1cc(-c2ccccc2F)n(S(=O)(=O)c2cccnc2)c1",
        "O=C(c1ccc(NS(=O)(=O)c2cccc3cccnc23)cc1)N1CCN(CC2CC2)CC1",
        "COc1ccc(-n2nccn2)c(C(=O)N2CCC[C@@]2(C)c2nc3c(C)c(Cl)ccc3[nH]2)c1",
    ]
    sms_list = [None, "O=S(=O)(Cl)c1cccnc1", "CCOC(=O)c1ccc(N)cc1", "C[C@@]1(C(=O)O)CCCN1"]
    n_steps_list = [1, 2, 5, 4]

    routes = generate_routes_batched(
        targets=targets_list,
        n_steps_list=n_steps_list,
        starting_materials=sms_list,
        beam_size=5,
        model="flash",
        config_path=Path("data/configs/dms_dictionary.yaml"),
        ckpt_dir=Path("data/checkpoints"),
    )
    from tqdm import tqdm

    old_r_coll = []

    for target, sm, n_steps in tqdm(zip(targets_list, sms_list, n_steps_list, strict=False), total=len(targets_list)):
        old_rs = generate_routes(
            target=target,
            n_steps=n_steps,
            starting_material=sm,
            beam_size=5,
            model="flash",
            config_path=Path("data/configs/dms_dictionary.yaml"),
            ckpt_dir=Path("data/checkpoints"),
            show_progress=False,
        )
        old_r_coll.append(old_rs)

    with open("b-hard-routes.txt", "w") as f:
        for i, (target, routes_for_target) in enumerate(zip(targets_list, routes, strict=False)):
            f.write(f"Target {i + 1}: {target}\n")
            f.write(f"Routes: {len(routes_for_target)}\n")
            for route in routes_for_target[:3]:
                f.write(route + "\n")
    with open("b1-routes.txt", "w") as f:
        for i, (target, routes_for_target) in enumerate(zip(targets_list, old_r_coll, strict=False)):
            f.write(f"Target {i + 1}: {target}\n")
            f.write(f"Routes: {len(routes_for_target)}\n")
            for route in routes_for_target[:3]:
                f.write(route + "\n")

    routes = generate_routes_batched(
        targets=targets_list * 16,
        n_steps_list=n_steps_list * 16,
        starting_materials=sms_list * 16,
        beam_size=5,
        model="flash",
        config_path=Path("data/configs/dms_dictionary.yaml"),
        ckpt_dir=Path("data/checkpoints"),
    )


if __name__ == "__main__":
    run_beam1()
    run_beam2()
    run_beam_hard()
