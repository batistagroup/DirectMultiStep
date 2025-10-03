"""
Example script demonstrating batched route generation.

This shows how to use the BatchedBeamSearch class to generate routes
for multiple targets simultaneously with different starting materials
and step counts.
"""

from pathlib import Path

from directmultistep.generate import generate_routes_batched

config_path = Path("data/configs/dms_dictionary.yaml")
ckpt_dir = Path("data/checkpoints")

targets = [
    "CNCc1ccccc1",
    "CCOc1ccccc1", 
    "c1ccccc1",
]

n_steps_list = [1, 2, 1]

starting_materials = [
    "CN",
    None,
    None,
]

routes = generate_routes_batched(
    targets=targets,
    n_steps_list=n_steps_list,
    starting_materials=starting_materials,
    beam_size=5,
    model="flash",
    config_path=config_path,
    ckpt_dir=ckpt_dir,
)

for i, (target, routes_for_target) in enumerate(zip(targets, routes)):
    print(f"\nTarget {i+1}: {target}")
    print(f"Starting material: {starting_materials[i]}")
    print(f"Number of steps: {n_steps_list[i]}")
    print(f"Generated {len(routes_for_target)} valid routes:")
    for j, route in enumerate(routes_for_target[:3], 1):
        print(f"  Route {j}: {route[:100]}..." if len(route) > 100 else f"  Route {j}: {route}")
