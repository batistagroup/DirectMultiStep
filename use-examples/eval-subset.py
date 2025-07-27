from pathlib import Path

from directmultistep.generation.eval import EvalConfig, ModelEvaluator
from directmultistep.model import ModelFactory
from directmultistep.training import TrainingConfig

__mode__ = "local"
assert __mode__ in ["local", "cluster"]

if __mode__ == "local" or __mode__ in ["cluster"]:
    base_path = Path(__file__).resolve().parent.parent

data_path = base_path / "data"

run_name = "van_sm_6x3_6x3_256_noboth_seed=42"
logbook_path = data_path / "configs" / "logbook" / run_name
train_conf = TrainingConfig.load(logbook_path / "training_config.yaml")
factory = ModelFactory.from_config_file(logbook_path / "model_config.yaml", compile_model=False)

ec = EvalConfig(
    data_path=data_path,
    run_name=run_name,
    eval_dataset="n1_50",
    epoch=46,
    use_sm=True,
    use_steps=True,
    beam_width=50,
    enc_active_experts=None,
    dec_active_experts=None,
)
ec.save(logbook_path / f"{ec.eval_name}_config.yaml")


if __name__ == "__main__":
    factory.check_for_eval_config_updates(ec)
    model = factory.create_model()
    device = ModelFactory.determine_device()
    # model = factory.load_lightning_checkpoint(model, ec.checkpoint_path, device=device)
    pblshd = data_path / "checkpoints" / "flash_ep=46.ckpt"
    model = factory.load_checkpoint(model, pblshd, device=device)

    evalObj = ModelEvaluator(model, ec, train_conf, device=device)

    evalObj.load_eval_dataset()
    evalObj.prepare_beam_search()

    all_beam_results_NS2 = evalObj.run_beam_search()

    top_ks = evalObj.calculate_top_k_accuracy()
    print(top_ks)

    # for pharma
    name_to_rank = evalObj.prepare_name_to_rank()
    print(name_to_rank)
