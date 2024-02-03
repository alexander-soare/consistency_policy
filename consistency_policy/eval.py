"""
Usage:
python -m consistency_policy.eval --checkpoint PATH/TO/CKPT
"""

import sys

# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)
import os
import pathlib
import click
import hydra
import torch
import dill
import wandb
import json
from diffusion_policy.workspace.base_workspace import BaseWorkspace


def eval(
    checkpoint,
    output_dir,
    device,
    eval_step_size,
    teacher_for_eval,
    n_train,
    n_train_vis,
    n_test,
    n_test_vis,
    test_start_seed,
    force: bool = False,
):
    if os.path.exists(output_dir) and not force:
        click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    # load checkpoint
    payload = torch.load(open(checkpoint, "rb"), pickle_module=dill)
    cfg = payload["cfg"]
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    # get policy from workspace
    policy = workspace.model
    policy.eval_step_size = eval_step_size
    policy.teacher_for_eval = teacher_for_eval

    device = torch.device(device)
    policy.to(device)
    policy.eval()

    # run eval
    cfg.task.env_runner.n_train = n_train
    cfg.task.env_runner.n_train_vis = n_train_vis
    cfg.task.env_runner.n_test = n_test
    cfg.task.env_runner.n_test_vis = n_test_vis
    cfg.task.env_runner.test_start_seed = test_start_seed
    env_runner = hydra.utils.instantiate(cfg.task.env_runner, output_dir=output_dir)
    runner_log = env_runner.run(policy)

    # dump log to json
    json_log = dict()
    for key, value in runner_log.items():
        if isinstance(value, wandb.sdk.data_types.video.Video):
            json_log[key] = value._path
        else:
            json_log[key] = value
    out_path = os.path.join(output_dir, "eval_log.json")
    json.dump(json_log, open(out_path, "w"), indent=2, sort_keys=True)
    print("test/mean_score:", json_log["test/mean_score"])
    return cfg, json_log


@click.command()
@click.option("-c", "--checkpoint", required=True)
@click.option("-o", "--output-dir", required=True)
@click.option("-d", "--device", default="cuda:0")
@click.option("-s", "--eval-step-size", type=int, default=1)
@click.option("-t", "--teacher-for-eval", is_flag=True, default=False)
@click.option("--n-train", type=int, default=0)
@click.option("--n-train-vis", type=int, default=0)
@click.option("--n-test", type=int, default=50)
@click.option("--n-test-vis", type=int, default=0)
@click.option("--test-start-seed", type=int, default=100000)
def main(
    checkpoint,
    output_dir,
    device,
    eval_step_size,
    teacher_for_eval,
    n_train,
    n_train_vis,
    n_test,
    n_test_vis,
    test_start_seed,
):
    eval(
        checkpoint=checkpoint,
        output_dir=output_dir,
        device=device,
        eval_step_size=eval_step_size,
        teacher_for_eval=teacher_for_eval,
        n_train=n_train,
        n_train_vis=n_train_vis,
        n_test=n_test,
        n_test_vis=n_test_vis,
        test_start_seed=test_start_seed,
    )


if __name__ == "__main__":
    main()
