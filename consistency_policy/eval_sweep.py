import click
import json
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

from consistency_policy.eval import eval

matplotlib.use("TkAgg")


def plot_results(results_dct: dict, title: str = ""):
    fig, ax = plt.subplots(1, 1)
    for k in ["ddim", "cm"]:
        num_steps = np.array(
            [
                int(math.ceil(results_dct["num_train_timesteps"] / int(s.replace("step_size_", ""))))
                for s in results_dct[k]
            ]
        )
        if "beta_mean" in results_dct[k][next(iter(results_dct[k].keys()))]:
            means = np.array([results_dct[k][s]["beta_mean"] for s in results_dct[k]])
            lowers = np.array([results_dct[k][s]["beta_lower"] for s in results_dct[k]])
            uppers = np.array([results_dct[k][s]["beta_upper"] for s in results_dct[k]])
        else:
            means = np.array([results_dct[k][s]["mean"] for s in results_dct[k]])
            stds = np.array([results_dct[k][s]["std"] for s in results_dct[k]])
            lowers = means - stds
            uppers = means + stds
        uppers = np.clip(uppers, -1, 1)
        lowers = np.clip(lowers, -1, 1)
        rev_argsort = np.argsort(num_steps)[::-1]
        num_steps = num_steps[rev_argsort]
        means = means[rev_argsort]
        lowers = lowers[rev_argsort]
        uppers = uppers[rev_argsort]
        # This loop makes sure only consecutive steps have lines between them.
        ix = 0
        while ix < len(num_steps) - 1:
            indices = [ix]
            while True:
                ix += 1
                if ix == len(num_steps):
                    break
                if num_steps[ix] + 1 == num_steps[indices[-1]]:
                    indices.append(ix)
                else:
                    break
            indices = np.array(indices)
            ax.errorbar(
                num_steps[indices].astype(str),
                means[indices],
                np.row_stack([means[indices] - lowers[indices], uppers[indices] - means[indices]]),
                linestyle="-",
                marker="o" if k == "ddim" else "x",
                markersize=10,
                capsize=2,
                elinewidth=0.5,
                label=("Diffusion Policy" if k == "ddim" else "Consistency Policy") if 0 in indices else None,
                color="lightcoral" if k == "ddim" else "forestgreen",
            )
            ix = indices[-1] + 1
    ax.grid()
    ax.set_xlabel("# inference steps")
    ax.set_ylabel("Mean score")
    ax.legend()
    ax.set_title(title)
    plt.show()
    return fig


@click.command()
@click.option("-j", "--results-json", required=True)
@click.option("-c", "--checkpoint", required=True)
@click.option("-o", "--output-dir", required=True)
@click.option("-d", "--device", default="cuda:0")
@click.option("--n-test", type=int, default=50)
@click.option("--n-chunks", type=int, default=1)
@click.option("--test-start-seed", type=int, default=100000)
@click.option("--eval-step-sizes", type=str, required=True, help="pass as a comma seperated list of numbers")
def eval_sweep(results_json, checkpoint, output_dir, device, n_test, n_chunks, test_start_seed, eval_step_sizes):
    results_dct = {"checkpoint": checkpoint, "n_test": n_test, "test_start_seed": test_start_seed, "ddim": {}, "cm": {}}
    for teacher_for_eval in [True, False]:
        for eval_step_size in [int(s) for s in eval_step_sizes.split(",")]:
            assert n_test % n_chunks == 0
            chunk_size = n_test // n_chunks
            scores = []
            for chunk_ix in range(n_chunks):
                cfg, json_log = eval(
                    eval_step_size=eval_step_size,
                    checkpoint=checkpoint,
                    output_dir=output_dir,
                    device=device,
                    teacher_for_eval=teacher_for_eval,
                    n_test=chunk_size,
                    test_start_seed=test_start_seed + chunk_ix * chunk_size,
                    n_train=0,
                    n_train_vis=0,
                    n_test_vis=0,
                    force=True,
                )
                scores += [v for k, v in json_log.items() if k.startswith("test/sim_max_reward_")]
            num_train_timesteps = cfg.policy.noise_scheduler.num_train_timesteps
            results_dct["num_train_timesteps"] = num_train_timesteps
            mean = np.mean(scores)
            std = np.std(scores)
            k1 = "ddim" if teacher_for_eval else "cm"
            k2 = f"step_size_{eval_step_size}"
            results_dct[k1][k2] = {
                "mean": mean,
                "std": std,
                "scores": scores,
            }
            scores = np.array(scores)
            if cfg.task.task_name != "pusht":
                # All the other task reward either 0 or 1 so use a beta distribution.
                confidence_interval = 0.682
                lower_percentile = (1 - confidence_interval) / 2
                upper_percentile = 1 - lower_percentile
                alpha = np.count_nonzero(scores == 1) + 1
                beta = np.count_nonzero(scores == 0) + 1
                results_dct[k1][k2]["beta_mean"] = scipy.stats.beta.mean(alpha, beta)
                results_dct[k1][k2]["beta_lower"] = scipy.stats.beta.ppf(lower_percentile, alpha, beta)
                results_dct[k1][k2]["beta_upper"] = scipy.stats.beta.ppf(upper_percentile, alpha, beta)
                results_dct[k1][k2]["beta_std"] = scipy.stats.beta.std(alpha, beta)
            with open(results_json, "w") as f:
                json.dump(results_dct, f, indent=2)


if __name__ == "__main__":
    eval_sweep()
