import json

import click
import matplotlib.pyplot as plt

from consistency_policy.eval_sweep import plot_results

plt.rcParams.update({'font.size': 14})


@click.command()
@click.option("-j", "--json-results", required=True)
@click.option("-t", "--title", default="")
@click.option("-o", "--output-file", default="")
def main(json_results, title, output_file):
    with open(json_results, "r") as f:
        results_dct = json.load(f)
    fig = plot_results(results_dct, title)
    if len(output_file) > 0:
        fig.savefig(output_file)


if __name__ == "__main__":
    main()
