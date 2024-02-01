"""
Usage:
Training:
python -m consistency_policy.train --config-name=train_diffusion_lowdim_workspace
"""
from pathlib import Path
import sys

import hydra
from omegaconf import OmegaConf

from diffusion_policy.workspace.base_workspace import BaseWorkspace

# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)


@hydra.main(
    version_base=None, config_path=str(Path(__file__).parent.joinpath("train_configs"))
)
def main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    OmegaConf.resolve(cfg)
    print(OmegaConf.to_yaml(cfg))

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
