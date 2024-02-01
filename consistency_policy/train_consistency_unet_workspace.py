"""
This is a mashup of train_diffusion_unet_hybrid_workspace and train_diffusion_unet_lowdim_workspace.

"""

from copy import deepcopy
from inspect import signature
from enum import Enum
import math
import os.path as osp
import random
from typing import Tuple

from deepdiff import DeepDiff
from deepdiff.operator import BaseOperator
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import dill
import hydra
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.models.base_nets as rmbn
import robomimic.utils.obs_utils as ObsUtils
from timm.utils import freeze
import torch
from torch import nn
from torch.utils.data import DataLoader
import tqdm
import wandb

from consistency_policy.consistency_model import ConsistencyUnetPolicy, PolicyType
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to, replace_submodules
from diffusion_policy.common.robomimic_config_util import get_robomimic_config
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
import diffusion_policy.model.vision.crop_randomizer as dmvc
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.workspace.train_diffusion_unet_lowdim_workspace import TrainDiffusionUnetLowdimWorkspace


matplotlib.use("Agg")


class TaskName(Enum):
    PUSHT = "pusht"
    TRANSPORT = "transport"
    TOOLHANG = "tool_hang"


class TensorOperator(BaseOperator):
    """Custom operator for comparing Tensors with DeepDiff."""

    def give_up_diffing(self, level, diff_instance):
        if torch.equal(level.t1, level.t2):
            return True


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


class TrainConsistencyUnetWorkspace(BaseWorkspace):
    include_keys = ["global_step", "epoch"]

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        set_seed(cfg.training.seed)

        policy_type = PolicyType(cfg.policy.policy_type)
        self.policy_type = policy_type
        task_name = TaskName(cfg.task.task_name)
        self.task_name = task_name

        # Raise assertions for things I haven't implemented.
        if policy_type == PolicyType.LOWDIM:
            assert not cfg.policy.pred_action_steps_only
            assert not cfg.policy.obs_as_local_cond
        elif policy_type == PolicyType.HYBRID:
            assert cfg.policy.obs_as_global_cond
        else:
            raise RuntimeError("policy_type not implemented")

        # Configure model.
        if policy_type == PolicyType.HYBRID:
            shape_meta = cfg.policy.shape_meta
            crop_shape = cfg.policy.crop_shape
            action_shape = shape_meta["action"]["shape"]
            assert len(action_shape) == 1
            action_dim = action_shape[0]
            obs_shape_meta = shape_meta["obs"]
            obs_config = {"low_dim": [], "rgb": [], "depth": [], "scan": []}
            obs_key_shapes = dict()
            for key, attr in obs_shape_meta.items():
                shape = attr["shape"]
                obs_key_shapes[key] = list(shape)
                type = attr.get("type", "low_dim")
                if type == "rgb":
                    obs_config["rgb"].append(key)
                elif type == "low_dim":
                    obs_config["low_dim"].append(key)
                else:
                    raise RuntimeError(f"Unsupported obs type: {type}")

            config = get_robomimic_config(algo_name="bc_rnn", hdf5_type="image", task_name="square", dataset_type="ph")

            with config.unlocked():
                # set config with shape_meta
                config.observation.modalities.obs = obs_config

                if crop_shape is None:
                    for key, modality in config.observation.encoder.items():
                        if modality.obs_randomizer_class == "CropRandomizer":
                            modality["obs_randomizer_class"] = None
                else:
                    # set random crop parameter
                    ch, cw = crop_shape
                    for key, modality in config.observation.encoder.items():
                        if modality.obs_randomizer_class == "CropRandomizer":
                            modality.obs_randomizer_kwargs.crop_height = ch
                            modality.obs_randomizer_kwargs.crop_width = cw

            # init global state
            ObsUtils.initialize_obs_utils_with_config(config)

            # Load model.
            # This was a weird bug where the order of the keys mattered. Not swapping the order meant that even with
            # loading the state_dict of the teacher model, the encoder would not work properly.
            if task_name == TaskName.PUSHT:
                assert list(obs_key_shapes.keys()) == ["agent_pos", "image"]
                obs_key_shapes = {"image": obs_key_shapes["image"], "agent_pos": obs_key_shapes["agent_pos"]}
            elif task_name == TaskName.TRANSPORT:
                assert list(obs_key_shapes.keys()) == [
                    "robot0_eef_pos",
                    "robot0_eef_quat",
                    "robot0_eye_in_hand_image",
                    "robot0_gripper_qpos",
                    "robot1_eef_pos",
                    "robot1_eef_quat",
                    "robot1_eye_in_hand_image",
                    "robot1_gripper_qpos",
                    "shouldercamera0_image",
                    "shouldercamera1_image",
                ]
                correct_order = [
                    "shouldercamera0_image",
                    "robot0_eye_in_hand_image",
                    "robot0_eef_pos",
                    "robot0_eef_quat",
                    "robot0_gripper_qpos",
                    "shouldercamera1_image",
                    "robot1_eye_in_hand_image",
                    "robot1_eef_pos",
                    "robot1_eef_quat",
                    "robot1_gripper_qpos",
                ]
                obs_key_shapes = {k: obs_key_shapes[k] for k in correct_order}
            elif task_name == TaskName.TOOLHANG:
                assert list(obs_key_shapes.keys()) == [
                    "robot0_eef_pos",
                    "robot0_eef_quat",
                    "robot0_eye_in_hand_image",
                    "robot0_gripper_qpos",
                    "sideview_image",
                ]
                correct_order = [
                    "sideview_image",
                    "robot0_eye_in_hand_image",
                    "robot0_eef_pos",
                    "robot0_eef_quat",
                    "robot0_gripper_qpos",
                ]
                obs_key_shapes = {k: obs_key_shapes[k] for k in correct_order}
            policy: PolicyAlgo = algo_factory(
                algo_name=config.algo_name,
                config=config,
                obs_key_shapes=obs_key_shapes,
                ac_dim=action_dim,
                device="cpu",
            )

            obs_encoder = policy.nets["policy"].nets["encoder"].nets["obs"]

            if cfg.policy.obs_encoder_group_norm:
                # Replace batch norm with group norm.
                replace_submodules(
                    root_module=obs_encoder,
                    predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                    func=lambda x: nn.GroupNorm(num_groups=x.num_features // 16, num_channels=x.num_features),
                )

            if cfg.policy.eval_fixed_crop:
                replace_submodules(
                    root_module=obs_encoder,
                    predicate=lambda x: isinstance(x, rmbn.CropRandomizer),
                    func=lambda x: dmvc.CropRandomizer(
                        input_shape=x.input_shape,
                        crop_height=x.crop_height,
                        crop_width=x.crop_width,
                        num_crops=x.num_crops,
                        pos_enc=x.pos_enc,
                    ),
                )

            unet_kwargs = {k: v for k, v in cfg.policy.items() if k in signature(ConditionalUnet1D).parameters.keys()}
            unet_kwargs["input_dim"] = action_dim + (
                0 if cfg.policy.obs_as_global_cond else obs_encoder.output_shape()[0]
            )
            unet_kwargs["local_cond_dim"] = None
            unet_kwargs["global_cond_dim"] = (
                obs_encoder.output_shape()[0] * cfg.policy.n_obs_steps if cfg.policy.obs_as_global_cond else None
            )
            net = ConditionalUnet1D(**unet_kwargs)
        elif policy_type == PolicyType.LOWDIM:
            obs_encoder = None
            net = hydra.utils.instantiate(cfg.policy.model)
        else:
            raise RuntimeError("policy_type not implemented.")

        # Configure teacher network (or if do_distill=False this ends being used only to initialize the consistency
        # model weights).
        teacher_net = None
        if cfg.training.teacher_checkpoint is not None:
            teacher_payload = torch.load(open(cfg.training.teacher_checkpoint, "rb"), pickle_module=dill)
            teacher_cfg = teacher_payload["cfg"]
            teacer_cls = hydra.utils.get_class(teacher_cfg._target_)
            teacher_workspace = teacer_cls(teacher_cfg)
            teacher_workspace.load_payload(teacher_payload, exclude_keys=None, include_keys=None)
            # Copy the teacher directly from the loaded workspace as we don't want to include dropout.
            teacher_net = deepcopy(teacher_workspace.ema_model.model)
            net.load_state_dict(teacher_net.state_dict())

            # Double check that net and teacher_net are really the same (apart from dropout).
            diff = DeepDiff(
                net, teacher_net, custom_operators=[TensorOperator(types=(torch.Tensor, torch._C._TensorMeta))]
            )
            assert diff == {} or all("dropout" in k for k in diff["values_changed"].keys())

            freeze(teacher_net)
            if policy_type == PolicyType.HYBRID:
                obs_encoder.load_state_dict(teacher_workspace.ema_model.obs_encoder.state_dict())

                # Double check that this is the same as cloning directly from the workspace.
                obs_encoder_ = deepcopy(teacher_workspace.ema_model.obs_encoder)
                diff = DeepDiff(
                    obs_encoder,
                    obs_encoder_,
                    custom_operators=[TensorOperator(types=(torch.Tensor, torch._C._TensorMeta))],
                )
                assert diff == {}
                del obs_encoder_

                freeze(obs_encoder)
            del teacher_payload
            del teacher_workspace
        ema_net = deepcopy(net)

        freeze(ema_net)  # freeze only applies to SGD, so the EMA model can still be updated
        policy_kwargs = {k: v for k, v in cfg.policy.items() if k in signature(ConsistencyUnetPolicy).parameters.keys()}
        policy_kwargs.update(
            {
                "net": net,
                "ema_net": ema_net,
                "obs_encoder": obs_encoder,
                "teacher_net": teacher_net if cfg.training.do_distill else None,
                "noise_scheduler": hydra.utils.instantiate(cfg.policy.noise_scheduler),
                "timestep_sampler": hydra.utils.instantiate(cfg.training.timestep_sampler),
            }
        )
        if policy_type == PolicyType.HYBRID:
            policy_kwargs["action_dim"] = action_dim
            policy_kwargs["obs_dim"] = 0 if cfg.policy.obs_as_global_cond else obs_encoder.output_shape()[0]
        self.model = ConsistencyUnetPolicy(**policy_kwargs)
        if self.output_dir is not None:
            fig = self.model.plot_schedule()
            plt.savefig(osp.join(self.output_dir, "schedule_params.png"))
            plt.close(fig)

        # Configure training state
        self.optimizer = hydra.utils.instantiate(cfg.optimizer, params=self.model.parameters())

        # TODO
        # self.optimizer.load_state_dict(payload["state_dicts"]["optimizer"])

        self.global_step = 0
        self.epoch = 0

    def calc_warmup_params(self, step: int, total_steps: int) -> Tuple[int, float]:
        """Calculate warmup parameters for consistency training as presented in CM paper.

        One difference is that I make the schedule finish a quarter of the way through the training so that there's
        plenty of training time with all the diffusion timesteps.
        """
        timesteps = self.cfg.policy.noise_scheduler.num_train_timesteps
        # Goes from 1 at the beginning of training to timesteps a quarter of the way through training.
        N = min(
            timesteps,
            int(math.ceil(math.sqrt(step / (total_steps // 4) * ((timesteps + 1) ** 2 - 2**2) + 2**2) - 1)),
        )
        ema_rate = np.exp(np.log(self.cfg.ema_rate) * 2 / (N + 1))
        return N, ema_rate

    def run(self):
        cfg = deepcopy(self.cfg)

        # Maybe resume training.
        if cfg.training.resume:
            lask_ckpt_path = self.get_checkpoint_path()
            if lask_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lask_ckpt_path}")
                self.load_checkpoint(path=lask_ckpt_path)

        # Configure training dataset.
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)

        # Configure validation dataset.
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        # Set normalizer.
        normalizer = dataset.get_normalizer()
        self.model.set_normalizer(normalizer)

        # Configure LR scheduler.
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(len(train_dataloader) * cfg.training.num_epochs)
            // cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step - 1,
        )

        # Configure env runner.
        env_runner = hydra.utils.instantiate(cfg.task.env_runner, output_dir=self.output_dir)

        # Configure logging.
        if cfg.use_wandb:
            wandb_run = wandb.init(
                dir=str(self.output_dir),
                config=OmegaConf.to_container(cfg, resolve=True),
                **cfg.logging,
            )
            wandb.config.update({"output_dir": self.output_dir})

        # Configure checkpointing.
        topk_manager = TopKCheckpointManager(save_dir=osp.join(self.output_dir, "checkpoints"), **cfg.checkpoint.topk)

        # Move everything to the training device.
        device = torch.device(cfg.training.device)
        self.model.to(device)
        optimizer_to(self.optimizer, device)

        # Keep one training batch for sampling.
        train_sampling_batch = None

        # Debug settings.
        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        total_steps = cfg.training.num_epochs * len(train_dataloader)
        log_path = osp.join(self.output_dir, "logs.json.txt")
        with JsonLogger(log_path) as json_logger:
            for _ in range(cfg.training.num_epochs):
                step_log = dict()
                # ========= train for this epoch ==========
                self.model.train()
                train_losses = list()
                with tqdm.tqdm(
                    train_dataloader,
                    desc=f"Training epoch {self.epoch}",
                    leave=False,
                    mininterval=cfg.training.tqdm_interval_sec,
                ) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        # Device transfer
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        if train_sampling_batch is None:
                            train_sampling_batch = batch

                        # Maybe calculate warmup parameters.
                        if cfg.training.do_consistency_training_warmup:
                            N, ema_rate = self.calc_warmup_params(self.global_step, total_steps)
                        else:
                            N = cfg.policy.noise_scheduler.num_train_timesteps
                            ema_rate = self.cfg.training.ema_rate

                        # Compute loss
                        raw_loss, loss_info = self.model.compute_loss(batch, max_timestep=N)
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        loss.backward()

                        # Step optimizer
                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()

                        # Update ema
                        self.model.step_ema(ema_rate)

                        # Logging
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            "train_loss": raw_loss_cpu,
                            **loss_info,
                            "global_step": self.global_step,
                            "epoch": self.epoch,
                            "lr": lr_scheduler.get_last_lr()[0],
                            "N": N,
                            "ema_rate": ema_rate,
                        }

                        is_last_batch = batch_idx == (len(train_dataloader) - 1)
                        if cfg.use_wandb:
                            wandb_run.log(step_log, commit=False, step=self.global_step)
                        if not is_last_batch:
                            # Log of last step is combined with validation and rollout
                            json_logger.log(step_log)
                            self.global_step += 1

                        if (cfg.training.max_train_steps is not None) and batch_idx >= (
                            cfg.training.max_train_steps - 1
                        ):
                            break

                # At the end of each epoch, replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log["train_loss"] = train_loss

                # # ========= eval for this epoch ==========
                self.model.eval()

                # Run rollout
                if (self.epoch % cfg.training.rollout_every) == 0:
                    runner_log = env_runner.run(self.model)
                    # log all
                    step_log.update(runner_log)

                # Run validation
                if (self.epoch % cfg.training.val_every) == 0:
                    with torch.no_grad():
                        val_losses = list()
                        with tqdm.tqdm(
                            val_dataloader,
                            desc=f"Validation epoch {self.epoch}",
                            leave=False,
                            mininterval=cfg.training.tqdm_interval_sec,
                        ) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                loss, _ = self.model.compute_loss(batch)
                                val_losses.append(loss)
                                if (cfg.training.max_val_steps is not None) and batch_idx >= (
                                    cfg.training.max_val_steps - 1
                                ):
                                    break
                        if len(val_losses) > 0:
                            val_loss = torch.mean(torch.tensor(val_losses)).item()
                            # log epoch average validation loss
                            step_log["val_loss"] = val_loss

                # Run diffusion sampling on a training batch
                if (self.epoch % cfg.training.sample_every) == 0:
                    with torch.no_grad():
                        # sample trajectory from training set, and evaluate difference
                        if self.policy_type == PolicyType.LOWDIM:
                            batch = train_sampling_batch
                            obs_dict = {"obs": batch["obs"]}
                        elif self.policy_type == PolicyType.HYBRID:
                            batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                            obs_dict = batch["obs"]
                        else:
                            raise RuntimeError("policy_type not implemented")

                        gt_action = batch["action"]

                        result = self.model.predict_action(obs_dict)
                        if self.policy_type == PolicyType.LOWDIM:
                            if cfg.pred_action_steps_only:
                                pred_action = result["action"]
                                start = cfg.n_obs_steps - 1
                                end = start + cfg.n_action_steps
                                gt_action = gt_action[:, start:end]
                            else:
                                pred_action = result["action_pred"]
                        elif self.policy_type == PolicyType.HYBRID:
                            pred_action = result["action_pred"]
                        else:
                            raise RuntimeError("policy_type not implemented")
                        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                        # log
                        step_log["train_action_mse_error"] = mse.item()
                        # release RAM
                        del batch
                        del obs_dict
                        del gt_action
                        del result
                        del pred_action
                        del mse

                # Checkpoint
                if (self.epoch % cfg.training.checkpoint_every) == 0:
                    # checkpointing
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    # sanitize metric names
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace("/", "_")
                        metric_dict[new_key] = value

                    # We can't copy the last checkpoint here
                    # since save_checkpoint uses threads.
                    # therefore at this point the file might have been empty!
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)
                # ========= eval end for this epoch ==========

                # end of epoch
                # log of last step is combined with validation and rollout
                if cfg.use_wandb:
                    wandb_run.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1
