from enum import Enum
from typing import Optional, Dict, Union, Tuple

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import torch
from torch import nn
from torch import Tensor, LongTensor, BoolTensor

from consistency_policy.samplers import ScheduleSampler, StratifiedUniformSampler
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy


matplotlib.use('Agg')


def sample_n_uniform(batch_size: int, N: int, device: torch.device) -> LongTensor:
    """Sample uniformly in [1, N]."""
    return torch.randint(1, N + 1, size=(batch_size,), device=device)


def sample_n_stratified(batch_size: int, N: int, n_strata: int, device: torch.device) -> LongTensor:
    """Sample uniformly in `n_strata` intervals along [1, N].

    If n_strata does not divide batch_size evenly, sample the remainder uniformly over the entire interval [1, N].
    """
    n_per_strata, remainder = divmod(batch_size, n_strata)
    strata_bounds = np.linspace(1, N + 1, n_strata + 1)
    strata_bounds = np.column_stack([strata_bounds[:-1], strata_bounds[1:]])
    strata_samples: list[NDArray[np.int64]] = []
    for l, u in strata_bounds:
        strata_samples.append(np.floor(np.random.uniform(l, u, size=(n_per_strata,))))
    if remainder != 0:
        strata_samples.append(np.floor(np.random.uniform(1, N, size=(remainder,))))
    strata_samples = np.concatenate(strata_samples)
    return torch.from_numpy(strata_samples).long().to(device)


def update_ema_parameters(ema_net: nn.Module, net: nn.Module, alpha: float):
    """
    Logic nicked from diffusers EMAModel.
    """
    for ema_module, module in zip(ema_net.modules(), net.modules()):
        for (n_p_ema, p_ema), (n_p, p) in zip(
            ema_module.named_parameters(recurse=False), module.named_parameters(recurse=False)
        ):
            assert n_p_ema == n_p, "Parameter names don't match for EMA model update"
            if isinstance(p, dict):
                raise RuntimeError("Dict parameter not supported")
            if isinstance(module, nn.modules.batchnorm._BatchNorm) or not p.requires_grad:
                # Copy BatchNorm parameters, and non-trainable parameters directly.
                p_ema.copy_(p.to(dtype=p_ema.dtype).data)
            with torch.no_grad():
                p_ema.mul_(alpha)
                # Why p.data.to() instead of p.to().data as above? I don't know, but I won't mess with it.
                p_ema.add_(p.data.to(dtype=p_ema.dtype), alpha=1 - alpha)


class PolicyType(Enum):
    LOWDIM = "lowdim"
    IMAGE = "image"
    HYBRID = "hybrid"


class ConsistencyUnetPolicy(BaseLowdimPolicy):
    """Consistency training and distillation training as per CM paper.

    CM paper: https://arxiv.org/abs/2303.01469
    Also:
    LCM paper: https://arxiv.org/abs/2310.04378
    EDM paper: https://arxiv.org/abs/2206.00364
    """

    def __init__(
        self,
        policy_type: Union[str, PolicyType],
        net: ConditionalUnet1D,
        ema_net: ConditionalUnet1D,
        noise_scheduler: DDPMScheduler,
        timestep_sampler: ScheduleSampler,
        horizon: int,
        n_action_steps: int,
        n_obs_steps: int,
        obs_dim: int,
        action_dim: int,
        obs_as_global_cond: bool = False,
        eval_step_size: int = 1,
        skip_steps: int = 1,
        use_ema_target: bool = True,
        obs_encoder: Optional[MultiImageObsEncoder] = None,
        teacher_net: Optional[ConditionalUnet1D] = None,
        oa_step_convention: bool = False,
        **kwargs,  # ignore unnecessary arguments from diffusion model config
    ):
        """
        Args:
            policy_type:
            net: The underlying neural network for the consistency model.
            ema_net: The EMA version of `net`.
            noise_scheduler: See DDPMScheduler docs.
            timestep_sampler: Object that has a sample method for sampling timesteps and corresponding loss weights.
            horizon: Horizon (in time steps) used for generating trajectories.
            n_action_steps: The number of steps actually executed on the robot without replanning (may be less then the
                planning horizon).
            n_obs_steps: Number of observation steps for conditioning during inference.
            obs_dim: Observation feature dimension.
            action_dim: Action feature dimension.
            obs_as_global_cond: Whether to use the observation as global conditioning for the denoising network.
            eval_step_size: Step size for ODE solver. The last step is always whatever it takes to get to t=0.
            skip_steps: How many steps to "skip" as outlined in the "skipping time steps" technique of the LCM paper.
                Note that "skipping" 1 step means comparing step n with step n+1 (instead of n+2 which one might
                naturally infer from the word "skip").
            use_ema_target: Whether the target network should be the EMA model. iCM suggests this unnecessary for
                consistency distillation and detrimental for consistency training.
            obs_encoder: Encoder for the observations (if doing image policy).
            teacher_net: The teacher network to distill from. If provided, do consistency distillation instead of
                standard consistency training.
            oa_step_convention: If True, the (i)th action is applied based on the (i-1)th observation and results in the
                (i)th observation. If False, the (i)th action is applied based on the (i)th observation and results in
                the (i+1)th observation.

        """
        super().__init__()

        if not isinstance(policy_type, PolicyType):
            policy_type = PolicyType(policy_type)

        if policy_type == PolicyType.LOWDIM:
            assert obs_encoder is None
        elif policy_type == PolicyType.HYBRID:
            assert obs_encoder is not None
        else:
            raise ValueError("policy_type not implemented")
        self.policy_type = policy_type

        self.net = net
        self.ema_net = ema_net
        self.teacher_net = teacher_net
        assert skip_steps >= 1
        self.skip_steps = skip_steps
        self.do_distill = self.teacher_net is not None
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        assert n_action_steps <= self.horizon
        self.obs_as_global_cond = obs_as_global_cond
        self.eval_step_size = eval_step_size
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.oa_step_convention = oa_step_convention
        self.use_ema_target = use_ema_target
        self.obs_encoder = obs_encoder
        self.noise_scheduler = noise_scheduler
        self.timestep_sampler = timestep_sampler

        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False,
        )

        # alphas and sigmas following the formulation in LCM paper: q(xt|x0) = N(xt | alpha(t) * x0, sigma(t)**2 I)
        # Get alpha(t_n) with alpha[n].
        # Clamping the alphas keeps the errors from the last timestep from dominating the loss. The minimum alpha value
        # still needs to be small enough so that the final step of the forward diffusion is essentially not discernible
        # from pure gaussian noise.
        self.alphas = torch.clamp(torch.sqrt(noise_scheduler.alphas_cumprod), min=1 / 80)
        self.alphas = torch.concat([torch.tensor([1]), self.alphas])
        self.sigmas = torch.sqrt(1 - noise_scheduler.alphas_cumprod)
        self.sigmas = torch.concat([torch.tensor([0]), self.sigmas])
        self.N = len(self.alphas) - 1

        # Preconditioning.
        # Assume the data is normalized to unit variance.
        sigma_data = 1
        # Note: Since DDPM is variance preserving, cin actually comes out to all 1s.
        self.cin = torch.ones_like(self.alphas)
        # Impulse function (from LCM github).
        timestep = torch.arange(self.N + 1)
        self.cskip = sigma_data**2 / ((timestep / 0.1) ** 2 + sigma_data**2)
        self.cout = (timestep / 0.1) / ((timestep / 0.1) ** 2 + sigma_data**2) ** 0.5

        # Loss weighting attempts to minimize the dependence of the loss on the specific schedule parameters. To get the
        # loss weight comparing losses for denoising x_n and x_{n+skip_steps}, access loss_weights[n+skip_steps].
        self.loss_fn = nn.functional.mse_loss
        loss_weights = torch.zeros_like(self.alphas)
        alphas_sq = self.alphas[:-skip_steps] ** 2
        alphas_skip_sq = self.alphas[skip_steps:] ** 2
        loss_weights[skip_steps:] = (
            alphas_sq * alphas_skip_sq / (alphas_sq + alphas_skip_sq - 2 * alphas_sq * alphas_skip_sq)
        )
        alpha_sq = self.alphas[0] ** 2
        for k in range(1, skip_steps):
            alpha_skip_sq = self.alphas[k] ** 2
            loss_weights[k] = alpha_sq * alpha_skip_sq / (alpha_sq + alpha_skip_sq - 2 * alpha_sq * alpha_skip_sq)
        self.loss_weights = loss_weights

        # Debug. Uses the teacher for evaluation instead of the consistency model.
        self.teacher_for_eval = False

    def train(self, mode: bool = True):
        self.net.train(mode)
        if self.obs_encoder is not None:
            self.obs_encoder.train(mode)
        # Teacher net is always in eval mode.
        self.teacher_net.eval()

    def _schedule_to_device(self, device: str):
        self.alphas = self.alphas.to(device)
        self.sigmas = self.sigmas.to(device)
        self.cin = self.cin.to(device)
        self.cskip = self.cskip.to(device)
        self.cout = self.cout.to(device)
        self.loss_weights = self.loss_weights.to(device)

    def plot_schedule(self, windowed: bool = False) -> plt.Figure:
        fig, ax = plt.subplots(3, 1, figsize=(10, 10))
        n = np.arange(self.N + 1)
        ax[0].set_title("Forward-diffusion schedule")
        ax[0].plot(n, self.alphas.cpu().numpy(), label="ɑ")
        ax[0].plot(n, self.sigmas.cpu().numpy(), label="σ")
        ax[1].set_title("Preconditioning schedule")
        ax[1].plot(n, self.cin.cpu().numpy(), label="cin")
        ax[1].plot(n, self.cskip.cpu().numpy(), label="cskip")
        ax[1].plot(n, self.cout.cpu().numpy(), label="cout")
        ax[2].set_title("Loss weight")
        ax[2].plot(n, self.loss_weights.cpu().numpy())
        ax[2].set_yscale("log")
        for ax_ in ax:
            ax_.legend()
        plt.tight_layout()
        if windowed:
            plt.show()
        return fig

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def step_ema(self, ema_rate: float):
        update_ema_parameters(self.ema_net, self.net, ema_rate)

    def ddim_euler_step(
        self,
        x: Tensor,
        from_n: Union[int, LongTensor],
        to_n: Union[int, LongTensor],
        global_cond: Optional[Tensor] = None,
    ) -> Tensor:
        """Do an Euler step with DDIM from a given timestep to another timestep.

        Args:
            x: A batch of noisy trajectories.
            from_n: The timestep (or batch of timesteps) to go from.
            to_n: The timestep (or batch of timesteps) to go to.
            global_cond: Conditioning input.
        """
        self._schedule_to_device(x.device)
        # Validate from_n and to_n and massage them into the right data structures.
        assert type(from_n) == type(to_n)
        if isinstance(from_n, int):
            assert from_n > to_n
            from_n = torch.full(x.shape[:1], from_n, device=x.device)
            to_n = torch.full(x.shape[:1], to_n, device=x.device)
        elif not isinstance(from_n, Tensor):
            raise ValueError("from_n and to_n should be either int or Tensor")
        assert torch.all(from_n >= 1)
        assert torch.all(to_n >= 0)
        assert torch.all(from_n > to_n)
        from_n_view = from_n.view(-1, *([1] * (x.ndim - 1)))
        to_n_view = to_n.view(-1, *([1] * (x.ndim - 1)))

        # DP work uses n-1 to indicate denoising from the nth step of the forward-diffusion.
        pred = self.teacher_net(x, from_n - 1, global_cond=global_cond)

        if self.noise_scheduler.config.prediction_type == "sample":
            x0_pred = pred
            if self.noise_scheduler.config.clip_sample:
                x0_pred = x0_pred.clamp(
                    -self.noise_scheduler.config.clip_sample_range, self.noise_scheduler.config.clip_sample_range
                )
            eps_pred = (x - self.alphas[from_n_view] * x0_pred) / self.sigmas[from_n_view]
        elif self.noise_scheduler.config.prediction_type == "epsilon":
            eps_pred = pred
            x0_pred = (x - self.sigmas[from_n_view] * eps_pred) / self.alphas[from_n_view]
            if self.noise_scheduler.config.clip_sample:
                x0_pred = x0_pred.clamp(
                    -self.noise_scheduler.config.clip_sample_range, self.noise_scheduler.config.clip_sample_range
                )
                eps_pred = (x - self.alphas[from_n_view] * x0_pred) / self.sigmas[from_n_view]
        else:
            raise RuntimeError(f"prediction_type '{self.noise_scheduler.config.prediction_type}' not implemented")
        # DDPM reverse sigma. See eqn (12) in DDIM paper.
        # rev_sigma = torch.where(
        #     from_n_view == 1,
        #     self.sigmas[1] / self.alphas[1],
        #     self.sigmas[to_n_view]
        #     / self.sigmas[from_n_view]
        #     * torch.sqrt(1 - (self.alphas[from_n_view] / self.alphas[to_n_view]) ** 2)
        # )
        x = torch.where(
            to_n_view == 0,
            x0_pred,
            self.alphas[to_n_view] * x0_pred + self.sigmas[to_n_view] * eps_pred,
        )

        return x

    def evaluate_consistency_model(
        self, x: Tensor, n: LongTensor, global_cond: Optional[Tensor] = None, use_ema: bool = True
    ) -> Tensor:
        """
        Note: Only works for n >= 1. For n=0 the result would be the identity operation.
        """
        self._schedule_to_device(x.device)
        assert torch.all(n >= 1)
        n_view = n.view(-1, *([1] * (x.ndim - 1)))
        if use_ema:
            net = self.ema_net
        else:
            net = self.net
        # DP work uses n-1 to indicate denoising from the nth step of the forward-diffusion.
        pred = net(self.cin[n_view] * x, n - 1, global_cond=global_cond)
        if self.noise_scheduler.config.prediction_type == "epsilon":
            # Reparameterization trick: xt = alpha * x0 + sigma * epsilon -> x0 = (xt - sigma * epsilon) / alpha
            pred = (x - self.sigmas[n_view] * pred) / self.alphas[n_view]
        elif self.noise_scheduler.config.prediction_type != "sample":
            raise AssertionError("prediction_type not implemented")
        return self.cskip[n_view] * x + self.cout[n_view] * pred

    def generate_trajectory(
        self,
        step_size: int = 1,
        batch_size: int = 1,
        inpaint: Optional[Tuple[Tensor, BoolTensor]] = None,
        global_cond: Optional[Tensor] = None,
        use_ema: bool = True,
    ) -> Tensor:
        """Generate a trajectory following algorithm 4 in the CM paper.
        Args:
            inpaint: Two (batch, horizon, action_obs_dim) tensors. The first contains the reference values, and the
                second contains the inpainting mask where 1s indicate regions that need to be inpainted. The generated
                output should then be equal to the reference values wherever the inpainting mask has 0s.
            global_cond: Conditioning input.
        """
        tau = list(range(self.N, 0, -step_size)) + [0]
        shape = (batch_size, self.horizon, self.action_dim + (0 if self.obs_as_global_cond else self.obs_dim))
        x = torch.randn(shape, device=self.device) * self.sigmas[tau[0]]
        if inpaint is not None:
            assert (
                inpaint[0].shape == inpaint[1].shape == shape
            ), f"inpainting reference and mask should have shape {shape}"
            x[~inpaint[1]] = inpaint[0][~inpaint[1]]
        for i, n in enumerate(tau[:-1]):
            from_n = n
            to_n = tau[i + 1]  # i + 1 because tau is in reverse order.
            if self.teacher_for_eval:
                # Debug only.
                eps_pred = self.teacher_net(x, from_n - 1, global_cond=global_cond)
                x = (x - eps_pred * self.sigmas[from_n]) / self.alphas[from_n]
            else:
                x = self.evaluate_consistency_model(
                    x, torch.full(x.shape[:1], from_n, device=self.device), global_cond=global_cond, use_ema=use_ema
                )
            if to_n != 0:
                if self.teacher_for_eval:
                    # Debug only.
                    x = self.alphas[to_n] * x + self.sigmas[to_n] * eps_pred
                else:
                    x = self.alphas[to_n] * x + self.sigmas[to_n] * torch.randn_like(x)
            if inpaint is not None:
                x[~inpaint[1]] = inpaint[0][~inpaint[1]]
            if self.noise_scheduler.config.clip_sample:
                x = x.clamp(
                    -self.noise_scheduler.config.clip_sample_range, self.noise_scheduler.config.clip_sample_range
                )
        return x

    def predict_action(self, obs_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key

        TODO: Incorporate masking and conditioning. PushT lowdim doesn't need it.
        """
        # Prepare normalized observation data.
        if self.policy_type == PolicyType.LOWDIM:
            nobs = self.normalizer["obs"].normalize(obs_dict["obs"])
            batch_size = nobs.shape[0]
        elif self.policy_type == PolicyType.HYBRID:
            nobs = self.normalizer.normalize(obs_dict)
            batch_size = next(iter(nobs.values())).shape[0]
        else:
            raise RuntimeError("policy_type not implemented")

        global_cond = None
        inpaint = None
        if self.policy_type == PolicyType.LOWDIM:
            if self.obs_as_global_cond:
                global_cond = nobs[:, : self.n_obs_steps].reshape(nobs.shape[0], -1)
            else:
                # Inpainting.
                inpaint_reference = torch.zeros(
                    (batch_size, self.horizon, self.action_dim + self.obs_dim), dtype=self.dtype, device=self.device
                )
                inpaint_reference[:, : self.n_obs_steps, self.action_dim :] = nobs[:, : self.n_obs_steps]
                inpaint_mask = torch.ones_like(inpaint_reference, dtype=bool)
                inpaint_mask[:, : self.n_obs_steps, self.action_dim :] = False
                inpaint = (inpaint_reference, inpaint_mask)
        elif self.policy_type == PolicyType.HYBRID:
            # Keep just a certain number of observations and reshape B, T, ... to B*T.
            nobs = dict_apply(nobs, lambda v: v[:, : self.n_obs_steps, ...].reshape(-1, *v.shape[2:]))
            nobs_features = self.obs_encoder(nobs)
            # Reshape to (B, D_obs).
            global_cond = nobs_features.reshape(batch_size, -1)
        else:
            raise RuntimeError("policy_type not implemented")

        # "n" in "ntrajectory" for "normalized".
        ntrajectory = self.generate_trajectory(
            self.eval_step_size, batch_size, inpaint=inpaint, global_cond=global_cond, use_ema=True
        )

        # Clip and Unnormalize.
        naction_pred = ntrajectory[..., : self.action_dim]
        # Hmm, clipping not very helpful...
        # clip_sample_range = getattr(self.noise_scheduler.config, "clip_sample_range", 1)
        # naction_pred = naction_pred.clamp(-clip_sample_range, clip_sample_range)
        action_pred = self.normalizer["action"].unnormalize(naction_pred)

        # Get actions to be executed on the robot without re-planning.
        if self.oa_step_convention or self.policy_type == PolicyType.HYBRID:
            # The (i)th action is applied based on the (i)th observation and results in the (i+1)th observation.
            start = self.n_obs_steps - 1
        else:
            # The (i)th action is applied based on the (i-1)th observation and results in the (i)th observation.
            start = self.n_obs_steps
        end = start + self.n_action_steps
        action = action_pred[:, start:end]

        result = {
            # Actions to be executed on the robot without re-planning.
            "action": action,
            # Full horizon action prediction.
            "action_pred": action_pred,
        }

        if self.policy_type == PolicyType.LOWDIM and not self.obs_as_global_cond:
            nobs_pred = ntrajectory[..., self.action_dim :]
            obs_pred = self.normalizer["obs"].unnormalize(nobs_pred)
            action_obs_pred = obs_pred[:, start:end]
            # The part of the observation prediction that aligns with the actions to be executed on the robot.
            result["action_obs_pred"] = action_obs_pred
            # Full horizon observation prediction.
            result["obs_pred"] = obs_pred

        return result

    def compute_loss(self, batch: Dict[str, Tensor], max_timestep: Optional[int] = None) -> Tuple[Tensor, dict]:
        """
        Args:
            batch: Dictionary containing a batch of observations and a corresponding batch of actions.
            max_timestep: Used for consistency training warmup. It limits the range of uniform sampling for the forward-
                diffusion timestep.
        Returns:
            Scalar loss tensor.

        TODO: Incorporate masking and conditioning. PushT lowdim doesn't need it.
        """
        # Get normalized observations and actions.
        if self.policy_type == PolicyType.LOWDIM:
            nbatch = self.normalizer.normalize(batch)
            nobs = nbatch["obs"]  # (batch, horizon, obs_dim)
            naction = nbatch["action"]  # (batch, horizon, action_dim)
        elif self.policy_type == PolicyType.HYBRID:
            nobs = self.normalizer.normalize(batch["obs"])
            naction = self.normalizer["action"].normalize(batch["action"])
        else:
            raise RuntimeError("policy_type not implemented")

        # Create inputs (and maybe conditioning).
        global_cond = None
        if self.policy_type == PolicyType.LOWDIM:
            if self.obs_as_global_cond:
                global_cond = nobs[:, : self.n_obs_steps].reshape(nobs.shape[0], -1)
                x = naction
            else:
                x = torch.cat([naction, nobs], dim=-1)
        elif self.policy_type == PolicyType.HYBRID:
            x = naction
            # Keep just a certain number of observations and reshape B, T, ... to B*T.
            nobs = dict_apply(nobs, lambda v: v[:, : self.n_obs_steps, ...].reshape(-1, *v.shape[2:]))
            with torch.no_grad():#, torch.autocast(self.device.type):
                nobs_features = self.obs_encoder(nobs)
            # Reshape to (B, D_obs).
            global_cond = nobs_features.reshape(naction.shape[0], -1)
        else:
            raise RuntimeError("policy_type not implemented")

        # Tells us where we'll be inpainting.
        condition_mask = self.mask_generator(x.shape)
        loss_mask = (~condition_mask).to(x.dtype)

        # Sample noise.
        z = torch.randn_like(x)
        # Note: It appears the DDPMScheduler.add_noise effectively takes timesteps-1 a an argument. So if we want to
        # noise to the 100th timestep for example, we would pass in 99. Here I'll treat n as the 100, and pass in n-1.
        N = max_timestep if max_timestep is not None else self.N
        assert N == self.N, "I broke this and didn't get around to fixing it."
        nk, sampler_loss_weights = self.timestep_sampler.sample(x.shape[0], device=x.device)
        n = torch.clamp(nk - self.skip_steps, min=0)
        nk_minus1 = nk - 1
        n_minus1 = n - 1
        inputs = self.noise_scheduler.add_noise(x, z, nk_minus1)

        # Here we actually include the inpainting in the training inputs.
        inputs[condition_mask] = x[condition_mask]

        # Keep random state for target model so that dropout is the same.
        rng_state = torch.random.get_rng_state()
        outputs = self.evaluate_consistency_model(inputs, nk, global_cond=global_cond, use_ema=False)
        loss_info = {
            f"sampler_weights/{nk_:03d}": self.timestep_sampler.weights()[nk_ - 1]
            for nk_ in [1, *range(10, self.N + 1, 10)]
            if nk_ in nk
        }
        with torch.no_grad():#, torch.autocast(outputs.device.type):
            # For n == 0 no noise is added.
            if self.do_distill:
                # Use the teacher model with DDIM-Euler.
                ema_inputs = self.ddim_euler_step(inputs, nk, n, global_cond=global_cond)
                loss_info.update(
                    {
                        f"ddim_loss/{nk_:03d}": self.loss_fn(ema_inputs[nk == nk_], x[nk == nk_]).item()
                        for nk_ in [1, *range(10, self.N + 1, 10)]
                    }
                )
            else:
                ema_inputs = torch.zeros_like(x)
                ema_inputs[n == 0] = x[n == 0]
                ema_inputs[n > 0] = self.noise_scheduler.add_noise(x[n > 0], z[n > 0], n_minus1[n > 0])
            ema_inputs[condition_mask] = x[condition_mask]
            ema_outputs = torch.zeros_like(x)
            # For n == 0 the preconditioning guarantees the identity function.
            ema_outputs[n == 0] = ema_inputs[n == 0]
            # Make sure dropout is the same as first forward pass.
            torch.random.set_rng_state(rng_state)
            ema_outputs[n > 0] = self.evaluate_consistency_model(
                ema_inputs[n > 0],
                n[n > 0],
                global_cond=None if global_cond is None else global_cond[n > 0],
                use_ema=self.use_ema_target,
            )
        loss = self.loss_fn(outputs, ema_outputs, reduction="none")
        # Don't include loss for inpainting regions.
        loss = loss * loss_mask
        loss = loss.mean(dim=[i for i in range(1, x.ndim)])
        with torch.no_grad():
            # Unweighted losses for each timestep.
            loss_info.update(
                {f"unweighted_loss/{nk_:03d}": loss[nk == nk_].mean().item() for nk_ in [1, *range(10, self.N + 1, 10)]}
            )
        # I'm going to have to explain why this sqrt does so well.
        loss *= torch.sqrt(self.loss_weights[nk])
        self.timestep_sampler.update_with_all_losses(nk, loss)
        loss *= sampler_loss_weights
        return loss.mean(), loss_info
