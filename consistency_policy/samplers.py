"""
Nicked from CM repo: https://github.dev/openai/consistency_models with my notes and maybe some tweaks.
"""

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray
import torch as th
from torch import LongTensor, Tensor


class ScheduleSampler(ABC):
    """
    A distribution over timesteps in the diffusion process, intended to reduce
    variance of the objective.

    By default, samplers perform unbiased importance sampling, in which the
    objective's mean is unchanged.
    However, subclasses may override sample() to change how the resampled
    terms are reweighted, allowing for actual changes in the objective.
    """

    def update_with_all_losses(self, ts, losses):
        return

    @abstractmethod
    def weights(self):
        """
        Get a numpy array of weights, one per diffusion step. Diffusion steps are in [1, N].
        The weights tell us how highly we weigh the importance of a timestep. It's probability of being sampled will be
        equal to the normalized weight. The loss for each exemplar with that timestep will be scaled inversely so that
        the overall loss per timestep evens out.

        The weights needn't be normalized, but must be positive.
        """

    def sample(self, batch_size, device):
        """
        Importance-sample timesteps for a batch.

        :param batch_size: the number of timesteps.
        :param device: the torch device to save to.
        :return: a tuple (timesteps, weights):
                 - timesteps: a tensor of timestep indices.
                 - weights: a tensor of weights to scale the resulting losses.
        """
        w = self.weights()
        p = w / np.sum(w)
        indices_np = np.random.choice(len(p), size=(batch_size,), p=p) + 1
        indices = th.from_numpy(indices_np).long().to(device)
        weights_np = 1 / (len(p) * p[indices_np - 1])
        weights = th.from_numpy(weights_np).float().to(device)
        return indices, weights


class UniformSampler(ScheduleSampler):
    def __init__(self, num_timesteps: int):
        self._weights = np.ones([num_timesteps])

    def weights(self):
        return self._weights


class LossSecondMomentResampler(ScheduleSampler):
    """
    Consider the RMS of the past history of losses by timestep. Use it to up-weigh the probability of sampling that
    timestep next, and inversely down-weigh the loss for that timestep (which evens out to the same amount of loss
    weight for that timestep overall, although divided over more exemplars).
    """

    def __init__(self, num_timesteps: int, history_per_term=10, uniform_prob=0.001):
        self.num_timesteps = num_timesteps
        self.history_per_term = history_per_term
        self.uniform_prob = uniform_prob
        self._loss_history = np.zeros([num_timesteps, history_per_term], dtype=np.float64)
        self._loss_counts = np.zeros([num_timesteps], dtype=np.int)

    def weights(self) -> NDArray[np.float64]:
        if not self._warmed_up():
            return np.ones([self.num_timesteps], dtype=np.float64) / self.num_timesteps
        weights = np.sqrt(np.mean(self._loss_history**2, axis=-1))
        weights /= np.sum(weights)
        # Compute a blending: weights * (1 - self.uniform_prob) + uniform_weighting * self.uniform_prob.
        weights *= 1 - self.uniform_prob
        weights += self.uniform_prob / len(weights)
        return weights

    def update_with_all_losses(self, ts: LongTensor, losses: Tensor):
        # -1 to move from [1, N] to [0, N-1].
        ts = ts - 1
        for t, loss in zip(ts, losses.detach().clone()):
            if self._loss_counts[t] == self.history_per_term:
                # Shift out the oldest loss term.
                self._loss_history[t, :-1] = self._loss_history[t, 1:]
                self._loss_history[t, -1] = loss
            else:
                self._loss_history[t, self._loss_counts[t]] = loss
                self._loss_counts[t] += 1

    def _warmed_up(self):
        return (self._loss_counts == self.history_per_term).all()


class LossEMAResampler(ScheduleSampler):
    """
    Consider the EMA of the past history of losses by timestep. Use it to up-weigh the probability of sampling that
    timestep next, and inversely down-weigh the loss for that timestep (which evens out to the same amount of loss
    weight for that timestep overall, although divided over more exemplars).
    """

    def __init__(self, num_timesteps: int, alpha: float, uniform_prob=0.001):
        self.num_timesteps = num_timesteps
        self.alpha = alpha
        self.uniform_prob = uniform_prob
        # The number of steps needed for the decay to go to 10%.
        self._warmup_steps = int(round(np.log(0.1) / np.log(alpha)))
        self._loss_ema = np.zeros([num_timesteps], dtype=np.float64)
        self._loss_counts = np.zeros([num_timesteps], dtype=np.int)

    def weights(self) -> NDArray[np.float64]:
        if not self._warmed_up():
            return np.ones([self.num_timesteps], dtype=np.float64) / self.num_timesteps
        weights = self._loss_ema.copy()
        weights /= np.sum(weights)
        # Compute a blending: weights * (1 - self.uniform_prob) + uniform_weighting * self.uniform_prob.
        weights *= 1 - self.uniform_prob
        weights += self.uniform_prob / len(weights)
        return weights

    def update_with_all_losses(self, ts: LongTensor, losses: Tensor):
        # -1 to move from [1, N] to [0, N-1].
        ts = (ts - 1).cpu().numpy()
        losses = losses.detach().cpu().numpy()
        for t in np.unique(ts):
            if self._loss_counts[t] == 0:
                self._loss_ema[t] = losses[ts == t].mean()
            else:
                self._loss_ema[t] = self.alpha * self._loss_ema[t] + (1 - self.alpha) * losses[ts == t].mean()
            if self._loss_counts[t] != self._warmup_steps:
                self._loss_counts[t] += 1

    def _warmed_up(self):
        return (self._loss_counts == self._warmup_steps).all()


class StratifiedUniformSampler(ScheduleSampler):
    def __init__(self, num_timesteps: int, num_strata: int):
        self.num_timesteps = num_timesteps
        self.num_strata = num_strata
        self._weights = np.ones([num_timesteps])

    def weights(self):
        return self._weights

    def sample(self, batch_size, device):
        n_per_strata, remainder = divmod(batch_size, self.num_strata)
        strata_bounds = np.linspace(1, self.num_timesteps + 1, self.num_strata + 1)
        strata_bounds = np.column_stack([strata_bounds[:-1], strata_bounds[1:]])
        strata_samples: list[NDArray[np.int64]] = []
        for l, u in strata_bounds:
            strata_samples.append(np.floor(np.random.uniform(l, u, size=(n_per_strata,))))
        if remainder != 0:
            strata_samples.append(np.floor(np.random.uniform(1, self.num_timesteps, size=(remainder,))))
        indices_np = np.concatenate(strata_samples)
        return th.from_numpy(indices_np).long().to(device), th.ones((len(indices_np),)).float().to(device)
