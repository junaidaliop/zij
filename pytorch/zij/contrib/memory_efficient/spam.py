# Adapted from https://github.com/kozistr/pytorch_optimizer (commit 3d08fa0)
# Copyright (c) Hyeongchan Kim (kozistr). Licensed under the Apache-2.0 license.
# See THIRD_PARTY_NOTICES.md for details.
r"""Implementation of the SPAM optimizer."""

import math

import torch
from torch.nn import Parameter, ParameterList
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

from ...core.optimizer import Optimizer, ParamsT

__all__ = ["SPAM"]


class _CosineDecay:
    """Cosine decay of a scalar, backed by :class:`CosineAnnealingLR`.

    Drives the post-reset warmup factor: the death rate decays from its
    initial value to ``eta_min`` over ``t_max`` steps and is held at
    ``eta_min`` afterward.
    """

    def __init__(self, death_rate, t_max, eta_min=0.0, last_epoch=-1):
        self.sgd = SGD(ParameterList([Parameter(torch.zeros(1))]), lr=death_rate)
        self.cosine_stepper = CosineAnnealingLR(self.sgd, t_max + 1, eta_min, last_epoch)
        self.t_max = t_max
        self.eta_min = eta_min

    def step(self, current_step):
        self.cosine_stepper.last_epoch = current_step
        self.cosine_stepper.step()

    def get_death_rate(self, current_step):
        if current_step >= self.t_max:
            return self.eta_min
        self.step(current_step)
        return self.sgd.param_groups[0]["lr"]


class SPAM(Optimizer):
    r"""Implements SPAM, Spike-Aware Adam with momentum reset for stable training.

    SPAM augments Adam with two stabilizing mechanisms. Spike-aware
    clipping caps any gradient coordinate whose squared value exceeds a
    multiple of its running second moment, replacing it with a magnitude
    bounded by that second moment:

    .. math::
       \begin{aligned}
       g_{t,i} &\leftarrow \mathrm{sign}(g_{t,i})\,
           \sqrt{\tau\, v_{t-1,i}}
           &&\text{if } g_{t,i}^2 > \tau\, v_{t-1,i} \\
       m_t &= \beta_1 m_{t-1} + (1 - \beta_1)\, g_t \\
       v_t &= \beta_2 v_{t-1} + (1 - \beta_2)\, g_t^2 \\
       \theta_t &= \theta_{t-1} - \gamma\, \phi_t\,
           \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
       \end{aligned}

    where :math:`\theta_t` are the parameters, :math:`\tau` is the fixed
    spike-detection ``threshold`` (default 5000, never updated), :math:`\hat{m}_t`
    and :math:`\hat{v}_t` are the bias-corrected moments, and :math:`\phi_t`
    is a cosine warmup factor. Every ``update_proj_gap`` steps the moments
    :math:`m, v` are reset to zero and the warmup restarts, which clears
    accumulated momentum after a spike. For two-dimensional parameters a
    random binary mask of fraction ``density`` selects the coordinates that
    keep momentum (sparse momentum), and the mask is resampled at each reset.

    Reference: Tianjin Huang, Ziquan Zhu, Gaojie Jin, Lu Liu, Zhangyang
    Wang, Shiwei Liu, "SPAM: Spike-Aware Adam with Momentum Reset for
    Stable LLM Training", ICLR 2025.
    https://arxiv.org/abs/2501.06842
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        density: float = 1.0,
        weight_decay: float = 0.0,
        warmup_epoch: int = 50,
        threshold: int = 5000,
        grad_accu_steps: int = 20,
        update_proj_gap: int = 500,
        eps: float = 1e-6,
        maximize: bool = False,
    ) -> None:
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        for i, beta in enumerate(betas):
            if not 0.0 <= beta < 1.0:
                raise ValueError(f"Invalid beta parameter at index {i}: {beta}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= density <= 1.0:
            raise ValueError(f"Invalid density value: {density}")
        if warmup_epoch < 0:
            raise ValueError(f"Invalid warmup_epoch value: {warmup_epoch}")
        if threshold < 0:
            raise ValueError(f"Invalid threshold value: {threshold}")
        if grad_accu_steps < 0:
            raise ValueError(f"Invalid grad_accu_steps value: {grad_accu_steps}")
        if update_proj_gap < 1:
            raise ValueError(f"Invalid update_proj_gap value: {update_proj_gap}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")

        self.density = density
        self.warmup_epoch = warmup_epoch
        self.threshold = threshold
        self.grad_accu_steps = grad_accu_steps
        self.update_proj_gap = update_proj_gap
        self.maximize = maximize

        defaults = {
            "lr": lr,
            "betas": betas,
            "weight_decay": weight_decay,
            "eps": eps,
        }
        super().__init__(params, defaults)

        self.warmup = _CosineDecay(0.99, self.warmup_epoch)
        self.init_masks()

        self.state["total_step"] = 0
        self.state["current_step"] = self.warmup_epoch + 1

    @staticmethod
    def _random_mask(m, n, density, device):
        total = m * n
        non_zero = int(density * total)
        tensor = torch.zeros(total, dtype=torch.bool, device=device)
        if non_zero > 0:
            tensor[torch.randperm(total, device=device)[:non_zero]] = True
        return tensor.view(m, n)

    def update_mask_random(self, p, old_mask):
        new_mask = torch.rand_like(p) < self.density

        exp_avg = torch.zeros_like(p[new_mask])
        exp_avg_sq = torch.zeros_like(p[new_mask])

        intersection_mask = new_mask & old_mask
        new_intersection_indices = intersection_mask[new_mask]
        old_intersection_indices = intersection_mask[old_mask]

        state = self.state[p]
        exp_avg[new_intersection_indices] = state["exp_avg"][old_intersection_indices]
        exp_avg_sq[new_intersection_indices] = state["exp_avg_sq"][old_intersection_indices]

        state["exp_avg"] = exp_avg
        state["exp_avg_sq"] = exp_avg_sq

        return new_mask

    def update_masks(self):
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if p.dim() == 2 and "mask" in state:
                    state["mask"] = self.update_mask_random(p, state["mask"])
                    p.mask = state["mask"]

    def init_masks(self):
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if p.dim() == 2 and "mask" not in state:
                    state["mask"] = self._random_mask(
                        p.shape[0], p.shape[1], self.density, p.device
                    )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        scale_factor = 1.0 - self.warmup.get_death_rate(self.state["current_step"])

        for group in self.param_groups:
            if "step" not in group:
                group["step"] = 0
            group["step"] += 1

            beta1, beta2 = group["betas"]

            bias_correction1 = 1.0 - beta1 ** group["step"]
            bias_correction2_sq = math.sqrt(1.0 - beta2 ** group["step"])

            step_size = group["lr"] * bias_correction2_sq / bias_correction1

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("SPAM does not support sparse gradients")
                if torch.is_complex(p):
                    raise RuntimeError("SPAM does not support complex parameters")
                if self.maximize:
                    grad = -grad

                state = self.state[p]

                if "mask" in state:
                    grad = grad[state["mask"]]

                if ("exp_avg" not in state) or (
                    self.state["total_step"] + 1
                ) % self.update_proj_gap == 0:
                    state["exp_avg"] = torch.zeros_like(grad)
                    state["exp_avg_sq"] = torch.zeros_like(grad)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

                if self.threshold != 0:
                    current_step = self.state["total_step"] + 1
                    if current_step >= self.grad_accu_steps and (
                        self.update_proj_gap == 0
                        or current_step % self.update_proj_gap >= self.grad_accu_steps
                    ):
                        mask = grad.pow(2) > (self.threshold * exp_avg_sq)
                        grad = grad.clone()
                        grad[mask] = grad[mask].sign() * torch.sqrt(
                            exp_avg_sq[mask] * self.threshold
                        )

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                de_nom = exp_avg_sq.sqrt().add_(group["eps"])

                if "mask" in state:
                    grad_full = torch.zeros_like(p.grad)
                    grad_full[state["mask"]] = exp_avg / de_nom
                    p.add_(grad_full, alpha=-step_size * scale_factor)
                else:
                    p.addcdiv_(exp_avg, de_nom, value=-step_size * scale_factor)

                if group["weight_decay"] > 0.0:
                    target = p[state["mask"]] if "mask" in state else p
                    target.mul_(1.0 - group["weight_decay"] * group["lr"])

        self.state["total_step"] += 1
        self.state["current_step"] += 1

        if (self.state["total_step"] != 0) and (
            self.state["total_step"] + 1
        ) % self.update_proj_gap == 0:
            self.update_masks()
            self.state["current_step"] = 0
            self.warmup = _CosineDecay(0.99, self.warmup_epoch)

        return loss
