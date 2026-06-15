# Adapted from https://github.com/kozistr/pytorch_optimizer (commit 3d08fa0)
# Copyright (c) Hyeongchan Kim (kozistr). Licensed under the Apache-2.0 license.
# See THIRD_PARTY_NOTICES.md for details.
r"""Implementation of the FOCUS optimizer."""

import math
from typing import Callable, Optional

import torch

from ...core.optimizer import Optimizer, ParamsT

__all__ = ["FOCUS"]


class FOCUS(Optimizer):
    r"""Implements FOCUS, a sign-momentum optimizer with attraction to a moving
    target.

    FOCUS extends Signum with an attraction force toward an exponential moving
    average of the parameters. The momentum supplies the descent direction
    through its sign, while the attraction term pulls the parameters toward the
    smoothed target with strength :math:`\gamma`.

    .. math::
       \begin{aligned}
            m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t                           \\
            \bar{\theta}_t &= \beta_2 \bar{\theta}_{t-1} + (1 - \beta_2) \theta_t \\
            \hat{\theta}_t &= \frac{\bar{\theta}_t}{1 - \beta_2^t}               \\
            \theta_t &= \theta_t - \eta \, \omega \, \hat{\theta}_t              \\
            \theta_{t+1} &= \theta_t - \eta \left(
                \mathrm{sign}(m_t)
                + \gamma \mathrm{sign}(\theta_t - \hat{\theta}_t)
            \right)
       \end{aligned}

    where :math:`m_t` is the gradient moment, :math:`\bar{\theta}_t` the moving
    average of the parameters with bias-corrected form :math:`\hat{\theta}_t`,
    :math:`\eta` the learning rate, :math:`\gamma` the attraction strength, and
    :math:`\omega` the decoupled weight decay applied toward the moving target.

    Reference: Yizhou Liu, Ziming Liu, Jeff Gore, "FOCUS: First Order
    Concentrated Updating Scheme", arXiv 2025.
    https://arxiv.org/abs/2501.12243
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-2,
        betas: tuple[float, float] = (0.9, 0.999),
        gamma: float = 0.1,
        weight_decay: float = 0.0,
        maximize: bool = False,
    ) -> None:
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= gamma < 1.0:
            raise ValueError(f"Invalid gamma value: {gamma}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        self.maximize = maximize

        defaults = {
            "lr": lr,
            "betas": betas,
            "gamma": gamma,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if "step" not in group:
                group["step"] = 0
            group["step"] += 1

            lr = group["lr"]
            beta1, beta2 = group["betas"]
            gamma = group["gamma"]
            weight_decay = group["weight_decay"]

            bias_correction2 = 1.0 - math.pow(beta2, group["step"])

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("FOCUS does not support sparse gradients")
                if self.maximize:
                    grad = grad.neg()

                state = self.state[p]

                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)
                    state["pbar"] = torch.zeros_like(p)

                exp_avg, pbar = state["exp_avg"], state["pbar"]

                if torch.is_complex(p):
                    p = torch.view_as_real(p)
                    grad = torch.view_as_real(grad)
                    exp_avg = torch.view_as_real(exp_avg)
                    pbar = torch.view_as_real(pbar)

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                pbar.mul_(beta2).add_(p, alpha=1.0 - beta2)

                pbar_hat = pbar / bias_correction2

                if weight_decay > 0.0:
                    p.add_(pbar_hat, alpha=-lr * weight_decay)

                update = (p - pbar_hat).sign_().mul_(gamma).add_(torch.sign(exp_avg))

                p.add_(update, alpha=-lr)

        return loss
