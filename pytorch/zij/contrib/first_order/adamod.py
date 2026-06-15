# Adapted from https://github.com/kozistr/pytorch_optimizer (commit 3d08fa02cb66)
# Copyright (c) 2021 Hyeongchan Kim (kozistr). Licensed under the Apache-2.0 license.
# See THIRD_PARTY_NOTICES.md for details.
r"""Implementation of the AdaMod optimizer."""

import math
from typing import Callable, Optional

import torch

from ...core.optimizer import Optimizer, ParamsT

__all__ = ["AdaMod"]


class AdaMod(Optimizer):
    r"""Implements AdaMod, an Adam variant that bounds the per-parameter
    learning rates by an exponential moving average of their past values.

    .. math::
       \begin{aligned}
            m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t                               \\
            v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2                              \\
            \eta_t &= \frac{\alpha \sqrt{1 - \beta_2^t}}{(1 - \beta_1^t)
                (\sqrt{v_t} + \epsilon)}                                             \\
            s_t &= \beta_3 s_{t-1} + (1 - \beta_3) \eta_t                            \\
            \hat{\eta}_t &= \min(\eta_t, s_t)                                        \\
            \theta_t &= \theta_{t-1} - \hat{\eta}_t \odot m_t
       \end{aligned}

    The adaptive learning rate :math:`\eta_t` computed by Adam is smoothed by a
    third exponential moving average :math:`s_t` with decay :math:`\beta_3`, and
    each element of the update is capped at this momental bound. This restrains
    the large learning rates that can appear early in training.

    Reference: Jianbang Ding, Xuancheng Ren, Ruixuan Luo, Xu Sun, "An Adaptive
    and Momental Bound Method for Stochastic Learning", 2019.
    https://arxiv.org/abs/1910.12249
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        betas: tuple[float, float, float] = (0.9, 0.99, 0.9999),
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        eps: float = 1e-8,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= betas[2] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 2: {betas[2]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")

        defaults = {
            "lr": lr,
            "betas": betas,
            "weight_decay": weight_decay,
            "weight_decouple": weight_decouple,
            "fixed_decay": fixed_decay,
            "eps": eps,
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

            beta1, beta2, beta3 = group["betas"]

            bias_correction1 = 1.0 - beta1 ** group["step"]
            bias_correction2_sq = math.sqrt(1.0 - beta2 ** group["step"])
            step_size = group["lr"] * bias_correction2_sq / bias_correction1

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdaMod does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    state["exp_avg_lr"] = torch.zeros_like(p)

                exp_avg, exp_avg_sq, exp_avg_lr = (
                    state["exp_avg"],
                    state["exp_avg_sq"],
                    state["exp_avg_lr"],
                )

                if group["weight_decay"] != 0.0:
                    if group["weight_decouple"]:
                        decay = (
                            group["weight_decay"]
                            if group["fixed_decay"]
                            else group["lr"] * group["weight_decay"]
                        )
                        p.mul_(1.0 - decay)
                    else:
                        grad = grad.add(p, alpha=group["weight_decay"])

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                denom = exp_avg_sq.sqrt().add_(group["eps"])

                update = torch.full_like(denom, fill_value=step_size)
                update.div_(denom)

                exp_avg_lr.mul_(beta3).add_(update, alpha=1.0 - beta3)

                torch.min(update, exp_avg_lr, out=update)
                update.mul_(exp_avg)

                p.add_(-update)

        return loss
