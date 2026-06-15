# Adapted from https://github.com/kozistr/pytorch_optimizer (commit 3d08fa02cb66)
# Copyright (c) 2021 Hyeongchan Kim (kozistr). Licensed under the Apache-2.0 license.
# See THIRD_PARTY_NOTICES.md for details.
r"""Implementation of the AdaPNM optimizer."""

import math
from typing import Callable, Optional

import torch

from ...core.optimizer import Optimizer, ParamsT

__all__ = ["AdaPNM"]


class AdaPNM(Optimizer):
    r"""Implements AdaPNM, the adaptive (Adam) form of positive-negative
    momentum.

    .. math::
       \begin{aligned}
            m_t &= \beta_1^2 m_{t-2} + (1 - \beta_1^2) g_t                        \\
            v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2                          \\
            \hat{v}_t &= \max(\hat{v}_{t-1}, v_t)                                 \\
            \pi_t &= \frac{(1 + \beta_3) m_t - \beta_3 m_{t-1}}
                {\sqrt{(1 + \beta_3)^2 + \beta_3^2}}                              \\
            \theta_t &= \theta_{t-1} - \frac{\eta}{1 - \beta_1^t}
                \frac{\pi_t}{\sqrt{\hat{v}_t / (1 - \beta_2^t)} + \epsilon}
       \end{aligned}

    Two momentum buffers are kept and their roles swap every step, so the buffer
    that receives the current gradient is decayed by :math:`\beta_1^2` and is two
    steps stale relative to itself. The update direction :math:`\pi_t` mixes the
    fresh positive momentum :math:`m_t` with the previous (negative) momentum
    :math:`m_{t-1}` and renormalizes by
    :math:`\sqrt{(1 + \beta_3)^2 + \beta_3^2}` so that its variance matches a
    plain momentum term. The difference amplifies the stochastic gradient noise,
    which the paper links to improved generalization. The denominator is the Adam
    second moment, taken with the AMSGrad running maximum when ``ams_bound`` is
    set.

    Reference: Zeke Xie, Li Yuan, Zhanxing Zhu, Masashi Sugiyama, "Positive-
    Negative Momentum: Manipulating Stochastic Gradient Noise to Improve
    Generalization", ICML 2021.
    https://arxiv.org/abs/2103.17182
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        betas: tuple[float, float, float] = (0.9, 0.999, 1.0),
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        ams_bound: bool = True,
        eps: float = 1e-8,
        maximize: bool = False,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= betas[2] <= 1.0:
            raise ValueError(f"Invalid beta parameter at index 2: {betas[2]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")

        self.maximize = maximize

        defaults = {
            "lr": lr,
            "betas": betas,
            "weight_decay": weight_decay,
            "weight_decouple": weight_decouple,
            "fixed_decay": fixed_decay,
            "ams_bound": ams_bound,
            "eps": eps,
        }
        super().__init__(params, defaults)

    def _init_group(self, group: dict) -> None:
        if "step" not in group:
            group["step"] = 0

        for p in group["params"]:
            if p.grad is None:
                continue

            grad = p.grad
            if grad.is_sparse:
                raise RuntimeError("AdaPNM does not support sparse gradients")

            state = self.state[p]
            if len(state) == 0:
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)
                state["neg_exp_avg"] = torch.zeros_like(p)
                if group["ams_bound"]:
                    state["max_exp_avg_sq"] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            self._init_group(group)
            group["step"] += 1

            beta1, beta2, beta3 = group["betas"]

            beta1_sq = beta1**2
            noise_norm = math.sqrt((1.0 + beta3) ** 2 + beta3**2)

            bias_correction1 = 1.0 - beta1 ** group["step"]
            bias_correction2_sq = math.sqrt(1.0 - beta2 ** group["step"])

            step_size = group["lr"] / bias_correction1

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if self.maximize:
                    grad = grad.neg()

                state = self.state[p]

                if group["weight_decay"] != 0.0:
                    if group["weight_decouple"]:
                        decay = group["weight_decay"] * (
                            1.0 if group["fixed_decay"] else group["lr"]
                        )
                        p.mul_(1.0 - decay)
                    else:
                        grad = grad.add(p, alpha=group["weight_decay"])

                exp_avg_sq = state["exp_avg_sq"]

                if group["step"] % 2 == 1:
                    exp_avg, neg_exp_avg = state["exp_avg"], state["neg_exp_avg"]
                else:
                    exp_avg, neg_exp_avg = state["neg_exp_avg"], state["exp_avg"]

                exp_avg.mul_(beta1_sq).add_(grad, alpha=1.0 - beta1_sq)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                if group["ams_bound"]:
                    max_exp_avg_sq = state["max_exp_avg_sq"]
                    torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    de_nom = max_exp_avg_sq.add(1e-15)
                else:
                    de_nom = exp_avg_sq.add(1e-15)
                de_nom.sqrt_().add_(group["eps"]).div_(bias_correction2_sq)

                pn_momentum = (
                    exp_avg.mul(1.0 + beta3)
                    .add_(neg_exp_avg, alpha=-beta3)
                    .mul_(1.0 / noise_norm)
                )

                p.addcdiv_(pn_momentum, de_nom, value=-step_size)

        return loss
