# Adapted from https://github.com/Gunale0926/Grams (commit f91a375)
# Copyright (c) 2024 Yang Cao, Xiaoyu Li, Zhao Song. Licensed under the Apache-2.0 license.
# See THIRD_PARTY_NOTICES.md for details.
r"""Implementation of the Grams optimizer."""

import math
from typing import Callable, Optional

import torch

from ...core.optimizer import Optimizer, ParamsT

__all__ = ["Grams"]


class Grams(Optimizer):
    r"""Implements Grams, gradient descent with adaptive momentum scaling.

    .. math::
       \begin{aligned}
            m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t                            \\
            v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2                          \\
            \hat{m}_t &= m_t / (1 - \beta_1^t), \quad
                \hat{v}_t = v_t / (1 - \beta_2^t)                                 \\
            \theta_t &= \theta_{t-1} - \eta \,
                \mathrm{sign}(g_t) \odot
                \frac{\lvert \hat{m}_t \rvert}{\sqrt{\hat{v}_t} + \epsilon}
       \end{aligned}

    The update direction is taken from the sign of the current gradient, while
    the Adam-style first moment supplies only the per-coordinate magnitude, so
    direction and magnitude are decoupled. Bias correction is applied through
    the step size when ``correct_bias`` is set, and weight decay is decoupled.

    Reference: Yang Cao, Xiaoyu Li, Zhao Song,
    "Grams: Gradient Descent with Adaptive Momentum Scaling", arXiv 2024.
    https://arxiv.org/abs/2412.17107
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)"
            )
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)"
            )
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "correct_bias": correct_bias,
        }
        super().__init__(params, defaults)
        self.init_lr = lr

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                if "step" not in state:
                    state["step"] = 0

                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(grad)
                    state["exp_avg_sq"] = torch.zeros_like(grad)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                if group["weight_decay"] > 0.0:
                    p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))

                exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if group["correct_bias"]:
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = (
                        step_size * math.sqrt(bias_correction2) / bias_correction1
                    )

                grad = grad.sign().mul_(exp_avg.abs())
                p.addcdiv_(grad, denom, value=-step_size)

        return loss
