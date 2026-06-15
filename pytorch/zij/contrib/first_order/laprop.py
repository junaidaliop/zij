# Adapted from https://github.com/Z-T-WANG/LaProp-Optimizer (commit a419916)
# Copyright (c) 2020 Wang, T. Zhikang. Licensed under the MIT license.
# See THIRD_PARTY_NOTICES.md for details.
r"""Implementation of the LaProp optimizer."""

import torch

from ...core.optimizer import Optimizer, ParamsT

__all__ = ["LaProp"]


class LaProp(Optimizer):
    r"""Implements LaProp, which separates momentum from adaptivity in Adam.

    LaProp divides the gradient by the second-moment estimate *before*
    accumulating momentum, so the momentum buffer holds already-normalized
    steps rather than raw gradients.

    .. math::
       \begin{aligned}
            v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2                          \\
            m_t &= \beta_1 m_{t-1} + (1 - \beta_1)
                \frac{g_t}{\sqrt{v_t / (1 - \beta_2^t)} + \epsilon}               \\
            \theta_t &= \theta_{t-1} - \frac{\eta}{1 - \beta_1^t} \, m_t
       \end{aligned}

    The bias-correction terms are tracked as exponential moving averages so
    that a learning rate that changes across steps is handled correctly.

    Reference: Liu Ziyin, Zhikang T. Wang, Masahito Ueda,
    "LaProp: Separating Momentum and Adaptivity in Adam", arXiv 2020.
    https://arxiv.org/abs/2002.04839
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 4e-4,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-15,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
        centered: bool = False,
    ) -> None:
        self.steps_before_using_centered = 10

        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "amsgrad": amsgrad,
            "centered": centered,
        }
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)
            group.setdefault("centered", False)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            amsgrad = group["amsgrad"]
            centered = group["centered"]
            beta1, beta2 = group["betas"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("LaProp does not support sparse gradients")

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_lr_1"] = 0.0
                    state["exp_avg_lr_2"] = 0.0
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    if centered:
                        state["exp_mean_avg_beta2"] = torch.zeros_like(p)
                    if amsgrad:
                        state["max_exp_avg_sq"] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

                state["step"] += 1

                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                state["exp_avg_lr_1"] = (
                    state["exp_avg_lr_1"] * beta1 + (1 - beta1) * group["lr"]
                )
                state["exp_avg_lr_2"] = state["exp_avg_lr_2"] * beta2 + (1 - beta2)

                bias_correction1 = (
                    state["exp_avg_lr_1"] / group["lr"] if group["lr"] != 0.0 else 1.0
                )
                step_size = 1 / bias_correction1
                bias_correction2 = state["exp_avg_lr_2"]

                denom = exp_avg_sq
                if centered:
                    exp_mean_avg_beta2 = state["exp_mean_avg_beta2"]
                    exp_mean_avg_beta2.mul_(beta2).add_(grad, alpha=1 - beta2)
                    if state["step"] > self.steps_before_using_centered:
                        denom = denom - exp_mean_avg_beta2**2

                if amsgrad:
                    if not (
                        centered and state["step"] <= self.steps_before_using_centered
                    ):
                        max_exp_avg_sq = state["max_exp_avg_sq"]
                        torch.max(max_exp_avg_sq, denom, out=max_exp_avg_sq)
                        denom = max_exp_avg_sq

                denom = denom.div(bias_correction2).sqrt_().add_(group["eps"])
                step_of_this_grad = grad / denom
                exp_avg.mul_(beta1).add_(
                    step_of_this_grad, alpha=(1 - beta1) * group["lr"]
                )

                p.add_(exp_avg, alpha=-step_size)
                if group["weight_decay"] != 0:
                    p.add_(p, alpha=-group["weight_decay"])

        return loss
