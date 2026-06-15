# Adapted from https://github.com/jettify/pytorch-optimizer (commit 19c3e41)
# Copyright (c) 2020 Nikolay Novik. Licensed under the Apache-2.0 license.
# See THIRD_PARTY_NOTICES.md for details.
"""Implementation of the DiffGrad optimizer."""

import math

import torch

from ...core.optimizer import Optimizer, ParamsT

__all__ = ["DiffGrad"]


class DiffGrad(Optimizer):
    r"""Implements DiffGrad, Adam scaled by a gradient-change friction coefficient.

    The friction coefficient :math:`\xi_t \in [0.5, 1]` is the sigmoid of the
    absolute change between consecutive gradients, so the first moment is damped
    toward half its value where consecutive gradients agree (the gradient is
    stable) and passed through nearly unchanged where they differ sharply (the
    gradient is changing quickly):

    .. math::
       \begin{aligned}
       m_t &= \beta_1 m_{t-1} + (1 - \beta_1)\, g_t \\
       v_t &= \beta_2 v_{t-1} + (1 - \beta_2)\, g_t^2 \\
       \xi_t &= \frac{1}{1 + e^{-|g_{t-1} - g_t|}} \\
       \theta_t &= \theta_{t-1} - \eta\,
                   \frac{\sqrt{1 - \beta_2^t}}{1 - \beta_1^t}\,
                   \frac{\xi_t\, m_t}{\sqrt{v_t} + \epsilon}
       \end{aligned}

    Reference: Shiv Ram Dubey, Soumendu Chakraborty, Swalpa Kumar Roy,
    Snehasis Mukherjee, Satish Kumar Singh, Bidyut Baran Chaudhuri,
    "diffGrad: An Optimization Method for Convolutional Neural Networks",
    IEEE Transactions on Neural Networks and Learning Systems 2020.
    https://arxiv.org/abs/1909.11015
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError(
                        "DiffGrad does not support sparse gradients, "
                        "please consider SparseAdam instead"
                    )

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    state["previous_grad"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                exp_avg, exp_avg_sq, previous_grad = (
                    state["exp_avg"],
                    state["exp_avg_sq"],
                    state["previous_grad"],
                )

                state["step"] += 1

                if group["weight_decay"] != 0:
                    grad = grad.add(p, alpha=group["weight_decay"])

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                # diffgrad friction coefficient from the gradient change
                diff = torch.abs(previous_grad - grad)
                dfc = 1.0 / (1.0 + torch.exp(-diff))
                state["previous_grad"] = grad.clone()

                exp_avg1 = exp_avg * dfc

                step_size = (
                    group["lr"] * math.sqrt(bias_correction2) / bias_correction1
                )

                p.addcdiv_(exp_avg1, denom, value=-step_size)

        return loss
