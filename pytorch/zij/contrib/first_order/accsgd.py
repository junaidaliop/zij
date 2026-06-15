# Adapted from https://github.com/jettify/pytorch-optimizer (commit 19c3e41)
# Copyright (c) 2020 Nikolay Novik. Licensed under the Apache-2.0 license.
# See THIRD_PARTY_NOTICES.md for details.
"""Implementation of the AccSGD optimizer."""

import copy

import torch

from ...core.optimizer import Optimizer, ParamsT

__all__ = ["AccSGD"]


class AccSGD(Optimizer):
    r"""Implements AccSGD, an accelerated stochastic gradient method.

    AccSGD couples a short, plain SGD step with a long, momentum-like step and
    blends the two iterates each update. With :math:`\eta` the learning rate,
    :math:`\kappa` the long-to-short step ratio, :math:`\xi` the statistical
    advantage parameter, and a constant :math:`0 < c \le 1`, the derived
    coefficients are

    .. math::
       \begin{aligned}
            \alpha &= 1 - \frac{c^2\,\xi}{\kappa}, \qquad
            \beta = 1 - \alpha, \qquad
            \zeta = \frac{c}{c + \beta},                                        \\
            \tilde{w}_t &= \beta\Big[(\tfrac{1}{\beta} - 1)\,\tilde{w}_{t-1}
                - \tfrac{\eta\kappa}{c}\,g_t + \theta_{t-1}\Big],               \\
            \theta_t &= \zeta\,(\theta_{t-1} - \eta\,g_t)
                + (1 - \zeta)\,\tilde{w}_t,
       \end{aligned}

    where :math:`\tilde{w}_t` is the accelerated running iterate, initialized to
    :math:`\theta_0`.

    Reference: Prateek Jain, Sham M. Kakade, Rahul Kidambi, Praneeth Netrapalli,
    Aaron Sidford, "Accelerating Stochastic Gradient Descent For Least Squares
    Regression", COLT 2018. https://arxiv.org/abs/1704.08227
    Companion analysis: Rahul Kidambi, Praneeth Netrapalli, Prateek Jain,
    Sham M. Kakade, "On the insufficiency of existing momentum schemes for
    Stochastic Optimization", ICLR 2018. https://arxiv.org/abs/1803.05591
    Reference implementation: https://github.com/rahulkidambi/AccSGD
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        kappa: float = 1000.0,
        xi: float = 10.0,
        small_const: float = 0.7,
        weight_decay: float = 0.0,
    ) -> None:
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = {
            "lr": lr,
            "kappa": kappa,
            "xi": xi,
            "small_const": small_const,
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
            weight_decay = group["weight_decay"]
            large_lr = (group["lr"] * group["kappa"]) / group["small_const"]
            alpha = 1.0 - (
                (group["small_const"] * group["small_const"] * group["xi"])
                / group["kappa"]
            )
            beta = 1.0 - alpha
            zeta = group["small_const"] / (group["small_const"] + beta)

            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = copy.deepcopy(p)
                buf = state["momentum_buffer"]
                buf.mul_((1.0 / beta) - 1.0)
                buf.add_(d_p, alpha=-large_lr)
                buf.add_(p)
                buf.mul_(beta)

                p.add_(d_p, alpha=-group["lr"])
                p.mul_(zeta)
                p.add_(buf, alpha=1.0 - zeta)

        return loss
