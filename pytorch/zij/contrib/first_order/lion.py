# Adapted from https://github.com/lucidrains/lion-pytorch (commit 6a74fdc)
# Copyright (c) 2023 Phil Wang. Licensed under the MIT license.
# See THIRD_PARTY_NOTICES.md for details.
"""Implementation of the Lion optimizer."""

from __future__ import annotations

from typing import Callable

import torch

from ...core.optimizer import Optimizer

__all__ = ["Lion"]


def exists(val):
    return val is not None


def update_fn(p, grad, exp_avg, lr, wd, beta1, beta2):
    # stepweight decay

    p.data.mul_(1.0 - lr * wd)

    # weight update

    update = exp_avg.clone().mul_(beta1).add(grad, alpha=1.0 - beta1).sign_()
    p.add_(update, alpha=-lr)

    # decay the momentum running average coefficient

    exp_avg.mul_(beta2).add_(grad, alpha=1.0 - beta2)


class Lion(Optimizer):
    r"""Implements Lion, a sign-momentum optimizer discovered by symbolic search.

    .. math::
       \begin{aligned}
       c_t &= \beta_1 m_{t-1} + (1 - \beta_1)\, g_t \\
       \theta_t &= \theta_{t-1} - \gamma \left( \mathrm{sign}(c_t)
                   + \lambda \theta_{t-1} \right) \\
       m_t &= \beta_2 m_{t-1} + (1 - \beta_2)\, g_t
       \end{aligned}

    where :math:`m_t` is the single momentum buffer, :math:`\lambda` is the
    decoupled weight decay, and the update direction is the element-wise sign of
    the interpolated momentum :math:`c_t`. The interpolation rate
    :math:`\beta_1` and the momentum decay :math:`\beta_2` are passed as
    ``betas``.

    Reference: Xiangning Chen et al., "Symbolic Discovery of Optimization
    Algorithms", NeurIPS 2023.
    https://arxiv.org/abs/2302.06675
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
        decoupled_weight_decay: bool = False,
    ):
        assert lr > 0.0
        assert all([0.0 <= beta <= 1.0 for beta in betas])

        self._init_lr = lr
        self.decoupled_wd = decoupled_weight_decay

        defaults = dict(
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
        )

        super().__init__(params, defaults)

        self.update_fn = update_fn

    @torch.no_grad()
    def step(self, closure: Callable | None = None):
        loss = None
        if exists(closure):
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in filter(lambda p: exists(p.grad), group["params"]):
                grad, lr, wd, beta1, beta2, state, decoupled_wd, init_lr = (
                    p.grad,
                    group["lr"],
                    group["weight_decay"],
                    *group["betas"],
                    self.state[p],
                    self.decoupled_wd,
                    self._init_lr,
                )

                # maybe decoupled weight decay

                if decoupled_wd:
                    wd /= init_lr

                # init state - exponential moving average of gradient values

                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]

                self.update_fn(p, grad, exp_avg, lr, wd, beta1, beta2)

        return loss
