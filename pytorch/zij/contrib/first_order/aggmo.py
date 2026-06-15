# Adapted from https://github.com/jettify/pytorch-optimizer (commit 19c3e41)
# Copyright (c) Nikolay Novik (jettify). Licensed under the Apache-2.0 license.
# See THIRD_PARTY_NOTICES.md for details.
r"""Implementation of the AggMo optimizer."""

import torch

from ...core.optimizer import Optimizer, ParamsT

__all__ = ["AggMo"]


class AggMo(Optimizer):
    r"""Implements AggMo, momentum descent aggregated over several damping rates.

    AggMo keeps one velocity buffer per damping coefficient :math:`\beta^{(i)}`
    and averages their contributions, so that the small coefficients react
    quickly to gradient changes while the large ones supply passive damping:

    .. math::
       \begin{aligned}
            v_t^{(i)} &= \beta^{(i)} v_{t-1}^{(i)} - g_t,
                \quad i = 1, \dots, K                                       \\
            \theta_t &= \theta_{t-1} + \frac{\eta}{K}
                \sum_{i=1}^{K} v_t^{(i)}
       \end{aligned}

    Reference: James Lucas, Shengyang Sun, Richard Zemel, Roger Grosse,
    "Aggregated Momentum: Stability Through Passive Damping", ICLR 2019.
    https://arxiv.org/abs/1804.00325
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        betas: tuple[float, ...] = (0.0, 0.9, 0.99),
        weight_decay: float = 0.0,
    ) -> None:
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        for i, beta in enumerate(betas):
            if not 0.0 <= beta < 1.0:
                raise ValueError(f"Invalid beta parameter at index {i}: {beta}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = {"lr": lr, "betas": betas, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    @classmethod
    def from_exp_form(
        cls,
        params: ParamsT,
        lr: float = 1e-3,
        a: float = 0.1,
        k: int = 3,
        weight_decay: float = 0.0,
    ) -> "AggMo":
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")

        betas = tuple(1 - a**i for i in range(k))
        return cls(params, lr, betas, weight_decay)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            betas = group["betas"]
            total_mom = float(len(betas))

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = {
                        beta: torch.zeros_like(p, memory_format=torch.preserve_format)
                        for beta in betas
                    }

                for beta in betas:
                    buf = state["momentum_buffer"][beta]
                    buf.mul_(beta).add_(grad)
                    p.sub_(buf, alpha=group["lr"] / total_mom)

        return loss
