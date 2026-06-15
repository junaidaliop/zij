# Adapted from https://github.com/jettify/pytorch-optimizer (commit 19c3e41)
# Copyright (c) 2020 Nikolay Novik. Licensed under the Apache-2.0 license.
# See THIRD_PARTY_NOTICES.md for details.
r"""Implementation of the PID optimizer."""

import torch

from ...core.optimizer import Optimizer, ParamsT

__all__ = ["PID"]


class PID(Optimizer):
    r"""Implements PID, an SGD variant cast as a PID controller.

    Classical momentum is the integral term of a PID controller acting on the
    gradient. PID adds the derivative term, the momentum-smoothed change in the
    gradient between consecutive steps, so that the update reacts to both the
    accumulated history and the instantaneous trend of the gradient:

    .. math::
       \begin{aligned}
            I_t &= \mu I_{t-1} + (1 - \tau) g_t                                 \\
            D_t &= \mu D_{t-1} + (1 - \mu)(g_t - g_{t-1})                        \\
            \theta_t &= \theta_{t-1} - \eta \left( g_t + k_i I_t + k_d D_t
                \right)
       \end{aligned}

    where :math:`\mu` is the momentum, :math:`\tau` the dampening,
    :math:`k_i` the integral gain, and :math:`k_d` the derivative gain.

    Reference: Wangpeng An, Haoqian Wang, Qingyun Sun, Jun Xu, Qionghai Dai,
    Lei Zhang, "A PID Controller Approach for Stochastic Optimization of Deep
    Networks", CVPR 2018.
    http://www4.comp.polyu.edu.hk/~cslzhang/paper/CVPR18_PID.pdf
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        momentum: float = 0.0,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        integral: float = 5.0,
        derivative: float = 10.0,
    ) -> None:
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if integral < 0.0:
            raise ValueError(f"Invalid PID integral value: {integral}")
        if derivative < 0.0:
            raise ValueError(f"Invalid PID derivative value: {derivative}")

        defaults = {
            "lr": lr,
            "momentum": momentum,
            "dampening": dampening,
            "weight_decay": weight_decay,
            "integral": integral,
            "derivative": derivative,
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
            momentum = group["momentum"]
            dampening = group["dampening"]
            integral = group["integral"]
            derivative = group["derivative"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                d_p = p.grad
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)

                if momentum != 0:
                    state = self.state[p]

                    if "i_buffer" not in state:
                        i_buf = state["i_buffer"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )
                        i_buf.mul_(momentum).add_(d_p)
                    else:
                        i_buf = state["i_buffer"]
                        i_buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

                    if "grad_buffer" not in state:
                        g_buf = state["grad_buffer"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )
                        g_buf = d_p

                        d_buf = state["d_buffer"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )
                        d_buf.mul_(momentum).add_(d_p - g_buf)
                    else:
                        d_buf = state["d_buffer"]
                        g_buf = state["grad_buffer"]
                        d_buf.mul_(momentum).add_(
                            d_p - g_buf, alpha=1 - momentum
                        )
                        state["grad_buffer"] = d_p.clone()

                    d_p = d_p.add(i_buf, alpha=integral).add_(
                        d_buf, alpha=derivative
                    )

                p.add_(d_p, alpha=-group["lr"])

        return loss
