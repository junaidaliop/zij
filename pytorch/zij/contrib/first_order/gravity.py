# Adapted from https://github.com/kozistr/pytorch_optimizer (commit 3d08fa0)
# Copyright (c) Hyeongchan Kim (kozistr). Licensed under the Apache-2.0 license.
# See THIRD_PARTY_NOTICES.md for details.
r"""Implementation of the Gravity optimizer."""

from typing import Callable, Optional

import torch

from ...core.optimizer import Optimizer, ParamsT

__all__ = ["Gravity"]


class Gravity(Optimizer):
    r"""Implements Gravity, a kinematic optimizer.

    Gravity treats each parameter as a point mass rolling down an inclined
    plane whose slope is the gradient, and integrates a constant-acceleration
    kinematic step. The per-coordinate step is largest for moderate gradients
    and saturates as the gradient grows, giving a bounded velocity increment.
    The velocity buffer is seeded from a normal distribution and smoothed by a
    running average whose decay anneals from :math:`\frac{1}{2}` toward
    :math:`\beta` as training proceeds.

    .. math::
       \begin{aligned}
            V_0 &\sim \mathcal{N}\!\left(0, \sigma^2\right),\ \sigma = \frac{\alpha}{\eta}  \\
            \hat{\beta}_t &= \frac{\beta t + 1}{t + 2}                         \\
            m_t &= \frac{1}{\max_i |g_{t,i}|}                                  \\
            \zeta_t &= \frac{g_t}{1 + (g_t / m_t)^2}                           \\
            V_t &= \hat{\beta}_t V_{t-1} + (1 - \hat{\beta}_t) \zeta_t         \\
            \theta_t &= \theta_{t-1} - \eta V_t
       \end{aligned}

    where :math:`g_t` is the gradient, :math:`m_t` the reciprocal of the largest
    gradient magnitude, :math:`\zeta_t` the saturating gravity step,
    :math:`V_t` the velocity buffer, :math:`\eta` the learning rate,
    :math:`\alpha` the velocity initialization scale, and :math:`\beta` the
    asymptotic running-average decay.

    Reference: Dariush Bahrami, Sadegh Pouriyan Zadeh, "Gravity Optimizer: a
    Kinematic Approach on Optimization in Deep Learning", arXiv 2021.
    https://arxiv.org/abs/2101.09192
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-2,
        alpha: float = 0.01,
        beta: float = 0.9,
        maximize: bool = False,
    ) -> None:
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"Invalid alpha value: {alpha}")
        if not 0.0 <= beta <= 1.0:
            raise ValueError(f"Invalid beta value: {beta}")

        self.maximize = maximize

        defaults = {"lr": lr, "alpha": alpha, "beta": beta}
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
            alpha = group["alpha"]
            beta = group["beta"]

            beta_t = (beta * group["step"] + 1.0) / (group["step"] + 2.0)

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Gravity does not support sparse gradients")
                if self.maximize:
                    grad = grad.neg()

                state = self.state[p]

                if len(state) == 0:
                    state["v"] = torch.empty_like(p).normal_(mean=0.0, std=alpha / lr)

                v = state["v"]

                if torch.is_complex(p):
                    p = torch.view_as_real(p)
                    grad = torch.view_as_real(grad)
                    v = torch.view_as_real(v)

                m = 1.0 / grad.abs().max()
                zeta = grad / (1.0 + (grad / m) ** 2)

                v.mul_(beta_t).add_(zeta, alpha=1.0 - beta_t)

                p.add_(v, alpha=-lr)

        return loss
