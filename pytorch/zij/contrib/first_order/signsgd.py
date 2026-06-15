# Adapted from https://github.com/kozistr/pytorch_optimizer (commit 3d08fa0)
# Copyright (c) Hyeongchan Kim (kozistr). Licensed under the Apache-2.0 license.
# See THIRD_PARTY_NOTICES.md for details.
r"""Implementation of the SignSGD optimizer."""

from typing import Callable, Optional

import torch

from ...core.optimizer import Optimizer, ParamsT

__all__ = ["SignSGD"]


class SignSGD(Optimizer):
    r"""Implements SignSGD, sign-based stochastic gradient descent.

    SignSGD compresses each gradient to its element-wise sign before applying
    the update, which keeps only one bit per coordinate. With ``momentum`` set
    to zero the update is the plain sign of the gradient:

    .. math::
       \theta_t = \theta_{t-1} - \gamma \mathrm{sign}(g_t)

    With a positive momentum coefficient the method becomes Signum, which takes
    the sign of an exponential moving average of the gradients:

    .. math::
       \begin{aligned}
       m_t &= \beta m_{t-1} + (1 - \beta)\, g_t \\
       \theta_t &= \theta_{t-1} - \gamma \mathrm{sign}(m_t)
       \end{aligned}

    where :math:`m_t` is the momentum buffer, :math:`\gamma` the learning rate,
    and :math:`\beta` the momentum coefficient. Decoupled weight decay scales the
    parameters by :math:`(1 - \gamma \lambda)` before the update; with
    ``weight_decouple=False`` the weight decay :math:`\lambda` is instead added
    to the gradient as an L2 penalty.

    Reference: Jeremy Bernstein, Yu-Xiang Wang, Kamyar Azizzadenesheli,
    Anima Anandkumar, "signSGD: Compressed Optimisation for Non-Convex
    Problems", ICML 2018.
    https://arxiv.org/abs/1802.04434
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        maximize: bool = False,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum <= 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        self.maximize = maximize

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            weight_decouple=weight_decouple,
        )
        super().__init__(params, defaults)

    def _init_group(self, group) -> None:
        if "step" not in group:
            group["step"] = 0

        for p in group["params"]:
            if p.grad is None:
                continue
            if p.grad.is_sparse:
                raise RuntimeError("SignSGD does not support sparse gradients.")

            state = self.state[p]
            if group["momentum"] > 0.0 and "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(p)

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

            momentum = group["momentum"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            weight_decouple = group["weight_decouple"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if self.maximize:
                    grad = grad.neg()

                state = self.state[p]

                if weight_decouple:
                    p.mul_(1.0 - weight_decay * lr)
                elif weight_decay > 0.0:
                    grad = grad.add(p, alpha=weight_decay)

                if momentum > 0.0:
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(grad, alpha=1.0 - momentum)
                else:
                    buf = grad

                update = torch.sgn(buf) if torch.is_complex(buf) else torch.sign(buf)
                p.add_(update, alpha=-lr)

        return loss
