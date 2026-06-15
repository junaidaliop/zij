# Adapted from https://github.com/kozistr/pytorch_optimizer (commit 3d08fa02cb66)
# Copyright (c) 2021 Hyeongchan Kim (kozistr). Licensed under the Apache-2.0 license.
# See THIRD_PARTY_NOTICES.md for details.
r"""Implementation of the FAdam optimizer."""

import math
from typing import Callable, Optional

import torch

from ...core.optimizer import Optimizer, ParamsT

__all__ = ["FAdam"]


class FAdam(Optimizer):
    r"""Implements FAdam (Fisher Adam), recasting Adam as natural gradient descent
    with a diagonal empirical Fisher information matrix.

    This is Fisher Adam, not a fractional Adam variant: the name FAdam refers to
    the Fisher information interpretation, where the second-moment buffer is read
    as a diagonal empirical Fisher and the update is a natural gradient step.

    .. math::
       \begin{aligned}
            f_t &= \beta_2 f_{t-1} + (1 - \beta_2) g_t^2                             \\
            \bar{g}_t &= \frac{g_t}{f_t^{p} + \epsilon}                              \\
            \hat{g}_t &= \frac{\bar{g}_t}{\max(1, \lVert \bar{g}_t \rVert_{rms} / c)}\\
            m_t &= \beta_1 m_{t-1} + (1 - \beta_1) \hat{g}_t                         \\
            w_t &= \frac{\theta_{t-1}}{f_t^{p} + \epsilon}                           \\
            \hat{w}_t &= \frac{w_t}{\max(1, \lVert w_t \rVert_{rms} / c)}            \\
            \theta_t &= \theta_{t-1} - \eta \left( m_t + \lambda \hat{w}_t \right)
       \end{aligned}

    The buffer :math:`f_t` accumulates the squared gradient as a diagonal
    empirical Fisher. The gradient is divided by :math:`f_t^{p}` to form the
    natural gradient :math:`\bar{g}_t` (with :math:`p = 1/2` recovering the Adam
    denominator), both the natural gradient and the weight-decay term are
    root-mean-square clipped to a maximum norm :math:`c`, and momentum is applied
    to the clipped natural gradient. The decoupled weight decay :math:`\lambda`
    is itself preconditioned by the Fisher.

    Note: following the official implementation, the Fisher EMA uses a debiased
    decay :math:`\hat{\beta}_2 = \beta_2 (1 - \beta_2^{t-1}) / (1 - \beta_2^{t})`
    in place of :math:`\beta_2`, and the stability constant is scaled by the
    gradient RMS, :math:`\epsilon_t = \min(\mathrm{RMS}(g_t), 1)\,\epsilon`, so
    the denominator is :math:`f_t^p + \epsilon_t`.

    Reference: Dongseong Hwang, "FAdam: Adam is a natural gradient optimizer using
    diagonal empirical Fisher information", 2024.
    https://arxiv.org/abs/2405.12807
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 0.1,
        clip: float = 1.0,
        p: float = 0.5,
        eps: float = 1e-8,
        momentum_dtype: torch.dtype = torch.float32,
        fim_dtype: torch.dtype = torch.float32,
        maximize: bool = False,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 < clip:
            raise ValueError(f"Invalid clip value: {clip}")
        if not 0.0 < p:
            raise ValueError(f"Invalid p value: {p}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")

        self.momentum_dtype = momentum_dtype
        self.fim_dtype = fim_dtype
        self.maximize = maximize

        defaults = {
            "lr": lr,
            "betas": betas,
            "weight_decay": weight_decay,
            "clip": clip,
            "p": p,
            "eps": eps,
        }
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

            beta1, beta2 = group["betas"]

            beta2_n = math.pow(beta2, group["step"])
            curr_beta2 = (beta2_n - beta2) / (beta2_n - 1.0)

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("FAdam does not support sparse gradients")
                if torch.is_complex(p):
                    raise RuntimeError("FAdam does not support complex parameters")

                if self.maximize:
                    grad = grad.neg()

                state = self.state[p]
                if len(state) == 0:
                    state["momentum"] = torch.zeros_like(p, dtype=self.momentum_dtype)
                    state["fim"] = torch.zeros_like(p, dtype=self.fim_dtype)

                momentum, fim = state["momentum"], state["fim"]

                fim.mul_(curr_beta2).addcmul_(grad, grad, value=1.0 - curr_beta2)

                rms_grad = grad.pow(2).mean().sqrt_()
                curr_eps = min(rms_grad, 1) * group["eps"]

                fim_base = fim.pow(group["p"]).add_(curr_eps)
                grad_nat = grad / fim_base

                rms = grad_nat.pow(2).mean().sqrt_()
                divisor = max(1, rms) / group["clip"]
                grad_nat.div_(divisor)

                momentum.mul_(beta1).add_(grad_nat, alpha=1.0 - beta1)

                grad_weights = p / fim_base

                rms = grad_weights.pow(2).mean().sqrt_()
                divisor = max(1, rms) / group["clip"]
                grad_weights.div_(divisor)

                grad_weights.mul_(group["weight_decay"]).add_(momentum)

                p.add_(grad_weights, alpha=-group["lr"])

        return loss
