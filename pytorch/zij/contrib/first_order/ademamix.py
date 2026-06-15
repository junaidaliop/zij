# Adapted from https://github.com/apple/ml-ademamix (commit c3d73c0)
# Copyright (c) 2024 Apple Inc. Licensed under the MIT license.
# See THIRD_PARTY_NOTICES.md for details.
"""Implementation of the AdEMAMix optimizer."""

import math
from typing import Callable, Iterable, Optional

import torch

from ...core.optimizer import Optimizer

__all__ = ["AdEMAMix"]


def _linear_warmup_scheduler(step, alpha_end, alpha_start=0, warmup=1):
    if step < warmup:
        a = step / float(warmup)
        return (1.0 - a) * alpha_start + a * alpha_end
    return alpha_end


def _linear_hl_warmup_scheduler(step, beta_end, beta_start=0, warmup=1):
    def f(beta, eps=1e-8):
        return math.log(0.5) / math.log(beta + eps) - 1

    def f_inv(t):
        return math.pow(0.5, 1 / (t + 1))

    if step < warmup:
        a = step / float(warmup)
        return f_inv((1.0 - a) * f(beta_start) + a * f(beta_end))
    return beta_end


class AdEMAMix(Optimizer):
    r"""Implements AdEMAMix, an Adam variant mixing a fast and a slow gradient EMA.

    .. math::
       \begin{aligned}
       m_{1,t} &= \beta_1 m_{1,t-1} + (1 - \beta_1)\, g_t \\
       m_{2,t} &= \beta_3 m_{2,t-1} + (1 - \beta_3)\, g_t \\
       v_t &= \beta_2 v_{t-1} + (1 - \beta_2)\, g_t^2 \\
       \theta_t &= \theta_{t-1} - \eta\left(
                   \frac{m_{1,t} / (1 - \beta_1^t) + \alpha\, m_{2,t}}
                        {\sqrt{v_t / (1 - \beta_2^t)} + \epsilon}
                   + \lambda \theta_{t-1}\right)
       \end{aligned}

    where :math:`m_{1,t}` is the fast EMA with decay :math:`\beta_1`,
    :math:`m_{2,t}` the slow EMA with decay :math:`\beta_3`, :math:`\alpha`
    the coefficient mixing the two, and :math:`\lambda` the decoupled weight
    decay. The slow EMA :math:`m_{2,t}` is not bias-corrected. When
    ``beta3_warmup`` or ``alpha_warmup`` is set, :math:`\beta_3` and
    :math:`\alpha` are ramped from :math:`\beta_1` and :math:`0` over that many
    steps.

    Reference: Matteo Pagliardini, Pierre Ablin, David Grangier,
    "The AdEMAMix Optimizer: Better, Faster, Older", arXiv 2024.
    https://arxiv.org/abs/2409.03137
    """

    def __init__(self, params: Iterable[torch.Tensor], lr: float = 1e-3,
                 betas: tuple[float, float, float] = (0.9, 0.999, 0.9999),
                 alpha: float = 2.0,
                 beta3_warmup: Optional[int] = None,
                 alpha_warmup: Optional[int] = None,
                 eps: float = 1e-8, weight_decay: float = 0.0):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= betas[2] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 2: {betas[2]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= alpha:
            raise ValueError(f"Invalid alpha value: {alpha}")

        defaults = dict(lr=lr, betas=betas, eps=eps, alpha=alpha,
                        beta3_warmup=beta3_warmup, alpha_warmup=alpha_warmup,
                        weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            lmbda = group["weight_decay"]
            eps = group["eps"]
            beta1, beta2, beta3_final = group["betas"]
            beta3_warmup = group["beta3_warmup"]
            alpha_final = group["alpha"]
            alpha_warmup = group["alpha_warmup"]

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('AdEMAMix does not support sparse gradients.')

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    if beta1 != 0.0:  # save memory in case beta1 is 0.0
                        state['exp_avg_fast'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    else:
                        state['exp_avg_fast'] = None
                    state['exp_avg_slow'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg_fast = state['exp_avg_fast']
                exp_avg_slow = state['exp_avg_slow']
                exp_avg_sq = state['exp_avg_sq']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                if alpha_warmup is not None:
                    alpha = _linear_warmup_scheduler(state["step"], alpha_end=alpha_final,
                                                     alpha_start=0, warmup=alpha_warmup)
                else:
                    alpha = alpha_final

                if beta3_warmup is not None:
                    beta3 = _linear_hl_warmup_scheduler(state["step"], beta_end=beta3_final,
                                                        beta_start=beta1, warmup=beta3_warmup)
                else:
                    beta3 = beta3_final

                if beta1 != 0.0:
                    exp_avg_fast.mul_(beta1).add_(grad, alpha=1 - beta1)
                else:
                    exp_avg_fast = grad
                exp_avg_slow.mul_(beta3).add_(grad, alpha=1 - beta3)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

                update = (exp_avg_fast.div(bias_correction1) + alpha * exp_avg_slow) / denom

                update.add_(p, alpha=lmbda)

                p.add_(update, alpha=-lr)

        return loss
