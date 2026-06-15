# Adapted from https://github.com/kozistr/pytorch_optimizer (commit 3d08fa0)
# Copyright (c) Hyeongchan Kim (kozistr). Licensed under the Apache-2.0 license.
# See THIRD_PARTY_NOTICES.md for details.
r"""Implementation of the AdaShift optimizer."""

import math
from typing import Callable, Optional

import torch

from ...core.optimizer import Optimizer, ParamsT

__all__ = ["AdaShift"]


class AdaShift(Optimizer):
    r"""Implements AdaShift, an adaptive method that decorrelates the gradient
    from the second-moment estimate by a temporal shift.

    In Adam the gradient :math:`g_t` enters both the numerator and the
    denominator of the update, which biases the effective step size. AdaShift
    removes that correlation by accumulating the second moment from a gradient
    that is :math:`n` steps in the past, where :math:`n` is the window size
    :math:`\mathrm{keep\_num}`. A spatial reduction :math:`\phi` (the maximum
    over each parameter block by default) is applied to the shifted squared
    gradient so that the denominator is independent of the current gradient.

    .. math::
       \begin{aligned}
            m_t &= \beta_1 m_{t-1}
                   + \frac{1}{w} g_t
                   - \frac{\beta_1^{n}}{w} g_{t-n}                              \\
            v_t &= \beta_2 v_{t-1} + (1 - \beta_2)\, \phi(g_{t-n}^2)            \\
            \hat{v}_t &= \frac{v_t}{1 - \beta_2^{\,t-n}}                        \\
            \theta_t &= \theta_{t-1} - \eta \, \frac{m_t}{\sqrt{\hat{v}_t} + \epsilon}
       \end{aligned}

    where :math:`w = \sum_{i=0}^{n-1} \beta_1^i` normalizes the windowed first
    moment, :math:`g_{t-n}` is the oldest (evicted) gradient in the window, :math:`\eta`
    is the learning rate, and :math:`\epsilon` is added for numerical stability.
    No update is applied until the window has filled with :math:`n` gradients.

    Reference: Zhiming Zhou, Qingru Zhang, Guansong Lu, Hongwei Wang, Weinan
    Zhang, Yong Yu, "AdaShift: Decorrelation and Convergence of Adaptive
    Learning Rate Methods", ICLR 2019.
    https://arxiv.org/abs/1810.00143
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        keep_num: int = 10,
        reduce_func: Optional[Callable[[torch.Tensor], torch.Tensor]] = torch.max,
        eps: float = 1e-10,
        maximize: bool = False,
    ) -> None:
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if keep_num <= 0:
            raise ValueError(f"Invalid keep_num value: {keep_num}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")

        self.reduce_func: Callable[[torch.Tensor], torch.Tensor] = (
            reduce_func if reduce_func is not None else lambda x: x
        )
        self.maximize = maximize

        defaults = {
            "lr": lr,
            "betas": betas,
            "keep_num": keep_num,
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
            beta1, beta2 = group["betas"]
            keep_num = group["keep_num"]

            exp_weight_sum = sum(beta1**i for i in range(keep_num))
            first_grad_weight = beta1 ** (keep_num - 1) / exp_weight_sum
            last_grad_weight = 1.0 / exp_weight_sum

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdaShift does not support sparse gradients")
                if torch.is_complex(p):
                    raise RuntimeError("AdaShift does not support complex parameters")
                if self.maximize:
                    grad = grad.neg()

                state = self.state[p]
                if len(state) == 0:
                    # The gradient window is held in a ring-buffer tensor rather
                    # than a deque so the state survives a state_dict roundtrip.
                    state["count"] = 1
                    state["grad_window"] = grad.new_zeros((keep_num, *grad.shape))
                    state["grad_window"][0] = grad
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    continue

                count = state["count"]
                grad_window = state["grad_window"]

                # The oldest gradient in the window sits in the slot about to be
                # overwritten. The window must be full before an update is
                # applied: the second moment trails the first moment by
                # keep_num gradients.
                ready = count >= keep_num
                offset_grad = grad_window[count % keep_num].clone()
                grad_window[count % keep_num] = grad
                state["count"] = count + 1
                if not ready:
                    continue

                exp_avg = state["exp_avg"]
                exp_avg.sub_(offset_grad, alpha=first_grad_weight).mul_(beta1).add_(
                    grad, alpha=last_grad_weight
                )

                reduced_grad_sq = self.reduce_func(offset_grad.pow(2))

                exp_avg_sq = state["exp_avg_sq"]
                exp_avg_sq.mul_(beta2).add_(reduced_grad_sq, alpha=1.0 - beta2)

                bias_correction = 1.0 - math.pow(beta2, state["count"] - keep_num)
                denom = exp_avg_sq.div(bias_correction).sqrt_().add_(group["eps"])

                p.addcdiv_(exp_avg, denom, value=-group["lr"])

        return loss
