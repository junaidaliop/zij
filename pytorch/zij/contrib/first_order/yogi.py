# Adapted from https://github.com/jettify/pytorch-optimizer (commit 19c3e41)
# Copyright (c) Nikolay Novik (jettify). Licensed under the Apache-2.0 license.
# See THIRD_PARTY_NOTICES.md for details.
r"""Implementation of the Yogi optimizer."""

import math

import torch

from ...core.optimizer import Optimizer, ParamsT

__all__ = ["Yogi"]


class Yogi(Optimizer):
    r"""Implements Yogi, an adaptive method that controls the increase in the
    effective learning rate.

    Yogi replaces the multiplicative second-moment update of Adam with an
    additive, sign-based one, so that the second moment can decrease as well as
    increase and large gradients do not cause it to grow uncontrollably.

    .. math::
       \begin{aligned}
            m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t                          \\
            v_t &= v_{t-1} - (1 - \beta_2)
                \mathrm{sign}(v_{t-1} - g_t^2) \, g_t^2                   \\
            \hat{v}_t &= v_t / (1 - \beta_2^t)                                  \\
            \theta_t &= \theta_{t-1} - \frac{\eta}{1 - \beta_1^t}
                \frac{m_t}{\sqrt{\hat{v}_t} + \epsilon}
       \end{aligned}

    .. note::
       Following the upstream implementation, both the first moment
       :math:`m_0` and second moment :math:`v_0` are initialized to
       ``initial_accumulator`` (default ``1e-6``) rather than to zero.

    Reference: Manzil Zaheer, Sashank J. Reddi, Devendra Sachan,
    Satyen Kale, Sanjiv Kumar, "Adaptive Methods for Nonconvex Optimization",
    NeurIPS 2018.
    https://papers.nips.cc/paper_files/paper/2018/hash/90365351ccc7437a1309dc64e4db32a3-Abstract.html
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-2,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-3,
        initial_accumulator: float = 1e-6,
        weight_decay: float = 0.0,
    ) -> None:
        if not 0.0 < lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "initial_accumulator": initial_accumulator,
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
                        "Yogi does not support sparse gradients, "
                        "please consider SparseAdam instead"
                    )

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.full_like(
                        p, group["initial_accumulator"]
                    )
                    state["exp_avg_sq"] = torch.full_like(
                        p, group["initial_accumulator"]
                    )

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

                state["step"] += 1
                bias_correction1 = 1.0 - beta1 ** state["step"]
                bias_correction2 = 1.0 - beta2 ** state["step"]

                if group["weight_decay"] != 0.0:
                    grad = grad.add(p, alpha=group["weight_decay"])

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)

                grad_squared = grad.mul(grad)
                exp_avg_sq.addcmul_(
                    torch.sign(exp_avg_sq - grad_squared),
                    grad_squared,
                    value=-(1.0 - beta2),
                )

                denom = exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2)).add_(
                    group["eps"]
                )
                step_size = group["lr"] / bias_correction1
                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
