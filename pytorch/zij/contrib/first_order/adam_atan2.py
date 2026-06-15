# Adapted from https://github.com/lucidrains/adam-atan2-pytorch (commit 1bfca92)
# Copyright (c) 2024 Phil Wang. Licensed under the MIT license.
# See THIRD_PARTY_NOTICES.md for details.
"""Implementation of the Adam-atan2 optimizer."""

from __future__ import annotations

from typing import Callable

import torch

from ...core.optimizer import Optimizer, ParamsT

__all__ = ["AdamAtan2"]


def exists(val):
    return val is not None


class AdamAtan2(Optimizer):
    r"""Implements Adam-atan2, a scale-invariant epsilon-free variant of Adam.

    The standard Adam update divides the bias-corrected first moment by the
    square root of the bias-corrected second moment plus a small constant
    :math:`\epsilon` to avoid division by zero. Adam-atan2 replaces that
    division with :math:`\mathrm{atan2}`, which removes the
    :math:`\epsilon` hyperparameter and makes the update invariant to the scale
    of the gradient.

    .. math::
       \begin{aligned}
            m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t                          \\
            v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2                         \\
            \hat{m}_t &= m_t / (1 - \beta_1^t), \quad
                \hat{v}_t = v_t / (1 - \beta_2^t)                               \\
            \theta_t &= \theta_{t-1} - \gamma\, a \,
                \mathrm{atan2}\!\left(\hat{m}_t,\, b \sqrt{\hat{v}_t}\right)
       \end{aligned}

    where :math:`a` and :math:`b` are fixed constants that recover the scale
    and shape of the original Adam step.

    Reference: Katie Everett et al., "Scaling Exponents Across Parameterizations
    and Optimizers", ICML 2024.
    https://arxiv.org/abs/2407.05872
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
        regen_reg_rate: float = 0.0,
        decoupled_wd: bool = False,
        cautious_wd: bool = False,
        cautious_factor: float = 1.0,
        a: float = 1.27,
        b: float = 1.0,
    ):
        assert lr > 0.0
        assert all([0.0 <= beta <= 1.0 for beta in betas])
        assert weight_decay >= 0.0
        assert regen_reg_rate >= 0.0
        assert not (weight_decay > 0.0 and regen_reg_rate > 0.0)
        assert 0.0 <= cautious_factor <= 1.0

        self._init_lr = lr
        self.decoupled_wd = decoupled_wd

        defaults = dict(
            lr=lr,
            betas=betas,
            a=a,
            b=b,
            weight_decay=weight_decay,
            cautious_wd=cautious_wd,
            regen_reg_rate=regen_reg_rate,
            cautious_factor=cautious_factor,
        )

        super().__init__(params, defaults)

        # independent of lr

        if decoupled_wd:
            for group in self.param_groups:
                group["weight_decay"] /= lr
                group["regen_reg_rate"] /= lr

    @torch.no_grad()
    def step(self, closure: Callable | None = None):
        loss = None
        if exists(closure):
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in filter(lambda p: exists(p.grad), group["params"]):
                grad = p.grad
                lr = group["lr"]
                wd = group["weight_decay"]
                regen_rate = group["regen_reg_rate"]
                cautious_wd = group["cautious_wd"]
                cautious_factor = group["cautious_factor"]
                beta1, beta2 = group["betas"]
                a, b = group["a"], group["b"]
                state = self.state[p]

                # regenerative regularization from Kumar et al.
                # https://arxiv.org/abs/2308.11958

                if regen_rate > 0.0 and "param_init" in state:
                    param_init = state["param_init"]
                    p.lerp_(param_init, lr * regen_rate)

                # init state if needed

                if len(state) == 0:
                    state["steps"] = 0
                    state["exp_avg"] = torch.zeros_like(grad)
                    state["exp_avg_sq"] = torch.zeros_like(grad)

                    if regen_rate > 0.0:
                        state["param_init"] = p.clone()

                exp_avg, exp_avg_sq, steps = (
                    state["exp_avg"],
                    state["exp_avg_sq"],
                    state["steps"],
                )

                steps += 1

                # bias corrections

                bias_correct1 = 1.0 - beta1**steps
                bias_correct2 = 1.0 - beta2**steps

                # decay running averages

                exp_avg.lerp_(grad, 1.0 - beta1)
                exp_avg_sq.lerp_(grad * grad, 1.0 - beta2)

                # the proposed change to the update rule: atan2 in place of a
                # division with an epsilon in the denominator
                # a * atan2(exp_avg / bias_correct1, b * sqrt(exp_avg_sq / bias_correct2))

                den = exp_avg_sq.mul(b * b / bias_correct2).sqrt_()
                update = exp_avg.mul(1.0 / bias_correct1).atan2_(den)

                # cautious update - algorithm 2 in https://arxiv.org/abs/2411.16085

                if cautious_factor < 1.0:
                    align_mask = (update * grad) > 0
                    scale = torch.where(
                        align_mask, torch.ones_like(grad), cautious_factor
                    )
                    update *= scale / scale.mean().clamp(min=1e-5)

                # maybe weight decay

                if wd > 0.0:
                    # maybe cautious weight decay
                    # https://arxiv.org/abs/2510.12402

                    wd_mask = (update * p > 0).float() if cautious_wd else 1.0

                    p.mul_(1.0 - lr * wd * wd_mask)

                # update parameters

                p.add_(update, alpha=-lr * a)

                # increment steps

                state["steps"] = steps

        return loss
